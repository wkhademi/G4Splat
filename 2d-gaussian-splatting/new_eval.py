import os
import sys
import json
import h5py
import torch
import imageio
import trimesh
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from glob import glob
from PIL import Image
from argparse import ArgumentParser
from pytorch3d.ops import knn_points
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from scene import Scene
from chamfer import chamfer_distance
from gaussian_renderer import render
from gaussian_renderer import GaussianModel
from utils.mesh_utils import GaussianExtractor
from arguments import ModelParams, PipelineParams, get_combined_args
from mast3r.colmap.colmap_utils import read_cameras_text, undistorted_camera, camera_to_intrinsic


ssim = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).cuda()


def fnv_hash_vec(arr):
    """
    FNV64-1A
    """
    assert arr.ndim == 2
    # Floor first for negative coordinates
    arr = arr.copy()
    arr = arr.astype(np.uint64, copy=False)
    hashed_arr = np.uint64(14695981039346656037) * np.ones(arr.shape[0], dtype=np.uint64)
    for j in range(arr.shape[1]):
        hashed_arr *= np.uint64(1099511628211)
        hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
    return hashed_arr


def ravel_hash_vec(arr):
    """
    Ravel the coordinates after subtracting the min coordinates.
    """
    assert arr.ndim == 2
    arr = arr.copy()
    arr -= arr.min(0)
    arr = arr.astype(np.uint64, copy=False)
    arr_max = arr.max(0).astype(np.uint64) + 1

    keys = np.zeros(arr.shape[0], dtype=np.uint64)
    # Fortran style indexing
    for j in range(arr.shape[1] - 1):
        keys += arr[:, j]
        keys *= arr_max[j + 1]
    keys += arr[:, -1]
    return keys


def voxelize(coord, voxel_size=0.05, hash_type='fnv', mode='random'):
    '''
    Voxelization of the input coordinates
    Parameters:
        coord: input coordinates (N x D)
        voxel_size: Size of the voxels
        hash_type: Type of the hashing function, can be chosen from 'ravel' and 'fnv'
        mode: 'random', 'deterministic' or 'multiple' mode. In training mode one selects a random point within the voxel as the representation of the voxel.
              In deterministic model right now one always uses the first point. Usually random mode is preferred for training. In 'multiple' mode, we will return
              multiple sets of indices, so that each point will be covered in at least one of these sets
    Returns:
        idx_unique: the indices of the points so that there is at most one point for each voxel
    '''
    coord = coord - coord.min(0)
    discrete_coord = np.floor(coord / np.array(voxel_size))
    if hash_type == 'ravel':
        key = ravel_hash_vec(discrete_coord)
    else:
        key = fnv_hash_vec(discrete_coord)

    idx_sort = np.argsort(key)
    key_sort = key[idx_sort]
    _, count = np.unique(key_sort, return_counts=True)
    if mode == 'deterministic':
        # idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + torch.randint(count.max(), (count.size,)).numpy() % count
        idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + np.zeros((count.size,), dtype=np.int32)
        idx_unique = idx_sort[idx_select]
        return idx_unique
    elif mode == 'multiple':  # mode is 'multiple'
        idx_data = []
        for i in range(count.max()):
            idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
            idx_part = idx_sort[idx_select]
            idx_data.append(idx_part)
        return idx_data
    else:  # mode == 'random'
        # idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + np.random.randint(0, count.max(), count.size) % count
        idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + torch.randint(count.max(), (count.size,)).numpy() % count
        idx_unique = idx_sort[idx_select]
        return idx_unique

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def filter_complete_points(
    valid_points, 
    complete_points, 
    max_coord=25.0,
    min_coord=-25.0,
    voxel_size=1.0
):
    """
    Chunk complete_points into voxel chunks and only keep chunks that contain points from 
    either input points or gt_points.
    """    
    # Calculate number of voxels in each dimension
    coord_range = max_coord - min_coord
    num_voxels = int(np.ceil(coord_range / voxel_size))

    # get voxels occupied by complete points
    complete_points_shifted = complete_points - min_coord
    voxel_indices = np.floor(complete_points_shifted / voxel_size).astype(int)
    voxel_indices_1d = voxel_indices[:,0] + \
                    voxel_indices[:,1] * num_voxels + \
                    voxel_indices[:,2] * num_voxels * num_voxels
    complete_unique_voxels, chunk_indices = np.unique(voxel_indices_1d, return_inverse=True)
    complete_unique_voxels = complete_unique_voxels.tolist()

    # get voxels occupied by valid points
    valid_points_shifted = valid_points - min_coord
    voxel_indices = np.floor(valid_points_shifted / voxel_size).astype(int)
    voxel_indices_1d = voxel_indices[:,0] + \
                    voxel_indices[:,1] * num_voxels + \
                    voxel_indices[:,2] * num_voxels * num_voxels
    valid_unique_voxels, _ = np.unique(voxel_indices_1d, return_inverse=True)
    valid_unique_voxels = valid_unique_voxels.tolist()

    filtered_complete_points = []
    for voxel_idx in valid_unique_voxels:
        if voxel_idx in complete_unique_voxels:
            chunk_mask = chunk_indices == complete_unique_voxels.index(voxel_idx)
            filtered_complete_points.append(complete_points[chunk_mask])

    filtered_complete_points = np.concatenate(filtered_complete_points, axis=0)
    return filtered_complete_points


def backproject(depth, extrinsics, K):
    """
    Backproject depth map into 3D in world coordinate system.
    """
    u, v = np.meshgrid(range(depth.shape[1]), range(depth.shape[0]))
    u = u.reshape([1, -1])[0]
    v = v.reshape([1, -1])[0]

    z = depth[v, u]

    # remove invalid pixels (i.e., zero depth)
    non_inf_indices = np.argwhere(z > 0).T[0]
    z = z[non_inf_indices]
    u = u[non_inf_indices]
    v = v[non_inf_indices]

    # calculate coordinates
    x = (u - K[0][2]) * z / K[0][0]
    y = (v - K[1][2]) * z / K[1][1]

    point_cam = np.vstack([x, y, z]).T

    # camera to world coordinates
    R = extrinsics[:3, :3]
    T = extrinsics[:3, 3]
    point_canonical = point_cam.dot(R.T) + T

    return point_canonical


def eval_scene(model, pipeline, args):
    print(f"Evaluating scene {os.path.basename(os.path.dirname(args.source_path))} with model iteration {args.iteration}")

    dataset, iteration, pipe = model.extract(args), args.iteration, pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    train_scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    if args.render_all_img:
        dataset.eval = True
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    num_train_views = len(scene.getTrainCameras()) - len(train_scene.getTrainCameras())

    scene_id = os.path.basename(os.path.dirname(args.source_path))
    data_dir = "/media/wkhademi/MainDrive/GradSchool/Code/G4Splat/data"
    if "scannetplusplus" in args.source_path:
        scene_data_dir = os.path.join(data_dir, "scannetplusplus", scene_id)
        rgb_data_dir = os.path.join(scene_data_dir, "images")
        depth_data_dir = os.path.join(scene_data_dir, "depths")

        # load intrinsics from COLMAP format
        intrinsics_file = f'{scene_data_dir}/cameras.txt'
        colmap_camera = list(read_cameras_text(intrinsics_file).values())[0]
        pinhole_camera = undistorted_camera(colmap_camera)
        intrinsics = camera_to_intrinsic(pinhole_camera)

        # rescale intrinsics to match loaded image size
        height, width = pinhole_camera.height, pinhole_camera.width
        s_x = 480.0 / float(width)
        s_y = 360.0 / float(height)
        intrinsics[0, 0] *= s_x
        intrinsics[1, 1] *= s_y
        intrinsics[0, 2] = 480.0 / 2.
        intrinsics[1, 2] = 360.0 / 2.

        gt_images = []
        gt_depths = []
        gt_backprojected_point_cloud = []
        for idx, camera in enumerate(scene.getTrainCameras()):
            image_idx = camera.image_name.split("_")[0]

            # load pose
            c2w = np.loadtxt(os.path.join(scene_data_dir, "extrinsics", f"{int(image_idx)}.txt"))

            # load image
            gt_image = Image.open(os.path.join(rgb_data_dir, f"{image_idx}_rgb.png"))
            gt_image = np.array(gt_image)

            # load depth
            gt_depth = np.load(os.path.join(depth_data_dir, f"{image_idx}_depth.npy"))

            # backproject depth
            view_points = backproject(gt_depth, c2w, intrinsics)
            gt_backprojected_point_cloud.append(view_points)

            # only store test views
            if idx < num_train_views:
                gt_images.append(gt_image)
                gt_depths.append(gt_depth)
        gt_backprojected_point_cloud = np.concatenate(gt_backprojected_point_cloud, axis=0)
        gt_complete_point_cloud_data = np.load(
            os.path.join(scene_data_dir, "gt_mesh", "complete_scene.npz")
        )
        gt_complete_point_cloud = gt_complete_point_cloud_data["points"]
        gt_complete_point_cloud[:, 2] *= -1
        # Rotate the pose -90 degrees about the x-axis (in radians)
        angle = np.pi / 2
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle),  np.cos(angle)]
        ])
        gt_complete_point_cloud = gt_complete_point_cloud @ Rx.T
        gt_complete_point_cloud = filter_complete_points(gt_backprojected_point_cloud, gt_complete_point_cloud)
    elif "threedfront" in args.source_path:
        scene_data_dir = os.path.join(data_dir, "threedfront", scene_id)
        rgb_data_dir = os.path.join(scene_data_dir, "images")
        depth_data_dir = os.path.join(scene_data_dir, "poses")

        # load intrinsics
        intrinsics = np.load(os.path.join(scene_data_dir, "cam_K.npy"))

        gt_images = []
        gt_depths = []
        gt_backprojected_point_cloud = []
        for idx, camera in enumerate(scene.getTrainCameras()):
            image_idx = camera.image_name.split("_")[0]

            # load image
            gt_image = Image.open(os.path.join(rgb_data_dir, f"{image_idx}_rgb.png"))
            gt_image = np.array(gt_image)
            
            with h5py.File(os.path.join(depth_data_dir, f"{int(image_idx)}.hdf5"), "r") as f:
                # load pose
                c2w = np.array(f["cam_Ts"])

                # blender camera to opengl camera
                y_z_swap = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
                c2w = y_z_swap.dot(c2w)
                c2w[:3, 0] *= -1

                # opengl camera to opencv (COLMAP) camera
                c2w[:3, 1] *= -1
                c2w[:3, 2] *= -1

                # Rotate the pose -90 degrees about the x-axis (in radians)
                angle = np.pi / 2
                Rx = np.array([
                    [1, 0, 0, 0],
                    [0, np.cos(angle), -np.sin(angle), 0],
                    [0, np.sin(angle),  np.cos(angle), 0],
                    [0, 0, 0, 1]
                ])
                c2w = Rx @ c2w

                # load depth
                gt_depth = np.array(f["depth"])[:, ::-1].astype(np.float32)
                gt_depth[gt_depth >= 10000000000.0] = 0.0

                # backproject depth
                view_points = backproject(gt_depth, c2w, intrinsics)
                gt_backprojected_point_cloud.append(view_points)

            if idx < num_train_views:
                gt_images.append(gt_image)
                gt_depths.append(gt_depth)
        gt_backprojected_point_cloud = np.concatenate(gt_backprojected_point_cloud, axis=0)
        gt_complete_point_cloud_data = np.load(
            os.path.join(scene_data_dir, "gt_mesh", "scene.npz")
        )
        gt_complete_point_cloud = gt_complete_point_cloud_data["points"]
        gt_complete_point_cloud[:, 2] *= -1
        # Rotate the pose -90 degrees about the x-axis (in radians)
        angle = np.pi / 2
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle),  np.cos(angle)]
        ])
        gt_complete_point_cloud = gt_complete_point_cloud @ Rx.T
        gt_complete_point_cloud = filter_complete_points(gt_backprojected_point_cloud, gt_complete_point_cloud)
    
    # prepare prediction
    predicted_mesh = trimesh.load(
        os.path.join(os.path.dirname(args.source_path), "tetra_meshes", "tetra_mesh_binary_search_7_iter_7000.ply")
    )
    predicted_point_cloud = predicted_mesh.sample(500000)
    predicted_point_cloud = filter_complete_points(gt_backprojected_point_cloud, predicted_point_cloud)
    idx = voxelize(predicted_point_cloud, voxel_size=0.02)
    predicted_point_cloud = torch.tensor(predicted_point_cloud[idx]).to("cuda").float()
    predicted_point_cloud = predicted_point_cloud.view(1, -1, 3)

    # prepare ground truth
    gt_point_cloud = gt_complete_point_cloud
    idx = voxelize(gt_point_cloud, voxel_size=0.02)
    gt_point_cloud = torch.tensor(gt_point_cloud[idx]).to("cuda").float()
    gt_point_cloud = gt_point_cloud.view(1, -1, 3)

    # load partial point cloud
    partial_point_cloud = trimesh.load(
        os.path.join(args.source_path, "points.ply")
    )
    partial_point_cloud = np.array(partial_point_cloud.vertices)
    idx = voxelize(partial_point_cloud, voxel_size=0.02)
    partial_point_cloud = torch.tensor(partial_point_cloud[idx]).to("cuda").float()
    partial_point_cloud = partial_point_cloud.view(1, -1, 3)

    # prepare missing
    dists, nn_idx, _ = knn_points(gt_point_cloud, partial_point_cloud, K=1)
    dists = dists.squeeze(-1) ** 0.5
    missing_point_cloud = gt_point_cloud[dists > 0.02].reshape(1, -1, 3)

    # Compute Chamfer Distance
    cd, _ = chamfer_distance(gt_point_cloud, predicted_point_cloud)
    cd = cd.item()

    # Compute Partial One-Sided Chamfer Distance
    partial_cd, _ = chamfer_distance(partial_point_cloud, predicted_point_cloud, single_directional=True)
    partial_cd = partial_cd.item()

    # Compute Complete One-Sided Chamfer Distance
    complete_cd, _ = chamfer_distance(missing_point_cloud, predicted_point_cloud, single_directional=True)
    complete_cd = complete_cd.item()
    
    # Color the input point cloud red ([255, 0, 0]) and the complete point cloud black ([0, 0, 0])
    # red = np.tile(np.array([[255, 0, 0]]), (gt_backprojected_point_cloud.shape[0], 1))
    # black = np.tile(np.array([[0, 0, 0]]), (gt_complete_point_cloud.shape[0], 1))
    # gt_input_point_cloud = trimesh.PointCloud(gt_backprojected_point_cloud, colors=red)
    # gt_point_cloud = trimesh.PointCloud(gt_complete_point_cloud, colors=black)

    # trimesh.Scene([partial_point_cloud, gt_input_point_cloud, gt_point_cloud]).show()
    
    # render test viewpoints
    gaussExtractor = GaussianExtractor(gaussians, render, pipe, bg_color=bg_color)  
    gaussExtractor.reconstruction(scene.getTrainCameras())

    # save out renders
    for idx, camera in enumerate(scene.getTrainCameras()):
        if idx < num_train_views:
            continue
        image_idx = camera.image_name.split("_")[0]

        rendered_image = gaussExtractor.rgbmaps[idx].squeeze().permute(1, 2, 0).numpy()
        rendered_image = (rendered_image * 255).clip(0, 255).astype(np.uint8)
        image_path = os.path.join(args.output_path, f"rendered_rgb_{image_idx}.png")
        imageio.imwrite(image_path, rendered_image)

        rendered_depth = gaussExtractor.depthmaps[idx].squeeze().numpy()
        max_depth = np.max(gt_depths)
        min_depth = np.min(gt_depths)
        rendered_depth = (255 * ((rendered_depth - min_depth) / (max_depth - min_depth))).astype(np.uint8)
        depth_path = os.path.join(args.output_path, f"rendered_depth_{image_idx}.png")
        imageio.imwrite(depth_path, rendered_depth)

    # PSNR
    rendered_images = torch.clip(torch.stack(gaussExtractor.rgbmaps[num_train_views:], dim=0), 0.0, 1.0)
    rendered_images = rendered_images.cuda()
    gt_images = torch.tensor(gt_images).permute(0, 3, 1, 2).cuda().float() / 255.
    mse = F.mse_loss(rendered_images, gt_images, reduction='none')
    mse_per_image = mse.reshape(mse.shape[0], -1).mean(dim=1)
    psnr_per_image = -10 * torch.log10(mse_per_image + 1e-8)
    psnr = psnr_per_image.mean().item()

    # SSIM
    ssim_value = ssim(rendered_images, gt_images)
    ssim_value = ssim_value.item()

    # LPIPS
    lpips_value = lpips(rendered_images, gt_images)
    lpips_value = lpips_value.item()

    # L1 Depth Error
    rendered_depths = torch.stack(gaussExtractor.depthmaps[num_train_views:], dim=0).squeeze().numpy()
    gt_depths = np.array(gt_depths)
    l1_depth = np.mean(np.abs(rendered_depths - gt_depths))

    print(f"Evaluation Metrics:\n"
              f"PSNR: {psnr:.3f} dB\n"
              f"SSIM: {ssim_value:.4f}\n"
              f"LPIPS: {lpips_value:.4f}\n"
              f"L1 Depth: {l1_depth: .4f}\n"
              f"Chamfer Distance: {cd:.4f}\n"
              f"Partial One-Sided Chamfer Distance: {partial_cd:.4f}\n"
              f"Complete One-Sided Chamfer Distance: {complete_cd:.4f}")

    eval_metrics = {}
    eval_metrics['cd'] = cd
    eval_metrics['partial_cd'] = partial_cd
    eval_metrics['complete_cd'] = complete_cd
    eval_metrics['psnr'] = psnr
    eval_metrics['ssim'] = ssim_value
    eval_metrics['lpips'] = lpips_value
    eval_metrics['l1_depth'] = l1_depth

    return eval_metrics


if __name__ == '__main__':

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--render_all_img", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    dataset, iteration, pipe = model.extract(args), args.iteration, pipeline.extract(args)

    scenes = os.listdir(args.source_path)
    source_path = args.source_path

    avg_metrics = {
        "psnr": [],
        "ssim": [],
        "lpips": [],
        "l1_depth": [],
        "cd": [],
        "partial_cd": [],
        "complete_cd": [],
    }
    for scene in scenes:
        scene_source_path = os.path.join(source_path, scene, "mast3r_sfm")
        scene_model_path = os.path.join(source_path, scene, "free_gaussians")
        scene_output_path = os.path.join(source_path, scene, "renders")
        os.makedirs(scene_output_path, exist_ok=True)

        args.source_path = scene_source_path
        args.model_path = scene_model_path
        args.output_path = scene_output_path

        metrics = eval_scene(model, pipeline, args)
        for k, v, in metrics.items():
            avg_metrics[k].append(v)

    for k, v in avg_metrics.items():
        print(f' Average {k}: {np.mean(v):.4f}')

    