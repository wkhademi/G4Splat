import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import shutil
import argparse

import h5py
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation


def dedup_preserve_order(items):
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def link_or_copy(src, dst, copy):
    if os.path.exists(dst) or os.path.islink(dst):
        os.remove(dst)
    if copy:
        shutil.copy(src, dst)
    else:
        os.symlink(os.path.realpath(src), dst)


def write_rotated_pose(src_hdf5, dst_hdf5, R4):
    """Copy a BlenderProc render and rotate its world-space camera-to-world pose (cam_Ts) by R4.
    Depth is left untouched: mast3r/run_mast3r.py's "poses/" (3D-FRONT style) loader
    unconditionally applies `depth[:, ::-1]` to whatever it reads, and the camera-pose axis
    conversion it chains afterwards (blender -> opengl -> opencv -> Rx90) assumes that mirrored
    convention. Colors/normals/segmaps are otherwise copied as-is.
    """
    if os.path.exists(dst_hdf5) or os.path.islink(dst_hdf5):
        os.remove(dst_hdf5)
    shutil.copy(src_hdf5, dst_hdf5)
    with h5py.File(dst_hdf5, 'r+') as f:
        cam_Ts = np.array(f['cam_Ts'])
        f['cam_Ts'][...] = R4 @ cam_Ts


def write_flipped_image(src_png, dst_png):
    """Mirror the RGB image left-right to match the depth flip mast3r/run_mast3r.py applies,
    so depth and color stay pixel-aligned for the same view.
    """
    if os.path.exists(dst_png) or os.path.islink(dst_png):
        os.remove(dst_png)
    img = Image.open(src_png)
    img.transpose(Image.FLIP_LEFT_RIGHT).save(dst_png)


# geometry/complete_scene.npz (points/normals) is stored in a different raw axis
# convention than cam_Ts: verified empirically (nearest-neighbor distance between this
# GT point cloud and the camera-backprojected point cloud) that this fixed rotation
# converts it into the same raw convention cam_Ts is in, before the per-scene
# floor-alignment rotation is applied.
MESH_TO_POSE_CONVENTION = np.array([
    [1, 0, 0],
    [0, 0, -1],
    [0, 1, 0],
])


def write_rotated_gt_mesh(src_npz, dst_npz, R):
    """Rotate the ground-truth scene point cloud (points + normals) so it matches the
    same world-frame alignment applied to the camera poses. Normals are stored as
    uint8-encoded unit vectors ((n+1)/2*255) and must be decoded/rotated/re-encoded.
    """
    if os.path.exists(dst_npz) or os.path.islink(dst_npz):
        os.remove(dst_npz)
    data = np.load(src_npz)
    combined = R @ MESH_TO_POSE_CONVENTION
    points = data['points'] @ combined.T
    normals = data['normals'].astype(np.float32) / 255.0 * 2.0 - 1.0
    normals = normals @ combined.T
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.clip(norms, 1e-8, None)
    normals = np.clip(np.round((normals + 1.0) / 2.0 * 255.0), 0, 255).astype(np.uint8)
    np.savez(dst_npz, points=points.astype(data['points'].dtype), colors=data['colors'], normals=normals)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert scenes from the BlenderProc-rendered "replica-processed" layout '
                    'into the images/, poses/, gt_mesh/ layout already used for 3D-FRONT scenes.'
    )
    parser.add_argument('--replica_root', type=str, default='/media/wkhademi/BackupDrive/Datasets/replica-processed',
        help='Root directory of the raw replica-processed dataset (contains <scene>/renderings, <scene>/geometry, cam_K.npy).')
    parser.add_argument('--scenes_json', type=str, default='/media/wkhademi/MainDrive/GradSchool/Code/replica_test_scenes.json',
        help='JSON file mapping scene name -> {input_view_ids, render_view_ids} used to build the train/test split.')
    parser.add_argument('--output_root', type=str, default='data/replica',
        help='Directory to write converted scenes to (one subdirectory per scene).')
    parser.add_argument('--copy', action='store_true',
        help='Copy source files instead of symlinking them (symlinks are used by default to avoid duplicating large renders).')
    args = parser.parse_args()

    with open(args.scenes_json, 'r') as f:
        scenes = json.load(f)

    os.makedirs(args.output_root, exist_ok=True)

    for scene_name, scene_cfg in scenes.items():
        print(f"Preparing scene {scene_name}...")

        scene_src_dir = os.path.join(args.replica_root, scene_name)
        renderings_dir = os.path.join(scene_src_dir, 'renderings')
        geometry_path = os.path.join(scene_src_dir, 'geometry', 'complete_scene.npz')
        cam_K_path = os.path.join(args.replica_root, 'cam_K.npy')

        # rotation that aligns the scene mesh with an axis-aligned ground plane;
        # applied consistently to camera poses and the ground-truth point cloud
        R = Rotation.from_quat(scene_cfg['rotation']).as_matrix()
        R4 = np.eye(4)
        R4[:3, :3] = R

        # train views come from input_view_ids (duplicates removed, order preserved),
        # test views come from render_view_ids
        train_view_files = dedup_preserve_order(scene_cfg['input_view_ids'])
        test_view_files = scene_cfg['render_view_ids']
        ordered_view_files = train_view_files + test_view_files

        scene_dst_dir = os.path.join(args.output_root, scene_name)
        images_dst_dir = os.path.join(scene_dst_dir, 'images')
        poses_dst_dir = os.path.join(scene_dst_dir, 'poses')
        gt_mesh_dst_dir = os.path.join(scene_dst_dir, 'gt_mesh')
        os.makedirs(images_dst_dir, exist_ok=True)
        os.makedirs(poses_dst_dir, exist_ok=True)
        os.makedirs(gt_mesh_dst_dir, exist_ok=True)

        # re-index views to 0, 1, 2, ... in train-then-test order
        for new_idx, view_file in enumerate(ordered_view_files):
            view_stem = view_file.split('.')[0]

            src_hdf5 = os.path.join(renderings_dir, f'{view_stem}.hdf5')
            src_png = os.path.join(renderings_dir, f'{view_stem}.png')

            dst_hdf5 = os.path.join(poses_dst_dir, f'{new_idx}.hdf5')
            dst_png = os.path.join(images_dst_dir, f'{new_idx:06d}_rgb.png')

            write_rotated_pose(src_hdf5, dst_hdf5, R4)
            write_flipped_image(src_png, dst_png)

        # shared intrinsics
        link_or_copy(cam_K_path, os.path.join(scene_dst_dir, 'cam_K.npy'), args.copy)

        # ground-truth scene point cloud
        write_rotated_gt_mesh(geometry_path, os.path.join(gt_mesh_dst_dir, 'scene.npz'), R)

        # train/test split, following the 3D-FRONT "split-Nviews.json" convention
        num_train = len(train_view_files)
        num_test = len(test_view_files)
        split = {
            'train': list(range(num_train)),
            'test': list(range(num_train, num_train + num_test)),
        }
        split_path = os.path.join(scene_dst_dir, f'split-{num_train}views.json')
        with open(split_path, 'w') as f:
            json.dump(split, f, indent=4)

        print(f"  {num_train} train views, {num_test} test views -> {scene_dst_dir}")

    print("Done.")
