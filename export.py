import os
import json
import h5py
import numpy as np
from PIL import Image

json_file = "/media/wkhademi/MainDrive/GradSchool/Code/scannetplusplus_test_scenes.json"
output_dir = "/media/wkhademi/MainDrive/GradSchool/Code/G4Splat/data/scannetplusplus"

with open(json_file, 'r') as f:
    scene_dict = json.load(f)
    scene_id_list = list(scene_dict.keys())

for scene_id in scene_id_list:
    scene_output_dir = os.path.join(output_dir, scene_id)

    scene_file = os.path.join(scene_output_dir, f"{scene_id}.h5")

    input_view_files = []
    render_view_files = []

    for input_view_id in scene_dict[scene_id]["input_view_ids"]:
        if input_view_id not in input_view_files:
            input_view_files.append(input_view_id)
    for render_view_id in scene_dict[scene_id]["render_view_ids"]:
        if render_view_id not in render_view_files:
            render_view_files.append(render_view_id)

    view_files = input_view_files + render_view_files

    with h5py.File(scene_file, 'r') as f:
        for idx, view_file in enumerate(view_files):
            image_name = view_file.split(".")[0]

            # load image and save to as PNG
            image = f['render_rgb'][image_name][()].astype(np.uint8)
            image = Image.fromarray(image)
            rgb_save_file_path = os.path.join(scene_output_dir, "images", f"{idx:06d}_rgb.png")
            image.save(rgb_save_file_path)

            # load depth and save as numpy array
            depth = f['render_depth'][image_name][()].astype(np.float32) / 1000.0
            depth_save_file_path = os.path.join(scene_output_dir, "depths", f"{idx:06d}_depth.npy")
            np.save(depth_save_file_path, depth)
            
            

