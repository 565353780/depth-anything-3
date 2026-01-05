import sys
sys.path.append('../camera-control')
sys.path.append('../da3')

import os

from src.depth_anything_3.utils.export.glb import export_to_glb

from depth_anything_three.Module.detector import Detector


def demo():
    home = os.environ['HOME']
    model_folder_path = home + "/chLi/Model/DepthAnythingV3/DA3-GIANT-1.1/"
    device = 'cuda:5'
    render_data_file_path = home + "/chLi/Dataset/MM/Match/nezha/da3/render_data.npy"
    use_ray_pose = False

    detector = Detector(model_folder_path, device)

    prediction = detector.detectRenderDataFile(render_data_file_path, use_ray_pose)

    export_to_glb(
        prediction,
        home + "/chLi/Dataset/MM/Match/nezha/da3/",
        filter_white_bg=True,
        conf_thresh_percentile=90,
        ensure_thresh_percentile=90,
    )
    return True
