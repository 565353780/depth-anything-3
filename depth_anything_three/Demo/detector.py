import sys
sys.path.append('../../MATCH/camera-control')

import os

from depth_anything_3.utils.export.glb import export_to_glb

from depth_anything_three.Module.detector import Detector


def demo():
    shape_id = '003fe719725150960b14cdd6644afe5533743ef63d3767c495ab76a6384a9b2f'

    home = os.environ['HOME']
    model_folder_path = home + "/chLi/Model/DepthAnythingV3/DA3-GIANT-1.1/"
    device = 'cuda:5'
    render_data_file_path = home + "/chLi/Dataset/pixel_align/" + shape_id + "/da3_gt.npy"
    use_ray_pose = False

    detector = Detector(model_folder_path, device)

    prediction = detector.detectRenderDataFile(render_data_file_path, use_ray_pose, return_dict=True)

    export_to_glb(
        prediction,
        home + "/chLi/Dataset/pixel_align/" + shape_id + "/da3/",
        filter_white_bg=True,
        conf_thresh_percentile=90,
        ensure_thresh_percentile=90,
    )
    return True
