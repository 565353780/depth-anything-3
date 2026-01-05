import os

from depth_anything_three.Module.detector import Detector


def demo():
    home = os.environ['HOME']
    model_folder_path = home + "/chLi/Model/DepthAnythingV3/DA3-GIANT-1.1/"
    device = 'cuda:5'
    render_data_file_path = home + "/chLi/Dataset/MM/Match/nezha/da3/render_data.npy"
    use_ray_pose = False

    detector = Detector(model_folder_path, device)
    prediction = detector.detectRenderDataFile(render_data_file_path, use_ray_pose)
    return True
