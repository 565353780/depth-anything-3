import sys
sys.path.append('../camera-control')

import os
import numpy as np
import open3d as o3d

from depth_anything_three.Module.detector import Detector


def demo():
    home = os.environ['HOME']
    model_folder_path = home + "/chLi/Model/DepthAnythingV3/DA3-GIANT-1.1/"
    device = 'cuda:5'
    render_data_file_path = home + "/chLi/Dataset/MM/Match/nezha/da3/render_data.npy"
    use_ray_pose = False

    detector = Detector(model_folder_path, device)

    prediction = detector.detectRenderDataFile(render_data_file_path, use_ray_pose)

    points = detector.visPrediction(prediction)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.cpu().numpy().astype(np.float64))
    o3d.io.write_point_cloud(home + "/chLi/Dataset/MM/Match/nezha/da3/render_data.ply", pcd)
    return True
