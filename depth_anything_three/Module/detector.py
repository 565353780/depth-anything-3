import os
import torch
import numpy as np
from typing import Optional

from depth_anything_3.api import DepthAnything3

from camera_control.Module.camera import Camera


class Detector(object):
    def __init__(
        self,
        model_folder_path: Optional[str]=None,
        device: str='cuda:0',
    ) -> None:
        self.device = device

        self.model: DepthAnything3 = None

        if model_folder_path is not None:
            self.loadModel(model_folder_path, device)
        return

    def loadModel(
        self,
        model_folder_path: str,
        device: str='cuda:0',
    ) -> bool:
        if not os.path.exists(model_folder_path):
            print('[ERROR][Detector::loadModel]')
            print('\t model folder not exist!')
            print('\t model_folder_path:', model_folder_path)
            return False

        self.device = device

        self.model = DepthAnything3.from_pretrained(model_folder_path)
        self.model = self.model.to(device=device)
        return True

    @torch.no_grad()
    def detect(
        self,
        images: torch.Tensor,
        extrinsics: Optional[np.ndarray]=None,
        intrinsics: Optional[np.ndarray]=None,
        use_ray_pose: bool = False,
    ) -> dict:
        prediction = self.model.inference(
            image=images,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            use_ray_pose=use_ray_pose,
        )
        return prediction

    @torch.no_grad()
    def detectRenderData(
        self,
        render_data_dict: dict,
        use_ray_pose: bool = False,
    ) -> dict:
        images = render_data_dict['images']
        extrinsics = render_data_dict['extrinsics']
        intrinsics = render_data_dict['intrinsics']

        image_list = [image for image in images]

        return self.detect(image_list, extrinsics, intrinsics, use_ray_pose)

    @torch.no_grad()
    def detectRenderDataFile(
        self,
        render_data_file_path: str,
        use_ray_pose: bool = False,
    ) -> Optional[dict]:
        if not os.path.exists(render_data_file_path):
            print('[ERROR][Detector::detectRenderDataFile]')
            print('\t render data file not exist!')
            print('\t render_data_file_path:', render_data_file_path)
            return None

        render_data_dict = np.load(render_data_file_path, allow_pickle=True).item()

        return self.detectRenderData(render_data_dict, use_ray_pose)

    @torch.no_grad()
    def visPrediction(
        self,
        prediction: dict,
        conf_min: float = 0.5,
    ) -> dict:
        height, width = prediction.processed_images.shape[1:3]

        points_list = []

        for i in range(prediction.depth.shape[0]):
            depth = prediction.depth[i]  # [H, W]
            conf = prediction.conf[i]    # [H, W]
            extrinsic = prediction.extrinsics[i]  # [3, 4]
            intrinsic = prediction.intrinsics[i]  # [3, 3]

            # 创建相机对象
            camera = Camera(
                width=width,
                height=height,
                fx=intrinsic[0][0],
                fy=intrinsic[1][1],
                cx=intrinsic[0][2],
                cy=intrinsic[1][2],
            )

            world2camera_cv = np.eye(4, dtype=extrinsic.dtype)
            world2camera_cv[:3, :] = extrinsic
            camera.setWorld2CameraByWorld2CameraCV(world2camera_cv)

            # 根据置信度阈值筛选
            valid_mask = conf >= conf_min  # [H, W]

            # 获取有效像素的索引
            valid_indices = np.where(valid_mask)  # (row_indices, col_indices)
            v_indices = valid_indices[0]  # 行索引（对应v）
            u_indices = valid_indices[1]  # 列索引（对应u）

            # 获取筛选后的深度和置信度
            valid_depth = depth[valid_mask]  # [N_valid]

            # 生成UV坐标（归一化到[0, 1]）
            # UV坐标系：原点在左下角(0,0)，u向右，v向上
            u_normalized = u_indices / width   # [N_valid]
            v_normalized = v_indices / height  # [N_valid]
            uv = np.stack([u_normalized, v_normalized], axis=-1)  # [N_valid, 2]

            # 使用projectUV2Points将UV和深度转换为世界坐标系的点
            points = camera.projectUV2Points(uv, valid_depth)  # [N_valid, 3]

            print(points.shape)
            points_list.append(points)

        points = torch.cat(points_list, dim=0)

        return points
