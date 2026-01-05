import os
import torch
import numpy as np
from typing import Optional

from depth_anything_3.api import DepthAnything3


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

        # prediction.processed_images : [N, H, W, 3] uint8   array
        print(prediction.processed_images.shape)
        # prediction.depth            : [N, H, W]    float32 array
        print(prediction.depth.shape)  
        # prediction.conf             : [N, H, W]    float32 array
        print(prediction.conf.shape)  
        # prediction.extrinsics       : [N, 3, 4]    float32 array # opencv w2c or colmap format
        print(prediction.extrinsics.shape)
        # prediction.intrinsics       : [N, 3, 3]    float32 array
        print(prediction.intrinsics.shape)
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
