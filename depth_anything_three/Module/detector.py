import os
import torch
import numpy as np
from typing import Optional, Union, List

from camera_control.Method.data import toNumpy
from camera_control.Module.camera import Camera

from depth_anything_3.api import DepthAnything3, Prediction


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
        use_ray_pose: bool=False,
    ) -> Prediction:
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
        return_dict: bool=False,
    ) -> Optional[Union[List[Camera], Prediction]]:
        images = render_data_dict['images']
        extrinsics = render_data_dict['extrinsics']
        intrinsics = render_data_dict['intrinsics']

        image_list = [image for image in images]

        prediction = self.detect(image_list, extrinsics, intrinsics, use_ray_pose)

        if return_dict:
            return prediction

        extrinsic_44_list = []
        for i in range(images.shape[0]):
            extrinsic_44 = np.zeros((4, 4), dtype=prediction.extrinsics.dtype)
            extrinsic_44[:3, :4] = prediction.extrinsics[i]
            extrinsic_44[3, :] = np.array([0, 0, 0, 1], dtype=prediction.extrinsics.dtype)
            extrinsic_44_list.append(extrinsic_44)
        pred_extrinsics = extrinsic_44_list

        camera_list = []

        for i in range(images.shape[0]):
            camera = Camera.fromDA3Pose(pred_extrinsics[i], prediction.intrinsics[i])

            camera.loadImage((prediction.processed_images[i].astype(np.float64) / 255.0)[..., ::-1])
            camera.loadDepth(prediction.depth[i], prediction.conf[i])

            camera_list.append(camera)
        return camera_list

    @torch.no_grad()
    def detectCameras(
        self,
        camera_list: List[Camera],
        use_ray_pose: bool = False,
        return_dict: bool=False,
    ) -> Optional[Union[List[Camera], Prediction]]:
        images = []
        extrinsics = []
        intrinsics = []

        for camera in range(camera_list):
            image = camera.image_cv
            extrinsic = toNumpy(camera.world2cameraCV, np.float32)
            intrinsic = toNumpy(camera.intrinsic, np.float32)

            images.append(image)
            extrinsics.append(extrinsic)
            intrinsics.append(intrinsic)

        render_data = {
            'images': images,
            'extrinsics': extrinsics,
            'intrinsics': intrinsics,
        }

        return self.detectRenderData(render_data, use_ray_pose, return_dict)

    @torch.no_grad()
    def detectRenderDataFile(
        self,
        render_data_file_path: str,
        use_ray_pose: bool = False,
        return_dict: bool=False,
    ) -> Optional[Union[List[Camera], Prediction]]:
        if not os.path.exists(render_data_file_path):
            print('[ERROR][Detector::detectRenderDataFile]')
            print('\t render data file not exist!')
            print('\t render_data_file_path:', render_data_file_path)
            return None

        render_data_dict = np.load(render_data_file_path, allow_pickle=True).item()

        return self.detectRenderData(render_data_dict, use_ray_pose, return_dict)
