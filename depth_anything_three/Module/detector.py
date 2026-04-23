import gc
import os
import torch
import numpy as np

from copy import deepcopy
from typing import Optional, Union, List, Tuple

from camera_control.Method.data import toNumpy
from camera_control.Module.camera import Camera

from depth_anything_3.api import DepthAnything3, Prediction


def _tensorToCPU(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, (list, tuple)):
        converted = [_tensorToCPU(v) for v in value]
        return type(value)(converted)
    if isinstance(value, dict):
        return {k: _tensorToCPU(v) for k, v in value.items()}
    return value


def _predictionToCPU(prediction: Prediction) -> Prediction:
    for name in (
        'depth', 'sky', 'conf', 'extrinsics', 'intrinsics',
        'processed_images',
    ):
        if not hasattr(prediction, name):
            continue
        setattr(prediction, name, _tensorToCPU(getattr(prediction, name)))
    if getattr(prediction, 'gaussians', None) is not None:
        g = prediction.gaussians
        for name in ('means', 'scales', 'rotations', 'harmonics', 'opacities'):
            if hasattr(g, name):
                setattr(g, name, _tensorToCPU(getattr(g, name)))
    if getattr(prediction, 'aux', None) is not None:
        prediction.aux = _tensorToCPU(prediction.aux)
    return prediction


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
        self.model = self.model.to(device='cpu')
        return True

    def _toGPU(self) -> None:
        if self.model is not None:
            self.model.to(device=self.device)
        return

    def _toCPU(self) -> None:
        if self.model is not None:
            self.model.to(device='cpu')
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return

    @torch.no_grad()
    def detect(
        self,
        images: np.ndarray,
        extrinsics: Optional[np.ndarray]=None,
        intrinsics: Optional[np.ndarray]=None,
        use_ray_pose: bool=False,
    ) -> Tuple[List[Camera], Prediction]:
        self._toGPU()
        try:
            prediction = self.model.inference(
                image=images,
                extrinsics=extrinsics,
                intrinsics=intrinsics,
                use_ray_pose=use_ray_pose,
            )

            prediction = _predictionToCPU(prediction)

            extrinsic_44_list = []
            for i in range(len(images)):
                extrinsic_44 = np.zeros((4, 4), dtype=prediction.extrinsics.dtype)
                extrinsic_44[:3, :4] = prediction.extrinsics[i]
                extrinsic_44[3, :] = np.array([0, 0, 0, 1], dtype=prediction.extrinsics.dtype)
                extrinsic_44_list.append(extrinsic_44)
            pred_extrinsics = extrinsic_44_list

            camera_list = []

            for i in range(len(images)):
                if intrinsics is not None:
                    camera = Camera.fromDA3Pose(pred_extrinsics[i], intrinsics[i])
                else:
                    camera = Camera.fromDA3Pose(pred_extrinsics[i], prediction.intrinsics[i])

                camera.loadImage((images[i].astype(np.float64) / 255.0)[..., ::-1])
                camera.loadDepth(prediction.depth[i], prediction.conf[i])

                camera_list.append(camera)

            return camera_list, prediction
        finally:
            self._toCPU()

    @torch.no_grad()
    def detectCameras(
        self,
        camera_list: List[Camera],
        use_ray_pose: bool = False,
    ) -> Union[Tuple[List[Camera], Prediction], Tuple[None, None]]:
        images = []
        extrinsics = []
        intrinsics = []

        for camera in camera_list:
            image = camera.toImageVisCV(use_mask=False)
            extrinsic = toNumpy(camera.world2cameraCV, np.float32)
            intrinsic = toNumpy(camera.intrinsic, np.float32)

            images.append(image)
            extrinsics.append(extrinsic)
            intrinsics.append(intrinsic)

        pred_camera_list, prediction = self.detect(
            images,
            extrinsics,
            intrinsics,
            use_ray_pose,
        )

        final_camera_list = deepcopy(camera_list)
        for i in range(len(final_camera_list)):
            pred_depth = pred_camera_list[i].depth
            pred_conf = pred_camera_list[i].conf
            if isinstance(pred_depth, torch.Tensor):
                pred_depth = pred_depth.detach().cpu()
            if isinstance(pred_conf, torch.Tensor):
                pred_conf = pred_conf.detach().cpu()
            final_camera_list[i].loadDepth(pred_depth, pred_conf)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return final_camera_list, prediction
