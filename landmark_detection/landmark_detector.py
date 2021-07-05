import torch
import numpy as np

import cv2
from PIL import Image

from pathlib import Path
import os.path as osp
import sys

# This is require for properly loading model
path_to_models = osp.dirname(osp.dirname(osp.abspath(__file__)))
path_to_models = str(Path(path_to_models) / 'SAN')
sys.path.append(path_to_models)

from SAN import models
from SAN.datasets import pil_loader
from SAN.san_vision import transforms
from SAN.datasets import Point_Meta


class LandmarkDetector(object):
    def __init__(self, model_path: Path, device: str = None, benchmark: bool = True):
        self.model_path = model_path

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        snapshot = torch.load(self.model_path, map_location=self.device)
        self.param = snapshot['args']
        self.transform = transforms.Compose([
            transforms.PreCrop(self.param.pre_crop_expand),
            transforms.TrainScale2WH((self.param.crop_width, self.param.crop_height)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.net = models.__dict__[self.param.arch](self.param.modelconfig, None)
        self.net.train(False).to(self.device)

        weights = models.remove_module_dict(snapshot['state_dict'])
        self.net.load_state_dict(weights)
        self.meta = None

    def _preprocess_image(self, img: np.ndarray) -> torch.Tensor:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img, 'RGB')
        img_tensor, self.meta = self.transform(img, self.meta)
        return img_tensor

    def predict(self, d: dict):
        try:
            image = d['image']
            bbox = d['box']

            self.meta = Point_Meta(self.param.num_pts, None, np.array(bbox), '', 'custom')

            image = self._preprocess_image(image)
            temp_save_wh = self.meta.temp_save_wh

            cropped_size = torch.IntTensor([temp_save_wh[1], temp_save_wh[0], temp_save_wh[2], temp_save_wh[3]])

            # network forward
            with torch.no_grad():
                inputs = image.unsqueeze(0).to(self.device)
                _, batch_locs, batch_scos, _ = self.net(inputs)

            # obtain the locations on the image in the orignial size
            np_batch_locs, np_batch_scos, cropped_size = batch_locs.cpu().numpy(), batch_scos.cpu().numpy(), cropped_size.numpy()
            locations, scores = np_batch_locs[0, :-1, :], np.expand_dims(np_batch_scos[0, :-1], -1)

            scale_h, scale_w = cropped_size[0] * 1. / inputs.size(-2), cropped_size[1] * 1. / inputs.size(-1)

            locations[:, 0], locations[:, 1] = locations[:, 0] * scale_w + cropped_size[2], locations[:, 1] * scale_h + \
                                               cropped_size[3]

            res_arr = np.ones((locations.shape[0], locations.shape[1] + 1))
            res_arr[:, :2] = locations.round().astype(np.float)
            res_arr[:, 2] = scores.reshape(scores.shape[0])

            return_dict = {
                'landmarks': res_arr,
                'error': ""
            }

            return return_dict

        except Exception as err:
            return_dict = {
                'landmarks': "",
                'error': str(err)
            }
            return return_dict
