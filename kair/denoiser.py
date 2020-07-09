import cv2
import numpy as np
import torch
from kair.utils import utils_image as util
from kair.models.network_ffdnet import FFDNet as net


class Denoiser:
    def __init__(self, model_path, device, noise_level=15):
        model = net(in_nc=3, out_nc=3, nc=96, nb=12, act_mode='R')
        model.load_state_dict(torch.load(model_path), strict=True)
        model.eval()
        for k, v in model.named_parameters():
            v.requires_grad = False
        self._model = model.to(device)
        self._device = device

        self.noise_level = noise_level

    def __call__(self, in_img, is_bgr=True, *args, **kwargs):
        """
        gets bgr returns bgr
        :param in_img:
        :param is_bgr:
        :param args:
        :param kwargs:
        :return:
        """
        in_img = cv2.cvtColor(in_img, cv2.COLOR_BGR2RGB)
        in_img = util.uint2single(in_img)
        np.random.seed(seed=0)  # for reproducibility

        in_img += np.random.normal(0, self.noise_level / 255., in_img.shape)
        in_img = util.single2tensor4(in_img)
        in_img = in_img.to(self._device)

        sigma = torch.full((1, 1, 1, 1), self.noise_level / 255.).type_as(in_img)

        out_img = self._model(in_img, sigma)
        out_img = util.tensor2uint(out_img)
        out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
        return out_img

