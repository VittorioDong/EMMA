import lama_inpaint
import numpy as np
import torch
import yaml
from lama_inpaint import inpaint_img_with_builded_lama
from omegaconf import OmegaConf
from PIL import Image
from saicinpainting.training.trainers import load_checkpoint
from sttn_video_inpaint import build_sttn_model, inpaint_video_with_builded_sttn

from ...pipeline import BasePipeline


class InpaintAnythingPipeline(BasePipeline):
    def __init__(self, inpainter_target, lama_config_path, lama_model_path, sttn_model_path):
        if inpainter_target == 'lama':
            self.inpainter = build_lama_model(
                config_p=lama_config_path,
                ckpt_p=lama_model_path,
                device='cpu',
            )
        elif inpainter_target == 'sttn':
            self.inpainter = build_sttn_model(
                model_type='sttn',
                ckpt_p=sttn_model_path,
                device='cpu',
            )
        else:
            raise NotImplementedError
        self.inpainter_target = inpainter_target
        self.device = 'cpu'

    def to(self, device):
        self.device = device
        self.inpainter.to(device)
        return self

    def __call__(self, image, mask):
        if self.inpainter_target == 'lama':
            image = np.array(image)
            mask = np.array(mask)
            inpaint_image = inpaint_img_with_builded_lama(self.inpainter, image, mask, device=self.device)
            inpaint_image = Image.fromarray(inpaint_image)
            return inpaint_image
        elif self.inpainter_target == 'sttn':
            inpaint_image = inpaint_video_with_builded_sttn(self.inpainter, [image], [mask], device=self.device)[0]
            return inpaint_image
        else:
            raise NotImplementedError


def build_lama_model(config_p, ckpt_p, device='cuda'):
    device = torch.device(device)

    with open(config_p, 'r') as f:
        model_config = OmegaConf.create(yaml.safe_load(f))

    model_config.training_model.predict_only = True
    model_config.visualizer.kind = 'noop'

    model = load_checkpoint(model_config, ckpt_p, map_location=device, strict=False)
    model.to(device)
    model.freeze()
    return model


setattr(lama_inpaint, 'build_lama_model', build_lama_model)
