import math

import torch
import torch.nn as nn
from diffusers.image_processor import VaeImageProcessor

from .... import utils
from ....models import GuiderModel
from .utils import add_control_model_name, repeat_data


class GuiderMixin:
    def load_guider_model(self, pretrained_model_path, **kwargs):
        kwargs.setdefault('torch_dtype', self.dtype)
        if isinstance(pretrained_model_path, list):
            pretrained_model_paths = pretrained_model_path
        else:
            pretrained_model_paths = [pretrained_model_path]
        guiders = []
        for pretrained_model_path in pretrained_model_paths:
            guider = GuiderModel.from_pretrained(pretrained_model_path, **kwargs)
            guiders.append(guider)
        self.guider = nn.ModuleList(guiders)
        self.guider_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor,
            do_convert_rgb=True,
            do_normalize=True,
        )
        add_control_model_name('guider')

    def forward_guider(
        self,
        image,
        height,
        width,
        batch_size=1,
        num_images_per_prompt=1,
        num_frames=None,
        chunk_size=None,
    ):
        if image is None:
            return None
        device = self._execution_device
        images = image if isinstance(image, list) else [image]
        latents_list = []
        for i, image in enumerate(images):
            image = self.guider_image_processor.preprocess(image, height=height, width=width)
            image = image.to(device=device, dtype=self.guider[i].dtype)
            total_size = len(image)
            chunk_size = min(chunk_size or total_size, total_size)
            if chunk_size == total_size:
                latents = self.guider[i](image)
            else:
                slices = list(torch.arange(total_size).chunk(int(math.ceil(total_size / chunk_size))))
                latents = utils.apply_slice(self.guider[i], image, slices=slices)
            latents = repeat_data(
                latents,
                batch_size=batch_size,
                num_images_per_prompt=num_images_per_prompt,
                num_frames=num_frames,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
            )
            if self.do_classifier_free_guidance:
                latents[: latents.shape[0] // 2] = 0
            latents_list.append(latents)
        latents = torch.cat(latents_list, dim=1)
        return latents
