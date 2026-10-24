import math

import torch
from diffusers import ControlNetModel
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.controlnet import MultiControlNetModel
from diffusers.utils.torch_utils import is_compiled_module

from .... import utils
from .utils import add_control_model_name, repeat_data


class ControlNetMixin:
    def load_controlnet_model(self, pretrained_model_path, **kwargs):
        kwargs.setdefault('torch_dtype', self.dtype)
        if not isinstance(pretrained_model_path, list):
            pretrained_model_path = [pretrained_model_path]
        controlnets = []
        for i in range(len(pretrained_model_path)):
            controlnet = ControlNetModel.from_pretrained(pretrained_model_path[i], **kwargs)
            controlnets.append(controlnet)
        if len(controlnets) == 1:
            controlnet = controlnets[0]
        else:
            controlnet = MultiControlNetModel(controlnets)
        self.controlnet = controlnet
        self.controlnet_image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False)
        add_control_model_name('controlnet')

    def prepare_controlnet(
        self,
        image,
        timesteps,
        height,
        width,
        batch_size=1,
        num_images_per_prompt=1,
        num_frames=None,
        conditioning_scale=1.0,
        guidance_start=0.0,
        guidance_end=1.0,
    ):
        if image is None:
            self.cn_image = None
            return
        controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet
        # align format for control guidance
        if not isinstance(guidance_start, list) and isinstance(guidance_end, list):
            guidance_start = len(guidance_end) * [guidance_start]
        elif not isinstance(guidance_end, list) and isinstance(guidance_start, list):
            guidance_end = len(guidance_start) * [guidance_end]
        elif not isinstance(guidance_start, list) and not isinstance(guidance_end, list):
            mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1
            guidance_start, guidance_end = mult * [guidance_start], mult * [guidance_end]
        # Define call parameters
        if isinstance(controlnet, MultiControlNetModel) and isinstance(conditioning_scale, float):
            conditioning_scale = [conditioning_scale] * len(controlnet.nets)
        # Create tensor stating which controlnets to keep
        keeps = []
        for i in range(len(timesteps)):
            keeps_i = [1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e) for s, e in zip(guidance_start, guidance_end)]
            keeps.append(keeps_i[0] if isinstance(controlnet, ControlNetModel) else keeps_i)
        # Prepare image
        device = self._execution_device
        if isinstance(controlnet, ControlNetModel):
            image = self.controlnet_image_processor.preprocess(image, height=height, width=width)
            image = image.to(device, dtype=self.controlnet.dtype)
            image = repeat_data(
                image,
                batch_size=batch_size,
                num_images_per_prompt=num_images_per_prompt,
                num_frames=num_frames,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
            )

        elif isinstance(controlnet, MultiControlNetModel):
            images = []
            for image_i in image:
                image_i = self.controlnet_image_processor.preprocess(image_i, height=height, width=width)
                image_i = image_i.to(device, dtype=self.controlnet.dtype)
                image_i = repeat_data(
                    image_i,
                    batch_size=batch_size,
                    num_images_per_prompt=num_images_per_prompt,
                    num_frames=num_frames,
                    do_classifier_free_guidance=self.do_classifier_free_guidance,
                )
                images.append(image_i)
            image = images
        else:
            assert False
        # outputs
        self.cn_image = image
        self.cn_conditioning_scale = conditioning_scale
        self.cn_keeps = keeps

    def forward_controlnet(
        self,
        cur_step,
        latent_model_input,
        timestep,
        prompt_embeds,
        added_cond_kwargs=None,
        chunk_size=None,
    ):
        if self.cn_image is None:
            return None, None
        control_model_input = latent_model_input
        controlnet_prompt_embeds = prompt_embeds
        controlnet_added_cond_kwargs = added_cond_kwargs
        if isinstance(self.cn_keeps[cur_step], list):
            cond_scale = [c * s for c, s in zip(self.cn_conditioning_scale, self.cn_keeps[cur_step])]
        else:
            controlnet_cond_scale = self.cn_conditioning_scale
            if isinstance(controlnet_cond_scale, list):
                controlnet_cond_scale = controlnet_cond_scale[0]
            cond_scale = controlnet_cond_scale * self.cn_keeps[cur_step]
        if self.do_classifier_free_guidance:
            total_size = len(control_model_input) // 2
        else:
            total_size = len(control_model_input)
        chunk_size = min(chunk_size or total_size, total_size)
        if chunk_size == total_size:
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                control_model_input,
                timestep,
                encoder_hidden_states=controlnet_prompt_embeds,
                controlnet_cond=self.cn_image,
                conditioning_scale=cond_scale,
                guess_mode=False,
                added_cond_kwargs=controlnet_added_cond_kwargs,
                return_dict=False,
            )
        else:
            slices = list(torch.arange(total_size).chunk(int(math.ceil(total_size / chunk_size))))
            if self.do_classifier_free_guidance:
                for i in range(len(slices)):
                    slices[i] = torch.cat([slices[i], slices[i] + total_size])
            down_block_res_samples, mid_block_res_sample = utils.apply_slice(
                self.controlnet,
                control_model_input,
                timestep,
                encoder_hidden_states=controlnet_prompt_embeds,
                controlnet_cond=self.cn_image,
                conditioning_scale=cond_scale,
                guess_mode=False,
                added_cond_kwargs=controlnet_added_cond_kwargs,
                return_dict=False,
                slices=slices,
            )
        return down_block_res_samples, mid_block_res_sample
