import functools
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torchvision.transforms.functional as F
from diffusers import DiffusionPipeline
from diffusers.image_processor import PipelineImageInput
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from einops import rearrange
from PIL import Image

from ....exports.transformer_engine import apply_fp8_autowrap
from ....models.diffusion.cache import DreamTransferAdaCache, FBCache
from ....models.diffusion.dream_transfer import (
    DreamTransferControlNet3DModel,
    DreamTransferEncoder3DModel,
    DreamTransferMultiControlNet3DModel,
    DreamTransferTransformer3DModel,
    T5TextEncoder,
    get_vae,
)
from ....nn import ModuleDict
from ....schedulers import EDMRESMultistepScheduler


class DreamTransferPipeline(DiffusionPipeline):
    def __init__(
        self,
        text_encoder,
        transformer,
        vae,
        scheduler,
        controlnet=None,
        encoder=None,
    ):
        self.register_modules(
            text_encoder=text_encoder,
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
            controlnet=controlnet,
            encoder=encoder,
        )
        self.vae_scale_factor_temporal = self.vae.temporal_compression_factor if getattr(self, 'vae', None) else 8
        self.vae_scale_factor_spatial = self.vae.spatial_compression_factor if getattr(self, 'vae', None) else 8
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
        self.cache = None

        model_names = ['text_encoder']
        if encoder is not None:
            model_names.append('encoder')
            self._optional_components.append('encoder')
        if controlnet is not None:
            model_names.append('controlnet')
            self._optional_components.append('controlnet')
        model_names.extend(['transformer', 'vae'])
        self.model_cpu_offload_seq = '->'.join(model_names)

    @classmethod
    def from_pretrained(
        cls,
        transformer_model_path,
        text_encoder_model_path,
        vae_model_path,
        controlnet_model_path=None,
        encoder_model_path=None,
        lora_model_path=None,
        lora_fuse=False,
    ):
        transformer = DreamTransferTransformer3DModel.from_pretrained(transformer_model_path)
        transformer.to(torch.bfloat16)
        text_encoder = T5TextEncoder(text_encoder_model_path)
        vae = get_vae(vae_model_path)
        scheduler = EDMRESMultistepScheduler()
        kwargs = dict(
            text_encoder=text_encoder,
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
        )
        if controlnet_model_path is not None:
            if isinstance(controlnet_model_path, dict):
                controlnets = dict()
                encoders = dict()
                for name, model_path in controlnet_model_path.items():
                    controlnet = DreamTransferControlNet3DModel.from_pretrained(model_path)
                    controlnet.to(torch.bfloat16)
                    if controlnet.fp8:
                        apply_fp8_autowrap(controlnet, use_autocast_during_eval=True)
                    if controlnet.encoder is not None:
                        encoders[name] = controlnet.encoder
                        controlnet.encoder = None
                    controlnets[name] = controlnet
                kwargs['controlnet'] = DreamTransferMultiControlNet3DModel(controlnets)
                if len(encoders) > 0:
                    assert encoder_model_path is None
                    kwargs['encoder'] = ModuleDict(encoders)
            else:
                controlnet = DreamTransferControlNet3DModel.from_pretrained(controlnet_model_path)
                controlnet.to(torch.bfloat16)
                if controlnet.fp8:
                    apply_fp8_autowrap(controlnet, use_autocast_during_eval=True)
                if controlnet.encoder is not None:
                    assert encoder_model_path is None
                    kwargs['encoder'] = controlnet.encoder
                    controlnet.encoder = None
                kwargs['controlnet'] = controlnet
        if encoder_model_path is not None:
            encoder = DreamTransferEncoder3DModel.from_pretrained(encoder_model_path)
            encoder.to(torch.bfloat16)
            kwargs['encoder'] = encoder
        if lora_model_path is not None:
            transformer.load_lora(lora_model_path, fuse=lora_fuse)
        if transformer.fp8:
            apply_fp8_autowrap(transformer, use_autocast_during_eval=True)
        pipe = cls(**kwargs)
        return pipe

    def set_attn_op(self, backend):
        self.transformer.set_attn_op(backend)
        if self.controlnet is not None:
            if isinstance(self.controlnet, DreamTransferMultiControlNet3DModel):
                for model in self.controlnet.nets.values():
                    model.set_attn_op(backend)
            else:
                self.controlnet.set_attn_op(backend)

    def set_cache(self, mode, **kwargs):
        if mode == 'fbcache':
            self.cache = FBCache(self.transformer, **kwargs)
        elif mode == 'adacache':
            self.cache = DreamTransferAdaCache(self.transformer, **kwargs)
        else:
            assert False

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
    ):
        if prompt_embeds is None:
            if isinstance(prompt, str):
                prompt = [prompt]
            assert len(prompt) == self.transformer.n_views
            prompt_embeds = self.text_encoder.encode_prompts(prompt)[0]
        if self.do_classifier_free_guidance:
            if negative_prompt_embeds is None:
                if negative_prompt is None:
                    negative_prompt = [''] * prompt_embeds.shape[0]
                elif isinstance(negative_prompt, str):
                    negative_prompt = [negative_prompt]
                assert len(negative_prompt) == self.transformer.n_views
                negative_prompt_embeds = self.text_encoder.encode_prompts(negative_prompt)[0]
        prompt_embeds = rearrange(prompt_embeds, 'v t d -> 1 (v t) d')
        if self.do_classifier_free_guidance:
            negative_prompt_embeds = rearrange(negative_prompt_embeds, 'v t d -> 1 (v t) d')
        return prompt_embeds, negative_prompt_embeds

    def encode_video(self, video, height, width, device):
        video = self.video_processor.preprocess_video(video, height, width)
        video = video.to(device=device, dtype=self.vae.dtype)
        video = rearrange(video, 'b c (v t) h w -> (b v) c t h w', v=self.transformer.n_views)
        video_latent = self.vae.encode(video) * self.vae.scaling_factor
        video_latent = rearrange(video_latent, '(b v) c t h w -> b c (v t) h w', v=self.transformer.n_views)
        return video_latent

    def prepare_latents(
        self,
        shape: tuple,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        seed: int = -1,
        rand_type: str = 'torch',
        latents: Optional[torch.Tensor] = None,
        is_strength_max: bool = True,
    ) -> torch.Tensor:
        if latents is None:
            if rand_type == 'np':
                if seed > 0:
                    rng = np.random.RandomState(seed)
                else:
                    rng = np.random.RandomState(seed)
                random_array = rng.standard_normal(shape).astype(np.float32)
                latents = torch.from_numpy(random_array).to(dtype=dtype, device=device)
            elif rand_type == 'torch':
                generator = torch.Generator(device=device)
                if seed > 0:
                    generator.manual_seed(seed)
                latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            else:
                assert False
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        if is_strength_max:
            latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_cond_latents(
        self,
        image: PipelineImageInput,
        num_frames: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        if isinstance(image, list) or isinstance(image, torch.Tensor):
            images = image
        else:
            images = [image]
        n_views = self.transformer.n_views
        if len(images) == 0:
            latents_shape = [
                1,
                self.vae.channel,
                n_views * ((num_frames - 1) // self.vae_scale_factor_temporal + 1),
                height // self.vae_scale_factor_spatial,
                width // self.vae_scale_factor_spatial,
            ]
            cond_latents = torch.zeros(latents_shape, device=device, dtype=dtype)
            cond_masks = torch.zeros((1, 1, cond_latents.shape[2], 1, 1), device=device, dtype=dtype)
            return cond_latents, cond_masks
        assert len(images) % n_views == 0
        num_images = len(images) // self.transformer.n_views
        assert num_images <= num_frames and (num_images - 1) % self.vae_scale_factor_temporal == 0
        num_cond_frames = 1 + (num_images - 1) // self.vae_scale_factor_temporal
        if isinstance(images, torch.Tensor):
            cond_images = 2.0 * images - 1.0
        else:
            cond_images = [np.array(image) for image in images]
            cond_images = np.stack(cond_images, axis=0)
            cond_images = rearrange(cond_images, 't h w c -> t c h w')
            cond_images = cond_images / 127.5 - 1.0
            cond_images = torch.from_numpy(cond_images)
        cond_images = cond_images.to(device, dtype)
        cond_images = F.resize(
            cond_images,
            size=(height, width),  # type: ignore
            interpolation=F.InterpolationMode.BICUBIC,
            antialias=True,
        )
        cond_images = rearrange(cond_images, '(v t) c h w -> v c t h w', v=n_views)
        padding_shape = list(cond_images.shape)
        padding_shape[2] = num_frames - cond_images.shape[2]
        padding_frames = cond_images.new_zeros(padding_shape)
        cond_images = torch.cat([cond_images, padding_frames], dim=2)
        cond_latents = self.vae.encode(cond_images) * self.vae.scaling_factor
        cond_masks = torch.zeros((n_views, 1, cond_latents.shape[2], 1, 1), device=device, dtype=dtype)
        cond_masks[:, :, :num_cond_frames] = 1
        cond_latents = rearrange(cond_latents, 'v c t h w -> 1 c (v t) h w')
        cond_masks = rearrange(cond_masks, 'v c t h w -> 1 c (v t) h w')
        return cond_latents, cond_masks

    def prepare_control_video(self, control_video, height, width, device, dtype):
        def _encode(video, encoder=None):
            video = self.video_processor.preprocess_video(video, height, width)
            video = video.to(device=device, dtype=dtype)
            video = rearrange(video, 'b c (v t) h w -> (b v) c t h w', v=self.transformer.n_views)
            if encoder is not None:
                video_latent = encoder(video)
            else:
                video_latent = self.vae.encode(video) * self.vae.scaling_factor
            video_latent = rearrange(video_latent, '(b v) c t h w -> b c (v t) h w', v=self.transformer.n_views)
            return video_latent

        if isinstance(control_video, dict):
            control_latents = dict()
            for key, val in control_video.items():
                if self.encoder is not None:
                    if isinstance(self.encoder, ModuleDict):
                        encoder_i = functools.partial(self.encoder, key)
                    else:
                        encoder_i = self.encoder
                else:
                    encoder_i = None
                control_latent = _encode(val, encoder_i)
                if self.do_classifier_free_guidance:
                    control_latent = torch.cat([control_latent, control_latent])
                control_latents[key] = control_latent
        else:
            control_latents = _encode(control_video, self.encoder)
            if self.do_classifier_free_guidance:
                control_latents = torch.cat([control_latents, control_latents])
        return control_latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1.0

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        image: PipelineImageInput = None,
        video: Optional[Union[List[Image.Image]]] = None,
        guidance_scale: float = 7.0,
        num_inference_steps: int = 35,
        fps: int = 24,
        num_frames: int = 121,
        height: int = 704,
        width: int = 1280,
        latents: Optional[torch.Tensor] = None,
        extra_latents: Optional[torch.Tensor] = None,
        control_video: Optional[Union[List[Image.Image], Dict[str, List[Image.Image]]]] = None,
        control_weight: Optional[Union[float, Dict[str, float]]] = 1.0,
        frame_repeat: int = 1,
        seed: int = -1,
        rand_type: str = 'torch',
        augment_sigma: float = 0.001,
        sigma_max: float = 80.0,
        output_type: Optional[str] = 'pil',
    ):
        if self.cache is not None:
            self.cache.num_steps = num_inference_steps
            self.cache.reset()
        self._guidance_scale = guidance_scale
        self._interrupt = False

        batch_size = 1
        device = self._execution_device
        dtype = self.transformer.dtype
        self.vae.set_pixel_chunk_duration(num_frames)

        generator = None
        if seed > 0:
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(prompt, negative_prompt=negative_prompt)
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        prompt_embeds = prompt_embeds.to(dtype)

        self.scheduler.config.sigma_max = sigma_max
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        sigmas = self.scheduler.sigmas.to(device)

        # Prepare latent variables
        if video is not None:
            assert len(video) == self.transformer.n_views * num_frames
            latents = self.encode_video(video, height, width, device=device)
            noise = randn_tensor(latents.shape, generator=generator, device=device, dtype=torch.float32)
            latents = latents + noise * sigma_max
        else:
            shape = (
                batch_size,
                self.vae.channel,
                self.transformer.n_views * ((num_frames - 1) // self.vae_scale_factor_temporal + 1),
                int(height) // self.vae_scale_factor_spatial,
                int(width) // self.vae_scale_factor_spatial,
            )
            latents = self.prepare_latents(
                shape,
                dtype=torch.float32,
                device=device,
                seed=seed,
                rand_type=rand_type,
                latents=latents,
            )

        padding_mask = torch.zeros(1, 1, height, width, device=device, dtype=dtype)
        fps = torch.tensor([fps], device=device, dtype=dtype)
        if self.do_classifier_free_guidance:
            padding_mask = torch.cat([padding_mask, padding_mask], dim=0)
            fps = torch.cat([fps, fps], dim=0)

        if image is not None:
            cond_latents, cond_masks = self.prepare_cond_latents(
                image,
                num_frames,
                height,
                width,
                device=device,
                dtype=torch.float32,
            )
            cond_sigma = torch.tensor([augment_sigma], device=device, dtype=cond_latents.dtype)
            masks_input = cond_masks.repeat(batch_size, 1, 1, cond_latents.shape[-2], cond_latents.shape[-1])
        else:
            cond_latents = cond_masks = None

        if control_video is not None:
            if isinstance(control_video, dict):
                for key in control_video:
                    assert len(control_video[key]) == self.transformer.n_views * num_frames
            else:
                assert len(control_video) == self.transformer.n_views * num_frames
            control_latents = self.prepare_control_video(control_video, height, width, device=device, dtype=dtype)
            if isinstance(control_latents, dict) and isinstance(control_weight, dict):
                assert len(control_latents) == len(control_weight)
                for key in control_latents:
                    assert key in control_weight
            elif isinstance(control_latents, dict) and isinstance(control_weight, float):
                control_weights = dict()
                for key in control_latents:
                    control_weights[key] = control_weight
                control_weight = control_weights
            else:
                assert isinstance(control_latents, torch.Tensor) and isinstance(control_weight, float)

        if self.transformer.n_views > 1:
            frame_repeats = torch.zeros((1, self.transformer.n_views), device=device, dtype=dtype)
            if self.do_classifier_free_guidance:
                frame_repeats = torch.cat([frame_repeats, frame_repeats], dim=0)
            frame_repeats[:, -1] = frame_repeat
            frame_repeats[:, -2] = frame_repeat
        else:
            frame_repeats = None

        # Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                latent_model_input = latents
                if image is not None:
                    cond_noise = torch.randn(
                        cond_latents.shape,
                        device=device,
                        dtype=torch.float32,
                        generator=generator,
                    )
                    latent_model_input = self.scheduler.add_condition_inputs(
                        latent_model_input,
                        sigmas[i],
                        cond_sample=cond_latents,
                        cond_mask=cond_masks,
                        cond_noise=cond_noise,
                        cond_sigma=cond_sigma,
                    )
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                if image is not None:
                    latent_model_input = torch.cat([latent_model_input, masks_input], dim=1)
                if extra_latents is not None:
                    latent_model_input = torch.cat([latent_model_input, extra_latents], dim=1)
                if self.do_classifier_free_guidance:
                    latent_model_input = torch.cat([latent_model_input] * 2)
                latent_model_input = latent_model_input.to(dtype)

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0]).to(latent_model_input.dtype)

                if control_video is not None:
                    x_ctrl = self.controlnet(
                        latent_model_input,
                        timestep,
                        crossattn_emb=prompt_embeds,
                        fps=fps,
                        padding_mask=padding_mask,
                        frame_repeat=frame_repeats,
                        control_cond=control_latents,
                        control_weight=control_weight,
                    )
                else:
                    x_ctrl = None

                noise_pred = self.transformer(
                    latent_model_input,
                    timestep,
                    crossattn_emb=prompt_embeds,
                    fps=fps,
                    padding_mask=padding_mask,
                    frame_repeat=frame_repeats,
                    x_ctrl=x_ctrl,
                )

                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_text + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                noise_pred = noise_pred.float()

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred,
                    timestep=t,
                    sample=latents,
                    cond_sample=cond_latents,
                    cond_mask=cond_masks,
                    return_dict=False,
                )[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if not output_type == 'latent':
            latents = latents.to(self.vae.dtype) / self.vae.scaling_factor
            latents = rearrange(latents, 'b c (v t) h w -> (b v) c t h w', v=self.transformer.n_views)
            video = self.vae.decode(latents)
            video = rearrange(video, '(b v) c t h w -> b c (v t) h w', v=self.transformer.n_views)
            video = self.video_processor.postprocess_video(video=video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        return video
