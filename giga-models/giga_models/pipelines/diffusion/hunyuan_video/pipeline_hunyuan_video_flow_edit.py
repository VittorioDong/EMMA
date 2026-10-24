from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from diffusers.pipelines.hunyuan_video.pipeline_output import HunyuanVideoPipelineOutput
from diffusers.utils import is_torch_xla_available, logging

from .pipeline_hunyuan_video import DEFAULT_PROMPT_TEMPLATE, HunyuanVideoPipeline, retrieve_timesteps

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class HunyuanVideoFlowEditPipeline(HunyuanVideoPipeline):
    def encode_video(self, video, height, width, device):
        video = self.video_processor.preprocess_video(video, height, width)
        video = video.to(device=device, dtype=self.vae.dtype)
        video_latent = self.vae.encode(video, return_dict=False)[0].mode()
        video_latent = video_latent * self.vae.config.scaling_factor
        return video_latent

    @torch.no_grad()
    def __call__(
        self,
        src_video: Union[str, torch.Tensor],
        src_prompt: Union[str, List[str]] = None,
        src_prompt_2: Optional[Union[str, List[str]]] = None,
        dst_prompt: Union[str, List[str]] = None,
        dst_prompt_2: Optional[Union[str, List[str]]] = None,
        src_guidance_scale: float = 1.5,
        dst_guidance_scale: float = 5.5,
        num_inference_steps: int = 50,
        skip_steps: int = 0,
        refine_steps: int = 0,
        n_avg: int = 1,
        height: int = 720,
        width: int = 1280,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        output_type: Optional[str] = 'pil',
        return_dict: bool = True,
        prompt_template: Dict[str, Any] = DEFAULT_PROMPT_TEMPLATE,
        max_sequence_length: int = 256,
        seed: Optional[int] = -1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    ):
        device = self._execution_device
        dtype = self.transformer.dtype

        if seed > 0:
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)

        # Encode input prompt
        src_prompt_embeds, src_pooled_prompt_embeds, src_prompt_attention_mask = self.encode_prompt(
            prompt=src_prompt,
            prompt_2=src_prompt_2,
            prompt_template=prompt_template,
            num_videos_per_prompt=1,
            device=device,
            max_sequence_length=max_sequence_length,
        )
        src_prompt_embeds = src_prompt_embeds.to(dtype)
        src_prompt_attention_mask = src_prompt_attention_mask.to(dtype)
        if src_pooled_prompt_embeds is not None:
            src_pooled_prompt_embeds = src_pooled_prompt_embeds.to(dtype)

        dst_prompt_embeds, dst_pooled_prompt_embeds, dst_prompt_attention_mask = self.encode_prompt(
            prompt=dst_prompt,
            prompt_2=dst_prompt_2,
            prompt_template=prompt_template,
            num_videos_per_prompt=1,
            device=device,
            max_sequence_length=max_sequence_length,
        )
        dst_prompt_embeds = dst_prompt_embeds.to(dtype)
        dst_prompt_attention_mask = dst_prompt_attention_mask.to(dtype)
        if dst_pooled_prompt_embeds is not None:
            dst_pooled_prompt_embeds = dst_pooled_prompt_embeds.to(dtype)

        # Prepare latent variables
        video_latents = self.encode_video(src_video, height, width, device).to(dtype)

        sigmas = np.linspace(1.0, 0.0, num_inference_steps + 1)[:-1]
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
        )
        sigmas = self.scheduler.sigmas
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        src_guidance = torch.tensor([src_guidance_scale] * video_latents.shape[0], dtype=dtype, device=device) * 1000.0
        dst_guidance = torch.tensor([dst_guidance_scale] * video_latents.shape[0], dtype=dtype, device=device) * 1000.0

        edit_zt = video_latents.clone()
        src_x0 = video_latents.clone()

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if i < skip_steps:
                    continue

                timestep = t.expand(edit_zt.shape[0]).to(edit_zt.dtype)

                if i < num_inference_steps - refine_steps:
                    # Calculate the average of the V predictions
                    v_delta_avg = torch.zeros_like(src_x0)
                    for k in range(n_avg):
                        fwd_noise = torch.randn(
                            src_x0.shape,
                            dtype=src_x0.dtype,
                            device=device,
                            generator=generator,
                        )
                        src_zt = (1 - sigmas[i]) * src_x0 + sigmas[i] * fwd_noise
                        dst_zt = edit_zt + src_zt - src_x0

                        src_vt = self.transformer(
                            hidden_states=src_zt,
                            timestep=timestep,
                            encoder_hidden_states=src_prompt_embeds,
                            encoder_attention_mask=src_prompt_attention_mask,
                            pooled_projections=src_pooled_prompt_embeds,
                            guidance=src_guidance,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                        )[0]
                        dst_vt = self.transformer(
                            hidden_states=dst_zt,
                            timestep=timestep,
                            encoder_hidden_states=dst_prompt_embeds,
                            encoder_attention_mask=dst_prompt_attention_mask,
                            pooled_projections=dst_pooled_prompt_embeds,
                            guidance=dst_guidance,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                        )[0]
                        v_delta_avg += (1 / n_avg) * (dst_vt - src_vt)

                    # propagate direct ODE
                    edit_zt = edit_zt.to(torch.float32)
                    edit_zt = edit_zt + v_delta_avg * (sigmas[i + 1] - sigmas[i])
                    edit_zt = edit_zt.to(dtype)

                else:
                    if i == num_inference_steps - refine_steps:
                        # initialize SDEDIT-style generation phase
                        fwd_noise = torch.randn(
                            src_x0.shape,
                            dtype=src_x0.dtype,
                            device=device,
                            generator=generator,
                        )
                        src_zt = (1 - sigmas[i]) * src_x0 + sigmas[i] * fwd_noise
                        dst_zt = edit_zt + src_zt - src_x0
                    else:
                        dst_zt = edit_zt

                    dst_vt = self.transformer(
                        hidden_states=dst_zt,
                        timestep=timestep,
                        encoder_hidden_states=dst_prompt_embeds,
                        encoder_attention_mask=dst_prompt_attention_mask,
                        pooled_projections=dst_pooled_prompt_embeds,
                        guidance=dst_guidance,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]

                    # propagate direct ODE
                    edit_zt = edit_zt.to(torch.float32)
                    edit_zt = edit_zt + dst_vt * (sigmas[i + 1] - sigmas[i])
                    edit_zt = edit_zt.to(dtype)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        latents = edit_zt
        if not output_type == 'latent':
            latents = latents.to(self.vae.dtype) / self.vae.config.scaling_factor
            video = self.vae.decode(latents, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return HunyuanVideoPipelineOutput(frames=video)
