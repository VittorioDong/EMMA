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


class HunyuanVideoRFInversionPipeline(HunyuanVideoPipeline):
    def encode_video(self, video, height, width, device):
        video = self.video_processor.preprocess_video(video, height, width)
        video = video.to(device=device, dtype=self.vae.dtype)
        video_latent = self.vae.encode(video, return_dict=False)[0].mode()
        video_latent = video_latent * self.vae.config.scaling_factor
        return video_latent

    @torch.no_grad()
    def invert(
        self,
        video: List[torch.Tensor],
        prompt: Union[str, List[str]] = None,
        prompt_2: Union[str, List[str]] = None,
        guidance_scale: float = 0.0,
        num_inversion_steps: int = 28,
        gamma: float = 0.5,
        height: int = 720,
        width: int = 1280,
        prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        prompt_template: Dict[str, Any] = DEFAULT_PROMPT_TEMPLATE,
        max_sequence_length: int = 256,
    ):
        device = self._execution_device

        # Encode input prompt
        prompt_embeds, pooled_prompt_embeds, prompt_attention_mask = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_template=prompt_template,
            num_videos_per_prompt=1,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            device=device,
            max_sequence_length=max_sequence_length,
        )

        transformer_dtype = self.transformer.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        prompt_attention_mask = prompt_attention_mask.to(transformer_dtype)
        if pooled_prompt_embeds is not None:
            pooled_prompt_embeds = pooled_prompt_embeds.to(transformer_dtype)

        latents = self.encode_video(video, height, width, device).to(transformer_dtype)

        # prepare timesteps
        sigmas = np.linspace(1.0, 0.0, num_inversion_steps + 1)[:-1]
        timesteps, num_inversion_steps = retrieve_timesteps(
            self.scheduler,
            num_inversion_steps,
            device,
            sigmas=sigmas,
        )
        timesteps = torch.cat([self.scheduler.timesteps, torch.tensor([0], dtype=timesteps.dtype, device=device)])
        sigmas = self.scheduler.sigmas

        guidance = torch.tensor([guidance_scale] * latents.shape[0], dtype=transformer_dtype, device=device) * 1000.0

        y_t = latents
        y_1 = torch.randn_like(y_t)

        with self.progress_bar(total=num_inversion_steps) as progress_bar:
            for i in range(num_inversion_steps, 0, -1):
                timestep = timesteps[i].expand(y_t.shape[0]).to(transformer_dtype)

                u_t_i = self.transformer(
                    hidden_states=y_t,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_attention_mask,
                    pooled_projections=pooled_prompt_embeds,
                    guidance=guidance,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]

                u_t_i_cond = (y_1 - y_t) / (1 - sigmas[i])
                u_t_i = u_t_i + gamma * (u_t_i_cond - u_t_i)

                y_t = y_t + u_t_i * (sigmas[i - 1] - sigmas[i])

                progress_bar.update()

        return y_t, latents

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Union[str, List[str]] = None,
        inverted_latents: Optional[torch.FloatTensor] = None,
        video_latents: Optional[torch.FloatTensor] = None,
        start_timestep: float = 0,
        stop_timestep: float = 0.25,
        eta: float = 1.0,
        decay_eta: Optional[bool] = False,
        eta_decay_power: Optional[float] = 1.0,
        num_inference_steps: int = 50,
        guidance_scale: float = 6.0,
        prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = 'pil',
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        prompt_template: Dict[str, Any] = DEFAULT_PROMPT_TEMPLATE,
        max_sequence_length: int = 256,
    ):
        r"""The call function to the pipeline for generation.

        Returns:
            [`~HunyuanVideoPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`HunyuanVideoPipelineOutput`] is returned, otherwise a `tuple` is returned
                where the first element is a list with the generated images and the second element is a list of `bool`s
                indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.
        """
        self._current_timestep = None
        self._interrupt = False

        device = self._execution_device

        # Encode input prompt
        prompt_embeds, pooled_prompt_embeds, prompt_attention_mask = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_template=prompt_template,
            num_videos_per_prompt=1,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            device=device,
            max_sequence_length=max_sequence_length,
        )

        transformer_dtype = self.transformer.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        prompt_attention_mask = prompt_attention_mask.to(transformer_dtype)
        if pooled_prompt_embeds is not None:
            pooled_prompt_embeds = pooled_prompt_embeds.to(transformer_dtype)

        # Prepare latent variables
        latents = inverted_latents.to(transformer_dtype)

        # Prepare timesteps
        sigmas = np.linspace(1.0, 0.0, num_inference_steps + 1)[:-1]
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
        )
        sigmas = self.scheduler.sigmas
        start_timestep = int(start_timestep * num_inference_steps)
        stop_timestep = min(int(stop_timestep * num_inference_steps), num_inference_steps)

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        # Prepare guidance condition
        guidance = torch.tensor([guidance_scale] * latents.shape[0], dtype=transformer_dtype, device=device) * 1000.0

        y_0 = video_latents.clone()

        # Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                timestep = t.expand(latents.shape[0]).to(transformer_dtype)

                noise_pred = self.transformer(
                    hidden_states=latents,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_attention_mask,
                    pooled_projections=pooled_prompt_embeds,
                    guidance=guidance,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]
                v_t = noise_pred

                v_t_cond = (y_0 - latents) / (0 - sigmas[i])
                eta_t = eta if start_timestep <= i < stop_timestep else 0.0
                if decay_eta:
                    eta_t = eta_t * (1 - i / num_inference_steps) ** eta_decay_power  # Decay eta over the loop
                v_t = v_t + eta_t * (v_t_cond - v_t)

                latents = latents + v_t * (sigmas[i + 1] - sigmas[i])

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

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
