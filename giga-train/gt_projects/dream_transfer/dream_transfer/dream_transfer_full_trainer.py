import functools

import torch
from einops import rearrange
from giga_models import (
    DreamTransferControlNet3DModel,
    DreamTransferTransformer3DModel,
    LoRAPeftWrapper,
)
from giga_models.models.diffusion.dream_transfer import get_vae
from giga_models.nn import EDMLoss, ModuleDict
from giga_train import Trainer, utils
from peft import LoraConfig


class DreamTransferFullTrainer(Trainer):
    def get_models(self, model_config):
        model = dict()
        # vae
        self.vae = get_vae(model_config.vae_model_path)
        self.vae.to(self.device)
        # transformer
        transformer_model_path = model_config.transformer_model_path
        transformer = DreamTransferTransformer3DModel.from_pretrained(transformer_model_path)
        transformer.requires_grad_(True)
        transformer.to(self.device, dtype=self.dtype)
        with_lora = model_config.get('with_lora', False)
        if with_lora:
            lora_rank = model_config.lora_rank
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_rank,
                init_lora_weights=True,
                target_modules=['to_q.0', 'to_k.0', 'to_v.0', 'to_out.0'],
            )
            transformer.add_adapter(lora_config)
            transformer = LoRAPeftWrapper(transformer)
            model.update(transformer=transformer)
        else:
            self.apply_activation_checkpointing([transformer])
        # self.transformer = transformer
        model.update(transformer=transformer)

        # controlnet
        controlnet_model_path = model_config.get('controlnet_model_path', None)
        if controlnet_model_path is not None:
            controlnet = DreamTransferControlNet3DModel.from_pretrained(controlnet_model_path)
        else:
            controlnet_cfg = model_config.get('controlnet', dict())
            controlnet = DreamTransferControlNet3DModel(**controlnet_cfg)
        self.with_encoder = controlnet.encoder is not None
        model.update(controlnet=controlnet)
        # loss
        self.edm_loss = EDMLoss(sigma_method=3)
        # model
        checkpoint = model_config.get('checkpoint', None)
        strict = model_config.get('strict', True)
        self.load_checkpoint(checkpoint, list(model.values()), strict=strict)
        model = ModuleDict(model)
        model.to(self.dtype)
        model.train()
        if with_lora:
            model.load_state_dict_mode = 'each'
        return model

    def forward_step(self, batch_dict):
        controlnet = functools.partial(self.model, 'controlnet')
        if hasattr(self, 'transformer'):
            transformer = self.transformer
        else:
            transformer = functools.partial(self.model, 'transformer')
        images = batch_dict['images'] # torch.Size([1, 121, 3, 480, 1920])
        prompt_embeds = batch_dict['prompt_embeds']
        batch_size = images.shape[0]
        num_frames = images.shape[1]
        self.vae.set_pixel_chunk_duration(num_frames)
        # condition
        padding_mask = torch.zeros((batch_size, 1, images.shape[-2], images.shape[-1]), dtype=self.dtype, device=self.device)
        fps = torch.tensor([batch_dict['fps']], device=self.device, dtype=self.dtype)
        # images
        latents = self.forward_vae(images)
        input_latents, timesteps = self.edm_loss.add_noise(latents)
        # ref images
        if 'ref_images' in batch_dict:
            ref_images = batch_dict['ref_images']
            ref_masks = batch_dict['ref_masks'].to(self.dtype)
            ref_latents = self.forward_vae(ref_images)
            # augment
            augment_sigma = torch.normal(mean=-3.0, std=2.0, size=(batch_size,), device=self.device)
            augment_sigma = torch.exp(augment_sigma).to(dtype=self.dtype)
            while len(augment_sigma.shape) < len(ref_latents.shape):
                augment_sigma = augment_sigma.unsqueeze(-1)
            noise = torch.randn_like(ref_latents)
            augment_latents = ref_latents + noise * augment_sigma
            augment_c_in = 1 / ((augment_sigma**2 + self.edm_loss.sigma_data**2) ** 0.5)
            augment_latents = augment_latents * augment_c_in
            # add
            input_latents = ref_masks * augment_latents + (1 - ref_masks) * input_latents
            input_masks = ref_masks.repeat(1, 1, 1, input_latents.shape[-2], input_latents.shape[-1])
            input_latents = torch.cat([input_latents, input_masks], dim=1)
        input_latents = input_latents.to(self.dtype)
        timesteps = timesteps.to(self.dtype)
        prompt_embeds = prompt_embeds.to(self.dtype)
        # encoder
        cn_images = batch_dict['cn_images']
        if cn_images.shape[1]>121:
            cn_images = cn_images[:,:121,:,:,:]
        #print("cn_images",cn_images.shape)
        #[1,121,480,640,3]
        if self.with_encoder:
            if cn_images.shape[1] == num_frames:
                cn_images = rearrange(cn_images, 'b t c h w -> b c t h w')
            cn_latents = cn_images
        else:
            cn_latents = self.forward_vae(cn_images)
        # controlnet
        #print('input_latents',input_latents.shape)
        #print('timesteps',timesteps.shape)
        #print('prompt_embeds',prompt_embeds.shape)
        #print('padding_mask',padding_mask.shape)
        #print('cn_latents',cn_latents.shape)
        x_ctrl = controlnet(
            x=input_latents,
            timesteps=timesteps,
            crossattn_emb=prompt_embeds,
            padding_mask=padding_mask,
            fps=fps,
            control_cond=cn_latents.to(self.dtype),
        )
        utils.to_dtype(x_ctrl, self.dtype)
        # forward
        model_pred = transformer(
            x=input_latents,
            timesteps=timesteps,
            crossattn_emb=prompt_embeds,
            padding_mask=padding_mask,
            fps=fps,
            x_ctrl=x_ctrl,
        )
        # compute loss
        denoised_latents = self.edm_loss.denoise(model_pred.float())
        if 'ref_images' in batch_dict:
            denoised_latents = ref_masks * ref_latents + (1 - ref_masks) * denoised_latents
        loss = self.edm_loss.compute_loss(denoised_latents)
        return loss

    def forward_vae(self, images):
        images = images.to(self.vae.dtype)
        with torch.no_grad():
            images = rearrange(images, 'b t c h w -> b c t h w')
            #print(images.shape)
            latents = self.vae.encode(images) * self.vae.scaling_factor
        return latents
