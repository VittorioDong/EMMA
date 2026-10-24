import torch
from diffusers.utils.torch_utils import randn_tensor

from ....models.diffusion.referencenet import ReferenceControl, ReferenceNet
from .utils import add_control_model_name


class ReferenceNetMixin:
    def load_referencenet_model(self, pretrained_model_path, **kwargs):
        kwargs.setdefault('torch_dtype', self.dtype)
        referencenet = ReferenceNet.from_pretrained(pretrained_model_path, **kwargs)
        self.referencenet = referencenet
        add_control_model_name('referencenet')

    def prepare_referencenet(
        self,
        image,
        height,
        width,
        batch_size=1,
        num_images_per_prompt=1,
        num_frames=None,
        style_fidelity=0.5,
        generator=None,
    ):
        if image is None:
            self.ref_image_latents = None
            return
        self.ref_writer = ReferenceControl(
            unet=self.referencenet,
            mode='write',
            style_fidelity=style_fidelity,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
        )
        self.ref_reader = ReferenceControl(
            unet=self.unet,
            mode='read',
            style_fidelity=style_fidelity,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
        )
        ref_image_latents = self.encode_vae_image(
            image=image,
            height=height,
            width=width,
            batch_size=batch_size,
            num_images_per_prompt=num_images_per_prompt,
            num_frames=num_frames,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            sample_mode='argmax',
            add_noise=False,
            generator=generator,
        )
        self.ref_image_latents = ref_image_latents

    def forward_referencenet(
        self,
        cur_step,
        timestep,
        add_noise=True,
        num_frames=None,
        chunk_size=None,
        generator=None,
        **kwargs,
    ):
        if self.ref_image_latents is not None:
            if add_noise:
                noise = randn_tensor(
                    self.ref_image_latents.shape,
                    generator=generator,
                    device=self.ref_image_latents.device,
                    dtype=self.ref_image_latents.dtype,
                )
                ref_input = self.scheduler.add_noise(self.ref_image_latents, noise, timestep)
                ref_input = self.scheduler.scale_model_input(ref_input, timestep)
            else:
                ref_input = self.ref_image_latents
                timestep = torch.zeros_like(timestep)
            self.ref_writer.clear()
            self.forward_unet(
                cur_step=cur_step,
                latent_model_input=ref_input,
                timestep=timestep,
                num_frames=num_frames,
                chunk_size=chunk_size,
                unet=self.referencenet,
                **kwargs,
            )
            self.ref_reader.update(self.ref_writer)

    def reset_referencenet(self):
        if self.ref_image_latents is not None:
            self.ref_writer.reset()
            self.ref_reader.reset()
