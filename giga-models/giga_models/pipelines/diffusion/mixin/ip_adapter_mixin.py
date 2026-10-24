import numpy as np
import torch
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from .... import utils
from ....models import IPAdapterModel
from .utils import add_control_model_name, repeat_data


class IPAdapterMixin:
    def load_ip_adapter_model(self, pretrained_model_path, image_model_path=None, face_model_path=None, **kwargs):
        kwargs.setdefault('torch_dtype', self.dtype)
        ip_adapter = IPAdapterModel.from_pretrained(pretrained_model_path, **kwargs)
        ip_adapter.to_unet(self.unet, is_train=False)
        self.ip_adapter = ip_adapter
        if image_model_path is None:
            image_model_path = utils.get_model_path('laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
        self.ip_adapter_image_processor = CLIPImageProcessor()
        self.ip_adapter_image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_model_path, **kwargs)
        if face_model_path is not None:
            face_model = FaceAnalysis(
                name='buffalo_l',
                root=face_model_path,
                allowed_modules=['detection', 'recognition'],
            )
            face_model.prepare(ctx_id=-1, det_size=(640, 640))
            self.ip_adapter_face_model = face_model
        add_control_model_name(['ip_adapter', 'ip_adapter_image_encoder'])

    def _get_ip_adapter_image_embeds(
        self,
        image,
        batch_size=1,
        num_images_per_prompt=1,
        num_frames=None,
        method=1,
    ):
        device = self._execution_device
        if method == 1:
            pixel_values = self.ip_adapter_image_processor(
                images=image,
                size=dict(height=224, width=224),
                do_center_crop=False,
                return_tensors='pt',
            ).pixel_values
            pixel_values = pixel_values.to(device=device, dtype=self.ip_adapter_image_encoder.dtype)
            image_embeds = self.ip_adapter_image_encoder(pixel_values).image_embeds
            image_embeds = repeat_data(
                image_embeds,
                batch_size=batch_size,
                num_images_per_prompt=num_images_per_prompt,
                num_frames=num_frames,
            )
            if self.do_classifier_free_guidance:
                uncond_image_embeds = torch.zeros_like(image_embeds)
                image_embeds = torch.cat([uncond_image_embeds, image_embeds])
            return image_embeds

        elif method == 2:
            pixel_values = self.ip_adapter_image_processor(
                images=image,
                size=dict(height=224, width=224),
                do_center_crop=False,
                return_tensors='pt',
            ).pixel_values
            pixel_values = pixel_values.to(device=device, dtype=self.ip_adapter_image_encoder.dtype)
            image_embeds = self.ip_adapter_image_encoder(pixel_values, output_hidden_states=True).hidden_states[-2]
            image_embeds = repeat_data(
                image_embeds,
                batch_size=batch_size,
                num_images_per_prompt=num_images_per_prompt,
                num_frames=num_frames,
            )
            if self.do_classifier_free_guidance:
                uncond_image_embeds = self.ip_adapter_image_encoder(torch.zeros_like(pixel_values), output_hidden_states=True).hidden_states[-2]
                uncond_image_embeds = repeat_data(
                    uncond_image_embeds,
                    batch_size=batch_size,
                    num_images_per_prompt=num_images_per_prompt,
                    num_frames=num_frames,
                )
                image_embeds = torch.cat([uncond_image_embeds, image_embeds])
            return image_embeds

        elif method == 3:
            image = np.array(image)[:, :, ::-1]
            faces = self.ip_adapter_face_model.get(image)
            faces = sorted(faces, key=lambda x: -(x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
            faceid_embeds = repeat_data(
                faceid_embeds,
                batch_size=batch_size,
                num_images_per_prompt=num_images_per_prompt,
                num_frames=num_frames,
            )
            if self.do_classifier_free_guidance:
                uncond_faceid_embeds = torch.zeros_like(faceid_embeds)
                faceid_embeds = torch.cat([uncond_faceid_embeds, faceid_embeds])
            faceid_embeds = faceid_embeds.to(device=device, dtype=self.ip_adapter.dtype)
            return faceid_embeds, faces[0]
        else:
            assert False

    def encode_ip_adapter_image(
        self,
        image,
        batch_size=1,
        num_images_per_prompt=1,
        num_frames=None,
    ):
        if image is None:
            return None
        class_name = self.ip_adapter.image_proj.__class__.__name__
        if class_name == 'ImageProjection':
            image_embeds = self._get_ip_adapter_image_embeds(
                image,
                batch_size=batch_size,
                num_images_per_prompt=num_images_per_prompt,
                num_frames=num_frames,
                method=1,
            )
            image_embeds = self.ip_adapter.image_proj(image_embeds)
            return image_embeds
        elif class_name in ('IPAdapterFullImageProjection', 'IPAdapterPlusImageProjection'):
            image_embeds = self._get_ip_adapter_image_embeds(
                image,
                batch_size=batch_size,
                num_images_per_prompt=num_images_per_prompt,
                num_frames=num_frames,
                method=2,
            )
            image_embeds = self.ip_adapter.image_proj(image_embeds)
            return image_embeds
        elif class_name == 'FaceProjection':
            faceid_embeds = self._get_ip_adapter_image_embeds(
                image,
                batch_size=batch_size,
                num_images_per_prompt=num_images_per_prompt,
                num_frames=num_frames,
                method=3,
            )[0]
            faceid_embeds = self.ip_adapter.image_proj(faceid_embeds)
            return faceid_embeds
        elif class_name == 'FacePlusProjection':
            faceid_embeds, face = self._get_ip_adapter_image_embeds(
                image,
                batch_size=batch_size,
                num_images_per_prompt=num_images_per_prompt,
                num_frames=num_frames,
                method=3,
            )
            image = np.array(image)[:, :, ::-1]
            face_image = face_align.norm_crop(image, landmark=face.kps, image_size=224)
            image_embeds = self._get_ip_adapter_image_embeds(
                face_image,
                batch_size=batch_size,
                num_images_per_prompt=num_images_per_prompt,
                num_frames=num_frames,
                method=2,
            )
            image_embeds = self.ip_adapter.image_proj(faceid_embeds, image_embeds)
            return image_embeds
        else:
            assert False
