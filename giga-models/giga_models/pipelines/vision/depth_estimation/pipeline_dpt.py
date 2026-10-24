import numpy as np
import torch
from PIL import Image
from transformers import DPTFeatureExtractor, DPTForDepthEstimation, pipeline

from .... import utils
from ...pipeline import BasePipeline


class DPTForDepthEstimationPipeline(BasePipeline):
    def __init__(self, model_name):
        self.model = utils.wrap_call(pipeline)('depth-estimation', model=model_name)
        self.device = 'cpu'

    def to(self, device):
        self.device = device
        self.model.device = torch.device(device)
        self.model.model.to(device)
        return self

    def __call__(self, image):
        image = self.model(image)['depth']
        image = np.array(image)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        image = Image.fromarray(image)
        return image


class DPTForDepthEstimation2Pipeline(BasePipeline):
    def __init__(self, model_name):
        self.feature_extractor = DPTFeatureExtractor.from_pretrained(
            model_name,
            local_files_only=utils.is_offline_mode(),
        )
        self.model = DPTForDepthEstimation.from_pretrained(
            model_name,
            local_files_only=utils.is_offline_mode(),
        )
        self.device = 'cpu'

    def to(self, device):
        self.device = device
        self.model.to(device)
        return self

    def __call__(self, image):
        pixel_values = self.feature_extractor(images=image, return_tensors='pt').pixel_values.to(self.device)
        with torch.no_grad():
            depth_map = self.model(pixel_values).predicted_depth
        depth_map = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1),
            size=(image.height, image.width),
            mode='bicubic',
            align_corners=False,
        )
        depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        image = torch.cat([depth_map] * 3, dim=1)
        image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
        image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
        return image
