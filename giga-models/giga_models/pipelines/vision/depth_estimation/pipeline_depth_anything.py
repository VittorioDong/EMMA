import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from ...pipeline import BasePipeline


class DepthAnythingForDepthEstimationPipeline(BasePipeline):
    def __init__(self, model_path):
        self.image_processor = AutoImageProcessor.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        self.model = AutoModelForDepthEstimation.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        self.device = 'cpu'

    def to(self, device):
        self.device = device
        self.model.to(device)
        return self

    def __call__(self, image):
        inputs = self.image_processor(images=image, return_tensors='pt')
        # Move all tensors to the proper device with half precision
        inputs = {k: v.to(self.device, dtype=torch.float16) for k, v in inputs.items()}

        with torch.no_grad():
            with torch.autocast(device_type=torch.device(self.device).type, dtype=torch.float16):
                outputs = self.model(**inputs)
                depth_map = outputs.predicted_depth

        # Interpolate the predicted depth to the original image size
        depth_map = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1),
            size=image.size[::-1],  # PIL image size is (width, height), interpolate expects (height, width)
            mode='bicubic',
            align_corners=False,
        )

        # Convert the output tensor to a numpy array and save as a depth image
        depth_map = depth_map.squeeze().unsqueeze(-1)
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8) * 255.0
        image = torch.cat([depth_map] * 3, dim=-1)
        image = image.cpu().numpy().clip(0, 255).astype(np.uint8)
        image = Image.fromarray(image)
        return image
