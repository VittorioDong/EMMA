import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation

from .... import utils
from ...pipeline import BasePipeline
from .palette import get_palette


class UperNetForSemanticSegmentationPipeline(BasePipeline):
    def __init__(self, model_name):
        self.processor = AutoImageProcessor.from_pretrained(
            model_name,
            local_files_only=utils.is_offline_mode(),
        )
        self.model = UperNetForSemanticSegmentation.from_pretrained(
            model_name,
            local_files_only=utils.is_offline_mode(),
        )
        self.palette = get_palette(rgb=True)
        self.device = 'cpu'

    def to(self, device):
        self.device = device
        self.model.to(device)
        return self

    def __call__(self, image):
        pixel_values = self.processor(image, return_tensors='pt').pixel_values.to(self.device)
        with torch.no_grad():
            outputs = self.model(pixel_values)
        image = self.processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
        image = image.cpu().numpy()
        color_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(self.palette):
            color_image[image == label, :] = color
        color_image = color_image.astype(np.uint8)
        color_image = Image.fromarray(color_image)
        return color_image
