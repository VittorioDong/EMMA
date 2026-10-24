from controlnet_aux import MLSDdetector

from .... import utils
from ...pipeline import BasePipeline


class MLSDPipeline(BasePipeline):
    def __init__(self, model_name):
        self.model = utils.wrap_call(MLSDdetector.from_pretrained)(model_name)
        self.device = 'cpu'

    def to(self, device):
        self.device = device
        self.model.model.to(device)
        return self

    def __call__(self, image):
        return self.model(image).resize(image.size)
