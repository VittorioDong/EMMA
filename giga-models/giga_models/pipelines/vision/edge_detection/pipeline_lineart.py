import torch
from controlnet_aux import LineartDetector

from .... import utils
from ...pipeline import BasePipeline


class LineartPipeline(BasePipeline):
    def __init__(self, model_name):
        with torch.cuda.device(utils.get_device()):
            self.model = utils.wrap_call(LineartDetector.from_pretrained)(model_name)
        self.model.model.to('cpu')
        self.model.model_coarse.to('cpu')
        self.device = 'cpu'

    def to(self, device):
        self.device = device
        self.model.model.to(device)
        self.model.model_coarse.to(device)
        return self

    def __call__(self, image):
        return self.model(image).resize(image.size)
