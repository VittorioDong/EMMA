import torch
from controlnet_aux import NormalBaeDetector

from .... import utils
from ...pipeline import BasePipeline


class NormalBaePipeline(BasePipeline):
    def __init__(self, model_name):
        with torch.cuda.device(utils.get_device()):
            self.model = utils.wrap_call(NormalBaeDetector.from_pretrained)(model_name)
        self.model.model.to('cpu')
        self.device = 'cpu'

    def to(self, device):
        self.device = device
        self.model.model.to(device)
        return self

    def __call__(self, image):
        return self.model(image).resize(image.size)
