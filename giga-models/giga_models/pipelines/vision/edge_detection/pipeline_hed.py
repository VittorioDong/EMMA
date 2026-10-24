from controlnet_aux import HEDdetector

from .... import utils
from ...pipeline import BasePipeline


class HEDPipeline(BasePipeline):
    def __init__(self, model_name):
        self.model = utils.wrap_call(HEDdetector.from_pretrained)(model_name)
        self.device = 'cpu'

    def to(self, device):
        self.device = device
        self.model.netNetwork.to(device)
        return self

    def __call__(self, image, scribble=False):
        return self.model(image, scribble=scribble).resize(image.size)
