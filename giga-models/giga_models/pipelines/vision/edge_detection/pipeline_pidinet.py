from controlnet_aux import PidiNetDetector

from .... import utils
from ...pipeline import BasePipeline


class PidiNetPipeline(BasePipeline):
    def __init__(self, model_name):
        self.model = utils.wrap_call(PidiNetDetector.from_pretrained)(model_name)
        self.device = 'cpu'

    def __call__(self, image, safe=False, scribble=False):
        return self.model(image, safe=safe, scribble=scribble).resize(image.size)
