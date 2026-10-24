from controlnet_aux import ContentShuffleDetector

from ...pipeline import BasePipeline


class ContentShufflePipeline(BasePipeline):
    def __init__(self):
        self.model = ContentShuffleDetector()

    def __call__(self, image):
        return self.model(image).resize(image.size)
