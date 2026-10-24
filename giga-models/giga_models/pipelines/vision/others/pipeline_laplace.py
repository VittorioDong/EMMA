import cv2
import numpy as np

from ...pipeline import BasePipeline


class LaplacePipeline(BasePipeline):
    def __init__(self):
        pass

    def __call__(self, image):
        image = np.array(image.convert('RGB')).astype(np.uint8)
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        laplace = cv2.Laplacian(gray_image, cv2.CV_64F).var()  # laplace >=20
        return laplace
