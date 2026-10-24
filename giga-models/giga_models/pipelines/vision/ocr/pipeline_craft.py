import sys
from collections import OrderedDict

import cv2
import numpy as np
import torch
from torch.autograd import Variable
from torchvision.transforms import Normalize

from ...pipeline import BasePipeline

try:
    import craft  # noqa F401
except ImportError:
    from ....utils import git_clone

    repo_path = git_clone('https://github.com/clovaai/CRAFT-pytorch.git')
    sys.path.insert(0, repo_path)

import craft_utils
import imgproc
from craft import CRAFT

mean = (0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255)
std = (0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255)
norm = Normalize(mean, std)


class OCRCRAFTPipeline(BasePipeline):
    def __init__(self, model_path):
        self.net = CRAFT()
        self.net.load_state_dict(copyStateDict(torch.load(model_path)))
        self.net.eval()
        self.device = 'cpu'

    def to(self, device):
        self.device = device
        self.net.to(self.device)
        return self

    def __call__(self, image):
        image = np.array(image.convert('RGB')).astype(np.uint8)
        bboxes, polys, score_text = get_text_bboxes(self.net, image, device=self.device)
        ocr_score = get_ocr_score(bboxes, image.shape[1], image.shape[0])  # ocr_score<=0.07
        return ocr_score


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith('module'):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = '.'.join(k.split('.')[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def get_text_bboxes(net, image, text_threshold=0.7, link_threshold=0.4, low_text=0.4, poly=False, device='cpu'):
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, 1280, interpolation=cv2.INTER_LINEAR, mag_ratio=1.5)
    ratio_h = ratio_w = 1 / target_ratio
    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
    x = x.to(device)
    # forward pass
    with torch.no_grad():
        y, feature = net(x)
    # make score and link map
    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()
    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)
    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None:
            polys[k] = boxes[k]
    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)
    return boxes, polys, ret_score_text


def get_ocr_score(bboxes, image_width, image_height):
    ocr = 0
    for b in bboxes:
        ocr += np.sqrt((b[0][0] - b[1][0]) ** 2 + (b[0][1] - b[1][1]) ** 2) * np.sqrt((b[1][0] - b[2][0]) ** 2 + (b[1][1] - b[2][1]) ** 2)
    ocr /= image_width * image_height
    return ocr
