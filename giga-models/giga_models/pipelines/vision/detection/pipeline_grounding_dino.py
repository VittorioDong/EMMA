import groundingdino.datasets.transforms as T
import torch
from groundingdino.util.inference import load_model, predict
from torchvision.ops import box_convert

from ...pipeline import BasePipeline


class GroundingDINOPipeline(BasePipeline):
    def __init__(self, config_path, model_path):
        self.device = 'cpu'
        self.model = load_model(config_path, model_path, device=self.device)
        self.transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def to(self, device):
        self.device = device
        self.model.to(self.device)
        return self

    def get_grounding_boxes(self, image, image_shape, det_labels, box_threshold=0.35, text_threshold=0.25):
        if not isinstance(det_labels, list):
            det_labels = [det_labels]
        caption = ''
        for det_label in det_labels:
            det_label = det_label.lower().strip()
            caption += '{}.'.format(det_label)
        boxes, logits, phrases = predict(
            model=self.model,
            image=image,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device,
        )
        h, w = image_shape[0], image_shape[1]
        boxes = boxes * torch.Tensor([w, h, w, h])
        boxes = box_convert(boxes=boxes, in_fmt='cxcywh', out_fmt='xyxy').numpy()
        logits = logits.numpy()
        return boxes, phrases, logits

    def __call__(self, image, det_labels, box_threshold=0.35, text_threshold=0.25):
        image_shape = (image.height, image.width)
        image = self.transform(image, None)[0]
        pred_boxes, pred_labels, pred_scores = self.get_grounding_boxes(
            image, image_shape, det_labels, box_threshold=box_threshold, text_threshold=text_threshold
        )
        return pred_boxes, pred_labels, pred_scores
