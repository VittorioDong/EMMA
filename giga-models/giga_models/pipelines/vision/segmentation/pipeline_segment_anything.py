import random

import numpy as np
import torch
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry

from ...pipeline import BasePipeline


class SegmentAnythingPipeline(BasePipeline):
    def __init__(self, model_path, model_name):
        self.sam = sam_model_registry[model_name]()
        self.sam.load_state_dict(torch.load(model_path))
        self.sam_predictor = SamPredictor(self.sam)
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)
        self.device = 'cpu'

    def to(self, device):
        self.device = device
        self.sam.to(device)
        return self

    def masks_to_points(self, masks):
        if len(masks) == 0:
            return None, None
        random.seed(1912)
        point_coords = []
        num_obj = len(masks)
        max_num_point = -1
        for mask in masks:
            idxs = np.nonzero(mask)
            num_points = min(max(1, int(len(idxs[0]) * 0.01)), 32)
            sampled_idx = random.sample(range(0, len(idxs[0])), num_points)
            points = []
            for i in range(len(idxs)):
                points.append(idxs[i][sampled_idx])
            points = np.array(points).reshape(2, -1).transpose(1, 0)[:, ::-1]
            point_coords.append(points)
            if len(points) > max_num_point:
                max_num_point = len(points)
        if max_num_point > 0:
            point_coords_np = np.zeros((num_obj, max_num_point, 2))
            point_labels_np = -np.ones((num_obj, max_num_point))
            for i in range(num_obj):
                point_coords_np[i, : len(point_coords[i])] = point_coords[i]
                point_labels_np[i, : len(point_coords[i])] = 1
            return point_coords_np, point_labels_np
        else:
            return None, None

    def __call__(self, image, point_coords=None, point_labels=None, boxes=None, masks=None):
        image = np.array(image)
        if masks is not None:
            assert point_coords is None and point_labels is None
            point_coords, point_labels = self.masks_to_points(masks)
        if point_coords is not None:
            if point_labels is None:
                point_labels = np.ones(point_coords.shape[:-1], dtype=np.int)
            point_coords = self.sam_predictor.transform.apply_coords(point_coords, image.shape[:2])
            point_coords = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
            point_labels = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
        if boxes is not None:
            boxes = self.sam_predictor.transform.apply_boxes(boxes, image.shape[:2])
            boxes = torch.as_tensor(boxes, dtype=torch.float, device=self.device)
        if point_coords is not None or boxes is not None:
            self.sam_predictor.set_image(image)
            masks = self.sam_predictor.predict_torch(
                point_coords=point_coords,
                point_labels=point_labels,
                boxes=boxes,
                multimask_output=False,
            )[0]
            masks = masks.squeeze(1).detach().cpu().numpy()
        else:
            masks = self.mask_generator.generate(image)
            masks = np.stack([mask['segmentation'] for mask in masks])
        return masks
