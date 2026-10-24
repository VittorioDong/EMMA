import cv2
import numpy as np
from PIL import Image
from rtmlib import Wholebody
from rtmlib.visualization import coco133, openpose134

from ...pipeline import BasePipeline


class RTMPosePipeline(BasePipeline):
    def __init__(self, det_path, pose_path, to_openpose=True, conf_thres=0.6):
        self.model = Wholebody(
            det=det_path,
            pose=pose_path,
            to_openpose=to_openpose,
            mode='performance',
            backend='onnxruntime',
            device='cpu',
        )
        self.to_openpose = to_openpose
        self.conf_thres = conf_thres
        self.device = 'cpu'

    def to(self, device):
        if device == 'cuda':
            device = 'cuda:0'
        self.device = device
        if device == 'cpu':
            providers = ['CPUExecutionProvider']
            provider_options = None
        elif device.startswith('cuda'):
            providers = ['CUDAExecutionProvider']
            provider_options = [{'device_id': int(device[-1])}]
        else:
            assert False
        self.model.det_model.session.set_providers(providers, provider_options)
        self.model.pose_model.session.set_providers(providers, provider_options)
        return self

    def __call__(self, input_image, return_image=False):
        input_image = np.array(input_image, dtype=np.uint8)
        keypoints, scores = self.model(input_image)

        if return_image:
            black_frame = np.zeros_like(input_image)
            skeleton_frame = plot_person_kpts(
                black_frame,
                keypoints,
                scores=scores,
                kpt_thr=self.conf_thres,
                openpose_format=True,
                line_width=4,
            )
            output_image = Image.fromarray(skeleton_frame)
            return output_image
        else:
            return keypoints, scores


def plot_person_kpts(
    pose_vis_img: np.ndarray,
    keypoints: np.ndarray,
    scores: np.ndarray,
    kpt_thr: float = 0.6,
    openpose_format: bool = True,
    line_width: int = 4,
) -> np.ndarray:
    """Plot a single person in-place update the pose image."""
    for kpts, ss in zip(keypoints, scores):
        try:
            pose_vis_img = draw_skeleton(pose_vis_img, kpts, ss, kpt_thr=kpt_thr, openpose_format=openpose_format, line_width=line_width)
        except ValueError as e:
            print(f'Error in draw_skeleton func, {e}')

    return pose_vis_img


def draw_skeleton(
    img: np.ndarray,
    keypoints: np.ndarray,
    scores: np.ndarray,
    kpt_thr: float = 0.6,
    openpose_format: bool = True,
    radius: int = 2,
    line_width: int = 4,
):
    skeleton_topology = openpose134 if openpose_format else coco133
    assert len(keypoints.shape) == 2
    keypoint_info, skeleton_info = (
        skeleton_topology['keypoint_info'],
        skeleton_topology['skeleton_info'],
    )
    vis_kpt = [s >= kpt_thr for s in scores]
    link_dict = {}
    for i, kpt_info in keypoint_info.items():
        kpt_color = tuple(kpt_info['color'])
        link_dict[kpt_info['name']] = kpt_info['id']

        kpt = keypoints[i]

        if vis_kpt[i]:
            img = cv2.circle(img, (int(kpt[0]), int(kpt[1])), int(radius), kpt_color, -1)

    for i, ske_info in skeleton_info.items():
        link = ske_info['link']
        pt0, pt1 = link_dict[link[0]], link_dict[link[1]]

        if vis_kpt[pt0] and vis_kpt[pt1]:
            link_color = ske_info['color']
            kpt0 = keypoints[pt0]
            kpt1 = keypoints[pt1]

            img = cv2.line(img, (int(kpt0[0]), int(kpt0[1])), (int(kpt1[0]), int(kpt1[1])), link_color, thickness=line_width)

    return img
