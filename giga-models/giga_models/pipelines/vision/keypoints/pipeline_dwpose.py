import cv2
import numpy as np
import torch
from mmengine.structures import InstanceData
from mmpose.apis import init_model as init_pose_estimator
from mmpose.structures import PoseDataSample, merge_data_samples
from mmpose.structures.bbox import bbox_xyxy2cs, get_warp_matrix

from ...pipeline import BasePipeline

KEYPOINT_INDEXES = {
    'nose': 0,
    'neck': 1,
    'shoulder_left': 2,
    'wrist_left': 3,
    'hand_left': 4,
    'shoulder_right': 5,
    'wrist_right': 6,
    'hand_right': 7,
    'waist_left': 8,
    'knee_left': 9,
    'foot_left': 10,
    'waist_right': 11,
    'knee_right': 12,
    'foot_right': 13,
    'eye_left': 14,
    'eye_right': 15,
    'ear_left': 16,
    'ear_right': 17,
}

instance_mapping_table = dict(
    bbox='bboxes',
    bbox_score='bbox_scores',
    keypoints='keypoints',
    keypoints_cam='keypoints_cam',
    keypoints_visible='keypoints_visible',
    bbox_scale='bbox_scales',
    head_size='head_size',
)


class DWposePipeline(BasePipeline):
    def __init__(self, pose_config, pose_ckpt):
        self.model = DWposeDetector(pose_config, pose_ckpt)
        self.device = 'cpu'

    def to(self, device):
        self.device = device
        self.model.to(device)
        return self

    def __call__(
        self,
        images,
        all_bboxes,
        include_body=True,
        include_hand=True,
        include_face=True,
        include_score=True,
        batch_size=32,
    ):
        if not isinstance(images, list):
            images = [images]
        if not isinstance(all_bboxes, list):
            all_bboxes = [all_bboxes]
        if len(images) != len(all_bboxes):
            raise ValueError('The length of images and all_bboxes must be the same')

        output = self.model(
            images,
            all_bboxes,
            include_body=include_body,
            include_hand=include_hand,
            include_face=include_face,
            include_score=include_score,
            batch_size=batch_size,
        )

        return output


class DWposeDetector:
    def __init__(self, pose_config, pose_ckpt):
        self.pose_estimator = init_pose_estimator(pose_config, pose_ckpt, device='cpu')
        self.device = 'cpu'

    def to(self, device):
        self.device = device
        self.pose_estimator.to(device)
        return self

    def __call__(
        self,
        ori_imgs,
        all_bboxes,
        include_body=True,
        include_hand=True,
        include_face=True,
        include_score=True,
        batch_size=32,
    ):
        def packed_pose(
            results,
            bboxes=torch.zeros([1, 4]),
            meta_keys=(
                'id',
                'img_id',
                'img_path',
                'category_id',
                'crowd_index',
                'ori_shape',
                'img_shape',
                'input_size',
                'input_center',
                'input_scale',
                'flip',
                'flip_direction',
                'flip_indices',
                'raw_ann_info',
                'dataset_name',
            ),
        ):
            data_sample = PoseDataSample()
            gt_instances = InstanceData()
            gt_instances.bboxes = bboxes
            gt_instances.bbox_scores = np.ones((1,))
            for key, packed_key in instance_mapping_table.items():
                if key in results:
                    gt_instances.set_field(results[key], packed_key)
            data_sample.gt_instances = gt_instances
            data_sample.gt_instance_labels = InstanceData().to_tensor()
            img_meta = {k: results[k] for k in meta_keys if k in results}
            data_sample.set_metainfo(img_meta)
            return data_sample

        def prepare_image(ori_img, bbox, input_size, mean_tensor, std_tensor):
            center, scale = bbox_xyxy2cs(bbox, padding=1.25)
            scale = self._fix_aspect_ratio(scale, aspect_ratio=input_size[0] / input_size[1])
            warp_mat = get_warp_matrix(center, scale, 0, output_size=input_size)
            ori_img = np.array(ori_img)
            img = cv2.warpAffine(ori_img, warp_mat, input_size, flags=cv2.INTER_LINEAR)
            img = img[:, :, ::-1].transpose((2, 0, 1))[np.newaxis]
            img_tensor = torch.from_numpy(img.copy()).contiguous().float()
            img_tensor = (img_tensor - mean_tensor) / std_tensor
            return img_tensor, center, scale

        input_size = (288, 384)
        mean_tensor = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1).float()
        std_tensor = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1).float()
        data_samples, ori_img_tensors, frame_idx_list = [], [], []
        for frame_idx, (ori_img, bboxes) in enumerate(zip(ori_imgs, all_bboxes), start=1):
            for bbox_idx, bbox in enumerate(bboxes):
                frame_idx_list.append(frame_idx)
                img_tensor, center, scale = prepare_image(ori_img, bbox, input_size, mean_tensor, std_tensor)
                ori_img_tensors.append(img_tensor)
                metainfo = {
                    'img_path': None,
                    'input_scale': scale,
                    'input_center': center,
                    'flip_indices': [
                        0,
                        2,
                        1,
                        4,
                        3,
                        6,
                        5,
                        8,
                        7,
                        10,
                        9,
                        12,
                        11,
                        14,
                        13,
                        16,
                        15,
                        20,
                        21,
                        22,
                        17,
                        18,
                        19,
                        39,
                        38,
                        37,
                        36,
                        35,
                        34,
                        33,
                        32,
                        31,
                        30,
                        29,
                        28,
                        27,
                        26,
                        25,
                        24,
                        23,
                        49,
                        48,
                        47,
                        46,
                        45,
                        44,
                        43,
                        42,
                        41,
                        40,
                        50,
                        51,
                        52,
                        53,
                        58,
                        57,
                        56,
                        55,
                        54,
                        68,
                        67,
                        66,
                        65,
                        70,
                        69,
                        62,
                        61,
                        60,
                        59,
                        64,
                        63,
                        77,
                        76,
                        75,
                        74,
                        73,
                        72,
                        71,
                        82,
                        81,
                        80,
                        79,
                        78,
                        87,
                        86,
                        85,
                        84,
                        83,
                        90,
                        89,
                        88,
                        112,
                        113,
                        114,
                        115,
                        116,
                        117,
                        118,
                        119,
                        120,
                        121,
                        122,
                        123,
                        124,
                        125,
                        126,
                        127,
                        128,
                        129,
                        130,
                        131,
                        132,
                        91,
                        92,
                        93,
                        94,
                        95,
                        96,
                        97,
                        98,
                        99,
                        100,
                        101,
                        102,
                        103,
                        104,
                        105,
                        106,
                        107,
                        108,
                        109,
                        110,
                        111,
                    ],
                    'input_size': input_size,
                    'dataset_name': 'coco_wholebody',
                    'ori_shape': (512, 896),
                    'img_shape': (512, 896),
                    'pad_shape': (384, 288),
                    'batch_input_shape': (384, 288),
                }
                data_samples.append(packed_pose(metainfo, bboxes=bbox[None]))

        pose_results_list_concat = []
        for idx in range(0, len(ori_img_tensors), batch_size):
            ori_img_tensor = torch.cat(ori_img_tensors[idx : idx + batch_size]).to(self.device)
            pose_results_list = self.pose_estimator.predict(ori_img_tensor, data_samples[idx : idx + batch_size])
            pose_results_list_concat.extend(pose_results_list)

        num_frames = len(ori_imgs)
        keypoints_and_scores = self._process_pose_results(pose_results_list_concat, frame_idx_list, num_frames)
        assert len(keypoints_and_scores) == len(ori_imgs)
        processed_keypoints_dict = []

        for i in range(len(keypoints_and_scores)):
            W, H = ori_imgs[i].size
            keypoints, scores = keypoints_and_scores[i]
            keypoints[..., 0] /= float(W)
            keypoints[..., 1] /= float(H)
            keypoints_per_frame = []
            for j in range(len(all_bboxes[i])):
                body = keypoints[j, :18].copy()
                body_score = scores[j : j + 1, :18]
                for body_idx in range(len(body_score[0])):
                    if body_score[0][body_idx] > 0.3:
                        body_score[0][body_idx] = int(body_idx)
                    else:
                        body_score[0][body_idx] = -1
                un_visible = scores < 0.3
                keypoints[un_visible] = -1
                faces = keypoints[j : j + 1, 24:92]
                hands = keypoints[j : j + 1, 92:113]
                hands = np.vstack([hands, keypoints[j : j + 1, 113:]])
                poses = dict()
                bodies = dict(candidate=body, subset=body_score)
                if include_body:
                    poses.update(bodies=bodies)
                if include_hand:
                    poses.update(hands=hands)
                if include_face:
                    poses.update(faces=faces)
                keypoints_per_frame.append(poses)
            processed_keypoints_dict.append(keypoints_per_frame)
        return processed_keypoints_dict

    def _process_pose_results(self, pose_results_list_concat, frame_idx_list, num_frames):
        keypoints_and_scores = []
        pose_results_dict = {i: [] for i in range(1, num_frames + 1)}

        for pose_results, frame_idx in zip(pose_results_list_concat, frame_idx_list):
            pose_results_dict[frame_idx].append(pose_results)

        for frame_idx in range(1, num_frames + 1):
            pose_results = pose_results_dict[frame_idx]

            if len(pose_results) == 0:
                keypoints = np.zeros((0, 134, 2), dtype=np.float32)
                scores = np.zeros((0, 134), dtype=np.float32)
                keypoints_and_scores.append([keypoints, scores])
                continue

            preds = merge_data_samples(pose_results).pred_instances
            keypoints = preds.get('transformed_keypoints', preds.keypoints)
            scores = preds.get('keypoint_scores', np.ones(keypoints.shape[:-1]))
            visible = preds.get('keypoints_visible', np.ones(keypoints.shape[:-1]))

            keypoints_info = np.concatenate((keypoints, scores[..., None], visible[..., None]), axis=-1)
            neck = np.mean(keypoints_info[:, [5, 6]], axis=1)
            neck[:, 2:4] = np.logical_and(keypoints_info[:, 5, 2:4] > 0.3, keypoints_info[:, 6, 2:4] > 0.3).astype(int)
            keypoints_info = np.insert(keypoints_info, 17, neck, axis=1)

            mmpose_idx = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
            openpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
            keypoints_info[:, openpose_idx] = keypoints_info[:, mmpose_idx]

            keypoints, scores, visible = keypoints_info[..., :2], keypoints_info[..., 2], keypoints_info[..., 3]
            keypoints_and_scores.append([keypoints, scores])

        return keypoints_and_scores

    def _fix_aspect_ratio(self, bbox_scale: np.ndarray, aspect_ratio: float):
        """Reshape the bbox to a fixed aspect ratio.

        Args:
            bbox_scale (np.ndarray): The bbox scales (w, h) in shape (n, 2)
            aspect_ratio (float): The ratio of ``w/h``

        Returns:
            np.darray: The reshaped bbox scales in (n, 2)
        """

        w, h = np.hsplit(bbox_scale, [1])
        bbox_scale = np.where(w > h * aspect_ratio, np.hstack([w, w / aspect_ratio]), np.hstack([h * aspect_ratio, h]))
        return bbox_scale
