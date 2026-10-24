import copy
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...pipeline import BasePipeline
from . import pose_utils


class PoseAlignerPipeline(BasePipeline):
    def __init__(self, model_dir):
        self.model = load_model(model_dir)
        self.mean_pose = np.load(os.path.join(model_dir, 'mixamo_data', 'meanpose.npy'))
        self.std_pose = np.load(os.path.join(model_dir, 'mixamo_data', 'stdpose.npy'))
        self.device = 'cpu'

    def to(self, device):
        self.device = device
        self.model.to(device)
        return self

    def __call__(self, src_poses, src_height, src_width, dst_poses_list, dst_height, dst_width):
        return self.align_one(src_poses, src_height, src_width, dst_poses_list, dst_height, dst_width)

    def align_one(self, src_poses, src_height, src_width, dst_poses_list, dst_height, dst_width):
        src_poses = copy.deepcopy(src_poses)
        dst_poses_list = copy.deepcopy(dst_poses_list)
        set_vis_indexes([src_poses])
        set_vis_indexes(dst_poses_list)
        scale_offset_poses([src_poses], scale=(src_width, src_height))
        scale_offset_poses(dst_poses_list, scale=[dst_width, dst_height])
        src_poses = move_poses(src_poses, dst_poses_list[0])
        src_motion = poses_to_motion([src_poses])
        dst_motion = poses_to_motion(dst_poses_list)
        src_motion = src_motion.repeat(dst_motion.shape[2], axis=2)
        input_height, input_width, scale = pad_to_height(512, dst_height, dst_width)
        src_motion = preprocess_motion(src_motion, self.mean_pose, self.std_pose, scale).to(self.device)
        dst_motion = preprocess_motion(dst_motion, self.mean_pose, self.std_pose, scale).to(self.device)
        output_motion = self.model.transfer(dst_motion, src_motion)
        output_motion = postprocess_motion(output_motion, self.mean_pose, self.std_pose, input_width // 2, input_height // 2, scale)
        output_poses_list = motion_to_poses(output_motion, dst_poses_list)
        scale_offset_poses(output_poses_list, scale=[1.0 / dst_width, 1.0 / dst_height])
        clip_poses(output_poses_list)
        for poses in output_poses_list:
            poses.pop('vis_indexes')
        return output_poses_list


def load_model(model_dir):
    sys.path.insert(0, model_dir)
    from model import AutoEncoder2x

    model = AutoEncoder2x(
        mot_en_channels=[30, 64, 96, 128],
        body_en_channels=[28, 32, 48, 64],
        de_channels=[192, 128, 64, 30],
        global_pool=F.max_pool1d,
        convpool=nn.MaxPool1d,
        compress=False,
    )
    weight_path = os.path.join(model_dir, 'model', 'pretrained_skeleton.pth')
    model.load_state_dict(torch.load(weight_path))
    model.eval()
    return model


def pad_to_height(dst_height, src_height, src_width):
    assert dst_height % 16 == 0
    scale = dst_height / src_height
    dst_width = int(src_width * scale)
    return dst_height, dst_width, scale


def poses_to_motion(poses_list):
    motion_list = []
    for poses in poses_list:
        bodies = poses['bodies']
        candidate = bodies['candidate']
        subset = bodies['subset']
        assert len(subset) == 1
        motion = candidate[:14].copy()
        subset = subset[0, :14]
        for i in range(len(subset)):
            if subset[i] < 0:
                motion[i] = -1
            else:
                assert motion[i].min() >= 0
        if len(motion_list) > 0:
            motion[np.where(motion == -1)] = motion_list[-1][np.where(motion == -1)]
        motion_list.append(motion)
    for i in range(len(motion_list) - 1, 0, -1):
        motion_list[i - 1][np.where(motion_list[i - 1] == -1)] = motion_list[i][np.where(motion_list[i - 1] == -1)]
    for i in range(len(motion_list)):
        motion_list[i][np.where(motion_list[i] == -1)] = 0
    pad_size = 8 - len(motion_list) % 8
    for i in range(pad_size):
        motion_list.append(motion_list[-1])
    motion = np.stack(motion_list, axis=2)
    center = (motion[8] + motion[11]) / 2
    motion = np.concatenate([motion[:8], center[None], motion[8:]], axis=0)
    return motion


def motion_to_poses(motion, poses_list):
    vis_indexes = poses_list[0]['vis_indexes']['bodies']
    box = get_box(poses_list[0]['bodies']['candidate'][vis_indexes])
    ground_index = get_ground_index(poses_list[0])
    ground_point = poses_list[0]['bodies']['candidate'][ground_index].copy()
    motion = motion[:, :, : len(poses_list)]
    for i in range(len(poses_list)):
        motion_i = motion[:, :, i]
        motion_i = np.concatenate([motion_i[:8], motion_i[9:]], axis=0)
        candidate = poses_list[i]['bodies']['candidate']
        subset = poses_list[i]['bodies']['subset']
        assert len(subset) == 1
        if subset[0][0] == 0:
            candidate[14:] = candidate[14:] - candidate[0] + motion_i[0]
        if 'hands' in poses_list[i]:
            hands = poses_list[i]['hands']
            assert len(hands) == 2
            if subset[0][4] == 4:
                hands[1] = hands[1] - candidate[4] + motion_i[4]
            if subset[0][7] == 7:
                hands[0] = hands[0] - candidate[7] + motion_i[7]
        if 'faces' in poses_list[i]:
            faces = poses_list[i]['faces']
            assert len(faces) == 1
            faces[0] = faces[0] - candidate[0] + motion_i[0]
        candidate[:14] = motion_i
        poses_list[i]['bodies']['candidate'] = candidate
    new_box = get_box(poses_list[0]['bodies']['candidate'][vis_indexes])
    scale = (box[3] - box[1]) / float((new_box[3] - new_box[1]))
    scale_offset_poses(poses_list, scale=(scale, scale))
    new_ground_point = poses_list[0]['bodies']['candidate'][ground_index]
    ground_offset = ground_point - new_ground_point
    scale_offset_poses(poses_list, offset=ground_offset)
    return poses_list


def preprocess_motion(motion, mean_pose, std_pose, scale):
    motion = motion * scale
    centers = motion[8, :, :]
    motion_proj = motion - centers
    # adding velocity
    velocity = np.c_[np.zeros((2, 1)), centers[:, 1:] - centers[:, :-1]].reshape(1, 2, -1)
    motion = np.r_[motion_proj[:8], motion_proj[9:], velocity]
    motion = (motion - mean_pose[:, :, np.newaxis]) / std_pose[:, :, np.newaxis]
    motion = motion.reshape((-1, motion.shape[-1]))
    return torch.Tensor(motion).unsqueeze(0)


def postprocess_motion(motion, mean_pose, std_pose, sx, sy, scale):
    motion = motion.detach().cpu().numpy()[0].reshape(-1, 2, motion.shape[-1])
    motion = motion * std_pose[:, :, np.newaxis] + mean_pose[:, :, np.newaxis]
    velocity = motion[-1].copy()
    motion_inv = np.r_[motion[:8], np.zeros((1, 2, motion.shape[-1])), motion[8:-1]]
    # restore centre position
    centers = np.zeros_like(velocity)
    sum = 0
    for i in range(motion.shape[-1]):
        sum += velocity[:, i]
        centers[:, i] = sum
    centers += np.array([[sx], [sy]])
    motion = motion_inv + centers.reshape((1, 2, -1))
    motion = motion / scale
    return motion


def move_poses(src_poses, dst_poses):
    src_vis_indexes = src_poses['vis_indexes']['bodies']
    dst_vis_indexes = dst_poses['vis_indexes']['bodies']
    vis_indexes = intersect_vis_indexes([src_vis_indexes, dst_vis_indexes])
    assert sum(vis_indexes) > 0
    src_box = get_box(src_poses['bodies']['candidate'][vis_indexes])
    dst_box = get_box(dst_poses['bodies']['candidate'][vis_indexes])
    src_h = src_box[3] - src_box[1]
    dst_h = dst_box[3] - dst_box[1]
    src_ctr = get_center(src_box)
    dst_ctr = get_center(dst_box)
    scale = dst_h / float(src_h)
    offset = (dst_ctr[0] - src_ctr[0] * scale, dst_ctr[1] - src_ctr[1] * scale)
    scale_offset_poses([src_poses], scale=(scale, scale), offset=offset)
    return src_poses


def set_vis_indexes(poses_list):
    for poses in poses_list:
        poses['vis_indexes'] = dict()
        if 'bodies' in poses:
            subset = poses['bodies']['subset']
            assert len(subset) == 1
            vis_indexes = [True] * subset.shape[1]
            for idx, state in enumerate(subset[0]):
                if state < 0:
                    vis_indexes[idx] = False
            poses['vis_indexes']['bodies'] = vis_indexes
        if 'hands' in poses:
            hands = poses['hands']
            assert len(hands) == 2
            vis_indexes = [[True] * hands.shape[1]] * hands.shape[0]
            for i in range(hands.shape[0]):
                for idx, point in enumerate(hands[i]):
                    if not pose_utils.point_in_image(point):
                        vis_indexes[i][idx] = False
            poses['vis_indexes']['hands'] = vis_indexes
        if 'faces' in poses:
            faces = poses['faces']
            assert len(faces) == 1
            vis_indexes = [True] * faces.shape[1]
            for idx, point in enumerate(faces[0]):
                if not pose_utils.point_in_image(point):
                    vis_indexes[idx] = False
            poses['vis_indexes']['faces'] = vis_indexes


def intersect_vis_indexes(indexes_list):
    vis_indexes = [True] * len(indexes_list[0])
    for idx in range(len(vis_indexes)):
        for i in range(len(indexes_list)):
            if not indexes_list[i][idx]:
                vis_indexes[idx] = False
                break
    return vis_indexes


def union_vis_indexes(indexes_list):
    vis_indexes = [False] * len(indexes_list[0])
    for idx in range(len(vis_indexes)):
        for i in range(len(indexes_list)):
            if indexes_list[i][idx]:
                vis_indexes[idx] = True
                break
    return vis_indexes


def scale_offset_poses(poses_list, scale=None, offset=None):
    for poses in poses_list:
        vis_indexes = poses['vis_indexes']
        if 'bodies' in poses:
            candidate = poses['bodies']['candidate'][vis_indexes['bodies']].copy()
            if scale is not None:
                candidate *= scale
            if offset is not None:
                candidate += offset
            poses['bodies']['candidate'][vis_indexes['bodies']] = candidate
        if 'hands' in poses:
            hands = poses['hands']
            for i in range(hands.shape[0]):
                hands_i = hands[i][vis_indexes['hands'][i]].copy()
                if scale is not None:
                    hands_i *= scale
                if offset is not None:
                    hands_i += offset
                poses['hands'][i][vis_indexes['hands'][i]] = hands_i
        if 'faces' in poses:
            faces = poses['faces'][0][vis_indexes['faces']].copy()
            if scale is not None:
                faces *= scale
            if offset is not None:
                faces += offset
            poses['faces'][0][vis_indexes['faces']] = faces


def clip_poses(poses_list):
    for poses in poses_list:
        vis_indexes = poses['vis_indexes']
        if 'bodies' in poses:
            candidate = poses['bodies']['candidate'][vis_indexes['bodies']].copy()
            subset = poses['bodies']['subset'][0][vis_indexes['bodies']].copy()
            poses['bodies']['candidate'][:] = -1
            poses['bodies']['subset'][0][:] = -1
            poses['bodies']['candidate'][vis_indexes['bodies']] = candidate
            poses['bodies']['subset'][0][vis_indexes['bodies']] = subset
        if 'hands' in poses:
            hands = poses['hands']
            for i in range(hands.shape[0]):
                hands_i = hands[i][vis_indexes['hands'][i]].copy()
                poses['hands'][i][:] = -1
                poses['hands'][i][vis_indexes['hands'][i]] = hands_i
        if 'faces' in poses:
            faces = poses['faces'][0][vis_indexes['faces']].copy()
            poses['faces'][0][:] = -1
            poses['faces'][0][vis_indexes['faces']] = faces


def get_ground_index(poses):
    indexes = poses['vis_indexes']['bodies'].copy()
    ignore_indexes = [3, 4, 6, 7]  # ignore hands
    for ignore_index in ignore_indexes:
        indexes[ignore_index] = False
    assert sum(indexes) > 0
    vis_bodies = poses['bodies']['candidate'][indexes]
    vis_ind = np.argsort(vis_bodies[:, 1])[-1]
    ground_index = np.where(indexes)[0][vis_ind]
    return ground_index


def get_box(points):
    if len(points) > 0:
        x1, y1 = np.min(points, axis=0)
        x2, y2 = np.max(points, axis=0)
        box = [x1, y1, x2 + 1, y2 + 1]
    else:
        box = [0, 0, 0, 0]
    return box


def get_center(box):
    x1, y1, x2, y2 = box
    x_ctr = (x1 + x2) / 2
    y_ctr = (y1 + y2) / 2
    return x_ctr, y_ctr
