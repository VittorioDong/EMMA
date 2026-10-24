import copy

import numpy as np
from controlnet_aux.dwpose import util
from PIL import Image

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


def draw_poses(poses, height, width, draw_body=True, draw_hand=False, draw_face=False, canvas=None):
    if canvas is None:
        canvas = np.zeros(shape=(height, width, 3), dtype=np.uint8)
    if draw_body:
        bodies = poses['bodies']
        candidate = bodies['candidate']
        subset = bodies['subset']
        canvas = util.draw_bodypose(canvas, candidate, subset)
    if draw_hand:
        hands = poses['hands'].copy()
        subset = poses['bodies']['subset']
        for i in range(len(subset)):
            if subset[i][4] < 0:
                hands[i * 2 + 1] = -1
            if subset[i][7] < 0:
                hands[i * 2] = -1
        canvas = util.draw_handpose(canvas, hands)
    if draw_face:
        faces = poses['faces']
        canvas = util.draw_facepose(canvas, faces)
    canvas = Image.fromarray(canvas)
    return canvas


def get_num_poses(poses):
    num_poses = len(poses['bodies']['subset'])
    if num_poses > 0:
        if poses['bodies']['subset'].max() == -1:
            num_poses = 0
    return num_poses


def get_boxes(poses, height, width, include_body=True, include_hand=False, include_face=False):
    num_poses = get_num_poses(poses)
    boxes = np.zeros((num_poses, 4), dtype=np.float32)
    for i in range(num_poses):
        points = []
        if include_body:
            bodies = poses['bodies']
            candidate = bodies['candidate']
            subset = bodies['subset'][i]
            subset = subset[subset >= 0].astype(np.int32)
            candidate = candidate[subset]
            points.append(candidate)
        if include_hand:
            hands = poses['hands'].reshape((2, num_poses, -1, 2))
            hands = hands[:, i].reshape((-1, 2))
            hands = hands[hands.sum(-1) >= 0]
            points.append(hands)
        if include_face:
            faces = poses['faces'][i]
            faces = faces[faces.sum(-1) >= 0]
            points.append(faces)
        points = np.concatenate(points, axis=0)
        if len(points) > 0:
            x1, y1 = np.min(points, axis=0)
            x2, y2 = np.max(points, axis=0)
            x1 = max(min(x1 * width, width - 1), 0)
            y1 = max(min(y1 * height, height - 1), 0)
            x2 = max(min(x2 * width, width - 1), 0)
            y2 = max(min(y2 * height, height - 1), 0)
            boxes[i] = [x1, y1, x2, y2]
        else:
            boxes[i] = [0, 0, width - 1, height - 1]
    return boxes


def get_face_centers(poses, height, width):
    num_poses = get_num_poses(poses)
    face_centers = np.zeros((num_poses, 2), dtype=np.float32)
    for i in range(num_poses):
        points = []
        if 'bodies' in poses:
            bodies = poses['bodies']
            candidate = bodies['candidate']
            subset = bodies['subset'][i, [0]]
            subset = subset[subset >= 0].astype(np.int32)
            candidate = candidate[subset]
            points.append(candidate)
        if 'faces' in poses:
            faces = poses['faces'][i, 27:]
            faces = faces[faces.sum(-1) >= 0]
            points.append(faces)
        points = np.concatenate(points, axis=0)
        if len(points) > 0:
            x1, y1 = np.min(points, axis=0)
            x2, y2 = np.max(points, axis=0)
            x1 = max(min(x1 * width, width - 1), 0)
            y1 = max(min(y1 * height, height - 1), 0)
            x2 = max(min(x2 * width, width - 1), 0)
            y2 = max(min(y2 * height, height - 1), 0)
            ctr_x = (x1 + x2) / 2
            ctr_y = (y1 + y2) / 2
            face_centers[i] = [ctr_x, ctr_y]
        else:
            face_centers[i] = [-1, -1]
    return face_centers


def get_poses(poses, idx):
    idx_list = idx if isinstance(idx, list) else [idx]
    poses = copy.deepcopy(poses)
    if 'bodies' in poses:
        bodies = poses['bodies']
        candidate = bodies['candidate'].reshape((-1, 18, 2))
        subset = bodies['subset']
        candidate = candidate[idx_list]
        candidate = candidate.reshape((-1, 2))
        subset = subset[idx_list].reshape(-1)
        new_subset = np.arange(subset.shape[0], dtype=subset.dtype)
        new_subset[subset < 0] = subset[subset < 0]
        new_subset = new_subset.reshape((-1, 18))
        poses['bodies']['candidate'] = candidate
        poses['bodies']['subset'] = new_subset
    if 'hands' in poses:
        hands = poses['hands'].reshape((2, len(poses['hands']) // 2, -1, 2))
        hands = hands[:, idx_list]
        hands = hands.reshape((-1, poses['hands'].shape[1], 2))
        poses['hands'] = hands
    if 'faces' in poses:
        faces = poses['faces']
        poses['faces'] = faces[idx_list]
    return poses


def merge_poses(poses_list):
    if len(poses_list) == 1:
        return poses_list[0]
    new_poses = dict()
    if 'bodies' in poses_list[0]:
        candidate_list = [poses['bodies']['candidate'] for poses in poses_list]
        subset_list = [poses['bodies']['subset'] for poses in poses_list]
        candidate = np.concatenate(candidate_list, axis=0)
        subset = np.concatenate(subset_list, axis=0).reshape(-1)
        new_subset = np.arange(subset.shape[0], dtype=subset.dtype)
        new_subset[subset < 0] = subset[subset < 0]
        new_subset = new_subset.reshape((-1, 18))
        new_poses['bodies'] = {
            'candidate': candidate,
            'subset': new_subset,
        }
    if 'hands' in poses_list[0]:
        hands_list = [poses['hands'] for poses in poses_list]
        hands = np.concatenate(hands_list, axis=1)
        hands = hands.reshape((2 * len(hands_list), -1, 2))
        new_poses['hands'] = hands
    if 'faces' in poses_list[0]:
        faces_list = [poses['faces'] for poses in poses_list]
        faces = np.concatenate(faces_list, axis=0)
        new_poses['faces'] = faces
    return new_poses


def merge_pose_images(pose_images):
    if len(pose_images) == 1:
        return pose_images[0]
    new_pose_image = np.zeros((pose_images[0].height, pose_images[0].width, 3), dtype=np.uint8)
    for pose_image in pose_images:
        pose_image = np.array(pose_image)
        new_pose_image[pose_image > 0] = pose_image[pose_image > 0]
    new_pose_image = Image.fromarray(new_pose_image)
    return new_pose_image


def filter_poses(poses, area_mode='area', min_area=None, max_area=None, max_poses=None):
    num_poses = get_num_poses(poses)
    areas = []
    for i in range(num_poses):
        points = []
        if 'bodies' in poses:
            bodies = poses['bodies']
            candidate = bodies['candidate']
            subset = bodies['subset'][i]
            subset = subset[subset >= 0].astype(np.int32)
            candidate = candidate[subset]
            points.append(candidate)
        if 'hands' in poses:
            hands = poses['hands'].reshape((2, num_poses, -1, 2))
            hands = hands[:, i].reshape((-1, 2))
            hands = hands[hands.sum(-1) >= 0]
            points.append(hands)
        if 'faces' in poses:
            faces = poses['faces'][i]
            faces = faces[faces.sum(-1) >= 0]
            points.append(faces)
        points = np.concatenate(points, axis=0)
        if len(points) > 0:
            x1, y1 = np.min(points, axis=0)
            x2, y2 = np.max(points, axis=0)
            x1 = max(min(x1, 1), 0)
            y1 = max(min(y1, 1), 0)
            x2 = max(min(x2, 1), 0)
            y2 = max(min(y2, 1), 0)
            if area_mode == 'area':
                areas.append((x2 - x1) * (y2 - y1))
            elif area_mode == 'height':
                areas.append(y2 - y1)
            elif area_mode == 'width':
                areas.append(x2 - x1)
            else:
                assert False
        else:
            areas.append(0)
    indexes = np.array(areas).argsort()[::-1]
    idx_list = []
    for idx in indexes:
        if min_area is not None and areas[idx] < min_area:
            continue
        if max_area is not None and areas[idx] > max_area:
            continue
        idx_list.append(idx)
    if max_poses is not None:
        idx_list = idx_list[:max_poses]
    return get_poses(poses, idx_list)


def point_in_image(point):
    if point[0] < 0 or point[1] < 0 or point[0] >= 1 or point[1] >= 1:
        return False
    return True
