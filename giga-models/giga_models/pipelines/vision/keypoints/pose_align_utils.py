import copy

import numpy as np

from . import pose_utils


def align_poses(
    ref_image,
    ref_poses,
    poses_list,
    pose_image_height,
    pose_image_width,
    align_idx=0,
    method=1,
    crop_ref_image=True,
):
    ref_poses, poses_list = _process_poses(
        ref_poses=ref_poses,
        poses_list=poses_list,
        ref_image_height=ref_image.height,
        ref_image_width=ref_image.width,
        pose_image_height=pose_image_height,
        pose_image_width=pose_image_width,
    )
    all_pose_images = []
    ref_height_list = []
    all_poses_list = []
    num_poses = pose_utils.get_num_poses(ref_poses)
    for i in range(num_poses):
        ref_poses_i = pose_utils.get_poses(ref_poses, i)
        poses_list_i = [pose_utils.get_poses(poses, i) for poses in poses_list]
        pose_images, ref_height, poses_list_i = _align_poses_single(
            ref_image=ref_image,
            ref_poses=ref_poses_i,
            poses_list=poses_list_i,
            pose_image_height=pose_image_height,
            pose_image_width=pose_image_width,
            align_idx=align_idx,
            method=method,
        )
        all_pose_images.append(pose_images)
        ref_height_list.append(ref_height)
        all_poses_list.append(poses_list_i)
    pose_images = [pose_utils.merge_pose_images(_) for _ in zip(*all_pose_images)]
    poses_list = [pose_utils.merge_poses(_) for _ in zip(*all_poses_list)]
    ref_height = max(ref_height_list)
    if crop_ref_image and ref_height > 0:
        crop_box = (0, 0, ref_image.width, int(ref_height))
        ref_image = ref_image.crop(crop_box)
        pose_images = [pose_image.crop(crop_box) for pose_image in pose_images]
    assert ref_image.height == pose_images[0].height and ref_image.width == pose_images[0].width
    return ref_image, pose_images, ref_poses, poses_list


def _process_poses(
    ref_poses,
    poses_list,
    ref_image_height,
    ref_image_width,
    pose_image_height,
    pose_image_width,
):
    ref_boxes = pose_utils.get_boxes(ref_poses, ref_image_height, ref_image_width)
    boxes_list = [pose_utils.get_boxes(poses, pose_image_height, pose_image_width) for poses in poses_list]
    ref_indexes = list(np.argsort(ref_boxes[:, 0]))
    indexes_list = [list(np.argsort(boxes[:, 0])) for boxes in boxes_list]
    for i, indexes in enumerate(indexes_list):
        if len(indexes) > len(ref_indexes):
            indexes = indexes[: len(ref_indexes)]
        elif len(indexes) < len(ref_indexes):
            extra = len(ref_indexes) - len(indexes)
            for j in range(extra):
                indexes.append(indexes[-1])
        indexes_list[i] = indexes
    ref_poses = pose_utils.get_poses(ref_poses, ref_indexes)
    poses_list = [pose_utils.get_poses(poses, indexes) for poses, indexes in zip(poses_list, indexes_list)]
    return ref_poses, poses_list


def _align_poses_single(
    ref_image,
    ref_poses,
    poses_list,
    pose_image_height,
    pose_image_width,
    align_idx=0,
    method=1,
):
    ref_poses = copy.deepcopy(ref_poses)
    poses_list = copy.deepcopy(poses_list)
    _set_vis_indexes([ref_poses])
    _set_vis_indexes(poses_list)
    _scale_offset_poses([ref_poses], scale=(ref_image.width, ref_image.height))
    _scale_offset_poses(poses_list, scale=(pose_image_width, pose_image_height))
    scale = _get_body_scale(ref_poses, poses_list[align_idx])
    _scale_offset_poses(poses_list, scale=scale)
    offset = _get_body_offset(ref_poses, poses_list[align_idx])
    _scale_offset_poses(poses_list, offset=offset)
    if method == 1:
        body_scales = _get_body_detail_scales(ref_poses, poses_list[align_idx])
        # hand_scale = _get_hand_scale(ref_poses, poses_list[align_idx])
        if 'faces' in ref_poses:
            face_scale = _get_face_scale(ref_poses, poses_list[align_idx])
            nose_neck_scale = body_scales['nose_neck']
            if face_scale[1] < 0.9:
                nose_neck_scale = min(nose_neck_scale, 1.2)
            body_scales['nose_neck'] = nose_neck_scale
        for poses in poses_list:
            assert poses['bodies']['subset'][0][0] == 0 and poses['bodies']['subset'][0][1] == 1
            nose_coord = poses['bodies']['candidate'][0]
            neck_coord = _compute_coord(body_scales, poses, ['nose', 'neck'], nose_coord)
            shoulder_left_coord = _compute_coord(body_scales, poses, ['neck', 'shoulder_left'], neck_coord)
            wrist_left_coord = _compute_coord(body_scales, poses, ['shoulder_left', 'wrist_left'], shoulder_left_coord)
            hand_left_coord = _compute_coord(body_scales, poses, ['wrist_left', 'hand_left'], wrist_left_coord)
            shoulder_right_coord = _compute_coord(body_scales, poses, ['neck', 'shoulder_right'], neck_coord)
            wrist_right_coord = _compute_coord(body_scales, poses, ['shoulder_right', 'wrist_right'], shoulder_right_coord)
            hand_right_coord = _compute_coord(body_scales, poses, ['wrist_right', 'hand_right'], wrist_right_coord)
            waist_left_coord = _compute_coord(body_scales, poses, ['neck', 'waist_left'], neck_coord)
            knee_left_coord = _compute_coord(body_scales, poses, ['waist_left', 'knee_left'], waist_left_coord)
            foot_left_coord = _compute_coord(body_scales, poses, ['knee_left', 'foot_left'], knee_left_coord)
            waist_right_coord = _compute_coord(body_scales, poses, ['neck', 'waist_right'], neck_coord)
            knee_right_coord = _compute_coord(body_scales, poses, ['waist_right', 'knee_right'], waist_right_coord)
            foot_right_coord = _compute_coord(body_scales, poses, ['knee_right', 'foot_right'], knee_right_coord)
            # process faces
            if 'faces' in poses:
                faces_in_body = poses['bodies']['candidate'][[-4, -3, -2, -1, 0]]
                vis_indexes_in_body = [poses['vis_indexes']['bodies'][i] for i in [-4, -3, -2, -1, 0]]
                faces = np.concatenate([poses['faces'][0], faces_in_body], axis=0)
                vis_indexes = poses['vis_indexes']['faces'] + vis_indexes_in_body
                faces[vis_indexes] *= face_scale
                faces[vis_indexes] += nose_coord - faces[-1]
                faces_in_body = faces[-5:-1, :]
                poses['faces'] = faces[:-5][None]
            else:
                faces_in_body = poses['bodies']['candidate'][[-4, -3, -2, -1, 0]]
                vis_indexes_in_body = [poses['vis_indexes']['bodies'][i] for i in [-4, -3, -2, -1, 0]]
                faces_in_body[vis_indexes_in_body] += nose_coord - faces_in_body[-1]
                faces_in_body = faces_in_body[-5:-1, :]
            # process hands
            if 'hands' in poses:
                hands_list = []
                for i in range(2):
                    if i == 0:
                        if poses['vis_indexes']['bodies'][7]:
                            hand_idx = 7
                            hand_coord = hand_right_coord
                        elif poses['vis_indexes']['bodies'][6]:
                            hand_idx = 6
                            hand_coord = wrist_right_coord
                        else:
                            hand_idx = -1
                    else:
                        if poses['vis_indexes']['bodies'][4]:
                            hand_idx = 4
                            hand_coord = hand_left_coord
                        elif poses['vis_indexes']['bodies'][3]:
                            hand_idx = 3
                            hand_coord = wrist_left_coord
                        else:
                            hand_idx = -1
                    hands = poses['hands'][i]
                    vis_indexes = poses['vis_indexes']['hands'][i]
                    if hand_idx > 0:
                        hands[vis_indexes] += hand_coord - poses['bodies']['candidate'][hand_idx : hand_idx + 1]
                    else:
                        hands[:] = -1
                        for k in range(len(vis_indexes)):
                            vis_indexes[k] = False
                    hands_list.append(hands)
                poses['hands'] = np.stack(hands_list, axis=0)
            # process bodies
            bodies = [
                nose_coord,
                neck_coord,
                shoulder_left_coord,
                wrist_left_coord,
                hand_left_coord,
                shoulder_right_coord,
                wrist_right_coord,
                hand_right_coord,
                waist_left_coord,
                knee_left_coord,
                foot_left_coord,
                waist_right_coord,
                knee_right_coord,
                foot_right_coord,
            ] + faces_in_body.tolist()
            bodies = np.array(bodies)
            poses['bodies']['candidate'] = bodies
    else:
        assert method == 0
    offset = _get_ground_offset(ref_poses, poses_list[align_idx])
    _scale_offset_poses(poses_list, offset=offset)
    # pose_images
    pose_images = _draw_pose_images(
        poses_list,
        height=ref_image.height,
        width=ref_image.width,
    )
    # crop ref_image
    ref_vis_indexes = ref_poses['vis_indexes']['bodies']
    tem_vis_indexes = _union_vis_indexes([poses['vis_indexes']['bodies'] for poses in poses_list])
    if sum(ref_vis_indexes) > sum(tem_vis_indexes):
        cross_indexes = _intersect_vis_indexes([ref_vis_indexes, tem_vis_indexes])
        vis_bodies = ref_poses['bodies']['candidate'][cross_indexes]
        ref_height = np.max(vis_bodies, axis=0)[1]
    else:
        ref_height = -1
    _scale_offset_poses(poses_list, scale=(1.0 / ref_image.width, 1.0 / ref_image.height))
    _set_vis_indexes(poses_list)
    _clip_poses(poses_list)
    for poses in poses_list:
        poses.pop('vis_indexes')
    return pose_images, ref_height, poses_list


def _point_in_image(point):
    if point[0] < 0 or point[1] < 0 or point[0] >= 1 or point[1] >= 1:
        return False
    return True


def _set_vis_indexes(poses_list):
    for poses in poses_list:
        poses['vis_indexes'] = dict()
        if 'bodies' in poses:
            candidate = poses['bodies']['candidate']
            subset = poses['bodies']['subset']
            if len(subset) == 1:
                vis_indexes = [True] * subset.shape[1]
                for idx, state in enumerate(subset[0]):
                    if state < 0 or not _point_in_image(candidate[idx]):
                        vis_indexes[idx] = False
            elif len(subset) == 0:
                vis_indexes = []
            else:
                assert False
            poses['vis_indexes']['bodies'] = vis_indexes
        if 'hands' in poses:
            hands = poses['hands']
            if len(hands) == 2:
                vis_indexes = [[True] * hands.shape[1]] * hands.shape[0]
                for i in range(hands.shape[0]):
                    for idx, point in enumerate(hands[i]):
                        if not _point_in_image(point):
                            vis_indexes[i][idx] = False
            elif len(hands) == 0:
                vis_indexes = []
            else:
                assert False
            poses['vis_indexes']['hands'] = vis_indexes
        if 'faces' in poses:
            faces = poses['faces']
            if len(faces) == 1:
                vis_indexes = [True] * faces.shape[1]
                for idx, point in enumerate(faces[0]):
                    if not _point_in_image(point):
                        vis_indexes[idx] = False
            elif len(faces) == 0:
                vis_indexes = []
            else:
                assert False
            poses['vis_indexes']['faces'] = vis_indexes


def _intersect_vis_indexes(indexes_list):
    vis_indexes = [True] * len(indexes_list[0])
    for idx in range(len(vis_indexes)):
        for i in range(len(indexes_list)):
            if not indexes_list[i][idx]:
                vis_indexes[idx] = False
                break
    return vis_indexes


def _union_vis_indexes(indexes_list):
    vis_indexes = [False] * len(indexes_list[0])
    for idx in range(len(vis_indexes)):
        for i in range(len(indexes_list)):
            if indexes_list[i][idx]:
                vis_indexes[idx] = True
                break
    return vis_indexes


def _scale_offset_poses(poses_list, scale=None, offset=None):
    for poses in poses_list:
        vis_indexes = poses['vis_indexes']
        if 'bodies' in poses:
            if len(vis_indexes['bodies']) > 0:
                candidate = poses['bodies']['candidate'][vis_indexes['bodies']].copy()
                if scale is not None:
                    candidate *= scale
                if offset is not None:
                    candidate += offset
                poses['bodies']['candidate'][vis_indexes['bodies']] = candidate
            else:
                assert len(poses['bodies']['candidate']) == 0
        if 'hands' in poses:
            if len(vis_indexes['hands']) > 0:
                hands = poses['hands']
                for i in range(hands.shape[0]):
                    hands_i = hands[i][vis_indexes['hands'][i]].copy()
                    if scale is not None:
                        hands_i *= scale
                    if offset is not None:
                        hands_i += offset
                    poses['hands'][i][vis_indexes['hands'][i]] = hands_i
            else:
                assert len(poses['hands']) == 0
        if 'faces' in poses:
            if len(vis_indexes['faces']) > 0:
                faces = poses['faces'][0][vis_indexes['faces']].copy()
                if scale is not None:
                    faces *= scale
                if offset is not None:
                    faces += offset
                poses['faces'][0][vis_indexes['faces']] = faces
            else:
                assert len(poses['faces']) == 0


def _clip_poses(poses_list):
    for poses in poses_list:
        vis_indexes = poses['vis_indexes']
        if 'bodies' in poses:
            if len(vis_indexes['bodies']) > 0:
                candidate = poses['bodies']['candidate'][vis_indexes['bodies']].copy()
                subset = poses['bodies']['subset'][0][vis_indexes['bodies']].copy()
                poses['bodies']['candidate'][:] = -1
                poses['bodies']['subset'][0][:] = -1
                poses['bodies']['candidate'][vis_indexes['bodies']] = candidate
                poses['bodies']['subset'][0][vis_indexes['bodies']] = subset
            else:
                assert len(poses['bodies']['candidate']) == 0
        if 'hands' in poses:
            if len(vis_indexes['hands']) > 0:
                hands = poses['hands']
                for i in range(hands.shape[0]):
                    hands_i = hands[i][vis_indexes['hands'][i]].copy()
                    poses['hands'][i][:] = -1
                    poses['hands'][i][vis_indexes['hands'][i]] = hands_i
            else:
                assert len(poses['hands']) == 0
        if 'faces' in poses:
            if len(vis_indexes['faces']) > 0:
                faces = poses['faces'][0][vis_indexes['faces']].copy()
                poses['faces'][0][:] = -1
                poses['faces'][0][vis_indexes['faces']] = faces
            else:
                assert len(poses['faces']) == 0


def _get_body_scale(ref_poses, poses):
    assert ref_poses['bodies']['subset'][0][2] == 2 and ref_poses['bodies']['subset'][0][5] == 5
    assert poses['bodies']['subset'][0][2] == 2 and poses['bodies']['subset'][0][5] == 5
    # scale_w
    left_shoulder, right_shoulder = ref_poses['bodies']['candidate'][2], ref_poses['bodies']['candidate'][5]
    ref_w = np.linalg.norm(left_shoulder - right_shoulder)
    left_shoulder, right_shoulder = poses['bodies']['candidate'][2], poses['bodies']['candidate'][5]
    aln_w = np.linalg.norm(left_shoulder - right_shoulder)
    scale_w = ref_w / aln_w
    # scale_h
    cross_indexes = _intersect_vis_indexes([ref_poses['vis_indexes']['bodies'], poses['vis_indexes']['bodies']])
    ignore_indexes = [3, 4, 6, 7]  # ignore hands
    for ignore_index in ignore_indexes:
        cross_indexes[ignore_index] = False
    assert sum(cross_indexes) > 0
    vis_bodies = ref_poses['bodies']['candidate'][cross_indexes]
    h_min = np.min(vis_bodies, axis=0)[1]
    h_max = np.max(vis_bodies, axis=0)[1]
    ref_h = h_max - h_min
    vis_bodies = poses['bodies']['candidate'][cross_indexes]
    h_min = np.min(vis_bodies, axis=0)[1]
    h_max = np.max(vis_bodies, axis=0)[1]
    aln_h = h_max - h_min
    scale_h = ref_h / aln_h
    return scale_w, scale_h


def _get_body_offset(ref_poses, poses):
    assert ref_poses['bodies']['subset'][0][0] == 0
    assert poses['bodies']['subset'][0][0] == 0
    ref_coord = ref_poses['bodies']['candidate'][0]
    aln_coord = poses['bodies']['candidate'][0]
    offset_w = int(ref_coord[0] - aln_coord[0])
    offset_h = int(ref_coord[1] - aln_coord[1])
    return offset_w, offset_h


def _get_face_scale(ref_poses, poses):
    cross_indexes = _intersect_vis_indexes([ref_poses['vis_indexes']['faces'], poses['vis_indexes']['faces']])
    assert sum(cross_indexes) > 0
    ref_w, ref_h = _get_wh(ref_poses['faces'][0][cross_indexes])
    aln_w, aln_h = _get_wh(poses['faces'][0][cross_indexes])
    assert ref_w > 0 and aln_w > 0 and ref_h > 0 and aln_h > 0
    scale_w = ref_w / aln_w
    scale_h = ref_h / aln_h
    return scale_w, scale_h


def _get_hand_scale(ref_poses, poses):
    idx = 0
    cross_indexes = _intersect_vis_indexes([ref_poses['vis_indexes']['hands'][0], poses['vis_indexes']['hands'][0]])
    cross_indexes_2 = _intersect_vis_indexes([ref_poses['vis_indexes']['hands'][1], poses['vis_indexes']['hands'][1]])
    if sum(cross_indexes_2) > sum(cross_indexes):
        idx = 1
        cross_indexes = cross_indexes_2
    assert sum(cross_indexes) > 0
    ref_w, ref_h = _get_wh(ref_poses['hands'][idx][cross_indexes])
    aln_w, aln_h = _get_wh(poses['hands'][idx][cross_indexes])
    assert ref_w > 0 and aln_w > 0 and ref_h > 0 and aln_h > 0
    scale_w = ref_w / aln_w
    scale_h = ref_h / aln_h
    return scale_w, scale_h


def _get_body_detail_scales(ref_poses, poses):
    all_indexes = [
        [['nose', 'neck']],
        [['neck', 'shoulder_left'], ['neck', 'shoulder_right']],
        [['neck', 'waist_left'], ['neck', 'waist_right']],
        [['waist_left', 'knee_left'], ['waist_right', 'knee_right']],
        [['knee_left', 'foot_left'], ['knee_right', 'foot_right']],
    ]
    all_indexes_2 = [
        [['shoulder_left', 'wrist_left'], ['shoulder_right', 'wrist_right']],
        [['wrist_left', 'hand_left'], ['wrist_right', 'hand_right']],
    ]
    all_scales = dict()
    ref_candidate = ref_poses['bodies']['candidate']
    ref_vis_indexes = ref_poses['vis_indexes']['bodies']
    candidate = poses['bodies']['candidate']
    vis_indexes = poses['vis_indexes']['bodies']
    for indexes in all_indexes:
        scales = []
        ref_dist_max = -10000
        aln_dist_max = -10000
        for indexes_i in indexes:
            src_idx = pose_utils.KEYPOINT_INDEXES[indexes_i[0]]
            dst_idx = pose_utils.KEYPOINT_INDEXES[indexes_i[1]]
            if ref_vis_indexes[src_idx] and ref_vis_indexes[dst_idx] and vis_indexes[src_idx] and vis_indexes[dst_idx]:
                ref_dist = np.linalg.norm(ref_candidate[src_idx] - ref_candidate[dst_idx])
                aln_dist = np.linalg.norm(candidate[src_idx] - candidate[dst_idx])
                ref_dist_max = max(ref_dist, ref_dist_max)
                aln_dist_max = max(aln_dist, aln_dist_max)
                scales.append(ref_dist / aln_dist)
            else:
                scales.append(-1)
        for i in range(len(scales)):
            if scales[i] != -1:
                scales[i] = ref_dist_max / aln_dist_max
        for i, indexes_i in enumerate(indexes):
            name = '_'.join(indexes_i)
            assert name not in all_scales
            all_scales[name] = scales[i]
    for indexes in all_indexes_2:
        for i, indexes_i in enumerate(indexes):
            name = '_'.join(indexes_i)
            assert name not in all_scales
            all_scales[name] = all_scales['neck_waist_left']
    return all_scales


def _compute_coord(body_scales, poses, indexes, base_coord):
    src_idx = pose_utils.KEYPOINT_INDEXES[indexes[0]]
    dst_idx = pose_utils.KEYPOINT_INDEXES[indexes[1]]
    candidate = poses['bodies']['candidate']
    scale = body_scales['_'.join(indexes)]
    coord = candidate[dst_idx] - candidate[src_idx]
    if scale > 0:
        coord *= scale
    coord += base_coord
    return coord


def _get_ground_offset(ref_poses, poses):
    cross_indexes = _intersect_vis_indexes([ref_poses['vis_indexes']['bodies'], poses['vis_indexes']['bodies']])
    ignore_indexes = [3, 4, 6, 7]  # ignore hands
    for ignore_index in ignore_indexes:
        cross_indexes[ignore_index] = False
    assert sum(cross_indexes) > 0
    vis_bodies = ref_poses['bodies']['candidate'][cross_indexes]
    vis_ind = np.argsort(vis_bodies[:, 1])[-1]
    ref_coord = vis_bodies[vis_ind]
    vis_bodies = poses['bodies']['candidate'][cross_indexes]
    aln_coord = vis_bodies[vis_ind]
    # offset_w = int(ref_coord[0] - aln_coord[0])
    offset_w = 0
    offset_h = int(ref_coord[1] - aln_coord[1])
    return offset_w, offset_h


def _get_wh(points):
    x_min = np.min(points[:, 0])
    x_max = np.max(points[:, 0])
    y_min = np.min(points[:, 1])
    y_max = np.max(points[:, 1])
    w = x_max - x_min
    h = y_max - y_min
    return w, h


def _get_face_indexes():
    left_face_indexes = [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        18,
        19,
        20,
        21,
        22,
        32,
        33,
        37,
        38,
        39,
        40,
        41,
        42,
        49,
        50,
        51,
        59,
        60,
        61,
        62,
        68,
    ]
    mid_face_indexes = [9, 28, 29, 30, 31, 34, 52, 58, 63, 67]
    right_face_indexes = [
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        23,
        24,
        25,
        26,
        27,
        35,
        36,
        43,
        44,
        45,
        46,
        47,
        48,
        53,
        54,
        55,
        56,
        57,
        64,
        65,
        66,
    ]
    face_indexes = left_face_indexes + mid_face_indexes + right_face_indexes
    assert len(list(set(face_indexes))) == len(face_indexes)
    assert len(left_face_indexes) == len(right_face_indexes)
    left_face_indexes = [idx - 1 for idx in left_face_indexes + mid_face_indexes]
    right_face_indexes = [idx - 1 for idx in right_face_indexes + mid_face_indexes]
    left_face_indexes.sort()
    right_face_indexes.sort()
    return left_face_indexes, right_face_indexes


def _draw_pose_images(poses_list, height, width):
    poses_list = copy.deepcopy(poses_list)
    _scale_offset_poses(poses_list, offset=(width // 2, height // 2))
    _scale_offset_poses(poses_list, scale=(1.0 / (width * 2), 1.0 / (height * 2)))
    _set_vis_indexes(poses_list)
    _clip_poses(poses_list)
    pose_images = []
    for poses in poses_list:
        pose_image = pose_utils.draw_poses(
            poses,
            height=height * 2,
            width=width * 2,
            draw_body='bodies' in poses,
            draw_hand='hands' in poses,
            draw_face='faces' in poses,
        )
        pose_image = pose_image.crop((width // 2, height // 2, width // 2 + width, height // 2 + height))
        pose_images.append(pose_image)
    return pose_images
