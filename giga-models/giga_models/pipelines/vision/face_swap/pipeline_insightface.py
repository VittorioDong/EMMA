import os

import numpy as np
from insightface.app import FaceAnalysis
from insightface.model_zoo import model_zoo
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

from ...pipeline import BasePipeline


class InsightFaceSwapPipeline(BasePipeline):
    def __init__(self, model_path):
        model_dir = os.path.dirname(model_path)
        self.app = FaceAnalysis(
            name='buffalo_l',
            root=model_dir,
            allowed_modules=['detection', 'recognition'],
        )
        self.app.prepare(ctx_id=-1, det_size=(640, 640))
        self.swapper = model_zoo.get_model(model_path, root=model_dir)
        self.device = 'cpu'
        self.tracker = Tracker()

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
        models = list(self.app.models.values()) + [self.swapper]
        for model in models:
            model.session.set_providers(providers, provider_options)
        return self

    def get_faces(self, image, points=None, mode=None, max_faces=None):
        image = np.array(image)[:, :, ::-1]
        faces = self.app.get(image)
        if len(faces) == 0:
            return faces
        if points is not None:
            faces = self.filter_faces(faces, points)
        if mode is not None:
            faces = self.sort_faces(faces, mode)
        if max_faces is not None:
            faces = faces[:max_faces]
        return faces

    def filter_faces(self, faces, points):
        new_faces = []
        for face in faces:
            for point in points:
                if point_in_box(point, face.bbox):
                    new_faces.append(face)
                    break
        return new_faces

    def sort_faces(self, faces, mode):
        if mode == 'area':
            faces = sorted(faces, key=lambda x: -(x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        elif mode == 'width':
            faces = sorted(faces, key=lambda x: -(x.bbox[2] - x.bbox[0]))
        elif mode == 'height':
            faces = sorted(faces, key=lambda x: -(x.bbox[3] - x.bbox[1]))
        elif mode == 'x1':
            faces = sorted(faces, key=lambda x: x.bbox[0])
        elif mode == 'y1':
            faces = sorted(faces, key=lambda x: x.bbox[1])
        else:
            assert False
        return faces

    def align_faces(self, src_faces, dst_faces):
        centers = get_centers(src_faces)
        new_dst_faces = []
        used_j = []
        for i, src_face in enumerate(src_faces):
            for j, dst_face in enumerate(dst_faces):
                if j not in used_j and point_in_box(centers[i], dst_face.bbox):
                    dst_face.track_id = src_face.track_id
                    new_dst_faces.append(dst_face)
                    used_j.append(j)
                    break
        return new_dst_faces

    def __call__(
        self,
        src_image=None,
        dst_image_list=None,
        src_faces=None,
        dst_faces_list=None,
        src_params=dict(),
        dst_params=dict(),
        is_aligned=False,
    ):
        if src_faces is None:
            src_faces = self.get_faces(src_image, **src_params)
        if len(src_faces) == 0:
            return dst_image_list
        if dst_faces_list is None:
            dst_faces_list = []
            for i, dst_image in enumerate(dst_image_list):
                if i == 0:
                    dst_faces_list.append(self.get_faces(dst_image, **dst_params))
                else:
                    dst_faces_list.append(self.get_faces(dst_image))
        assert len(dst_image_list) == len(dst_faces_list)
        src_faces = self.tracker([src_faces])[0]
        if is_aligned:
            dst_faces_list[0] = self.align_faces(src_faces, dst_faces_list[0])
        dst_faces_list = self.tracker(dst_faces_list)
        src_faces_dict = {src_face.track_id: src_face for src_face in src_faces}
        new_dst_image_list = []
        for i in range(len(dst_image_list)):
            dst_image = dst_image_list[i]
            dst_image = np.array(dst_image)[:, :, ::-1]
            dst_faces = dst_faces_list[i]
            for dst_face in dst_faces:
                if dst_face.track_id in src_faces_dict:
                    src_face = src_faces_dict[dst_face.track_id]
                    dst_image = self.swapper.get(dst_image, dst_face, src_face, paste_back=True)
            dst_image = Image.fromarray(dst_image[:, :, ::-1])
            new_dst_image_list.append(dst_image)
        return new_dst_image_list


def point_in_box(point, box):
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def get_centers(faces):
    centers = []
    for face in faces:
        x1, y1, x2, y2 = face.bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        centers.append((cx, cy))
    return centers


class Tracker:
    def __init__(self, cosine_threshes=[0.8, 0.4], iou_thresh=0.01):
        self.cosine_threshes = cosine_threshes
        self.iou_thresh = iou_thresh

    def __call__(self, faces_list):
        cache = dict()
        cur_track_id = 0
        cur_track_id = self.add_new_track(faces_list[0], cur_track_id, cache)
        for n in range(1, len(faces_list)):
            faces = faces_list[n]
            used_track_ids = []
            faces = self.track_by_embed_sim(faces, used_track_ids, cache, cosine_thresh=self.cosine_threshes[0])
            faces = self.track_by_iou(faces, faces_list[n - 1], used_track_ids, cache, iou_thresh=self.iou_thresh)
            faces = self.track_by_embed_sim(faces, used_track_ids, cache, cosine_thresh=self.cosine_threshes[1])
            cur_track_id = self.add_new_track(faces, cur_track_id, cache)
        return faces_list

    def track_by_embed_sim(self, faces, used_track_ids, cache, cosine_thresh):
        left_faces = []
        for face in faces:
            last_all_faces = []
            for track_id in cache:
                if track_id not in used_track_ids:
                    last_all_faces += cache[track_id]
            if len(last_all_faces) > 0:
                last_all_embeds = np.stack([last_face.embedding for last_face in last_all_faces])
                cosine_sim = cosine_similarity(face.embedding[None], last_all_embeds)[0]
                if cosine_sim.max() >= cosine_thresh:
                    j = cosine_sim.argmax()
                    track_id = last_all_faces[j].track_id
                    face.track_id = track_id
                    cache[track_id].append(face)
                    assert track_id not in used_track_ids
                    used_track_ids.append(track_id)
                else:
                    left_faces.append(face)
            else:
                left_faces.append(face)
        return left_faces

    def track_by_iou(self, faces, all_last_faces, used_track_ids, cache, iou_thresh):
        last_faces = []
        for last_face in all_last_faces:
            if last_face.track_id not in used_track_ids:
                last_faces.append(last_face)
        if len(faces) > 0 and len(last_faces) > 0:
            boxes = np.stack([face.bbox for face in faces])
            last_boxes = np.stack([last_face.bbox for last_face in last_faces])
            iou = self.compute_iou(boxes, last_boxes)
            used_face_ids = []
            while True:
                if iou.max() >= iou_thresh:
                    idx = iou.argmax()
                    i = idx // iou.shape[1]
                    j = idx % iou.shape[1]
                    track_id = last_faces[j].track_id
                    faces[i].track_id = track_id
                    cache[track_id].append(faces[i])
                    assert track_id not in used_track_ids
                    used_track_ids.append(track_id)
                    used_face_ids.append(i)
                    iou[i] = -1
                    iou[:, j] = -1
                else:
                    break
            left_faces = []
            for i in range(len(faces)):
                if i not in used_face_ids:
                    left_faces.append(faces[i])
        elif len(faces) > 0:
            left_faces = faces
        else:
            left_faces = []
        return left_faces

    def add_new_track(self, faces, cur_track_id, cache):
        if len(faces) > 0 and 'track_id' in faces[0]:
            track_ids = [face.track_id for face in faces]
            cur_track_id = max(track_ids) + 1
        else:
            for face in faces:
                face.track_id = cur_track_id
                cur_track_id += 1
        for face in faces:
            cache[face.track_id] = [face]
        return cur_track_id

    def compute_iou(self, boxes_a, boxes_b, eps=1e-6):
        if boxes_a.shape[0] * boxes_b.shape[0] == 0:
            return np.zeros((boxes_a.shape[0], boxes_b.shape[0]))
        lt = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])  # (n, m, 2)
        rb = np.minimum(boxes_a[:, None, 2:4], boxes_b[None, :, 2:4])  # (n, m, 2)
        wh = np.maximum(rb - lt, 0)  # (n, m, 2)
        overlap = wh[:, :, 0] * wh[:, :, 1]  # (n, m)
        area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
        area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
        union = area_a[:, None] + area_b[None, :] - overlap  # (n, m)
        union = np.maximum(union, eps)
        return overlap / union
