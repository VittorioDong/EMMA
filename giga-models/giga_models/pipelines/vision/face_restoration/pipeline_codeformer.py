import os.path

import numpy as np
import torch
from codeformer.basicsr.utils import img2tensor, tensor2img
from codeformer.basicsr.utils.registry import ARCH_REGISTRY
from codeformer.facelib.detection.retinaface import retinaface
from codeformer.facelib.utils import misc
from codeformer.facelib.utils.face_restoration_helper import FaceRestoreHelper
from PIL import Image
from torchvision.transforms.functional import normalize

from ...pipeline import BasePipeline


class CodeFormerPipeline(BasePipeline):
    def __init__(self, model_path):
        misc.ROOT_DIR = os.path.dirname(model_path)
        self.net = ARCH_REGISTRY.get('CodeFormer')(
            dim_embd=512,
            codebook_size=1024,
            n_head=8,
            n_layers=9,
            connect_list=['32', '64', '128', '256'],
        )
        checkpoint = torch.load(model_path)['params_ema']
        self.net.load_state_dict(checkpoint)
        self.net.eval()
        self.device = 'cpu'
        retinaface.device = self.device
        self.face_helper = FaceRestoreHelper(
            upscale_factor=1,
            face_size=512,
            crop_ratio=(1, 1),
            det_model='retinaface_resnet50',
            save_ext='png',
            use_parse=True,
            device=self.device,
        )

    def to(self, device):
        retinaface.device = device
        self.device = device
        self.net.to(device)
        self.face_helper.device = device
        self.face_helper.face_detector.to(device)
        self.face_helper.face_detector.mean_tensor = self.face_helper.face_detector.mean_tensor.to(device)
        self.face_helper.face_parse.to(device)
        return self

    def process_faces(self, image, mode=None, max_faces=None, points=None):
        self.face_helper.clean_all()
        self.face_helper.read_image(np.array(image)[:, :, ::-1])
        self.face_helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
        if len(self.face_helper.det_faces) == 0:
            return
        if mode is not None:
            boxes = np.array(self.face_helper.det_faces)
            if mode == 'area':
                data = -(boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            elif mode == 'width':
                data = -(boxes[:, 2] - boxes[:, 0])
            elif mode == 'height':
                data = -(boxes[:, 3] - boxes[:, 1])
            elif mode == 'x1':
                data = boxes[:, 0]
            elif mode == 'y1':
                data = boxes[:, 1]
            else:
                assert False
            indexes = data.argsort()
            self.face_helper.det_faces = [self.face_helper.det_faces[idx] for idx in indexes]
            self.face_helper.all_landmarks_5 = [self.face_helper.all_landmarks_5[idx] for idx in indexes]
        if max_faces is not None:
            self.face_helper.det_faces = self.face_helper.det_faces[:max_faces]
            self.face_helper.all_landmarks_5 = self.face_helper.all_landmarks_5[:max_faces]
        if points is not None:
            indexes = []
            for point in points:
                for idx, det_face in enumerate(self.face_helper.det_faces):
                    if idx in indexes:
                        continue
                    if point_in_box(point, det_face):
                        indexes.append(idx)
            self.face_helper.det_faces = [self.face_helper.det_faces[idx] for idx in indexes]
            self.face_helper.all_landmarks_5 = [self.face_helper.all_landmarks_5[idx] for idx in indexes]

    def __call__(self, image, mode=None, max_faces=None, points=None):
        self.process_faces(image, mode=mode, max_faces=max_faces, points=points)
        if len(self.face_helper.det_faces) > 0:
            self.face_helper.align_warp_face()
            for cropped_face in self.face_helper.cropped_faces:
                cropped_face_t = img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
                normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                cropped_face_t = cropped_face_t.unsqueeze(0).to(self.device)
                with torch.no_grad():
                    output = self.net(cropped_face_t, w=0.5, adain=True)[0]
                restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                restored_face = restored_face.astype('uint8')
                self.face_helper.add_restored_face(restored_face, cropped_face)
            self.face_helper.get_inverse_affine()
            restored_image = self.face_helper.paste_faces_to_input_image()
            restored_image = Image.fromarray(restored_image[:, :, ::-1])
            restored_image = restored_image.resize(image.size)
        else:
            restored_image = image
        self.face_helper.clean_all()
        return restored_image


def point_in_box(point, box):
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]
