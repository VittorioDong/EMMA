import torch

from ....models import MotionCtrlModel
from .utils import add_control_model_name


class MotionCtrlMixin:
    def load_motionctrl_model(self, pretrained_model_path, **kwargs):
        kwargs.setdefault('torch_dtype', self.dtype)
        motionctrl = MotionCtrlModel.from_pretrained(pretrained_model_path, **kwargs)
        motionctrl.to_unet(self.unet)
        self.motionctrl = motionctrl
        add_control_model_name('motionctrl')

    def prepare_motionctrl(self, camera_poses):
        if camera_poses is None:
            return
        camera_poses = camera_poses.to(self._execution_device, dtype=self.motionctrl.dtype)
        camera_poses = camera_poses.unsqueeze(0)
        if self.do_classifier_free_guidance:
            camera_poses = torch.cat([camera_poses] * 2)
            camera_poses[: camera_poses.shape[0] // 2] = 0
        self.motionctrl.camera_poses = camera_poses
