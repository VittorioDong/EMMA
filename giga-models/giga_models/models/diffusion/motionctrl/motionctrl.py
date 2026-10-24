import functools

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin


class MotionCtrlModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, camera_poses_cfg):
        super().__init__()
        self.camera_pose_names = []
        self.camera_pose_modules = nn.ModuleList()
        for camera_pose_name, camera_pose_cfg in camera_poses_cfg:
            camera_pose_module = nn.Linear(**camera_pose_cfg)
            self.camera_pose_names.append(camera_pose_name)
            self.camera_pose_modules.append(camera_pose_module)
        self.camera_poses = None

    @classmethod
    def from_unet(cls, unet, add_in_channels=12):
        camera_poses_cfg = []
        for name, module in unet.named_modules():
            if module.__class__.__name__ == 'TemporalBasicTransformerBlock':
                query_dim = module.attn1.to_out[0].weight.shape[0]
                camera_pose_cfg = dict(
                    in_features=query_dim + add_in_channels,
                    out_features=query_dim,
                )
                camera_poses_cfg.append((name, camera_pose_cfg))
        motion_ctrl = cls(camera_poses_cfg)
        return motion_ctrl

    def to_unet(self, unet):
        for name, module in unet.named_modules():
            if module.__class__.__name__ == 'TemporalBasicTransformerBlock':
                module.camera_pose_proj = functools.partial(self, name)

    def forward(self, key, hidden_states):
        seq_length = hidden_states.shape[0] // self.camera_poses.shape[0]
        camera_poses = self.camera_poses.repeat_interleave(seq_length, dim=0)
        hidden_states = torch.cat([hidden_states, camera_poses], dim=-1)
        idx = self.camera_pose_names.index(key)
        hidden_states = self.camera_pose_modules[idx](hidden_states)
        return hidden_states
