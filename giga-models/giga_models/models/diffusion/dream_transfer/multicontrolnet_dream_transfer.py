import os
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from diffusers.models.modeling_utils import ModelMixin

from .controlnet_dream_transfer import DreamTransferControlNet3DModel


class DreamTransferMultiControlNet3DModel(ModelMixin):
    def __init__(self, controlnets: Dict[str, DreamTransferControlNet3DModel]):
        super().__init__()
        self.nets = nn.ModuleDict(controlnets)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        crossattn_emb: torch.Tensor,
        crossattn_mask: Optional[torch.Tensor] = None,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        frame_repeat: Optional[torch.Tensor] = None,
        control_cond: Dict[str, torch.Tensor] = None,
        control_weight: Dict[str, float] = 1.0,
    ) -> torch.Tensor | List[torch.Tensor] | Tuple[torch.Tensor, List[torch.Tensor]]:
        assert len(control_cond) == len(control_weight)
        outputs = dict()
        for key in control_cond:
            outputs_i = self.nets[key](
                x,
                timesteps,
                crossattn_emb=crossattn_emb,
                crossattn_mask=crossattn_mask,
                fps=fps,
                padding_mask=padding_mask,
                frame_repeat=frame_repeat,
                control_cond=control_cond[key],
                control_weight=control_weight[key],
            )
            for name, val in outputs_i.items():
                if name not in outputs:
                    outputs[name] = val
                else:
                    outputs[name] += val
        return outputs

    def save_pretrained(self, save_directory: Union[str, os.PathLike], **kwargs):
        for name, controlnet in self.nets.items():
            controlnet.save_pretrained(os.path.join(save_directory, name), **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_path: Optional[Union[str, os.PathLike]], **kwargs):
        names = os.listdir(pretrained_model_path)
        controlnets = dict()
        for name in names:
            controlnet = DreamTransferControlNet3DModel.from_pretrained(os.path.join(pretrained_model_path, name), **kwargs)
            controlnets[name] = controlnet
        if len(controlnets) == 0:
            raise ValueError(f'No ControlNets found under {os.path.dirname(pretrained_model_path)}. ' f'Expected at least {pretrained_model_path}.')
        return cls(controlnets)
