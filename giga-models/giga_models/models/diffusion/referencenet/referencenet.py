import json
import os

import torch.nn as nn

from ....utils import load_state_dict
from ..unet import UNet2DConditionModel


class ReferenceNet:
    @staticmethod
    def from_pretrained(pretrained_model_path, **kwargs):
        try:
            model = UNet2DConditionModel.from_pretrained(pretrained_model_path, **kwargs)
            _skip_modules(model)
        except ValueError:
            torch_dtype = kwargs.pop('torch_dtype', None)
            config_path = os.path.join(pretrained_model_path, 'config.json')
            config = json.load(open(config_path, 'r'))
            model = UNet2DConditionModel.from_config(config)
            _skip_modules(model)
            state_dict = load_state_dict(pretrained_model_path)
            model.load_state_dict(state_dict)
            if torch_dtype is not None:
                model = model.to(dtype=torch_dtype)
        return model


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, input, *args, **kwargs):
        return input


def _skip_modules(model):
    model.up_blocks[-1].attentions[-1].transformer_blocks[-1].attn1 = None
    model.up_blocks[-1].attentions[-1].transformer_blocks[-1].norm2 = Identity()
    model.up_blocks[-1].attentions[-1].transformer_blocks[-1].attn2 = None
    model.up_blocks[-1].attentions[-1].transformer_blocks[-1].norm3 = Identity()
    model.up_blocks[-1].attentions[-1].transformer_blocks[-1].ff = Identity()
    model.up_blocks[-1].attentions[-1].proj_out = Identity()
    model.conv_norm_out = None
    model.conv_act = None
    model.conv_out = Identity()
