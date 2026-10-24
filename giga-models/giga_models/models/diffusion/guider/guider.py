import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.autoencoders.autoencoder_kl_cogvideox import CogVideoXCausalConv3d as _CogVideoXCausalConv3d
from diffusers.models.downsampling import CogVideoXDownsample3D
from diffusers.models.modeling_utils import ModelMixin
from torch.nn import functional as F


class GuiderModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, guider_cfg):
        super().__init__()
        guider_cfg = guider_cfg.copy()
        guider_type = guider_cfg.pop('type')
        self.guider = globals()[guider_type](**guider_cfg)
        self.gradient_checkpointing = False

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False

    def forward(self, *args, **kwargs):
        if self.gradient_checkpointing:
            output = torch.utils.checkpoint.checkpoint(self.guider, *args, use_reentrant=False, **kwargs)
        else:
            output = self.guider(*args, **kwargs)
        return output


class GuiderNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=4,
        block_out_channels=(16, 32, 96, 256),
        init_zero=True,
    ):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([])
        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            self.blocks.append(nn.SiLU())
            self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))
            self.blocks.append(nn.SiLU())
        self.conv_out = nn.Conv2d(block_out_channels[-1], out_channels, kernel_size=3, padding=1)
        if init_zero:
            self.conv_out = zero_module(self.conv_out)

    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)
        for block in self.blocks:
            embedding = block(embedding)
        embedding = self.conv_out(embedding)
        return embedding


class GuiderNet2(nn.Module):
    def __init__(self, in_channels=3, mid_channels=4, out_channels=8):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(mid_channels, mid_channels, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(mid_channels, out_channels, 4, 2, 1),
        )

    def forward(self, x):
        return self.layers(x)


class GuiderNet3DCogVideoX(nn.Module):
    def __init__(self, in_channels=3, mid_channels=4, out_channels=8):
        super().__init__()
        self.layers = nn.Sequential(
            CogVideoXDownsample3D(in_channels, mid_channels, 3, 2, compress_time=True),
            nn.SiLU(),
            CogVideoXDownsample3D(mid_channels, mid_channels, 3, 2, compress_time=True),
            nn.SiLU(),
            CogVideoXDownsample3D(mid_channels, out_channels, 3, 2),
        )

    def forward(self, x):
        return self.layers(x)


class GuiderNet3DCogVideoX2(nn.Module):
    def __init__(self, in_channels=16, mid_channels=16, out_channels=16):
        super().__init__()
        self.layers = nn.Sequential(
            CogVideoXCausalConv3d(in_channels, mid_channels, 3, 1),
            nn.SiLU(),
            CogVideoXCausalConv3d(mid_channels, mid_channels, 3, 1),
            nn.SiLU(),
            CogVideoXCausalConv3d(mid_channels, mid_channels, 3, 1),
            nn.SiLU(),
            CogVideoXCausalConv3d(mid_channels, mid_channels, 3, 1),
            nn.SiLU(),
            CogVideoXCausalConv3d(mid_channels, out_channels, 3, 1),
        )

    def forward(self, x):
        return self.layers(x)


class CogVideoXCausalConv3d(_CogVideoXCausalConv3d):
    def fake_context_parallel_forward(self, inputs):
        kernel_size = self.time_kernel_size
        if kernel_size > 1:
            cached_inputs = [inputs[:, :, :1]] * (kernel_size - 1)
            inputs = torch.cat(cached_inputs + [inputs], dim=2)
        return inputs


def pad_at_dim(t, pad, dim=-1):
    dims_from_right = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = (0, 0) * dims_from_right
    return F.pad(t, (*zeros, *pad), mode='constant')


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module
