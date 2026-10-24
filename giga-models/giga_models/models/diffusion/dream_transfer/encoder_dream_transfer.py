# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin


def cast_tuple(t: Any, length: int = 1) -> Any:
    return t if isinstance(t, tuple) else ((t,) * length)


def replication_pad(x):
    return torch.cat([x[:, :, :1, ...], x], dim=2)


class CausalConv3d(nn.Module):
    def __init__(
        self,
        chan_in: int = 1,
        chan_out: int = 1,
        kernel_size: Union[int, Tuple[int, int, int]] = 3,
        pad_mode: str = 'constant',
        **kwargs,
    ):
        super().__init__()
        kernel_size = cast_tuple(kernel_size, 3)

        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size

        dilation = kwargs.pop('dilation', 1)
        stride = kwargs.pop('stride', 1)
        time_stride = kwargs.pop('time_stride', 1)
        time_dilation = kwargs.pop('time_dilation', 1)
        padding = kwargs.pop('padding', 1)

        self.pad_mode = pad_mode
        time_pad = time_dilation * (time_kernel_size - 1) + (1 - time_stride)
        self.time_pad = time_pad

        self.spatial_pad = (padding, padding, padding, padding)

        stride = (time_stride, stride, stride)
        dilation = (time_dilation, dilation, dilation)
        self.conv3d = nn.Conv3d(
            chan_in,
            chan_out,
            kernel_size,
            stride=stride,
            dilation=dilation,
            **kwargs,
        )

    def _replication_pad(self, x: torch.Tensor) -> torch.Tensor:
        x_prev = x[:, :, :1, ...].repeat(1, 1, self.time_pad, 1, 1)
        x = torch.cat([x_prev, x], dim=2)
        padding = self.spatial_pad + (0, 0)
        return F.pad(x, padding, mode=self.pad_mode, value=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._replication_pad(x)
        return self.conv3d(x)


class CausalDownsample3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = CausalConv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            time_stride=2,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = (0, 1, 0, 1, 0, 0)
        x = F.pad(x, pad, mode='constant', value=0)
        x = replication_pad(x)
        x = self.conv(x)
        return x


class DreamTransferEncoder3DModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            CausalDownsample3d(in_channels, mid_channels),
            nn.SiLU(),
            CausalDownsample3d(mid_channels, mid_channels),
            nn.SiLU(),
            CausalDownsample3d(mid_channels, out_channels),
        )

    def forward(self, x):
        return self.layers(x)
