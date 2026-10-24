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

"""ControlNet Encoder based on DreamTransferTransformer3DModel."""

from typing import List, Optional, Tuple

import torch
from diffusers.configuration_utils import register_to_config
from einops import rearrange
from torch import nn
from torchvision import transforms

from ....acceleration import get_sequence_parallel_group, split_forward_gather_backward
from .encoder_dream_transfer import DreamTransferEncoder3DModel
from .module.blocks import PatchEmbed, zero_module
from .transformer_dream_transfer import DreamTransferTransformer3DModel


class DreamTransferControlNet3DModel(DreamTransferTransformer3DModel):
    """ControlNet Encoder based on DreamTransferTransformer3DModel.

    Heavily borrowed from DreamTransferTransformer3DModel with minor modifications.
    """

    @register_to_config
    def __init__(
        self,
        max_img_h: int,
        max_img_w: int,
        max_frames: int,
        in_channels: int,
        out_channels: int,
        patch_spatial: tuple,
        patch_temporal: int,
        concat_padding_mask: bool = True,
        # attention settings
        block_config: str = 'FA-CA-MLP',
        model_channels: int = 768,
        num_blocks: int = 3,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        block_x_format: str = 'BTHWD',
        # cross attention settings
        crossattn_emb_channels: int = 1024,
        use_cross_attn_mask: bool = False,
        # positional embedding settings
        pos_emb_cls: str = 'sincos',
        pos_emb_learnable: bool = False,
        pos_emb_interpolation: str = 'crop',
        affline_emb_norm: bool = False,  # whether or not to normalize the affine embedding
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        rope_h_extrapolation_ratio: float = 1.0,
        rope_w_extrapolation_ratio: float = 1.0,
        rope_t_extrapolation_ratio: float = 1.0,
        extra_per_block_abs_pos_emb: bool = False,
        extra_per_block_abs_pos_emb_type: str = 'sincos',
        extra_h_extrapolation_ratio: float = 1.0,
        extra_w_extrapolation_ratio: float = 1.0,
        extra_t_extrapolation_ratio: float = 1.0,
        n_views: int = 1,
        view_condition_dim: int = 6,
        encoder_cfg: dict = None,
        hint_channels: int = 16,
    ):
        super().__init__(
            max_img_h=max_img_h,
            max_img_w=max_img_w,
            max_frames=max_frames,
            in_channels=in_channels,
            out_channels=out_channels,
            patch_spatial=patch_spatial,
            patch_temporal=patch_temporal,
            concat_padding_mask=concat_padding_mask,
            block_config=block_config,
            model_channels=model_channels,
            num_blocks=num_blocks,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            block_x_format=block_x_format,
            crossattn_emb_channels=crossattn_emb_channels,
            use_cross_attn_mask=use_cross_attn_mask,
            pos_emb_cls=pos_emb_cls,
            pos_emb_learnable=pos_emb_learnable,
            pos_emb_interpolation=pos_emb_interpolation,
            affline_emb_norm=affline_emb_norm,
            use_adaln_lora=use_adaln_lora,
            adaln_lora_dim=adaln_lora_dim,
            rope_h_extrapolation_ratio=rope_h_extrapolation_ratio,
            rope_w_extrapolation_ratio=rope_w_extrapolation_ratio,
            rope_t_extrapolation_ratio=rope_t_extrapolation_ratio,
            extra_per_block_abs_pos_emb=extra_per_block_abs_pos_emb,
            extra_per_block_abs_pos_emb_type=extra_per_block_abs_pos_emb_type,
            extra_h_extrapolation_ratio=extra_h_extrapolation_ratio,
            extra_w_extrapolation_ratio=extra_w_extrapolation_ratio,
            extra_t_extrapolation_ratio=extra_t_extrapolation_ratio,
            with_decode_head=False,
            n_views=n_views,
            view_condition_dim=view_condition_dim,
        )
        if encoder_cfg is not None:
            encoder_type = encoder_cfg.pop('type', 'DreamTransferEncoder3DModel')
            # Older checkpoints may store the pre-release encoder class name.
            if encoder_type.endswith('Encoder3DModel'):
                self.encoder = DreamTransferEncoder3DModel(**encoder_cfg)
            else:
                raise ValueError(f'Unsupported encoder type: {encoder_type}')
        else:
            self.encoder = None
        self.hint_channels = hint_channels
        self.build_hint_patch_embed()
        hint_nf = [16, 16, 32, 32, 96, 96, 256]
        nonlinearity = nn.SiLU()
        input_hint_block = [nn.Linear(model_channels, hint_nf[0]), nonlinearity]
        for i in range(len(hint_nf) - 1):
            input_hint_block += [nn.Linear(hint_nf[i], hint_nf[i + 1]), nonlinearity]
        self.input_hint_block = nn.Sequential(*input_hint_block)
        # Initialize weights
        self.initialize_weights()
        self.zero_blocks = nn.ModuleDict()
        for idx in range(num_blocks):
            self.zero_blocks[f'block{idx}'] = zero_module(nn.Linear(model_channels, model_channels))
        self.input_hint_block.append(zero_module(nn.Linear(hint_nf[-1], model_channels)))

    def build_hint_patch_embed(self):
        concat_padding_mask, in_channels, patch_spatial, patch_temporal, model_channels = (
            self.concat_padding_mask,
            self.hint_channels,
            self.patch_spatial,
            self.patch_temporal,
            self.model_channels,
        )
        in_channels = in_channels + 1 if concat_padding_mask else in_channels
        self.x_embedder2 = PatchEmbed(
            spatial_patch_size=patch_spatial,
            temporal_patch_size=patch_temporal,
            in_channels=in_channels,
            out_channels=model_channels,
            bias=False,
        )

    def load_state_dict_from_transformer(self, transformer):
        base_state_dict = transformer.state_dict()
        state_dict = self.state_dict()
        for key in base_state_dict:
            if key in state_dict:
                state_dict[key][:] = base_state_dict[key]
        self.load_state_dict(state_dict, strict=False)

    def prepare_hint_embedded_sequence(
        self, x_B_C_T_H_W: torch.Tensor, fps: Optional[torch.Tensor] = None, padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.concat_padding_mask:
            padding_mask = transforms.functional.resize(
                padding_mask, list(x_B_C_T_H_W.shape[-2:]), interpolation=transforms.InterpolationMode.NEAREST
            )
            padding_mask = padding_mask.unsqueeze(1).repeat(1, 1, x_B_C_T_H_W.shape[2], 1, 1)
            x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, padding_mask], dim=1)

        x_B_T_H_W_D = self.x_embedder2(x_B_C_T_H_W)

        if 'rope' in self.pos_emb_cls.lower():
            return x_B_T_H_W_D

        if 'fps_aware' in self.pos_emb_cls:
            x_B_T_H_W_D = x_B_T_H_W_D + self.pos_embedder(x_B_T_H_W_D, fps=fps)  # [B, T, H, W, D]
        else:
            x_B_T_H_W_D = x_B_T_H_W_D + self.pos_embedder(x_B_T_H_W_D)  # [B, T, H, W, D]
        return x_B_T_H_W_D

    def encode_hint(
        self,
        hint: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert hint.size(1) <= self.hint_channels, f'Expected hint channels <= {self.hint_channels}, got {hint.size(1)}'
        if hint.size(1) < self.hint_channels:
            padding_shape = list(hint.shape)
            padding_shape[1] = self.hint_channels - hint.size(1)
            hint = torch.cat([hint, torch.zeros(*padding_shape, dtype=hint.dtype, device=hint.device)], dim=1)

        hint_B_T_H_W_D = self.prepare_hint_embedded_sequence(hint, fps=fps, padding_mask=padding_mask)
        hint = rearrange(hint_B_T_H_W_D, 'B T H W D -> T H W B D')

        guided_hint = self.input_hint_block(hint)
        return guided_hint

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        crossattn_emb: torch.Tensor,
        crossattn_mask: Optional[torch.Tensor] = None,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        frame_repeat: Optional[torch.Tensor] = None,
        control_cond: torch.Tensor = None,
        control_weight: float = 1.0,
    ) -> torch.Tensor | List[torch.Tensor] | Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: (B, C, T, H, W) tensor of spatial-temp inputs
            timesteps: (B, ) tensor of timesteps
            crossattn_emb: (B, N, D) tensor of cross-attention embeddings
            crossattn_mask: (B, N) tensor of cross-attention masks
        """
        sp_group = get_sequence_parallel_group()
        if sp_group is not None:
            x = rearrange(x, 'B C (V T) H W -> (B V) C T H W', V=self.n_views)
            x = split_forward_gather_backward(x, dim=2, group=sp_group)
            x = rearrange(x, '(B V) C T H W -> B C (V T) H W', V=self.n_views)

        if self.encoder is not None:
            control_cond = rearrange(control_cond, 'B C (V T) H W -> (B V) C T H W', V=self.n_views)
            control_cond = self.encoder(control_cond)
            control_cond = rearrange(control_cond, '(B V) C T H W -> B C (V T) H W', V=self.n_views)

        if sp_group is not None:
            control_cond = rearrange(control_cond, 'B C (V T) H W -> (B V) C T H W', V=self.n_views)
            control_cond = split_forward_gather_backward(control_cond, dim=2, group=sp_group)
            control_cond = rearrange(control_cond, '(B V) C T H W -> B C (V T) H W', V=self.n_views)

        guided_hint = self.encode_hint(control_cond, fps=fps, padding_mask=padding_mask)

        inputs = self.forward_before_blocks(
            x=x,
            timesteps=timesteps,
            crossattn_emb=crossattn_emb,
            crossattn_mask=crossattn_mask,
            fps=fps,
            padding_mask=padding_mask,
            frame_repeat=frame_repeat,
        )
        x, affline_emb_B_D, crossattn_emb, crossattn_mask, rope_emb_L_1_1_D, adaln_lora_B_3D = (
            inputs['x'],
            inputs['affline_emb_B_D'],
            inputs['crossattn_emb'],
            inputs['crossattn_mask'],
            inputs['rope_emb_L_1_1_D'],
            inputs['adaln_lora_B_3D'],
        )
        extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = inputs['extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D']
        if extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D is not None:
            assert x.shape == extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape, f'{x.shape} != {extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape}'

        outputs = {}
        for name, block in self.blocks.items():
            assert self.blocks['block0'].x_format == block.x_format, f'First block has x_format {self.blocks[0].x_format}, got {block.x_format}'

            x = block(
                x,
                affline_emb_B_D,
                crossattn_emb,
                crossattn_mask,
                rope_emb_L_1_1_D=rope_emb_L_1_1_D,
                adaln_lora_B_3D=adaln_lora_B_3D,
                extra_per_block_pos_emb=extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
            )
            if guided_hint is not None:
                x = x + guided_hint
                guided_hint = None

            if isinstance(control_weight, (float, int)) or control_weight.ndim < 2:
                hint_val = self.zero_blocks[name](x) * control_weight

            else:  # Spatial-temporal weights [num_controls, B, 1, T, H, W]
                control_feat = self.zero_blocks[name](x)
                T, H, W, _, _ = control_feat.shape

                # Get current feature dimensions
                weight_map = control_weight  # [B, 1, T, H, W]
                if weight_map.shape[2:5] != (T, H, W):
                    assert weight_map.shape[2] == 8 * (T - 1) + 1
                    weight_map_i = [
                        torch.nn.functional.interpolate(
                            weight_map[:, :, :1, :, :],
                            size=(1, H, W),
                            mode='trilinear',
                            align_corners=False,
                        )
                    ]
                    for wi in range(1, weight_map.shape[2], 8):
                        weight_map_i += [
                            torch.nn.functional.interpolate(
                                weight_map[:, :, wi : wi + 8],
                                size=(1, H, W),
                                mode='trilinear',
                                align_corners=False,
                            )
                        ]
                    weight_map = torch.cat(weight_map_i, dim=2)
                # Reshape to match THWBD format
                weight_map = weight_map.permute(2, 3, 4, 0, 1)  # [T, H, W, B, 1]
                hint_val = control_feat * weight_map

            outputs[name] = hint_val

        return outputs
