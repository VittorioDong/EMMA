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

"""A general implementation of adaln-modulated VIT-like~(DiT) transformer for
video processing."""
import json
import os
from collections import OrderedDict
from typing import List, Optional, Tuple

import accelerate
import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.models.model_loading_utils import load_model_dict_into_meta
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils.peft_utils import get_peft_kwargs, set_weights_and_activate_adapters
from einops import rearrange
from peft import LoraConfig, set_peft_model_state_dict
from torch import nn
from torchvision import transforms

from ....acceleration import gather_forward_split_backward, get_sequence_parallel_group, split_forward_gather_backward
from ....exports.transformer_engine import convert_model_torch_to_te
from ....utils import load_state_dict
from .module.attention import Attention, get_normalization
from .module.blocks import FinalLayer, GeneralDITTransformerBlock, PatchEmbed, TimestepEmbedding, Timesteps
from .module.position_embedding import LearnablePosEmbAxis, SinCosPosEmbAxis, VideoRopePosition3DEmb


class DreamTransferTransformer3DModel(ModelMixin, ConfigMixin, PeftAdapterMixin):
    """A general implementation of adaln-modulated VIT-like~(DiT) transformer
    for video processing.

    Args:
        max_img_h (int): Maximum height of the input images.
        max_img_w (int): Maximum width of the input images.
        max_frames (int): Maximum number of frames in the video sequence.
        in_channels (int): Number of input channels (e.g., RGB channels for color images).
        out_channels (int): Number of output channels.
        patch_spatial (tuple): Spatial resolution of patches for input processing.
        patch_temporal (int): Temporal resolution of patches for input processing.
        concat_padding_mask (bool): If True, includes a mask channel in the input to handle padding.
        block_config (str): Configuration of the transformer block. See Notes for supported block types.
        model_channels (int): Base number of channels used throughout the model.
        num_blocks (int): Number of transformer blocks.
        num_heads (int): Number of heads in the multi-head attention layers.
        mlp_ratio (float): Expansion ratio for MLP blocks.
        block_x_format (str): Format of input tensor for transformer blocks ('BTHWD' or 'THWBD').
        crossattn_emb_channels (int): Number of embedding channels for cross-attention.
        use_cross_attn_mask (bool): Whether to use mask in cross-attention.
        pos_emb_cls (str): Type of positional embeddings.
        pos_emb_learnable (bool): Whether positional embeddings are learnable.
        pos_emb_interpolation (str): Method for interpolating positional embeddings.
        affline_emb_norm (bool): Whether to normalize affine embeddings.
        use_adaln_lora (bool): Whether to use AdaLN-LoRA.
        adaln_lora_dim (int): Dimension for AdaLN-LoRA.
        rope_h_extrapolation_ratio (float): Height extrapolation ratio for RoPE.
        rope_w_extrapolation_ratio (float): Width extrapolation ratio for RoPE.
        rope_t_extrapolation_ratio (float): Temporal extrapolation ratio for RoPE.
        extra_per_block_abs_pos_emb (bool): Whether to use extra per-block absolute positional embeddings.
        extra_per_block_abs_pos_emb_type (str): Type of extra per-block positional embeddings.
        extra_h_extrapolation_ratio (float): Height extrapolation ratio for extra embeddings.
        extra_w_extrapolation_ratio (float): Width extrapolation ratio for extra embeddings.
        extra_t_extrapolation_ratio (float): Temporal extrapolation ratio for extra embeddings.

    Notes:
        Supported block types in block_config:
        * cross_attn, ca: Cross attention
        * full_attn: Full attention on all flattened tokens
        * mlp, ff: Feed forward block
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
        num_blocks: int = 10,
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
        with_decode_head: float = True,
        n_views: int = 1,
        view_condition_dim: int = 6,
    ) -> None:
        super().__init__()
        self.max_img_h = max_img_h
        self.max_img_w = max_img_w
        self.max_frames = max_frames
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_spatial = patch_spatial
        self.patch_temporal = patch_temporal
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.model_channels = model_channels
        self.use_cross_attn_mask = use_cross_attn_mask
        self.concat_padding_mask = concat_padding_mask
        # positional embedding settings
        self.pos_emb_cls = pos_emb_cls
        self.pos_emb_learnable = pos_emb_learnable
        self.pos_emb_interpolation = pos_emb_interpolation
        self.affline_emb_norm = affline_emb_norm
        self.rope_h_extrapolation_ratio = rope_h_extrapolation_ratio
        self.rope_w_extrapolation_ratio = rope_w_extrapolation_ratio
        self.rope_t_extrapolation_ratio = rope_t_extrapolation_ratio
        self.extra_per_block_abs_pos_emb = extra_per_block_abs_pos_emb
        self.extra_per_block_abs_pos_emb_type = extra_per_block_abs_pos_emb_type.lower()
        self.extra_h_extrapolation_ratio = extra_h_extrapolation_ratio
        self.extra_w_extrapolation_ratio = extra_w_extrapolation_ratio
        self.extra_t_extrapolation_ratio = extra_t_extrapolation_ratio
        self.n_views = n_views
        self.view_condition_dim = view_condition_dim

        self.build_patch_embed()
        self.build_pos_embed()
        self.block_x_format = block_x_format
        self.use_adaln_lora = use_adaln_lora
        self.adaln_lora_dim = adaln_lora_dim
        self.t_embedder = nn.Sequential(
            Timesteps(model_channels),
            TimestepEmbedding(model_channels, model_channels, use_adaln_lora=use_adaln_lora),
        )

        self.blocks = nn.ModuleDict()

        for idx in range(num_blocks):
            self.blocks[f'block{idx}'] = GeneralDITTransformerBlock(
                x_dim=model_channels,
                context_dim=crossattn_emb_channels,
                num_heads=num_heads,
                block_config=block_config,
                mlp_ratio=mlp_ratio,
                x_format=self.block_x_format,
                use_adaln_lora=use_adaln_lora,
                adaln_lora_dim=adaln_lora_dim,
                n_views=n_views,
            )

        if with_decode_head:
            self.build_decode_head()
        if self.affline_emb_norm:
            self.affline_norm = get_normalization('R', model_channels)
        else:
            self.affline_norm = nn.Identity()

        if n_views > 1:
            self.view_embeddings = nn.Embedding(n_views, view_condition_dim)
            self.repeat_frame_embedding = nn.Linear(1, view_condition_dim)

        self.fp8 = False
        self.fp8_config = dict()

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding
        nn.init.normal_(self.t_embedder[1].linear_1.weight, std=0.02)
        if self.t_embedder[1].linear_1.bias is not None:
            nn.init.constant_(self.t_embedder[1].linear_1.bias, 0)
        nn.init.normal_(self.t_embedder[1].linear_2.weight, std=0.02)
        if self.t_embedder[1].linear_2.bias is not None:
            nn.init.constant_(self.t_embedder[1].linear_2.bias, 0)

        # Zero-out adaLN modulation layers in DiT blocks:
        for transformer_block in self.blocks.values():
            for block in transformer_block.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                if block.adaLN_modulation[-1].bias is not None:
                    nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def to_fp8(self, **kwargs):
        self.fp8_config['convert_cfg'] = kwargs
        convert_model_torch_to_te(self, **kwargs)
        self.fp8 = True

    def state_dict(self, *args, destination=None, **kwargs):
        assert len(args) == 0
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()
        super().state_dict(destination=destination, **kwargs)
        if not self.fp8:
            ignore_keys = []
            for key, val in destination.items():
                if key.endswith('_extra_state'):
                    ignore_keys.append(key)
            for key in ignore_keys:
                destination.pop(key)
        return destination

    def save_config(self, save_directory, *args, **kwargs):
        super().save_config(save_directory, *args, **kwargs)
        if self.fp8:
            save_path = os.path.join(save_directory, 'config_fp8.json')
            json.dump(self.fp8_config, open(save_path, 'w'), indent=2)

    @classmethod
    def load_config(cls, pretrained_model_name_or_path, *args, **kwargs):
        config = super().load_config(pretrained_model_name_or_path, *args, **kwargs)
        fp8_config_path = os.path.join(pretrained_model_name_or_path, 'config_fp8.json')
        if os.path.exists(fp8_config_path):
            fp8_config = json.load(open(fp8_config_path, 'r'))
            config[0]['_fp8_config'] = fp8_config
        return config

    @classmethod
    def from_config(cls, config, *args, **kwargs):
        fp8_config = config.pop('_fp8_config', None)
        model = super().from_config(config, *args, **kwargs)
        if fp8_config is not None:
            model.fp8_config = fp8_config
            model.to_fp8(**fp8_config['convert_cfg'])
        return model

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        fp8_config_path = os.path.join(pretrained_model_name_or_path, 'config_fp8.json')
        if os.path.exists(fp8_config_path):
            kwargs['low_cpu_mem_usage'] = False
        try:
            model = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        except ValueError as _:  # noqa F841
            config = super().load_config(pretrained_model_name_or_path)
            with accelerate.init_empty_weights():
                model = super().from_config(config)
            state_dict = load_state_dict(pretrained_model_name_or_path)
            load_model_dict_into_meta(model, state_dict)
            model.eval()
        return model

    def load_lora(self, model_path_dict, fuse=False):
        if isinstance(model_path_dict, str):
            model_path_dict = {'default': model_path_dict}
        adapter_names = list(model_path_dict.keys())
        lora_scales = []
        for adapter_name in adapter_names:
            model_info = model_path_dict[adapter_name]
            if isinstance(model_info, str):
                model_path, lora_scale = model_info, 1.0
            else:
                model_path, lora_scale = model_info
            state_dict = load_state_dict(model_path)
            rank = {}
            for key, val in state_dict.items():
                if 'lora_B' in key and val.ndim > 1:
                    rank[key] = val.shape[1]
            lora_config_kwargs = get_peft_kwargs(rank, network_alpha_dict=None, peft_state_dict=state_dict)
            lora_config = LoraConfig(**lora_config_kwargs)
            self.add_adapter(lora_config, adapter_name=adapter_name)
            fp8_config_path = os.path.join(model_path, 'config_fp8.json')
            if os.path.exists(fp8_config_path):
                fp8_config = json.load(open(fp8_config_path, 'r'))
                self.fp8_config = fp8_config
                self.to_fp8(**fp8_config['convert_cfg'])
            incompatible_keys = set_peft_model_state_dict(self, state_dict, adapter_name=adapter_name)
            assert len(incompatible_keys.unexpected_keys) == 0
            lora_scales.append(lora_scale)
        set_weights_and_activate_adapters(self, adapter_names=adapter_names, weights=lora_scales)
        if fuse:
            self.fuse_lora(adapter_names=adapter_names)

    def build_decode_head(self):
        self.final_layer = FinalLayer(
            hidden_size=self.model_channels,
            spatial_patch_size=self.patch_spatial,
            temporal_patch_size=self.patch_temporal,
            out_channels=self.out_channels,
            use_adaln_lora=self.use_adaln_lora,
            adaln_lora_dim=self.adaln_lora_dim,
        )

    def build_patch_embed(self):
        (
            concat_padding_mask,
            in_channels,
            patch_spatial,
            patch_temporal,
            model_channels,
        ) = (
            self.concat_padding_mask,
            self.in_channels,
            self.patch_spatial,
            self.patch_temporal,
            self.model_channels,
        )
        in_channels = in_channels + 1 if concat_padding_mask else in_channels
        if self.n_views > 1:
            in_channels = in_channels + self.view_condition_dim
        self.x_embedder = PatchEmbed(
            spatial_patch_size=patch_spatial,
            temporal_patch_size=patch_temporal,
            in_channels=in_channels,
            out_channels=model_channels,
            bias=False,
        )

    def build_pos_embed(self):
        if self.pos_emb_cls == 'rope3d':
            cls_type = VideoRopePosition3DEmb
        else:
            raise ValueError(f'Unknown pos_emb_cls {self.pos_emb_cls}')

        kwargs = dict(
            model_channels=self.model_channels,
            len_h=self.max_img_h // self.patch_spatial,
            len_w=self.max_img_w // self.patch_spatial,
            len_t=self.max_frames // self.patch_temporal,
            is_learnable=self.pos_emb_learnable,
            interpolation=self.pos_emb_interpolation,
            head_dim=self.model_channels // self.num_heads,
            h_extrapolation_ratio=self.rope_h_extrapolation_ratio,
            w_extrapolation_ratio=self.rope_w_extrapolation_ratio,
            t_extrapolation_ratio=self.rope_t_extrapolation_ratio,
            n_views=self.n_views,
        )
        self.pos_embedder = cls_type(**kwargs)

        if self.extra_per_block_abs_pos_emb:
            assert self.extra_per_block_abs_pos_emb_type in [
                'learnable',
                'sincos',
            ], f'Unknown extra_per_block_abs_pos_emb_type {self.extra_per_block_abs_pos_emb_type}'
            kwargs['h_extrapolation_ratio'] = self.extra_h_extrapolation_ratio
            kwargs['w_extrapolation_ratio'] = self.extra_w_extrapolation_ratio
            kwargs['t_extrapolation_ratio'] = self.extra_t_extrapolation_ratio
            if self.extra_per_block_abs_pos_emb_type == 'learnable':
                self.extra_pos_embedder = LearnablePosEmbAxis(**kwargs)
            elif self.extra_per_block_abs_pos_emb_type == 'sincos':
                self.extra_pos_embedder = SinCosPosEmbAxis(**kwargs)

    def set_attn_op(self, backend):
        for module in self.modules():
            if isinstance(module, Attention):
                module.set_attn_op(backend)

    def prepare_embedded_sequence(
        self,
        x_B_C_T_H_W: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        frame_repeat: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Prepares an embedded sequence tensor by applying positional
        embeddings and handling padding masks.

        Args:
            x_B_C_T_H_W (torch.Tensor): video
            fps (Optional[torch.Tensor]): Frames per second tensor to be used for positional embedding when required.
                                    If None, a default value (`self.base_fps`) will be used.
            padding_mask (Optional[torch.Tensor]): current it is not used

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - A tensor of shape (B, T, H, W, D) with the embedded sequence.
                - An optional positional embedding tensor, returned only if the positional embedding class
                (`self.pos_emb_cls`) includes 'rope'. Otherwise, None.

        Notes:
            - If `self.concat_padding_mask` is True, a padding mask channel is concatenated to the input tensor.
            - The method of applying positional embeddings depends on the value of `self.pos_emb_cls`.
            - If 'rope' is in `self.pos_emb_cls` (case insensitive), the positional embeddings are generated using
                the `self.pos_embedder` with the shape [T, H, W].
            - If "fps_aware" is in `self.pos_emb_cls`, the positional embeddings are generated using the
            `self.pos_embedder` with the fps tensor.
            - Otherwise, the positional embeddings are generated without considering fps.
        """
        if self.concat_padding_mask:
            padding_mask = transforms.functional.resize(
                padding_mask, list(x_B_C_T_H_W.shape[-2:]), interpolation=transforms.InterpolationMode.NEAREST
            )
            padding_mask = padding_mask.unsqueeze(1).repeat(1, 1, x_B_C_T_H_W.shape[2], 1, 1)
            x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, padding_mask], dim=1)

        if self.n_views > 1:
            view_indices = torch.arange(self.n_views).to(x_B_C_T_H_W.device)  # View indices [0, 1, ..., V-1]
            view_embedding = self.view_embeddings(view_indices)  # Shape: [V, embedding_dim]
            view_embedding = rearrange(view_embedding, 'V D -> D V')
            view_embedding = view_embedding.unsqueeze(0).unsqueeze(3).unsqueeze(4).unsqueeze(5)  # Shape: [1, D, V, 1, 1, 1]
            if frame_repeat is None:
                frame_repeat = torch.zeros([x_B_C_T_H_W.shape[0], view_embedding.shape[1]]).to(view_embedding.device).to(view_embedding.dtype)
            frame_repeat_embedding = self.repeat_frame_embedding(frame_repeat.unsqueeze(-1))
            frame_repeat_embedding = rearrange(frame_repeat_embedding, 'B V D -> B D V')
            view_embedding = view_embedding + frame_repeat_embedding.unsqueeze(3).unsqueeze(4).unsqueeze(5)
            x_B_C_V_T_H_W = rearrange(x_B_C_T_H_W, 'B C (V T) H W -> B C V T H W', V=self.n_views)
            view_embedding = view_embedding.expand(
                x_B_C_V_T_H_W.shape[0],
                view_embedding.shape[1],
                view_embedding.shape[2],
                x_B_C_V_T_H_W.shape[3],
                x_B_C_V_T_H_W.shape[4],
                x_B_C_V_T_H_W.shape[5],
            )  # Shape: [B, D, V, T, H, W]
            x_B_C_V_T_H_W = torch.cat([x_B_C_V_T_H_W, view_embedding], dim=1)
            x_B_C_T_H_W = rearrange(x_B_C_V_T_H_W, ' B C V T H W -> B C (V T) H W', V=self.n_views)

        x_B_T_H_W_D = self.x_embedder(x_B_C_T_H_W)

        if self.extra_per_block_abs_pos_emb:
            extra_pos_emb = self.extra_pos_embedder(x_B_T_H_W_D, fps=fps)
        else:
            extra_pos_emb = None

        if 'rope' in self.pos_emb_cls.lower():
            return x_B_T_H_W_D, self.pos_embedder(x_B_T_H_W_D, fps=fps), extra_pos_emb

        if 'fps_aware' in self.pos_emb_cls:
            x_B_T_H_W_D = x_B_T_H_W_D + self.pos_embedder(x_B_T_H_W_D, fps=fps)  # [B, T, H, W, D]
        else:
            x_B_T_H_W_D = x_B_T_H_W_D + self.pos_embedder(x_B_T_H_W_D)  # [B, T, H, W, D]

        return x_B_T_H_W_D, None, extra_pos_emb

    def decoder_head(
        self,
        x_B_T_H_W_D: torch.Tensor,
        emb_B_D: torch.Tensor,
        origin_shape: Tuple[int, int, int, int, int],  # [B, C, T, H, W]
        adaln_lora_B_3D: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, C, T_before_patchify, H_before_patchify, W_before_patchify = origin_shape
        x_BT_HW_D = rearrange(x_B_T_H_W_D, 'B T H W D -> (B T) (H W) D')
        x_BT_HW_D = self.final_layer(x_BT_HW_D, emb_B_D, adaln_lora_B_3D=adaln_lora_B_3D)
        # This is to ensure x_BT_HW_D has the correct shape because
        # when we merge T, H, W into one dimension, x_BT_HW_D has shape (B * T * H * W, 1*1, D).
        x_BT_HW_D = x_BT_HW_D.view(
            B * T_before_patchify // self.patch_temporal,
            H_before_patchify // self.patch_spatial * W_before_patchify // self.patch_spatial,
            -1,
        )
        x_B_D_T_H_W = rearrange(
            x_BT_HW_D,
            '(B T) (H W) (p1 p2 t C) -> B C (T t) (H p1) (W p2)',
            p1=self.patch_spatial,
            p2=self.patch_spatial,
            H=H_before_patchify // self.patch_spatial,
            W=W_before_patchify // self.patch_spatial,
            t=self.patch_temporal,
            B=B,
        )
        return x_B_D_T_H_W

    def forward_before_blocks(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        crossattn_emb: torch.Tensor,
        crossattn_mask: Optional[torch.Tensor] = None,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        frame_repeat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, H, W) tensor of spatial-temp inputs
            timesteps: (B, ) tensor of timesteps
            crossattn_emb: (B, N, D) tensor of cross-attention embeddings
            crossattn_mask: (B, N) tensor of cross-attention masks
        """
        if fps is not None:
            fps = fps.flatten()
        x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = self.prepare_embedded_sequence(
            x,
            fps=fps,
            padding_mask=padding_mask,
            frame_repeat=frame_repeat,
        )
        # logging affline scale information
        timesteps_B_D, adaln_lora_B_3D = self.t_embedder(timesteps.flatten())
        affline_emb_B_D = self.affline_norm(timesteps_B_D)

        if self.use_cross_attn_mask:
            crossattn_mask = crossattn_mask[:, None, None, :].to(dtype=torch.bool)  # [B, 1, 1, length]
        else:
            crossattn_mask = None

        if self.blocks['block0'].x_format == 'THWBD':
            x = rearrange(x_B_T_H_W_D, 'B T H W D -> T H W B D')
            if extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D is not None:
                extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = rearrange(extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D, 'B T H W D -> T H W B D')
            crossattn_emb = rearrange(crossattn_emb, 'B M D -> M B D')

            if crossattn_mask:
                crossattn_mask = rearrange(crossattn_mask, 'B M -> M B')

        elif self.blocks['block0'].x_format == 'BTHWD':
            x = x_B_T_H_W_D
        else:
            raise ValueError(f'Unknown x_format {self.blocks[0].x_format}')
        output = {
            'x': x,
            'affline_emb_B_D': affline_emb_B_D,
            'crossattn_emb': crossattn_emb,
            'crossattn_mask': crossattn_mask,
            'rope_emb_L_1_1_D': rope_emb_L_1_1_D,
            'adaln_lora_B_3D': adaln_lora_B_3D,
            'extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D': extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
        }
        return output

    def forward_blocks(self, x, *args, x_ctrl=None, **kwargs):
        for name, block in self.blocks.items():
            x = block(x, *args, **kwargs)
            if x_ctrl is not None and name in x_ctrl:
                x = x + x_ctrl[name]
        return x

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        crossattn_emb: torch.Tensor,
        crossattn_mask: Optional[torch.Tensor] = None,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        frame_repeat: Optional[torch.Tensor] = None,
        x_ctrl: Optional[dict] = None,
    ) -> torch.Tensor | List[torch.Tensor] | Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: (B, C, T, H, W) tensor of spatial-temp inputs
            timesteps: (B, ) tensor of timesteps
            crossattn_emb: (B, N, D) tensor of cross-attention embeddings
            crossattn_mask: (B, N) tensor of cross-attention masks
            condition_video_augment_sigma: (B,) used in lvg(long video generation), we add noise with this sigma to
                augment condition input, the lvg model will condition on the condition_video_augment_sigma value;
                we need forward_before_blocks pass to the forward_before_blocks function.
        """
        sp_group = get_sequence_parallel_group()
        if sp_group is not None:
            x = rearrange(x, 'B C (V T) H W -> (B V) C T H W', V=self.n_views)
            x = split_forward_gather_backward(x, dim=2, group=sp_group)
            x = rearrange(x, '(B V) C T H W -> B C (V T) H W', V=self.n_views)

        original_shape = x.shape
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

        x = self.forward_blocks(
            x,
            affline_emb_B_D,
            crossattn_emb,
            crossattn_mask,
            rope_emb_L_1_1_D=rope_emb_L_1_1_D,
            adaln_lora_B_3D=adaln_lora_B_3D,
            extra_per_block_pos_emb=extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
            x_ctrl=x_ctrl,
        )

        x_B_T_H_W_D = rearrange(x, 'T H W B D -> B T H W D')

        x_B_D_T_H_W = self.decoder_head(
            x_B_T_H_W_D=x_B_T_H_W_D,
            emb_B_D=affline_emb_B_D,
            origin_shape=original_shape,
            adaln_lora_B_3D=adaln_lora_B_3D,
        )

        if sp_group is not None:
            x_B_D_T_H_W = rearrange(x_B_D_T_H_W, 'B C (V T) H W -> (B V) C T H W', V=self.n_views)
            x_B_D_T_H_W = gather_forward_split_backward(x_B_D_T_H_W, dim=2, group=sp_group)
            x_B_D_T_H_W = rearrange(x_B_D_T_H_W, '(B V) C T H W -> B C (V T) H W', V=self.n_views)

        return x_B_D_T_H_W
