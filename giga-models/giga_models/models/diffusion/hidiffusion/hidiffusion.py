from types import MethodType
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from diffusers.models.attention import _chunked_feed_forward
from diffusers.utils import is_torch_version


class HiDiffusionControl:
    def __init__(
        self,
        unet,
        max_timestep,
        ratios=(0.4, 0.0),
    ):
        self.unet = unet
        self.max_timestep = max_timestep
        self.ratios = ratios
        self.module_names = {
            'basic_block_names': [
                'down_blocks.0.attentions.0.transformer_blocks.0',
                'down_blocks.0.attentions.1.transformer_blocks.0',
                'up_blocks.3.attentions.0.transformer_blocks.0',
                'up_blocks.3.attentions.1.transformer_blocks.0',
                'up_blocks.3.attentions.2.transformer_blocks.0',
            ],
            'down_block_names': ['down_blocks.1'],
            'downsample_conv_names': ['down_blocks.0.downsamplers.0.conv'],
            'up_block_names': ['up_blocks.2'],
            'upsample_conv_names': ['up_blocks.2.upsamplers.0.conv'],
        }
        self.modules = []
        self.register_hooks()

    def register_hooks(self):
        self.unet.register_forward_pre_hook(UNetHook.forward_pre)
        for name, module in self.unet.named_modules():
            if name in self.module_names['basic_block_names']:
                hook = BasicTransformerBlockHook
            elif name in self.module_names['down_block_names']:
                hook = CrossAttnDownBlockSpatioTemporalHook
            elif name in self.module_names['downsample_conv_names']:
                hook = Downsample2DConvHook
            elif name in self.module_names['up_block_names']:
                hook = CrossAttnUpBlockSpatioTemporalHook
            elif name in self.module_names['upsample_conv_names']:
                hook = Upsample2DConvHook
            else:
                hook = None
            if hook is not None:
                module._original_forward = module.forward
                module.forward = MethodType(hook.forward, module)
                idx = 1 if hook in (CrossAttnDownBlockSpatioTemporalHook, CrossAttnUpBlockSpatioTemporalHook) else 0
                module._info = {
                    'max_timestep': self.max_timestep,
                    'timestep': 0,
                    't1_ratio': self.ratios[idx],
                }
                self.modules.append(module)

    def clear(self):
        for module in self.modules:
            module._info.pop('input_h', None)
            module._info.pop('input_w', None)
            module._info['timestep'] = 0

    def reset(self):
        for module in self.modules:
            module.forward = module._original_forward
            module._info = {}


class UNetHook:
    @staticmethod
    def forward_pre(self, args):
        for module in self.modules():
            if hasattr(module, '_info'):
                module._info['input_h'] = args[0].shape[2]
                module._info['input_w'] = args[0].shape[3]
        return None


class BasicTransformerBlockHook:
    @staticmethod
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.FloatTensor:
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Self-Attention
        batch_size = hidden_states.shape[0]

        if self.norm_type == 'ada_norm':
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.norm_type == 'ada_norm_zero':
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        elif self.norm_type in ['layer_norm', 'layer_norm_i2vgen']:
            norm_hidden_states = self.norm1(hidden_states)
        elif self.norm_type == 'ada_norm_continuous':
            norm_hidden_states = self.norm1(hidden_states, added_cond_kwargs['pooled_text_emb'])
        elif self.norm_type == 'ada_norm_single':
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
            ).chunk(6, dim=1)
            norm_hidden_states = self.norm1(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
            norm_hidden_states = norm_hidden_states.squeeze(1)
        else:
            raise ValueError('Incorrect norm used')

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        # MSW-MSA
        info = self._info
        rand_num = torch.rand(1)
        B, N, C = hidden_states.shape
        input_h = info['input_h']
        input_w = info['input_w']
        downsample_ratio = int(((input_h * input_w) // N) ** 0.5)
        h, w = (input_h // downsample_ratio, input_w // downsample_ratio)
        widow_size = (h // 2, w // 2)
        if rand_num <= 0.25:
            shift_size = (0, 0)
        if rand_num > 0.25 and rand_num <= 0.5:
            shift_size = (widow_size[0] // 4, widow_size[1] // 4)
        if rand_num > 0.5 and rand_num <= 0.75:
            shift_size = (widow_size[0] // 4 * 2, widow_size[1] // 4 * 2)
        if rand_num > 0.75 and rand_num <= 1:
            shift_size = (widow_size[0] // 4 * 3, widow_size[1] // 4 * 3)
        norm_hidden_states = window_partition(norm_hidden_states, widow_size, shift_size, h, w)

        # 1. Prepare GLIGEN inputs
        cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        gligen_kwargs = cross_attention_kwargs.pop('gligen', None)

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )
        if self.norm_type == 'ada_norm_zero':
            attn_output = gate_msa.unsqueeze(1) * attn_output
        elif self.norm_type == 'ada_norm_single':
            attn_output = gate_msa * attn_output

        attn_output = window_reverse(attn_output, widow_size, h, w, shift_size)

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 1.2 GLIGEN Control
        if gligen_kwargs is not None:
            hidden_states = self.fuser(hidden_states, gligen_kwargs['objs'])

        # 3. Cross-Attention
        if self.attn2 is not None:
            if self.norm_type == 'ada_norm':
                norm_hidden_states = self.norm2(hidden_states, timestep)
            elif self.norm_type in ['ada_norm_zero', 'layer_norm', 'layer_norm_i2vgen']:
                norm_hidden_states = self.norm2(hidden_states)
            elif self.norm_type == 'ada_norm_single':
                # For PixArt norm2 isn't applied here:
                # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L70C1-L76C103
                norm_hidden_states = hidden_states
            elif self.norm_type == 'ada_norm_continuous':
                norm_hidden_states = self.norm2(hidden_states, added_cond_kwargs['pooled_text_emb'])
            else:
                raise ValueError('Incorrect norm')

            if self.pos_embed is not None and self.norm_type != 'ada_norm_single':
                norm_hidden_states = self.pos_embed(norm_hidden_states)

            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states

        # 4. Feed-forward
        # i2vgen doesn't have this norm 🤷‍♂️
        if self.norm_type == 'ada_norm_continuous':
            norm_hidden_states = self.norm3(hidden_states, added_cond_kwargs['pooled_text_emb'])
        elif not self.norm_type == 'ada_norm_single':
            norm_hidden_states = self.norm3(hidden_states)

        if self.norm_type == 'ada_norm_zero':
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        if self.norm_type == 'ada_norm_single':
            norm_hidden_states = self.norm2(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
        else:
            ff_output = self.ff(norm_hidden_states)

        if self.norm_type == 'ada_norm_zero':
            ff_output = gate_mlp.unsqueeze(1) * ff_output
        elif self.norm_type == 'ada_norm_single':
            ff_output = gate_mlp * ff_output

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states


class CrossAttnDownBlockSpatioTemporalHook:
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        image_only_indicator: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor, Tuple[torch.FloatTensor, ...]]:
        info = self._info
        max_timestep = info['max_timestep']
        t1_ratio = info['t1_ratio']
        t1 = int(max_timestep * t1_ratio)
        timestep = info['timestep']

        output_states = ()
        blocks = list(zip(self.resnets, self.attentions))
        for i, (resnet, attn) in enumerate(blocks):
            if self.training and self.gradient_checkpointing:  # TODO

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {'use_reentrant': False} if is_torch_version('>=', '1.11.0') else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet),
                    hidden_states,
                    temb,
                    image_only_indicator,
                    **ckpt_kwargs,
                )

                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                    return_dict=False,
                )[0]
            else:
                hidden_states = resnet(
                    hidden_states,
                    temb,
                    image_only_indicator=image_only_indicator,
                )
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                    return_dict=False,
                )[0]

            if i == 0:
                if timestep < t1:
                    hidden_states = F.avg_pool2d(hidden_states, kernel_size=(2, 2))

            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states = output_states + (hidden_states,)

        timestep += 1
        if timestep == max_timestep:
            timestep = 0
        info['timestep'] = timestep

        return hidden_states, output_states


class CrossAttnUpBlockSpatioTemporalHook:
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        res_hidden_states_tuple: Tuple[torch.FloatTensor, ...],
        temb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        image_only_indicator: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        info = self._info
        max_timestep = info['max_timestep']
        t1_ratio = info['t1_ratio']
        t1 = int(max_timestep * t1_ratio)
        timestep = info['timestep']

        for i, (resnet, attn) in enumerate(zip(self.resnets, self.attentions)):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if self.training and self.gradient_checkpointing:  # TODO

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {'use_reentrant': False} if is_torch_version('>=', '1.11.0') else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet),
                    hidden_states,
                    temb,
                    image_only_indicator,
                    **ckpt_kwargs,
                )
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                    return_dict=False,
                )[0]
            else:
                hidden_states = resnet(
                    hidden_states,
                    temb,
                    image_only_indicator=image_only_indicator,
                )
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                    return_dict=False,
                )[0]

            if i == 1:
                if timestep < t1:
                    re_size = (int(hidden_states.shape[-2] * 2), int(hidden_states.shape[-1] * 2))
                    hidden_states = F.interpolate(hidden_states, size=re_size, mode='bicubic')

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        timestep += 1
        if timestep == max_timestep:
            timestep = 0
        info['timestep'] = timestep

        return hidden_states


class Downsample2DConvHook:
    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        info = self._info
        max_timestep = info['max_timestep']
        t1_ratio = info['t1_ratio']
        t1 = int(max_timestep * t1_ratio)
        timestep = info['timestep']

        if timestep < t1:
            info['stride'] = self.stride
            info['padding'] = self.padding
            info['dilation'] = self.dilation
            self.stride = (4, 4)
            self.padding = (2, 2)
            self.dilation = (2, 2)

        hidden_states = F.conv2d(hidden_states, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        if timestep < t1:
            self.stride = info['stride']
            self.padding = info['padding']
            self.dilation = info['dilation']
        timestep += 1
        if timestep == max_timestep:
            timestep = 0
        info['timestep'] = timestep

        return hidden_states


class Upsample2DConvHook:
    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        info = self._info
        max_timestep = info['max_timestep']
        t1_ratio = info['t1_ratio']
        t1 = int(max_timestep * t1_ratio)
        timestep = info['timestep']

        if timestep < t1:
            hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode='bicubic')

        hidden_states = F.conv2d(hidden_states, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        timestep += 1
        if timestep == max_timestep:
            timestep = 0
        info['timestep'] = timestep

        return hidden_states


def window_partition(x, window_size, shift_size, H, W):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, N, C = x.shape
    # H, W = int(N**0.5), int(N**0.5)
    x = x.view(B, H, W, C)
    if type(shift_size) == list or type(shift_size) == tuple:
        if shift_size[0] > 0:
            x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
    else:
        if shift_size > 0:
            x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2))
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    windows = windows.view(-1, window_size[0] * window_size[1], C)
    return windows


def window_reverse(windows, window_size, H, W, shift_size):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B, N, C = windows.shape
    windows = windows.view(-1, window_size[0], window_size[1], C)
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    if type(shift_size) == list or type(shift_size) == tuple:
        if shift_size[0] > 0:
            x = torch.roll(x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))
    else:
        if shift_size > 0:
            x = torch.roll(x, shifts=(shift_size, shift_size), dims=(1, 2))
    x = x.view(B, H * W, C)
    return x
