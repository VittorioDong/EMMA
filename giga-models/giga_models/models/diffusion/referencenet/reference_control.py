from types import MethodType
from typing import Any, Dict, Optional

import torch
from diffusers.models.attention import _chunked_feed_forward


class ReferenceControl:
    def __init__(
        self,
        unet,
        mode,
        style_fidelity=0.5,
        do_classifier_free_guidance=False,
    ):
        self.unet = unet
        self.mode = mode
        self.style_fidelity = style_fidelity
        self.do_classifier_free_guidance = do_classifier_free_guidance
        self.modules = []
        self.register_hooks()

    def register_hooks(self):
        for name, module in self.unet.named_modules():
            if module.__class__.__name__ == 'BasicTransformerBlock' and '.attentions.' in name:
                module._original_forward = module.forward
                module.forward = MethodType(BasicTransformerBlockHook.forward, module)
                module.mode = self.mode
                module.do_classifier_free_guidance = self.do_classifier_free_guidance
                module.style_fidelity = self.style_fidelity
                module.bank = []
                self.modules.append(module)

    def set_mode(self, mode):
        for module in self.modules:
            module.mode = mode

    def update(self, writer):
        assert len(self.modules) == len(writer.modules)
        for i in range(len(self.modules)):
            self.modules[i].bank = writer.modules[i].bank

    def clear(self):
        for module in self.modules:
            module.bank.clear()

    def reset(self):
        for module in self.modules:
            module.forward = module._original_forward
            module.bank.clear()


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
    ) -> torch.FloatTensor:
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Self-Attention
        batch_size = hidden_states.shape[0]

        if self.use_ada_layer_norm:
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.use_ada_layer_norm_zero:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        elif self.use_layer_norm:
            norm_hidden_states = self.norm1(hidden_states)
        elif self.use_ada_layer_norm_single:
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

        # 1. Retrieve lora scale.
        lora_scale = cross_attention_kwargs.get('scale', 1.0) if cross_attention_kwargs is not None else 1.0

        # 2. Prepare GLIGEN inputs
        cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        gligen_kwargs = cross_attention_kwargs.pop('gligen', None)

        if self.only_cross_attention:
            attn_output = self.attn1(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )
        else:
            if self.mode == 'write':
                self.bank.append(norm_hidden_states.clone())
                if self.attn1 is not None:
                    attn_output = self.attn1(
                        norm_hidden_states,
                        attention_mask=attention_mask,
                        **cross_attention_kwargs,
                    )
                else:
                    attn_output = norm_hidden_states
            elif self.mode == 'read':
                prev_norm_hidden_states = self.bank[0]
                self.bank = self.bank[1:]
                new_norm_hidden_states = torch.cat([norm_hidden_states, prev_norm_hidden_states], dim=1)
                attn_output = self.attn1(
                    norm_hidden_states,
                    encoder_hidden_states=new_norm_hidden_states,
                    attention_mask=attention_mask,
                    **cross_attention_kwargs,
                )
                if self.do_classifier_free_guidance and self.style_fidelity > 0:
                    attn_output_uc = attn_output.clone()
                    half_size = attn_output_uc.shape[0] // 2
                    attn_output_uc[:half_size] = self.attn1(
                        norm_hidden_states[:half_size],
                        attention_mask=attention_mask,
                        **cross_attention_kwargs,
                    )
                    attn_output = self.style_fidelity * attn_output_uc + (1.0 - self.style_fidelity) * attn_output

        if self.use_ada_layer_norm_zero:
            attn_output = gate_msa.unsqueeze(1) * attn_output
        elif self.use_ada_layer_norm_single:
            attn_output = gate_msa * attn_output

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 2.5 GLIGEN Control
        if gligen_kwargs is not None:
            hidden_states = self.fuser(hidden_states, gligen_kwargs['objs'])

        # 3. Cross-Attention
        if self.attn2 is not None:
            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm2(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero or self.use_layer_norm:
                norm_hidden_states = self.norm2(hidden_states)
            elif self.use_ada_layer_norm_single:
                # For PixArt norm2 isn't applied here:
                # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L70C1-L76C103
                norm_hidden_states = hidden_states
            else:
                raise ValueError('Incorrect norm')

            if self.pos_embed is not None and self.use_ada_layer_norm_single is False:
                norm_hidden_states = self.pos_embed(norm_hidden_states)

            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states

        # 4. Feed-forward
        if not self.use_ada_layer_norm_single:
            norm_hidden_states = self.norm3(hidden_states)

        if self.use_ada_layer_norm_zero:
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        if self.use_ada_layer_norm_single:
            norm_hidden_states = self.norm2(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size, lora_scale=lora_scale)
        else:
            ff_output = self.ff(norm_hidden_states, scale=lora_scale)

        if self.use_ada_layer_norm_zero:
            ff_output = gate_mlp.unsqueeze(1) * ff_output
        elif self.use_ada_layer_norm_single:
            ff_output = gate_mlp * ff_output

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states
