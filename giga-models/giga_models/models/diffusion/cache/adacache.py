from typing import Optional

import torch
import torch.nn as nn

from ...diffusion.dream_transfer.module.blocks import adaln_norm_state


class DreamTransferAdaCache(nn.Module):
    def __init__(
        self,
        model,
        cache_res: str = 'full_attn',
        cache_loc: list = [13],
        codebook: dict = {0.08: 6, 0.16: 5, 0.24: 4, 0.32: 3, 0.40: 2, 1.00: 1},
        num_steps: int = -1,
    ):
        super().__init__()
        self.cache_res = cache_res
        self.cache_loc = cache_loc
        self.codebook = codebook
        self.num_steps = num_steps

        model._original_forward_blocks = model.forward_blocks
        model.forward_blocks = self.forward_blocks

        self.block_stream = []
        for i, (name, block) in enumerate(model.blocks.items()):
            local_block_stream = []
            for j, local_block in enumerate(block.blocks):
                local_block._original_forward = local_block.forward
                local_block.forward = self.block_forward
                local_block.blk_id = i
                if i in cache_loc:
                    local_block.in_cache_loc = True
                else:
                    local_block.in_cache_loc = False
                local_block_stream.append(local_block)
            self.block_stream.append(local_block_stream)
        self.model = model

        self.reset()

    def reset(self):
        self.block_id = 0
        self.local_block_id = 0
        self.fwd_id = 0
        self.attn_cache_rate = 1
        self.prev_compute_step = []
        for i in range(len(self.block_stream)):
            steps = []
            for j in range(len(self.block_stream[i])):
                steps.append(0)
            self.prev_compute_step.append(steps)
        self.ada_dict = {
            'avg_diff': 0,
            'count': 0,
            'attn_cache_rate': self.attn_cache_rate,
            'new_attn_cache_rate': self.attn_cache_rate,
        }
        self.block_cache = None
        self.residual_cache = None

    def clear(self):
        self.model.forward_blocks = self.model._original_forward_blocks
        for i, (name, block) in enumerate(self.model.blocks.items()):
            for j, local_block in enumerate(block.blocks):
                local_block.forward = local_block._original_forward
        self.reset()

    def forward_blocks(self, x, *args, x_ctrl=None, **kwargs):
        self.ada_dict = {
            'avg_diff': 0,
            'count': 0,
            'attn_cache_rate': self.attn_cache_rate,
            'new_attn_cache_rate': self.attn_cache_rate,
        }
        ori_x = x
        attn_cache_rate = self.ada_dict['attn_cache_rate']
        local_prev_compute_step = self.prev_compute_step[self.block_id][self.local_block_id]
        self.attn_next_step = local_prev_compute_step + attn_cache_rate
        self.mlp_next_step = local_prev_compute_step + attn_cache_rate
        skip_cache_steps = [0, self.num_steps - 2, self.num_steps - 1]
        attn_active, mlp_active = False, False
        if (self.fwd_id == self.attn_next_step) or (self.fwd_id in skip_cache_steps):
            attn_active = True
        if (self.fwd_id == self.mlp_next_step) or (self.fwd_id in skip_cache_steps):
            mlp_active = True
        if (not attn_active) and (not mlp_active):
            x = x + self.residual_cache
            self.fwd_id += 1
            self.block_id = 0
            self.local_block_id = 0
            if self.fwd_id >= self.num_steps:
                self.reset()
            return x
        else:
            for i, (name, block) in enumerate(self.model.blocks.items()):
                x = block(x, *args, **kwargs)
                if x_ctrl is not None and name in x_ctrl:
                    x = x + x_ctrl[name]
            self.residual_cache = x - ori_x
            self.fwd_id += 1
            self.attn_cache_rate = self.ada_dict['new_attn_cache_rate']
            self.block_id = 0
            self.local_block_id = 0
            if self.fwd_id >= self.num_steps:
                self.reset()
            return x

    def block_forward(
        self,
        x: torch.Tensor,
        emb_B_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        crossattn_mask: Optional[torch.Tensor] = None,
        rope_emb_L_1_1_D: Optional[torch.Tensor] = None,
        adaln_lora_B_3D: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        block = self.block_stream[self.block_id]
        local_block = block[self.local_block_id]

        if local_block.use_adaln_lora:
            shift_B_D, scale_B_D, gate_B_D = (local_block.adaLN_modulation(emb_B_D) + adaln_lora_B_3D).chunk(local_block.n_adaln_chunks, dim=1)
        else:
            shift_B_D, scale_B_D, gate_B_D = local_block.adaLN_modulation(emb_B_D).chunk(local_block.n_adaln_chunks, dim=1)

        shift_1_1_1_B_D, scale_1_1_1_B_D, gate_1_1_1_B_D = (
            shift_B_D.unsqueeze(0).unsqueeze(0).unsqueeze(0),
            scale_B_D.unsqueeze(0).unsqueeze(0).unsqueeze(0),
            gate_B_D.unsqueeze(0).unsqueeze(0).unsqueeze(0),
        )

        attn_cache_rate = self.ada_dict['attn_cache_rate']
        local_prev_compute_step = self.prev_compute_step[self.block_id][self.local_block_id]
        self.attn_next_step = local_prev_compute_step + attn_cache_rate
        self.mlp_next_step = local_prev_compute_step + attn_cache_rate
        skip_cache_steps = [0, self.num_steps - 2, self.num_steps - 1]
        attn_active, mlp_active = False, False
        if (self.fwd_id == self.attn_next_step) or (self.fwd_id in skip_cache_steps):
            attn_active = True
        if (self.fwd_id == self.mlp_next_step) or (self.fwd_id in skip_cache_steps):
            mlp_active = True

        if local_block.block_type in ['mlp', 'ff']:
            if mlp_active:
                mlp_res = gate_1_1_1_B_D * local_block.block(
                    adaln_norm_state(local_block.norm_state, x, scale_1_1_1_B_D, shift_1_1_1_B_D),
                )
                x = x + mlp_res
                if local_block.in_cache_loc and (type(self.block_cache) is torch.Tensor) and self.cache_res in ['mlp', 'ff']:
                    cache_in = self.block_cache
                    res_in = mlp_res
                    attn_cache_rate = self.compute_next_step(
                        cache_in,
                        res_in,
                        self.codebook,
                        attn_cache_rate,
                        self.ada_dict,
                        1,
                    )
                    self.ada_dict['new_attn_cache_rate'] = attn_cache_rate
                if local_block.in_cache_loc and self.cache_res in ['mlp', 'ff']:
                    self.block_cache = mlp_res
                self.prev_compute_step[self.block_id][self.local_block_id] = self.fwd_id
            else:
                raise ValueError(f'Block {local_block.blk_id} is in cache location but not active for MLP at step {self.fwd_id}. ')

        elif local_block.block_type in ['full_attn', 'fa']:
            if attn_active:
                attn_res = gate_1_1_1_B_D * local_block.block(
                    adaln_norm_state(local_block.norm_state, x, scale_1_1_1_B_D, shift_1_1_1_B_D),
                    context=None,
                    rope_emb_L_1_1_D=rope_emb_L_1_1_D,
                )
                x = x + attn_res
                if local_block.in_cache_loc and (type(self.block_cache) is torch.Tensor) and self.cache_res in ['full_attn', 'fa']:
                    cache_in = self.block_cache
                    res_in = attn_res
                    attn_cache_rate = self.compute_next_step(
                        cache_in,
                        res_in,
                        self.codebook,
                        attn_cache_rate,
                        self.ada_dict,
                        1,
                    )
                    self.ada_dict['new_attn_cache_rate'] = attn_cache_rate
                if local_block.in_cache_loc and self.cache_res in ['full_attn', 'fa']:
                    self.block_cache = attn_res
                self.prev_compute_step[self.block_id][self.local_block_id] = self.fwd_id
            else:
                raise ValueError(f'Block {local_block.blk_id} is in cache location but not active for Full Attention at step {self.fwd_id}. ')

        elif local_block.block_type in ['cross_attn', 'ca']:
            if attn_active:
                attn_res = gate_1_1_1_B_D * local_block.block(
                    adaln_norm_state(local_block.norm_state, x, scale_1_1_1_B_D, shift_1_1_1_B_D),
                    context=crossattn_emb,
                    crossattn_mask=crossattn_mask,
                    rope_emb_L_1_1_D=rope_emb_L_1_1_D,
                )
                x = x + attn_res
                if local_block.in_cache_loc and (type(self.block_cache) is torch.Tensor) and self.cache_res in ['cross_attn', 'ca']:
                    cache_in = self.block_cache
                    res_in = attn_res
                    attn_cache_rate = self.compute_next_step(
                        cache_in,
                        res_in,
                        self.codebook,
                        attn_cache_rate,
                        self.ada_dict,
                        1,
                    )
                    self.ada_dict['new_attn_cache_rate'] = attn_cache_rate
                if local_block.in_cache_loc and self.cache_res in ['cross_attn', 'ca']:
                    self.block_cache = attn_res
                self.prev_compute_step[self.block_id][self.local_block_id] = self.fwd_id
            else:
                raise ValueError(f'Block {local_block.blk_id} is in cache location but not active for Cross Attention at step {self.fwd_id}. ')

        else:
            raise ValueError(f'Unknown block type: {local_block.block_type}')

        if self.local_block_id == len(block) - 1:
            self.local_block_id = 0
            self.block_id += 1
        else:
            self.local_block_id += 1

        return x

    def compute_next_step(self, cache, res, codebook, prev_rate, ada_dict, norm_ord):
        # distance metric [synced across layers if needed]
        cache_diff = (cache - res).norm(dim=(0, 1, 2, 3, 4), p=norm_ord) / cache.norm(dim=(0, 1, 2, 3, 4), p=norm_ord)
        # normalize across steps
        cache_diff = cache_diff / prev_rate
        # sync across layers
        cache_diff = (cache_diff + ada_dict['count'] * ada_dict['avg_diff']) / (ada_dict['count'] + 1)
        ada_dict['avg_diff'], ada_dict['count'] = cache_diff, ada_dict['count'] + 1
        cache_diff = cache_diff

        metric_thres, cache_rates = list(codebook.keys()), list(codebook.values())
        if cache_diff < metric_thres[0]:
            new_rate = cache_rates[0]
        elif cache_diff < metric_thres[1]:
            new_rate = cache_rates[1]
        elif cache_diff < metric_thres[2]:
            new_rate = cache_rates[2]
        elif cache_diff < metric_thres[3]:
            new_rate = cache_rates[3]
        elif cache_diff < metric_thres[4]:
            new_rate = cache_rates[4]
        else:
            new_rate = cache_rates[-1]

        return new_rate
