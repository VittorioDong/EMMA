import functools
import math

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention_processor import IPAdapterAttnProcessor2_0
from diffusers.models.embeddings import ImageProjection, IPAdapterFullImageProjection, IPAdapterPlusImageProjection
from diffusers.models.lora import LoRALinearLayer
from diffusers.models.modeling_utils import ModelMixin


class IPAdapterModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, processors_cfg, image_proj_cfg, lora_layers_cfg=None):
        super().__init__()
        self.processor_names = []
        self.processors = nn.ModuleList()
        for processor_name, processor_cfg in processors_cfg:
            processor = IPAdapterAttnProcessor2_0(**processor_cfg)
            assert processor_name not in self.processor_names
            self.processor_names.append(processor_name)
            self.processors.append(processor)
        image_proj_cfg = image_proj_cfg.copy()
        image_proj_type = image_proj_cfg.pop('type')
        if image_proj_type == 'ImageProjection':
            image_proj = ImageProjection(**image_proj_cfg)
        elif image_proj_type == 'IPAdapterFullImageProjection':
            image_proj = IPAdapterFullImageProjection(**image_proj_cfg)
        elif image_proj_type == 'IPAdapterPlusImageProjection':
            image_proj = IPAdapterPlusImageProjection(**image_proj_cfg)
        elif image_proj_type == 'FaceProjection':
            image_proj = FaceProjection(**image_proj_cfg)
        elif image_proj_type == 'FacePlusProjection':
            image_proj = FacePlusProjection(**image_proj_cfg)
        else:
            assert False
        self.image_proj = image_proj
        self.lora_layer_names = []
        self.lora_layers = nn.ModuleList()
        if lora_layers_cfg is not None:
            for lora_layer_name, lora_layer_cfg in lora_layers_cfg:
                lora_layer = nn.ModuleDict()
                for l_name, l_cfg in lora_layer_cfg.items():
                    l_cfg = l_cfg.copy()
                    l_type = l_cfg.pop('type')
                    if l_type == 'LoRALinearLayer':
                        lora_layer[l_name] = LoRALinearLayer(**l_cfg)
                    else:
                        assert False
                assert lora_layer_name not in self.lora_layer_names
                self.lora_layer_names.append(lora_layer_name)
                self.lora_layers.append(lora_layer)

    @classmethod
    def from_unet(cls, unet, num_tokens, image_proj_cfg, with_lora=False, lora_rank=4):
        processors_cfg = []
        lora_layers_cfg = []
        conditions_attn = ['.attentions.']
        conditions_attn2 = ['.attentions.', '.attn2']
        for name, module in unet.named_modules():
            if module.__class__.__name__ == 'Attention':
                if meet_condition(name, conditions_attn2):
                    to_k_shape = module.to_k.weight.shape
                    processor_cfg = dict(
                        hidden_size=to_k_shape[0],
                        cross_attention_dim=to_k_shape[1],
                        num_tokens=num_tokens,
                    )
                    processors_cfg.append((name, processor_cfg))
                if with_lora and meet_condition(name, conditions_attn):
                    name = 'lora.' + name
                    lora_layer_cfg = dict(
                        to_q=dict(
                            type='LoRALinearLayer',
                            in_features=module.to_q.in_features,
                            out_features=module.to_q.out_features,
                            rank=lora_rank,
                        ),
                        to_k=dict(
                            type='LoRALinearLayer',
                            in_features=module.to_k.in_features,
                            out_features=module.to_k.out_features,
                            rank=lora_rank,
                        ),
                        to_v=dict(
                            type='LoRALinearLayer',
                            in_features=module.to_v.in_features,
                            out_features=module.to_v.out_features,
                            rank=lora_rank,
                        ),
                        to_out=dict(
                            type='LoRALinearLayer',
                            in_features=module.to_out[0].in_features,
                            out_features=module.to_out[0].out_features,
                            rank=lora_rank,
                        ),
                    )
                    lora_layers_cfg.append((name, lora_layer_cfg))
        ip_adapter = cls(
            processors_cfg=processors_cfg,
            image_proj_cfg=image_proj_cfg,
            lora_layers_cfg=lora_layers_cfg if len(lora_layers_cfg) > 0 else None,
        )
        return ip_adapter

    def _fuse_lora(self, module, key, key2, scale=1.0, *args, **kwargs):
        idx = self.lora_layer_names.index(key)
        lora_layer = self.lora_layers[idx][key2]
        if scale != 1.0:
            lora_layer.network_alpha = scale * lora_layer.rank
        module.lora_layer = lora_layer
        module._fuse_lora(*args, **kwargs)

    def to_unet(self, unet, scale=1.0, is_train=True):
        for name, module in unet.named_modules():
            if module.__class__.__name__ == 'Attention':
                if name in self.processor_names:
                    module.processor = functools.partial(self, name)
                name = 'lora.' + name
                if name in self.lora_layer_names:
                    if is_train:
                        assert scale == 1.0
                        module.to_q.lora_layer = functools.partial(self, name, 'to_q')
                        module.to_k.lora_layer = functools.partial(self, name, 'to_k')
                        module.to_v.lora_layer = functools.partial(self, name, 'to_v')
                        module.to_out[0].lora_layer = functools.partial(self, name, 'to_out')
                    else:
                        self._fuse_lora(module.to_q, name, 'to_q', scale)
                        self._fuse_lora(module.to_k, name, 'to_k', scale)
                        self._fuse_lora(module.to_v, name, 'to_v', scale)
                        self._fuse_lora(module.to_out[0], name, 'to_out', scale)

    def set_scale(self, scale):
        for processor in self.processors:
            processor.scale = scale

    def forward(self, key, *args, **kwargs):
        if key == 'image_proj':
            module = self.image_proj
        elif key in self.processor_names:
            idx = self.processor_names.index(key)
            module = self.processors[idx]
        elif key in self.lora_layer_names:
            idx = self.lora_layer_names.index(key)
            key2, args = args[0], args[1:]
            module = self.lora_layers[idx][key2]
        else:
            assert False
        return module(*args, **kwargs)


class FaceProjection(torch.nn.Module):
    def __init__(self, cross_attention_dim=768, id_embeddings_dim=512, num_tokens=4):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(id_embeddings_dim, id_embeddings_dim * 2),
            torch.nn.GELU(),
            torch.nn.Linear(id_embeddings_dim * 2, cross_attention_dim * num_tokens),
        )
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, id_embeds):
        x = self.proj(id_embeds)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        x = self.norm(x)
        return x


class FacePlusProjection(torch.nn.Module):
    def __init__(
        self,
        cross_attention_dim=768,
        id_embeddings_dim=512,
        clip_embeddings_dim=1280,
        num_tokens=4,
        shortcut=False,
    ):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(id_embeddings_dim, id_embeddings_dim * 2),
            torch.nn.GELU(),
            torch.nn.Linear(id_embeddings_dim * 2, cross_attention_dim * num_tokens),
        )
        self.norm = torch.nn.LayerNorm(cross_attention_dim)
        self.perceiver_resampler = FacePerceiverResampler(
            dim=cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=cross_attention_dim // 64,
            embedding_dim=clip_embeddings_dim,
            output_dim=cross_attention_dim,
            ff_mult=4,
        )
        self.shortcut = shortcut

    def forward(self, id_embeds, clip_embeds, scale=1.0):
        x = self.proj(id_embeds)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        x = self.norm(x)
        out = self.perceiver_resampler(x, clip_embeds)
        if self.shortcut:
            out = x + scale * out
        return out


class FacePerceiverResampler(torch.nn.Module):
    def __init__(
        self,
        *,
        dim=768,
        depth=4,
        dim_head=64,
        heads=16,
        embedding_dim=1280,
        output_dim=768,
        ff_mult=4,
    ):
        super().__init__()

        self.proj_in = torch.nn.Linear(embedding_dim, dim)
        self.proj_out = torch.nn.Linear(dim, output_dim)
        self.norm_out = torch.nn.LayerNorm(output_dim)
        self.layers = torch.nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                torch.nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def forward(self, latents, x):
        x = self.proj_in(x)
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
        latents = self.proj_out(latents)
        return self.norm_out(latents)


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            latent (torch.Tensor): latent features
                shape (b, n2, D)
        """
        x = self.norm1(x)
        latents = self.norm2(latents)

        b, l, _ = latents.shape

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1)  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v

        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)

        return self.to_out(out)


def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


def reshape_tensor(x, heads):
    bs, length, width = x.shape
    # (bs, length, width) --> (bs, length, n_heads, dim_per_head)
    x = x.view(bs, length, heads, -1)
    # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
    x = x.transpose(1, 2)
    # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)
    x = x.reshape(bs, heads, length, -1)
    return x


def meet_condition(name, conditions):
    if conditions is None:
        return True
    for condition in conditions:
        if condition not in name:
            return False
    return True
