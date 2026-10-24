from types import MethodType

import accelerate
import torch
import torch.nn as nn
import transformer_engine.common.recipe as te_recipe
import transformer_engine.pytorch as te
from accelerate.utils.transformer_engine import contextual_fp8_autocast


@torch.no_grad()
def convert_model_torch_to_te(model, prefix='', ignore_modules=None):
    """Recursively converts torch layers of a model to their
    `transformers_engine` counterpart."""
    for name, module in model.named_children():
        submodule_prefix = prefix + ('.' if prefix else '') + name
        if _need_ignore(submodule_prefix, module, ignore_modules):
            continue
        if isinstance(module, nn.Linear):
            if any(p % 16 != 0 for p in module.weight.shape):
                continue
            has_bias = module.bias is not None
            te_module = te.Linear(module.in_features, module.out_features, bias=has_bias, params_dtype=module.weight.dtype)
            te_module.weight.copy_(module.weight)
            te_module.weight.requires_grad = module.weight.requires_grad
            if has_bias:
                te_module.bias.copy_(module.bias)
                te_module.bias.requires_grad = module.bias.requires_grad
            setattr(model, name, te_module)

        elif isinstance(module, nn.LayerNorm):
            if module.weight is None:
                continue
            te_module = te.LayerNorm(module.normalized_shape, eps=module.eps, params_dtype=module.weight.dtype)
            te_module.weight.copy_(module.weight)
            te_module.weight.requires_grad = module.weight.requires_grad
            if module.bias is not None:
                te_module.bias.copy_(module.bias)
                te_module.bias.requires_grad = module.bias.requires_grad
            setattr(model, name, te_module)

        else:
            convert_model_torch_to_te(module, submodule_prefix, ignore_modules)
    return model


def apply_fp8_autowrap(model, **recipe_kwargs):
    use_during_eval = recipe_kwargs.pop('use_autocast_during_eval', False)
    fp8_recipe = te_recipe.DelayedScaling(**recipe_kwargs)
    new_forward = contextual_fp8_autocast(model.forward, fp8_recipe, use_during_eval)

    if hasattr(model.forward, '__func__'):
        model.forward = MethodType(new_forward, model)
    else:
        model.forward = new_forward

    return model


def reset_accelerate_te(**recipe_kwargs):
    def _convert_model(model, *args, **kwargs):
        pass

    setattr(accelerate.accelerator, 'convert_model', _convert_model)
    setattr(accelerate.utils, 'convert_model', _convert_model)
    setattr(accelerate.utils.other, 'convert_model', _convert_model)
    setattr(accelerate.utils.transformer_engine, 'convert_model', _convert_model)

    def _apply_fp8_autowrap(model, *args, **kwargs):
        return apply_fp8_autowrap(model, **recipe_kwargs)

    setattr(accelerate.accelerator, 'apply_fp8_autowrap', _apply_fp8_autowrap)
    setattr(accelerate.utils, 'apply_fp8_autowrap', _apply_fp8_autowrap)
    setattr(accelerate.utils.transformer_engine, 'apply_fp8_autowrap', _apply_fp8_autowrap)


def _need_ignore(prefix, module, ignore_modules):
    if ignore_modules is None:
        return False
    for ignore_module in ignore_modules:
        if isinstance(ignore_module, str):
            if ignore_module in prefix:
                return True
            if module.__class__.__name__ == ignore_module:
                return True
        elif isinstance(module, ignore_module):
            return True
    return False
