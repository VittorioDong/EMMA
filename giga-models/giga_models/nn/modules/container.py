from collections import OrderedDict
from typing import List

import torch.nn as nn


class ModuleList(nn.ModuleList):
    def forward(self, idx, *args, **kwargs):
        return self[idx](*args, **kwargs)


class ModuleDict(nn.ModuleDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.load_state_dict_mode = None

    @property
    def dtype(self):
        return list(self.values())[0].dtype

    def forward(self, key, *args, **kwargs):
        return self[key](*args, **kwargs)

    def load_state_dict(self, state_dict, strict: bool = True, **kwargs):
        if self.load_state_dict_mode == 'each':
            incompatible_keys = None

            model_names = list(self.keys())
            for model_name in model_names:
                keys = list(state_dict.keys())
                sub_model_name = model_name + '.'
                sub_state_dict = OrderedDict()
                for key in keys:
                    if key.startswith(sub_model_name):
                        sub_key = key[len(sub_model_name) :]
                        sub_state_dict[sub_key] = state_dict.pop(key)
                ret_keys = self[model_name].load_state_dict(sub_state_dict, strict=strict, **kwargs)
                if incompatible_keys is None:
                    incompatible_keys = ret_keys
                else:
                    incompatible_keys.missing_keys.extend(ret_keys.missing_keys)
                    incompatible_keys.unexpected_keys.extend(ret_keys.unexpected_keys)

            error_msgs: List[str] = []
            missing_keys = list(state_dict.keys())
            if len(missing_keys) > 0:
                incompatible_keys.missing_keys.extend(missing_keys)

            if strict:
                if len(incompatible_keys.unexpected_keys) > 0:
                    error_msgs.insert(
                        0,
                        'Unexpected key(s) in state_dict: {}. '.format(', '.join(f'"{k}"' for k in incompatible_keys.unexpected_keys)),
                    )
                if len(incompatible_keys.missing_keys) > 0:
                    error_msgs.insert(
                        0,
                        'Missing key(s) in state_dict: {}. '.format(', '.join(f'"{k}"' for k in incompatible_keys.missing_keys)),
                    )

            if len(error_msgs) > 0:
                raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(self.__class__.__name__, '\n\t'.join(error_msgs)))
            return incompatible_keys

        else:
            assert self.load_state_dict_mode is None
            return super().load_state_dict(state_dict, strict=strict, **kwargs)
