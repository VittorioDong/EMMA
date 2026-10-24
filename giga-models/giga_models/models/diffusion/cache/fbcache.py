import torch.nn as nn


class FBCache(nn.Module):
    def __init__(
        self,
        model,
        thresh,
        num_steps=-1,
    ):
        super().__init__()
        model._original_forward_blocks = model.forward_blocks
        model.forward_blocks = self.forward
        self.model = model
        self.thresh = thresh
        self.num_steps = num_steps

        self.cur_step = 0
        self.prev_first_x_res = None
        self.x_res = None

    def reset(self):
        self.cur_step = 0
        self.prev_first_x_res = None
        self.x_res = None

    def clear(self):
        self.model.forward_blocks = self.model._original_forward_blocks
        self.reset()

    def forward(self, x, *args, x_ctrl=None, **kwargs):
        ori_x = x
        for i, (name, block) in enumerate(self.model.blocks.items()):
            x = block(x, *args, **kwargs)
            if x_ctrl is not None and name in x_ctrl:
                x = x + x_ctrl[name]
            if i == 0:
                first_x_res = x - ori_x
                use_cache = False
                if self.prev_first_x_res is not None and self.x_res is not None:
                    diff = calculate_similarity(self.prev_first_x_res, first_x_res)
                    if diff < self.thresh:
                        use_cache = True
                if use_cache:
                    x = self.x_res + x
                    self.cur_step += 1
                    if self.cur_step >= self.num_steps:
                        self.reset()
                    return x
                else:
                    self.prev_first_x_res = first_x_res
                    ori_x = x
        self.x_res = x - ori_x
        self.cur_step += 1
        if self.cur_step >= self.num_steps:
            self.reset()
        return x


def calculate_similarity(t1, t2):
    mean_diff = (t1 - t2).abs().mean()
    mean_t1 = t1.abs().mean()
    diff = mean_diff / mean_t1
    return diff.item()
