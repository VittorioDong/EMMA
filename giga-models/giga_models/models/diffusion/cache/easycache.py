import torch.nn as nn


class EasyCache(nn.Module):
    def __init__(
        self,
        model,
        thresh,
        ret_steps,
        num_steps=-1,
    ):
        super().__init__()
        model._original_forward = model.forward
        model.forward = self.forward

        self.model = model
        self.thresh = thresh
        self.ret_steps = ret_steps
        self.num_steps = num_steps

        self.cur_step = 0
        self.accumulated_error = 0
        self.k = None
        self.prev_raw_input = None
        self.prev_prev_raw_input = None
        self.prev_output = None
        self.cache = None

    def reset(self):
        self.cur_step = 0
        self.accumulated_error = 0
        self.k = None
        self.prev_raw_input = None
        self.prev_prev_raw_input = None
        self.prev_output = None
        self.cache = None

    def clear(self):
        self.model.forward = self.model._original_forward
        self.reset()

    def forward(self, x, *args, **kwargs):
        raw_input = x.clone()

        if self.cur_step < self.ret_steps or self.cur_step >= self.num_steps - 1:
            should_calc = True
            self.accumulated_error = 0
        else:
            # Check if previous inputs and outputs exist
            if self.prev_raw_input is not None and self.prev_output is not None and self.k is not None:
                if self.prev_raw_input.shape[1] != raw_input.shape[1]:
                    raw_input = raw_input[:, : self.prev_raw_input.shape[1]]
                raw_input_change = (raw_input[-1:] - self.prev_raw_input).abs().mean()
                output_norm = self.prev_output.abs().mean()
                pred_change = self.k * (raw_input_change / output_norm)
                self.accumulated_error += pred_change
                if self.accumulated_error < self.thresh:
                    should_calc = False
                else:
                    should_calc = True
                    self.accumulated_error = 0
            else:
                should_calc = True
                self.accumulated_error = 0

        self.prev_raw_input = raw_input[-1:].clone()

        if not should_calc and self.cache is not None:
            if self.cache.shape[1] != raw_input.shape[1]:
                raw_input = raw_input[:, : self.cache.shape[1]]
            output = raw_input + self.cache
            self.cur_step += 1
            if self.cur_step >= self.num_steps:
                self.reset()
            return output

        output = self.model._original_forward(x, *args, **kwargs)

        if output.shape[1] != raw_input.shape[1]:
            raw_input = raw_input[:, : output.shape[1]]
            self.prev_raw_input = self.prev_raw_input[:, : output.shape[1]]

        # store the cache for next step
        self.cache = output - raw_input
        if self.prev_output is not None and self.prev_prev_raw_input is not None:
            output_change = (output[-1:] - self.prev_output).abs().mean()
            input_change = (self.prev_raw_input - self.prev_prev_raw_input).abs().mean()
            self.k = output_change / input_change

        # update the previous state
        self.prev_prev_raw_input = self.prev_raw_input
        self.prev_output = output[-1:].clone()

        self.cur_step += 1
        if self.cur_step >= self.num_steps:
            self.reset()

        return output
