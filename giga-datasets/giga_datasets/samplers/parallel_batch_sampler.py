import torch


class ParallelBatchSampler(torch.utils.data.Sampler):
    def __init__(self, batch_sampler, sp_size):
        self.batch_sampler = batch_sampler
        self.sp_size = sp_size
        self.batch_size = getattr(batch_sampler, 'batch_size', None)
        self.drop_last = getattr(batch_sampler, 'drop_last', False)

    def set_epoch(self, epoch):
        if hasattr(self.batch_sampler, 'set_epoch'):
            self.batch_sampler.set_epoch(epoch)

    def __len__(self):
        return len(self.batch_sampler) * self.sp_size

    def __iter__(self):
        for batch in self.batch_sampler:
            for i in range(self.sp_size):
                yield batch
