import math
import random

import numpy as np
import torch

from ..datasets import ConcatDataset, Dataset


class BucketBatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size_per_gpu, batch_size, shuffle=True, infinite=True, seed=6666):
        self.num_gpus = batch_size // batch_size_per_gpu
        self.shuffle = shuffle
        self.infinite = infinite
        self.seed = seed
        self.epoch = 0
        self.all_indices_dict = _get_indices_dict(dataset)
        self.batch_sizes = list(self.all_indices_dict.keys())
        self.total_size = 0
        self.total_size_dict = dict()
        for batch_size_per_gpu, indices_list in self.all_indices_dict.items():
            if batch_size_per_gpu not in self.total_size_dict:
                self.total_size_dict[batch_size_per_gpu] = []
            for indices in indices_list:
                data_size = len(indices)
                batch_size = batch_size_per_gpu * self.num_gpus
                total_size = int(math.ceil(data_size / batch_size)) * batch_size
                self.total_size += total_size // batch_size_per_gpu
                self.total_size_dict[batch_size_per_gpu].append(total_size)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.total_size

    def __iter__(self):
        while True:
            np.random.seed(self.seed + self.epoch)
            self.epoch += 1
            all_indices_list = []
            for batch_size_per_gpu, indices_list in self.all_indices_dict.items():
                batch_size = batch_size_per_gpu * self.num_gpus
                for i in range(len(indices_list)):
                    indices = np.zeros((0,), dtype=np.int64)
                    total_size = self.total_size_dict[batch_size_per_gpu][i]
                    while len(indices) < total_size:
                        indices_i = indices_list[i]
                        if self.shuffle:
                            indices_i = np.random.permutation(indices_i)
                        num_data = min(len(indices_i), total_size - len(indices))
                        indices = np.hstack((indices, indices_i[:num_data]))
                    indices = indices.reshape((-1, batch_size))
                    indices = indices.tolist()
                    all_indices_list.extend(indices)
            assert len(all_indices_list) * self.num_gpus == self.total_size
            if self.shuffle:
                random.shuffle(all_indices_list)
            for indices_list in all_indices_list:
                assert len(indices_list) % self.num_gpus == 0
                batch_size_per_gpu = len(indices_list) // self.num_gpus
                for i in range(self.num_gpus):
                    start = i * batch_size_per_gpu
                    end = (i + 1) * batch_size_per_gpu
                    indices = indices_list[start:end]
                    yield indices
            if not self.infinite:
                break


def _process_dataset(dataset):
    if isinstance(dataset, ConcatDataset):
        return [d.datasets[0] for d in dataset.datasets]
    elif isinstance(dataset, Dataset):
        return [dataset.datasets[0]]
    else:
        assert False


def _get_indices_dict(dataset):
    dataset_list = _process_dataset(dataset)
    all_indices_dict = dict()
    count = 0
    for dataset in dataset_list:
        for i in range(len(dataset)):
            data_dict = dataset[i]
            batch_size_per_gpu, bucket_index = data_dict['bucket_info']
            if batch_size_per_gpu not in all_indices_dict:
                all_indices_dict[batch_size_per_gpu] = dict()
            indices_dict = all_indices_dict[batch_size_per_gpu]
            if bucket_index not in indices_dict:
                indices_dict[bucket_index] = []
            indices_dict[bucket_index].append(count)
            count += 1
    new_all_indices_dict = dict()
    for batch_size_per_gpu, indices_dict in all_indices_dict.items():
        indices_list = []
        for indices in indices_dict.values():
            indices = np.array(indices, dtype=np.int64)
            indices_list.append(indices)
        new_all_indices_dict[batch_size_per_gpu] = indices_list
    return new_all_indices_dict
