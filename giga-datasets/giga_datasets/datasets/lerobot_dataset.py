import os

# from lerobot.datasets.lerobot_dataset import LeRobotDataset as _LeRobotDataset
# from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata as _LeRobotDatasetMetadata

from .base_dataset import BaseDataset


class LeRobotDataset(BaseDataset):
    def __init__(self, data_path, data_size=None, delta_info=None, meta_name=None, **kwargs):
        super(LeRobotDataset, self).__init__(data_path=data_path)
        self.data_size = data_size
        self.delta_info = delta_info
        self.meta_name = meta_name
        self.kwargs = kwargs
        self.dataset = None

    @classmethod
    def load(cls, data_or_config):
        from .dataset import load_config

        config = load_config(data_or_config)
        keys = list(config.keys())
        for key in keys:
            if key.startswith('_'):
                config.pop(key)
        return cls(**config)

    def open(self):
        if self.dataset is None:
            repo_id = os.path.basename(self.data_path)
            dataset_meta = _LeRobotDatasetMetadata(repo_id, root=self.data_path)
            if self.delta_info is not None:
                delta_timestamps = dict()
                for delta_name, delta_size in self.delta_info.items():
                    delta_timestamps[delta_name] = [i / dataset_meta.fps for i in range(delta_size)]
            else:
                delta_timestamps = None
            self.dataset = _LeRobotDataset(
                repo_id,
                root=self.data_path,
                delta_timestamps=delta_timestamps,
                **self.kwargs,
            )
            if self.data_size is not None:
                assert self.data_size == len(self.dataset)
            else:
                self.data_size = len(self.dataset)

    def close(self):
        if self.dataset is not None:
            self.dataset = None
        super(LeRobotDataset, self).close()

    def __len__(self):
        if self.data_size is None:
            self.open()
        return self.data_size

    def _get_data(self, index):
        data_dict = self.dataset[index]
        if self.meta_name is not None:
            assert self.meta_name not in data_dict
            data_dict[self.meta_name] = self.dataset.meta
        return data_dict
