import os
import shutil

import imageio
from decord import VideoReader
from PIL import Image

from .. import utils
from .base_dataset import BaseDataset


class FileDataset(BaseDataset):
    def __init__(self, data_size, data_type, data_name=None, **kwargs):
        super(FileDataset, self).__init__(**kwargs)
        self.data_size = data_size
        self.data_type = data_type
        self.data_name = data_name
        self.data_dict = dict()

    @classmethod
    def load(cls, data_or_config):
        from .dataset import load_config

        config = load_config(data_or_config)
        config_path = config.get('config_path', None)
        data_path = config.get('data_path', None)
        data_size = config['data_size']
        data_type = config['data_type']
        data_name = config['data_name']
        return cls(config_path=config_path, data_path=data_path, data_size=data_size, data_type=data_type, data_name=data_name)

    def open(self):
        if len(self.data_dict) == 0:
            data_path = os.path.join(self.data_path, 'data.pkl')
            if os.path.exists(data_path):
                self.data_dict = utils.load_file(data_path)
            else:
                data_dir = os.path.join(self.data_path, 'data')
                data_names = os.listdir(data_dir)
                for data_name in data_names:
                    data_index = data_name.split('.')[0]
                    assert data_index not in self.data_dict
                    self.data_dict[data_index] = os.path.join(data_dir, data_name)
            if self.data_size is not None:
                assert self.data_size == len(self.data_dict)
            else:
                self.data_size = len(self.data_dict)

    def close(self):
        if self.data_dict is not None:
            self.data_dict.clear()
            self.data_dict = None
        super(FileDataset, self).close()

    def __len__(self):
        if self.data_size is None:
            self.open()
        return self.data_size

    def _get_data(self, index):
        data = self.data_dict[str(index)]
        if self.data_type == 'image':
            data = Image.open(data)
        elif self.data_type == 'video':
            data = VideoReader(data)
        elif self.data_type == 'dict':
            data = utils.load_file(data)
        else:
            assert self.data_type == 'raw'
        if self.data_name is not None:
            data_dict = {self.data_name: data}
        else:
            data_dict = data
        return data_dict


class FileWriter:
    def __init__(self, data_path, rmtree=True):
        if os.path.exists(data_path) and rmtree:
            shutil.rmtree(data_path)
        self.save_dir = os.path.join(data_path, 'data')
        self.data_path = data_path
        self.data_type = None
        self.data_dict = dict()
        self.key_names = []
        self._count = 0
        os.makedirs(self.save_dir, exist_ok=True)

    def close(self):
        if len(self.data_dict) > 0:
            data_path = os.path.join(self.data_path, 'data.pkl')
            utils.save_file(data_path, self.data_dict)
            self.data_dict = dict()
            if len(os.listdir(self.save_dir)) == 0:
                shutil.rmtree(self.save_dir)
        self.data_type = None
        self.key_names = []
        self._count = 0

    def write_image(self, index, image, ext=None):
        if self.data_type is None:
            self.data_type = 'image'
        else:
            assert self.data_type == 'image'
        if isinstance(image, str):
            if ext is None:
                ext = image.split('.')[-1]
            save_path = os.path.join(self.save_dir, f'{index}.{ext}')
            shutil.copyfile(image, save_path)
        elif isinstance(image, bytes):
            if ext is None:
                ext = 'png'
            save_path = os.path.join(self.save_dir, f'{index}.{ext}')
            open(save_path, 'wb').write(image)
        elif isinstance(image, Image.Image):
            if ext is None:
                ext = 'png'
            save_path = os.path.join(self.save_dir, f'{index}.{ext}')
            image.save(save_path)
        else:
            assert False
        self._count += 1

    def write_video(self, index, video, ext=None, fps=None):
        if self.data_type is None:
            self.data_type = 'video'
        else:
            assert self.data_type == 'video'
        if isinstance(video, str):
            if ext is None:
                ext = video.split('.')[-1]
            save_path = os.path.join(self.save_dir, f'{index}.{ext}')
            shutil.copyfile(video, save_path)
        elif isinstance(video, bytes):
            if ext is None:
                ext = 'mp4'
            save_path = os.path.join(self.save_dir, f'{index}.{ext}')
            open(save_path, 'wb').write(video)
        elif isinstance(video, list):
            assert len(video) > 0
            for i in range(len(video)):
                assert isinstance(video[i], Image.Image)
            if ext is None:
                ext = 'mp4'
            assert fps is not None
            save_path = os.path.join(self.save_dir, f'{index}.{ext}')
            imageio.mimsave(save_path, video, fps=fps)
        else:
            assert False
        self._count += 1

    def write_dict(self, index, data):
        if self.data_type is None:
            self.data_type = 'dict'
        else:
            assert self.data_type == 'dict'
        assert isinstance(data, dict)
        self.key_names = list(set(self.key_names + list(data.keys())))
        save_path = os.path.join(self.save_dir, f'{index}.pkl')
        utils.save_file(save_path, data)
        self._count += 1

    def write_image_path(self, index, image_path):
        if self.data_type is None:
            self.data_type = 'image_path'
        else:
            assert self.data_type == 'image_path'
        index = str(index)
        assert index not in self.data_dict
        self.data_dict[index] = image_path
        self._count += 1

    def write_video_path(self, index, video_path):
        if self.data_type is None:
            self.data_type = 'video_path'
        else:
            assert self.data_type == 'video_path'
        index = str(index)
        assert index not in self.data_dict
        self.data_dict[index] = video_path
        self._count += 1

    def write_config(self, **kwargs):
        config_path = os.path.join(self.data_path, 'config.json')
        data_type = self.data_type
        data_name = kwargs.pop('data_name', None)
        if data_name is not None:
            assert self.data_type != 'dict'
        else:
            if self.data_type in ('image', 'image_path'):
                data_type = 'image'
                data_name = 'image'
            elif self.data_type in ('video', 'video_path'):
                data_type = 'video'
                data_name = 'video'
        if self.data_type == 'dict':
            key_names = self.key_names
        else:
            key_names = [data_name]
        key_names.sort()
        config = {
            '_class_name': 'FileDataset',
            '_key_names': key_names,
            'data_size': self._count,
            'data_type': data_type,
            'data_name': data_name,
        }
        config.update(kwargs)
        utils.save_file(config_path, config)
