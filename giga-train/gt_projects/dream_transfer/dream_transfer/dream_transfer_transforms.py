import copy
import random

import numpy as np
import torch
from giga_datasets import video_utils
from giga_train import TRANSFORMS, PromptTransform
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F


@TRANSFORMS.register
class DreamTransferTransform:
    def __init__(
        self,
        size_method=1,
        dst_size=None,
        num_frames=1,
        stride=1,
        image_cfg=None,
        cn_video_cfg=None,
        edge_cfg=None,
        depth_cfg=None,
        camera_cfg=None,
        prompt_cfg=None,
        is_train=False,
    ):
        self.is_train = is_train
        self.video_transform = VideoTransform(
            size_method=size_method,
            dst_size=dst_size,
            num_frames=num_frames,
            stride=stride,
            image_cfg=image_cfg,
            cn_video_cfg=cn_video_cfg,
            edge_cfg=edge_cfg,
            camera_cfg=camera_cfg,
            depth_cfg = depth_cfg
        )
        self.prompt_transform = PromptTransform(is_train=is_train, **prompt_cfg)

    def __call__(self, data_dict):
        data_dict = self.prompt_transform.preprocess(data_dict, num_frames=self.video_transform.num_frames)
        data_dict = self.video_transform(data_dict)
        data_dict = self.prompt_transform(data_dict)
        if self.is_train:
            new_data_dict = {}
            if 'input_fps' in data_dict:
                new_data_dict['fps'] = data_dict['input_fps']
            if 'input_images' in data_dict:
                new_data_dict['images'] = data_dict['input_images']
            if 'input_ref_images' in data_dict:
                new_data_dict['ref_images'] = data_dict['input_ref_images']
                new_data_dict['ref_masks'] = data_dict['input_ref_masks']
            if 'prompt_embeds' in data_dict:
                new_data_dict['prompt_embeds'] = data_dict['prompt_embeds']
            if 'depth' in data_dict:
                new_data_dict['depth'] = data_dict['depth']
            if 'input_cn_images' in data_dict:
                new_data_dict['cn_images'] = data_dict['input_cn_images']
        else:
            assert False
        keys = list(new_data_dict.keys())
        for key in keys:
            if new_data_dict[key] is None:
                new_data_dict.pop(key)
        return new_data_dict


class VideoTransform:
    def __init__(
        self,
        size_method=1,
        dst_size=None,
        num_frames=1,
        stride=1,
        image_cfg=None,
        cn_video_cfg=None,
        edge_cfg=None,
        camera_cfg=None,
        depth_cfg=None
    ):
        self.size_method = size_method
        self.dst_size = dst_size
        self.num_frames = num_frames
        self.stride = stride
        self.image_cfg = image_cfg
        self.cn_video_cfg = cn_video_cfg
        self.edge_cfg = edge_cfg
        self.camera_cfg = camera_cfg
        self.depth_cfg = depth_cfg
        if image_cfg is not None:
            self.mask_generator = MaskGenerator(**image_cfg['mask_generator'])
        else:
            self.mask_generator = None
        self.normalize = transforms.Normalize([0.5], [0.5])

    def __call__(self, data_dict):
        num_frames, new_height, new_width, dst_height, dst_width = self.get_new_height_width(data_dict)
        x1 = random.randint(0, new_width - dst_width)
        y1 = random.randint(0, new_height - dst_height)
        video = data_dict['video']
        video_length = data_dict['video_length']
        
        #fps = video.get_avg_fps()
        fps = 24
        assert video_length <= len(video)
        sample_indexes = self.get_sample_indexes(data_dict, num_frames)
        #images = video_utils.sample_video(video, sample_indexes, method=2)
        images = video[sample_indexes]
        data_dict['input_fps'] = fps
        input_images = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous()
        height, width = input_images.shape[2], input_images.shape[3]
        assert data_dict['video_height'] == height and data_dict['video_width'] == width
        input_images = F.resize(input_images, (new_height, new_width), InterpolationMode.BILINEAR)
        input_images = F.crop(input_images, y1, x1, dst_height, dst_width)
        input_images = input_images / 255.0
        input_images = self.normalize(input_images)
        data_dict['input_images'] = input_images
        if self.image_cfg is not None:
            ref_masks, ref_latent_masks = self.mask_generator.get_mask(input_images.shape[0])
            ref_masks = ref_masks[:, None, None, None]
            ref_latent_masks = ref_latent_masks[None, :, None, None]
            ref_images = copy.deepcopy(input_images)
            ref_images = ref_images * ref_masks
            data_dict['input_ref_images'] = ref_images
            data_dict['input_ref_masks'] = ref_latent_masks
        if self.cn_video_cfg is not None:
            cn_video_name = self.cn_video_cfg['name']
            cn_video = data_dict[cn_video_name]
            cn_images = video_utils.sample_video(cn_video, sample_indexes, method=2)
            input_cn_images = torch.from_numpy(cn_images).permute(0, 3, 1, 2).contiguous()
            input_cn_images = F.resize(input_cn_images, (new_height, new_width), InterpolationMode.BILINEAR)
            input_cn_images = F.crop(input_cn_images, y1, x1, dst_height, dst_width)
            input_cn_images = input_cn_images / 255.0
            input_cn_images = self.normalize(input_cn_images)
            data_dict['input_cn_images'] = input_cn_images
        if self.edge_cfg is not None:
            # low_threshold = self.edge_cfg.get('low_threshold', 100)
            # high_threshold = self.edge_cfg.get('high_threshold', 200)
            # edge_images = [cv2.Canny(image, low_threshold, high_threshold) for image in images]
            # edge_images = np.stack(edge_images)[:, :, :, None]
            # edge_images = np.repeat(edge_images, 3, axis=3)
            # import pdb; pdb.set_trace()
            edge_images = data_dict.get('edge', None)
            # import imageio
            # imageio.mimsave("edge.mp4", edge_images, fps=15)
            input_edge_images = torch.from_numpy(edge_images).permute(0, 3, 1, 2).contiguous()
            input_edge_images = F.resize(input_edge_images, (new_height, new_width), InterpolationMode.BILINEAR)
            input_edge_images = F.crop(input_edge_images, y1, x1, dst_height, dst_width)
            input_edge_images = input_edge_images / 255.0
            input_edge_images = self.normalize(input_edge_images)
            data_dict['input_cn_images'] = input_edge_images
        if self.depth_cfg is not None:
            # 获取参数
            depth = data_dict.get('depth', None)
            depth = depth / 255.0
            depth = torch.from_numpy(depth).permute(0, 3, 1, 2).contiguous()
            depth = F.resize(depth, (new_height, new_width), InterpolationMode.BILINEAR)
            depth = F.crop(depth, y1, x1, dst_height, dst_width)
            depth = self.normalize(depth)
            data_dict['input_cn_images'] = depth
        if self.camera_cfg is not None:
            temporal_factor = self.camera_cfg.get('temporal_factor', 1)
            spatial_factor = self.camera_cfg.get('spatial_factor', 1)
            intrinsic = data_dict['camera_intrinsic']
            world2cam = []
            for idx in sample_indexes:
                world2cam.append(data_dict['camera_poses'][idx])
            world2cam = np.stack(world2cam)
            world2cam = np.concatenate([world2cam[:1], world2cam[(1 + temporal_factor // 2) :: temporal_factor]])
            fx = intrinsic[0, 0]
            fy = intrinsic[1, 1]
            cx = intrinsic[0, 2]
            cy = intrinsic[1, 2]
            # scale intrinsic
            scale_x = float(new_width) / width
            scale_y = float(new_height) / height
            fx = fx * width * scale_x
            fy = fy * height * scale_y
            cx = cx * width * scale_x - x1
            cy = cy * height * scale_y - y1
            fx = fx / spatial_factor
            fy = fy / spatial_factor
            cx = cx / spatial_factor
            cy = cy / spatial_factor
            # rel cam_to_world
            rel_cam2world = to_rel_camera_poses(world2cam)
            plucker_embedding = ray_condition(
                [fx, fy, cx, cy],
                rel_cam2world,
                height=dst_height // spatial_factor,
                width=dst_width // spatial_factor,
            )
            data_dict['input_cn_images'] = plucker_embedding
        return data_dict

    def get_new_height_width(self, data_dict):
        if 'video_height' in data_dict:
            num_frames = self.num_frames
        else:
            num_frames = 1
        if self.size_method == 1:
            dst_width, dst_height = self.dst_size
        elif self.size_method == 2:
            if len(data_dict['video_info']) == 2:
                dst_width, dst_height = data_dict['video_info']
            elif len(data_dict['video_info']) == 3:
                dst_width, dst_height, num_frames = data_dict['video_info']
            else:
                assert False
        else:
            assert False
        # if float(dst_height) / height < float(dst_width) / width:
        #     new_height = int(round(float(dst_width) / width * height))
        #     new_width = dst_width
        # else:
        #     new_height = dst_height
        #     new_width = int(round(float(dst_height) / height * width))
        # 不保持原始宽高比，直接将新尺寸设为目标尺寸
        new_width = dst_width
        new_height = dst_height
        assert new_width > 0 and new_height > 0, "目标尺寸必须为正数"

        return num_frames, new_height, new_width, dst_height, dst_width

    def get_sample_indexes(self, data_dict, num_frames):
        if 'video_valid_range' in data_dict:
            valid_range = data_dict['video_valid_range']
        else:
            valid_range = (0, data_dict['video_length'])
        video_length = valid_range[1] - valid_range[0]
        assert num_frames <= video_length
        sample_length = min(video_length, (num_frames - 1) * self.stride + 1)
        start_idx = valid_range[0] + random.randint(0, video_length - sample_length)
        sample_indexes = np.linspace(start_idx, start_idx + sample_length - 1, num_frames, dtype=int)
        return sample_indexes


class MaskGenerator:
    def __init__(self, max_ref_frames, factor=8, start=1):
        assert max_ref_frames > 0 and (max_ref_frames - 1) % factor == 0
        self.max_ref_frames = max_ref_frames
        self.factor = factor
        self.start = start
        self.max_ref_latents = 1 + (max_ref_frames - 1) // factor
        assert self.start <= self.max_ref_latents

    def get_mask(self, num_frames):
        assert num_frames > 0 and (num_frames - 1) % self.factor == 0 and num_frames >= self.max_ref_frames
        num_latents = 1 + (num_frames - 1) // self.factor
        num_ref_latents = random.randint(self.start, self.max_ref_latents)
        if num_ref_latents > 0:
            num_ref_frames = 1 + (num_ref_latents - 1) * self.factor
        else:
            num_ref_frames = 0
        ref_masks = torch.zeros((num_frames,), dtype=torch.float32)
        ref_masks[:num_ref_frames] = 1
        ref_latent_masks = torch.zeros((num_latents,), dtype=torch.float32)
        ref_latent_masks[:num_ref_latents] = 1
        return ref_masks, ref_latent_masks


def to_rel_camera_poses(world2cam):
    abs_world2cam = np.zeros((world2cam.shape[0], 4, 4), dtype=np.float32)
    abs_world2cam[:, 3, 3] = 1
    abs_world2cam[:, :3] = world2cam
    rel_cam2world = [np.eye(4)]
    for i in range(1, abs_world2cam.shape[0]):
        rel_cam2world_i = abs_world2cam[0] @ np.linalg.inv(abs_world2cam[i])
        rel_cam2world.append(rel_cam2world_i)
    rel_cam2world = np.stack(rel_cam2world, dtype=np.float32)
    return rel_cam2world


def ray_condition(intrinsic, cam2world, height, width):
    cam2world = torch.as_tensor(cam2world)

    j, i = torch.meshgrid(
        torch.linspace(0, height - 1, height, dtype=cam2world.dtype),
        torch.linspace(0, width - 1, width, dtype=cam2world.dtype),
        indexing='ij',
    )
    i = i.reshape(-1) + 0.5  # HW
    j = j.reshape(-1) + 0.5  # HW

    fx, fy, cx, cy = intrinsic

    zs = torch.ones_like(i)  # [B, HxW]
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs

    directions = torch.stack((xs, ys, zs), dim=-1)  # (HW, 3)
    directions = directions / directions.norm(dim=-1, keepdim=True)  # (HW, 3)

    rays_d = directions[None] @ cam2world[:, :3, :3].transpose(-1, -2)  # (T, HW, 3)
    rays_o = cam2world[:, :3, 3]  # (T, 3)
    rays_o = rays_o[:, None].expand_as(rays_d)  # (T, HW, 3)

    rays_od = torch.cross(rays_o, rays_d, dim=-1)  # (T, HW, 3)
    plucker = torch.cat([rays_d, rays_od], dim=-1)
    plucker = plucker.reshape(cam2world.shape[0], height, width, 6)  # (T, H, H, 6)
    plucker = plucker.permute(3, 0, 1, 2)  # (6, T, H, W)

    return plucker
