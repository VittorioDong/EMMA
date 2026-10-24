import sys

import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader
from einops import rearrange
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from ...pipeline import BasePipeline

try:
    import unimatch  # noqa F401
except ImportError:
    from ....utils import git_clone

    repo_path = git_clone('https://github.com/autonomousvision/unimatch.git')
    sys.path.insert(0, repo_path)

from unimatch.unimatch import UniMatch


class UniMatchPipeline(BasePipeline):
    def __init__(self, model_path=None):
        self.model = UniMatch(
            feature_channels=128,
            num_scales=2,
            upsample_factor=4,
            num_head=1,
            ffn_dim_expansion=4,
            num_transformer_layers=6,
            reg_refine=True,
            task='flow',
        ).eval()
        ckpt = torch.load(model_path)
        self.model.load_state_dict(ckpt['model'])
        self.device = 'cpu'

    def to(self, device):
        self.device = device
        self.model.to(self.device)
        return self

    def __call__(self, video_path, video_valid_range=None):
        video = VideoReader(video_path)
        if video_valid_range is None:
            video_valid_range = (0, len(video))
        images = read_video_frames(video, video_valid_range)
        optical_flow = float(get_unimatch(images, self.model, self.device)[0])
        return optical_flow


def read_video_frames(video, video_valid_range):
    start_index = video_valid_range[0]
    end_index = video_valid_range[1]
    frame_inds = list(range(start_index, end_index, 10))
    frame_inds = frame_inds[:8]
    images = video.get_batch([i for i in range(0, end_index)]).asnumpy().astype(np.uint8)
    images = [Image.fromarray(images[i]) for i in frame_inds]
    # transform
    images = torch.stack([pil_to_tensor(x) for x in images])  # shape: [N, C, H, W]; dtype: torch.uint8
    images = images.float()
    H, W = images.shape[-2:]
    if H > W:
        images = rearrange(images, 'N C H W -> N C W H')
    images = F.interpolate(images, size=(320, 576), mode='bilinear', align_corners=True)
    return images


def get_unimatch(images, model, device):
    images = images.to(device)
    images = images.unsqueeze(0)
    B = images.shape[0]
    batch_0 = rearrange(images[:, :-1], 'B N C H W -> (B N) C H W').contiguous()
    batch_1 = rearrange(images[:, 1:], 'B N C H W -> (B N) C H W').contiguous()
    with torch.no_grad():
        res = model(
            batch_0,
            batch_1,
            attn_type='swin',
            attn_splits_list=[2, 8],
            corr_radius_list=[-1, 4],
            prop_radius_list=[-1, 1],
            num_reg_refine=6,
            task='flow',
            pred_bidir_flow=False,
        )
        flow_maps = res['flow_preds'][-1].cpu()  # [B * (N-1), 2, H, W]
        flow_maps = rearrange(flow_maps, '(B N) C H W -> B N H W C', B=B)
        flow_scores = flow_maps.abs().mean(dim=[1, 2, 3, 4])
        flow_scores = flow_scores.tolist()
    return flow_scores
