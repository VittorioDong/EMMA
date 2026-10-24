import os.path
import sys

import numpy as np
import torch
from decord import VideoReader

try:
    import video_llama  # noqa F401
except ImportError:
    from ....utils import git_clone

    repo_path = git_clone('https://github.com/snap-research/Panda-70M.git')
    sys.path.insert(0, os.path.join(repo_path, 'captioning'))

from video_llama.models import VideoLLAMA
from video_llama.processors import AlproVideoEvalProcessor

from ...pipeline import BasePipeline


class PandaVideoLLaMAPipeline(BasePipeline):
    def __init__(self, model_path, q_former_model_path, llama_model_path):
        self.processor = AlproVideoEvalProcessor(
            image_size=224,
            n_frms=8,
        )
        self.model = VideoLLAMA(
            q_former_model=q_former_model_path,
            llama_model=llama_model_path,
            input_prompt=True,
            num_query_token=32,
            max_caption_len=48,
            max_prompt_len=200,
        )
        state_dict = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(state_dict['model'], strict=False)
        self.device = 'cpu'

    def to(self, device):
        self.device = device
        self.model.to(device)
        return self

    def __call__(self, video, video_valid_range=None, prompt=None):
        if prompt is None:
            prompt = 'Please faithfully summarize the following video in one sentence.'
        frames = load_video(video, video_valid_range, self.processor.n_frms)
        frames = frames.permute(3, 0, 1, 2).contiguous().float()  # (C, T, H, W)
        frames = self.processor.transform(frames).to(self.device)
        output_prompt = self.model.inference(frames[None], [prompt])[0]
        output_prompt = output_prompt.capitalize() + '.'
        return output_prompt


def load_video(video, video_valid_range=None, num_frames=0):
    if isinstance(video, str):
        video = VideoReader(video)
    if video_valid_range is None:
        video_valid_range = (0, len(video))
    video_length = video_valid_range[1] - video_valid_range[0]
    num_frames = num_frames if num_frames > 0 else video_length
    num_frames = min(num_frames, video_length)
    sample_indexes = np.linspace(video_valid_range[0], video_valid_range[1] - 1, num_frames, dtype=int)
    frames = video.get_batch(sample_indexes)
    if not isinstance(frames, torch.Tensor):
        frames = torch.from_numpy(frames)
    return frames
