import numpy as np
import torch
from decord import VideoReader
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...pipeline import BasePipeline


class CogVLM2Pipeline(BasePipeline):
    def __init__(self, model_path, torch_dtype):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype, trust_remote_code=True)
        self.model.eval()
        self.model.to('cpu')
        self.dtype = torch_dtype
        self.device = 'cpu'
        self.num_frames = 24

    def to(self, device):
        self.device = device
        self.model.to(device)
        return self

    def __call__(self, video, video_valid_range=None, prompt=None):
        if prompt is None:
            prompt = 'Please describe this video in detail.'
        frames = load_video(video, video_valid_range, num_frames=self.num_frames)
        history = []
        inputs = self.model.build_conversation_input_ids(
            tokenizer=self.tokenizer,
            query=prompt,
            images=[frames],
            history=history,
            template_version='chat',
        )
        inputs = {
            'input_ids': inputs['input_ids'].unsqueeze(0).to(self.device),
            'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to(self.device),
            'attention_mask': inputs['attention_mask'].unsqueeze(0).to(self.device),
            'images': [[inputs['images'][0].to(self.device).to(self.dtype)]],
        }
        gen_kwargs = {
            'max_new_tokens': 2048,
            'pad_token_id': 128002,
            'top_k': 1,
            'do_sample': False,
            'top_p': 0.1,
            'temperature': 0.1,
        }
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1] :]
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response


def load_video(video, video_valid_range=None, num_frames=24):
    if isinstance(video, str):
        video = VideoReader(video)
    if video_valid_range is None:
        video_valid_range = (0, len(video))
    video_length = video_valid_range[1] - video_valid_range[0]
    num_frames = num_frames if num_frames > 0 else video_length
    num_frames = min(num_frames, video_length)
    sample_indexes = np.linspace(video_valid_range[0], video_valid_range[1] - 1, num_frames, dtype=int)
    if isinstance(video, VideoReader):
        frames = video.get_batch(sample_indexes)
        if not isinstance(frames, torch.Tensor):
            frames = frames.asnumpy()
    else:
        frames = video[sample_indexes]
    if not isinstance(frames, torch.Tensor):
        frames = torch.from_numpy(frames)
    frames = frames.permute(3, 0, 1, 2)
    return frames
