import numpy as np
import torch
from decord import VideoReader
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor

from ...pipeline import BasePipeline

VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
VIDEO_TOTAL_PIXELS = 24576 * 28 * 28
FRAME_FACTOR = 2


class QwenPipeline(BasePipeline):
    def __init__(self, model_path, device='cuda', use_vllm=False):
        self.use_vllm = use_vllm
        self.processor = AutoProcessor.from_pretrained(model_path)
        if self.use_vllm:
            from vllm import LLM, SamplingParams

            self.model = LLM(
                model=model_path,
                limit_mm_per_prompt={'image': 0, 'video': 1},
                quantization=None,
                max_seq_len_to_capture=32768,
                max_model_len=32768,
                gpu_memory_utilization=0.85,
                max_num_batched_tokens=32768,
                device=device,
            )
            self.sampling_params = SamplingParams(
                temperature=0.1,
                top_p=0.001,
                repetition_penalty=1.05,
                max_tokens=512,
                stop_token_ids=[],
            )
        else:
            from transformers import Qwen2_5_VLForConditionalGeneration

            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype='auto', device_map=device)

    def __call__(self, video, video_valid_range=None, num_frames=-1, prompt=None):
        if prompt is None:
            prompt = 'Elaborate on the visual and narrative elements of the video in detail.'
        frames = load_video(video, video_valid_range, num_frames=num_frames)
        mix_pixels = VIDEO_MIN_PIXELS
        max_pixels = max(
            min(VIDEO_MAX_PIXELS, int(VIDEO_TOTAL_PIXELS / len(frames) * FRAME_FACTOR)),
            int(VIDEO_MIN_PIXELS * 1.05),
        )
        messages = [
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'video',
                        'video': frames,
                        'mix_pixels': mix_pixels,
                        'max_pixels': max_pixels,
                    },
                    {
                        'type': 'text',
                        'text': prompt,
                    },
                ],
            },
        ]
        text_prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        if self.use_vllm:
            if not isinstance(video_inputs[0], torch.Tensor):
                video_inputs = [np.array(_) for _ in video_inputs[0]]
                video_inputs = np.stack(video_inputs)
                video_inputs = torch.tensor(video_inputs).permute(0, 3, 1, 2)
                video_inputs = [video_inputs]
            llm_inputs = {
                'prompt': text_prompt,
                'multi_modal_data': dict(video=video_inputs),
            }
            outputs = self.model.generate(
                llm_inputs,
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )
            output_text = outputs[0].outputs[0].text
        else:
            inputs = self.processor(
                text=[text_prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors='pt',
            )
            inputs = inputs.to(self.model.device)
            generated_ids = self.model.generate(**inputs, max_new_tokens=512)
            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            output_text = output_text[0]
        return output_text


def load_video(video, video_valid_range=None, num_frames=-1):
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
        frames = frames.numpy() if isinstance(frames, torch.Tensor) else frames.asnumpy()
        frames = [Image.fromarray(frame) for frame in frames]
    else:
        frames = []
        for i in sample_indexes:
            frame = video[i]
            if not isinstance(frame, Image.Image):
                frame = Image.fromarray(frame)
            frames.append(frame)
    return frames
