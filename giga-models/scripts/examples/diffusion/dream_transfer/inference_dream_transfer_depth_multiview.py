import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import imageio
import numpy as np
import torch
from accelerate.utils import set_seed
from giga_datasets import image_utils
from giga_models import DreamTransferPipeline
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F
from tqdm import tqdm


def build_parser():
    parser = argparse.ArgumentParser(description='Run DreamTransfer depth-conditioned multiview inference.')
    parser.add_argument('--transformer_model', required=True)
    parser.add_argument('--controlnet_model', required=True)
    parser.add_argument('--text_encoder_model', required=True)
    parser.add_argument('--vae_model', required=True)
    parser.add_argument('--depth_model', required=True, help='Path to the Video-Depth-Anything checkpoint.')
    parser.add_argument(
        '--depth_model_source',
        help='Optional path to a Video-Depth-Anything source checkout when it is not installed.',
    )
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--runmode', default='color', help='Prompt file suffix, for example color or lighting.')
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=None)
    parser.add_argument('--width', type=int, default=1920)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--num_frames', type=int, default=121)
    parser.add_argument('--num_inference_steps', type=int, default=30)
    parser.add_argument('--seed', type=int, default=6666)
    parser.add_argument('--is_test', action='store_true', help='Process at most two chunks per video.')
    parser.add_argument('--disable_acceleration', action='store_true')
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--testset_dir', '--input_dir', dest='testset_dir', required=True)
    return parser

def split_combined_images(images: List[Image.Image]) -> Tuple[List[Image.Image], List[Image.Image], List[Image.Image]]:
    """
    将拼接的图像按左右顺序切分为三个子图像
    
    参数:
        images: 包含 PIL Image 对象的列表，每个图像由三个子图像横向拼接而成
    
    返回:
        元组，包含三个列表，分别对应左、中、右三个子图像
    """
    # 初始化三个子图像列表
    cam_left_wrist_images = []
    cam_high_images = []
    cam_right_wrist_images = []
    
    # 子图像尺寸和间距
    sub_width = images[0].size[0] // 3
    sub_height = images[0].size[1]
    gap = 0
    
    for image in images:
        # 验证图像尺寸
        width, height = image.size
        expected_width = sub_width * 3 + gap * 2
        expected_height = sub_height
        
        if width != expected_width or height != expected_height:
            raise ValueError(f"图像尺寸不符合预期: 期望 {expected_width}x{expected_height}, 实际 {width}x{height}")
        
        # 切分图像
        left_box = (0, 0, sub_width, sub_height)
        middle_box = (sub_width + gap, 0, sub_width * 2 + gap, sub_height)
        right_box = (sub_width * 2 + gap * 2, 0, sub_width * 3 + gap * 2, sub_height)
        
        # 裁剪并添加到对应列表
        left_image = image.crop(left_box)
        middle_image = image.crop(middle_box)
        right_image = image.crop(right_box)
        
        cam_left_wrist_images.append(left_image)
        cam_high_images.append(middle_image)
        cam_right_wrist_images.append(right_image)
    
    return cam_left_wrist_images, cam_high_images, cam_right_wrist_images

def merge_split_images(
    left_images: List[Image.Image], 
    middle_images: List[Image.Image], 
    right_images: List[Image.Image]
) -> List[Image.Image]:
    """
    将三个子图像列表按左右顺序合并为原始的拼接图像列表
    
    参数:
        left_images: 左相机子图像列表
        middle_images: 中相机子图像列表
        right_images: 右相机子图像列表
    
    返回:
        合并后的原始拼接图像列表
    """
    # 验证三个列表长度相同
    if not (len(left_images) == len(middle_images) == len(right_images)):
        raise ValueError("三个子图像列表的长度必须相同")
    
    # 子图像尺寸和间距
    sub_width = left_images[0].size[0]
    sub_height = left_images[0].size[1]
    gap = 0
    
    # 初始化合并后的图像列表
    merged_images = []
    
    # 遍历每个索引，合并对应的子图像
    for i in range(len(left_images)):
        # 创建新的空白图像，尺寸为 1920x480
        merged_image = Image.new('RGB', (sub_width * 3 + gap * 2, sub_height))
        
        # 将三个子图像粘贴到正确位置
        merged_image.paste(left_images[i], (0, 0))
        merged_image.paste(middle_images[i], (sub_width + gap, 0))
        merged_image.paste(right_images[i], (sub_width * 2 + gap * 2, 0))
        
        # 添加到合并后的列表
        merged_images.append(merged_image)
    
    return merged_images

def inference_depth(depth_model, device, left_images, middle_images, right_images):
    depth_multiview = []
    for single_view_video in [left_images, middle_images, right_images]:
        video_list = []
        for frame in single_view_video:
            video_list.append(np.array(frame))
        video_list = np.array(video_list)
        height, width, _ = video_list[0].shape
        depth_map, _ = depth_model.infer_video_depth(
            video_list,
            target_fps=24,
            input_size=518,
            device=device,
        )
        depth_map = torch.tensor(depth_map, device=device)
        # 调整深度图尺寸
        depth_map = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1),
            size=(height, width),
            mode='bicubic',
            align_corners=False,
        ).squeeze().unsqueeze(-1)

        # 归一化并生成伪彩色深度图
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8) * 255.0
        depth_image_np = torch.cat([depth_map] * 3, dim=-1).cpu().numpy().clip(0, 255).astype(np.uint8)
        depth_image = []
        for img in depth_image_np:
            depth_image.append(Image.fromarray(img.astype('uint8')).convert('RGB'))
        depth_multiview.append(depth_image)
    return depth_multiview


def concat_video(cam_high_video: np.ndarray,
                 cam_left_wrist_video: np.ndarray,
                 cam_right_wrist_video: np.ndarray) -> np.ndarray:
    """
    横向拼接三个分辨率相同的视角视频（左→中→右顺序：左手腕 → 正面 → 右手腕）
    
    前提条件：
        三个输入视频的分辨率完全一致：帧数、高度、宽度、通道数均相同
        输入数组形状均为 [帧数N, 高度H, 宽度W, 通道数C]（C通常为3，RGB/BGR）
    
    参数:
        cam_high_video: 正面视角视频数组（中），形状 [N, H, W, C]
        cam_left_wrist_video: 左手腕视角视频数组（左），形状 [N, H, W, C]
        cam_right_wrist_video: 右手腕视角视频数组（右），形状 [N, H, W, C]
    
    返回:
        np.ndarray: 横向拼接后的视频数组，形状 [N, H, 3*W, C]
                   通道顺序与输入一致（输入BGR则输出BGR，输入RGB则输出RGB）
    """
    # 1. 校验输入视频的形状一致性（基于前提条件增强容错）
    videos = [cam_left_wrist_video, cam_high_video, cam_right_wrist_video]
    video_names = ["left", "high", "right"]
    target_shape = videos[0].shape  # 以第一个视频形状为基准
    
    for idx, (video, name) in enumerate(zip(videos, video_names)):
        # 校验维度（必须是4维：帧数、高、宽、通道）
        if video.ndim != 4:
            raise ValueError(f"{name}视频格式错误！需为4维数组[帧数, 高度, 宽度, 通道]，当前维度：{video.ndim}")
        # 校验形状与基准一致
        if video.shape != target_shape:
            raise ValueError(
                f"三个视频分辨率不一致！\n"
                f"基准形状（{video_names[0]}）：{target_shape}\n"
                f"{name}视频形状：{video.shape}"
            )
        # 校验非空（至少1帧）
        if video.shape[0] == 0:
            raise ValueError(f"{name}视频为空（0帧），无法拼接")
    
    # 2. 横向拼接（axis=2 对应宽度维度，沿水平方向堆叠）
    # 顺序：左手腕（左）→ 正面（中）→ 右手腕（右）
    concatenated_video = np.concatenate(videos, axis=2)
    
    return concatenated_video


def video_reader(video_path: str) -> np.ndarray:
    """
    读取视频文件（支持AV1编码），返回 (T, H, W, 3) 格式的RGB视频数组

    Args:
        video_path (str): 视频文件路径

    Returns:
        list: 视频帧列表，每个元素为 (H, W, 3) 的RGB图像
    """
    try:
        # 使用imageio读取视频，自动处理AV1解码
        reader = imageio.get_reader(video_path, format='ffmpeg')
        frames = []
        for frame in reader:
            # frame 已经是 (H, W, 3) 的RGB格式，直接添加
            frames.append(frame)
        reader.close()
        # 转换为 numpy 数组（空视频返回形状为 [0, H, W, 3] 的数组，便于后续拼接）
        if not frames:
            raise Warning(f"视频 {video_path} 无有效帧（可能是空视频）")

        return np.stack(frames, axis=0)

    except Exception as e:
        raise IOError(f"读取视频 {video_path} 失败: {str(e)}")

def is_video_processed(save_dir, save_name, check_subvideos=True):
    """
    检查视频是否已经处理完成
    Args:
        save_dir: 保存目录
        save_name: 视频保存名称（不含后缀）
        check_subvideos: 是否检查子视频片段（可选，增加鲁棒性）
    Returns:
        bool: True表示已处理，False表示未处理
    """
    # 检查主视频文件是否存在
    main_video_path = os.path.join(save_dir, f"{save_name}.mp4")
    if not os.path.exists(main_video_path):
        return False
    
    return True

def inference_dream_transfer_giga_folder(
    device, 
    suffix,
    width,
    height,
    depth_model,
    pipe,
    runmode=None,
    start_idx=0,
    end_idx=None,
    num_frames=121,
    num_inference_steps=30,
    seed=6666,
    is_test=False,
    disable_acceleration=False,
    save_dir=None,
    testset_dir=None,
):
    device = torch.device(device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    mode = 'depth'
    
    if depth_model is not None:
        depth_model = depth_model.to(device).eval()

    # 加速
    if not disable_acceleration and pipe is not None:
        # attn 改用 sageattn
        pipe.set_attn_op('sage')
        pipe.set_cache(mode='adacache')

    if pipe is not None:
        pipe.to(device)
    
    # load data
    negative_prompt = 'The video captures a series of frames showing ugly scenes, static with no motion, motion blur, \
                    over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, \
                    underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, \
                    jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special \
                    effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and \
                    flickering. Overall, the video is of poor quality.'
    
    video_paths_file = os.path.join(testset_dir, 'video_paths.json')
    with open(video_paths_file, 'r', encoding='utf-8') as f:
        video_paths = json.load(f)
    prompt_path = os.path.join(testset_dir, f'prompt_{runmode}.json')
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompts = json.load(f)

    data_list = []
    # import pdb; pdb.set_trace()

    for video_id, video_path in tqdm(video_paths.items(), desc="Reading videos", total=len(video_paths)):
        video_id = int(video_id)
        cam_high_path = video_path['cam_high']
        cam_left_wrist_path = video_path['cam_left_wrist']
        cam_right_wrist_path = video_path['cam_right_wrist']

        # 选择数据
        if video_id < start_idx or (end_idx is not None and video_id >= end_idx):
            continue

        data_dict = dict(
            prompt=prompts[str(video_id)],
            cam_high_path=cam_high_path,
            cam_left_wrist_path=cam_left_wrist_path,
            cam_right_wrist_path=cam_right_wrist_path,
            video_id=video_id,
            save_name=f'transfer_episode_{video_id}_{suffix}',
            save_fps=30,
        )
        data_list.append(data_dict)
    
    print(f"\n📌 Total unprocessed videos to handle: {len(data_list)}")

    # inference
    for n in tqdm(range(len(data_list)), desc=f"Transfering {len(data_list)} full videos"):
        print(f'\n🚀 Starting inference for video_id: {data_list[n]["video_id"]}')
        start = time.time()
        set_seed(seed)
        data_dict = data_list[n]
        prompt = data_dict['prompt']
        cam_high_path = data_dict['cam_high_path']
        cam_left_wrist_path = data_dict['cam_left_wrist_path']
        cam_right_wrist_path = data_dict['cam_right_wrist_path']
        # 读取视频
        cam_high_video = video_reader(cam_high_path)
        cam_left_wrist_video = video_reader(cam_left_wrist_path)
        cam_right_wrist_video = video_reader(cam_right_wrist_path)
        video_id = data_dict['video_id']
        valid_range = (0, len(cam_high_video))
        video_length = valid_range[1] - valid_range[0]

        # 如果视频少于 121 帧，则跳过
        # if video_length < num_frames:
        #     continue

        # sample_indexes = np.arange(video_length)
        video = concat_video(cam_high_video, cam_left_wrist_video, cam_right_wrist_video)
        # images = video_utils.sample_video(video, sample_indexes, method=2)
        images = [Image.fromarray(image) for image in video]
        video_height = images[0].height
        video_width = images[0].width
        dst_width, dst_height = image_utils.get_image_size((video_width, video_height), (width, height), mode='fixed', multiple=16)

        # 不保持原始宽高比，直接将新尺寸设为目标尺寸
        new_width = dst_width
        new_height = dst_height
        assert dst_width <= new_width and dst_height <= new_height
        x1 = (new_width - dst_width) // 2
        y1 = (new_height - dst_height) // 2

        def resize_images(images_, interpolation=InterpolationMode.BILINEAR):
            new_images = []
            for image in images_:
                image = F.resize(image, (new_height, new_width), interpolation)
                image = F.crop(image, y1, x1, dst_height, dst_width)
                new_images.append(image)
            return new_images

        input_images = resize_images(images)

        # 从 resize 后的拼接图像中切分三个视角，保证深度图与输入图像分辨率一致 (dst_width x dst_height)
        left_images, middle_images, right_images = split_combined_images(input_images)
        cn_left_images, cn_middle_images, cn_right_images = inference_depth(depth_model, device, left_images, middle_images, right_images)

        input_cn_images = merge_split_images(cn_left_images, cn_middle_images, cn_right_images)

        video_splits = (video_length + num_frames - 1) // num_frames
        output_image_all = []

        # 创建子视频保存目录
        subvideo_save_dir = os.path.join(save_dir, f'subvideos_{data_dict["save_name"]}')
        os.makedirs(subvideo_save_dir, exist_ok=True)
        
        if is_test:
            # sample_count = min(3, video_splits)
            # if sample_count == 1:
            #     selected_indices = [0]
            # else:
            #     interval = max(1, video_splits // (sample_count - 1))
            #     selected_indices = [i * interval for i in range(sample_count)]
            #     selected_indices[-1] = min(selected_indices[-1], video_splits - 1)
            selected_indices = range(min(2, video_splits))
        else:
            selected_indices = range(video_splits)

        for split_idx in tqdm(selected_indices, desc=f"Transfering video_id:{video_id}"):
            # import pdb; pdb.set_trace()
            # 计算当前块的起始/结束帧
            start_frame = split_idx * num_frames
            end_frame = min(start_frame + num_frames, video_length)
            current_block_frames = end_frame - start_frame

            # 提取当前块的原始帧和深度帧
            block_input_images = input_images[start_frame:end_frame]
            block_cn_images = input_cn_images[start_frame:end_frame]

            # 补帧到121帧（不足时重复最后一帧）
            pad_frames = num_frames - len(block_input_images)
            if pad_frames > 0 and len(block_input_images) > 0:
                # 对输入图像补帧
                last_input = block_input_images[-1]
                block_input_images += [last_input] * pad_frames
                # 对深度图补帧
                last_cn = block_cn_images[-1]
                block_cn_images += [last_cn] * pad_frames
            
            # 确保输入帧数符合模型要求
            assert len(block_input_images) == num_frames and len(block_cn_images) == num_frames, \
                f"Split {split_idx} input frame count error!"
            
            output_frames = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=[],
                video = block_input_images,
                guidance_scale=5.0,
                num_inference_steps=num_inference_steps,
                num_frames=num_frames,
                height=dst_height,
                width=dst_width,
                control_video={
                    mode: block_cn_images,
                },
                control_weight={
                    mode: 1.0,
                },
                sigma_max=70.0,  # 70.0 if video is not None else 80.0
                seed=seed,
            )[0]
            # import pdb; pdb.set_trace()
            # for k in range(cutoff_frame_id, len(output_images)):
            #     output_image_all.append(output_images[k])
            # 只保留有效帧（去掉补帧部分），并添加到全局列表
            valid_frames = output_frames[:current_block_frames]
            output_image_all.extend(valid_frames)

             # 保存当前子视频片段
            sub_vis_frames = []
            for i in range(current_block_frames):
                # 取当前块的原始帧、深度帧、生成帧
                raw_frame = input_images[start_frame + i]
                cn_frame = input_cn_images[start_frame + i]
                gen_frame = valid_frames[i]
                # 垂直拼接可视化
                vis_frame = image_utils.concat_images([raw_frame, cn_frame, gen_frame], direction='vertical', pad=0)
                sub_vis_frames.append(vis_frame)
            
            # import pdb; pdb.set_trace()

            subvideo_save_path = os.path.join(subvideo_save_dir, f'{data_dict["save_name"]}_split_{split_idx}.mp4')
            imageio.mimsave(subvideo_save_path, sub_vis_frames, fps=data_dict['save_fps'])
            print(f"Saved subvideo {i} to {subvideo_save_path}")

        # 4. 保存最终完整视频（核心修正：帧对齐）
        end = time.time()
        save_fps = data_dict['save_fps']
        vis_images = []
        # 确保生成帧数量和原始帧一致
        assert len(output_image_all) == video_length, \
            f"Video {video_id} frame count mismatch! Generated: {len(output_image_all)}, Original: {video_length}"
        
        # 逐帧拼接可视化（原始 + 深度 + 生成）
        for i in range(video_length):
            raw_frame = input_images[i]
            cn_frame = input_cn_images[i]
            gen_frame = output_image_all[i]
            vis_frame = image_utils.concat_images([raw_frame, cn_frame, gen_frame], direction='vertical', pad=0)
            vis_images.append(vis_frame)
        
        # import pdb; pdb.set_trace()
        
        # 保存完整视频
        save_path = os.path.join(save_dir, f'{data_dict["save_name"]}.mp4')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        imageio.mimsave(save_path, vis_images, fps=save_fps)

        # 打印日志
        print(f"\n⏱️ Time for video {video_id}: {end-start:.2f} seconds")
        print(f"✅ Saved full video to {save_path} (total frames: {len(vis_images)})")
        print(f"🎉 Video {video_id} processing completed!\n")

def load_models(
    transformer_model_path,
    controlnet_model_path,
    text_encoder_model_path,
    vae_model_path,
    depth_model_path,
    depth_model_source=None,
):
    mode = 'depth'
    pipe = DreamTransferPipeline.from_pretrained(
        transformer_model_path=transformer_model_path,
        text_encoder_model_path=text_encoder_model_path,
        vae_model_path=vae_model_path,
        controlnet_model_path={
            mode: controlnet_model_path,
        },
    )

    if depth_model_source:
        sys.path.insert(0, str(Path(depth_model_source).expanduser().resolve()))
    try:
        from video_depth_anything.video_depth import VideoDepthAnything
    except ImportError as exc:
        raise ImportError(
            'Video-Depth-Anything is required. Install it or pass --depth_model_source.'
        ) from exc

    model_config = {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
    depth_model = VideoDepthAnything(**model_config)
    depth_model.load_state_dict(torch.load(depth_model_path, map_location='cpu'), strict=True)

    return depth_model, pipe

def main():
    args = build_parser().parse_args()
    depth_model, pipe = load_models(
        transformer_model_path=args.transformer_model,
        controlnet_model_path=args.controlnet_model,
        text_encoder_model_path=args.text_encoder_model,
        vae_model_path=args.vae_model,
        depth_model_path=args.depth_model,
        depth_model_source=args.depth_model_source,
    )

    inference_dream_transfer_giga_folder(
        device=args.device,
        suffix=args.runmode,
        width=args.width,
        height=args.height,
        depth_model=depth_model,
        pipe=pipe,
        runmode=args.runmode,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        is_test=args.is_test,
        disable_acceleration=args.disable_acceleration,
        save_dir=args.save_dir,
        testset_dir=args.testset_dir,
    )

if __name__ == "__main__":
    main()
