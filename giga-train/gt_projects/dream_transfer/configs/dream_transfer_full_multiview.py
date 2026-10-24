import os
from pathlib import Path

dst_size = (1920, 480)
num_frames = 121
project_root = Path(__file__).resolve().parents[2]
config = dict(
    project_dir=os.getenv('DREAMTRANSFER_OUTPUT_DIR', 'outputs/dream_transfer'),
    launch=dict(
        gpu_ids=[0],
        distributed_type='DEEPSPEED',
        deepspeed_config=dict(
            deepspeed_config_file=str(project_root / 'accelerate_configs' / 'zero2.json'),
        ),
        until_completion=True,
    ),
    dataloaders=dict(
        train=dict(
            data_or_config=[os.environ['DREAMTRANSFER_TRAIN_DATA']],
            batch_size_per_gpu=1,
            num_workers=0,
            filter=dict(
                mode='overall_func',
                func='configs.dream_transfer_full_multiview.filter_data',
                dst_size=dst_size,
                min_num_frames=num_frames,
                min_area=dst_size[0] * dst_size[1],
                min_size=4,
            ),
            transform=dict(
                type='DreamTransferTransform',
                size_method=2,
                dst_size=dst_size,
                num_frames=num_frames,
                image_cfg=dict(
                    mask_generator=dict(
                        max_ref_frames=1,
                        start=0,
                    ),
                ),
                depth_cfg=dict(

                ),
                is_train=True,
                prompt_cfg=dict(
                    prompt_mode='from_dict',
                    prompt_embeds_mode='from_dict',
                    prompt_names=['prompt_qwen'],
                    default_prompt_prob=0.2,
                    cache_path=os.environ['DREAMTRANSFER_PROMPT_CACHE'],
                ),
            ),
            sampler=dict(
                type='BucketSampler',
            ),
            collator=dict(
                is_equal=True,
            ),
        ),
        test=dict(),
    ),
    models=dict(
        vae_model_path=os.environ['DREAMTRANSFER_VAE_MODEL'],
        transformer_model_path=os.environ['DREAMTRANSFER_TRANSFORMER_MODEL'],
        controlnet_model_path=os.environ['DREAMTRANSFER_CONTROLNET_MODEL'],
    ),
    optimizers=dict(
        type='AdamW',
        lr=1e-5,
        weight_decay=1e-2,
    ),
    schedulers=dict(
        type='ConstantScheduler',
    ),
    train=dict(
        resume=False,
        max_epochs=4,
        gradient_accumulation_steps=1,
        mixed_precision='bf16',  # fp16, bf16
        checkpoint_interval=50,
        checkpoint_total_limit=1000,
        checkpoint_safe_serialization=False,
        checkpoint_strict=False,
        log_with='tensorboard',
        log_interval=1,
        with_ema=True,
        activation_checkpointing=True,
        activation_class_names=['GeneralDITTransformerBlock'],
    ),
    test=dict(),
)


def filter_data(
    all_data_list,
    dst_size=(1280, 704),
    min_num_frames=121,
    multiple=16,
    min_area=-1,
    min_size=1,
):
    from giga_datasets import image_utils

    video_info_dict = dict()
    for n, data_list in enumerate(all_data_list):
        for m, data_dict in enumerate(data_list):
            video_length = data_dict['video_length']
            video_height = data_dict['video_height']
            video_width = data_dict['video_width']
            if video_height * video_width < min_area:
                print("video_height:",video_height)
                print("video_width:",video_width)
                continue
            # scores
            scores = data_dict.get('scores', None)
            if scores is not None:
                optical_flow_unimatch = scores['optical_flow_unimatch']
                aesthetic = scores['aesthetic']
                ocr = scores['ocr']
                laplacian = scores['laplacian']
                if optical_flow_unimatch < 2:
                    continue
                if aesthetic < 4:
                    continue
                if ocr > 0.07:
                    continue
                if laplacian < 20:
                    continue
            # video_valid_range
            video_valid_range = data_dict.get('video_valid_range', None)
            if video_valid_range is None:
                video_valid_range = (0, video_length)
            video_length = video_valid_range[1] - video_valid_range[0]
            
            if video_length < min_num_frames:
                #print("video_length:",video_length)
                continue
            data_dict['video_valid_range'] = video_valid_range
            # image size
            dst_width, dst_height = image_utils.get_image_size(
                (video_width, video_height),
                dst_size,
                mode='fixed',
                multiple=multiple,
            )
            video_info = (dst_width, dst_height)
            if video_info not in video_info_dict:
                video_info_dict[video_info] = []
            video_info_dict[video_info].append((n, m))

    new_all_data_list = [[] for _ in range(len(all_data_list))]
    bucket_index = 0
    for video_info, data_indexes in video_info_dict.items():
        if len(data_indexes) >= min_size:
            for n, m in data_indexes:
                data_dict = all_data_list[n][m]
                data_dict['bucket_index'] = bucket_index
                data_dict['video_info'] = video_info
                new_all_data_list[n].append(data_dict)
            bucket_index += 1
    return new_all_data_list
