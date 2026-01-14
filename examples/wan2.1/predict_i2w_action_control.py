# Modifications Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.

# This file is a modified version of code from VideoXFun
# (https://github.com/aigc-apps/VideoX-Fun),
# which is licensed under the Apache License, Version 2.0.
import os
import sys

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from PIL import Image
from transformers import AutoTokenizer
from datetime import datetime
import pandas as pd
current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from microworld.models import (AutoencoderKLWan, AutoTokenizer, CLIPModel,
                              WanT5EncoderModel, WanActionControlNetModel, WanActionAdaLNModel)
from microworld.models.cache_utils import get_teacache_coefficients
from microworld.pipeline.pipeline_wan_action_i2w import WanActionI2WPipeline
from microworld.utils.lora_utils import merge_lora, unmerge_lora
from microworld.utils.utils import (filter_kwargs, get_image_to_video_latent, get_video_to_video_latent,
                                   save_videos_grid, replace_parameters_by_name, parse_action_list)
from microworld.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from microworld.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

# GPU memory mode, which can be choosen in [model_full_load, model_cpu_offload, sequential_cpu_offload].
# model_full_load means that the entire model will be moved to the GPU.
# 
# model_cpu_offload means that the entire model will be moved to the CPU after use, which can save some GPU memory.
# 
# sequential_cpu_offload means that each layer of the model will be moved to the CPU after use, 
# resulting in slower speeds but saving a large amount of GPU memory.
GPU_memory_mode     = "model_full_load"

# Support TeaCache.
enable_teacache     = True

teacache_threshold  = 0.20
# The number of steps to skip TeaCache at the beginning of the inference process, which can
# reduce the impact of TeaCache on generated video quality.
num_skip_start_steps = 5
# Whether to offload TeaCache tensors to cpu to save a little bit of GPU memory.
teacache_offload    = False

# Skip some cfg steps in inference
# Recommended to be set between 0.00 and 0.25
cfg_skip_ratio      = 0

# Config and model path
config_path         = "config/wan2.1/wan_civitai.yaml"
# model path
model_name          = "models/Diffusion_Transformer/Wan2.1-I2V-14B-480P"
train_mode          = "adaln" #"controlnet" # "adaln"
# Choose the sampler in "Flow", "Flow_Unipc", "Flow_DPM++"
sampler_name        = "Flow_Unipc"
# [NOTE]: Noise schedule shift parameter. Affects temporal dynamics. 
# Used when the sampler is in "Flow_Unipc", "Flow_DPM++".
# If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
# If you want to generate a 720p video, it is recommended to set the shift value to 5.0.
shift               = 3 

# Load pretrained model if need
transformer_path    = "models/I2W/transformer"
vae_path            = None
lora_path           = "models/I2W/lora_diffusion_pytorch_model.safetensors"

# Other params
sample_size         = [352, 640]
video_length        = 81
fps                 = 16

# Use torch.float16 if GPU does not support torch.bfloat16
# ome graphics cards, such as v100, 2080ti, do not support torch.bfloat16
weight_dtype            = torch.bfloat16
# validation_image_start or video_path should be set one of them
validation_image_start  = "asset/street_night.jpg" #"asset/longshan-temple.jpg" #"asset/panda.jpg" #"asset/london.jpg" #"asset/camel.jpg" #"asset/cozy-living-rooms.jpg" #"asset/cliff.jpg"
# validation_image_start  = None

# prompts

# city street night
prompt = "First-person perspective walking down a lively city street at night. Neon signs and bright billboards glow on both sides, cars drive past with headlights and taillights streaking slightly. camera motion directly aligned with user actions, immersive urban night scene."

# temple
# prompt = "First-person perspective standing in front of an ornate traditional Chinese temple. The symmetrical facade features red lanterns, intricate carvings, and a curved tiled roof decorated with dragons. Bright daytime lighting, consistent environment, camera motion directly aligned with user actions, immersive and interactive exploration."

# panda
# prompt = "First-person perspective of standing close to a relaxed giant panda lying on its back in a bamboo forest. The panda looks peaceful and playful, surrounded by grass and bamboo stalks. Soft daylight, camera motion controlled by user actions, gentle camera motion, immersive POV, realistic scale, natural wildlife scene."

# camle
# prompt = "First-person perspective of standing in a rocky desert valley, looking at a camel a few meters ahead. The camel stands calmly on uneven stones, its long legs and single hump clearly visible. Bright midday sunlight, \
# dry air, muted earth tones, distant barren mountains. Natural handheld camera feeling, camera motion controlled by user actions, smooth movement, cinematic realism."

# london
# prompt = "First-person perspective walking through a narrow urban alley, old red brick industrial buildings on both sides, cobblestone street stretching forward with strong depth, metal walkways connecting buildings above, \
# overcast daylight, soft diffused lighting, cool and muted color tones, quiet and empty environment, no people, camera motion controlled by user actions, smooth movement, stable horizon, realistic scale and geometry, high realism, cinematic urban scene"

# cozy living room
# prompt = "First-person perspective inside a cozy living room, walking around a warm fireplace, soft carpet underfoot, furniture arranged neatly, bookshelves, plants, and warm table lamps on both sides, \
# warm indoor lighting, calm and quiet atmosphere, natural head-level camera movement, camera motion driven by user actions, realistic scale and depth, high realism, cinematic lighting, no people, no distortion."

# cliff
# prompt = "First-person perspective coastal exploration scene, walking along a cliffside stone path with wooden railings, green bushes lining the walkway, ocean to the left with gentle waves, distant islands visible under a clear sky, \
# realistic head-mounted camera view, smooth forward motion, stable horizon, natural human eye level, high realism, consistent environment, camera motion directly aligned with user actions, immersive and interactive exploration"

negative_prompt     = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
guidance_scale      = 6.0
seed                = 43
num_inference_steps = 30
lora_weight         = 1.0
save_path           = "samples/wan-videos-i2w"

# [w, s, a, d, shift, ctrl, _, mouse_y, mouse_x]
action_list = [[20, "0 1 0 1 0 0 0 0 0"], [40, "1 0 0 0 0 0 0 0 0"], [80, "0 0 1 0 0 0 0 0 0"], "40"]


device = "cuda"
config = OmegaConf.load(config_path)

keyboard_actions, mouse_actions = parse_action_list(action_list)
# add batch
keyboard_actions = keyboard_actions[None].to(device=device, dtype=weight_dtype)
mouse_actions = mouse_actions[None].to(device=device, dtype=weight_dtype)

if train_mode == 'controlnet':
    model_class = WanActionControlNetModel
elif train_mode == 'adaln':
    model_class = WanActionAdaLNModel
else:
    raise ValueError(f"Invalid train mode: {train_mode}")
print(f"model_class: {model_class}")
transformer = model_class.from_pretrained(
    transformer_path if transformer_path is not None else os.path.join(model_name, config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
    transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
)

# Get Vae
vae = AutoencoderKLWan.from_pretrained(
    os.path.join(model_name, config['vae_kwargs'].get('vae_subpath', 'vae')),
    additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
).to(weight_dtype)

if vae_path is not None:
    print(f"From checkpoint: {vae_path}")
    if vae_path.endswith("safetensors"):
        from safetensors.torch import load_file, safe_open
        state_dict = load_file(vae_path)
    else:
        state_dict = torch.load(vae_path, map_location="cpu")
    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

    m, u = vae.load_state_dict(state_dict, strict=False)
    print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

# Get Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    os.path.join(model_name, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
)

# Get Text encoder
text_encoder = WanT5EncoderModel.from_pretrained(
    os.path.join(model_name, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
    additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
)
text_encoder = text_encoder.eval()

# Get Clip Image Encoder
clip_image_encoder = CLIPModel.from_pretrained(
    os.path.join(model_name, config['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder')),
).to(weight_dtype)
clip_image_encoder = clip_image_encoder.eval()

# Get Scheduler
Choosen_Scheduler = scheduler_dict = {
    "Flow": FlowMatchEulerDiscreteScheduler,
    "Flow_Unipc": FlowUniPCMultistepScheduler,
    "Flow_DPM++": FlowDPMSolverMultistepScheduler,
}[sampler_name]
if sampler_name == "Flow_Unipc" or sampler_name == "Flow_DPM++":
    config['scheduler_kwargs']['shift'] = 1
scheduler = Choosen_Scheduler(
    **filter_kwargs(Choosen_Scheduler, OmegaConf.to_container(config['scheduler_kwargs']))
)

# Get Pipeline
pipeline = WanActionI2WPipeline(
    transformer=transformer,
    vae=vae,
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    scheduler=scheduler,
    clip_image_encoder=clip_image_encoder
)


if GPU_memory_mode == "sequential_cpu_offload":
    replace_parameters_by_name(transformer, ["modulation",], device=device)
    transformer.freqs = transformer.freqs.to(device=device)
    pipeline.enable_sequential_cpu_offload(device=device)
elif GPU_memory_mode == "model_cpu_offload":
    pipeline.enable_model_cpu_offload(device=device)
else:
    pipeline.to(device=device)

coefficients = get_teacache_coefficients(model_name) if enable_teacache else None
if coefficients is not None:
    print(f"Enable TeaCache with threshold {teacache_threshold} and skip the first {num_skip_start_steps} steps.")
    pipeline.transformer.enable_teacache(
        coefficients, num_inference_steps, teacache_threshold, num_skip_start_steps=num_skip_start_steps, offload=teacache_offload
    )

if cfg_skip_ratio is not None:
    print(f"Enable cfg_skip_ratio {cfg_skip_ratio}.")
    pipeline.transformer.enable_cfg_skip(cfg_skip_ratio, num_inference_steps)

generator = torch.Generator(device=device).manual_seed(seed)

if lora_path is not None:
    pipeline = merge_lora(pipeline, lora_path, lora_weight, device=device)

with torch.no_grad():
    video_length = int((video_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if video_length != 1 else 1
    latent_frames = (video_length - 1) // vae.config.temporal_compression_ratio + 1

    input_video, input_video_mask, clip_image = get_image_to_video_latent(validation_image_start, None, video_length=video_length, sample_size=sample_size)
    input_video_mask = torch.zeros_like(input_video[:, :1])
    input_video_mask[:, :, 1:] = 255
    clip_image = (input_video[0, :, 0]*255).to(torch.uint8).permute(1,2,0)
    clip_image = Image.fromarray(clip_image.cpu().numpy())

    sample = pipeline(
        prompt, 
        mouse_actions=mouse_actions,
        keyboard_actions=keyboard_actions,
        num_frames = video_length,
        negative_prompt = negative_prompt,
        height      = sample_size[0],
        width       = sample_size[1],
        generator   = generator,
        guidance_scale = guidance_scale,
        num_inference_steps = num_inference_steps,
        video      = input_video,
        mask_video   = input_video_mask,
        clip_image = clip_image,
        shift = shift,
    ).videos

if lora_path is not None:
    pipeline = unmerge_lora(pipeline, lora_path, lora_weight, device=device)

all_sample = torch.concat([sample, input_video], dim=-1)

def save_results():
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    # index = len([path for path in os.listdir(save_path)]) + 1
    # prefix = str(index).zfill(8)
    now = datetime.now()
    prefix = now.strftime("%Y%m%d-%H%M")
    if video_length == 1:
        video_path = os.path.join(save_path, prefix + ".png")

        image = sample[0, :, 0]
        image = image.transpose(0, 1).transpose(1, 2)
        image = (image * 255).numpy().astype(np.uint8)
        image = Image.fromarray(image)
        image.save(video_path)
    else:
        video_path = os.path.join(save_path, prefix + ".mp4")
        save_videos_grid(all_sample, video_path, fps=fps)
    print(f"Saved to {video_path}")

save_results()