# Training with Driving Data (`data_5d_80x144`)

This guide covers how to train Micro-World on the custom driving simulation dataset
captured in the `data_5d_80x144` format. A custom `MicroWorldDataset` class reads
the PNG frames directly — no intermediate video conversion needed.

---

## Data Format

```
data_5d_80x144/
  <capture_YYYYMMDDHHMMSSSSSS>/         # timestamped capture session
    Run_000001/
      frame_0000.png ... frame_0299.png  # 300 frames, 144×80 RGBA, 25 FPS
      input.csv                          # per-frame vehicle inputs (no header)
      info.txt                           # FPS: 25 / NumFrame: 300
    Run_000002/
      ...
  <capture_YYYYMMDDHHMMSSSSSS>/
    ...
```

### `input.csv` fields

| Column | Description |
|--------|-------------|
| `input_name` | One of: `steering`, `throttle`, `camera_dx`, `camera_dy`, `camera_drot` |
| `seconds` | Timestamp in seconds (float) |
| `frame_number` | Frame index 0–299 |
| `value` | Numeric value for that input at that frame |

Five rows per frame (one per input type), 300 frames × 5 = 1,500 rows per file.

### Action mapping to Micro-World format

| Driving input | Micro-World action | Notes |
|---------------|--------------------|-------|
| `steering < -0.1` | `ad = 1` (left key) | threshold configurable |
| `steering > 0.1` | `ad = 2` (right key) | |
| `throttle > 0.1` | `ws = 1` (forward key) | threshold configurable |
| `camera_dx` | `yaw_delta` | scaled × 0.05, clamped ±5 |
| `camera_dy` | `pitch_delta` | scaled × 0.05, clamped ±5 |
| `camera_drot` | *(ignored)* | redundant with camera_dx |
| — | `scs = 0` | no jump/sneak/sprint |

Each 300-frame run is split into **3 non-overlapping 81-frame clips**
(frames 0–80, 81–161, 162–242). The trailing 57 frames are discarded.

---

## Setup

### 1. Install dependencies

Inside the container, install requirements while preserving the existing
`torch 2.9.0+cu130` installation:

```bash
pip install \
  "accelerate>=0.25.0" \
  "diffusers==0.34.0" \
  "transformers>=4.46.2,<5.0.0" \
  einops safetensors omegaconf sentencepiece \
  func_timeout \
  "imageio[ffmpeg]" "imageio[pyav]" \
  tensorboard tomesd torchdiffeq torchsde \
  scikit-image opencv-python-headless albumentations \
  ftfy beautifulsoup4 timm pandas deepspeed
```

> **Note:** Use `opencv-python-headless` (not `opencv-python`) — the full build
> requires `libGL.so.1` which is not present in the container.

> **Note:** Pin `transformers<5.0.0` — `diffusers==0.34.0` depends on
> `FLAX_WEIGHTS_NAME` which was removed in transformers 5.x.

### 2. Download model weights

```bash
# Base diffusion model (~14 GB)
hf download Wan-AI/Wan2.1-T2V-1.3B \
  --local-dir models/Diffusion_Transformer/Wan2.1-T2V-1.3B

# Pretrained action LoRA weights
hf download amd/Micro-World-T2W \
  --local-dir models/T2W
```

Expected layout after download:

```
models/
  Diffusion_Transformer/
    Wan2.1-T2V-1.3B/
      diffusion_pytorch_model.safetensors
      Wan2.1_VAE.pth
      models_t5_umt5-xxl-enc-bf16.pth
      config.json
      ...
  T2W/
    lora_diffusion_pytorch_model.safetensors
    transformer/
    ...
```

---

## Training

### Single GPU

```bash
accelerate launch --num_processes 1 \
  scripts/wan2.1/train_game_action_t2w.py \
  --config_path="config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path=models/Diffusion_Transformer/Wan2.1-T2V-1.3B \
  --train_data_dir=../data_5d_80x144 \
  --dataset_type microworld \
  --video_sample_size 80 144 \
  --video_sample_stride=1 \
  --video_sample_n_frames=81 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --training_with_video_token_length \
  --dataloader_num_workers=4 \
  --num_train_epochs=100 \
  --checkpointing_steps=500 \
  --learning_rate=1e-05 \
  --lr_scheduler="cosine_with_restarts" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=1e-1 \
  --adam_epsilon=1e-8 \
  --vae_mini_batch=64 \
  --max_grad_norm=0.05 \
  --network_alpha 12 \
  --tracker_project_name "MicroWorld-Driving" \
  --checkpoints_total_limit=5 \
  --trainable_modules "action" \
  --lora_skip_name "action" \
  --validation_steps=500 \
  --motion_sub_loss \
  --lora_path "models/T2W/lora_diffusion_pytorch_model.safetensors" \
  --train_mode "adaln"
```

### Multi-GPU (DGX / multi-node)

Replace the `accelerate launch` prefix with the cluster launcher from
`scripts/wan2.1/train_game_action_t2w.sh`, swapping the dataset variables:

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-1.3B"
export DATASET_NAME="../data_5d_80x144"

accelerate launch --num_processes $((GPUS_PER_NODE*WORLD_SIZE)) \
  --num_machines $WORLD_SIZE --machine_rank=$RANK \
  --mixed_precision="bf16" \
  --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT \
  --use_deepspeed \
  --deepspeed_config_file config/zero_stage2_config.json \
  --deepspeed_multinode_launcher standard \
  scripts/wan2.1/train_game_action_t2w.py \
    --dataset_type microworld \
    --train_data_dir=$DATASET_NAME \
    --video_sample_size 80 144 \
    ... (rest of args as above, omit --train_data_meta)
```

---

## Key Arguments

| Argument | Value | Notes |
|----------|-------|-------|
| `--dataset_type` | `microworld` | Selects `MicroWorldDataset` instead of `VideoGameDataset` |
| `--train_data_dir` | path to `data_5d_80x144/` | Root of capture directories |
| `--video_sample_size` | `80 144` | Height × Width — matches source frame size exactly |
| `--video_sample_n_frames` | `81` | Must satisfy 4n+1; 81 frames = 3.24 s at 25 FPS |
| `--microworld_steering_thresh` | `0.1` | Steering dead-zone before mapping to A/D key |
| `--microworld_throttle_thresh` | `0.1` | Throttle threshold before mapping to W key |
| `--microworld_mouse_scale` | `0.05` | Scales raw `camera_dx/dy` pixel values to model range |
| `--microworld_mouse_clamp` | `5.0` | Clips scaled mouse deltas to suppress rare spikes |
| `--microworld_prompt` | `""` | Text caption used for all clips (empty = unconditional) |

---

## Dataset Statistics

With the default `clip_len=81`:

| Metric | Value |
|--------|-------|
| Captures | 6 |
| Runs per capture | 132–151 |
| Total runs | ~837 |
| Clips per run | 3 |
| **Total training clips** | **~2,478** |
| Frame size | 144×80 RGBA → RGB |
| Framerate | 25 FPS |
| Clip duration | 3.24 s |

---

## Monitoring

TensorBoard logs are written to `output_dir/<timestamp>/logs/`:

```bash
tensorboard --logdir output_dir/
```

Sanity-check GIFs (rendered at startup) are saved to
`output_dir/<timestamp>/sanity_check/`.

---

## Implementation Notes

- **No video conversion.** `MicroWorldDataset` reads PNGs directly on each
  `__getitem__` call. Action CSVs are cached in memory per run to avoid
  re-reading the same file for the 3 clips it produces.
- **VAE compatibility.** 80÷8 = 10, 144÷8 = 18 — both whole numbers, so the
  spatial compression in the VAE works without padding.
- **Temporal constraint.** The model requires frame counts of the form 4n+1.
  81 = 4×20+1 ✓.
- **Source file:** [`microworld/data/dataset_microworld.py`](microworld/data/dataset_microworld.py)
