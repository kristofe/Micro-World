<div align="center">
  <br>
  <br>
  <h1>AMD Micro-World: action-controlled Interactive world model</h1>
</div>

# Table of Contents
- [Introduction](#introduction)
- [Quick Start](#quick-start)
- [Video Result](#video-result)
- [Acknowledgement](#acknowledgement)
- [License](#license)

# Introduction
Micro-World is a series of action-controlled interactive world models developed by the AMD AIG team and trained on AMD Instinct™ MI250/MI325 GPUs. It includes both text-to-video and image-to-video variants, enabling a wide range of application scenarios. Built on Wan as the base model, Micro-World is trained on a MineCraft dataset and is designed to generate high-quality, open-domain visual environments.

# Quick Start
## Installation
Our model is build on AMD GPU and ROCm enviroment.

### a. From docker
We strongly recommend docker enviroment.

```
# build image
docker build -t microworld:latest .

# enter image
docker run -it --rm --name=agent --network=host \
  --device=/dev/kfd --device=/dev/dri --group-add=video --group-add=render \
  --ipc=host \
  -e NCCL_IB_DISABLE=1 \
  -e NCCL_SOCKET_IFNAME=lo \
  -e NCCL_SOCKET_FAMILY=AF_INET \
  -e NCCL_DEBUG=WARN \
  -e RCCL_MSCCLPP_ENABLE=0 \
  -e RCCL_MSCCL_ENABLE=0 \
  -e NCCL_MIN_NCHANNELS=16 -e NCCL_MAX_NCHANNELS=32 \
  microworld:latest
```

### b. Conda
```
conda create -n AMD_microworld python=3.12
conda activate AMD_microworld

# Install torch/torchvision
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/rocm6.4

# Important: Install ROCm flash-attn by offical guideline https://github.com/Dao-AILab/flash-attention
pip install -r requirements.txt
```

## Inference:

- **Step 1**: Download the corresponding model weights and place them in the `models` folder.
```
# Download T2W model weights:
hf download amd/Micro-World-T2W --local-dir models/T2W

# Download I2W model weights:
hf download amd/Micro-World-I2W --local-dir models/I2W
```
- **Step 2**: We provide action-controlled model inference scripts under the `examples` folder.
  - Modify config in the script, such as `transformer_path`, `lora_path`, `prompt`, `neg_prompt`, `GPU_memory_mode`, `validation_image_start`, `action_list`.
    - `validation_image_start` is the reference image path of image-to-video generation.
    - `action_list` follows formation of [[{end frame}, "w, s, a, d, shift, ctrl, _, mouse_y, mouse_x"],...,"{space frames}"]. For example, [[10, "0 1 0 0 0 0 0 0 0"], [80, "0 0 0 1 0 0 0 0 0"], "30 65"] indicates press s from frame 0 to frame 10, press d from frame 10 to frame 80, and press space at frame 30 and 65.
  - For example, you can run T2W action controled model inference using following command:
```
python examples/wan2.1/predict_t2w_action_control.py
```

## Training
We have provided our collected minecraft action controlled dataset, the game lora and action controlled model weights. 
- **Step 1**: Download your target *base model* weights and place them in the `models` folder.
```
# Download Wan2.1 T2V model weights:
hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir models/Diffusion_Transformer/Wan2.1-T2V-1.3B

# Download pretrained model weights and lora
hf download amd/Micro-World-T2W --local-dir models/T2W

# Download Wan2.1 I2V model weights:
hf download Wan-AI/Wan2.1-I2V-14B-480P --local-dir models/Diffusion_Transformer/Wan2.1-I2V-14B-480P

# Download pretrained model weights and lora
hf download amd/Micro-World-I2W --local-dir models/I2W

# Download train dataset:
hf download amd/Micro-World-MC-Dataset --local-dir datasets --repo-type=dataset
```
- **Step 2**: We provide action-controlled model training scripts under the `scripts` folder.
  - Modify config in the bash script.
  - For example, you can run T2W action controled model training using following command:
```
bash scripts/wan2.1/train_game_action_t2w.sh
```

## Evaluation
We follow the evaluation protocol from [Matrix-Game](https://github.com/SkyworkAI/Matrix-Game) to assess image quality, action controllability, and temporal stability.

- **Step 1** Download the IDM model and weights from [VPT](https://github.com/openai/Video-Pre-Training), and place them under `test_metrics/idm_model`.
- **Step 2** Clone [GameWorldScore](https://github.com/SkyworkAI/Matrix-Game/tree/main/Matrix-Game-1/GameWorldScore) and move the folder into `test_metrics/GameWorldScore`.
- **Step 3** Run the evaluation script.

```
cd test_metrics
python evaluate.py \
        --model idm_moddel/4x_idm.model \
        --weights idm_moddel/4x_idm.weights \
        --visual_metrics \ # evaluate image quality(PSNR, LPIPS, FVD)
        --control_metrics \ # evaluate action Controllability 
        --temporal_metrics \ # evaluate temporal stability
        --infer-demo-num 0 --n-frames 81 \
        --video-path your_testing_output_path \
        --ref-path dataset_val/video \
        --output-file "metrics_log/ground_truth/idm_res_pred.json" \
        --json-path dataset_val/metadata-detection/
```

# Video Result
## T2W Model
### In Domain
<table border="0" style="width: 100%; table-layout: fixed; text-align: center; margin-top: 20px;">
  <tr>
      <td style="vertical-align: top; width: 33%;">
          <video src="https://github.com/user-attachments/assets/01ecff57-5fc8-40c0-b7c1-1c72525b598c" width="100%" controls autoplay loop></video>
          <div style="margin-top: 8px; overflow:hidden; font-size: 14px;">
        W
        </div>
      </td>
       <td style="vertical-align: top; width: 33%;">
          <video src="https://github.com/user-attachments/assets/0156af1f-5fe2-4276-9cec-ba97b2476018" width="100%" controls autoplay loop></video>
          <div style="margin-top: 8px; font-size: 14px;">
        S
        </div>
     </td>
      <td style="vertical-align: top; width: 33%;">
          <video src="https://github.com/user-attachments/assets/d27268e5-9fbc-49f7-b3ca-882fb58f21b6" width="100%" controls autoplay loop></video>
          <div style="margin-top: 8px; font-size: 14px;">
        A
        </div>
      </td>
  </tr>
</table>

<table border="0" style="width: 100%; table-layout: fixed; text-align: center; margin-top: 20px;">
  <tr>
     <td style="vertical-align: top; width: 33%;">
          <video src="https://github.com/user-attachments/assets/aff52ef1-0c9c-4a03-961f-6aa5361b636d" width="100%" controls autoplay loop></video>
          <div style="margin-top: 8px; font-size: 14px;">
        D
        </div>
     </td>
      <td style="vertical-align: top; width: 33%;">
          <video src="https://github.com/user-attachments/assets/b5d37d89-5cf0-40a5-8504-61f68e944fb9" width="100%" controls autoplay loop></video>
          <div style="margin-top: 8px; font-size: 14px;">
        W+Ctrl
        </div>
      </td>
      <td style="vertical-align: top; width: 33%;">
          <video src="https://github.com/user-attachments/assets/1b0d50c8-a037-4671-a146-b77672260322" width="100%" controls autoplay loop></video>
          <div style="margin-top: 8px; font-size: 14px;">
        W+Shift
        </div>
     </td>
  </tr>
</table>

<table border="0" style="width: 100%; table-layout: fixed; text-align: center; margin-top: 20px;">
  <tr>
    <td style="vertical-align: top; width: 33%;">
          <video src="https://github.com/user-attachments/assets/b13a14d7-5882-42dd-872b-8f61b9ab7060" width="100%" controls autoplay loop></video>
          <div style="margin-top: 8px; font-size: 14px;">
        Multiple control
        </div>
     </td>
     <td style="vertical-align: top; width: 33%;">
          <video src="https://github.com/user-attachments/assets/1218bbda-7993-4075-881b-2e16002acda8" width="100%" controls autoplay loop></video>
          <div style="margin-top: 8px; font-size: 14px;">
        Mouse down and up
        </div>
     </td>
      <td style="vertical-align: top; width: 33%;">
          <video src="https://github.com/user-attachments/assets/31471313-d94b-4936-b23b-12e7f89fda87" width="100%" controls autoplay loop></video>
          <div style="margin-top: 8px; font-size: 14px;">
        Mouse right and left
        </div>
      </td>
  </tr>
</table>

### Open Domain
<table border="0" style="width: 100%; table-layout: fixed; text-align: center; margin-top: 20px;">
  <tr>
      <td style="vertical-align: top; width: 33%;">
          <video src="https://github.com/user-attachments/assets/25ec4ba8-4f65-4b26-8966-13437647f240" width="100%" controls autoplay loop></video>
          <div style="margin-top: 8px; text-align: left;">
            <details>
              <summary style="cursor: pointer; font-size: 13px;">View Prompt</summary>
              <div style="font-size: 12px; margin-top: 5px; color: #555;">
                A cozy living room with sunlight streaming through window, vintage furniture, soft shadows.
              </div>
            </details>
          </div>        
      </td>
      <td style="vertical-align: top; width: 33%;">
          <video src="https://github.com/user-attachments/assets/a92149a8-6c4d-4b9a-8ada-81b47b4c81e7" width="100%" controls autoplay loop></video>
        <div style="margin-top: 8px; text-align: left;">
            <details>
              <summary style="cursor: pointer; font-size: 13px;">View Prompt</summary>
              <div style="font-size: 12px; margin-top: 5px; color: #555;">
                A cozy living room with sunlight streaming through window, vintage furniture, soft shadows.
              </div>
            </details>
          </div>  
      </td>
       <td style="vertical-align: top; width: 33%;">
          <video src="https://github.com/user-attachments/assets/67b35842-04fd-4a0f-9a5c-6914d9f77e66" width="100%" controls autoplay loop></video>
          <div style="margin-top: 8px; text-align: left;">
            <details>
              <summary style="cursor: pointer; font-size: 13px;">View Prompt</summary>
              <div style="font-size: 12px; margin-top: 5px; color: #555;">
                Running along a cliffside path in a tropical island in first person perspective, with turquoise waters crashing against the rocks far below, the salty scent of the ocean carried by the breeze, and the sound of distant waves blending with the calls of seagulls as the path twists and turns along the jagged cliffs.
              </div>
            </details>
          </div>  
     </td>
  </tr>
</table>

<table border="0" style="width: 100%; table-layout: fixed; text-align: center; margin-top: 20px;">
  <tr>
      <td style="vertical-align: top; width: 33%;">
          <video src="https://github.com/user-attachments/assets/d4a46b8b-022d-4fca-964f-c1d477111f4e" width="100%" controls autoplay loop></video>
          <div style="margin-top: 8px; text-align: left;">
            <details>
              <summary style="cursor: pointer; font-size: 13px;">View Prompt</summary>
              <div style="font-size: 12px; margin-top: 5px; color: #555;">
                A young bear stands next to a large tree in a grassy meadow, its dark fur catching the soft daylight. The bear seems poised, observing its surroundings in a tranquil landscape, with rolling hills and sparse trees dotting the background under a pale blue sky.
              </div>
            </details>
          </div>  
     </td>
     <td style="vertical-align: top; width: 33%;">
          <video src="https://github.com/user-attachments/assets/b6f77a1a-58ce-43db-b6c5-efe09b7a9142" width="100%" controls autoplay loop></video>
          <div style="margin-top: 8px; text-align: left;">
            <details>
              <summary style="cursor: pointer; font-size: 13px;">View Prompt</summary>
              <div style="font-size: 12px; margin-top: 5px; color: #555;">
                A giant panda rests peacefully under a blooming cherry blossom tree, its black and white fur contrasting beautifully with the delicate pink petals. The ground is lightly sprinkled with fallen blossoms, and the tranquil setting is framed by the soft hues of the blossoms and the grassy field surrounding the tree.
              </div>
            </details>
          </div>  
     </td>      
     <td style="vertical-align: top; width: 33%;">
          <video src="https://github.com/user-attachments/assets/c9225344-8b0b-4249-ab77-c8e5c4dddacc" width="100%" controls autoplay loop></video>
          <div style="margin-top: 8px; text-align: left;">
            <details>
              <summary style="cursor: pointer; font-size: 13px;">View Prompt</summary>
              <div style="font-size: 12px; margin-top: 5px; color: #555;">
                Exploring an ancient jungle ruin in first person perspective surrounded by towering stone statues covered in moss and vines.
              </div>
            </details>
          </div> 
     </td>      
  </tr>
</table>

## I2W Model
We observe that fully decoupling the action module from game-specific styles in large-scale models remains challenging. As a result, we apply both the LoRA weights and the action module during inference for the I2W results.
<table border="0" style="width: 100%; table-layout: fixed; text-align: center; margin-top: 20px;">
  <tr>
      <td style="vertical-align: top; width: 50%;">
          <video src="https://github.com/user-attachments/assets/f135d5c9-0379-4ace-bf22-1671cef261af" width="100%" controls autoplay loop></video>
          <div style="margin-top: 8px; text-align: left;">
            <details>
              <summary style="cursor: pointer; font-size: 13px;">View Prompt</summary>
              <div style="font-size: 12px; margin-top: 5px; color: #555;">
                First-person perspective walking down a lively city street at night. Neon signs and bright billboards glow on both sides, cars drive past with headlights and taillights streaking slightly. camera motion directly aligned with user actions, immersive urban night scene.
              </div>
            </details>
          </div>        
      </td>
      <td style="vertical-align: top; width: 50%;">
          <video src="https://github.com/user-attachments/assets/2088d2da-95a6-4908-b7a2-f60458281b5e" width="100%" controls autoplay loop></video>
        <div style="margin-top: 8px; text-align: left;">
            <details>
              <summary style="cursor: pointer; font-size: 13px;">View Prompt</summary>
              <div style="font-size: 12px; margin-top: 5px; color: #555;">
                First-person perspective standing in front of an ornate traditional Chinese temple. The symmetrical facade features red lanterns, intricate carvings, and a curved tiled roof decorated with dragons. Bright daytime lighting, consistent environment, camera motion directly aligned with user actions, immersive and interactive exploration.
              </div>
            </details>
          </div>  
      </td>
  </tr>
</table>

<table border="0" style="width: 100%; table-layout: fixed; text-align: center; margin-top: 20px;">
  <tr>
       <td style="vertical-align: top; width: 50%;">
          <video src="https://github.com/user-attachments/assets/9e1185cc-5480-4059-8643-7b6e08fff0c1" width="100%" controls autoplay loop></video>
          <div style="margin-top: 8px; text-align: left;">
            <details>
              <summary style="cursor: pointer; font-size: 13px;">View Prompt</summary>
              <div style="font-size: 12px; margin-top: 5px; color: #555;">
                First-person perspective of standing in a rocky desert valley, looking at a camel a few meters ahead. The camel stands calmly on uneven stones, its long legs and single hump clearly visible. Bright midday sunlight, dry air, muted earth tones, distant barren mountains. Natural handheld camera feeling, camera motion controlled by user actions, smooth movement, cinematic realism.
              </div>
            </details>
          </div>  
     </td>
     <td style="vertical-align: top; width: 50%;">
          <video src="https://github.com/user-attachments/assets/c75d7344-7016-494e-be00-103d28e43738" width="100%" controls autoplay loop></video>
          <div style="margin-top: 8px; text-align: left;">
            <details>
              <summary style="cursor: pointer; font-size: 13px;">View Prompt</summary>
              <div style="font-size: 12px; margin-top: 5px; color: #555;">
                First-person perspective walking through a narrow urban alley, old red brick industrial buildings on both sides, cobblestone street stretching forward with strong depth, metal walkways connecting buildings above, overcast daylight, soft diffused lighting, cool and muted color tones, quiet and empty environment, no people, camera motion controlled by user actions, smooth movement, stable horizon, realistic scale and geometry, high realism, cinematic urban scene.
              </div>
            </details>
          </div>  
     </td>
  </tr>
</table>

<table border="0" style="width: 100%; table-layout: fixed; text-align: center; margin-top: 20px;">
  <tr>
     <td style="vertical-align: top; width: 50%;">
          <video src="https://github.com/user-attachments/assets/f6da97af-0d3a-4b6a-b80f-5ae3c03ccbf6" width="100%" controls autoplay loop></video>
          <div style="margin-top: 8px; text-align: left;">
            <details>
              <summary style="cursor: pointer; font-size: 13px;">View Prompt</summary>
              <div style="font-size: 12px; margin-top: 5px; color: #555;">
                First-person perspective coastal exploration scene, walking along a cliffside stone path with wooden railings, green bushes lining the walkway, ocean to the left with gentle waves, distant islands visible under a clear sky, realistic head-mounted camera view, smooth forward motion, stable horizon, natural human eye level, high realism, consistent environment, camera motion directly aligned with user actions, immersive and interactive exploration.
              </div>
            </details>
          </div>  
     </td>      
     <td style="vertical-align: top; width: 50%;">
          <video src="https://github.com/user-attachments/assets/b76a8aca-d1da-47ba-88e9-3da36f64429d" width="100%" controls autoplay loop></video>
          <div style="margin-top: 8px; text-align: left;">
            <details>
              <summary style="cursor: pointer; font-size: 13px;">View Prompt</summary>
              <div style="font-size: 12px; margin-top: 5px; color: #555;">
                First-person perspective inside a cozy living room, walking around a warm fireplace, soft carpet underfoot, furniture arranged neatly, bookshelves, plants, and warm table lamps on both sides, warm indoor lighting, calm and quiet atmosphere, natural head-level camera movement, camera motion driven by user actions, realistic scale and depth, high realism, cinematic lighting, no people, no distortion.
              </div>
            </details>
          </div> 
     </td>      
  </tr>
</table>

## Acknowledgement
Our codebase is built upon [Wan2.1](https://github.com/Wan-Video/Wan2.1/), [VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun). We sincerely thank the authors for open-sourcing their excellent codebases. 

Our datasets are collected using [MineDojo](https://github.com/MineDojo/MineDojo) and captioned with [miniCPM-V](https://github.com/OpenBMB/MiniCPM-V). We also extend our appreciation to the respective teams for their high-quality tools and contributions.
## License
This project is licensed under the [Apache License (Version 2.0)](https://github.com/modelscope/modelscope/blob/master/LICENSE).
