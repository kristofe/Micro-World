FROM rocm/pytorch:latest

RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y sudo \
    && apt-get install -y libheif-dev ffmpeg openssh-server parallel rclone zsh curl git locales

RUN pip install diffusers==0.34.0 opencv-python decord albumentations func_timeout comet_ml imageio imageio-ffmpeg einops ninja ipython omegaconf transformers accelerate
