# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.
import os
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.io import read_video
import tqdm
# --- Robust imports for torchmetrics classes across versions ---
# SSIM / PSNR
try:
    from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
except Exception:
    from torchmetrics import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

# LPIPS
try:
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
except Exception:
    try:
        # sometimes named LPIPS
        from torchmetrics.image.lpip import LPIPS as LearnedPerceptualImagePatchSimilarity
    except Exception:
        from torchmetrics import LearnedPerceptualImagePatchSimilarity  # last resort

# FVD
from GameWorldScore.GameWorld.third_party.fvd.calculate_fvd import calculate_fvd


def _device_auto() -> torch.device:
    """Auto-pick device. Works for CUDA and ROCm."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def _read_video_tensor(path: str,
                       size: Optional[Tuple[int, int]] = None,
                       to_float01: bool = True) -> torch.Tensor:
    """
    Read a video using torchvision.io.read_video.
    Returns a float tensor of shape [T, C, H, W].
    - If to_float01=True, normalize to [0,1].
    - If size is provided, resize spatially to (H, W).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Video not found: {path}")

    # video: [T, H, W, C] (uint8)
    video, _, _ = read_video(path, pts_unit="sec")
    if video.numel() == 0:
        raise ValueError(f"Empty or unreadable video: {path}")

    # -> [T, C, H, W], float
    frames = video.permute(0, 3, 1, 2).contiguous()
    frames = frames.float()
    if to_float01:
        frames = frames / 255.0

    if size is not None:
        # size is (H, W)
        frames = F.interpolate(frames, size=size, mode="bilinear", align_corners=False)

    return frames  # [T, C, H, W], float


def _uniform_sample_to_num_frames(frames: torch.Tensor,
                                  num_frames: int) -> torch.Tensor:
    """
    Uniformly sample a clip to exactly `num_frames`, padding by repeating last frame if too short.
    Input: [T, C, H, W]
    Output: [num_frames, C, H, W]
    """
    T = frames.shape[0]
    if T == 0:
        raise ValueError("Video has 0 frames after reading.")
    if T >= num_frames:
        idx = torch.linspace(0, T - 1, steps=num_frames).round().long()
        return frames.index_select(0, idx)
    else:
        pad = num_frames - T
        return torch.cat([frames, frames[-1:].repeat(pad, 1, 1, 1)], dim=0)


def _batch_iter(total: int, batch_size: int):
    for i in range(0, total, batch_size):
        yield i, min(i + batch_size, total)


def compute_visual_metrics(
    video_list: List[str],
    ref_list: Optional[List[str]] = None,
    *,
    # processing params
    size_for_frame_metrics: Tuple[int, int] = (256, 256),
    batch_size_frames: int = 16,
    lpips_net: str = "vgg",  # "vgg" | "alex" | "squeeze"
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Compute SSIM, PSNR, LPIPS (framewise) and FVD (setwise).
    - SSIM/PSNR/LPIPS: require ref_list with same length/order as video_list.
    - FVD: requires fvd_ref_list (the "real/reference" set). We compute FVD(video_list, fvd_ref_list).

    Returns a dict:
    {
      "ssim": float or None,
      "psnr": float or None,
      "lpips": float or None,
      "fvd": float or None,
      "details": {...}
    }
    """
    device = device or _device_auto()

    results: Dict[str, Any] = {
        "ssim": None,
        "psnr": None,
        "lpips": None,
        "fvd": None,
        "details": {},
    }

    # -----------------------------
    # Pairwise frame metrics (SSIM / PSNR / LPIPS)
    # -----------------------------
    if ref_list is not None:
        if len(video_list) != len(ref_list):
            raise ValueError("For framewise metrics, video_list and ref_list must have the same length.")
        all_v_frames, all_r_frames = [], []
        ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
        lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type=lpips_net).to(device)
        fvd_list = []
        print("Compute PSNR, SSIM, LPIPS, FVD...")
        for v_path, r_path in tqdm.tqdm(zip(video_list, ref_list), total=len(video_list), desc="Processing videos"):
            # Read & resize frames to shared spatial size
            v_frames = _read_video_tensor(v_path, size=size_for_frame_metrics, to_float01=True)  # [T, C, H, W]
            r_frames = _read_video_tensor(r_path, size=size_for_frame_metrics, to_float01=True)
            all_v_frames.append(v_frames)
            all_r_frames.append(r_frames)
            # Align in time (truncate to min length)
            T = min(v_frames.shape[0], r_frames.shape[0])
            if T == 0:
                continue
            v_frames = v_frames[:T]
            r_frames = r_frames[:T]

            for i, j in _batch_iter(T, batch_size_frames):
                v_b = v_frames[i:j].to(device)            # [B, C, H, W], in [0,1]
                r_b = r_frames[i:j].to(device)            # [B, C, H, W], in [0,1]

                # Global accumulators
                ssim_metric.update(v_b, r_b)
                psnr_metric.update(v_b, r_b)
                lpips_metric.update(v_b, r_b)  # LPIPS expects [-1,1]
        all_v_frames = torch.stack(all_v_frames, 0)
        all_r_frames = torch.stack(all_r_frames, 0)
        v_length = all_v_frames.shape[1]
        r_length = all_r_frames.shape[1]
        vr_length = min(v_length, r_length)
        all_r_frames = all_r_frames[:,:vr_length]
        all_v_frames = all_v_frames[:,:vr_length]
        curr_fvd = calculate_fvd(all_v_frames, all_r_frames, device = device, method='styleganv',only_final=False)
        results["ssim"] = ssim_metric.compute().item()
        results["psnr"] = psnr_metric.compute().item()
        results["lpips"] = lpips_metric.compute().item()
        results["fvd"] =  np.mean(list(curr_fvd['value']))

    return results
