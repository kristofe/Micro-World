# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Custom PyTorch Dataset for data captured in the data_5d_80x144 format.
# Reads PNG frames directly (no intermediate video encoding) and maps
# vehicle-style controls to the keyboard/mouse action format expected by
# the Micro-World training pipeline.
#
# Data layout:
#   <data_root>/
#     <capture_YYYYMMDDHHMMSSSSSS>/
#       Run_XXXXXX/
#         frame_0000.png ... frame_0299.png   (80×144 RGBA)
#         input.csv                            (no header: input_name,seconds,frame_number,value)
#         info.txt
#
# Action mapping (vehicle → keyboard+mouse):
#   steering < -thresh  → ad=1 (left)
#   steering >  thresh  → ad=2 (right)
#   else                → ad=0
#   throttle >  thresh  → ws=1 (forward)
#   else                → ws=0
#   camera_dx           → yaw_delta   (continuous, raw)
#   camera_dy           → pitch_delta (continuous, raw)
#   scs / collision / jump_invalid  → always 0

import os
import random

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

def keyboard_to_onehot(ws, ad, scs):
    """7-dim one-hot: [forward, backward, left, right, jump, sneak, sprint].
    Copied from dataset_game_video to avoid importing decord."""
    arr = [0] * 7
    if ws == 1:
        arr[0] = 1
    elif ws == 2:
        arr[1] = 1
    if ad == 1:
        arr[2] = 1
    elif ad == 2:
        arr[3] = 1
    if scs == 1:
        arr[4] = 1
    elif scs == 2:
        arr[5] = 1
    elif scs == 3:
        arr[6] = 1
    return arr


class MicroWorldDataset(Dataset):
    """Dataset that reads directly from data_5d_80x144-style captures.

    Each 300-frame run is split into non-overlapping clips of `clip_len`
    frames.  With clip_len=81 this yields 3 clips per run (frames 243-299
    are discarded).

    Returns the same dict structure as VideoGameDataset so it can be used
    as a drop-in replacement in the training script.
    """

    def __init__(
        self,
        data_root,
        clip_len=81,
        steering_thresh=0.1,
        throttle_thresh=0.1,
        mouse_scale=0.05,
        mouse_clamp=5.0,
        text_drop_ratio=0.1,
        prompt="",
    ):
        self.data_root = data_root
        self.clip_len = clip_len
        self.steering_thresh = steering_thresh
        self.throttle_thresh = throttle_thresh
        self.mouse_scale = mouse_scale
        self.mouse_clamp = mouse_clamp
        self.text_drop_ratio = text_drop_ratio
        self.prompt = prompt

        # Transforms matching VideoGameDataset (CenterCrop is identity for 80×144)
        self.video_transforms = transforms.Compose(
            [
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )

        # Build clip index
        self.clips = []
        self._action_cache = {}
        self._build_index()

        print(f"MicroWorldDataset: {len(self.clips)} clips from {data_root}")

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def _build_index(self):
        """Walk data_root and collect (run_path, start_frame, clip_name) tuples."""
        if not os.path.isdir(self.data_root):
            raise ValueError(f"data_root does not exist: {self.data_root}")

        for capture_name in sorted(os.listdir(self.data_root)):
            capture_path = os.path.join(self.data_root, capture_name)
            if not os.path.isdir(capture_path):
                continue
            for run_name in sorted(os.listdir(capture_path)):
                run_path = os.path.join(capture_path, run_name)
                if not os.path.isdir(run_path):
                    continue
                # Verify it looks like a run directory
                input_csv = os.path.join(run_path, "input.csv")
                if not os.path.isfile(input_csv):
                    continue
                # Count available frames
                n_frames = self._count_frames(run_path)
                n_clips = n_frames // self.clip_len
                for i in range(n_clips):
                    self.clips.append(
                        {
                            "run_path": run_path,
                            "start_frame": i * self.clip_len,
                            "name": f"{capture_name}__{run_name}__clip{i:02d}",
                        }
                    )

    @staticmethod
    def _count_frames(run_path):
        """Count frame_XXXX.png files in run_path."""
        count = 0
        for fname in os.listdir(run_path):
            if fname.startswith("frame_") and fname.endswith(".png"):
                count += 1
        return count

    # ------------------------------------------------------------------
    # Action CSV parsing
    # ------------------------------------------------------------------

    def _parse_actions(self, run_path):
        """Parse input.csv and return a dict {frame_int: {field: value}}.

        Cached per run_path to avoid re-reading when multiple clips come
        from the same run.
        """
        if run_path in self._action_cache:
            return self._action_cache[run_path]

        csv_path = os.path.join(run_path, "input.csv")
        df = pd.read_csv(
            csv_path,
            header=None,
            names=["input_name", "seconds", "frame_number", "value"],
        )
        df["frame_number"] = df["frame_number"].astype(int)

        # Pivot: rows = frames, columns = input_name
        pivoted = df.pivot_table(
            index="frame_number", columns="input_name", values="value", aggfunc="last"
        )

        result = {}
        for frame_num, row in pivoted.iterrows():
            result[int(frame_num)] = row.to_dict()

        self._action_cache[run_path] = result
        return result

    def _map_actions(self, frame_data):
        """Map one frame's vehicle inputs to (ws, ad, pitch_delta, yaw_delta)."""
        steering = frame_data.get("steering", 0.0) or 0.0
        throttle = frame_data.get("throttle", 0.0) or 0.0
        camera_dx = frame_data.get("camera_dx", 0.0) or 0.0
        camera_dy = frame_data.get("camera_dy", 0.0) or 0.0

        # Discretise to keyboard codes
        if steering < -self.steering_thresh:
            ad = 1  # left
        elif steering > self.steering_thresh:
            ad = 2  # right
        else:
            ad = 0

        ws = 1 if throttle > self.throttle_thresh else 0

        # Scale raw pixel displacements then clamp to remove rare spikes
        c = self.mouse_clamp
        pitch_delta = max(-c, min(c, float(camera_dy) * self.mouse_scale))
        yaw_delta = max(-c, min(c, float(camera_dx) * self.mouse_scale))

        return ws, ad, pitch_delta, yaw_delta

    # ------------------------------------------------------------------
    # Frame loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_frame_rgb(run_path, frame_idx):
        """Load a single RGBA PNG and return an RGB uint8 numpy array [H,W,3]."""
        fname = f"frame_{frame_idx:04d}.png"
        fpath = os.path.join(run_path, fname)
        img = Image.open(fpath).convert("RGB")
        return np.array(img, dtype=np.uint8)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        while True:
            try:
                sample = self._get_clip(idx)
                break
            except Exception as e:
                print(f"[MicroWorldDataset] Error loading clip {idx}: {e}")
                idx = random.randint(0, len(self.clips) - 1)
        return sample

    def _get_clip(self, idx):
        clip_info = self.clips[idx]
        run_path = clip_info["run_path"]
        start_frame = clip_info["start_frame"]
        clip_name = clip_info["name"]

        # ---- Load frames ----
        frames = []
        for f in range(start_frame, start_frame + self.clip_len):
            frame_arr = self._load_frame_rgb(run_path, f)
            frames.append(frame_arr)

        # Stack → [F, H, W, 3] → [F, 3, H, W], float [0,1]
        pixel_values = torch.from_numpy(np.stack(frames, axis=0))  # [F,H,W,3]
        pixel_values = pixel_values.permute(0, 3, 1, 2).contiguous().float() / 255.0
        pixel_values = self.video_transforms(pixel_values)

        # ---- Load actions ----
        action_dict = self._parse_actions(run_path)

        mouse_actions = []
        keyboard_actions = []
        for f in range(start_frame, start_frame + self.clip_len):
            frame_data = action_dict.get(f, {})
            ws, ad, pitch_delta, yaw_delta = self._map_actions(frame_data)
            scs = 0  # no jump/sneak/sprint in vehicle data
            mouse_actions.append([pitch_delta, yaw_delta])
            keyboard_actions.append(keyboard_to_onehot(ws, ad, scs))

        mouse_actions = torch.tensor(mouse_actions, dtype=torch.float32)     # [F, 2]
        keyboard_actions = torch.tensor(keyboard_actions, dtype=torch.float32)  # [F, 7]

        # ---- Text prompt ----
        text = "" if random.random() < self.text_drop_ratio else self.prompt

        return {
            "pixel_values": pixel_values,
            "text": text,
            "data_type": "video",
            "name": clip_name,
            "idx": idx,
            "mouse_actions": mouse_actions,
            "keyboard_actions": keyboard_actions,
        }
