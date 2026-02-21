# Modifications Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.

# This file is a modified version of code from VideoXFun
# (https://github.com/aigc-apps/VideoX-Fun),
# which is licensed under the Apache License, Version 2.0.

import csv
import gc
import json
import os
import random
from contextlib import contextmanager

import torch
import torchvision.transforms as transforms
try:
    from decord import VideoReader
except ImportError:
    VideoReader = None  # decord unavailable; VideoGameDataset will fail at runtime if used
from func_timeout import FunctionTimedOut, func_timeout
from torch.utils.data import BatchSampler, Sampler
from torch.utils.data.dataset import Dataset

VIDEO_READER_TIMEOUT = 20


def keyboard_to_onehot(ws, ad, scs):
    arr = [0] * 7
    if ws == 1:
        arr[0] = 1  # forward
    elif ws == 2:
        arr[1] = 1  # backward
    if ad == 1:
        arr[2] = 1  # left
    elif ad == 2:
        arr[3] = 1  # right
    if scs == 1:
        arr[4] = 1  # jump
    elif scs == 2:
        arr[5] = 1  # sneak
    elif scs == 3:
        arr[6] = 1  # sprint
    return arr

def process_collisions(action):
    if action["collision"] == 1:
        action["ws"] = 0
        action["ad"] = 0
    if action["jump_invalid"] == 1:
        action["scs"] = 0
    return action

class GameVideoSampler(BatchSampler):
    """A sampler wrapper for grouping images with similar aspect ratio into a same batch.

    Args:
        sampler (Sampler): Base sampler.
        dataset (Dataset): Dataset providing data information.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``.
        aspect_ratios (dict): The predefined aspect ratios.
    """

    def __init__(self,
                 sampler: Sampler,
                 dataset: Dataset,
                 batch_size: int,
                 drop_last: bool = False
                ) -> None:
        if not isinstance(sampler, Sampler):
            raise TypeError('sampler should be an instance of ``Sampler``, '
                            f'but got {sampler}')
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError('batch_size should be a positive integer value, '
                             f'but got batch_size={batch_size}')
        self.sampler = sampler
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        # buckets for each aspect ratio
        self.bucket = []

    def __iter__(self):
        for idx in self.sampler:
            self.bucket.append(idx)

            # yield a batch of indices in the same aspect ratio group
            if len(self.bucket) == self.batch_size:
                yield self.bucket[:]
                del self.bucket[:]

@contextmanager
def VideoReader_contextmanager(*args, **kwargs):
    vr = VideoReader(*args, **kwargs)
    try:
        yield vr
    finally:
        del vr
        gc.collect()

def get_video_reader_batch(video_reader, batch_index):
    frames = video_reader.get_batch(batch_index).asnumpy()
    return frames 

class VideoGameDataset(Dataset):
    def __init__(
        self,
        ann_path,
        data_root=None,
        video_sample_size=[352, 640], video_sample_stride=1, video_sample_n_frames=81,
        text_drop_ratio=0.1,
        use_action=False,
        process_collisions=True,
        enable_bucket=False,
    ):
        # Loading annotations from files
        print(f"loading annotations from {ann_path} ...")
        if ann_path.endswith('.csv'):
            with open(ann_path, 'r') as csvfile:
                self.dataset = list(csv.DictReader(csvfile))
        elif ann_path.endswith('.json'):
            self.dataset = json.load(open(ann_path))
    
        self.data_root = data_root

        self.length = len(self.dataset)
        print(f"data scale: {self.length}")
        # TODO: enable bucket training
        self.enable_bucket = enable_bucket
        self.text_drop_ratio = text_drop_ratio

        # Video params
        self.video_sample_stride    = video_sample_stride
        self.video_sample_n_frames  = video_sample_n_frames
        self.video_sample_size = tuple(video_sample_size) if not isinstance(video_sample_size, int) else (video_sample_size, video_sample_size)
        self.video_transforms = transforms.Compose(
            [
                # transforms.Resize(self.video_sample_size),
                transforms.CenterCrop(self.video_sample_size),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
        self.use_action = use_action
        self.process_collisions = process_collisions
    
    def get_batch(self, idx):
        data_info = self.dataset[idx]
        start_frame_index = int(data_info["start frame index"])
        end_frame_index = int(data_info["end frame index"])
        
        video_id, text, data_dir = data_info['original video name'], data_info['prompt'], data_info["dir"]
        json_id = video_id.replace("mp4", 'json')
        video_dir = os.path.join(self.data_root, data_dir, "video", video_id)

        with VideoReader_contextmanager(video_dir, num_threads=2) as video_reader:
            batch_index = range(start_frame_index, end_frame_index+1)
            try:
                sample_args = (video_reader, batch_index)
                pixel_values = func_timeout(
                    VIDEO_READER_TIMEOUT, get_video_reader_batch, args=sample_args
                )
                # resized_frames = []
                # for i in range(len(pixel_values)):
                #     frame = pixel_values[i]
                #     resized_frame = resize_frame(frame, self.shorter_size)
                #     resized_frames.append(resized_frame)
                # pixel_values = np.array(resized_frames)
            except FunctionTimedOut:
                raise ValueError(f"Read {idx} timeout.")
            except Exception as e:
                raise ValueError(f"Failed to extract frames from video. Error is {e}.")

            pixel_values = torch.from_numpy(pixel_values).permute(0, 3, 1, 2).contiguous()
            pixel_values = pixel_values / 255.
            del video_reader
            pixel_values = self.video_transforms(pixel_values)

        # Random use no text generation
        if random.random() < self.text_drop_ratio:
            text = ''

        sample = {"pixel_values": pixel_values,
                "text": text,
                "data_type": "video",
                "name": video_id.split(".")[0],
                "idx" : idx}
        if self.use_action:
            action_dir = os.path.join(self.data_root, data_dir, "metadata-detection", json_id)

            # Load actions from JSON file
            with open(action_dir, 'r') as f:
                action_data = json.load(f)
            actions = action_data.get("actions", {})

            # Filter actions for frames 1 to 1,999
            filtered_actions = {
                int(frame): {
                    "ws": action["ws"],
                    "ad": action["ad"],
                    "scs": action["scs"],
                    "pitch_delta": action["pitch_delta"] * 15,
                    "yaw_delta": action["yaw_delta"] * 15,
                    "collision": action["collision"],
                    "jump_invalid": action["jump_invalid"]
                }
                for frame, action in actions.items()
                if frame != "0" and int(frame) > start_frame_index and int(frame) <= end_frame_index
            }


            # Prepare mouse_actions and keyboard_actions tensors
            mouse_actions = []
            keyboard_actions = []

            for frame in range(start_frame_index, end_frame_index+1):
                action = filtered_actions.get(frame, None)
                if action:
                    # Extract mouse actions (pitch_delta, yaw_delta)
                    mouse_actions.append([action["pitch_delta"], action["yaw_delta"]])

                    # Extract keyboard actions (ws, ad, scs)
                    if self.process_collisions:
                        action = process_collisions(action)

                    keyboard_action_now = keyboard_to_onehot(action["ws"], action["ad"], action["scs"])
                    keyboard_actions.append(keyboard_action_now)
                else:
                    # Append zeros if no action is available for the frame
                    mouse_actions.append([0.0, 0.0])
                    keyboard_actions.append([0, 0, 0, 0, 0, 0, 0])

            # Convert to tensors and ensure correct shape
            mouse_actions = torch.tensor(mouse_actions, dtype=torch.float32)  # Shape [rn, mouse_dim]
            keyboard_actions = torch.tensor(keyboard_actions, dtype=torch.float32)  # Shape [rn, keyboard_dim]
            sample.update({
                "mouse_actions": mouse_actions,  # Include mouse actions in the sample
                "keyboard_actions": keyboard_actions,  # Include keyboard actions in the sample
            })

        return sample

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # if len(self.check_idx) != 0:
        #     idx = self.check_idx.pop()
        while True:
            sample = {}
            try:
                sample = self.get_batch(idx)
                if len(sample) > 0:
                    break
            except Exception as e:
                print(e, self.dataset[idx % len(self.dataset)])
                idx = random.randint(0, self.length-1)

        return sample
    

if __name__ == "__main__":
    dataset = VideoGameDataset(
        ann_path="datasets/dataset/video_captions_v0.csv",
        data_root="datasets/",
        video_sample_size=[360, 640],
        use_action=True)
    data = dataset[104]
    print(1)