# Modifications Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.
 
# This file is a modified version of code from MatrixGame
# https://github.com/SkyworkAI/Matrix-Game,
# which is licensed under the MIT License.
import os
from argparse import ArgumentParser
import clip 
import tqdm
from visual_metrics import compute_visual_metrics
from GameWorldScore.GameWorld.temporal_consistency import temporal_consistency
from GameWorldScore.GameWorld.motion_smoothness import motion_smoothness, MotionSmoothness
from GameWorldScore.GameWorld.utils import init_submodules
from GameWorldScore.GameWorld.third_party.IDM.IDM_benchmark import evaluate_IDM_quality
from GameWorldScore.GameWorld.scenario_consistency import scenario_consistency

def compute_temporal_consistency(video_list, device, submodules_list, **kwargs):
    vit_path, read_frame = submodules_list[0], submodules_list[1]
    clip_model, preprocess = clip.load(vit_path, device=device)
    all_results, video_results = temporal_consistency(clip_model, preprocess, video_list, device, read_frame)
    return all_results, video_results

def compute_motion_smoothness(video_list, device, submodules_list, **kwargs):
    config = submodules_list["config"] # pretrained/amt_model/AMT-S.yaml
    ckpt = submodules_list["ckpt"] # pretrained/amt_model/amt-s.pth
    motion = MotionSmoothness(config, ckpt, device)
    all_results, video_results = motion_smoothness(motion, video_list)
    return all_results, video_results

def compute_scenario_consistency(video_list,
                                  device=None,          
                                  submodules_list=None, 
                                  target_size=(256, 256),
                                  max_shift=4,
                                  **kwargs):

    all_results, video_results = scenario_consistency(video_list,
                                                       target_size=target_size,
                                                       max_shift=max_shift)
    return all_results, video_results    

func_dict = {
    "temporal_consistency": compute_temporal_consistency,
    "motion_smoothness": compute_motion_smoothness,
    "scenario_consistency": compute_scenario_consistency
}

if __name__ == "__main__":
    parser = ArgumentParser("Evaluate IDM quality for MC-LVM ")
    parser.add_argument("--control_metrics", action="store_true", help="Enable IDM metrics")
    parser.add_argument("--temporal_metrics", action="store_true", help="Enable temporal metrics")
    parser.add_argument("--physical_metrics", action="store_true", help="Enable physical metrics")
    parser.add_argument("--visual_metrics", action="store_true", help="Enable visual metrics")
    parser.add_argument("--weights", type=str, required=True, help="[IDM model config] Path to the '.weights' file to be loaded.")
    parser.add_argument("--model", type=str, required=True, help="[IDM model config] Path to the '.model' file to be loaded.")
    parser.add_argument("--json-path", type=str, required=True, help="[Eval Config] Path to .json contains actions.")
    parser.add_argument("--video-path", type=str, required=True, help="[Eval Config] Path to a .mp4 file.")
    parser.add_argument("--ref-path", type=str, required=True, help="[Eval Config] Path to a .mp4 file.")
    parser.add_argument("--infer-demo-num", type=int, default=0, help="[Inference Config] Number of frames to skip before starting evaluation.")
    parser.add_argument("--n-frames", type=int, default=32, help="[Inference Config] Number of frames to generation.")
    parser.add_argument("--output-file", type=str, default="[Eval Config] output/action_loss.jsonl", help="[Eval Config] Path to save the action loss.")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    video_names = os.listdir(args.video_path)
    video_names = [f for f in video_names if f.endswith(".mp4")]
    video_names = sorted(video_names)
    video_files = [os.path.join(args.video_path, f) for f in video_names]
    ref_files = [os.path.join(args.ref_path, f) for f in video_names]
    eval_num = min(500,len(video_files)) 
    video_files = video_files[:eval_num]
    ref_files = ref_files[:eval_num]
    if args.control_metrics:
        print("Evaluate action controllabilty...")
        (keyboard_precision, camera_loss_val, camera_precision), video_results = evaluate_IDM_quality(args.model, args.weights,args.json_path,video_files, args.infer_demo_num,args.n_frames,args.output_file)
        print(f"keyboard_precision: {keyboard_precision}")
        print(f"camera_precision: {camera_precision}")
    if args.temporal_metrics:
        dimension_list = ["temporal_consistency", "motion_smoothness"]
        submodules_dict = init_submodules(dimension_list,local=False, read_frame=False)
        results_dict = {}
        for k, v in submodules_dict.items():
            print(f"Compute {k}")
            func = func_dict[k]
            all_results, video_results = func(video_files, 'cuda', v)
            results_dict[k] = all_results
        for k, v in results_dict.items():
            print(f"{k}: {v}")

    if args.physical_metrics:
        dimension_list = ["scenario_consistency"]
        submodules_dict = init_submodules(dimension_list,local=False, read_frame=False)
        for k, v in submodules_dict.items():
            print(f"Compute {k}")
            func = func_dict[k]
            all_results, video_results = func(video_files, 'cuda', v)
            print(f'{k}: {all_results}')
    
    if args.visual_metrics:
        print("Compute visual metrics.")
        all_results = compute_visual_metrics(video_files, ref_files)
        for k, v in all_results.items():
            if k == 'details':
                continue
            print(f'{k}: {v}')