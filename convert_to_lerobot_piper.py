import h5py
import numpy as np
import glob
import os
import json
import shutil
import pandas as pd
import torch
import torchvision
from pathlib import Path
import math
import copy

# === 配置 ===
INPUT_DIR = 'data/handover_block/demo_randomized/data/'
OUTPUT_DIR = 'lerobot_dataset_handover_29_1820' 
FPS = 30
CHUNKS_SIZE = 1000 
TASK_DESCRIPTION = "Pick up the block and hand it over to the other arm."
VIDEO_KEY = "observation.images.head_camera"
# ============

def calculate_stats(files):
    print("📊 计算 Action/State 全局统计量，并严格对齐 1D/3D Numpy 形状契约...")
    all_qpos = []
    for file_path in files:
        try:
            with h5py.File(file_path, 'r') as f:
                if 'joint_action/vector' in f:
                    all_qpos.append(f['joint_action/vector'][:])
        except: continue
    
    if not all_qpos: return {}, 0
    all_qpos = np.concatenate(all_qpos, axis=0)
    total_count = all_qpos.shape[0]
    
    mean = np.mean(all_qpos, axis=0).tolist()
    std = np.std(all_qpos, axis=0).tolist()
    min_val = np.min(all_qpos, axis=0).tolist()
    max_val = np.max(all_qpos, axis=0).tolist()

    def to_311(val_list):
        return [[[v]] for v in val_list]

    # 👑 核心修复：所有的 count 必须包裹在列表中，使其成为 shape=(1,) 的 numpy 数组
    stats = {
        "action": { 
            "mean": mean, "std": std, "min": min_val, "max": max_val, 
            "count": [total_count]  # <- 终极修复点
        },
        "observation.state": { 
            "mean": mean, "std": std, "min": min_val, "max": max_val, 
            "count": [total_count]  # <- 终极修复点
        },
        VIDEO_KEY: {
            "mean": to_311([0.485, 0.456, 0.406]),
            "std": to_311([0.229, 0.224, 0.225]),
            "min": to_311([0.0, 0.0, 0.0]),
            "max": to_311([1.0, 1.0, 1.0]),
            "count": [total_count]  # <- 终极修复点
        }
    }
    return stats, total_count

def decode_images_to_tensor(images_bytes):
    frames = []
    for b in images_bytes:
        img_tensor = torch.frombuffer(b, dtype=torch.uint8)
        img_decoded = torchvision.io.decode_image(img_tensor)
        frames.append(img_decoded)
    if not frames: return None
    video_tensor = torch.stack(frames)
    video_tensor = video_tensor.permute(0, 2, 3, 1) # (T, H, W, C)
    return video_tensor

def save_episode(ep_idx, actions_raw, images_bytes, output_dir):
    chunk_idx = ep_idx // CHUNKS_SIZE
    valid_length = len(actions_raw) - 1
    if valid_length < 1: return
    images_valid = images_bytes[:valid_length]
    
    video_dir = os.path.join(output_dir, "videos", "chunk-000", VIDEO_KEY)
    os.makedirs(video_dir, exist_ok=True)
    video_path = os.path.join(video_dir, f"episode_{ep_idx:06d}.mp4")
    
    video_tensor = decode_images_to_tensor(images_valid)
    torchvision.io.write_video(video_path, video_tensor, fps=FPS, video_codec="libx264")
    
    data_chunk_dir = os.path.join(output_dir, "data", f"chunk-{chunk_idx:03d}")
    os.makedirs(data_chunk_dir, exist_ok=True)
    parquet_path = os.path.join(data_chunk_dir, f"episode_{ep_idx:06d}.parquet")
    
    frame_data = []
    for t in range(valid_length):
        frame_data.append({
            "observation.state": actions_raw[t].tolist(),
            "action": actions_raw[t + 1].tolist(),
            "episode_index": ep_idx,
            "frame_index": t,
            "timestamp": t / float(FPS),
            "next.done": t == (valid_length - 1),
            "task_index": 0
        })
    pd.DataFrame(frame_data).to_parquet(parquet_path)
    return valid_length

def gen_and_save():
    if os.path.exists(OUTPUT_DIR): shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)
    
    files = glob.glob(os.path.join(INPUT_DIR, '*.hdf5'))
    files.sort()
    
    global_stats, total_frames_from_stats = calculate_stats(files)
    
    total_frames = 0
    episodes_metadata = []
    
    print(f"🚀 [v2.1 Master] 数据组装与元数据打包中...")
    for ep_idx, file_path in enumerate(files):
        try:
            with h5py.File(file_path, 'r') as f:
                if 'joint_action/vector' not in f: continue
                actions_raw = f['joint_action/vector'][:]
                if 'observation/head_camera/rgb' in f:
                    images_bytes = [x if isinstance(x, (bytes, np.bytes_)) else x.tobytes() for x in f['observation/head_camera/rgb'][:]]
                else: continue
                
                length = save_episode(ep_idx, actions_raw, images_bytes, OUTPUT_DIR)
                if length:
                    ep_stats = {}
                    for k, v in global_stats.items():
                        ep_stats[k] = v.copy()
                        # 👑 每集的独立统计量，count 也必须包裹在列表中
                        ep_stats[k]["count"] = [length]
                    
                    episodes_metadata.append({
                        "episode_index": ep_idx, 
                        "length": length,
                        "stats": ep_stats
                    })
                    total_frames += length
                if (ep_idx+1) % 5 == 0: print(f"  已处理 {ep_idx+1} 集...")
        except Exception as e:
            print(f"❌ {file_path}: {e}")

    meta_dir = os.path.join(OUTPUT_DIR, "meta")
    os.makedirs(meta_dir, exist_ok=True)

    total_episodes = len(episodes_metadata)
    total_chunks = math.ceil(total_episodes / CHUNKS_SIZE)

    info = {
        "codebase_version": "v2.1",
        "robot_type": "piper",
        "fps": FPS,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_chunks": total_chunks,
        "total_videos": total_episodes,
        "chunks_size": CHUNKS_SIZE,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": f"videos/{{episode_chunk:03d}}/{VIDEO_KEY}/episode_{{episode_index:06d}}.mp4", 
        "features": {
            VIDEO_KEY: {
                "dtype": "video", "shape": [480, 640, 3], "names": ["height", "width", "channel"],
                "info": {"video.fps": FPS, "video.codec": "libx264"}
            },
            "observation.state": {"dtype": "float32", "shape": [14], "names": ["motor_positions"] * 14},
            "action": {"dtype": "float32", "shape": [14], "names": ["motor_positions"] * 14},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "next.done": {"dtype": "bool", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None}
        },
        "splits": {"train": f"0:{total_episodes}"}
    }
    
    with open(os.path.join(meta_dir, "info.json"), "w") as f: json.dump(info, f, indent=4)
    with open(os.path.join(meta_dir, "stats.json"), "w") as f: json.dump(global_stats, f, indent=4)
    with open(os.path.join(meta_dir, "tasks.jsonl"), "w") as f:
        f.write(json.dumps({"task_index": 0, "task": TASK_DESCRIPTION}) + "\n")
    
    with open(os.path.join(meta_dir, "episodes.jsonl"), "w") as f:
        for ep_meta in episodes_metadata:
            line = {"episode_index": ep_meta['episode_index'], "task_index": 0, "length": ep_meta['length']}
            f.write(json.dumps(line) + "\n")

    with open(os.path.join(meta_dir, "episodes_stats.jsonl"), "w") as f:
        for ep_meta in episodes_metadata:
            line = {
                "episode_index": ep_meta['episode_index'],
                "stats": ep_meta['stats']
            }
            f.write(json.dumps(line) + "\n")

    print(f"✅ 转换完成！所有 Numpy Shape 强校验锁已解除。")

if __name__ == "__main__":
    gen_and_save()