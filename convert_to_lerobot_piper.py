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

# === 配置 ===
INPUT_DIR = 'data/handover_block/demo_randomized/data/'
OUTPUT_DIR = 'lerobot_dataset_handover_29_1820' # 改名以示区分
FPS = 30
CHUNKS_SIZE = 1000
TASK_DESCRIPTION = "Pick up the block and hand it over to the other arm."
# ============

def calculate_stats(files):
    print("📊 计算 Action/State 统计量...")
    all_qpos = []
    for file_path in files:
        try:
            with h5py.File(file_path, 'r') as f:
                if 'joint_action/vector' in f:
                    all_qpos.append(f['joint_action/vector'][:])
        except: continue
    
    if not all_qpos: return {}
    all_qpos = np.concatenate(all_qpos, axis=0)
    
    # 计算统计量
    mean = np.mean(all_qpos, axis=0).tolist()
    std = np.std(all_qpos, axis=0).tolist()
    min_val = np.min(all_qpos, axis=0).tolist()
    max_val = np.max(all_qpos, axis=0).tolist()

    return {
        # Action 和 State 共享同一套物理统计量
        "action": { "mean": mean, "std": std, "min": min_val, "max": max_val },
        "observation.state": { "mean": mean, "std": std, "min": min_val, "max": max_val },
        "observation.images.head_camera": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "min": [0.0, 0.0, 0.0],
            "max": [1.0, 1.0, 1.0]
        }
    }

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
    
    # === 关键逻辑修正：时序偏移 (Shift) ===
    # 输入: 0 ~ N-1
    # 动作: 1 ~ N (目标是下一步的位置)
    
    # 我们不仅要裁剪数据，还要丢弃最后一帧图像（因为它没有对应的未来动作）
    valid_length = len(actions_raw) - 1
    if valid_length < 1: return # 太短了

    # 1. 处理图像 (只保留前 N-1 帧)
    # 注意：images_bytes 是 list，可以直接切片
    images_valid = images_bytes[:valid_length]
    
    # 保存视频
    video_chunk_dir = os.path.join(output_dir, "videos", f"chunk-{chunk_idx:03d}")
    os.makedirs(video_chunk_dir, exist_ok=True)
    video_path = os.path.join(video_chunk_dir, f"episode_{ep_idx:06d}.mp4")
    
    video_tensor = decode_images_to_tensor(images_valid)
    torchvision.io.write_video(video_path, video_tensor, fps=FPS)
    
    # 2. 处理 Parquet (动作偏移)
    data_chunk_dir = os.path.join(output_dir, "data", f"chunk-{chunk_idx:03d}")
    os.makedirs(data_chunk_dir, exist_ok=True)
    parquet_path = os.path.join(data_chunk_dir, f"episode_{ep_idx:06d}.parquet")
    
    frame_data = []
    for t in range(valid_length):
        # 当前时刻的状态 (Obs)
        current_qpos = actions_raw[t]
        
        # 下一时刻的状态 (Action) -> 这才是我们要让模型预测的！
        next_qpos = actions_raw[t + 1]
        
        frame_data.append({
            "observation.state": current_qpos.tolist(),
            "action": next_qpos.tolist(), # <--- 修正：指向未来
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
    
    stats = calculate_stats(files)
    total_frames = 0
    episodes_metadata = []
    
    print(f"🚀 [V11] 开始转换 (Shifted Actions) ...")
    
    for ep_idx, file_path in enumerate(files):
        try:
            with h5py.File(file_path, 'r') as f:
                if 'joint_action/vector' not in f: continue
                # 获取原始数据
                actions_raw = f['joint_action/vector'][:]
                
                if 'observation/head_camera/rgb' in f:
                    images_bytes = [x if isinstance(x, (bytes, np.bytes_)) else x.tobytes() for x in f['observation/head_camera/rgb'][:]]
                else: continue
                
                # 保存并获取修正后的长度
                length = save_episode(ep_idx, actions_raw, images_bytes, OUTPUT_DIR)
                
                if length:
                    episodes_metadata.append({"episode_index": ep_idx, "length": length})
                    total_frames += length
                
                if (ep_idx+1) % 5 == 0: print(f"  已处理 {ep_idx+1} 集...")

        except Exception as e:
            print(f"❌ {file_path}: {e}")

    # 生成 Meta 文件 (完全同前)
    meta_dir = os.path.join(OUTPUT_DIR, "meta")
    os.makedirs(meta_dir, exist_ok=True)

    info = {
        "codebase_version": "v2.0",
        "robot_type": "unknown",
        "fps": FPS,
        "total_episodes": len(episodes_metadata),
        "total_frames": total_frames,
        "chunks_size": CHUNKS_SIZE,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.mp4", 
        "features": {
            "observation.images.head_camera": {
                "dtype": "video", "shape": [480, 640, 3], "names": ["height", "width", "channel"],
                "info": {"video.fps": FPS, "video.codec": "av1"}
            },
            "observation.state": {
                "dtype": "float32", "shape": [14], "names": ["motor_positions"] * 14
            },
            "action": {
                "dtype": "float32", "shape": [14], "names": ["motor_positions"] * 14
            },
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "next.done": {"dtype": "bool", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None}
        },
        "splits": {"train": f"0:{len(episodes_metadata)}"}
    }
    
    with open(os.path.join(meta_dir, "info.json"), "w") as f: json.dump(info, f, indent=4)
    with open(os.path.join(meta_dir, "stats.json"), "w") as f: json.dump(stats, f, indent=4)
    with open(os.path.join(meta_dir, "tasks.jsonl"), "w") as f:
        f.write(json.dumps({"task_index": 0, "task": TASK_DESCRIPTION}) + "\n")
    with open(os.path.join(meta_dir, "episodes.jsonl"), "w") as f:
        for ep_meta in episodes_metadata:
            line = {"episode_index": ep_meta['episode_index'], "task_index": 0, "length": ep_meta['length']}
            f.write(json.dumps(line) + "\n")

    print("✅ V11 转换完成！Action 已对其进行 Shift 操作。")

if __name__ == "__main__":
    gen_and_save()