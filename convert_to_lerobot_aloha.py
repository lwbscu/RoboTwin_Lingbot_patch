#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert RoboTwin HDF5 episodes DIRECTLY to LeRobot v3.0 dataset.
"""

import argparse
import shutil
from pathlib import Path

import cv2
import h5py
import numpy as np

# 确保导入的是当前环境（V3.0）的 LeRobotDataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def _require(cond: bool, msg: str):
    if not cond:
        raise RuntimeError(msg)

def _rm_dir_strict(p: Path):
    if p.exists():
        shutil.rmtree(p)

def _decode_jpeg_like_to_chw_float32(data, *, expected_hw=(240, 320)) -> np.ndarray:
    if isinstance(data, np.ndarray):
        buf = data.astype(np.uint8, copy=False).reshape(-1)
    else:
        buf = np.frombuffer(data, dtype=np.uint8)

    img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("cv2.imdecode returned None.")

    h, w = img_bgr.shape[:2]
    eh, ew = expected_hw
    if (h, w) != (eh, ew):
        img_bgr = cv2.resize(img_bgr, (ew, eh), interpolation=cv2.INTER_AREA)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  
    img_f32 = img_rgb.astype(np.float32) / 255.0         
    chw = np.transpose(img_f32, (2, 0, 1))               
    return chw

def _as_f32_1d(x, *, dim: int, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    _require(arr.ndim == 1 and arr.shape[0] == dim, f"{name} must be shape ({dim},)")
    return arr


def convert(
    input_dir: Path,
    output_dir: Path,
    repo_id: str,
    fps: int,
    task_description: str,
    head_hw=(240, 320),
    front_hw=(240, 320),
):
    _require(input_dir.exists(), f"INPUT_DIR not found: {input_dir}")
    files = sorted(input_dir.glob("*.hdf5"))
    _require(len(files) > 0, f"No .hdf5 episodes found in: {input_dir}")

    _rm_dir_strict(output_dir)

    # 👑 V3.0 规范：只定义传感器和动作特征，千万不要定义 index, timestamp 等系统保留字
    base_features = {
        "observation.images.head_camera": {
            "dtype": "video",
            "shape": (3, int(head_hw[0]), int(head_hw[1])),
            "names": ["c", "h", "w"],
        },
        "observation.images.front_camera": {
            "dtype": "video",
            "shape": (3, int(front_hw[0]), int(front_hw[1])),
            "names": ["c", "h", "w"],
        },
        "observation.state": {"dtype": "float32", "shape": (14,), "names": ["motors"]},
        "action": {"dtype": "float32", "shape": (14,), "names": ["motors"]},
    }

    print(f"🚀 Creating V3.0 dataset at {output_dir} ...")
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        root=output_dir,
        features=base_features,
        use_videos=True,
    )

    print(f"Found {len(files)} episodes in: {input_dir}")

    for ep_idx, h5_path in enumerate(files):
        print(f"\nProcessing {h5_path.name} ...")

        with h5py.File(h5_path, "r") as f:
            # 基础结构校验
            _require("joint_action" in f and "vector" in f["joint_action"], f"Missing action in {h5_path.name}")
            _require("observation" in f and "head_camera" in f["observation"] and "rgb" in f["observation"]["head_camera"], "Missing head_camera")
            _require("observation" in f and "front_camera" in f["observation"] and "rgb" in f["observation"]["front_camera"], "Missing front_camera")

            action = np.asarray(f["joint_action"]["vector"][:], dtype=np.float32)
            
            # 容错：如果没有 qpos，用 action 代替
            if "qpos" in f["observation"]:
                qpos = np.asarray(f["observation"]["qpos"][:], dtype=np.float32)
            else:
                print("⚠️ [Warning] qpos not found, falling back to using 'action' as state.")
                qpos = action.copy()

            head_ds = f["observation"]["head_camera"]["rgb"]
            front_ds = f["observation"]["front_camera"]["rgb"]

            T = int(action.shape[0])

            for t in range(T):
                head_img = _decode_jpeg_like_to_chw_float32(head_ds[t], expected_hw=head_hw)
                front_img = _decode_jpeg_like_to_chw_float32(front_ds[t], expected_hw=front_hw)
                state14 = _as_f32_1d(qpos[t], dim=14, name="observation.state")
                act14 = _as_f32_1d(action[t], dim=14, name="action")

                # 👑 每一帧必须带上 task 描述字符串
                frame = {
                    "observation.images.head_camera": head_img,
                    "observation.images.front_camera": front_img,
                    "observation.state": state14,
                    "action": act14,
                    "task": task_description,
                }
                
                dataset.add_frame(frame)

            # V3.0 在这里会完成所有 Parquet 写入和后台 MP4 生成
            dataset.save_episode()
            print(f"✅ Saved episode {ep_idx} with {T} frames.")

    # 👑 优雅地关闭后台异步视频写入线程 (V3.0 标准操作)
    if hasattr(dataset, "stop_image_writer"):
        dataset.stop_image_writer()
        
    print(f"\n🎉 ALL DONE! The pure V3.0 dataset is ready at: {output_dir}")


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, default="data/handover_block/demo_randomized_aloha/data")
    ap.add_argument("--output_dir", type=str, default="lerobot_dataset_aloha_debug")
    ap.add_argument("--repo_id", type=str, default="aloha_handover_debug")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument(
        "--task",
        type=str,
        default="Pick up the block with the left arm and hand it over to the right arm.",
    )
    ap.add_argument("--head_h", type=int, default=240)
    ap.add_argument("--head_w", type=int, default=320)
    ap.add_argument("--front_h", type=int, default=240)
    ap.add_argument("--front_w", type=int, default=320)
    return ap


if __name__ == "__main__":
    args = build_argparser().parse_args()
    convert(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        repo_id=args.repo_id,
        fps=args.fps,
        task_description=args.task,
        head_hw=(args.head_h, args.head_w),
        front_hw=(args.front_h, args.front_w),
    )