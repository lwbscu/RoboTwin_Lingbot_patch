#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert RoboTwin HDF5 episodes to LeRobot v2.1 dataset (strictly compatible with your local lerobot/common/datasets/utils.py).

Key constraints enforced (per your utils.py):
- DO NOT include DEFAULT_FEATURES keys in per-frame `frame` (timestamp/index/frame_index/episode_index/task_index).
- MUST include per-frame "task" as a *string* (frame["task"] = "...").
- Dataset "features" passed to LeRobotDataset.create must NOT include "task" (task is handled specially in episode_buffer).
- Images must be np.ndarray or PIL.Image; we use np.ndarray float32 CHW in [0,1].
"""

import argparse
import shutil
from pathlib import Path

import cv2
import h5py
import numpy as np

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


# -------------------------
# Strict utilities
# -------------------------
def _require(cond: bool, msg: str):
    if not cond:
        raise RuntimeError(msg)


def _rm_dir_strict(p: Path):
    if p.exists():
        shutil.rmtree(p)


def _decode_jpeg_like_to_chw_float32(data, *, expected_hw=(240, 320)) -> np.ndarray:
    """
    Decode an encoded image (bytes or vlen uint8) via cv2.imdecode,
    output CHW float32 in [0,1], optionally resizing to expected_hw.
    """
    if isinstance(data, np.ndarray):
        buf = data.astype(np.uint8, copy=False).reshape(-1)
    else:
        # h5py can return bytes-like
        buf = np.frombuffer(data, dtype=np.uint8)

    img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("cv2.imdecode returned None (image decode failed).")

    h, w = img_bgr.shape[:2]
    eh, ew = expected_hw
    if (h, w) != (eh, ew):
        # Strict but robust: resize to match declared feature shape
        img_bgr = cv2.resize(img_bgr, (ew, eh), interpolation=cv2.INTER_AREA)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # HWC uint8
    img_f32 = img_rgb.astype(np.float32) / 255.0         # HWC float32
    chw = np.transpose(img_f32, (2, 0, 1))               # CHW
    return chw


def _as_f32_1d(x, *, dim: int, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    _require(arr.ndim == 1 and arr.shape[0] == dim, f"{name} must be shape ({dim},), got {arr.shape}")
    return arr


# -------------------------
# Main conversion
# -------------------------
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

    # IMPORTANT: wipe output to avoid FileExistsError (LeRobotDatasetMetadata.create uses exist_ok=False)
    _rm_dir_strict(output_dir)

    # Your strict LeRobot v2.1 utils.py requires:
    # - features must include ONLY "data" features (DEFAULT_FEATURES are appended internally)
    # - task is NOT a feature; it must be provided per-frame as a string
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

    print(f"Creating dataset at {output_dir} ...")
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        root=output_dir,
        features=base_features,
        use_videos=True,
    )

    # NOTE: In LeRobot v2.1, tasks are persisted when save_episode() runs (it adds unseen tasks).
    # We do not need to pre-add task here. But it's harmless to keep a single task string constant per frame.
    print(f"✅ Dataset created.\n   Features keys: {list(dataset.features.keys())}")

    print(f"Found {len(files)} episodes in: {input_dir}")

    for ep_idx, h5_path in enumerate(files):
        print(f"\nProcessing {h5_path.name} ...")

        with h5py.File(h5_path, "r") as f:
            # Validate expected RoboTwin HDF5 structure
            _require("joint_action" in f and "vector" in f["joint_action"], f"Missing joint_action/vector in {h5_path.name}")
            _require("observation" in f and "qpos" in f["observation"], f"Missing observation/qpos in {h5_path.name}")
            _require("head_camera" in f["observation"] and "rgb" in f["observation"]["head_camera"], f"Missing observation/head_camera/rgb in {h5_path.name}")
            _require("front_camera" in f["observation"] and "rgb" in f["observation"]["front_camera"], f"Missing observation/front_camera/rgb in {h5_path.name}")

            action = np.asarray(f["joint_action"]["vector"][:], dtype=np.float32)
            qpos = np.asarray(f["observation"]["qpos"][:], dtype=np.float32)

            head_ds = f["observation"]["head_camera"]["rgb"]
            front_ds = f["observation"]["front_camera"]["rgb"]

            T = int(action.shape[0])
            _require(qpos.shape[0] == T, f"Length mismatch: action {T} vs qpos {qpos.shape[0]} in {h5_path.name}")
            _require(len(head_ds) == T, f"Length mismatch: action {T} vs head_camera {len(head_ds)} in {h5_path.name}")
            _require(len(front_ds) == T, f"Length mismatch: action {T} vs front_camera {len(front_ds)} in {h5_path.name}")
            _require(action.shape[1] == 14, f"action dim must be 14, got {action.shape} in {h5_path.name}")
            _require(qpos.shape[1] == 14, f"qpos dim must be 14, got {qpos.shape} in {h5_path.name}")

            for t in range(T):
                head_img = _decode_jpeg_like_to_chw_float32(head_ds[t], expected_hw=head_hw)
                front_img = _decode_jpeg_like_to_chw_float32(front_ds[t], expected_hw=front_hw)
                state14 = _as_f32_1d(qpos[t], dim=14, name="observation/qpos[t]")
                act14 = _as_f32_1d(action[t], dim=14, name="joint_action/vector[t]")

                # CRITICAL (per your utils.py):
                # - frame MUST include "task" (str)
                # - frame MUST NOT include: timestamp/index/frame_index/episode_index/task_index
                frame = {
                    "observation.images.head_camera": head_img,
                    "observation.images.front_camera": front_img,
                    "observation.state": state14,
                    "action": act14,
                    "task": task_description,
                }

                dataset.add_frame(frame)

            # save_episode will:
            # - validate_episode_buffer
            # - compute task_index from "task" list (special key) and meta.tasks
            # - write parquet + meta + encode episode videos later
            dataset.save_episode()
            print(f"✅ Saved episode {ep_idx} with {T} frames.")

    print("\nEncoding videos ...")
    dataset.encode_videos()
    print(f"✅ DONE. Dataset ready at: {output_dir}")


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, default="data/handover_block/demo_randomized/data")
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
