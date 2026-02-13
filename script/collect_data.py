import sys
import os
import yaml
import numpy as np
import sapien
import importlib
import time
import traceback
import pickle
import h5py
import cv2
from argparse import ArgumentParser

sys.path.append("./")

# =========================================================
# 🩹 1. Sapien 3 兼容与维度适配补丁
# =========================================================
ArticulationClass = None
try:
    if hasattr(sapien, "physx"):
        ArticulationClass = sapien.physx.PhysxArticulation
    elif hasattr(sapien, "Articulation"):
        ArticulationClass = sapien.Articulation

    if ArticulationClass:
        _orig = ArticulationClass.set_qpos

        def safe_set_qpos(self, qpos):
            # 针对历史兼容：如果外部给 6D 但 articulation 是 38 dof，按老逻辑写入部分 index
            if len(qpos) == 6 and getattr(self, "dof", None) == 38:
                full = self.get_qpos()
                indices = [6, 14, 18, 22, 26, 30]
                for i, idx in enumerate(indices):
                    full[idx] = qpos[i]
                return _orig(self, full)
            return _orig(self, qpos)

        ArticulationClass.set_qpos = safe_set_qpos
except Exception as e:
    print(f"⚠️ Patch Error: {e}")

from envs import *  # noqa: F401,F403


# =========================================================
# ✅ STRICT UTILITIES
# =========================================================
def _ensure_finite(x: np.ndarray, name: str, frame_idx: int = None):
    if not np.all(np.isfinite(x)):
        prefix = f"[frame={frame_idx}] " if frame_idx is not None else ""
        raise RuntimeError(f"❌ HARD FAIL: {prefix}{name} contains NaN/Inf")

def _ensure_shape_14(x: np.ndarray, name: str, frame_idx: int = None):
    x = np.asarray(x).reshape(-1)
    if x.size != 14:
        prefix = f"[frame={frame_idx}] " if frame_idx is not None else ""
        raise RuntimeError(f"❌ HARD FAIL: {prefix}{name} dim != 14 (got {x.size})")
    return x.astype(np.float32)


# =========================================================
# 🛠️ 自定义数据合并函数（STRICT：qpos 必须存在且为真）
# =========================================================
def custom_merge_hdf5(save_path, episode_idx, *, strict_images=True):
    """
    强制接管数据写入逻辑，支持自动拆分 14D action/qpos。
    使用 vlen-uint8 存储图像，避免 h5py 的 NULL byte 错误。

    STRICT POLICY:
    - frame['qpos'] 必须存在且为 14D，否则直接 HARD FAIL
    - action 必须为 14D，否则直接 HARD FAIL
    - 图像如果缺帧（strict_images=True）直接 HARD FAIL
    """
    pkl_path = os.path.join(save_path, ".cache", f"episode{episode_idx}")
    if not os.path.exists(pkl_path):
        raise RuntimeError(f"❌ HARD FAIL: Cache folder not found: {pkl_path}")

    # 1) 读取所有帧 PKL
    file_list = sorted([int(x[:-4]) for x in os.listdir(pkl_path) if x.endswith(".pkl")])
    if not file_list:
        raise RuntimeError(f"❌ HARD FAIL: no pkl frames under: {pkl_path}")

    all_data = []
    for file_idx in file_list:
        fp = os.path.join(pkl_path, f"{file_idx}.pkl")
        with open(fp, "rb") as f:
            all_data.append(pickle.load(f))

    # 2) 探测图像键路径（必须固定一致）
    sample = all_data[0]
    head_key_path = None
    if "rgb_head_camera" in sample:
        head_key_path = ["rgb_head_camera"]
    elif "observation" in sample and isinstance(sample["observation"], dict) and "head_camera" in sample["observation"]:
        head_key_path = ["observation", "head_camera", "rgb"]

    front_key_path = None
    if "rgb_front_camera" in sample:
        front_key_path = ["rgb_front_camera"]
    elif "observation" in sample and isinstance(sample["observation"], dict) and "front_camera" in sample["observation"]:
        front_key_path = ["observation", "front_camera", "rgb"]

    if head_key_path is None or front_key_path is None:
        raise RuntimeError(
            f"❌ HARD FAIL: cannot locate image keys in pkl. "
            f"head_key_path={head_key_path}, front_key_path={front_key_path}. "
            f"sample_keys={list(sample.keys())}"
        )

    def get_by_path(frm, path):
        cur = frm
        for k in path:
            if not isinstance(cur, dict) or k not in cur:
                return None
            cur = cur[k]
        return cur

    def encode_jpg_uint8(img_rgb, *, frame_idx: int, cam_name: str):
        if img_rgb is None:
            raise RuntimeError(f"❌ HARD FAIL: missing {cam_name} image at frame {frame_idx}")
        img_rgb = np.asarray(img_rgb)
        if img_rgb.ndim != 3 or img_rgb.shape[-1] != 3:
            raise RuntimeError(
                f"❌ HARD FAIL: invalid {cam_name} image shape at frame {frame_idx}: {img_rgb.shape}"
            )
        # OpenCV expects BGR
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        success, encoded = cv2.imencode(".jpg", img_bgr)
        if not success:
            raise RuntimeError(f"❌ HARD FAIL: cv2.imencode failed for {cam_name} at frame {frame_idx}")
        # store as 1D uint8 array (not bytes)
        return encoded.flatten().astype(np.uint8)

    # 3) 提取动作/状态/图像（STRICT）
    joint_action_vector = []
    qpos_vector = []
    head_imgs = []
    front_imgs = []

    for i, frame in enumerate(all_data):
        # --- A) 动作：必须能构造 14D ---
        la = frame.get("left_action")
        ra = frame.get("right_action")

        if la is None or ra is None:
            unified_action = frame.get("action", frame.get("joint_action"))
            if unified_action is not None:
                if isinstance(unified_action, dict):
                    if "vector" in unified_action:
                        u_act = np.array(unified_action["vector"]).flatten()
                        if u_act.size == 14:
                            la, ra = u_act[:7], u_act[7:]
                        elif u_act.size == 12:
                            la, ra = np.append(u_act[:6], 0), np.append(u_act[6:], 0)
                    elif "left_arm" in unified_action:
                        l_arm = np.array(unified_action["left_arm"]).flatten()
                        l_grip = float(unified_action.get("left_gripper", 0))
                        r_arm = np.array(unified_action["right_arm"]).flatten()
                        r_grip = float(unified_action.get("right_gripper", 0))
                        la, ra = np.append(l_arm, l_grip), np.append(r_arm, r_grip)
                elif isinstance(unified_action, (list, np.ndarray, tuple)):
                    u_act = np.array(unified_action).flatten()
                    if u_act.size == 14:
                        la, ra = u_act[:7], u_act[7:]
                    elif u_act.size == 12:
                        la, ra = np.append(u_act[:6], 0), np.append(u_act[6:], 0)

        if la is None or ra is None:
            raise RuntimeError(
                f"❌ HARD FAIL: cannot parse action at frame {i}. available_keys={list(frame.keys())}"
            )

        la = np.array(la).flatten()
        ra = np.array(ra).flatten()
        if la.size == 6:
            la = np.append(la, 0)
        if ra.size == 6:
            ra = np.append(ra, 0)

        if la.size < 7 or ra.size < 7:
            raise RuntimeError(f"❌ HARD FAIL: action arm dims invalid at frame {i}: la={la.size}, ra={ra.size}")

        action_14d = np.concatenate([la[:7], ra[:7]]).astype(np.float32)
        action_14d = _ensure_shape_14(action_14d, "action_14d", frame_idx=i)
        _ensure_finite(action_14d, "action_14d", frame_idx=i)
        joint_action_vector.append(action_14d)

        # --- B) 状态：qpos 必须存在且为 14D（硬失败，不允许 fallback） ---
        if "qpos" not in frame or frame["qpos"] is None:
            raise RuntimeError(
                f"❌ HARD FAIL: missing frame['qpos'] at frame {i}. available_keys={list(frame.keys())}"
            )
        qpos_14d = _ensure_shape_14(np.array(frame["qpos"]).flatten(), "qpos", frame_idx=i)
        _ensure_finite(qpos_14d, "qpos", frame_idx=i)
        qpos_vector.append(qpos_14d)

        # --- C) 图像：默认严格检查每帧必须有 ---
        head_img = get_by_path(frame, head_key_path)
        front_img = get_by_path(frame, front_key_path)

        if strict_images:
            head_imgs.append(encode_jpg_uint8(head_img, frame_idx=i, cam_name="head_camera"))
            front_imgs.append(encode_jpg_uint8(front_img, frame_idx=i, cam_name="front_camera"))
        else:
            # 非严格模式（不建议）：缺帧则跳过
            if head_img is not None:
                head_imgs.append(encode_jpg_uint8(head_img, frame_idx=i, cam_name="head_camera"))
            if front_img is not None:
                front_imgs.append(encode_jpg_uint8(front_img, frame_idx=i, cam_name="front_camera"))

    # 4) Episode 一致性检查（必须完全对齐）
    joint_action_vector = np.asarray(joint_action_vector, dtype=np.float32)
    qpos_vector = np.asarray(qpos_vector, dtype=np.float32)

    if joint_action_vector.ndim != 2 or joint_action_vector.shape[1] != 14:
        raise RuntimeError(f"❌ HARD FAIL: action array must be (T,14), got {joint_action_vector.shape}")
    if qpos_vector.ndim != 2 or qpos_vector.shape[1] != 14:
        raise RuntimeError(f"❌ HARD FAIL: qpos array must be (T,14), got {qpos_vector.shape}")

    if joint_action_vector.shape[0] != qpos_vector.shape[0]:
        raise RuntimeError(
            f"❌ HARD FAIL: length mismatch action vs qpos: "
            f"action={joint_action_vector.shape[0]}, qpos={qpos_vector.shape[0]}"
        )

    T = joint_action_vector.shape[0]

    if strict_images:
        if len(head_imgs) != T or len(front_imgs) != T:
            raise RuntimeError(
                f"❌ HARD FAIL: image length mismatch: "
                f"T={T}, head_imgs={len(head_imgs)}, front_imgs={len(front_imgs)}"
            )

    # 5) 写入 HDF5
    hdf5_path = os.path.join(save_path, "data", f"episode{episode_idx}.hdf5")
    os.makedirs(os.path.dirname(hdf5_path), exist_ok=True)

    with h5py.File(hdf5_path, "w") as f:
        g_action = f.create_group("joint_action")
        g_action.create_dataset("vector", data=joint_action_vector)

        g_obs = f.create_group("observation")
        g_obs.create_dataset("qpos", data=qpos_vector)

        g_head = g_obs.create_group("head_camera")
        g_front = g_obs.create_group("front_camera")

        # vlen uint8
        dt = h5py.special_dtype(vlen=np.dtype("uint8"))
        g_head.create_dataset("rgb", data=head_imgs, dtype=dt)
        g_front.create_dataset("rgb", data=front_imgs, dtype=dt)

    # 6) 生成视频（head_camera）
    video_path = os.path.join(save_path, "video", f"episode{episode_idx}.mp4")
    os.makedirs(os.path.dirname(video_path), exist_ok=True)

    try:
        # decode first frame for size
        first = head_imgs[0]
        img0 = cv2.imdecode(first, cv2.IMREAD_COLOR)
        if img0 is None:
            raise RuntimeError("cv2.imdecode failed on first head frame")
        h, w, _ = img0.shape
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
        for enc in head_imgs:
            bgr = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError("cv2.imdecode failed during video writing")
            out.write(bgr)
        out.release()
    except Exception as e:
        raise RuntimeError(f"❌ HARD FAIL: video generation failed: {e}")

    print(f"✅ HDF5 & Video Saved: episode{episode_idx} | T={T} | {hdf5_path}")


# =========================================================
# MAIN COLLECTION LOGIC
# =========================================================
def run(TASK_ENV, args):
    print(f"Task Name: \033[34m{args['task_name']}\033[0m")
    os.makedirs(args["save_path"], exist_ok=True)

    seed_file_path = os.path.join(args["save_path"], "seed.txt")
    seed_list = []

    # -------------------------
    # Phase 1: Planning
    # -------------------------
    if not args["use_seed"]:
        print("\033[93m" + "[Phase 1: Planning]" + "\033[0m")
        args["collect_data"] = True
        args["save_data"] = False

        if os.path.exists(seed_file_path):
            os.remove(seed_file_path)

        generated_count = 0
        trial = 0

        while generated_count < args["episode_num"] and trial < args["episode_num"] * 50:
            current_seed = np.random.randint(0, 1000000) + trial
            if trial % 10 == 0:
                print(f"  > Planning Trial {trial}...")

            try:
                TASK_ENV.setup_demo(now_ep_num=generated_count, seed=current_seed, **args)
                TASK_ENV.scene.set_timestep(1 / 500)
                TASK_ENV.play_once()
                TASK_ENV.save_traj_data(generated_count)
                seed_list.append(current_seed)
                generated_count += 1
                print(f"  ✅ Plan Success: {generated_count}/{args['episode_num']}")
            except Exception:
                # planning fail is ok; continue searching
                pass

            trial += 1
            if hasattr(TASK_ENV, "close_env"):
                TASK_ENV.close_env(clear_cache=True)

        with open(seed_file_path, "w") as f:
            f.write(" ".join(map(str, seed_list)))

    # -------------------------
    # Phase 2: Rendering
    # -------------------------
    if os.path.exists(seed_file_path):
        with open(seed_file_path, "r") as file:
            content = file.read().strip()
            seed_list = [int(i) for i in content.split()] if content else []
    else:
        return

    print("\033[93m" + "[Phase 2: Rendering with STRICT Writer]" + "\033[0m")
    args["collect_data"] = True
    args["save_data"] = True
    args["need_plan"] = False
    args["render_freq"] = 0

    for episode_idx in range(len(seed_list)):
        print(f"Rendering Episode {episode_idx} (Seed {seed_list[episode_idx]})...")
        try:
            TASK_ENV.setup_demo(now_ep_num=episode_idx, seed=seed_list[episode_idx], **args)

            # load planned trajectories
            traj_data = TASK_ENV.load_tran_data(episode_idx)
            args["left_joint_path"] = traj_data["left_joint_path"]
            args["right_joint_path"] = traj_data["right_joint_path"]
            TASK_ENV.set_path_lst(args)

            TASK_ENV.play_once()

            # STRICT merge: will HARD FAIL if qpos/images missing
            custom_merge_hdf5(args["save_path"], episode_idx, strict_images=True)

            # ✅ 默认保留 cache 便于审计；只有显式要求才删
            if args.get("delete_cache", False):
                if hasattr(TASK_ENV, "remove_data_cache"):
                    TASK_ENV.remove_data_cache()

        except Exception as e:
            print(f"❌ Failed Render (episode={episode_idx}): {e}")
            traceback.print_exc()
            if hasattr(TASK_ENV, "close_env"):
                TASK_ENV.close_env(clear_cache=False)
            # 你要求严格：失败就直接抛出，不要悄悄跳过
            raise


def main(task_name=None, task_config=None, *, delete_cache=False):
    def class_decorator(task_name):
        envs_module = importlib.import_module(f"envs.{task_name}")
        return getattr(envs_module, task_name)()

    task = class_decorator(task_name)

    config_path = f"./task_config/{task_config}.yml"
    with open(config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    # ---- Overrides (same as your pipeline expectation) ----
    args["use_seed"] = False
    args["task_name"] = task_name
    args["task_config"] = task_config
    args["delete_cache"] = bool(delete_cache)

    # Embodiment resolution (same as your original)
    embodiment_type = args.get("embodiment")
    CONFIGS_PATH = "./task_config"
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        return _embodiment_types[embodiment_type]["file_path"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False

    if "aloha" in str(embodiment_type).lower():
        args["dual_arm_embodied"] = True

    def get_robot_cfg(path):
        with open(os.path.join(path, "config.yml"), "r") as f:
            return yaml.load(f, Loader=yaml.FullLoader)

    args["left_embodiment_config"] = get_robot_cfg(args["left_robot_file"])
    args["right_embodiment_config"] = get_robot_cfg(args["right_robot_file"])

    if len(embodiment_type) == 1:
        args["embodiment_name"] = str(embodiment_type[0])
    else:
        args["embodiment_name"] = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    args["save_path"] = os.path.join(args["save_path"], str(args["task_name"]), args["task_config"])

    if "data_type" not in args:
        args["data_type"] = {}
    args["data_type"]["rgb"] = True

    run(task, args)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("task_name", type=str)
    parser.add_argument("task_config", type=str)
    parser.add_argument(
        "--delete-cache",
        action="store_true",
        help="If set, delete .cache after each episode. Default is to KEEP cache for auditing.",
    )
    cli = parser.parse_args()
    main(task_name=cli.task_name, task_config=cli.task_config, delete_cache=cli.delete_cache)
