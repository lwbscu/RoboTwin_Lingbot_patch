#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import importlib
import numpy as np
import traceback
import os
import yaml
from collections.abc import Mapping, Sequence
from collections import deque

# ----------------------------
# Helpers
# ----------------------------
def _to_np(x):
    try:
        if isinstance(x, np.ndarray): return x
        if isinstance(x, (list, tuple)): return np.asarray(x)
    except: return None
    return None

def _summarize_array(arr: np.ndarray, max_elems=8):
    arr = np.asarray(arr)
    shape = arr.shape
    dtype = arr.dtype
    flat = arr.reshape(-1)
    head = flat[:max_elems]
    return f"shape={shape} dtype={dtype} head={np.array2string(head, precision=5, suppress_small=True)}"

# ----------------------------
# Scan & Search Logic
# ----------------------------
STATE_KEYWORDS = ["qpos", "qvel", "state", "states", "proprio", "joint", "joints", "action", "act"]

def scan_tree(obj, path="", max_depth=6, max_items=80):
    hits = 0
    def _rec(x, p, d):
        nonlocal hits
        if hits >= max_items or d > max_depth: return
        if isinstance(x, Mapping):
            for kk, vv in x.items(): _rec(vv, f"{p}/{kk}" if p else str(kk), d + 1)
            return
        if isinstance(x, Sequence) and not isinstance(x, (str, bytes, bytearray)):
            arr = _to_np(x)
            if arr is not None and arr.dtype != object and arr.size > 0:
                _emit_leaf(p, arr); return
            for i, vv in enumerate(list(x)[:20]): _rec(vv, f"{p}[{i}]", d + 1)
            return
        if isinstance(x, np.ndarray): _emit_leaf(p, x); return

    def _emit_leaf(p, arr):
        nonlocal hits
        if hits >= max_items: return
        key_lower = p.lower()
        looks_like_state_key = any(k in key_lower for k in STATE_KEYWORDS)
        arr = np.asarray(arr)
        if arr.dtype == object: return
        if arr.ndim >= 3 and (arr.shape[-1] in (3, 4) or arr.shape[0] in (3, 4)): return
        dims = set(arr.shape) if arr.ndim > 0 else set()
        looks_like_qpos_dim = (14 in dims) or (38 in dims) or (6 in dims) or (7 in dims)

        if looks_like_state_key or looks_like_qpos_dim:
            tag = []
            if looks_like_state_key: tag.append("key")
            if looks_like_qpos_dim: tag.append("dim")
            tag = ",".join(tag) if tag else "hit"
            print(f"[SCAN:{tag}] {p:70s} -> {_summarize_array(arr)}")
            hits += 1

    _rec(obj, path, 0)
    if hits == 0: print("[SCAN] no obvious state-like arrays found.")

def find_articulations(root, name="root", max_depth=4, max_nodes=2000):
    seen = set()
    q = deque([(root, name, 0)])
    arts = []
    while q and len(seen) < max_nodes:
        obj, path, d = q.popleft()
        if id(obj) in seen: continue
        seen.add(id(obj))
        if d > max_depth: continue

        # Check for get_qpos + get_active_joints
        try:
            if hasattr(obj, "get_qpos") and callable(getattr(obj, "get_qpos")):
                arts.append((path, obj))
        except: pass

        # Expand
        try:
            if hasattr(obj, "__dict__"):
                for k, v in obj.__dict__.items():
                    if not k.startswith("_") and not isinstance(v, (str, int, float, bool)):
                        q.append((v, f"{path}.{k}", d + 1))
        except: pass
    return arts

def try_fetch_obs(TASK_ENV):
    """
    Try common observation getters / fields in RoboTwin-style envs.
    Returns (obs_dict, source_str) or (None, reason).
    """
    # 1) common fields
    for attr in ["now_obs", "obs", "observation", "latest_obs"]:
        if hasattr(TASK_ENV, attr):
            v = getattr(TASK_ENV, attr)
            if isinstance(v, dict) and len(v) > 0:
                return v, f"TASK_ENV.{attr}"

    # 2) common methods
    cand_methods = [
        "get_obs", "get_observation", "_get_obs",
        "observe", "observation_fn", "get_state", "get_robot_state"
    ]
    for m in cand_methods:
        if hasattr(TASK_ENV, m) and callable(getattr(TASK_ENV, m)):
            try:
                v = getattr(TASK_ENV, m)()
                if isinstance(v, dict) and len(v) > 0:
                    return v, f"TASK_ENV.{m}()"
            except Exception:
                pass

    # 3) sometimes stored under internal buffers
    try:
        for k, v in TASK_ENV.__dict__.items():
            if isinstance(v, dict) and len(v) > 0:
                # heuristic: has camera/state-like keys
                ks = " ".join(map(str, list(v.keys())[:30])).lower()
                if any(w in ks for w in ["qpos","state","camera","rgb","images","proprio","joint"]):
                    return v, f"TASK_ENV.__dict__[{k}]"
    except Exception:
        pass

    return None, "no obs getter/field produced a non-empty dict"

def must_get_14d_qpos(obs_dict):
    candidates = [("qpos",), ("observation", "qpos"), ("state",), ("states",), ("proprio",), ("joint_positions",)]
    for path in candidates:
        cur = obs_dict
        valid = True
        for k in path:
            if isinstance(cur, dict) and k in cur: cur = cur[k]
            else: valid = False; break
        if valid:
            arr = _to_np(cur)
            if arr is not None and (arr.size == 14 or arr.shape[-1] == 14):
                return arr.reshape(-1), ".".join(path)
    raise RuntimeError("❌ HARD FAIL: cannot find 14D qpos in obs.")

def dump_active_joint_names(art, prefix="art"):
    # SAPIEN articulation usually supports get_active_joints()
    if hasattr(art, "get_active_joints") and callable(getattr(art, "get_active_joints")):
        try:
            joints = art.get_active_joints()
            names = []
            for j in joints:
                if hasattr(j, "name"): names.append(j.name)
                elif hasattr(j, "get_name"): names.append(j.get_name())
                else: names.append(str(j))
            print(f"\n[{prefix}] active_joints (index -> name):")
            for i, n in enumerate(names):
                print(f"  {i:02d}: {n}")
            return names, joints
        except Exception as e:
            print(f"⚠️ dump_active_joint_names failed: {e}")
    else:
        print(f"⚠️ articulation has no get_active_joints()")
    return None, None

def extract_14d_from_art_and_robot(art, robot, joint_names):
    """
    Build qpos14 = [L1..L6, Lgrip, R1..R6, Rgrip] with name-based indexing.
    HARD FAIL if any joint name missing.
    """
    if joint_names is None:
        raise RuntimeError("No joint names available for mapping")
        
    qpos_full = np.asarray(art.get_qpos()).reshape(-1)

    def find_idx(nm):
        for i, n in enumerate(joint_names):
            if n == nm: return i
        return None

    # 1. Get Left Arm Joints
    if hasattr(robot, "left_arm_joints_name"):
        left_names = list(robot.left_arm_joints_name)
    else:
        raise RuntimeError("robot has no left_arm_joints_name")

    # 2. Get Right Arm Joints
    right_names = None
    if hasattr(robot, "right_arm_joints_name"):
        right_names = list(robot.right_arm_joints_name)
    else:
        # convention fallback
        right_names = [str(n).replace("fl_", "fr_") for n in left_names]

    # 3. Map Indices
    L = []
    for n in left_names:
        idx = find_idx(str(n))
        if idx is None:
            raise RuntimeError(f"❌ HARD FAIL: cannot find left joint '{n}' in articulation active joints")
        L.append(qpos_full[idx])

    R = []
    for n in right_names:
        idx = find_idx(str(n))
        if idx is None:
            raise RuntimeError(f"❌ HARD FAIL: cannot find right joint '{n}' in articulation active joints")
        R.append(qpos_full[idx])

    # 4. Gripper Values (from robot fields, because they are often not in qpos or controlled differently)
    # Note: If gripper IS in qpos, you can use similar logic to find its index. 
    # But for now, we trust the robot.left/right_gripper_val as the "commanded/sensed" value.
    if not hasattr(robot, "left_gripper_val"):
        # Try finding "left_gripper_joint" in active joints
        idx = find_idx("left_gripper_joint") # guess
        if idx is not None:
             lgrip = qpos_full[idx]
        else:
             raise RuntimeError("❌ HARD FAIL: robot has no left_gripper_val")
    else:
        lgrip = float(robot.left_gripper_val)

    if hasattr(robot, "right_gripper_val"):
        rgrip = float(robot.right_gripper_val)
    else:
        # Try guess
        idx = find_idx("right_gripper_joint")
        if idx is not None:
             rgrip = qpos_full[idx]
        else:
             raise RuntimeError("❌ HARD FAIL: robot has no right_gripper_val")

    # Assemble
    qpos14 = np.asarray(L + [lgrip] + R + [rgrip], dtype=np.float32)
    return qpos14

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task_name", type=str)
    ap.add_argument("task_config", type=str)
    ap.add_argument("--steps", type=int, default=1)
    args = ap.parse_args()

    # 1. Import Env
    try:
        envs_module = importlib.import_module(f"envs.{args.task_name}")
        TASK_ENV = getattr(envs_module, args.task_name)()
        print(f"✅ Loaded TASK_ENV: {TASK_ENV}")
    except Exception:
        print("❌ Failed to import env."); raise

    # 2. Config
    print("\n=== LOADING CONFIGURATION ===")
    CONFIGS_PATH = "./task_config"
    cfg_path = f"./task_config/{args.task_config}.yml"
    if not os.path.exists(cfg_path):
        if os.path.exists(f"./{args.task_config}.yml"): cfg_path = f"./{args.task_config}.yml"
        else: raise FileNotFoundError(f"task config not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f: cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    
    # Overrides
    cfg.update({"use_seed": True, "need_plan": False, "collect_data": False, "save_data": False, "render_freq": 0, "task_name": args.task_name, "save_path": os.path.join(cfg.get("save_path", "data"), "_probe_tmp", args.task_name)})
    cfg.setdefault("data_type", {})["rgb"] = True

    # Embodiment
    embodiment_type = cfg.get("embodiment", None)
    if embodiment_type is None: raise RuntimeError("missing embodiment")
    emb_cfg_file = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    if not os.path.exists(emb_cfg_file):
         if os.path.exists("./configs/_embodiment_config.yml"):
             CONFIGS_PATH = "./configs"; emb_cfg_file = "./configs/_embodiment_config.yml"
         else: raise FileNotFoundError("missing embodiment config")
    print(f"✅ Using embodiment config: {emb_cfg_file}")
    with open(emb_cfg_file, "r") as f: _embodiment_types = yaml.load(f, Loader=yaml.FullLoader)
    
    def get_embodiment_file(emb):
        rel_path = _embodiment_types[emb]["file_path"]
        if os.path.exists(rel_path): return rel_path
        joined = os.path.join(CONFIGS_PATH, rel_path)
        if os.path.exists(joined): return joined
        return os.path.join("./configs", rel_path)

    if isinstance(embodiment_type, (list, tuple)) and len(embodiment_type) == 1:
        cfg["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        cfg["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        cfg["dual_arm_embodied"] = True
    elif isinstance(embodiment_type, (list, tuple)) and len(embodiment_type) == 3:
        cfg["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        cfg["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        cfg["embodiment_dis"] = embodiment_type[2]
        cfg["dual_arm_embodied"] = False
    else: cfg["dual_arm_embodied"] = True
    if "aloha" in str(embodiment_type).lower(): cfg["dual_arm_embodied"] = True

    def get_robot_cfg(path):
        target = os.path.join(path, "config.yml")
        if not os.path.exists(target): target = os.path.join(CONFIGS_PATH, path, "config.yml")
        with open(target, "r") as ff: return yaml.load(ff, Loader=yaml.FullLoader)

    if "left_robot_file" in cfg: cfg["left_embodiment_config"] = get_robot_cfg(cfg["left_robot_file"])
    if "right_robot_file" in cfg: cfg["right_embodiment_config"] = get_robot_cfg(cfg["right_robot_file"])

    # 3. Setup
    try:
        TASK_ENV.setup_demo(now_ep_num=0, seed=0, **cfg)
        print("✅ setup_demo succeeded.")
    except:
        print("❌ setup_demo FAILED."); traceback.print_exc(); raise

    # =========================================================================
    # PHASE 1: FETCH OBS (ROBUST)
    # =========================================================================
    print(f"\n==================== PHASE 1: FETCH OBS (Robust) ====================")
    obs0, src = try_fetch_obs(TASK_ENV)
    print("obs source:", src)
    
    if obs0 is None:
        print("⚠️ No obs available yet (likely only updated during play loop).")
        print("    >> Will proceed to check PHYSICS directly.")
    else:
        print("✅ obs keys:", list(obs0.keys())[:30])
        scan_tree(obs0, path=f"obs@{src}", max_depth=10, max_items=250)
        
        # Check for 14D
        try:
            _, p = must_get_14d_qpos(obs0)
            print(f"✅ OK: Found 14D qpos at '{p}'")
        except RuntimeError:
            print("⚠️ Obs dict currently has no 14D qpos.")
            print("    >> Will enforce qpos from PHYSICS articulation instead.")

    # =========================================================================
    # PHASE 2: PHYSICS PROBE (ARTICULATION + JOINT NAMES)
    # =========================================================================
    print(f"\n==================== PHASE 2: FIND PHYSICS ARTICULATION ====================")
    arts = []
    arts += find_articulations(TASK_ENV, "TASK_ENV", max_depth=3)
    if hasattr(TASK_ENV, "robot"): arts += find_articulations(TASK_ENV.robot, "TASK_ENV.robot", max_depth=4)
    
    uniq_arts = []
    seen = set()
    for p, o in arts:
        if id(o) not in seen:
            seen.add(id(o))
            uniq_arts.append((p, o))
            
    if not uniq_arts:
        raise RuntimeError("❌ HARD FAIL: No physics articulations found! (Is the robot loaded?)")
    
    print(f"✅ Found {len(uniq_arts)} articulation candidates:")
    
    valid_q14 = None
    
    for p, art in uniq_arts[:5]:
        print(f"\n--- Checking Candidate: {p} ---")
        try:
            # 1. Print Joint Names (The Proof)
            names, _ = dump_active_joint_names(art, prefix=p)
            
            # 2. Try to extract 14D
            if names:
                print("    > Attempting strict 14D extraction...")
                q14 = extract_14d_from_art_and_robot(art, TASK_ENV.robot, names)
                print(f"✅ STRICT qpos14 extracted! Head={q14[:6]}")
                print(f"   (L_grip={q14[6]:.4f}, R_grip={q14[13]:.4f})")
                valid_q14 = q14
                break # Found our hero
        except Exception as e:
            print(f"    > Extraction failed: {e}")

    # =========================================================================
    # PHASE 3: FINAL VERDICT
    # =========================================================================
    print(f"\n==================== PHASE 3: VERDICT ====================")
    
    if valid_q14 is not None:
        print(">>> SUCCESS: We found a valid physical articulation and mapped it to 14D qpos.")
        print("    Action: You can now safely use the `extract_14d_from_art_and_robot` logic in collect_data.py")
    else:
        print("❌ FAILURE: Could not extract 14D qpos from any articulation.")
        print("    Reason: Likely missing joint names or robot.left/right_arm_joints_name definition mismatch.")
        print("    Please check the Phase 2 logs above for exact joint names.")
        exit(1)

    # Cleanup
    if hasattr(TASK_ENV, "close_env"): 
        try: TASK_ENV.close_env(clear_cache=False)
        except: pass

if __name__ == "__main__":
    try: main()
    except Exception: pass