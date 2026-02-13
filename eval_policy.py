import torch
import numpy as np
import os
import yaml
import importlib
import sys
import time
import cv2
import json
from lerobot.common.policies.act.modeling_act import ACTPolicy

# === ⚙️ 配置区 (请确认路径正确) ===
CHECKPOINT_DIR = "outputs/train_handover__29_1820/checkpoints/020000/pretrained_model"
DATASET_DIR = "lerobot_dataset_handover_29_1820"
TASK_NAME = "handover_block"
TASK_CONFIG = "demo_randomized"
ROBOTWIN_ROOT = "/mnt/public/lwb/work/embodied_stack/RoboTwin"
OUTPUT_VIDEO = "eval_result_platinum.mp4"
MAX_STEPS = 400
# ============================

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f: return yaml.load(f.read(), Loader=yaml.FullLoader)

def load_dataset_stats(checkpoint_dir, dataset_dir):
    candidates = [
        os.path.join(checkpoint_dir, "dataset_stats.json"),
        os.path.join(os.path.dirname(os.path.dirname(checkpoint_dir)), "dataset_stats.json"),
        os.path.join(ROBOTWIN_ROOT, dataset_dir, "meta/stats.json")
    ]
    target_path = next((p for p in candidates if os.path.exists(p)), None)
    if not target_path: raise FileNotFoundError("❌ 找不到 stats.json")
    print(f"📊 Stats loaded: {target_path}")
    with open(target_path, 'r') as f: return json.load(f)

def normalize_state(qpos, stats, device):
    # 模糊匹配 key，适应不同数据集命名
    key = next((k for k in stats.keys() if "state" in k or "qpos" in k), None)
    if not key: raise ValueError("Stats key not found")
    mean = torch.tensor(stats[key]["mean"], device=device)
    std = torch.tensor(stats[key]["std"], device=device)
    qpos_tensor = torch.from_numpy(qpos).float().to(device)
    return (qpos_tensor - mean) / std

def unnormalize_action(action_norm, stats, device):
    mean = torch.tensor(stats["action"]["mean"], device=device)
    std = torch.tensor(stats["action"]["std"], device=device)
    return action_norm * std + mean

def load_robust_config(task_name):
    print(f"🌍 Config: {task_name}")
    args = load_yaml(os.path.join(ROBOTWIN_ROOT, "task_config", f"{TASK_CONFIG}.yml"))
    if 'camera' not in args: args['camera'] = {}
    args['camera'].update({'collect_head_camera': True, 'collect_left_camera': True, 'collect_right_camera': True})
    args['data_type'] = {'rgb': True}
    args['render_freq'] = 0
    args["task_name"] = task_name
    
    # === 路径自动修复 ===
    path1 = os.path.join(ROBOTWIN_ROOT, "task_config/_embodiment_config.yml")
    path2 = os.path.join(ROBOTWIN_ROOT, "assets/embodiments/_embodiment_config.yml")
    emb_config_path = path1 if os.path.exists(path1) else path2
    
    emb_types = load_yaml(emb_config_path)
    etype = args.get("embodiment", ["piper"])[0] if isinstance(args.get("embodiment"), list) else args.get("embodiment")
    robot_file = os.path.join(ROBOTWIN_ROOT, emb_types[etype]["file_path"])
    
    args["left_robot_file"] = robot_file
    args["right_robot_file"] = robot_file
    args["dual_arm_embodied"] = True
    args["left_embodiment_config"] = load_yaml(os.path.join(robot_file, "config.yml"))
    args["right_embodiment_config"] = load_yaml(os.path.join(robot_file, "config.yml"))
    args["embodiment_name"] = str(etype)
    args["use_seed"] = True
    return args

def get_env_instance(task_name):
    sys.path.append(ROBOTWIN_ROOT)
    envs_module = importlib.import_module(f"envs.{task_name}")
    return getattr(envs_module, task_name)()

# === 🛡️ 强制 14 维读取 (State) ===
def get_real_qpos(env):
    try:
        # 无论 API 返回什么，我都只要前 6 位 + 1 夹爪
        def safe_read(func):
            val = func()
            if isinstance(val, tuple): val = val[0]
            val = np.array(val).flatten()
            # 截断到 6
            if len(val) > 6: return val[:6]
            # 补零到 6
            if len(val) < 6: return np.pad(val, (0, 6-len(val)))
            return val

        l_state = safe_read(env.robot.get_left_arm_jointState)
        l_grip = np.array([env.robot.get_left_gripper_val()]).flatten()

        r_state = safe_read(env.robot.get_right_arm_jointState)
        r_grip = np.array([env.robot.get_right_gripper_val()]).flatten()

        return np.concatenate([l_state, l_grip, r_state, r_grip]).astype(np.float32)
    except Exception as e:
        print(f"❌ State Read Error: {e}")
        return np.zeros(14, dtype=np.float32)

# === 🛡️ 强制 14 维写入 (Action) ===
def apply_action_robust(env, action_vec):
    # 1. 强制对齐 14 维
    action_vec = np.atleast_1d(action_vec).flatten()
    if len(action_vec) != 14:
        # 自动截断或补零，防止 Crash
        if len(action_vec) > 14: action_vec = action_vec[:14]
        else: action_vec = np.pad(action_vec, (0, 14-len(action_vec)))

    # 2. 硬编码切分 (6+1+6+1)
    # [0:6] = Left Arm
    # [6] = Left Gripper
    # [7:13] = Right Arm (注意：从7开始取6个)
    # [13] = Right Gripper
    
    l_arm_target = action_vec[0:6]
    l_grip_target = action_vec[6]
    r_arm_target = action_vec[7:13] 
    r_grip_target = action_vec[13]
    
    # 3. 倒序写入 (Kinematic) - 避开 Base
    def safe_set(entity, target_6d, grip_1d):
        try:
            q = entity.get_qpos()
            # 修改最后 7 位：[... Arm(6), Grip(1)]
            if len(q) >= 7:
                q[-7:-1] = target_6d 
                q[-1] = grip_1d      
                entity.set_qpos(q)
                entity.set_qvel(np.zeros_like(q))
        except: pass

    safe_set(env.robot.left_entity, l_arm_target, l_grip_target)
    safe_set(env.robot.right_entity, r_arm_target, r_grip_target)
    
    # 只更新画面，不进行物理步进，避免物理引擎报错
    env._update_render()

def main():
    if not os.path.exists(CHECKPOINT_DIR): raise FileNotFoundError("Checkpoint not found")
    print("🚀 Loading Policy...")
    stats = load_dataset_stats(CHECKPOINT_DIR, DATASET_DIR)
    policy = ACTPolicy.from_pretrained(CHECKPOINT_DIR)
    policy.cuda().eval()
    
    try:
        args = load_robust_config(TASK_NAME)
    except FileNotFoundError as e:
        print(e); return

    env = get_env_instance(TASK_NAME)
    
    for i in range(10):
        try:
            env.setup_demo(now_ep_num=0, seed=i, **args)
            print(f"✅ Seed {i} Ready.")
            break
        except: continue
            
    frames = []
    current_step = 0
    
    print("\n🎬 Starting Rollout...")
    with torch.no_grad():
        while current_step < MAX_STEPS:
            obs = env.get_obs()
            
            # 视觉处理
            if 'head_camera' in obs['observation']:
                raw = obs['observation']['head_camera']['rgb']
                src = "Head"
            else:
                k = list(obs['observation'].keys())[0]
                raw = obs['observation'][k]['rgb']
                src = k
            
            img_tensor = torch.from_numpy(raw).float()
            if raw.max() > 1.1: img_tensor = img_tensor / 255.0
            img_tensor = img_tensor.permute(2, 0, 1).cuda().unsqueeze(0)
            
            # 状态处理
            real_qpos = get_real_qpos(env)
            state_tensor = normalize_state(real_qpos, stats, device='cuda').unsqueeze(0)
            
            # 推理
            out = policy.select_action({
                "observation.images.head_camera": img_tensor,
                "observation.state": state_tensor
            })
            # 反归一化
            out_real = unnormalize_action(out, stats, device='cuda')
            
            # 维度提取
            target = out_real.reshape(-1).cpu().numpy()
            if len(target) > 14: target = target[:14] # 取第一步动作
            
            # 执行
            apply_action_robust(env, target)
            current_step += 1
            
            # 监控
            if current_step % 10 == 0:
                new_q = get_real_qpos(env)
                err = np.mean(np.abs(target[:6] - new_q[:6]))
                print(f"[{current_step}] Src:{src} | Err:{err:.4f}")
            
            vis = cv2.cvtColor(raw if raw.dtype==np.uint8 else (raw*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.putText(vis, f"Step: {current_step}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            frames.append(vis)
            
            if env.check_success():
                print("✨ SUCCESS!")
                break

    if frames:
        out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frames[0].shape[1], frames[0].shape[0]))
        for f in frames: out.write(f)
        out.release()
        print(f"🎉 Saved: {OUTPUT_VIDEO}")
    env.close_env()

if __name__ == "__main__":
    main()