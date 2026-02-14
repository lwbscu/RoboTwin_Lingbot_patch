import os
import sys
import torch
import torch.nn.functional as F
import einops
import numpy as np
import yaml
import cv2
from pathlib import Path
from transformers import AutoTokenizer

# 注入你的框架路径
ROBOTWIN_ROOT = "/mnt/public/lwb/work/embodied_stack/RoboTwin"
RLINF_ROOT = "/mnt/public/lwb/work/embodied_stack/RLinf"
sys.path.append(ROBOTWIN_ROOT)
sys.path.append(RLINF_ROOT)

from rlinf.models.embodiment.lingbot_vla.lingbot_vla_action_model import LingBotVLAForRLRollout
import importlib

# === ⚙️ 核心评估配置区 ===
CHECKPOINT_PATH = "/mnt/public/lwb/work/embodied_stack/results/robotwin_sft_lingbot/checkpoints/global_step_50/actor/model_state_dict/full_weights.pt"
TASK_NAME = "handover_block"
TASK_CONFIG = "demo_randomized"
OUTPUT_VIDEO = "eval_lingbot_vla.mp4"
MAX_STEPS = 400
INSTRUCTION = "Pick up the block and hand it over to the other arm."
# =========================

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f: return yaml.load(f.read(), Loader=yaml.FullLoader)

def load_robust_config(task_name):
    args = load_yaml(os.path.join(ROBOTWIN_ROOT, "task_config", f"{TASK_CONFIG}.yml"))
    if 'camera' not in args: args['camera'] = {}
    args['camera'].update({'collect_head_camera': True, 'collect_left_camera': True, 'collect_right_camera': True})
    args['data_type'] = {'rgb': True}
    args['render_freq'] = 0
    args["task_name"] = task_name
    
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
    envs_module = importlib.import_module(f"envs.{task_name}")
    return getattr(envs_module, task_name)()

def get_real_qpos(env):
    try:
        def safe_read(func):
            val = func()
            if isinstance(val, tuple): val = val[0]
            val = np.array(val).flatten()
            if len(val) > 6: return val[:6]
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

def apply_action_robust(env, action_vec):
    action_vec = np.atleast_1d(action_vec).flatten()
    if len(action_vec) != 14:
        if len(action_vec) > 14: action_vec = action_vec[:14]
        else: action_vec = np.pad(action_vec, (0, 14-len(action_vec)))
    
    l_arm_target = action_vec[0:6]
    l_grip_target = action_vec[6]
    r_arm_target = action_vec[7:13] 
    r_grip_target = action_vec[13]
    
    def safe_set(entity, target_6d, grip_1d):
        try:
            q = entity.get_qpos()
            if len(q) >= 7:
                q[-7:-1] = target_6d 
                q[-1] = grip_1d      
                entity.set_qpos(q)
                entity.set_qvel(np.zeros_like(q))
        except: pass

    safe_set(env.robot.left_entity, l_arm_target, l_grip_target)
    safe_set(env.robot.right_entity, r_arm_target, r_grip_target)
    env._update_render()


class LingBotInferenceEngine:
    def __init__(self, ckpt_path, device):
        self.device = device
        from omegaconf import OmegaConf
        cfg = OmegaConf.create({
            "model_type": "lingbot_vla",
            "model_path": "/mnt/public/lwb/data/embodied/models/lingbot-vla-4b",
            "tokenizer_path": "/mnt/public/lwb/data/embodied/models/Qwen2.5-VL-3B-Instruct",
            "action_dim": 14,
            "max_state_dim": 75,
            "max_action_dim": 75,
            "num_action_chunks": 50,
            "precision": "bf16",
            "device": "cuda"
        })
        
        print("🤖 [Model] Initializing LingBotVLA...")
        self.model = LingBotVLAForRLRollout(cfg).to(device).to(torch.bfloat16)
        
        if os.path.exists(ckpt_path):
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state_dict, strict=False)
            print(f"✅ [Model] Successfully loaded SFT weights from Step 50!")
        else:
            print(f"⚠️ [Model] Checkpoint not found! Using raw base model.")
            
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_path)

        # 👑 Bounds_99 统计信息 (Robotwin 14D)
        arm_q01 = [-0.967696, -0.000316, -0.000818, -1.595294, -0.444409, -2.210820, -0.136485, -0.002513, -0.001647, -1.702366, -1.029245, -1.670216]
        arm_q99 = [0.170456, 2.579206, 2.479186, 1.263499, 1.228358, 1.462294, 1.096450, 2.605947, 2.503909, 1.310469, 1.074876, 2.104229]
        eff_q01 = [-1e-10, -1e-10]
        eff_q99 = [0.999800, 0.999800]
        
        self.q01 = torch.tensor(arm_q01[:7] + eff_q01[:1] + arm_q01[7:] + eff_q01[1:], device=self.device, dtype=torch.float32)
        self.q99 = torch.tensor(arm_q99[:7] + eff_q99[:1] + arm_q99[7:] + eff_q99[1:], device=self.device, dtype=torch.float32)

        self.action_buffer = []
        self.chunk_idx = 0

    def unnormalize_action(self, action_norm):
        return (action_norm + 1.0) / 2.0 * (self.q99 - self.q01) + self.q01

    @torch.no_grad()
    def get_action(self, obs, instruction):
        if self.chunk_idx < len(self.action_buffer) and self.chunk_idx < 8:
            act = self.action_buffer[self.chunk_idx]
            self.chunk_idx += 1
            return act

        # 1. 🛡️ 图像预处理防线: 缩放224 + 伪造三视角
        raw_img = obs['observation']['head_camera']['rgb'] if 'head_camera' in obs['observation'] else obs['observation'][list(obs['observation'].keys())[0]]['rgb']
        img_tensor = torch.from_numpy(raw_img).permute(2, 0, 1).unsqueeze(0).float()
        if raw_img.max() > 1.1: img_tensor /= 255.0
        
        base_img = F.interpolate(img_tensor, size=(224, 224), mode="bilinear", align_corners=False).to(torch.bfloat16)
        images = torch.stack([base_img, base_img, base_img], dim=1).to(self.device)

        # 2. 🛡️ 状态预处理防线: 严格 14D
        qpos = get_real_qpos(obs["env"]) # Hack: 通过 env 引用读取
        state_tensor = torch.tensor(qpos, dtype=torch.float32, device=self.device).unsqueeze(0)

        # 3. 组织推理数据
        data = {
            "observation": {
                "image": {"base_0_rgb": images[:, 0], "left_wrist_0_rgb": images[:, 1], "right_wrist_0_rgb": images[:, 2]},
                "state": state_tensor
            },
            "prompt": [instruction]
        }

        # 4. 模型预测
        output = self.model(forward_type="rollout", data=data)
        actions_norm = output["action"][0] # (50, 14)
        
        # 5. 逆归一化并缓存
        actions_unnorm = self.unnormalize_action(actions_norm).cpu().numpy()
        self.action_buffer = actions_unnorm
        self.chunk_idx = 1
        
        return actions_unnorm[0]


def main():
    print("🚀 Loading Env Configuration...")
    try: args = load_robust_config(TASK_NAME)
    except FileNotFoundError as e: print(e); return

    # 👑 [恢复占位]：我们把 mplib 还给它，让机器人实体能够顺利创建
    args["planner_backend"] = "none" 
    # 但是绝不允许它进行任何逆向运动学和防碰撞计算！
    args["need_plan"] = False        
    
    import logging
    import os
    os.environ["CUROBO_LOG_LEVEL"] = "ERROR"
    logging.getLogger("curobo").setLevel(logging.ERROR)

    env = get_env_instance(TASK_NAME)
    
    # 动态劫持存在 Bug 的渲染函数
    original_update_render = env._update_render
    def patched_update_render():
        try:
            original_update_render()
        except AttributeError as e:
            if "cameras" in str(e):
                if hasattr(env, "viewer") and env.viewer is not None:
                    env.viewer.update_render()
                elif hasattr(env, "scene"):
                    env.scene.update_render()
            else:
                raise e
    env._update_render = patched_update_render
    
    print("🌍 Starting Simulation Initialization...")
    env_ready = False
    for i in range(10):
        try:
            env.setup_demo(now_ep_num=0, seed=i, is_test=True, **args)
            print(f"✅ Simulation Seed {i} Ready.")
            env_ready = True
            break
        except Exception as e: 
            print(f"⚠️ Seed {i} setup failed: {e}")
            continue
            
    # 👑 [绝对防线]：如果环境没起来，直接掐断，绝不硬跑！
    if not env_ready:
        print("❌ 致命错误：环境初始化全部失败，请检查上方 Seed setup failed 的报错！")
        return
            
    engine = LingBotInferenceEngine(CHECKPOINT_PATH, device="cuda")
    
    frames = []
    current_step = 0
    print("\n🎬 Starting Autonomous Rollout with LingBotVLA...")
    
    while current_step < MAX_STEPS:
        obs = env.get_obs()
        obs["env"] = env 
        
        action = engine.get_action(obs, INSTRUCTION)
        action = np.clip(action, -3.14, 3.14) 
        
        apply_action_robust(env, action)
        current_step += 1
        
        raw = obs['observation']['head_camera']['rgb'] if 'head_camera' in obs['observation'] else obs['observation'][list(obs['observation'].keys())[0]]['rgb']
        vis = cv2.cvtColor(raw if raw.dtype==np.uint8 else (raw*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.putText(vis, f"Step: {current_step}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        frames.append(vis)
        
        if env.check_success():
            print("✨ SUCCESS: Task Completed by LingBot!")
            break

    if frames:
        out = cv2.VideoWriter(os.path.join(ROBOTWIN_ROOT, OUTPUT_VIDEO), cv2.VideoWriter_fourcc(*'mp4v'), 30, (frames[0].shape[1], frames[0].shape[0]))
        for f in frames: out.write(f)
        out.release()
        print(f"🎉 Saved Video to: {os.path.join(ROBOTWIN_ROOT, OUTPUT_VIDEO)}")
    env.close_env()

if __name__ == "__main__":
    main()