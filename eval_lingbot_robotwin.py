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
CHECKPOINT_PATH = "/mnt/public/lwb/work/embodied_stack/results/robotwin_sft_lingbot/checkpoints/global_step_70/actor/model_state_dict/full_weights.pt"
TASK_NAME = "handover_block"
TASK_CONFIG = "demo_randomized_aloha"
OUTPUT_VIDEO = "eval_lingbot_vla.mp4"
MAX_STEPS = 400
INSTRUCTION = "Pick up the block and hand it over to the other arm."
# =========================

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f: return yaml.load(f.read(), Loader=yaml.FullLoader)

def load_robust_config(task_name):
    args = load_yaml(os.path.join(ROBOTWIN_ROOT, "task_config", f"{TASK_CONFIG}.yml"))
    if 'camera' not in args: args['camera'] = {}
    # 👑 严格对齐 RoboTwin 的底层变量名
    args['camera'].update({
        'collect_head_camera': True, 
        'collect_front_camera': True, 
        'collect_wrist_camera': True   # 这一句就能同时开启 left_camera 和 right_camera
    })
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
    """
    最纯净的底层下发：不越俎代庖去搞数学 LERP，
    把力矩求解权还给底层的 SAPIEN PhysX 物理引擎！
    """
    action_vec = np.atleast_1d(action_vec).flatten()
    if len(action_vec) != 14:
        if len(action_vec) > 14: action_vec = action_vec[:14]
        else: action_vec = np.pad(action_vec, (0, 14-len(action_vec)))
    
    l_arm_target = action_vec[0:6]
    l_grip_target = action_vec[6]
    r_arm_target = action_vec[7:13] 
    r_grip_target = action_vec[13]

    # 👑 1. 模拟环境官方的调用方式：传 0 速度，但配合极短的 step
    zero_vel = np.zeros(6, dtype=np.float32)
    env.robot.set_arm_joints(l_arm_target, zero_vel, "left")
    env.robot.set_arm_joints(r_arm_target, zero_vel, "right")
    
    env.robot.set_gripper(l_grip_target, "left")
    env.robot.set_gripper(r_grip_target, "right")
    
    # 👑 2. 物理步进与渲染
    # 关键点：不要用 for 循环连续 step 10 次或 25 次！
    # 直接 step 1 次，或者参照 check_dataset.py 里的官方逻辑，
    # 把它当做单帧步进，让模型的 10Hz 输出频率自然形成平滑轨迹！
    env.scene.step()
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
            print(f"✅ [Model] Successfully loaded SFT weights!")
        else:
            print(f"⚠️ [Model] Checkpoint not found! Using raw base model.")
            
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_path)

        # 👑 [绝对正确的度量衡]：动态读取 JSON，确保 14D 顺序丝毫不差
        import json
        norm_path = "/mnt/public/lwb/work/embodied_stack/lingbot-vla/assets/norm_stats/robotwin_50.json"
        
        try:
            with open(norm_path, 'r') as f:
                norm_data = json.load(f)["norm_stats"]
            
            arm_q01 = norm_data["action.arm.position"]["q01"]
            arm_q99 = norm_data["action.arm.position"]["q99"]
            
            eff_q01 = norm_data["action.effector.position"]["q01"]
            eff_q99 = norm_data["action.effector.position"]["q99"]
            
            # 严格按照 ALOHA 14D 顺序拼接: [L_Arm(6), L_Grip(1), R_Arm(6), R_Grip(1)]
            combined_q01 = arm_q01[:6] + [eff_q01[0]] + arm_q01[6:] + [eff_q01[1]]
            combined_q99 = arm_q99[:6] + [eff_q99[0]] + arm_q99[6:] + [eff_q99[1]]
            
            self.q01 = torch.tensor(combined_q01, device=self.device, dtype=torch.float32)
            self.q99 = torch.tensor(combined_q99, device=self.device, dtype=torch.float32)
            
            print("✅ [Normalizer] Successfully loaded and aligned 14D bounds from robotwin_50.json")
            
        except Exception as e:
            print(f"❌ [Normalizer Error] Failed to load bounds: {e}")
            raise e

        self.action_buffer = []
        self.chunk_idx = 0

    # 👑 [恢复反归一化函数]：将 [-1, 1] 翻译回真实的物理弧度
    def unnormalize_action(self, action_norm):
        return (action_norm + 1.0) / 2.0 * (self.q99 - self.q01) + self.q01

    @torch.no_grad()
    def get_action(self, obs, instruction):
        if self.chunk_idx < len(self.action_buffer) and self.chunk_idx < 8:
            act = self.action_buffer[self.chunk_idx]
            self.chunk_idx += 1
            return act

        # 👑 提取真实的 3 视角，严格对接大模型的视觉骨干
        def process_img(cam_key):
            if cam_key not in obs['observation']:
                print(f"⚠️ Warning: Camera '{cam_key}' missing from env obs! Falling back to head_camera.")
                cam_key = 'head_camera' if 'head_camera' in obs['observation'] else list(obs['observation'].keys())[0]
            raw_img = obs['observation'][cam_key]['rgb']
            img_tensor = torch.from_numpy(raw_img).permute(2, 0, 1).unsqueeze(0).float()
            if raw_img.max() > 1.1: img_tensor /= 255.0
            return F.interpolate(img_tensor, size=(224, 224), mode="bilinear", align_corners=False).to(torch.bfloat16)

        # 把环境里渲染出的真实头、左腕、右腕，分别提取出来
        base_img = process_img('head_camera')
        left_wrist_img = process_img('left_camera')
        right_wrist_img = process_img('right_camera')

        qpos = get_real_qpos(obs["env"]) 
        state_tensor = torch.tensor(qpos, dtype=torch.float32, device=self.device).unsqueeze(0)

        # 👑 完美的坑位对齐
        data = {
            "observation": {
                "image": {
                    "base_0_rgb": base_img.to(self.device),             
                    "left_wrist_0_rgb": left_wrist_img.to(self.device),   
                    "right_wrist_0_rgb": right_wrist_img.to(self.device)  
                },
                "state": state_tensor
            },
            "prompt": [instruction]
        }

        output = self.model(forward_type="rollout", data=data)
        
        # 1. 提取 bfloat16 动作并转为 float32 (NumPy 才能接得住)
        actions_norm = output["action"][0].to(torch.float32)
        
        # 2. 调用正确的 14D 反归一化，翻译为物理弧度
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
    
    # 👑 【神谕测试】直接、唯一地使用我们刚才提取的黄金种子！
    target_seed = 951329
    
    try:
        env.setup_demo(now_ep_num=0, seed=target_seed, is_test=True, **args)
        print(f"✅ Simulation Seed {target_seed} Ready.")
        env_ready = True
    except Exception as e: 
        print(f"⚠️ Seed {target_seed} setup failed: {e}")
            
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