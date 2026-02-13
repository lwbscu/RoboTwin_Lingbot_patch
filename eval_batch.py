import torch
import numpy as np
import os
import yaml
import importlib
import sys
import time
from lerobot.common.policies.act.modeling_act import ACTPolicy

# === ⚙️ 配置区 ===
CHECKPOINT_DIR = "outputs/train_handover__29_1820/checkpoints/020000/pretrained_model"
TASK_NAME = "handover_block"
TASK_CONFIG = "demo_randomized" 
TEST_EPISODES = 20  # 测试多少轮
START_SEED = 0   # 测试种子的起始值 (避开训练集 0-300)
# ================

# 路径定义
ROBOTWIN_ROOT = "/mnt/public/lwb/work/embodied_stack/RoboTwin"
TASK_CONFIG_PATH = os.path.join(ROBOTWIN_ROOT, f"task_config/{TASK_CONFIG}.yml")
EMB_CONFIG_PATH = os.path.join(ROBOTWIN_ROOT, "task_config/_embodiment_config.yml")

def load_env_args(task_name):
    """单独加载配置，方便在循环中重置环境"""
    if not os.path.exists(TASK_CONFIG_PATH):
        raise FileNotFoundError(f"❌ 配置缺失: {TASK_CONFIG_PATH}")
    
    with open(TASK_CONFIG_PATH, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    
    args['task_name'] = task_name

    # 加载机器人配置
    if not os.path.exists(EMB_CONFIG_PATH):
        alt_path = os.path.join(ROBOTWIN_ROOT, "assets/embodiments/_embodiment_config.yml")
        if os.path.exists(alt_path):
            real_emb_path = alt_path
        else:
            raise FileNotFoundError(f"❌ 机器人配置表缺失: {EMB_CONFIG_PATH}")
    else:
        real_emb_path = EMB_CONFIG_PATH

    with open(real_emb_path, "r", encoding="utf-8") as f:
        emb_conf = yaml.load(f, Loader=yaml.FullLoader)

    robot_name = args["embodiment"][0]
    args["left_robot_file"] = os.path.join(ROBOTWIN_ROOT, emb_conf[robot_name]["file_path"])
    args["right_robot_file"] = os.path.join(ROBOTWIN_ROOT, emb_conf[robot_name]["file_path"])
    args["dual_arm_embodied"] = True
    
    def load_detail_config(urdf_dir):
        cfg_path = os.path.join(urdf_dir, "config.yml")
        with open(cfg_path) as f: return yaml.load(f, Loader=yaml.FullLoader)
        
    args["left_embodiment_config"] = load_detail_config(args["left_robot_file"])
    args["right_embodiment_config"] = load_detail_config(args["right_robot_file"])
    args["embodiment_name"] = robot_name
    
    # 强制覆盖关键参数
    args["render_freq"] = 0
    args["use_seed"] = True
    args["data_type"] = {
        "rgb": True, 
        "qpos": True, 
        "depth": False, 
        "pointcloud": False
    }
    return args

def get_env_instance(task_name):
    try:
        sys.path.append(ROBOTWIN_ROOT)
        envs_module = importlib.import_module(f"envs.{task_name}")
        env_class = getattr(envs_module, task_name)
        env = env_class()
        return env
    except Exception as e:
        raise ImportError(f"❌ 无法加载环境: {e}")

def execute_step_safe(env, action_vector):
    """直接控制，防止闪烁"""
    left_qpos = action_vector[:6]
    left_gripper = action_vector[6]
    right_qpos = action_vector[7:13]
    right_gripper = action_vector[13]

    zero_vel = np.zeros_like(left_qpos)
    env.robot.set_arm_joints(left_qpos, zero_vel, "left")
    env.robot.set_arm_joints(right_qpos, zero_vel, "right")
    env.robot.set_gripper(left_gripper, "left")
    env.robot.set_gripper(right_gripper, "right")
    
    env.scene.step()
    env._update_render()

def main():
    if not os.path.exists(CHECKPOINT_DIR):
        print(f"❌ 模型未找到: {CHECKPOINT_DIR}")
        return

    print("🚀 加载 ACT 策略...")
    policy = ACTPolicy.from_pretrained(CHECKPOINT_DIR)
    policy.cuda()
    policy.eval()

    print("🌍 加载环境配置...")
    args = load_env_args(TASK_NAME)
    env = get_env_instance(TASK_NAME)

    success_count = 0
    
    print(f"\n⚡️ 开始批量评估 (共 {TEST_EPISODES} 轮)")
    print("=" * 60)

    for i in range(TEST_EPISODES):
        current_seed = START_SEED + i
        print(f"🧪 [Ep {i+1}/{TEST_EPISODES}] Seed: {current_seed}", end=" ")
        
        # 重置环境
        try:
            env.setup_demo(now_ep_num=0, seed=current_seed, **args)
        except Exception as e:
            print(f"-> ❌ 环境重置失败: {e}")
            continue

        # 获取初始观测
        if hasattr(env, 'reset_env'): env.reset_env()
        obs = env.get_obs()
        
        episode_success = False
        
        with torch.no_grad():
            for step in range(400): # 每轮最多跑 400 步
                try:
                    img = obs['observation']['head_camera']['rgb'] 
                    qpos = obs['joint_action']['vector']
                except:
                    print("-> 数据获取失败", end="")
                    break

                # 预处理
                img_tensor = torch.from_numpy(img).float()
                if img.max() > 1.1: img_tensor = img_tensor / 255.0
                img_tensor = img_tensor.permute(2, 0, 1).cuda().unsqueeze(0)
                state_tensor = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                
                # 推理
                action = policy.select_action({
                    "observation.images.head_camera": img_tensor,
                    "observation.state": state_tensor
                })
                
                # 执行 (带稳定器)
                action_np = action.cpu().numpy().flatten()
                
                # 二值化夹爪 (防止抽搐)
                action_np[6] = 1.0 if action_np[6] > 0.5 else 0.0
                action_np[13] = 1.0 if action_np[13] > 0.5 else 0.0
                
                execute_step_safe(env, action_np)
                obs = env.get_obs()

                # 检查成功
                if hasattr(env, "check_success") and env.check_success():
                    episode_success = True
                    break
        
        if episode_success:
            print("-> ✅ 成功")
            success_count += 1
        else:
            print(f"-> ❌ 失败")
            
    print("=" * 60)
    rate = (success_count / TEST_EPISODES) * 100
    print(f"🏆 最终结果: {success_count} / {TEST_EPISODES} 成功")
    print(f"📊 成功率: {rate:.1f}%")
    env.close_env()

if __name__ == "__main__":
    main()