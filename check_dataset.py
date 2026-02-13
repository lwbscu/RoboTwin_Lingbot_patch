import pandas as pd
import numpy as np
import time
import os
import yaml
import sys
import importlib
import torch

# === 配置：指向你用来训练的那个数据集 ===
# 确认这是你训练用的那个 1246 数据集
DATASET_PATH = "lerobot_dataset_handover_1246/data/chunk-000/episode_000000.parquet"
TASK_NAME = "handover_block"
TASK_CONFIG = "demo_randomized"
ROBOTWIN_ROOT = "/mnt/public/lwb/work/embodied_stack/RoboTwin"
# ==========================================

def execute_step_safe(env, action_vector):
    """直接控制执行"""
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

def load_full_config(task_name):
    # ... (保持原有的加载逻辑不变，为了节省篇幅我简化了这里，请直接复用之前的 load_full_config 函数) ...
    # ⚠️ 请确保这里直接复制之前 check_dataset.py 里的 load_full_config 代码
    # 或者直接用你现在的 check_dataset.py，只修改 replay_dataset 函数
    task_config_path = os.path.join(ROBOTWIN_ROOT, f"task_config/{TASK_CONFIG}.yml")
    emb_config_path = os.path.join(ROBOTWIN_ROOT, "task_config/_embodiment_config.yml")
    if not os.path.exists(task_config_path): raise FileNotFoundError(f"❌ {task_config_path}")
    with open(task_config_path, "r", encoding="utf-8") as f: args = yaml.load(f.read(), Loader=yaml.FullLoader)
    args['task_name'] = task_name
    if not os.path.exists(emb_config_path):
        alt = os.path.join(ROBOTWIN_ROOT, "assets/embodiments/_embodiment_config.yml")
        emb_config_path = alt if os.path.exists(alt) else emb_config_path
    with open(emb_config_path, "r", encoding="utf-8") as f: emb_conf = yaml.load(f, Loader=yaml.FullLoader)
    r_name = args["embodiment"][0]
    args["left_robot_file"] = os.path.join(ROBOTWIN_ROOT, emb_conf[r_name]["file_path"])
    args["right_robot_file"] = os.path.join(ROBOTWIN_ROOT, emb_conf[r_name]["file_path"])
    args["dual_arm_embodied"] = True
    def ld(d):
        with open(os.path.join(d, "config.yml")) as f: return yaml.load(f, Loader=yaml.FullLoader)
    args["left_embodiment_config"] = ld(args["left_robot_file"])
    args["right_embodiment_config"] = ld(args["right_robot_file"])
    args["embodiment_name"] = r_name
    args["render_freq"] = 0
    args["use_seed"] = True
    return args

def get_env_raw(task_name):
    sys.path.append(ROBOTWIN_ROOT)
    envs_module = importlib.import_module(f"envs.{task_name}")
    env_class = getattr(envs_module, task_name)
    return env_class()

def replay_dataset():
    print(f"🕵️  严谨审查数据完整性: {DATASET_PATH}")
    if not os.path.exists(DATASET_PATH):
        print(f"❌ 数据文件未找到！")
        return

    df = pd.read_parquet(DATASET_PATH)
    actions = np.stack(df['action'].values)
    total_frames = len(actions)
    print(f"📊 数据总帧数: {total_frames}")

    # 初始化环境
    args = load_full_config(TASK_NAME)
    env = get_env_raw(TASK_NAME)
    # 强制 Seed 0
    env.setup_demo(now_ep_num=0, seed=0, **args)
    
    print("🎬 开始 Ground Truth 回放测试...")
    
    success_frame = -1
    dataset_is_valid = False
    
    # 记录夹爪状态
    gripper_closed_at = -1
    gripper_opened_at = -1

    for i in range(total_frames):
        action_vector = actions[i]
        execute_step_safe(env, action_vector)
        
        # 1. 检查是否成功
        if env.check_success():
            if success_frame == -1:
                print(f"✨ [SUCCESS] 在第 {i}/{total_frames} 帧检测到任务成功信号！")
                success_frame = i
                dataset_is_valid = True
        
        # 2. 检查夹爪逻辑 (左手夹爪是 Index 6)
        gripper_val = action_vector[6]
        if gripper_val < 0.1 and gripper_closed_at == -1:
            gripper_closed_at = i
            print(f"✊ [GRIP] 第 {i} 帧：左手夹爪闭合 (抓取尝试)")
        
        if gripper_closed_at != -1 and gripper_val > 0.9 and gripper_opened_at == -1:
            gripper_opened_at = i
            print(f"🖐 [RELEASE] 第 {i} 帧：左手夹爪再次张开 (递交尝试)")

    print("\n" + "="*40)
    print("⚖️  最终裁决报告")
    print("="*40)
    
    if dataset_is_valid:
        print("✅ 数据合格：Ground Truth 包含完整的成功演示。")
        print(f"   - 成功点: Frame {success_frame}")
        print(f"   - 总长度: {total_frames}")
        print("👉 如果 eval 失败，说明是 execute_step_safe 代码有问题，或者模型太笨。")
    else:
        print("❌ 数据不合格：整条轨迹回放完毕，环境判定【未成功】！")
        print("   原因分析：")
        if gripper_closed_at == -1:
            print("   - 机器人压根没关过夹爪 (Action有问题)")
        elif gripper_opened_at == -1:
            print("   - 机器人抓了东西，但直到视频结束都没松手递出去。")
            print("   - 结论：数据采集被截断了 (Truncated)。")
        else:
            print("   - 动作看起来做完了，但物体没到位，或者check_success条件太苛刻。")
            
        print("👉 既然老师(数据)都没做对，学生(模型)肯定学不会。")

    env.close_env()

if __name__ == "__main__":
    replay_dataset()