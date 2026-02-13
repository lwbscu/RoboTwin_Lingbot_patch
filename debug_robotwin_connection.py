import numpy as np
import os
import yaml
import importlib
import sys
import cv2
import time
import math

# === ⚙️ 配置区 ===
TASK_NAME = "handover_block"
TASK_CONFIG = "demo_randomized"
ROBOTWIN_ROOT = os.getcwd() 
OUTPUT_VIDEO = "debug_v5_reach_long.mp4" # 改个名字
TOTAL_STEPS = 450  # 15秒 (30FPS)
# ================

def load_config_force_agilex(task_name):
    """V5: 保持 AgileX 配置不变"""
    print(f"🌍 加载任务配置 (V5 - Long Reach): {task_name}...")
    config_path = os.path.join(ROBOTWIN_ROOT, f"task_config/{TASK_CONFIG}.yml")
    
    with open(config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name
    args['task_config'] = TASK_CONFIG

    # 锁定 aloha-agilex
    target_key = 'aloha-agilex'
    args["embodiment"] = [target_key, target_key, 0.6]
    
    emb_config_path = os.path.join(ROBOTWIN_ROOT, "task_config/_embodiment_config.yml")
    if not os.path.exists(emb_config_path):
        emb_config_path = os.path.join(ROBOTWIN_ROOT, "assets/embodiments/_embodiment_config.yml")

    with open(emb_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)
    
    rel_path = _embodiment_types[target_key]["file_path"]
    robot_file = os.path.join(ROBOTWIN_ROOT, rel_path)
    
    args["left_robot_file"] = robot_file
    args["right_robot_file"] = robot_file
    args["dual_arm_embodied"] = True
    args["embodiment_name"] = f"{target_key}_{target_key}"

    def get_emb_cfg(path):
        cfg_path = os.path.join(path, "config.yml")
        if not os.path.exists(cfg_path): return {} 
        with open(cfg_path, "r") as f: return yaml.load(f, Loader=yaml.FullLoader)
    
    args["left_embodiment_config"] = get_emb_cfg(args["left_robot_file"])
    args["right_embodiment_config"] = get_emb_cfg(args["right_robot_file"])

    if "data_type" not in args: args["data_type"] = {}
    args["data_type"]["rgb"] = True
    args["data_type"]["third_view"] = True
    args["data_type"]["qpos"] = True
    
    if "camera" not in args: args["camera"] = {}
    args["camera"]["collect_head_camera"] = True
    args["render_freq"] = 0 
    args["use_seed"] = True
    
    return args

def get_env_instance(task_name):
    sys.path.append(ROBOTWIN_ROOT)
    envs_module = importlib.import_module(f"envs.{task_name}")
    env_class = getattr(envs_module, task_name)
    return env_class()

def execute_action(env, action_14d):
    """透传动作"""
    l_arm = action_14d[:6]
    l_grip = action_14d[6]
    r_arm = action_14d[7:13]
    r_grip = action_14d[13]
    zero_vel = np.zeros(6)
    
    env.robot.set_arm_joints(l_arm, zero_vel, "left")
    env.robot.set_arm_joints(r_arm, zero_vel, "right")
    env.robot.set_gripper(l_grip, "left")
    env.robot.set_gripper(r_grip, "right")
    
    env.scene.step()
    env._update_render()

def main():
    print(f"🚀 启动 V5 测试 (15秒长视频 + 前后伸缩)...")
    try:
        args = load_config_force_agilex(TASK_NAME)
        env = get_env_instance(TASK_NAME)
        env.setup_demo(now_ep_num=0, seed=0, **args)
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return

    obs = env.get_obs()
    init_qpos = obs['joint_action']['vector']
    
    frames = []
    print(f"\n🎬 开始录制 {TOTAL_STEPS} 步 (约 {TOTAL_STEPS/30:.1f} 秒)...")
    
    for step in range(TOTAL_STEPS):
        target_action = init_qpos.copy()
        
        # === 动作设计：前后伸缩 (Reaching) ===
        # 使用慢速正弦波 (0.05) 让动作更舒缓
        # 通过同时控制 肩部(Joint 1) 和 肘部(Joint 2) 来实现“伸出去”的效果
        
        # 周期：约 120 步一个来回
        reach_val = math.sin(step * 0.05) 
        
        if len(target_action) >= 14:
            # === 左臂 ===
            # Joint 1 (大臂): 向下压 (0 到 0.8) -> 往前探
            target_action[1] += reach_val * 0.4 + 0.4 
            # Joint 2 (小臂): 伸直 (0 到 0.5)
            target_action[2] += reach_val * 0.3
            # Joint 3 (手腕): 稍微抬起，保持平衡
            target_action[3] -= reach_val * 0.2

            # === 右臂 (对称动作) ===
            target_action[8] += reach_val * 0.4 + 0.4
            target_action[9] += reach_val * 0.3
            target_action[10] -= reach_val * 0.2
            
            # === 夹爪 (配合伸手) ===
            # 伸出去的时候(reach_val > 0) 张开(1.0)
            # 缩回来的时候(reach_val < 0) 闭合(0.0)
            grip_val = 1.0 if reach_val > 0 else 0.0
            target_action[6] = grip_val
            target_action[13] = grip_val
        
        execute_action(env, target_action)
        obs = env.get_obs()
        
        # 获取图像
        img = None
        if 'third_view_rgb' in obs: img = obs['third_view_rgb']
        elif 'observation' in obs and 'head_camera' in obs['observation']: 
             img = obs['observation']['head_camera']['rgb']
        
        if img is not None:
            if img.shape[2] == 3: img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # 进度条
            cv2.putText(img, f"Time: {step/30:.1f}s / {TOTAL_STEPS/30:.1f}s", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(img, f"Reach: {'FORWARD' if reach_val>0 else 'BACK'}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
            
            frames.append(img)
            
        if step % 50 == 0:
            print(f"⏳ 进度: {step}/{TOTAL_STEPS} | Reach={reach_val:.2f}")

    if len(frames) > 0:
        h, w, _ = frames[0].shape
        out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
        for f in frames: out.write(f)
        out.release()
        print(f"\n🎉 长视频已保存: {OUTPUT_VIDEO}")
        print("请检查：1. 机械臂是否前后伸缩？ 2. 视频时长是否足够？")
    
    env.close_env()

if __name__ == "__main__":
    main()