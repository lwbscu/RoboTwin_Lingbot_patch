import torch
import numpy as np
import os
import yaml
import importlib
import sys
import cv2

# === 配置 ===
TASK_NAME = "handover_block"
TASK_CONFIG = "demo_randomized"
ROBOTWIN_ROOT = "/mnt/public/lwb/work/embodied_stack/RoboTwin"
OUTPUT_VIDEO = "debug_physics.mp4"
# ===========

def load_config(task_name):
    config_path = os.path.join(ROBOTWIN_ROOT, f"task_config/{TASK_CONFIG}.yml")
    with open(config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    
    # 尝试开启所有能开启的相机参数
    if 'camera' not in args: args['camera'] = {}
    if 'data_type' not in args: args['data_type'] = {}
    
    args['camera']['collect_head_camera'] = True
    args['camera']['collect_front_camera'] = True
    args['data_type']['rgb'] = True
    args['data_type']['observer'] = True 
    args['data_type']['third_view'] = True
    
    args['task_name'] = task_name
    emb_config_path = os.path.join(ROBOTWIN_ROOT, "task_config/_embodiment_config.yml")
    if not os.path.exists(emb_config_path):
        emb_config_path = os.path.join(ROBOTWIN_ROOT, "assets/embodiments/_embodiment_config.yml")
    with open(emb_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)
    
    embodiment_type = args.get("embodiment")
    def get_embodiment_file(etype):
        return os.path.join(ROBOTWIN_ROOT, _embodiment_types[etype]["file_path"])
    
    args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
    args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
    args["dual_arm_embodied"] = True 
    
    def get_emb_cfg(path):
        with open(os.path.join(path, "config.yml"), "r") as f: return yaml.load(f, Loader=yaml.FullLoader)
    args["left_embodiment_config"] = get_emb_cfg(args["left_robot_file"])
    args["right_embodiment_config"] = get_emb_cfg(args["right_robot_file"])
    args["embodiment_name"] = str(embodiment_type[0])
    args["use_seed"] = True
    args["render_freq"] = 0
    return args

def get_env_instance(task_name):
    sys.path.append(ROBOTWIN_ROOT)
    envs_module = importlib.import_module(f"envs.{task_name}")
    env_class = getattr(envs_module, task_name)
    return env_class()

def inspect_robot(env):
    print("\n🔍 ====== 机器人解剖开始 ======")
    print(f"Type of env.robot: {type(env.robot)}")
    print(f"Dir of env.robot: {[x for x in dir(env.robot) if not x.startswith('_')]}")
    
    # 尝试寻找内部的 articulation
    if hasattr(env.robot, 'robot'):
        print(f"Type of env.robot.robot: {type(env.robot.robot)}")
        if hasattr(env.robot.robot, 'get_qpos'):
            q = env.robot.robot.get_qpos()
            print(f"✅ Found qpos in env.robot.robot: Shape={q.shape}")
    
    if hasattr(env.robot, 'robots'):
        print(f"env.robot.robots (List): Length={len(env.robot.robots)}")
        for i, r in enumerate(env.robot.robots):
            if hasattr(r, 'get_qpos'):
                 q = r.get_qpos()
                 print(f"  👉 Robot[{i}].get_qpos(): Shape={q.shape}, Values={q[:3]}...")

    # 打印 Observation 结构
    obs = env.get_obs()
    print("\n📸 ====== 相机状态 ======")
    print(f"Keys in obs['observation']: {list(obs['observation'].keys())}")
    print("==========================\n")

def main():
    args = load_config(TASK_NAME)
    env = get_env_instance(TASK_NAME)
    
    # 寻找稳定种子
    for i in range(10):
        try:
            print(f"Trying Seed {i}...")
            env.setup_demo(now_ep_num=0, seed=i, **args)
            break
        except: continue
            
    inspect_robot(env)
    
    print("🦾 开始执行‘正弦波’动作测试 (排除模型干扰)...")
    frames = []
    
    for step in range(100):
        # 构造一个像波浪一样变化的动作，测试机械臂是否真的受控
        # 14维: [L(6), L_grip(1), R(6), R_grip(1)]
        val = np.sin(step / 10.0) * 0.5
        
        # 左臂动，右臂不动
        action = np.zeros(14)
        action[3] = val # 让左臂第4个关节摆动
        action[6] = 1.0 # 左手张开
        action[13] = 0.0 # 右手闭合
        
        # 执行
        left_qpos = action[:6]
        right_qpos = action[7:13]
        zero_vel = np.zeros_like(left_qpos)
        
        env.robot.set_arm_joints(left_qpos, zero_vel, "left")
        env.robot.set_arm_joints(right_qpos, zero_vel, "right")
        env.robot.set_gripper(action[6], "left")
        env.robot.set_gripper(action[13], "right")
        env.scene.step()
        env._update_render()
        
        # 重新读取状态
        # 这里尝试用最底层的 API 读取
        try:
            # 假设 Piper 结构
            real_val = env.robot.robots[0].get_qpos()[3]
            print(f"Step {step}: Send={val:.3f} | Real_Joint_3={real_val:.3f}")
        except:
            print(f"Step {step}: Cannot read internal qpos")

        # 录像
        obs = env.get_obs()
        # 优先拿 head_camera
        if 'head_camera' in obs['observation']:
             img = obs['observation']['head_camera']['rgb']
             if img.max() <= 1.1: img = (img * 255).astype(np.uint8)
             frames.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    if len(frames) > 0:
        h, w, _ = frames[0].shape
        out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
        for f in frames: out.write(f)
        out.release()
        print(f"Debug video saved to {OUTPUT_VIDEO}")

if __name__ == "__main__":
    main()