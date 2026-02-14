import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["SAPIEN_RENDER_BACKEND"] = "egl"

import torch
_ = torch.tensor([0.0]).cuda() # 预热 CUDA 
import sys
sys.path.append("./")

import sapien.core as sapien
from sapien.render import clear_cache
from collections import OrderedDict
import pdb
from envs import *
import yaml
import importlib
import json
import traceback
import os
import time
from argparse import ArgumentParser
import numpy as np

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)

# =========================================================
# 🔧 核心黑科技：万能抓取增强补丁 (Universal Grip Hack)
# =========================================================
def apply_grip_hack(robot_instance):
    """
    使用 *args 接收任意参数，智能识别并增强抓取力度。
    """
    if hasattr(robot_instance, '_original_set_gripper'):
        return

    robot_instance._original_set_gripper = robot_instance.set_gripper

    def aggressive_set_gripper(*args, **kwargs):
        args_list = list(args)
        for i, arg in enumerate(args_list):
            if isinstance(arg, (int, float, np.number)):
                # 如果是闭合指令 (<0.1)，强行改为 -0.04 (挤压)
                if arg < 0.1:
                    args_list[i] = -0.04
                break 
        return robot_instance._original_set_gripper(*args_list, **kwargs)

    robot_instance.set_gripper = aggressive_set_gripper
    print(f"💉 Grip Hack Applied to Robot: {robot_instance}")
# =========================================================


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No such task")
    return env_instance


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def main(task_name=None, task_config=None):
    task = class_decorator(task_name)
    config_path = f"./task_config/{task_config}.yml"

    with open(config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    # 【强制修复】无论配置文件写什么，代码层面强行关闭 use_seed，触发自动规划
    args['use_seed'] = False 

    args['task_name'] = task_name

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "missing embodiment files"
        return robot_file

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        
        # 👑 [核心修复1]：准确识别 ALOHA 为原生双臂，引导底层加载 curobo_left.yml
        if "aloha" in str(embodiment_type[0]).lower() or str(embodiment_type[0]).lower() == str(embodiment_type[1]).lower():
            args["dual_arm_embodied"] = True
        else:
            args["dual_arm_embodied"] = False

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    args["embodiment_name"] = embodiment_name
    args['task_config'] = task_config
    args["save_path"] = os.path.join(args["save_path"], str(args["task_name"]), args["task_config"])
    
    if "data_type" not in args: args["data_type"] = {}
    args["data_type"]["rgb"] = True
    
    run(task, args)


def run(TASK_ENV, args):
    epid, suc_num, fail_num, seed_list = 0, 0, 0, []

    print(f"Task Name: \033[34m{args['task_name']}\033[0m")

    # =========== Collect Seed (Planning Phase) ===========
    os.makedirs(args["save_path"], exist_ok=True)
    seed_file_path = os.path.join(args["save_path"], "seed.txt")

    # 【关键修复】如果 use_seed 为 False，执行自动规划逻辑
    if not args["use_seed"]:
        print("\033[93m" + "[Start Planning Phase (Generating New Data)]" + "\033[0m")
        
        # 清理旧的种子文件，防止干扰
        if os.path.exists(seed_file_path):
            os.remove(seed_file_path)
            
        traj_dir = os.path.join(args["save_path"], "_traj_data")
        os.makedirs(traj_dir, exist_ok=True)

        generated_count = 0
        trial = 0
        max_trials = args["episode_num"] * 10 # 防止死循环
        
        while generated_count < args["episode_num"] and trial < max_trials:
            current_seed = np.random.randint(0, 1000000) + trial
            print(f"  > Planning Trial {trial} (Seed {current_seed})...", end="\r")
            
            try:
                # 1. 尝试规划
                TASK_ENV.setup_demo(now_ep_num=generated_count, seed=current_seed, **args)
                
                # 👑 [核心修复2]：必须执行这行代码！让 Curobo 算运动学并生成专家轨迹！
                TASK_ENV.play_once()
                
                # 2. 如果顺利执行，保存轨迹数据
                TASK_ENV.save_traj_data(generated_count)
                
                seed_list.append(current_seed)
                generated_count += 1
                print(f"  ✅ Plan Success: {generated_count}/{args['episode_num']} (Seed {current_seed})   ")
                
            except Exception as e:
                # print(f"  ❌ Plan Fail: {e}")  # 你可以取消这行注释来查看偶尔规划失败的原因
                pass
            
            trial += 1
            
        # 保存生成的种子列表
        with open(seed_file_path, "w") as f:
            f.write(" ".join(map(str, seed_list)))
        
        print(f"🎉 Planning Complete! Generated {len(seed_list)} valid trajectories.")

    # =========== Load Seeds ===========
    print("\033[93m" + "Loading Seeds List".center(30, "-") + "\033[0m")
    if os.path.exists(seed_file_path):
        with open(seed_file_path, "r") as file:
            seed_list = file.read().split()
            seed_list = [int(i) for i in seed_list]
    else:
        print("❌ 错误：找不到 seed.txt (规划阶段可能全部失败)")
        return

    # =========== Collect Data (Rendering Phase) ===========
    args["collect_data"] = True 

    if args["collect_data"]:
        print("\033[93m" + "[Start Data Collection (Replay Mode)]" + "\033[0m")

        args["need_plan"] = False # 关键：采集阶段不需要再规划，直接读取 path
        args["render_freq"] = 0 
        args["save_data"] = True

        clear_cache_freq = args["clear_cache_freq"]

        st_idx = 0
        def exist_hdf5(idx):
            file_path = os.path.join(args["save_path"], 'data', f'episode{idx}.hdf5')
            return os.path.exists(file_path)

        while exist_hdf5(st_idx):
            st_idx += 1
            
        print(f"Resuming from episode index: {st_idx}")

        for episode_idx in range(st_idx, args["episode_num"]):
            if episode_idx >= len(seed_list):
                print("⚠️ Warning: Not enough seeds generated.")
                break
                
            print(f"\033[34mRendering Episode {episode_idx} (Seed {seed_list[episode_idx]})\033[0m")

            try:
                # 1. 设置环境
                if hasattr(TASK_ENV, 'robot') and TASK_ENV.robot is not None:
                    apply_grip_hack(TASK_ENV.robot)

                TASK_ENV.setup_demo(now_ep_num=episode_idx, seed=seed_list[episode_idx], **args)

                if hasattr(TASK_ENV, 'robot') and TASK_ENV.robot is not None:
                    apply_grip_hack(TASK_ENV.robot)

                # 2. 加载之前规划阶段保存的轨迹
                traj_data = TASK_ENV.load_tran_data(episode_idx)
                args["left_joint_path"] = traj_data["left_joint_path"]
                args["right_joint_path"] = traj_data["right_joint_path"]
                TASK_ENV.set_path_lst(args)

                info_file_path = os.path.join(args["save_path"], "scene_info.json")
                if not os.path.exists(info_file_path):
                    with open(info_file_path, "w", encoding="utf-8") as file:
                        json.dump({}, file, ensure_ascii=False)
                with open(info_file_path, "r", encoding="utf-8") as file:
                    info_db = json.load(file)

                # 3. 回放
                info = TASK_ENV.play_once()
                info_db[f"episode_{episode_idx}"] = info

                with open(info_file_path, "w", encoding="utf-8") as file:
                    json.dump(info_db, file, ensure_ascii=False, indent=4)

                # 4. 保存
                TASK_ENV.close_env(clear_cache=((episode_idx + 1) % clear_cache_freq == 0))
                TASK_ENV.merge_pkl_to_hdf5_video() 
                TASK_ENV.remove_data_cache()
                
                status = "✅ Success" if TASK_ENV.check_success() else "⚠️ Failed Check"
                print(f"✅ Saved Episode {episode_idx} | Status: {status}")

            except Exception as e:
                print(f"❌ Failed Episode {episode_idx}: {e}")
                # traceback.print_exc()
                TASK_ENV.close_env(clear_cache=False)
                time.sleep(1)


if __name__ == "__main__":


    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    parser = ArgumentParser()
    parser.add_argument("task_name", type=str)
    parser.add_argument("task_config", type=str)
    args = parser.parse_args()
    
    main(task_name=args.task_name, task_config=args.task_config)