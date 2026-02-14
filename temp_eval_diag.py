import sys
sys.path.append("/mnt/public/lwb/work/embodied_stack/RoboTwin")
sys.path.append("/mnt/public/lwb/work/embodied_stack/RLinf")
import torch
import numpy as np
from eval_lingbot_robotwin import LingBotInferenceEngine, get_env_instance, load_robust_config, get_real_qpos

env = get_env_instance("handover_block")
args = load_robust_config("handover_block")
args["planner_backend"] = "none"
args["need_plan"] = False
env.setup_demo(now_ep_num=0, seed=989255, is_test=True, **args)

engine = LingBotInferenceEngine("/mnt/public/lwb/work/embodied_stack/results/robotwin_sft_lingbot/checkpoints/global_step_50/actor/model_state_dict/full_weights.pt", "cuda")
obs = env.get_obs(); obs["env"] = env
action_unnorm = engine.get_action(obs, "Pick up the block and hand it over to the other arm.")
print("--- EVAL MODEL ACTION (UNNORM, Frame 0) ---")
print(action_unnorm.tolist())
