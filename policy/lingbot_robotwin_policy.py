import os
import torch
import torch.nn.functional as F
import einops
import numpy as np
from PIL import Image
from transformers import AutoTokenizer

# 注入 RLinf 路径以获取你魔改过的模型
import sys
sys.path.append("/mnt/public/lwb/work/embodied_stack/RLinf")
from rlinf.models.embodiment.lingbot_vla.lingbot_vla_action_model import LingBotVLAForRLRollout

class LingBotEvalWrapper:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # 👑 [严谨对齐] 硬编码来自 robotwin_50.json 的 Bounds_99 统计量
        # action.arm.position (12D) + action.effector.position (2D)
        arm_q01 = [-0.967696, -0.000316, -0.000818, -1.595294, -0.444409, -2.210820, -0.136485, -0.002513, -0.001647, -1.702366, -1.029245, -1.670216]
        arm_q99 = [0.170456, 2.579206, 2.479186, 1.263499, 1.228358, 1.462294, 1.096450, 2.605947, 2.503909, 1.310469, 1.074876, 2.104229]
        
        eff_q01 = [-1e-10, -1e-10]
        eff_q99 = [0.999800, 0.999800]
        
        self.q01 = torch.tensor(arm_q01[:7] + eff_q01[:1] + arm_q01[7:] + eff_q01[1:], device=self.device, dtype=torch.float32)
        self.q99 = torch.tensor(arm_q99[:7] + eff_q99[:1] + arm_q99[7:] + eff_q99[1:], device=self.device, dtype=torch.float32)

        # 缓存状态
        self.action_history = []
        self.step_idx = 0

    def unnormalize_action(self, action_norm):
        # Bounds_99 unnormalization 逻辑: (x + 1) / 2 * (q99 - q01) + q01
        action_unnorm = (action_norm + 1.0) / 2.0 * (self.q99 - self.q01) + self.q01
        return action_unnorm

    def reset(self):
        self.action_history = []
        self.step_idx = 0

    @torch.no_grad()
    def get_action(self, observation, instruction):
        # 如果缓存里还有 action，直接执行（Temporal Ensembling）
        if self.step_idx < len(self.action_history) and self.step_idx < 8: # 默认执行前 8 步
            action = self.action_history[self.step_idx]
            self.step_idx += 1
            return action

        # 1. 提取并清洗图像
        # 仿真器吐出的是 (H, W, C) numpy array, RGB
        raw_img = observation["image"]["head_camera"]
        img_tensor = torch.from_numpy(raw_img).permute(2, 0, 1).unsqueeze(0) # (1, 3, H, w)
        
        # 👑 [绝对防线] 强制缩放至 224x224 并伪装成 3 视角，防止 RoPE 越界！
        base_img = F.interpolate(img_tensor.float(), size=(224, 224), mode="bilinear", align_corners=False).to(torch.bfloat16) / 255.0
        
        # 构建 (Batch=1, Views=3, C=3, H=224, W=224)
        images = torch.stack([base_img, base_img, base_img], dim=1).to(self.device)

        # 2. 提取并清洗状态 (qpos)
        qpos = np.array(observation["qpos"]).flatten()
        if qpos.size == 12: # 如果仿真器少给夹爪，强行补齐
            qpos = np.append(qpos[:6], [0, 0] + qpos[6:].tolist() + [0])
        elif qpos.size > 14:
            qpos = qpos[:14]
        state_tensor = torch.tensor(qpos, dtype=torch.float32, device=self.device).unsqueeze(0)

        # 3. 组织模型输入
        data = {
            "observation": {
                "image": {"base_0_rgb": images[:, 0], "left_wrist_0_rgb": images[:, 1], "right_wrist_0_rgb": images[:, 2]},
                "state": state_tensor
            },
            "prompt": [instruction]
        }

        # 4. 执行推理获取 Action Chunk (1, 50, 14)
        output = self.model(forward_type="rollout", data=data)
        actions_norm = output["action"][0] # (50, 14)

        # 5. 逆归一化并缓存
        actions_unnorm = self.unnormalize_action(actions_norm).cpu().numpy()
        self.action_history = actions_unnorm
        self.step_idx = 1
        
        return actions_unnorm[0]


def get_model(args):
    # 模拟 cfg 结构加载你训练好的权重
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({
        "actor": {
            "model": {
                "model_type": "lingbot_vla",
                "model_path": "/mnt/public/lwb/data/embodied/models/lingbot-vla-4b",
                "tokenizer_path": "/mnt/public/lwb/data/embodied/models/Qwen2.5-VL-3B-Instruct",
                "action_dim": 14,
                "max_state_dim": 75,
                "max_action_dim": 75,
                "num_action_chunks": 50,
                "precision": "bf16",
                "device": "cuda"
            }
        }
    })

    print(f"🚀 [Eval Bridge] Loading trained LingBotVLA from FSDP Checkpoint...")
    # 这里直接实例化模型
    model = LingBotVLAForRLRollout(cfg.actor)
    
    # 👑 加载你训练出来的最新权重 (global_step_50)
    ckpt_path = "/mnt/public/lwb/work/embodied_stack/results/robotwin_sft_lingbot/checkpoints/global_step_50/actor/model_state_dict/full_weights.pt"
    if os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        print(f"✅ Successfully loaded checkpoint: {ckpt_path}")
    else:
        print(f"❌ Checkpoint not found at {ckpt_path}! Using base model weights for debugging.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).to(torch.bfloat16)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(cfg.actor.model.tokenizer_path)
    
    return LingBotEvalWrapper(model, tokenizer, device)


def eval(TASK_ENV, model_wrapper, observation):
    # 这是被 eval_policy.py 循环调用的函数
    instruction = TASK_ENV.instruction
    action = model_wrapper.get_action(observation, instruction)
    
    # 限制 Action 大小，防止仿真器崩溃 (物理引擎保护)
    action = np.clip(action, -3.14, 3.14) 
    
    TASK_ENV.step(action)


def reset_model(model_wrapper):
    model_wrapper.reset()