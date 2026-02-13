import sys
sys.path.insert(0, "/mnt/public/lwb/work/embodied_stack/RLinf/.venv/lerobot/src")
sys.path.insert(0, "/mnt/public/lwb/work/embodied_stack/lingbot-vla")

from lingbotvla.data.vla_data.base_dataset import RobotwinDataset
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from transformers import AutoTokenizer

dataset_path = "/mnt/public/lwb/work/embodied_stack/RoboTwin/lerobot_dataset_handover_29_1820"

class DummyConfig:
    img_size = 224
    norm_stats_file = "/mnt/public/lwb/work/embodied_stack/lingbot-vla/assets/norm_stats/robotwin_50.json"
    norm_type = "bounds_99_woclip"
    dataset_type = "robotwin" # 补齐可能需要的参数

tokenizer = AutoTokenizer.from_pretrained("/mnt/public/lwb/data/embodied/models/Qwen2.5-VL-3B-Instruct")
rob_ds = RobotwinDataset(
    repo_id=dataset_path,
    config=PI0Config(),
    tokenizer=tokenizer,
    data_config=DummyConfig(),
)

print("\n=== [探针] 抓取原始 KeyError(0) 堆栈 ===")
try:
    rob_ds.getdata(0)
    print("✅ 包装器读取成功！")
except Exception as e:
    import traceback
    traceback.print_exc()
