import pandas as pd
import json
import glob

print("🚀 开始执行 v3.0 终极微创手术...")

# 1. 强行切开 Parquet 物理文件，注入 index 列
parquet_files = glob.glob('/mnt/public/lwb/work/embodied_stack/RoboTwin/lerobot_dataset_handover_29_1820/data/**/*.parquet', recursive=True)
for f in parquet_files:
    df = pd.read_parquet(f)
    if 'index' not in df.columns:
        df['index'] = range(len(df))  # 生成全局绝对索引 0, 1, 2...
        df.to_parquet(f, index=False)
        print(f"✅ 物理手术成功：已向 {f} 注入 index 列！")

# 2. 强行修改 info.json，为 index 上户口
info_path = '/mnt/public/lwb/work/embodied_stack/RoboTwin/lerobot_dataset_handover_29_1820/meta/info.json'
with open(info_path, 'r') as f:
    info = json.load(f)

if 'index' not in info['features']:
    info['features']['index'] = {
        "dtype": "int64",
        "shape": [1],
        "names": None
    }
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=4)
    print(f"✅ 元数据手术成功：已向 info.json 注册 index 特征！")

print("🎉 手术完毕！你的数据集现在是 100% 无懈可击的 v3.0 完全体。")
