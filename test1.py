import h5py
import os
import numpy as np

# 指向生成的 HDF5 文件
file_path = "data/handover_block/demo_randomized/data/episode0.hdf5"

print(f"Checking: {file_path}")

if not os.path.exists(file_path):
    print("❌ 还没有生成 HDF5 文件，请等待 collect_data.sh 跑完至少一个 Episode。")
    exit()

try:
    with h5py.File(file_path, "r") as f:
        print("\n=== HDF5 Structure ===")
        print(f"Keys: {list(f.keys())}")
        
        # 1. 检查 Observation
        if "observation" in f:
            print(f"Observation Keys: {list(f['observation'].keys())}")
            
            # 2. 核心检查：Qpos 是否存在且为 14维
            if "qpos" in f["observation"]:
                qpos = f["observation"]["qpos"][:]
                print(f"✅ Found 'observation/qpos' | Shape: {qpos.shape} | Type: {qpos.dtype}")
                
                if qpos.shape[1] == 14:
                    print("🎉 SUCCESS! 维度正确 (14)！")
                    print(f"Sample data: {qpos[0]}")
                else:
                    print(f"❌ Dimension Mismatch! Expected 14, got {qpos.shape[1]}")
            else:
                print("❌ Critical: 'qpos' missing in observation!")
        else:
            print("❌ Critical: 'observation' group missing!")

        # 3. 检查 Action
        if "joint_action" in f and "vector" in f["joint_action"]:
            act = f["joint_action"]["vector"][:]
            print(f"✅ Found 'joint_action/vector'| Shape: {act.shape}")
        else:
            print("❌ Action missing!")

except Exception as e:
    print(f"Read Error: {e}")