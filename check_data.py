import h5py
import sys

# 指向第一个文件
target_file = 'data/handover_block/demo_randomized/data/episode0.hdf5'

print(f"🔍 深度扫描文件: {target_file}")

try:
    with h5py.File(target_file, 'r') as f:
        print("\n===== 📂 完整文件结构树 =====")
        
        def print_node(name, obj):
            indent = "  " * name.count('/')
            node_name = name.split('/')[-1]
            if isinstance(obj, h5py.Dataset):
                print(f"{indent}📄 {node_name}  [Shape: {obj.shape}]")
            elif isinstance(obj, h5py.Group):
                print(f"{indent}📁 {node_name}/")

        f.visititems(print_node)
        print("=============================\n")

except Exception as e:
    print(f"❌ 读取失败: {e}")
