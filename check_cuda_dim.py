import torch

# 模拟我们之前的修复逻辑
bsize = 1
h, w = 480, 640
patch_size = 14 # 假设值，我们需要确认
h_feat = h // patch_size
w_feat = w // patch_size
t_feat = 3 # 我们强注的 3 张图

image_grid_thw = torch.tensor([[t_feat, h_feat, w_feat]] * bsize)
total_tokens = image_grid_thw.prod()
print(f"\n🎯 预估参数分析:")
print(f"➡️ 单图 H_feat: {h_feat}, W_feat: {w_feat}")
print(f"➡️ 总 Temporal (相机数): {t_feat}")
print(f"➡️ 总计算 Token 数: {total_tokens}")

if total_tokens > 2000:
    print("\n⚠️ 警报: Token 数量可能超过了 Qwen2-VL 视觉编码器的预设缓存范围！")
