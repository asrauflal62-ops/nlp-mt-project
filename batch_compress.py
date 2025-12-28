import torch
import os

# 设置要扫描的根目录 (通常是 runs)
root_dir = "runs"

print(f"🔍 开始扫描 {root_dir} 下的所有模型文件...")

count = 0
for dirpath, dirnames, filenames in os.walk(root_dir):
    for filename in filenames:
        # 找到是以 .pt 结尾，且不是已经压缩过的文件
        if filename.endswith(".pt") and "lite" not in filename:
            full_path = os.path.join(dirpath, filename)
            # 生成新文件名，例如 best.pt -> best_lite.pt
            new_filename = filename.replace(".pt", "_lite.pt")
            new_full_path = os.path.join(dirpath, new_filename)

            # 如果 lite 文件已经存在，跳过 (避免重复跑)
            if os.path.exists(new_full_path):
                continue

            print(f"\n👉 处理中: {full_path}")
            
            try:
                # 1. 加载模型 (CPU模式)
                checkpoint = torch.load(full_path, map_location='cpu')
                
                # 2. 剥离优化器状态
                lite_checkpoint = {
                    'model': checkpoint['model'] if 'model' in checkpoint else checkpoint,
                    'config': checkpoint.get('config', {}),
                    'vocab': checkpoint.get('vocab', {})
                }
                
                # 3. 保存 Lite 版本
                torch.save(lite_checkpoint, new_full_path)
                
                # 4. 打印大小对比
                old_size = os.path.getsize(full_path) / (1024*1024)
                new_size = os.path.getsize(new_full_path) / (1024*1024)
                
                print(f"   ✅ 成功生成: {new_filename}")
                print(f"   📉 体积压缩: {old_size:.2f} MB -> {new_size:.2f} MB")
                
                if new_size > 100:
                    print("   ⚠️ 警告: 压缩后依然超过 100MB，GitHub 可能拒收。")
                
                count += 1
            except Exception as e:
                print(f"   ❌ 处理失败: {e}")

print(f"\n🎉 全部完成！共生成了 {count} 个 Lite 模型。")
