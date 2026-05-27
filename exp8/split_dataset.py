import json
import shutil
from pathlib import Path
from collections import defaultdict

# =========================
# 路径配置
# =========================
json_path = Path("./RSICD/dataset_rsicd.json")   # 标注文件
images_dir = Path("./RSICD/RSICD_images")              # 原始图像文件夹
output_dir = Path("./RSICD/images/")         # 输出根目录

train_dir = output_dir / "train"
val_dir = output_dir / "val"
test_dir = output_dir / "test"

# 创建输出文件夹
train_dir.mkdir(parents=True, exist_ok=True)
val_dir.mkdir(parents=True, exist_ok=True)
test_dir.mkdir(parents=True, exist_ok=True)

# =========================
# 读取 JSON
# =========================
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

images_info = data["images"]

# =========================
# 统计与分发
# =========================
split_map = {
    "train": train_dir,
    "val": val_dir,
    "test": test_dir
}

split_counts = defaultdict(int)
missing_files = []

for item in images_info:
    filename = item["filename"]   # 例如 airport_1.jpg
    split = item["split"]         # train / val / test

    if split not in split_map:
        print(f"[跳过] 未知 split: {split}, 文件: {filename}")
        continue

    src_path = images_dir / filename
    dst_path = split_map[split] / filename

    if not src_path.exists():
        missing_files.append(str(src_path))
        print(f"[缺失] {src_path} 不存在")
        continue

    # 复制文件到对应文件夹
    shutil.copy2(src_path, dst_path)
    split_counts[split] += 1

# =========================
# 输出结果
# =========================
print("\n分割完成！")
print(f"train: {split_counts['train']} 张")
print(f"val:   {split_counts['val']} 张")
print(f"test:  {split_counts['test']} 张")

if missing_files:
    print(f"\n有 {len(missing_files)} 个文件在 images 文件夹中未找到：")
    for p in missing_files[:20]:
        print(p)
    if len(missing_files) > 20:
        print("...（仅显示前20个）")