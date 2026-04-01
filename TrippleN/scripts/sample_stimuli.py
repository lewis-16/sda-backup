import scipy.io
import os
import shutil
import numpy as np

# Load cluster info
mat = scipy.io.loadmat('/media/ubuntu/sda/TrippleN/ClusInfo.mat')
cluster_idx = mat['Cluster_idx'].flatten()  # 1000 elements, values 1-12
print(f"类别分布: {np.bincount(cluster_idx)[1:]}")  # skip index 0

# 按类别抽样，每类至少抽41，不足则全抽
np.random.seed(42)
selected_files = []
target_per_category = 500 // 12  # 41 per category
print(f"每类目标抽样: {target_per_category}张")

total_selected = 0
for cat in range(1, 13):
    # 找出该类别的图片索引 (1-based for filename)
    indices = np.where(cluster_idx == cat)[0] + 1  # 转为1-based文件名
    n_available = len(indices)
    n_select = min(target_per_category, n_available)
    selected = np.random.choice(indices, size=n_select, replace=False)
    selected_files.extend([f"{i:04d}.bmp" for i in selected])
    total_selected += n_select
    print(f"  类别{cat}: {n_available}张 -> 抽{n_select}张")

print(f"从1000张中抽取: {total_selected}张")

# 补抽到500张
if total_selected < 500:
    remaining = set(range(1, 1001)) - set(int(f[:4]) for f in selected_files)
    extra = np.random.choice(list(remaining), size=500 - total_selected, replace=False)
    selected_files.extend([f"{i:04d}.bmp" for i in extra])
    print(f"补抽{500-total_selected}张, 总共{len(selected_files)}张")

print(f"从1000张中抽取: {len(selected_files)}张")

# 加上后面72张 (MFOB001-MFOB072)
extra_72 = [f"MFOB{i:03d}.bmp" for i in range(1, 73)]
print(f"加上72张: {len(extra_72)}张")

all_files = selected_files + extra_72
print(f"总共: {len(all_files)}张")

# 创建目标目录
dest_dir = '/media/ubuntu/sda/TrippleN/stimuli_new'
os.makedirs(dest_dir, exist_ok=True)

# 复制文件
src_dir = '/media/ubuntu/sda/TrippleN/stimuli'
for f in all_files:
    src = os.path.join(src_dir, f)
    dst = os.path.join(dest_dir, f)
    if not os.path.exists(src):
        print(f"Warning: {src} not found")
    else:
        shutil.copy2(src, dst)

print(f"完成! 复制到 {dest_dir}")
print(f"目标目录文件数: {len(os.listdir(dest_dir))}")
