# Ravens任务总览图生成器（纯Python版本）

import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def check_available_tasks():
    """检查可用的任务数据"""
    print("=== 检查可用任务 ===")
    
    available_tasks = []
    for item in os.listdir('.'):
        if os.path.isdir(item) and item.endswith('-train'):
            task_name = item.replace('-train', '')
            available_tasks.append(task_name)
    
    print(f"找到 {len(available_tasks)} 个任务: {available_tasks}")
    return available_tasks

def extract_task_image(dataset_dir):
    """从数据集中提取任务图像"""
    color_dir = os.path.join(dataset_dir, 'color')
    
    if not os.path.exists(color_dir):
        print(f"❌ {dataset_dir} 没有color目录")
        return None
    
    pkl_files = glob.glob(os.path.join(color_dir, '*.pkl'))
    
    if not pkl_files:
        print(f"❌ {dataset_dir} 没有pkl文件")
        return None
    
    # 使用第一个可用文件
    target_file = pkl_files[0]
    print(f"📂 读取文件: {target_file}")
    
    try:
        with open(target_file, 'rb') as f:
            image_data = pickle.load(f)
        
        print(f"📊 图像数据类型: {type(image_data)}")
        
        # 处理不同的图像数据格式
        if isinstance(image_data, np.ndarray):
            print(f"📏 数组形状: {image_data.shape}")
            
            if len(image_data.shape) == 5:
                # (batch, camera, height, width, channels)
                img = image_data[0, 0]
                print(f"✂️  提取形状: {img.shape}")
            elif len(image_data.shape) == 4:
                img = image_data[0]
            elif len(image_data.shape) == 3:
                img = image_data
            else:
                print(f"❌ 不支持的图像形状: {image_data.shape}")
                return None
                
        elif isinstance(image_data, dict):
            print(f"📋 字典键: {list(image_data.keys())}")
            if 'color' in image_data:
                img = image_data['color']
            else:
                print("❌ 字典中没有'color'键")
                return None
        else:
            print(f"❌ 不支持的数据类型: {type(image_data)}")
            return None
        
        # 确保像素值在正确范围
        if img.max() > 1.0:
            img = img / 255.0
        
        print(f"✅ 成功提取图像，最终形状: {img.shape}")
        return img
        
    except Exception as e:
        print(f"❌ 读取 {target_file} 失败: {e}")
        return None

def create_task_grid(tasks):
    """创建任务网格图"""
    print(f"\n=== 创建 {len(tasks)} 个任务的网格图 ===")
    
    # 收集所有图像
    images = {}
    valid_tasks = []
    
    for task in tasks:
        dataset_dir = f'{task}-train'
        print(f"\n处理任务: {task}")
        
        img = extract_task_image(dataset_dir)
        if img is not None:
            images[task] = img
            valid_tasks.append(task)
            print(f"✅ {task} 图像提取成功")
        else:
            print(f"❌ {task} 图像提取失败")
    
    if not valid_tasks:
        print("❌ 没有可用的图像数据")
        return
    
    print(f"\n📊 总共可用图像: {len(valid_tasks)}")
    
    # 计算布局 - 强制使用2x5布局（适合10个任务）
    n_tasks = len(valid_tasks)
    if n_tasks <= 5:
        rows, cols = 1, n_tasks
        figsize = (4 * n_tasks, 4)
    else:
        # 对于6个或更多任务，使用2x5布局
        rows, cols = 2, 5
        figsize = (20, 8)
    
    print(f"📐 布局: {rows}行 x {cols}列")
    
    # 创建图形 - 2x5布局
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    # 确保axes是2D数组，方便索引
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    # 绘制每个任务
    for i, task in enumerate(valid_tasks):
        row = i // cols
        col = i % cols
        
        axes[row, col].imshow(images[task])
        
        # 任务标题
        title = task.replace('-', ' ').title()
        axes[row, col].set_title(title, fontsize=12, fontweight='bold', pad=10)
        axes[row, col].axis('off')
    
    # 隐藏多余的子图
    for i in range(len(valid_tasks), rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    # 调整布局
    plt.tight_layout()
    
    # 添加总标题
    fig.suptitle('Ravens Framework: Collected Task Data', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.subplots_adjust(top=0.92)
    
    # 保存图像
    filename = 'ravens_task_overview.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"\n🎉 任务总览图已保存: {filename}")
    plt.show()

def main():
    """主函数"""
    print("🚀 Ravens任务总览图生成器")
    
    # 检查可用任务
    available_tasks = check_available_tasks()
    
    if not available_tasks:
        print("❌ 没有找到任务数据")
        print("请先收集数据:")
        print("python ravens/demos.py --assets_root=./ravens/environments/assets/ --disp=True --task=block-insertion --mode=train --n=2")
        return
    
    if len(available_tasks) < 2:
        print("⚠️  只有1个任务，建议收集更多任务数据")
    
    # 生成任务网格图
    create_task_grid(available_tasks)

if __name__ == "__main__":
    main()
