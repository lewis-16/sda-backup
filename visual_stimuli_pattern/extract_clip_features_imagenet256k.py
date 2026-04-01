"""
对imagenet_256k的54万张图片提取CLIP特征并保存为矩阵
参考: model_utah_train_meaning_two_stage.py
"""

import os
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from tqdm import tqdm
import glob
from pathlib import Path
import open_clip

# 配置参数
IMAGENET_256K_PATH = "/media/ubuntu/sda/visual_stimuli_pattern/imagenet_256"  # 请修改为实际的imagenet_256k路径
OUTPUT_FEATURES_PATH = "imagenet_256k_clip_features_768d.npy"  # 输出特征矩阵路径
OUTPUT_PATHS_PATH = "imagenet_256k_image_paths.txt"  # 输出图片路径列表（用于对应）
BATCH_SIZE = 256  # 批量处理大小，可根据GPU内存调整
FEATURE_DIM = 768  # CLIP ViT-L-14的特征维度
IMAGE_SIZE = 224  # CLIP输入图像尺寸
SAVE_INTERVAL = 10000  # 每处理多少张图片保存一次（避免内存溢出）

# 支持的图片扩展名
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.JPEG', '.JPG', '.PNG'}

def find_all_images(root_dir):
    """
    递归查找所有图片文件
    """
    image_paths = []
    root_path = Path(root_dir)
    
    print(f"正在扫描目录: {root_dir}")
    for ext in IMAGE_EXTENSIONS:
        pattern = f"**/*{ext}"
        found = list(root_path.glob(pattern))
        image_paths.extend([str(p) for p in found])
    
    # 去重并排序
    image_paths = sorted(list(set(image_paths)))
    print(f"找到 {len(image_paths)} 张图片")
    return image_paths

def preprocess_image_for_clip(image_path, clip_preprocess, size=224):
    """预处理图像用于CLIP"""
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((size, size))
        return clip_preprocess(img)
    except Exception as e:
        print(f"警告: 加载图片失败 {image_path}: {e}")
        # 返回一个黑色图像作为备用
        return clip_preprocess(Image.new('RGB', (size, size), (0, 0, 0)))

def extract_clip_features_batch(image_paths, clip_model, clip_preprocess, device, 
                                 batch_size=256, save_interval=10000, 
                                 output_features_path=None, output_paths_path=None):
    """
    批量提取CLIP特征（支持增量保存，避免内存溢出）
    """
    all_features = []
    valid_paths = []
    processed_count = 0
    
    # 获取CLIP模型的dtype
    clip_dtype = next(clip_model.parameters()).dtype
    
    # 如果输出文件已存在，询问是否继续
    if output_features_path and os.path.exists(output_features_path):
        response = input(f"输出文件 {output_features_path} 已存在。是否删除并重新开始？(y/n): ")
        if response.lower() == 'y':
            os.remove(output_features_path)
            if output_paths_path and os.path.exists(output_paths_path):
                os.remove(output_paths_path)
        else:
            print("使用增量追加模式...")
            # 加载已有的路径列表
            if output_paths_path and os.path.exists(output_paths_path):
                with open(output_paths_path, 'r', encoding='utf-8') as f:
                    valid_paths = [line.strip() for line in f if line.strip()]
                processed_count = len(valid_paths)
                print(f"已加载 {processed_count} 条已有路径，从第 {processed_count} 张图片继续处理")
                # 跳过已处理的图片
                image_paths = image_paths[processed_count:]
    
    # 使用tqdm显示进度
    for i in tqdm(range(0, len(image_paths), batch_size), desc="提取CLIP特征"):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        batch_valid_indices = []
        
        # 预处理当前batch的图片
        for idx, img_path in enumerate(batch_paths):
            try:
                img_tensor = preprocess_image_for_clip(img_path, clip_preprocess, size=IMAGE_SIZE)
                batch_images.append(img_tensor)
                batch_valid_indices.append(idx)
            except Exception as e:
                if processed_count % 1000 == 0:  # 减少错误输出频率
                    print(f"跳过图片 {img_path}: {e}")
                continue
        
        if len(batch_images) == 0:
            continue
        
        # 堆叠为batch
        try:
            img_batch = torch.stack(batch_images).to(device)
            img_batch = img_batch.to(clip_dtype)
            
            # 提取CLIP特征
            with torch.no_grad():
                clip_features = clip_model.encode_image(img_batch)
                clip_features = F.normalize(clip_features, dim=-1)  # L2归一化
            
            # 转移到CPU并转换为numpy
            clip_features_cpu = clip_features.cpu().numpy()
            all_features.append(clip_features_cpu)
            
            # 记录有效的图片路径
            batch_valid_paths = [batch_paths[idx] for idx in batch_valid_indices]
            valid_paths.extend(batch_valid_paths)
            
            processed_count += len(batch_images)
            
            # 定期保存，避免内存溢出
            if save_interval > 0 and processed_count % save_interval == 0:
                if len(all_features) > 0:
                    features_chunk = np.vstack(all_features)
                    # 追加保存到文件
                    if output_features_path:
                        if os.path.exists(output_features_path):
                            # 追加模式
                            existing = np.load(output_features_path)
                            combined = np.vstack([existing, features_chunk])
                            np.save(output_features_path, combined)
                            final_shape = combined.shape
                        else:
                            # 首次保存
                            np.save(output_features_path, features_chunk)
                            final_shape = features_chunk.shape
                        
                        # 同时保存路径列表
                        if output_paths_path:
                            with open(output_paths_path, 'w', encoding='utf-8') as f:
                                for path in valid_paths:
                                    f.write(f"{path}\n")
                        
                        print(f"\n已保存中间结果: {processed_count} 张图片，特征矩阵形状: {final_shape}")
                    # 清空内存
                    all_features = []
                    
        except Exception as e:
            print(f"\n处理batch时出错 (索引 {i}): {e}")
            import traceback
            traceback.print_exc()
            # 跳过这个batch，继续处理下一个
            continue
    
    # 合并剩余的特征
    if len(all_features) > 0:
        features_chunk = np.vstack(all_features)
        # 如果之前有保存的文件，需要合并
        if output_features_path and os.path.exists(output_features_path):
            existing = np.load(output_features_path)
            features_matrix = np.vstack([existing, features_chunk])
        else:
            features_matrix = features_chunk
        print(f"最终特征矩阵形状: {features_matrix.shape}")
        return features_matrix, valid_paths
    elif output_features_path and os.path.exists(output_features_path):
        # 只加载已保存的文件
        features_matrix = np.load(output_features_path)
        print(f"从文件加载特征矩阵，形状: {features_matrix.shape}")
        return features_matrix, valid_paths
    else:
        print("错误: 没有成功提取任何特征")
        return None, []

def main():
    # 检查设备 - 强制使用CUDA
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA不可用！请检查：\n"
            "1. 是否正确安装了支持CUDA的PyTorch版本\n"
            "2. 是否正确安装了NVIDIA驱动\n"
            "3. 运行命令: python -c 'import torch; print(torch.cuda.is_available())' 检查CUDA状态\n"
            "如果确实无法使用CUDA，请修改代码中的device设置"
        )
    
    device = 'cuda'
    print(f"使用设备: {device} (GPU)")
    print(f"CUDA设备: {torch.cuda.get_device_name(0)}")
    print(f"CUDA版本: {torch.version.cuda}")
    
    # 加载CLIP模型
    print("正在加载CLIP模型...")
    checkpoint_path = '/media/ubuntu/sda/visual_stimuli_pattern/ViT-L-14.pt'
    
    # 如果本地没有，尝试其他路径
    if not os.path.exists(checkpoint_path):
        possible_paths = [
            '/media/ubuntu/sda/visual_stimuli_pattern/ViT-L-14.pt',
            '/disk1/jinchentao/visual_decode/visual_reconstruction/VAR-CLIP-master/ViT-L-14.pt',
        ]
        for path in possible_paths:
            if os.path.exists(path):
                checkpoint_path = path
                break
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"找不到CLIP模型文件。请确保ViT-L-14.pt文件存在。尝试过的路径: {checkpoint_path}")
    
    print(f"加载CLIP模型: {checkpoint_path}")
    # TorchScript模型先加载到CPU，然后转移到CUDA
    clip_model = torch.jit.load(checkpoint_path, map_location='cpu')
    
    # 转移到CUDA设备
    try:
        clip_model = clip_model.to(device)
        # 测试模型是否正常工作
        test_input = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            _ = clip_model.encode_image(test_input)
        print("✓ CLIP模型加载成功并已转移到CUDA")
    except Exception as e:
        raise RuntimeError(
            f"无法将CLIP模型转移到CUDA设备: {e}\n"
            "这可能是因为：\n"
            "1. PyTorch版本与CUDA版本不匹配\n"
            "2. TorchScript模型与当前PyTorch版本不兼容\n"
            "请检查PyTorch和CUDA版本是否匹配"
        ) from e
    
    clip_model.eval()
    
    # 获取预处理器
    print("加载CLIP预处理器...")
    _, clip_preprocess, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained=None)
    
    # 查找所有图片
    if not os.path.exists(IMAGENET_256K_PATH):
        print(f"错误: imagenet_256k路径不存在: {IMAGENET_256K_PATH}")
        print("请修改脚本中的 IMAGENET_256K_PATH 变量为正确的路径")
        return
    
    image_paths = find_all_images(IMAGENET_256K_PATH)
    
    if len(image_paths) == 0:
        print("错误: 没有找到任何图片文件")
        return
    
    print(f"准备处理 {len(image_paths)} 张图片")
    print(f"批量大小: {BATCH_SIZE}")
    print(f"特征维度: {FEATURE_DIM}")
    
    # 提取特征
    print("\n开始提取CLIP特征...")
    features_matrix, valid_paths = extract_clip_features_batch(
        image_paths, 
        clip_model, 
        clip_preprocess, 
        device, 
        batch_size=BATCH_SIZE,
        save_interval=SAVE_INTERVAL,
        output_features_path=OUTPUT_FEATURES_PATH,
        output_paths_path=OUTPUT_PATHS_PATH
    )
    
    if features_matrix is None:
        print("错误: 特征提取失败")
        return
    
    # 保存特征矩阵（如果还没有保存）
    if features_matrix is not None:
        if not os.path.exists(OUTPUT_FEATURES_PATH) or SAVE_INTERVAL == 0:
            print(f"\n保存特征矩阵到: {OUTPUT_FEATURES_PATH}")
            print(f"特征矩阵形状: {features_matrix.shape} (样本数 x 特征维度)")
            np.save(OUTPUT_FEATURES_PATH, features_matrix)
            print(f"✓ 特征矩阵已保存")
        else:
            print(f"\n特征矩阵已保存在: {OUTPUT_FEATURES_PATH}")
            print(f"特征矩阵形状: {features_matrix.shape} (样本数 x 特征维度)")
        
        # 保存图片路径列表（用于对应）
        print(f"\n保存图片路径列表到: {OUTPUT_PATHS_PATH}")
        with open(OUTPUT_PATHS_PATH, 'w', encoding='utf-8') as f:
            for path in valid_paths:
                f.write(f"{path}\n")
        print(f"✓ 图片路径列表已保存 ({len(valid_paths)} 条路径)")
    
    # 打印统计信息
    print("\n" + "="*60)
    print("处理完成!")
    print("="*60)
    print(f"总图片数: {len(image_paths)}")
    print(f"成功处理: {len(valid_paths)}")
    print(f"特征矩阵形状: {features_matrix.shape}")
    print(f"特征矩阵大小: {features_matrix.nbytes / (1024**3):.2f} GB")
    print(f"输出文件:")
    print(f"  - 特征矩阵: {OUTPUT_FEATURES_PATH}")
    print(f"  - 路径列表: {OUTPUT_PATHS_PATH}")

if __name__ == "__main__":
    main()

