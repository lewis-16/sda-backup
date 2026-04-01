#!/usr/bin/env python3
"""
生成图片caption和特征提取脚本

功能：
1. 提取NSD数据集中图片的5个captions
2. 使用CLIP ViT-L-14提取文本特征（平均）
3. 使用AlexNet fc6提取图像特征
4. 将train2017和val2017分开保存，用作训练集和测试集
"""

import numpy as np
import h5py
import pandas as pd
from pycocotools.coco import COCO
import torch
from torchvision import models, transforms
from PIL import Image
import open_clip
import pickle
import os
import warnings
import tqdm

warnings.filterwarnings('ignore')


STIMULI_HDF5 = '/media/ubuntu/sda/TrippleN/nsd_stimuli.hdf5'
STIMULI_CSV = '/media/ubuntu/sda/TrippleN/nsd_stim_info_merged.csv'
COCO_DIR = '/media/ubuntu/sda/TrippleN/coco'
CLIP_MODEL_PATH = '/media/ubuntu/sda/TrippleN/model/ViT-L-14.pt'
OUTPUT_DIR = '/media/ubuntu/sda/TrippleN/customize'

CLIP_TEXT_DIM = 768
ALEXNET_FC6_DIM = 4096


def load_stimuli_info(csv_path):
    """加载NSD刺激物信息CSV文件"""
    print(f"加载刺激物信息: {csv_path}")
    df = pd.read_csv(csv_path, index_col=0)
    print(f"  总样本数: {len(df)}")
    print(f"  train2017: {(df['cocoSplit'] == 'train2017').sum()}")
    print(f"  val2017: {(df['cocoSplit'] == 'val2017').sum()}")
    return df


def load_images_from_hdf5(hdf5_path, indices=None):
    """从HDF5文件加载图片数据
    
    Parameters:
    -----------
    hdf5_path : str
        HDF5文件路径
    indices : array-like, optional
        要加载的图片索引，如果为None则加载全部
    
    Returns:
    --------
    images : np.ndarray
        图片数组，shape为 (N, H, W, 3)
    """
    print(f"加载HDF5图片数据: {hdf5_path}")
    with h5py.File(hdf5_path, 'r') as f:
        img_data = f['imgBrick']
        if indices is not None:
            images = img_data[np.array(indices)]
        else:
            images = img_data[:]
    print(f"  图片shape: {images.shape}")
    return images


def get_coco_annotations_path(coco_dir, split):
    """根据split获取COCO annotation文件路径"""
    if split == 'val2017':
        ann_file = os.path.join(coco_dir, 'annotations', 'captions_val2017.json')
    else:
        ann_file = os.path.join(coco_dir, 'annotations', 'captions_train2017.json')
    return ann_file


def extract_captions_single_split(df_subset, coco_dir, split_name):
    """从COCO数据集中提取单个split的captions
    
    Parameters:
    -----------
    df_subset : pd.DataFrame
        包含cocoId的DataFrame（已过滤为单个split）
    coco_dir : str
        COCO数据集目录
    split_name : str
        split名称 ('train2017' 或 'val2017')
    
    Returns:
    --------
    captions_matrix : np.ndarray
        shape为 (N, 5) 的captions矩阵
    coco_ids : np.ndarray
        对应的cocoId数组
    nsd_ids : np.ndarray
        对应的nsdId数组
    """
    n_samples = len(df_subset)
    print(f"\n提取 {split_name} Captions ({n_samples} 张图片)...")

    ann_file = get_coco_annotations_path(coco_dir, split_name)
    print(f"  加载COCO {split_name} annotations...")
    coco_caps = COCO(ann_file)

    captions_matrix = np.zeros((n_samples, 5), dtype=object)
    coco_ids = np.zeros(n_samples, dtype=np.int64)
    nsd_ids = np.zeros(n_samples, dtype=np.int64)

    df_reset = df_subset.reset_index()

    for idx, row in tqdm.tqdm(df_reset.iterrows(), total=n_samples, desc="  提取Captions", ncols=80):
        coco_id = row['cocoId']
        nsd_id = row['index']

        ann_ids = coco_caps.getAnnIds(imgIds=[coco_id])
        anns = coco_caps.loadAnns(ann_ids)

        n_captions = min(5, len(anns))
        captions_matrix[idx, :n_captions] = [a['caption'] for a in anns[:n_captions]]
        coco_ids[idx] = coco_id
        nsd_ids[idx] = nsd_id

    print(f"  Captions矩阵shape: {captions_matrix.shape}")
    return captions_matrix, coco_ids, nsd_ids


def extract_captions(df, coco_dir, max_samples=None):
    """从COCO数据集中提取图片的captions（合并处理）
    
    Parameters:
    -----------
    df : pd.DataFrame
        包含cocoId和cocoSplit的DataFrame
    coco_dir : str
        COCO数据集目录
    max_samples : int, optional
        最大处理样本数，None表示全部
    
    Returns:
    --------
    captions_matrix : np.ndarray
        shape为 (N, 5) 的captions矩阵
    coco_ids : np.ndarray
        对应的cocoId数组
    nsd_ids : np.ndarray
        对应的nsdId数组
    """
    if max_samples is not None:
        df_subset = df.iloc[:max_samples]
    else:
        df_subset = df
    
    n_samples = len(df_subset)
    print(f"\n[1/4] 提取Captions ({n_samples} 张图片)...")

    captions_matrix = np.zeros((n_samples, 5), dtype=object)
    coco_ids = np.zeros(n_samples, dtype=np.int64)
    nsd_ids = np.zeros(n_samples, dtype=np.int64)

    for idx, row in tqdm.tqdm(df_subset.iterrows(), total=n_samples, desc="  提取Captions", ncols=80):
        split = row['cocoSplit']
        coco_id = row['cocoId']
        nsd_id = row.name

        ann_file = get_coco_annotations_path(coco_dir, split)

        if idx == 0 or (not hasattr(extract_captions, 'coco_caps') or
                        extract_captions.current_split != split):
            if hasattr(extract_captions, 'coco_caps'):
                del extract_captions.coco_caps
            print(f"  加载COCO {split} annotations...")
            coco_caps = COCO(ann_file)
            extract_captions.coco_caps = coco_caps
            extract_captions.current_split = split
        else:
            coco_caps = extract_captions.coco_caps

        ann_ids = coco_caps.getAnnIds(imgIds=[coco_id])
        anns = coco_caps.loadAnns(ann_ids)

        row_idx = idx if max_samples is None else idx
        captions_matrix[row_idx % n_samples, :min(5, len(anns))] = [a['caption'] for a in anns[:5]]
        coco_ids[row_idx % n_samples] = coco_id
        nsd_ids[row_idx % n_samples] = nsd_id

    print(f"  Captions矩阵shape: {captions_matrix.shape}")
    return captions_matrix, coco_ids, nsd_ids


def load_clip_model(device):
    """加载CLIP ViT-L-14模型"""
    print(f"\n[2/4] 加载CLIP ViT-L-14模型...")
    model, _, _ = open_clip.create_model_and_transforms(
        'ViT-L-14',
        pretrained=CLIP_MODEL_PATH,
        weights_only=False
    )
    model.eval()
    model = model.to(device)
    print("  CLIP ViT-L-14模型加载完成")
    return model


def extract_clip_text_features(captions_matrix, clip_model, device, batch_size=32):
    """使用CLIP提取文本特征
    
    Parameters:
    -----------
    captions_matrix : np.ndarray
        shape为 (N, 5) 的captions矩阵
    clip_model : torch.nn.Module
        CLIP模型
    device : torch.device
        计算设备
    batch_size : int
        批处理大小
    
    Returns:
    --------
    text_features : np.ndarray
        shape为 (N, 768) 的文本特征矩阵
    """
    n_images = captions_matrix.shape[0]
    print(f"\n  提取CLIP文本特征 ({n_images} 张图片)...")

    tokenizer = open_clip.get_tokenizer('ViT-L-14')
    text_features = np.zeros((n_images, CLIP_TEXT_DIM), dtype=np.float32)

    for i in tqdm.tqdm(range(0, n_images, batch_size), desc="  CLIP文本特征", ncols=80):
        end_idx = min(i + batch_size, n_images)

        batch_captions = []
        for j in range(i, end_idx):
            batch_captions.extend(captions_matrix[j].tolist())

        tokens = tokenizer(batch_captions).to(device)

        with torch.no_grad():
            features = clip_model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
            features_np = features.cpu().numpy()

        for j in range(i, end_idx):
            start_emb = (j - i) * 5
            end_emb = start_emb + 5
            text_features[j] = np.mean(features_np[start_emb:end_emb], axis=0)

    print(f"  CLIP文本特征shape: {text_features.shape}")
    return text_features


def load_alexnet(device):
    """加载AlexNet模型"""
    print(f"\n[3/4] 加载AlexNet模型...")
    alexnet = models.alexnet(weights='IMAGENET1K_V1')
    alexnet.eval()
    alexnet = alexnet.to(device)
    print("  AlexNet模型加载完成")
    return alexnet


def extract_alexnet_fc6_features(images, alexnet, device, batch_size=32):
    """提取AlexNet fc6层特征
    
    Parameters:
    -----------
    images : np.ndarray
        图片数组，shape为 (N, H, W, 3)
    alexnet : torch.nn.Module
        AlexNet模型
    device : torch.device
        计算设备
    batch_size : int
        批处理大小
    
    Returns:
    --------
    fc6_features : np.ndarray
        shape为 (N, 4096) 的fc6特征矩阵
    """
    n_images = images.shape[0]
    print(f"\n  提取AlexNet fc6特征 ({n_images} 张图片)...")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    fc6_features = np.zeros((n_images, ALEXNET_FC6_DIM), dtype=np.float32)

    for i in tqdm.tqdm(range(0, n_images, batch_size), desc="  AlexNet fc6", ncols=80):
        end_idx = min(i + batch_size, n_images)
        batch_images = images[i:end_idx]

        batch_tensors = []
        for img in batch_images:
            img_pil = Image.fromarray(img)
            img_tensor = transform(img_pil)
            batch_tensors.append(img_tensor)

        batch_tensor = torch.stack(batch_tensors).to(device)

        with torch.no_grad():
            x = alexnet.features(batch_tensor)
            x = alexnet.avgpool(x)
            x = torch.flatten(x, 1)
            fc6_activations = alexnet.classifier[1](x)
            fc6_activations = torch.nn.functional.relu(fc6_activations)
            fc6_features[i:end_idx] = fc6_activations.cpu().numpy()

    print(f"  AlexNet fc6特征shape: {fc6_features.shape}")
    return fc6_features


def clear_gpu_memory():
    """清理GPU内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def process_split(df_subset, split_name, coco_dir, hdf5_path, clip_model, alexnet, device):
    """处理单个split的数据
    
    Parameters:
    -----------
    df_subset : pd.DataFrame
        该split的数据
    split_name : str
        split名称
    coco_dir : str
        COCO数据集目录
    hdf5_path : str
        HDF5文件路径
    clip_model : torch.nn.Module
        CLIP模型
    alexnet : torch.nn.Module
        AlexNet模型
    device : torch.device
        计算设备
    
    Returns:
    --------
    output_data : dict
        处理后的数据字典
    """
    n_samples = len(df_subset)
    print(f"\n{'='*60}")
    print(f"处理 {split_name} ({n_samples} 张图片)")
    print(f"{'='*60}")
    
    indices = df_subset.index.tolist()
    
    print("\n加载HDF5图片数据...")
    images = load_images_from_hdf5(hdf5_path, indices=indices)
    
    captions_matrix, coco_ids, nsd_ids = extract_captions_single_split(
        df_subset, coco_dir, split_name
    )
    
    print("\n提取CLIP文本特征...")
    text_embeddings = extract_clip_text_features(captions_matrix, clip_model, device)
    
    print("\n提取AlexNet fc6特征...")
    alexnet_embeddings = extract_alexnet_fc6_features(images, alexnet, device)

    output_data = {
        'nsd_ids': nsd_ids,
        'coco_ids': coco_ids,
        'captions': captions_matrix,
        'images': images,
        'text_embeddings': text_embeddings,
        'alexnet_embeddings': alexnet_embeddings,
        'split': split_name
    }

    return output_data


def save_split_data(output_data, output_name):
    """保存单个split的数据"""
    output_path = os.path.join(OUTPUT_DIR, output_name)
    print(f"\n保存 {output_name} 到: {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(output_data, f)
    print(f"  保存完成!")
    return output_path


def main(num_samples=None):
    """主函数 - 分别处理train和val数据
    
    Parameters:
    -----------
    num_samples : int, optional
        每个split处理的最大样本数，None表示全部
    """
    print("=" * 60)
    print("图片Caption和特征提取 (分开保存)")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    df = load_stimuli_info(STIMULI_CSV)
    
    df_train = df[df['cocoSplit'] == 'train2017'].copy()
    df_val = df[df['cocoSplit'] == 'val2017'].copy()
    
    if num_samples is not None:
        df_train = df_train.iloc[:num_samples]
        df_val = df_val.iloc[:num_samples]
    
    print(f"\n训练集 (train2017): {len(df_train)} 张图片")
    print(f"测试集 (val2017): {len(df_val)} 张图片")
    
    clip_model = load_clip_model(device)
    alexnet = load_alexnet(device)
    
    output_train = process_split(
        df_train, 'train2017', COCO_DIR, STIMULI_HDF5,
        clip_model, alexnet, device
    )
    save_split_data(output_train, 'train_features.pkl')
    
    clear_gpu_memory()
    
    output_val = process_split(
        df_val, 'val2017', COCO_DIR, STIMULI_HDF5,
        clip_model, alexnet, device
    )
    save_split_data(output_val, 'val_features.pkl')
    
    del clip_model, alexnet
    clear_gpu_memory()
    
    print("\n" + "=" * 60)
    print("处理完成!")
    print("=" * 60)
    print(f"\n输出文件:")
    print(f"  训练集: {OUTPUT_DIR}/train_features.pkl")
    print(f"  测试集: {OUTPUT_DIR}/val_features.pkl")
    print(f"\n每个文件包含:")
    print(f"  - nsd_ids: shape {output_train['nsd_ids'].shape}")
    print(f"  - coco_ids: shape {output_train['coco_ids'].shape}")
    print(f"  - images: shape {output_train['images'].shape}")
    print(f"  - captions: shape {output_train['captions'].shape}")
    print(f"  - text_embeddings: shape {output_train['text_embeddings'].shape} (768维)")
    print(f"  - alexnet_embeddings: shape {output_train['alexnet_embeddings'].shape} (4096维)")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='生成图片caption和特征提取（分开保存）')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='每个split处理的最大样本数，默认全部')
    args = parser.parse_args()
    
    main(num_samples=args.num_samples)
