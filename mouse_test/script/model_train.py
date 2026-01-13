from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS
from torchvision.transforms import InterpolationMode, transforms

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from random import shuffle
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import gc
import shutil
import sys
import time
import warnings
from functools import partial

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import pickle

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
from torchvision import transforms
import random
import PIL.Image as PImage, PIL.ImageDraw as PImageDraw
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed

# 导入指定目录下的models模块
import os
models_dir = "/disk1/jinchentao/visual_decode/visual_reconstruction/my_VAR_mua/results_260106_monkeyF_train"
sys.path.insert(0, models_dir)
from models import VQVAE, build_vae_var
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pytorch_msssim import ssim
from tqdm import tqdm
import math

# 导入CLIP相关模块
import open_clip
import torch.nn.functional as F

# 定义PSTH数据路径（留一法版本和原始版本）
base_data_dir = "/disk1/jinchentao/visual_decode/visual_reconstruction/fig5/mouse1/date_1215_results"
psth_loo_path = os.path.join(base_data_dir, "psth_matrix_loo.npy")
psth_raw_path = os.path.join(base_data_dir, "psth_matrix_raw.npy")
trial_image_id_path = os.path.join(base_data_dir, "trial_image_id.pkl")

def load_psth_data(psth_path, psth_type='loo'):
    """
    加载PSTH数据
    Args:
        psth_path: PSTH数据文件路径
        psth_type: 'loo' 或 'raw'，用于日志显示
    Returns:
        train_MUA: PSTH数据数组
    """
    print(f"\n{'='*60}")
    print(f"加载PSTH数据 ({psth_type.upper()}版本): {psth_path}")
    print(f"{'='*60}")
    
    train_MUA = np.load(psth_path)
    print(f"训练MUA数据形状: {train_MUA.shape}")
    
    # 检查数据中的NaN和Inf
    print(f"数据统计:")
    print(f"  - 包含NaN: {np.isnan(train_MUA).any()}")
    print(f"  - 包含Inf: {np.isinf(train_MUA).any()}")
    print(f"  - 最小值: {train_MUA.min():.6f}")
    print(f"  - 最大值: {train_MUA.max():.6f}")
    print(f"  - 均值: {train_MUA.mean():.6f}")
    print(f"  - 标准差: {train_MUA.std():.6f}")
    
    # 处理可能的NaN和Inf
    if np.isnan(train_MUA).any() or np.isinf(train_MUA).any():
        print("警告: 数据中包含NaN或Inf，将进行清理...")
        train_MUA = np.nan_to_num(train_MUA, nan=0.0, posinf=1.0, neginf=-1.0)
    
    return train_MUA

# 加载图像ID（两个版本共享）
with open(trial_image_id_path, 'rb') as f:
    trial_image_ids = pickle.load(f)  # list of image IDs (strings)

print(f"加载了 {len(trial_image_ids)} 个图像ID")

# 加载CLIP模型 (ViT-H-14) - 参考 model_utah_train_meaning_two_stage_H.py
device_for_clip = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\n加载CLIP模型 (ViT-H-14) 到设备: {device_for_clip}")

# 加载CLIP模型 - 使用ViT-H-14，从本地预训练权重加载
# 注意：open_clip.create_model_and_transforms返回 (model, preprocess_train, preprocess_val)
# 参考文件中使用第三个返回值（preprocess_val）作为clip_preprocess
clip_model_path = '/disk1/jinchentao/visual_decode/visual_reconstruction/my_VAR_mua/open_clip_pytorch_model.bin'
if os.path.exists(clip_model_path):
    print(f"从本地路径加载CLIP预训练权重: {clip_model_path}")
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        'ViT-H-14', 
        pretrained=clip_model_path
    )
else:
    print(f"警告: 本地权重文件不存在 ({clip_model_path})，尝试从laion2b加载")
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        'ViT-H-14',
        pretrained='laion2b_s32b_b79k'  # 使用在线预训练权重
    )
clip_model = clip_model.to(device_for_clip)
clip_model.eval()
print(f"✓ CLIP模型加载完成 (ViT-H-14, 特征维度: 1024)")

# 图像预处理函数（用于CLIP）
def preprocess_image_for_clip(image_path, size=224):
    """预处理图像用于CLIP"""
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((size, size))
        return clip_preprocess(img)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        # 返回一个空图像
        return clip_preprocess(Image.new('RGB', (size, size), (0, 0, 0)))


# 时间序列EP编码器
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1)]
        return x


class TemporalEPEncoder(nn.Module):
    """
    时间序列EP编码器（使用Conv1D替代Transformer）
    输入: (B, time_bins, neurons) = (B, 99, 51)
    输出: 
        - tokens: (B, n_token, Cvae) = (B, 128, 32) - 用于VAR模型的交叉注意力
        - condition_vector: (B, 1024) - 用于对比学习（与CLIP ViT-H-14对齐）
    
    架构特点：
    - 使用Conv1D处理时间维度（保留时间建模，但更轻量）
    - 先对神经元维度降维，减少参数量
    - 生成token序列用于VAR模型的交叉注意力
    - 同时生成1024维条件向量用于对比学习
    """
    def __init__(self, input_dim=93, time_bins=31, d_model=32, n_token=128, 
                 num_conv_layers=2, dropout=0.2, Cvae=32):
        super().__init__()
        self.input_dim = input_dim
        self.time_bins = time_bins
        self.d_model = d_model
        self.n_token = n_token
        self.Cvae = Cvae  # VAR模型的Cvae维度（默认32）
        
        # 输入投影：先对神经元维度降维，减少参数量
        # 对于大输入维度（如1000），使用多层降维
        if input_dim > 200:
            # 大维度：分步降维
            hidden_dim = min(input_dim // 4, d_model * 8)
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, d_model * 4),
                nn.LayerNorm(d_model * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model),
            )
        else:
            # 小维度：直接投影
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, d_model * 4),
                nn.LayerNorm(d_model * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model),
            )
        
        # 时间维度的1D卷积（替代Transformer，保留时间建模）
        # 使用渐进式通道扩展：d_model -> d_model*2 -> ... -> d_model
        conv_layers = []
        for i in range(num_conv_layers):
            if i == 0:
                in_channels = d_model
            else:
                in_channels = d_model * 2
            
            if i == num_conv_layers - 1:
                out_channels = d_model  # 最后一层回到d_model
            else:
                out_channels = d_model * 2  # 中间层扩展到d_model*2
            
            conv_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
        self.temporal_conv = nn.Sequential(*conv_layers)
        
        # 自适应池化到固定长度
        self.adaptive_pool = nn.AdaptiveAvgPool1d(n_token)
        
        # 最终投影层（如果需要进一步调整特征）
        self.final_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)  # 最终层使用较小的dropout
        )
        
        # Token投影到Cvae维度（用于VAR模型）
        # VAR模型期望tokens是(B, n_token, Cvae)，其中Cvae=32
        self.token_to_cvae = nn.Sequential(
            nn.Linear(d_model, Cvae),
            nn.LayerNorm(Cvae)
        )
        
        # 条件向量投影层：从token特征中提取1024维条件向量（用于对比学习）
        # 这个条件向量将与CLIP特征进行对比学习
        self.condition_proj = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.LayerNorm(d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model * 4, 1024),  # CLIP ViT-H-14的特征维度是1024
            nn.LayerNorm(1024)
        )
        
        # 位置编码：为token序列添加位置信息（可学习的嵌入）
        # 由于使用自适应池化，输出长度固定为n_token，所以位置编码大小也是n_token
        self.pos_embed = nn.Parameter(torch.zeros(1, n_token, d_model))
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # 初始化位置编码
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
    def forward(self, x, return_condition_vector=False):
        """
        前向传播：同时生成tokens和条件向量
        Args:
            x: (B, time_bins, input_dim) = (B, 99, 51) 或 (B, 40, 1000)
            return_condition_vector: 是否返回条件向量（用于对比学习）
        Returns:
            - 如果return_condition_vector=False: 返回tokens (B, n_token, Cvae) - 用于VAR模型
            - 如果return_condition_vector=True: 返回(tokens, condition_vector)
                - tokens: (B, n_token, Cvae) = (B, 128, 32) - 用于VAR模型的交叉注意力
                - condition_vector: (B, 1024) - 用于对比学习（与CLIP ViT-H-14对齐）
        """
        B = x.shape[0]
        
        # 检查输入
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"警告: 输入包含NaN或Inf，将进行清理")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 1. 输入投影（对神经元维度降维）
        x = self.input_proj(x)  # (B, time_bins, d_model)
        
        # 检查中间结果
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"警告: 输入投影输出包含NaN或Inf")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 2. 转换为卷积输入格式 (B, d_model, time_bins)
        x = x.transpose(1, 2)  # (B, d_model, time_bins)
        
        # 3. 时间维度的1D卷积（保留时间建模）
        x = self.temporal_conv(x)  # (B, d_model, time_bins)
        
        # 检查中间结果
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"警告: 卷积输出包含NaN或Inf")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 4. 自适应池化到n_token长度
        x = self.adaptive_pool(x)  # (B, d_model, n_token)
        
        # 5. 转回 (B, n_token, d_model)
        x = x.transpose(1, 2)  # (B, n_token, d_model)
        
        # 6. 最终投影
        x = self.final_proj(x)  # (B, n_token, d_model)
        
        # 7. 添加位置编码
        x = x + self.pos_embed  # (B, n_token, d_model)
        
        # 8. 投影到Cvae维度，生成tokens（用于VAR模型）
        tokens = self.token_to_cvae(x)  # (B, n_token, Cvae)
        
        # 检查tokens
        if torch.isnan(tokens).any() or torch.isinf(tokens).any():
            print(f"警告: tokens包含NaN或Inf")
            tokens = torch.nan_to_num(tokens, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 如果只需要tokens，直接返回
        if not return_condition_vector:
            return tokens  # (B, n_token, Cvae)
        
        # 9. 对token序列进行平均池化，得到全局表示（用于生成条件向量）
        token_mean = x.mean(dim=1)  # (B, d_model)
        
        # 10. 投影到1024维条件向量空间（用于对比学习）
        condition_vector = self.condition_proj(token_mean)  # (B, 1024)
        
        # 检查条件向量
        if torch.isnan(condition_vector).any() or torch.isinf(condition_vector).any():
            print(f"警告: 条件向量包含NaN或Inf")
            condition_vector = torch.nan_to_num(condition_vector, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return tokens, condition_vector  # (B, n_token, Cvae), (B, 1024)


class MUAClassificationDataset(Dataset):
    def __init__(self, mua_data, labels, image_paths=None, transform=None, load_clip_features=True, clip_model=None, device='cuda'):
        # 保持3D数据格式 (trials, time_bins, neurons)
        self.mua_data = torch.tensor(mua_data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.image_paths = image_paths
        self.transform = transform
        self.load_clip_features = load_clip_features
        self.clip_model = clip_model
        self.device = device
        
        # 图像变换
        self.image_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2 - 1)  # 归一化到[-1, 1]
        ])
        
        # 空图像张量作为备用
        self.empty_image_tensor = torch.zeros(3, 256, 256, dtype=torch.float32)
        
        # 预计算CLIP特征（如果启用）
        self.clip_features = None
        if self.load_clip_features and self.image_paths is not None and self.clip_model is not None:
            self._precompute_clip_features()
        
    def _precompute_clip_features(self):
        """预计算所有图像的CLIP特征（带缓存机制）"""
        import hashlib
        
        # 创建缓存目录
        cache_dir = '/disk1/jinchentao/visual_decode/visual_reconstruction/my_VAR_mua/clip_features_vitH_cache'
        os.makedirs(cache_dir, exist_ok=True)
        
        print("开始预计算CLIP特征（使用缓存机制）...")
        self.clip_features = []
        
        cached_count = 0
        computed_count = 0
        
        for i, image_path in enumerate(self.image_paths):
            if i % 1000 == 0:
                print(f"处理进度: {i}/{len(self.image_paths)} (缓存:{cached_count}, 计算:{computed_count})")
            
            # 生成缓存文件名（基于图像路径的hash）
            cache_key = hashlib.md5(image_path.encode()).hexdigest()
            cache_path = os.path.join(cache_dir, f"{cache_key}.pt")
            
            # 检查缓存是否存在
            if os.path.exists(cache_path):
                try:
                    cached_feature = torch.load(cache_path, map_location='cpu', weights_only=False)
                    self.clip_features.append(cached_feature)
                    cached_count += 1
                    continue
                except Exception as e:
                    print(f"Warning: Failed to load cache for {image_path}: {e}")
            
            # 计算CLIP特征
            try:
                # 预处理图像
                img_tensor = preprocess_image_for_clip(image_path)
                img_batch = img_tensor.unsqueeze(0).to(self.device)
                
                # 检查CLIP模型的dtype并转换输入
                clip_dtype = next(self.clip_model.parameters()).dtype
                img_batch = img_batch.to(clip_dtype)
                
                # 提取CLIP特征
                with torch.no_grad():
                    clip_feature = self.clip_model.encode_image(img_batch)
                    clip_feature = F.normalize(clip_feature, dim=-1)
                
                clip_feature_cpu = clip_feature.cpu()
                self.clip_features.append(clip_feature_cpu)
                
                # 保存到缓存
                try:
                    torch.save(clip_feature_cpu, cache_path)
                    computed_count += 1
                except Exception as e:
                    print(f"Warning: Failed to save cache for {image_path}: {e}")
                
            except Exception as e:
                print(f"Error processing image {i}: {e}")
                # 使用零特征作为备用
                self.clip_features.append(torch.zeros(1, 1024))  # CLIP ViT-H-14的特征维度是1024
        
        print(f"CLIP特征预计算完成: 总计{len(self.image_paths)}个, 缓存{cached_count}个, 新计算{computed_count}个")
        
    def __len__(self):
        return len(self.mua_data)
    
    def __getitem__(self, idx):
        # 返回完整的3D数据 (time_bins, neurons) = (99, 51)
        mua_sample = self.mua_data[idx]  # (99, 51)
        label = self.labels[idx]
        
        # 处理图像
        if self.image_paths is not None and idx < len(self.image_paths) and self.image_paths[idx] is not None:
            image_path = self.image_paths[idx]
            try:
                img = Image.open(image_path).convert('RGB')
                img = img.resize((256, 256))
                img_array = np.array(img, dtype=np.float32) / 255.0
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
                img_tensor = img_tensor * 2 - 1  # 归一化到[-1, 1]
            except (FileNotFoundError, OSError) as e:
                img_tensor = self.empty_image_tensor.clone()
                print(f"Warning: {image_path} not found or corrupted: {e}")
        else:
            img_tensor = self.empty_image_tensor.clone()
        
        # 获取CLIP特征
        if self.clip_features is not None:
            clip_feature = self.clip_features[idx].squeeze(0)  # 移除batch维度
        else:
            clip_feature = torch.zeros(1024)  # CLIP ViT-H-14的特征维度是1024
        
        if self.transform:
            mua_sample = self.transform(mua_sample)
            
        return mua_sample, img_tensor, label, clip_feature

def create_label_mapping_from_classes(class_labels, unique_classes):
    """
    根据类别名称创建标签映射
    Args:
        class_labels: 类别名称列表
        unique_classes: 唯一类别列表
    Returns:
        mapped_labels: 映射后的数值标签
        label_mapping: 标签映射字典
    """
    # 创建类别名称到数值的映射
    class_to_idx = {class_name: idx for idx, class_name in enumerate(unique_classes)}
    
    # 将类别名称转换为数值标签
    mapped_labels = np.array([class_to_idx[class_name] for class_name in class_labels])
    
    return mapped_labels, class_to_idx

# 重写VAR模型的forward方法（在train_with_psth_type函数中使用）
def var_forward_with_ep_encoder_tokens(self, neuron_activity, x_BLCv_wo_first_l):
    """
    修改后的VAR forward方法，使用EP_encoder的tokens输出
    根据VAR模型的原始实现逻辑：
    1. EP_encoder返回tokens: (B, n_token, Cvae) = (B, 128, 32)
    2. 通过对tokens做平均得到条件向量（用于AdaLN）
    3. 将tokens通过word_embed投影到embed_dim用于交叉注意力
    :param neuron_activity: 神经活动输入 (B, time_bins, input_dim) = (B, 99, 51)
    :param x_BLCv_wo_first_l: teacher forcing input (B, self.L-self.first_l, self.Cvae)
    :return: logits BLV, V is vocab_size
    """
    # 1. 获取EP_encoder输出的tokens序列（用于VAR模型的交叉注意力）
    neuro_tokens = self.neuro_encoder(neuron_activity, return_condition_vector=False)  # (B, n_token, Cvae)
    assert neuro_tokens.shape[1] == 128, f"neuro_tokens的token数量({neuro_tokens.shape[1]})应该是128"
    assert neuro_tokens.shape[2] == self.Cvae, f"neuro_tokens的Cvae维度({neuro_tokens.shape[2]})应该是{self.Cvae}"
    
    # 2. 先对神经token做平均，得到Cvae维度的向量
    neuro_token_mean = neuro_tokens.mean(dim=1)  # (B, Cvae)
    
    # 3. 将平均后的向量投影到embed_dim，生成条件嵌入（用于AdaLN）
    cond_BD = self.word_embed(neuro_token_mean)  # (B, embed_dim)
    assert cond_BD.shape[-1] == self.C, f"cond_BD维度({cond_BD.shape[-1]})与self.C({self.C})不匹配，请检查word_embed的输出维度"
    
    # 4. 将神经token投影到embed_dim（用于交叉注意力）
    neuro_tokens_proj = self.word_embed(neuro_tokens)  # (B, n_token, embed_dim)
    assert neuro_tokens_proj.shape[-1] == self.C, f"neuro_tokens_proj维度({neuro_tokens_proj.shape[-1]})与self.C({self.C})不匹配"
    
    # 4. 继续使用原始forward方法的后续逻辑
    bg, ed = self.begin_ends[self.prog_si] if self.prog_si >= 0 else (0, self.L)
    B = x_BLCv_wo_first_l.shape[0]
    
    with torch.amp.autocast('cuda', enabled=False):
        sos = cond_BD.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1)
        assert sos.shape[-1] == self.pos_start.shape[-1], f"sos维度({sos.shape[-1]})与pos_start维度({self.pos_start.shape[-1]})不匹配"
        
        if self.prog_si == 0: 
            x_BLC = sos
        else: 
            x_BLC = torch.cat((sos, self.word_embed(x_BLCv_wo_first_l.float())), dim=1)
        x_BLC += self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1)) + self.pos_1LC[:, :ed]
    
    attn_bias = self.attn_bias_for_masking[:, :, :ed, :ed]
    
    cond_BD_or_gss = self.shared_ada_lin(cond_BD)
    
    # hack: get the dtype if mixed precision is used
    temp = x_BLC.new_ones(8, 8)
    main_type = torch.matmul(temp, temp).dtype
    
    x_BLC = x_BLC.to(dtype=main_type)
    cond_BD_or_gss = cond_BD_or_gss.to(dtype=main_type)
    attn_bias = attn_bias.to(dtype=main_type)
    neuro_tokens_proj = neuro_tokens_proj.to(dtype=main_type)
    
    for i, b in enumerate(self.blocks):
        x_BLC = b(x=x_BLC, cond_BD=cond_BD_or_gss, attn_bias=attn_bias, neuro_tokens=neuro_tokens_proj)
    x_BLC = self.get_logits(x_BLC.float(), cond_BD)
    
    if self.prog_si == 0:
        if isinstance(self.word_embed, nn.Linear):
            x_BLC[0, 0, 0] += self.word_embed.weight[0, 0] * 0 + self.word_embed.bias[0] * 0
        else:
            s = 0
            for p in self.word_embed.parameters():
                if p.requires_grad:
                    s += p.view(-1)[0] * 0
            x_BLC[0, 0, 0] += s
    
    return x_BLC  # logits BLV, V is vocab_size

# 对比学习损失函数
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, features_1, features_2):
        """
        计算对比学习损失
        Args:
            features_1: EP_encoder特征 [batch_size, feature_dim]
            features_2: CLIP特征 [batch_size, feature_dim]
        """
        # 归一化特征
        features_1 = F.normalize(features_1, dim=1)
        features_2 = F.normalize(features_2, dim=1)
        
        # 计算相似度矩阵
        logits_12 = torch.matmul(features_1, features_2.T) / self.temperature
        logits_21 = torch.matmul(features_2, features_1.T) / self.temperature
        
        # 创建正样本标签（对角线）
        batch_size = features_1.size(0)
        labels = torch.arange(batch_size, device=features_1.device)
        
        # 计算交叉熵损失
        loss_12 = F.cross_entropy(logits_12, labels)
        loss_21 = F.cross_entropy(logits_21, labels)
        
        return (loss_12 + loss_21) / 2

class TrainingConfig:
    batch_size = 8
    num_workers = 0
    image_size = 256
    
    distributed = True
    backend = 'nccl' 
    init_method = 'env://'
    
    vae_config = {
        "in_channel": 3,
        "channel": 128,
        "n_res_block": 2,
        "n_res_channel": 64,
        "embed_dim": 64,
        "n_embed": 8192,
        "decay": 0.99
    }
    var_config = {
        "num_classes": 1000,
        "depth": 32,
        "embed_dim": 1024,
        "num_heads": 32,
        "mlp_ratio": 4.0,
        "drop_rate": 0.1,
        "attn_drop_rate": 0.0,
        "drop_path_rate": 0.1,
        "norm_eps": 1e-6,
        "shared_aln": True,
        "cond_drop_rate": 0.1,
        "attn_l2_norm": False,
        "patch_nums": (1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
        "flash_if_available": True,
        "fused_if_available": True,
    }
    
    # 微调配置
    freeze_pretrained = True  # 是否冻结预训练层（设置为True以完全冻结）
    new_layer_lr_multiplier = 10.0  # 新增层的学习率倍数（相对于基础学习率）
    
    lr = 1e-4
    weight_decay = 0.05
    betas = (0.9, 0.95)
    # EP_encoder专用学习率（降低以避免梯度爆炸）
    ep_encoder_lr_multiplier = 1.0  # 从10.0降低到1.0，使用基础学习率
    
    # 两步训练配置
    stage1_epochs = 20  # 第一阶段：EP_encoder与CLIP对齐训练
    stage2_epochs = 6  # 第二阶段：VAR模型训练
    contrastive_temp = 0.07  # 对比学习温度参数
    
    grad_accum = 1
    label_smooth = 0.1
    amp_enabled = True
    
    # 渐进式训练配置
    enable_progressive_training = False  # 是否启用渐进式训练（设置为False以关闭）
    prog_epochs = 10 
    prog_warmup_iters = 1000  
    
    log_interval = 50
    eval_interval = 1
    save_interval = 5
    ssim_interval = 1  # 每个epoch都计算SSIM并重建图像
    checkpoint_dir = "checkpoints"
    log_dir = "logs"
    # results_dir 将在训练时根据PSTH类型动态设置
    base_results_dir = "/disk1/jinchentao/visual_decode/visual_reconstruction/fig5/mouse1/date_1215_results"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42
    
    def __init__(self, psth_type='loo'):
        """
        初始化训练配置
        Args:
            psth_type: 'loo' 或 'raw'，指定使用哪种PSTH数据
        """
        self.psth_type = psth_type
        # 根据PSTH类型设置结果目录
        if psth_type == 'loo':
            self.results_dir = os.path.join(self.base_results_dir, "train_results_loo")
        elif psth_type == 'raw':
            self.results_dir = os.path.join(self.base_results_dir, "train_results_raw")
        else:
            raise ValueError(f"未知的PSTH类型: {psth_type}，必须是'loo'或'raw'")

class VARTrainer(object):
    def __init__(
        self, device, patch_nums, resos,
        vae_local, var_model, optimizer: torch.optim.Optimizer, label_smooth: float,
        clip_model=None, amp_enabled: bool = False, rank: int = 0
    ):
        super(VARTrainer, self).__init__()
        
        self.var_model = var_model
        self.vae_local = vae_local
        self.clip_model = clip_model
        self.quantize_local = vae_local.quantize
        self.optimizer = optimizer
        self.amp_enabled = amp_enabled
        self.device = device
        self.rank = rank
        
        if hasattr(self.var_model, 'rng'):
            self.var_model.rng = torch.Generator(device=device)
        
        self.label_smooth = label_smooth
        self.train_loss = nn.CrossEntropyLoss(label_smoothing=label_smooth, reduction='none')
        self.val_loss = nn.CrossEntropyLoss(label_smoothing=0.0, reduction='mean')
        
        # 对比学习损失
        self.contrastive_loss = ContrastiveLoss(temperature=0.07)
        
        self.L = sum(pn * pn for pn in patch_nums)
        self.last_l = patch_nums[-1] * patch_nums[-1]
        self.loss_weight = torch.ones(1, self.L, device=device) / self.L
        
        self.patch_nums, self.resos = patch_nums, resos
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(patch_nums):
            self.begin_ends.append((cur, cur + pn * pn))
            cur += pn*pn
        
        self.prog_it = 0
        self.last_prog_si = -1
        self.first_prog = True
        
        # 使用新的GradScaler API
        self.scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)
        
        # 训练阶段标志
        self.stage = 1  # 1: EP_encoder对齐, 2: VAR训练
    
    def set_stage(self, stage):
        """设置训练阶段"""
        self.stage = stage
        if stage == 2:
            # 第二阶段：冻结EP_encoder的条件向量部分（condition_proj），保留token生成部分可训练
            # 只冻结condition_proj，保留其他部分（input_proj, temporal_conv, final_proj, token_to_cvae, pos_embed）可训练
            for name, param in self.var_model.neuro_encoder.named_parameters():
                if 'condition_proj' in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            print("EP_encoder的condition_proj已冻结，token生成部分（input_proj, temporal_conv, final_proj, token_to_cvae）可训练")
        else:
            # 第一阶段：解冻整个EP_encoder（包括tokens和condition_proj）
            for param in self.var_model.neuro_encoder.parameters():
                param.requires_grad = True
            print("EP_encoder已解冻（训练tokens和condition_proj）")
    
    def train_step_stage1(self, neuron_activity, clip_features):
        """第一阶段：EP_encoder与CLIP对齐训练"""
        self.var_model.neuro_encoder.train()
        
        # 获取EP_encoder输出的tokens和1024维条件向量
        # 第一阶段只训练条件向量部分，tokens部分不参与对比学习
        _, condition_vector = self.var_model.neuro_encoder(neuron_activity, return_condition_vector=True)  # (B, n_token, Cvae), (B, 1024)
        
        # 确保数据类型一致（EP_encoder转换为float16以匹配CLIP）
        condition_vector = condition_vector.half()
        clip_features = clip_features.half()
        
        # 计算对比学习损失（与CLIP特征对齐）
        contrastive_loss = self.contrastive_loss(condition_vector, clip_features)
        
        return contrastive_loss, condition_vector
    
    @torch.no_grad()
    def eval_stage1(self, val_loader):
        """第一阶段：评估验证集loss"""
        self.var_model.neuro_encoder.eval()
        
        total_loss = 0.0
        total_samples = 0
        
        for neuron_activity, images, labels, clip_features in val_loader:
            neuron_activity = neuron_activity.to(self.device, non_blocking=True)
            clip_features = clip_features.to(self.device, non_blocking=True)
            
            # 获取EP_encoder输出的tokens和1024维条件向量
            # 第一阶段只评估条件向量部分
            _, condition_vector = self.var_model.neuro_encoder(neuron_activity, return_condition_vector=True)  # (B, n_token, Cvae), (B, 1024)
            
            # 确保数据类型一致
            condition_vector = condition_vector.half()
            clip_features = clip_features.half()
            
            # 计算对比学习损失（与CLIP特征对齐）
            loss = self.contrastive_loss(condition_vector, clip_features)
            
            batch_size = neuron_activity.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
        
        avg_loss = total_loss / total_samples
        return avg_loss

    @torch.no_grad()
    def eval_ep(self, ld_val: DataLoader, epoch=None, total_epochs=None):
        tot = 0
        L_mean, L_tail, acc_mean, acc_tail = 0, 0, 0, 0
        stt = time.time()
        training = self.var_model.training
        self.var_model.eval()
        
        # 添加验证进度条
        desc = f"Epoch {epoch+1}/{total_epochs} [Val]" if epoch is not None else "[Val]"
        val_pbar = tqdm(ld_val, desc=desc, ncols=100, leave=False)
        
        for neuron_activity, inp_B3HW, _, _  in val_pbar:  # 解包: (neuron_activity, img, label, clip_feature)
            B, V = neuron_activity.shape[0], self.vae_local.vocab_size
            inp_B3HW = inp_B3HW.to(self.device)
            neuron_activity = neuron_activity.to(self.device)
            
            # 强制使用VAE的img_to_idxBl方法，不使用零张量
            try:
                gt_idx_Bl = self.vae_local.img_to_idxBl(inp_B3HW)
                gt_BL = torch.cat(gt_idx_Bl, dim=1)
                x_BLCv_wo_first_l = self.quantize_local.idxBl_to_var_input(gt_idx_Bl)
            except Exception as e:
                print(f"Error in VAE img_to_idxBl: {e}")
                # 如果VAE方法失败，跳过这个batch
                continue
            
            logits_BLV = self.var_model(neuron_activity, x_BLCv_wo_first_l)
            
            L_mean += self.val_loss(logits_BLV.view(-1, V), gt_BL.view(-1)).item() * B
            L_tail += self.val_loss(
                logits_BLV[:, -self.last_l:].reshape(-1, V), 
                gt_BL[:, -self.last_l:].reshape(-1)
            ).item() * B
            acc_mean += (logits_BLV.argmax(dim=-1) == gt_BL).float().mean().item() * 100 * B
            acc_tail += (
                logits_BLV[:, -self.last_l:].argmax(dim=-1) == gt_BL[:, -self.last_l:]
            ).float().mean().item() * 100 * B
            tot += B
            
            # 更新验证进度条
            if tot > 0:
                val_pbar.set_postfix({
                    'loss': f'{L_mean/tot if tot > 0 else 0:.4f}',
                    'acc': f'{acc_mean/tot if tot > 0 else 0:.2f}%'
                })
        
        self.var_model.train(training)
        
        L_mean /= tot
        L_tail /= tot
        acc_mean /= tot
        acc_tail /= tot
        
        return L_mean, L_tail, acc_mean, acc_tail, tot, time.time()-stt

    def train_step(
        self, it: int, g_it: int, stepping: bool,
        inp_B3HW: torch.Tensor, neuron_activity: torch.Tensor, prog_si: int, prog_wp_it: float
    ):
        # 修复：检查模型是否有prog_si属性
        if hasattr(self.var_model, 'prog_si'):
            self.var_model.prog_si = prog_si
        if hasattr(self.vae_local.quantize, 'prog_si'):
            self.vae_local.quantize.prog_si = prog_si
            
        if self.last_prog_si != prog_si:
            if self.last_prog_si != -1: 
                self.first_prog = False
            self.last_prog_si = prog_si
            self.prog_it = 0
        self.prog_it += 1
        
        prog_wp = max(min(self.prog_it / prog_wp_it, 1), 0.01)
        if self.first_prog: 
            prog_wp = 1
        if prog_si == len(self.patch_nums) - 1: 
            prog_si = -1 

        B, V = neuron_activity.shape[0], self.vae_local.vocab_size
        
        # 使用新的autocast API
        with torch.amp.autocast('cuda', enabled=False):
            # 强制使用VAE的img_to_idxBl方法，不使用零张量
            try:
                gt_idx_Bl = self.vae_local.img_to_idxBl(inp_B3HW)
                gt_BL = torch.cat(gt_idx_Bl, dim=1)
                x_BLCv_wo_first_l = self.quantize_local.idxBl_to_var_input(gt_idx_Bl)
            except Exception as e:
                print(f"Error in VAE img_to_idxBl during training: {e}")
                # 如果VAE方法失败，跳过这个batch
                return 0.0, 0.0, 0.0, 1.0
            
            logits_BLV = self.var_model(neuron_activity, x_BLCv_wo_first_l)
            
            # 检查logits是否包含NaN或Inf
            if torch.isnan(logits_BLV).any() or torch.isinf(logits_BLV).any():
                print(f"警告: logits包含NaN或Inf，跳过此batch")
                return 0.0, 0.0, 0.0, 1.0

            pred_BL = logits_BLV.argmax(dim=-1)
            accuracy = (pred_BL == gt_BL).float().mean().item() * 100
            loss = self.train_loss(logits_BLV.view(-1, V), gt_BL.view(-1)).view(B, -1)
            
            # 检查loss是否包含NaN
            if torch.isnan(loss).any():
                print(f"警告: loss包含NaN，跳过此batch")
                return 0.0, 0.0, 0.0, 1.0
            
            if prog_si >= 0:
                bg, ed = self.begin_ends[prog_si]
                lw = self.loss_weight[:, :ed].clone()
                lw[:, bg:ed] *= prog_wp
            else:
                lw = self.loss_weight
                
            loss = loss.mul(lw).sum(dim=-1).mean()

        self.scaler.scale(loss).backward()
        
        grad_norm = 0
        if stepping:
            # 先unscale再clip，避免因为阈值过低导致频繁跳过step
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.var_model.parameters(), max_norm=1.0
            ).item()
            
            # 仅在出现NaN/Inf时跳过，并让scaler自动缩放
            if torch.isnan(torch.tensor(grad_norm)) or torch.isinf(torch.tensor(grad_norm)):
                print(f"警告: 梯度包含NaN/Inf，跳过此step")
                self.optimizer.zero_grad()
                self.scaler.update()
                return loss.item(), accuracy, grad_norm, self.scaler.get_scale()
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        
        if hasattr(self.var_model, 'prog_si'):
            self.var_model.prog_si = -1
        if hasattr(self.vae_local.quantize, 'prog_si'):
            self.vae_local.quantize.prog_si = -1
            
        return loss.item(), accuracy, grad_norm, self.scaler.get_scale()


def get_multiscale_reconstruction(var_model, vae_model, neuron_activity, max_scale=None, g_seed=None, cfg=5, top_k=1000, top_p=0.99, more_smooth=False):
    """
    获取指定scale的重建图片，支持渐进式生成到任意scale
    
    Args:
        var_model: VAR模型
        vae_model: VAE模型
        neuron_activity: 神经活动输入
        max_scale: 生成到第几个scale（0-based索引），None表示生成所有scale
        g_seed: 随机种子
        cfg: classifier-free guidance系数
        top_k, top_p: 采样参数
        more_smooth: 是否使用更平滑的采样
    
    Returns:
        重建的图像 tensor
    """
    B = neuron_activity.shape[0]
    device = neuron_activity.device
    
    # 1. 获取EP_encoder输出的tokens序列（用于VAR模型的交叉注意力）
    neuro_tokens = var_model.neuro_encoder(neuron_activity, return_condition_vector=False)  # (B, n_token, Cvae)
    
    # 2. 先对神经token做平均，得到Cvae维度的向量
    neuro_token_mean = neuro_tokens.mean(dim=1)  # (B, Cvae)
    
    # 3. 将平均后的向量投影到embed_dim，生成条件嵌入（用于AdaLN）
    cond_BD = var_model.word_embed(neuro_token_mean)  # (B, embed_dim)
    
    # 4. 将神经token投影到embed_dim（用于交叉注意力）
    neuro_tokens_proj = var_model.word_embed(neuro_tokens)  # (B, n_token, embed_dim)
    
    # 5. 初始化位置和层级嵌入
    lvl_pos = var_model.lvl_embed(var_model.lvl_1L) + var_model.pos_1LC
    
    # 6. 准备初始输入
    sos = cond_BD.unsqueeze(1).expand(B, var_model.first_l, -1) + var_model.pos_start.expand(B, var_model.first_l, -1) + lvl_pos[:, :var_model.first_l].expand(B, -1, -1).contiguous()
    
    f_hat = torch.zeros(B, var_model.Cvae, var_model.patch_nums[-1], var_model.patch_nums[-1], device=device)
    next_token_map = sos.clone()
    
    # 7. 开启KV缓存
    for b in var_model.blocks:
        b.attn.kv_caching(True)
    
    # 8. 确定生成到哪个scale
    if max_scale is None:
        max_scale = len(var_model.patch_nums) - 1
    else:
        max_scale = min(max_scale, len(var_model.patch_nums) - 1)
    
    # 9. 渐进式生成到指定scale
    from models.helpers import sample_with_top_k_top_p_, gumbel_softmax_with_rng
    # 设置随机数生成器
    if g_seed is None:
        rng = None
    else:
        if hasattr(var_model, 'rng'):
            var_model.rng.manual_seed(g_seed)
            rng = var_model.rng
        else:
            rng = torch.Generator(device=device)
            rng.manual_seed(g_seed)
    
    for si, pn in enumerate(var_model.patch_nums[:max_scale+1]):
        ratio = si / max(1, var_model.num_stages_minus_1)
        
        # 处理条件嵌入
        cond_BD_or_gss = var_model.shared_ada_lin(cond_BD)
        
        # 通过Transformer块（传递神经token）
        x = next_token_map
        for b in var_model.blocks:
            x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None, neuro_tokens=neuro_tokens_proj)
        
        # 获取logits
        logits = var_model.get_logits(x, cond_BD)
        
        # 采样
        idx_Bl = sample_with_top_k_top_p_(
            logits, 
            rng=rng, 
            top_k=top_k, 
            top_p=top_p, 
            num_samples=1
        )[:, :, 0]
        
        # 解码token
        if not more_smooth:
            h_BChw = var_model.vae_quant_proxy[0].embedding(idx_Bl)
        else:
            gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)
            h_BChw = gumbel_softmax_with_rng(
                logits.mul(1 + ratio), 
                tau=gum_t, 
                hard=False, 
                dim=-1, 
                rng=rng
            ) @ var_model.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
        
        h_BChw = h_BChw.transpose(1, 2).reshape(B, var_model.Cvae, pn, pn)
        
        # 更新f_hat
        f_hat, next_input = var_model.vae_quant_proxy[0].get_next_autoregressive_input(
            si, len(var_model.patch_nums), f_hat, h_BChw
        )
        
        # 准备下一阶段输入
        if si != max_scale and si != var_model.num_stages_minus_1:
            next_input = next_input.view(B, var_model.Cvae, -1).transpose(1, 2)
            next_token_map = var_model.word_embed(next_input)
            next_token_map += lvl_pos[:, var_model.begin_ends[si+1][0]:var_model.begin_ends[si+1][1]]
    
    # 关闭KV缓存
    for b in var_model.blocks:
        b.attn.kv_caching(False)
    
    # 解码图像
    result = var_model.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5)
    
    # 清理中间变量
    del f_hat, cond_BD, lvl_pos, next_token_map, neuro_tokens, neuro_tokens_proj
    if 'x' in locals():
        del x
    if 'logits' in locals():
        del logits
    
    return result


def compute_ssim_and_save_reconstructions(config, vae, var_model, val_loader, epoch, device, results_dir):
    """
    计算SSIM并保存重建图像到PDF
    """
    vae.eval()
    var_model.eval()
    
    ssim_values = []
    pdf_path = os.path.join(results_dir, f"reconstructions_epoch_{epoch}.pdf")
    
    # 诊断：收集token分布信息
    all_predicted_tokens = []
    all_gt_tokens = []
    token_entropy_list = []
    
    print(f"\n开始计算SSIM并生成重建图像PDF (Epoch {epoch})...")
    
    with torch.no_grad():
        with PdfPages(pdf_path) as pdf:
            # 添加SSIM计算进度条
            ssim_pbar = tqdm(
                enumerate(val_loader),
                total=len(val_loader),
                desc=f"Epoch {epoch} [SSIM & Reconstructions]",
                ncols=100
            )
            
            for batch_idx, (neuron_activity, images, labels, clip_features) in ssim_pbar:  # 解包: (neuron_activity, img, label, clip_feature)
                neuron_activity = neuron_activity.to(device)
                images = images.to(device)
                B = neuron_activity.shape[0]
                cfg = 2
                
                # 生成重建图像（采样方式）
                recon_B3HW = var_model.autoregressive_infer_cfg(
                    B=B, 
                    neuron_activity=neuron_activity, 
                    cfg=cfg, 
                    top_k=500, 
                    top_p=0.95, 
                    more_smooth=True
                )
                
            
                # 将图像归一化到[0,1]范围（SSIM需要）
                original_imgs = (images + 1) / 2  # 从[-1,1]转换到[0,1]
                # autoregressive_infer_cfg 返回的图像已经在[0,1]范围
                recon_imgs = torch.clamp(recon_B3HW, 0, 1)
                
                # 计算当前batch的SSIM（平均值）
                ssim_value = ssim(
                    recon_imgs, 
                    original_imgs, 
                    data_range=1.0, 
                    size_average=True
                ).item()
                ssim_values.append(ssim_value)
                
                # 对batch中的每个样本生成一页PDF
                for i in range(B):
                    # 创建2x3的子图布局
                    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                    axes = axes.flatten()
                    
                    # 第1张：原始图像
                    original_img = images[i].cpu().detach().numpy()
                    original_img = (original_img + 1) / 2  # 从[-1,1]转换到[0,1]
                    original_img = np.transpose(original_img, (1, 2, 0))  # 从CHW转换为HWC
                    axes[0].imshow(original_img)
                    axes[0].set_title("Original", fontsize=14, fontweight='bold')
                    axes[0].axis('off')
                    
                    # 第2张：完整重建图像
                    recon_img = recon_B3HW[i].cpu().numpy()
                    recon_img = np.clip(recon_img, 0, 1)  # 确保在[0,1]范围
                    recon_img = np.transpose(recon_img, (1, 2, 0))  # 从CHW转换为HWC
                    axes[1].imshow(recon_img)
                    axes[1].set_title("Full Reconstruction", fontsize=14, fontweight='bold')
                    axes[1].axis('off')
                    
                    # 第3-6张：显示scale 1到scale 4的渐进式重建
                    scale_indices = [0, 1, 2, 3]  # 对应scale 1-4
                    
                    for idx, scale_idx in enumerate(scale_indices):
                        random_seed = batch_idx * 1000 + i * 10 + epoch
                        
                        recon_scale = get_multiscale_reconstruction(
                            var_model=var_model,
                            vae_model=vae,
                            neuron_activity=neuron_activity[i:i+1],
                            max_scale=scale_idx,
                            g_seed=random_seed,
                            cfg=cfg,
                            top_k=900,
                            top_p=0.96,
                            more_smooth=False
                        )
                        
                        recon_img_scale = recon_scale[0].cpu().numpy()
                        # get_multiscale_reconstruction 返回的图像已经在[0,1]范围
                        recon_img_scale = np.clip(recon_img_scale, 0, 1)  # 确保在[0,1]范围
                        recon_img_scale = np.transpose(recon_img_scale, (1, 2, 0))  # 从CHW转换为HWC
                        
                        del recon_scale
                        torch.cuda.empty_cache()
                        
                        patch_num = var_model.patch_nums[scale_idx]
                        axes[idx + 2].imshow(recon_img_scale)
                        axes[idx + 2].set_title(
                            f"Scale {scale_idx + 1}\n({patch_num}×{patch_num} patches)", 
                            fontsize=12, fontweight='bold'
                        )
                        axes[idx + 2].axis('off')
                    
                    plt.suptitle(f"Epoch {epoch} - Batch {batch_idx}, Sample {i} (Label: {labels[i].item()})", 
                                fontsize=16, fontweight='bold', y=0.98)
                    plt.tight_layout()
                    pdf.savefig(fig, dpi=100)
                    plt.close(fig)
                
                # 清理显存
                del neuron_activity, images, recon_B3HW
                torch.cuda.empty_cache()
                
                # 更新SSIM进度条
                current_avg_ssim = np.mean(ssim_values) if len(ssim_values) > 0 else 0.0
                ssim_pbar.set_postfix({
                    'batch': f'{batch_idx+1}/{len(val_loader)}',
                    'avg_ssim': f'{current_avg_ssim:.4f}'
                })
    
    avg_ssim = np.mean(ssim_values)
    print(f"Epoch {epoch} - 平均SSIM: {avg_ssim:.4f}")
    
    # 保存SSIM结果
    ssim_file = os.path.join(results_dir, "ssim_results.txt")
    with open(ssim_file, 'a') as f:
        f.write(f"Epoch {epoch}: {avg_ssim:.4f}\n")
    
    var_model.train()
    return avg_ssim


def train_single_gpu(config, vae, var_model, train_dataset, val_dataset):
    torch.manual_seed(config.seed)
    if config.device == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    
    # 创建结果目录
    os.makedirs(config.results_dir, exist_ok=True)
    checkpoint_dir = os.path.join(config.results_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )

    
    # 创建初始优化器（将在第一阶段重新配置）
    optimizer_placeholder = torch.optim.AdamW(
        var_model.neuro_encoder.parameters(),
        lr=config.lr * config.ep_encoder_lr_multiplier,
        weight_decay=config.weight_decay,
        betas=config.betas
    )
    
    trainer = VARTrainer(
        device=config.device,
        patch_nums=config.var_config["patch_nums"],
        resos=(16, 32, 48, 64, 80, 96, 128, 160, 208, 256),
        vae_local=vae,
        var_model=var_model,
        optimizer=optimizer_placeholder,
        label_smooth=config.label_smooth,
        clip_model=clip_model,
        amp_enabled=config.amp_enabled
    )
    
    # ===== 第一阶段：EP_encoder与CLIP对齐训练 =====
    print("=" * 60)
    print("开始第一阶段：EP_encoder与CLIP对齐训练")
    print("=" * 60)
    
    trainer.set_stage(1)
    
    # 第一阶段只训练EP_encoder
    ep_encoder_params = []
    for name, param in var_model.neuro_encoder.named_parameters():
        if param.requires_grad:
            ep_encoder_params.append(param)
    
    optimizer_stage1 = torch.optim.AdamW(
        ep_encoder_params,
        lr=config.lr * config.ep_encoder_lr_multiplier,
        weight_decay=config.weight_decay,
        betas=config.betas
    )
    trainer.optimizer = optimizer_stage1
    
    print(f"第一阶段优化器配置:")
    print(f"  - EP_encoder学习率: {config.lr * config.ep_encoder_lr_multiplier:.2e}")
    print(f"  - 参数数量: {sum(p.numel() for p in ep_encoder_params if p.requires_grad)}")
    
    global_step = 0
    
    # 第一阶段训练循环
    for epoch in range(config.stage1_epochs):
        var_model.neuro_encoder.train()
        epoch_loss = 0.0
        epoch_samples = 0
        
        # 添加训练进度条
        train_pbar = tqdm(
            enumerate(train_loader), 
            total=len(train_loader),
            desc=f"Stage 1 - Epoch {epoch+1}/{config.stage1_epochs} [Train]",
            ncols=100,
            leave=False
        )
        
        for i, (neuron_activity, images, labels, clip_features) in train_pbar:
            neuron_activity = neuron_activity.to(config.device, non_blocking=True)
            clip_features = clip_features.to(config.device, non_blocking=True)
            
            stepping = (i + 1) % config.grad_accum == 0
            
            # 第一阶段训练步骤
            loss_value, ep_features = trainer.train_step_stage1(neuron_activity, clip_features)
            
            trainer.scaler.scale(loss_value).backward()
            
            grad_norm = 0
            if stepping:
                trainer.scaler.unscale_(trainer.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    var_model.neuro_encoder.parameters(), max_norm=1.0
                ).item()
                
                trainer.scaler.step(trainer.optimizer)
                trainer.scaler.update()
                trainer.optimizer.zero_grad()
            
            batch_size = neuron_activity.size(0)
            epoch_loss += loss_value.item() * batch_size
            epoch_samples += batch_size
            global_step += 1
            
            # 更新进度条显示
            train_pbar.set_postfix({
                'loss': f'{loss_value.item():.4f}',
                'avg_loss': f'{epoch_loss/epoch_samples:.4f}',
                'grad_norm': f'{grad_norm:.4f}'
            })
        
        avg_epoch_loss = epoch_loss / epoch_samples
        
        # 评估验证集loss
        if epoch % config.eval_interval == 0:
            val_loss = trainer.eval_stage1(val_loader)
            print(f"\nStage 1 - Epoch {epoch+1}/{config.stage1_epochs} Training: "
                  f"Train Loss = {avg_epoch_loss:.4f}, Val Loss = {val_loss:.4f}")
        else:
            print(f"Stage 1 - Epoch {epoch+1}/{config.stage1_epochs} Training: "
                  f"Contrastive Loss = {avg_epoch_loss:.4f}")
        
        # 每10个epoch保存checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'neuro_encoder_state_dict': var_model.neuro_encoder.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'epoch': epoch,
                'loss': avg_epoch_loss,
            }
            checkpoint_path = os.path.join(checkpoint_dir, f'stage1_ep_encoder_epoch_{epoch+1}.pth')
            torch.save(checkpoint, checkpoint_path)
            print(f"✓ Stage 1 Checkpoint已保存: {checkpoint_path}")
    
    # 保存第一阶段训练好的EP_encoder
    best_ep_encoder_path = os.path.join(checkpoint_dir, 'best_ep_encoder_clip_aligned.pth')
    torch.save(var_model.neuro_encoder.state_dict(), best_ep_encoder_path)
    print(f"\n第一阶段训练完成，EP_encoder已保存: {best_ep_encoder_path}")
    
    # ===== 第二阶段：VAR模型训练 =====
    print("\n" + "=" * 60)
    print("开始第二阶段：冻结EP_encoder后的VAR训练")
    print("=" * 60)
    
    trainer.set_stage(2)
    
    # 第二阶段训练VAR，部分冻结EP_encoder（冻结condition_proj，保留token生成部分可训练）
    # 配置优化器：包含VAR参数和EP_encoder的token生成部分（不包括condition_proj）
    var_params = []
    ep_encoder_token_params = []
    for name, param in var_model.named_parameters():
        if 'neuro_encoder' in name:
            # EP_encoder的token生成部分（不包括condition_proj）
            if 'condition_proj' not in name and param.requires_grad:
                ep_encoder_token_params.append(param)
        else:
            # VAR模型的其他参数
            if param.requires_grad:
                var_params.append(param)
    
    # 合并所有可训练参数
    all_params = var_params + ep_encoder_token_params
    
    optimizer_stage2 = torch.optim.AdamW(
        all_params,
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=config.betas
    )
    trainer.optimizer = optimizer_stage2
    
    print(f"第二阶段优化器配置:")
    print(f"  - VAR学习率: {config.lr:.2e}")
    print(f"  - VAR参数数量: {sum(p.numel() for p in var_params if p.requires_grad)}")
    print(f"  - EP_encoder token生成部分参数数量: {sum(p.numel() for p in ep_encoder_token_params if p.requires_grad)}")
    print(f"  - 总参数数量: {sum(p.numel() for p in all_params if p.requires_grad)}")
    
    # 跟踪最佳验证损失
    best_val_loss = float('inf')
    best_epoch = -1
    
    # 第二阶段训练循环
    for epoch in range(config.stage2_epochs):
        var_model.train()        
        # 渐进式训练：如果禁用，直接训练所有scale（prog_si=-1）
        if config.enable_progressive_training:
            prog_si = min(epoch // config.prog_epochs + 1, len(config.var_config["patch_nums"]) - 1)
        else:
            prog_si = -1  # -1表示训练所有scale，不使用渐进式训练
        
        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_samples = 0
        
        # 添加训练进度条
        train_pbar = tqdm(
            enumerate(train_loader), 
            total=len(train_loader),
            desc=f"Stage 2 - Epoch {epoch+1}/{config.stage2_epochs} [Train]",
            ncols=100,
            leave=False
        )
        
        for i, (neuron_activity, images, labels, clip_features) in train_pbar:
            images = images.to(config.device, non_blocking=True)
            neuron_activity = neuron_activity.to(config.device, non_blocking=True)
            
            stepping = (i + 1) % config.grad_accum == 0
            
            loss_value, accuracy, grad_norm, scale = trainer.train_step(
                it=i,
                g_it=global_step,
                stepping=stepping,
                inp_B3HW=images,
                neuron_activity=neuron_activity,
                prog_si=prog_si,
                prog_wp_it=config.prog_warmup_iters
            )
            
            batch_size = images.size(0)
            epoch_loss += loss_value * batch_size
            epoch_acc += accuracy * batch_size
            epoch_samples += batch_size

            global_step += 1
            
            # 更新进度条显示
            train_pbar.set_postfix({
                'loss': f'{loss_value:.4f}',
                'acc': f'{accuracy:.2f}%',
                'avg_loss': f'{epoch_loss/epoch_samples:.4f}',
                'avg_acc': f'{epoch_acc/epoch_samples:.2f}%'
            })
        
        avg_epoch_loss = epoch_loss / epoch_samples
        avg_epoch_acc = epoch_acc / epoch_samples
        
        print(f"\nStage 2 - Epoch {epoch+1}/{config.stage2_epochs} Training: "
              f"Loss = {avg_epoch_loss:.4f} "
              f"Acc = {avg_epoch_acc:.2f}%")

        if epoch % config.eval_interval == 0:
            var_model.eval()
            L_mean, L_tail, acc_mean, acc_tail, tot, eval_time = trainer.eval_ep(val_loader, epoch, config.stage2_epochs)
            print(f"\nStage 2 - Epoch {epoch+1} Validation: "
                  f"Loss = {L_mean:.4f}/{L_tail:.4f} "
                  f"Acc = {acc_mean:.2f}%/{acc_tail:.2f}% "
                  f"Time = {eval_time:.1f}s\n")
            
            # 更新最佳模型
            if L_mean < best_val_loss:
                best_val_loss = L_mean
                best_epoch = epoch
                # 保存最佳checkpoint
                best_checkpoint_path = os.path.join(checkpoint_dir, "best_checkpoint_stage2.pth")
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": var_model.state_dict(),
                    "optimizer_state_dict": optimizer_stage2.state_dict(),
                    "scaler_state_dict": trainer.scaler.state_dict(),
                    "val_loss": L_mean,
                    "val_acc": acc_mean
                }
                torch.save(checkpoint, best_checkpoint_path)
                print(f"保存最佳checkpoint (Epoch {epoch+1}, Val Loss: {L_mean:.4f})")
        
        # 每5个epoch保存checkpoint
        if epoch % config.save_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_stage2_epoch_{epoch+1}.pth")
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": var_model.state_dict(),
                "optimizer_state_dict": optimizer_stage2.state_dict(),
                "scaler_state_dict": trainer.scaler.state_dict(),
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"保存checkpoint: {checkpoint_path}")
        
        # 每个epoch计算SSIM并保存重建图像
        if epoch % config.ssim_interval == 0:
            avg_ssim = compute_ssim_and_save_reconstructions(
                config, vae, var_model, val_loader, epoch, config.device, config.results_dir
            )
            print(f"Stage 2 - Epoch {epoch+1} - SSIM计算完成，平均SSIM: {avg_ssim:.4f}")


def train_with_psth_type(psth_type='loo'):
    """
    使用指定类型的PSTH数据进行训练
    Args:
        psth_type: 'loo' 或 'raw'，指定使用哪种PSTH数据
    """
    print(f"\n{'='*80}")
    print(f"开始训练 - PSTH类型: {psth_type.upper()}")
    print(f"{'='*80}\n")
    
    # 1. 加载PSTH数据
    if psth_type == 'loo':
        psth_path = psth_loo_path
    elif psth_type == 'raw':
        psth_path = psth_raw_path
    else:
        raise ValueError(f"未知的PSTH类型: {psth_type}，必须是'loo'或'raw'")
    
    train_MUA = load_psth_data(psth_path, psth_type)
    
    # 2. 构建图像路径（共享）
    image_base_dir = "/disk1/jinchentao/visual_decode/nature_scene"
    image_paths = []
    for img_id in trial_image_ids:
        image_path = os.path.join(image_base_dir, f"natural_scene_{img_id}.tiff")
        image_paths.append(image_path)
    
    # 3. 根据图像ID创建类别标签（共享）
    unique_image_ids = sorted(list(set(trial_image_ids)))  # 118个唯一图像
    num_classes = len(unique_image_ids)  # 118个类别
    print(f"唯一图像数量: {num_classes}")
    
    # 创建图像ID到类别索引的映射
    image_id_to_class = {img_id: idx for idx, img_id in enumerate(unique_image_ids)}
    class_labels = np.array([image_id_to_class[img_id] for img_id in trial_image_ids])
    unique_classes = np.arange(num_classes)
    
    num_samples = train_MUA.shape[0]
    print(f"样本数量: {num_samples}")
    print(f"时间步数: {train_MUA.shape[1]}")
    print(f"神经元数量: {train_MUA.shape[2]}")
    print(f"类别数量: {num_classes}")
    
    # 4. 获取类别标签
    mapped_labels, label_mapping = create_label_mapping_from_classes(class_labels, unique_classes)
    
    print(f"标签映射完成，映射了 {len(label_mapping)} 个类别")
    print(f"标签范围: {mapped_labels.min()} - {mapped_labels.max()}")
    
    # 5. 固定100张图片作为测试集，其余为训练集
    test_size = 100
    all_indices = list(range(len(train_MUA)))
    np.random.seed(42)
    np.random.shuffle(all_indices)
    test_indices = all_indices[:test_size]
    train_indices = all_indices[test_size:]
    
    print(f"训练集大小: {len(train_indices)}")
    print(f"测试集大小: {len(test_indices)}")
    
    # 6. 创建数据集（启用CLIP特征预计算）
    print("创建训练数据集...")
    train_dataset = MUAClassificationDataset(
        train_MUA[train_indices], 
        mapped_labels[train_indices],
        image_paths=[image_paths[i] for i in train_indices] if image_paths else None,
        load_clip_features=True,
        clip_model=clip_model,
        device=device_for_clip
    )
    
    print("创建测试数据集...")
    test_dataset = MUAClassificationDataset(
        train_MUA[test_indices], 
        mapped_labels[test_indices],
        image_paths=[image_paths[i] for i in test_indices] if image_paths else None,
        load_clip_features=True,
        clip_model=clip_model,
        device=device_for_clip
    )
    
    # 7. 构建模型（每次训练都创建新的模型实例）
    MODEL_DEPTH = 16
    time_bins = train_MUA.shape[1]
    actual_input_dim = train_MUA.shape[2]
    num_classes_ep = num_classes
    
    print(f"\nEP配置 - 时间步数: {time_bins}, 神经元数量: {actual_input_dim}, 类别数量: {num_classes_ep}")
    
    vae_ckpt = '/disk1/jinchentao/visual_decode/visual_reconstruction/my_VAR_mua/vae_ch160v4096z32.pth'
    var_ckpt = f'/disk1/jinchentao/visual_decode/visual_reconstruction/my_VAR_mua/var_d{MODEL_DEPTH}.pth'
    
    FOR_512_px = MODEL_DEPTH == 16
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"开始构建VAE和VAR模型...")
    print(f"  - 设备: {device}")
    print(f"  - VAR深度: {MODEL_DEPTH}")
    print(f"  - 输入维度: {actual_input_dim}, 类别数: {num_classes_ep}")
    start_time = time.time()
    vae, var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,   
        device=device, patch_nums=patch_nums, input_dim = actual_input_dim, num_classes_ep = num_classes_ep,
        num_classes=1000, depth=MODEL_DEPTH, shared_aln=FOR_512_px
    )
    build_time = time.time() - start_time
    print(f"✓ VAE和VAR模型构建完成，耗时: {build_time:.2f}秒")
    
    # 加载VAE权重
    print(f"开始加载VAE权重...")
    vae_start_time = time.time()
    vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
    vae_load_time = time.time() - vae_start_time
    print(f"✓ 已加载VAE权重: {vae_ckpt} (耗时: {vae_load_time:.2f}秒)")
    
    # 加载预训练VAR权重
    print(f"开始加载预训练VAR权重: {var_ckpt}")
    var_load_start_time = time.time()
    try:
        var_state_dict = torch.load(var_ckpt, map_location='cpu')
        var_load_dict_time = time.time() - var_load_start_time
        print(f"  - 权重文件加载完成 (耗时: {var_load_dict_time:.2f}秒)")
        
        var_load_start_time = time.time()
        var.load_state_dict(var_state_dict, strict=False)
        var_load_time = time.time() - var_load_start_time
        print(f"✓ 已加载预训练VAR权重 (耗时: {var_load_time:.2f}秒)")
        
        loaded_keys = set(var_state_dict.keys())
        print(f"  - checkpoint包含 {len(loaded_keys)} 个参数键")
        neuro_encoder_keys = [k for k in loaded_keys if 'neuro_encoder' in k]
        if neuro_encoder_keys:
            print(f"  - 注意: checkpoint中包含 {len(neuro_encoder_keys)} 个neuro_encoder相关键（将被跳过，因为会被替换）")
                
    except Exception as e:
        print(f"⚠️  警告: 加载预训练VAR权重失败: {e}")
        print(f"  将从头开始训练")
        var._pretrained_keys = set()
        var._new_keys = set()
    
    # 替换VAR模型的neuro_encoder
    neuro_d_model = var.Cvae  # 32
    n_token = 128
    
    var.neuro_encoder = TemporalEPEncoder(
        input_dim=actual_input_dim,
        time_bins=time_bins,
        d_model=neuro_d_model,
        n_token=n_token,
        num_conv_layers=2,
        dropout=0.2,
        Cvae=var.Cvae
    ).to(device)
    
    # 重写VAR模型的forward方法
    original_var_forward = var.forward
    import types
    var.forward = types.MethodType(var_forward_with_ep_encoder_tokens, var)
    
    print(f"\n已创建轻量级时间序列EP编码器 ({psth_type.upper()}版本):")
    print(f"  - 输入: (B, {time_bins}, {actual_input_dim})")
    print(f"  - 输出1 (tokens): (B, {n_token}, {var.Cvae})")
    print(f"  - 输出2 (condition_vector): (B, 1024)")
    
    # 8. 创建训练配置
    config = TrainingConfig(psth_type=psth_type)
    
    # 9. 开始训练
    train_single_gpu(config, vae=vae, var_model=var, train_dataset=train_dataset, val_dataset=test_dataset)
    
    # 10. 训练完成后保存最终模型
    final_model_path = os.path.join(config.results_dir, f"var_mouse1_1215_{psth_type}_final.pth")
    torch.save(var, final_model_path)
    print(f"\n{'='*80}")
    print(f"训练完成 ({psth_type.upper()}版本)，最终模型已保存至: {final_model_path}")
    print(f"{'='*80}\n")
    
    # 清理显存
    del vae, var, train_dataset, test_dataset
    torch.cuda.empty_cache()
    gc.collect()

# 分别训练两个版本
if __name__ == "__main__":
    # 训练留一法版本
    # train_with_psth_type(psth_type='loo')
    
    # 训练原始版本
    train_with_psth_type(psth_type='raw')
    
    print(f"\n{'='*80}")
    print("所有训练完成！")
    print(f"{'='*80}")

