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
from models import VQVAE, build_vae_var

# 导入CLIP相关模块
import open_clip
import torch.nn.functional as F

# 加载训练数据
train_MUA = np.load("/disk1/jinchentao/visual_decode/visual_reconstruction/dataset/train_MUA_MonkeyF.npy")  # shape: (22248, 503)
train_csv_path = "/disk1/jinchentao/visual_decode/visual_reconstruction/dataset/train_image_MonkeyF.csv"

print(f"训练MUA数据形状: {train_MUA.shape}")

# 读取CSV文件获取类别和图像路径信息
df = pd.read_csv(train_csv_path)
print(f"CSV数据形状: {df.shape}")
print(f"CSV列名: {df.columns.tolist()}")

# 获取所有唯一的类别
unique_classes = df['class'].unique()
num_classes = len(unique_classes)
print(f"类别数量: {num_classes}")
print(f"前10个类别: {unique_classes[:10]}")

# 添加图像路径加载
def load_image_paths_for_train():
    """
    为训练数据集加载对应的图像路径
    返回图像路径列表，与train_MUA数据对应
    """
    # 直接使用CSV文件中的local_path列
    image_paths = df['local_path'].values
    return image_paths

# 加载图像路径
image_paths = load_image_paths_for_train()
print(f"加载了 {len(image_paths)} 个图像路径")

# 检查数据维度
print(f"train_MUA shape: {train_MUA.shape}")
print(f"image_paths length: {len(image_paths)}")
print(f"CSV数据长度: {len(df)}")
print(f"数据一致性检查: {train_MUA.shape[0] == len(image_paths) == len(df)}")

# 加载CLIP模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {device}")

# 加载CLIP模型 - 使用ViT-L-14
# 加载 TorchScript 模型
checkpoint_path = '/disk1/jinchentao/visual_decode/visual_reconstruction/VAR-CLIP-master/ViT-L-14.pt'
clip_model = torch.jit.load(checkpoint_path, map_location='cpu')
clip_model = clip_model.to(device)
clip_model.eval()

# 获取预处理器
_, clip_preprocess, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained=None)

# 图像预处理函数
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

class MUAClassificationDataset(Dataset):
    def __init__(self, mua_data, labels, image_paths=None, transform=None, load_clip_features=True):
        self.mua_data = torch.tensor(mua_data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.image_paths = image_paths
        self.transform = transform
        self.load_clip_features = load_clip_features
        
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
        if self.load_clip_features and self.image_paths is not None:
            self._precompute_clip_features()
        
    def _precompute_clip_features(self):
        """预计算所有图像的CLIP特征（带缓存机制）"""
        import hashlib
        
        # 创建缓存目录
        cache_dir = '/disk1/jinchentao/visual_decode/visual_reconstruction/my_VAR_mua/clip_features_vitL_cache'
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
                img_batch = img_tensor.unsqueeze(0).to(device)
                
                # 检查CLIP模型的dtype并转换输入
                clip_dtype = next(clip_model.parameters()).dtype
                img_batch = img_batch.to(clip_dtype)
                
                # 提取CLIP特征
                with torch.no_grad():
                    clip_feature = clip_model.encode_image(img_batch)
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
                self.clip_features.append(torch.zeros(1, 768))  # CLIP ViT-L-14的特征维度是768
        
        print(f"CLIP特征预计算完成: 总计{len(self.image_paths)}个, 缓存{cached_count}个, 新计算{computed_count}个")
        
    def __len__(self):
        return len(self.mua_data)
    
    def __getitem__(self, idx):
        mua_sample = self.mua_data[idx]
        label = self.labels[idx]
        
        # 处理图像
        if self.image_paths is not None and idx < len(self.image_paths):
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
            clip_feature = torch.zeros(768)  # CLIP ViT-L-14的特征维度是768
        
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

# 获取类别标签
class_labels = df['class'].values
mapped_labels, label_mapping = create_label_mapping_from_classes(class_labels, unique_classes)

print(f"标签映射完成，映射了 {len(label_mapping)} 个类别")
print(f"标签范围: {mapped_labels.min()} - {mapped_labels.max()}")

# 根据类别进行分层划分
train_indices, test_indices = train_test_split(
    range(len(train_MUA)), 
    test_size=0.2, 
    random_state=42, 
    stratify=mapped_labels
)

print(f"训练集大小: {len(train_indices)}")
print(f"测试集大小: {len(test_indices)}")

# 创建数据集（启用CLIP特征预计算）
print("创建训练数据集...")
train_dataset = MUAClassificationDataset(
    train_MUA[train_indices], 
    mapped_labels[train_indices],
    image_paths=[image_paths[i] for i in train_indices],
    load_clip_features=True
)

print("创建测试数据集...")
test_dataset = MUAClassificationDataset(
    train_MUA[test_indices], 
    mapped_labels[test_indices],
    image_paths=[image_paths[i] for i in test_indices],
    load_clip_features=True
)

# 模型配置
MODEL_DEPTH = 16   
actual_input_dim = train_MUA.shape[1]  # 神经元数量 (503)
num_classes_ep = num_classes  # 使用实际的类别数量

print(f"EP配置 - 输入维度: {actual_input_dim}, 类别数量: {num_classes_ep}")

vae_ckpt, var_ckpt = 'vae_ch160v4096z32.pth', f'var_d{MODEL_DEPTH}.pth'

FOR_512_px = MODEL_DEPTH == 16

patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
vae, var = build_vae_var(
    V=4096, Cvae=32, ch=160, share_quant_resi=4,   
    device=device, patch_nums=patch_nums, input_dim = actual_input_dim, num_classes_ep = num_classes_ep,
    num_classes=len(label_mapping), depth=MODEL_DEPTH, shared_aln=FOR_512_px
)

vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu', weights_only=False), strict=True)

# ==================== 关键修复：加载VAR预训练权重 ====================
print("正在加载VAR预训练权重...")

# 尝试多个可能的路径
possible_paths = [
    f'../VAR-main/var_d{MODEL_DEPTH}.pth',
    f'weight/var_d{MODEL_DEPTH}.pth',
    f'var_d{MODEL_DEPTH}.pth',
    '../VAR-main/output/ar-ckpt-best.pth',  # VAR训练生成的checkpoint
]

var_checkpoint_path = None
for path in possible_paths:
    if os.path.exists(path):
        var_checkpoint_path = path
        print(f"找到预训练权重: {path}")
        break

if var_checkpoint_path is None:
    print("⚠ 警告: 未找到VAR预训练权重文件")
    raise FileNotFoundError(f"请确保VAR预训练权重文件存在")
try:
    var_pretrained = torch.load(var_checkpoint_path, map_location='cpu', weights_only=False)
    
    # 提取VAR的state_dict（因为checkpoint可能包含其他键）
    if 'model' in var_pretrained:
        var_state_dict = var_pretrained['model']
    elif isinstance(var_pretrained, dict) and any('blocks' in k for k in var_pretrained.keys()):
        var_state_dict = var_pretrained
    else:
        var_state_dict = var_pretrained
    
    # 过滤掉不匹配的键（如ep_encoder相关参数）
    model_state_dict = var.state_dict()
    pretrained_dict = {k: v for k, v in var_state_dict.items() 
                      if k in model_state_dict and model_state_dict[k].shape == v.shape}
    
    # 打印加载信息
    loaded_keys = set(pretrained_dict.keys())
    model_keys = set(model_state_dict.keys())
    
    print(f"✓ 成功加载 {len(loaded_keys)} 个预训练参数")
    print(f"✗ 跳过了 {len(model_keys - loaded_keys)} 个新参数（包括EP_encoder）")
    
    # 加载权重（strict=False允许不匹配的参数）
    var.load_state_dict(pretrained_dict, strict=False)
    print("✓ VAR预训练权重加载完成")
    
except Exception as e:
    print(f"⚠ 无法加载VAR预训练权重: {e}")
    print("⚠ 将使用随机初始化的VAR模型（可能导致训练困难）")

class TrainingConfig:
    batch_size = 256
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
    
    lr = 1e-4
    weight_decay = 0.05
    betas = (0.9, 0.95)
    
    epochs = 60
    grad_accum = 1
    label_smooth = 0.1
    amp_enabled = True
    
    prog_epochs = 10 
    prog_warmup_iters = 1000  
    
    log_interval = 50
    eval_interval = 1
    save_interval = 5
    checkpoint_dir = "checkpoints"
    log_dir = "logs"
    
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    seed = 42
    
    # 两步训练配置
    stage1_epochs = 120  
    stage2_epochs = 120  
    contrastive_temp = 0.07  # 对比学习温度参数

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
        # 统一数据类型为float16
        features_1 = features_1.half()
        features_2 = features_2.half()
        
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

class TwoStageVARTrainer(object):
    def __init__(
        self, device, patch_nums, resos,
        vae_local, var_model, clip_model,
        optimizer: torch.optim.Optimizer, label_smooth: float,
        amp_enabled: bool = False, rank: int = 0
    ):
        super(TwoStageVARTrainer, self).__init__()
        
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
            # 冻结EP_encoder
            for param in self.var_model.ep_encoder.parameters():
                param.requires_grad = False
            print("EP_encoder已冻结")
        else:
            # 解冻EP_encoder
            for param in self.var_model.ep_encoder.parameters():
                param.requires_grad = True
            print("EP_encoder已解冻")

    def train_step_stage1(self, neuron_activity, clip_features):
        """第一阶段：EP_encoder与CLIP对齐训练"""
        self.var_model.ep_encoder.train()
        
        # 获取EP_encoder特征
        _, ep_features = self.var_model.ep_encoder(neuron_activity)
        
        # 确保数据类型一致（EP_encoder转换为float16以匹配CLIP）
        ep_features = ep_features.half()
        clip_features = clip_features.half()
        
        # 计算对比学习损失
        contrastive_loss = self.contrastive_loss(ep_features, clip_features)
        
        return contrastive_loss, ep_features
    
    @torch.no_grad()
    def eval_stage1(self, val_loader):
        """第一阶段：评估验证集loss"""
        self.var_model.ep_encoder.eval()
        
        total_loss = 0.0
        total_samples = 0
        
        for neuron_activity, images, labels, clip_features in val_loader:
            neuron_activity = neuron_activity.to(self.device, non_blocking=True)
            clip_features = clip_features.to(self.device, non_blocking=True)
            
            # 获取EP_encoder特征
            _, ep_features = self.var_model.ep_encoder(neuron_activity)
            
            # 确保数据类型一致
            ep_features = ep_features.half()
            clip_features = clip_features.half()
            
            # 计算对比学习损失
            loss = self.contrastive_loss(ep_features, clip_features)
            
            batch_size = neuron_activity.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
        
        avg_loss = total_loss / total_samples
        return avg_loss

    def train_step_stage2(
        self, it: int, g_it: int, stepping: bool,
        inp_B3HW: torch.Tensor, neuron_activity: torch.Tensor, prog_si: int, prog_wp_it: float
    ):
        """第二阶段：VAR模型训练"""
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

            pred_BL = logits_BLV.argmax(dim=-1)
            accuracy = (pred_BL == gt_BL).float().mean().item() * 100
            loss = self.train_loss(logits_BLV.view(-1, V), gt_BL.view(-1)).view(B, -1)
            
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
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.var_model.parameters(), max_norm=1.0
            ).item()
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        
        if hasattr(self.var_model, 'prog_si'):
            self.var_model.prog_si = -1
        if hasattr(self.vae_local.quantize, 'prog_si'):
            self.vae_local.quantize.prog_si = -1
            
        return loss.item(), accuracy, grad_norm, self.scaler.get_scale()

    @torch.no_grad()
    def eval_ep(self, ld_val: DataLoader):
        tot = 0
        L_mean, L_tail, acc_mean, acc_tail = 0, 0, 0, 0
        stt = time.time()
        training = self.var_model.training
        self.var_model.eval()
        
        for neuron_activity, inp_B3HW, _, _ in ld_val:
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
        
        self.var_model.train(training)
        
        L_mean /= tot
        L_tail /= tot
        acc_mean /= tot
        acc_tail /= tot
        
        return L_mean, L_tail, acc_mean, acc_tail, tot, time.time()-stt

def train_two_stage(config, vae, var_model, clip_model, train_dataset, val_dataset):
    torch.manual_seed(config.seed)
    if config.device == 'cuda':
        torch.cuda.manual_seed_all(config.seed)
    
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

    optimizer = torch.optim.AdamW(
        var_model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=config.betas
    )
    
    trainer = TwoStageVARTrainer(
        device=config.device,
        patch_nums=config.var_config["patch_nums"],
        resos=(16, 32, 48, 64, 80, 96, 128, 160, 208, 256),
        vae_local=vae,
        var_model=var_model,
        clip_model=clip_model,
        optimizer=optimizer,
        label_smooth=config.label_smooth,
        amp_enabled=config.amp_enabled
    )
    
    global_step = 0
    
    # ===== 第一阶段：EP_encoder与CLIP对齐训练 =====
    print("=" * 60)
    print("开始第一阶段：EP_encoder与CLIP对齐训练")
    print("=" * 60)
    
    trainer.set_stage(1)
    
    for epoch in range(config.stage1_epochs):
        var_model.train()
        epoch_loss = 0.0
        epoch_samples = 0
        
        for i, (neuron_activity, images, labels, clip_features) in enumerate(train_loader):
            neuron_activity = neuron_activity.to(config.device, non_blocking=True)
            clip_features = clip_features.to(config.device, non_blocking=True)
            
            stepping = (i + 1) % config.grad_accum == 0
            
            # 第一阶段训练步骤
            loss_value, ep_features = trainer.train_step_stage1(neuron_activity, clip_features)
            
            trainer.scaler.scale(loss_value).backward()
            
            if stepping:
                trainer.scaler.unscale_(trainer.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    var_model.ep_encoder.parameters(), max_norm=1.0
                ).item()
                
                trainer.scaler.step(trainer.optimizer)
                trainer.scaler.update()
                trainer.optimizer.zero_grad()
            
            batch_size = neuron_activity.size(0)
            epoch_loss += loss_value.item() * batch_size
            epoch_samples += batch_size
        
        avg_epoch_loss = epoch_loss / epoch_samples
        
        # 评估验证集loss
        if epoch % config.eval_interval == 0:
            val_loss = trainer.eval_stage1(val_loader)
            print(f"Stage 1 - Epoch {epoch+1}/{config.stage1_epochs} Training: "
                  f"Train Loss = {avg_epoch_loss:.4f}, Val Loss = {val_loss:.4f}")
        else:
            print(f"Stage 1 - Epoch {epoch+1}/{config.stage1_epochs} Training: "
                  f"Contrastive Loss = {avg_epoch_loss:.4f}")
        
        # 每10个epoch保存checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'ep_encoder_state_dict': var_model.ep_encoder.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'epoch': epoch,
                'loss': avg_epoch_loss,
            }
            torch.save(checkpoint, f'stage1_ep_encoder_epoch_{epoch+1}.pth')
            print(f"✓ Stage 1 Checkpoint已保存: stage1_ep_encoder_epoch_{epoch+1}.pth")
    
    # 保存第一阶段训练好的EP_encoder
    torch.save(var_model.ep_encoder.state_dict(), 'best_ep_encoder_clip_aligned_MonkeyF_clip_L.pth')
    print("第一阶段训练完成，EP_encoder已保存")
    
    # ===== 第二阶段：VAR模型训练 =====
    print("=" * 60)
    print("开始第二阶段：冻结EP_encoder后的VAR训练")
    print("=" * 60)
    
    trainer.set_stage(2)
    
    for epoch in range(config.stage2_epochs):
        var_model.train()        
        prog_si = min(epoch // config.prog_epochs + 1, len(config.var_config["patch_nums"]) - 1)
        
        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_samples = 0
        
        for i, (neuron_activity, images, labels, clip_features) in enumerate(train_loader):
            images = images.to(config.device, non_blocking=True)
            neuron_activity = neuron_activity.to(config.device, non_blocking=True)
            
            stepping = (i + 1) % config.grad_accum == 0
            
            loss_value, accuracy, grad_norm, scale = trainer.train_step_stage2(
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
        
        avg_epoch_loss = epoch_loss / epoch_samples
        avg_epoch_acc = epoch_acc / epoch_samples
        
        print(f"Stage 2 - Epoch {epoch+1}/{config.stage2_epochs} Training: "
              f"Loss = {avg_epoch_loss:.4f} "
              f"Acc = {avg_epoch_acc:.2f}%")

        if epoch % config.eval_interval == 0:
            var_model.eval()
            L_mean, L_tail, acc_mean, acc_tail, tot, eval_time = trainer.eval_ep(val_loader)
            print(f"Stage 2 - Epoch {epoch+1} Validation: "
                  f"Loss = {L_mean:.4f}/{L_tail:.4f} "
                  f"Acc = {acc_mean:.2f}%/{acc_tail:.2f}% "
                  f"Time = {eval_time:.1f}s\n")
        
        # 每10个epoch保存checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'var_state_dict': var_model.state_dict(),
                'ep_encoder_state_dict': var_model.ep_encoder.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'epoch': epoch,
                'loss': avg_epoch_loss,
                'acc': avg_epoch_acc,
            }
            torch.save(checkpoint, f'var_utah_train_monkeyF_two_stage_epoch_{epoch+1}.pth')
            print(f"✓ Checkpoint已保存: var_utah_train_monkeyF_two_stage_epoch_{epoch+1}.pth")

config = TrainingConfig()

# 开始两步训练
train_two_stage(config, vae=vae, var_model=var, clip_model=clip_model, 
                train_dataset=train_dataset, val_dataset=test_dataset)

# 保存最终模型
torch.save(var, 'var_utah_train_monkeyF_two_stage.pth')
print("两步训练完成，模型已保存")

# 测试阶段：评估语义重建效果
print("=" * 60)
print("开始评估语义重建效果")
print("=" * 60)

# 这里可以添加语义重建的评估代码
# 例如：生成图像并评估其与原始图像的语义相似性
