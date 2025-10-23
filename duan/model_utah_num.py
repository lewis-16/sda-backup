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


# 加载原始数据
filtered_test_MUA_full = np.load("/disk1/jinchentao/visual_decode/visual_reconstruction/dataset/filtered_test_MUA_MonkeyN.npy")
filtered_labels = np.load("/disk1/jinchentao/visual_decode/visual_reconstruction/dataset/filtered_labels_MonkeyN.npy")

print(f"原始数据维度: {filtered_test_MUA_full.shape}")

# 定义要测试的维度
TEST_DIMENSIONS = [100, 200, 400, 600]
print(f"将测试的输入维度: {TEST_DIMENSIONS}")

# 添加神经元采样函数
def sample_neurons(mua_data, n_neurons, seed=42):
    """
    从MUA数据中随机采样指定数量的神经元
    
    Args:
        mua_data: 原始MUA数据 (n_samples, n_neurons)
        n_neurons: 要采样的神经元数量
        seed: 随机种子
    
    Returns:
        sampled_data: 采样后的数据 (n_samples, n_neurons)
        selected_indices: 被选中的神经元索引
    """
    np.random.seed(seed)
    total_neurons = mua_data.shape[1]
    
    if n_neurons >= total_neurons:
        print(f"  警告: 请求的神经元数({n_neurons})大于等于总数({total_neurons})，使用全部神经元")
        return mua_data, np.arange(total_neurons)
    
    # 随机选择神经元索引
    selected_indices = np.random.choice(total_neurons, n_neurons, replace=False)
    selected_indices = np.sort(selected_indices)  # 排序以保持一致性
    
    # 采样数据
    sampled_data = mua_data[:, selected_indices]
    
    print(f"  从 {total_neurons} 个神经元中随机采样 {n_neurons} 个")
    print(f"  采样后数据维度: {sampled_data.shape}")
    
    return sampled_data, selected_indices

# 添加图像路径加载
def load_image_paths_for_utah():
    """
    为Utah数据集加载对应的图像路径
    返回图像路径列表，与filtered_labels对应
    """
    # 读取CSV文件获取图像路径
    csv_path = "/disk1/jinchentao/visual_decode/visual_reconstruction/dataset/test_image_MonkeyN.csv"
    df = pd.read_csv(csv_path)
    
    # 创建图像路径列表，与filtered_labels对应
    image_paths = []
    for label in filtered_labels:
        if label < len(df):
            image_path = df.iloc[label]['local_path']
            image_paths.append(image_path)
        else:
            # 如果标签超出范围，使用空路径
            image_paths.append("")
    
    return image_paths

# 加载图像路径
image_paths = load_image_paths_for_utah()
print(f"加载了 {len(image_paths)} 个图像路径")

# 检查数据维度
print(f"filtered_labels shape: {filtered_labels.shape}")
print(f"unique labels: {len(np.unique(filtered_labels))}")


class MUAClassificationDataset(Dataset):
    def __init__(self, mua_data, labels, image_paths=None, transform=None):
        self.mua_data = torch.tensor(mua_data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.image_paths = image_paths
        self.transform = transform
        
        # 图像变换
        self.image_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2 - 1)  # 归一化到[-1, 1]
        ])
        
        # 空图像张量作为备用
        self.empty_image_tensor = torch.zeros(3, 256, 256, dtype=torch.float32)
        
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
        
        if self.transform:
            mua_sample = self.transform(mua_sample)
            
        return mua_sample, img_tensor, label

def create_label_mapping(original_labels, num_classes=91):

    unique_labels = np.unique(original_labels)
    
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
    
    mapped_labels = np.array([label_mapping[label] for label in original_labels])
    
    
    return mapped_labels, label_mapping

# 创建标签映射（所有维度共用）
mapped_labels, label_mapping = create_label_mapping(filtered_labels, num_classes=91)

# 创建训练/测试索引（所有维度共用）
train_indices, test_indices = train_test_split(
    range(len(filtered_test_MUA_full)), 
    test_size=0.2, 
    random_state=42, 
    stratify=mapped_labels
)

print(f"训练样本数: {len(train_indices)}, 测试样本数: {len(test_indices)}")

MODEL_DEPTH = 16
num_classes = len(np.unique(filtered_labels))

# 用于存储所有维度的结果
all_dimension_results = {}

vae_ckpt, var_ckpt = 'vae_ch160v4096z32.pth', f'var_d{MODEL_DEPTH}.pth'
FOR_512_px = MODEL_DEPTH == 16
patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

# 定义训练配置（所有维度共用）
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
        "depth": 16,
        "embed_dim": 1024,
        "num_heads": 16,
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
    
    epochs = 45
    grad_accum = 1
    label_smooth = 0.1
    amp_enabled = True
    
    prog_epochs = 5 
    prog_warmup_iters = 1000  
    
    log_interval = 50
    eval_interval = 1
    save_interval = 5
    checkpoint_dir = "checkpoints"
    log_dir = "logs"
    
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    seed = 42

# 辅助函数定义（用于评估）
def compute_correlation(img1, img2):
    """计算Pearson相关系数"""
    img1_flat = img1.flatten()
    img2_flat = img2.flatten()
    
    mean1 = img1_flat.mean()
    mean2 = img2_flat.mean()
    
    numerator = ((img1_flat - mean1) * (img2_flat - mean2)).sum()
    denominator = torch.sqrt(((img1_flat - mean1) ** 2).sum() * ((img2_flat - mean2) ** 2).sum())
    
    if denominator > 0:
        return (numerator / denominator).item()
    else:
        return 0.0

def compute_cosine_similarity(img1, img2):
    """计算余弦相似度"""
    img1_flat = img1.flatten()
    img2_flat = img2.flatten()
    
    dot_product = (img1_flat * img2_flat).sum()
    norm1 = torch.sqrt((img1_flat ** 2).sum())
    norm2 = torch.sqrt((img2_flat ** 2).sum())
    
    if norm1 > 0 and norm2 > 0:
        return (dot_product / (norm1 * norm2)).item()
    else:
        return 0.0

def correct_brightness_contrast_batch(recon_imgs, original_imgs, method='linear'):
    """
    批量校正重建图像的亮度和对比度
    
    Args:
        recon_imgs: 重建图像 tensor (B, C, H, W), 范围[0,1]
        original_imgs: 原始图像 tensor (B, C, H, W), 范围[0,1]
        method: 校正方法 ('linear', 'histogram', 'stretch', 'none')
    
    Returns:
        corrected_imgs: 校正后的图像 tensor (B, C, H, W), 范围[0,1]
    """
    if method == 'none':
        return recon_imgs
    
    B = recon_imgs.shape[0]
    corrected_imgs = recon_imgs.clone()
    
    for i in range(B):
        if method == 'linear':
            # 方法1: 线性亮度对比度调整（最优缩放和偏移）
            # 对每个通道分别处理
            for c in range(3):
                recon_flat = recon_imgs[i, c].flatten()
                orig_flat = original_imgs[i, c].flatten()
                
                # 计算最优的缩放因子和偏移量（最小二乘法）
                recon_mean = recon_flat.mean()
                orig_mean = orig_flat.mean()
                recon_std = recon_flat.std()
                orig_std = orig_flat.std()
                
                # 标准化匹配：scale = std_orig/std_recon, offset = mean_orig - scale*mean_recon
                if recon_std > 1e-6:
                    scale = orig_std / recon_std
                    offset = orig_mean - scale * recon_mean
                    corrected_imgs[i, c] = recon_imgs[i, c] * scale + offset
                else:
                    corrected_imgs[i, c] = recon_imgs[i, c]
            
            # 限制到[0,1]范围
            corrected_imgs[i] = torch.clamp(corrected_imgs[i], 0, 1)
            
        elif method == 'histogram':
            # 方法2: 直方图匹配
            recon_np = recon_imgs[i].cpu().numpy().transpose(1, 2, 0)  # HWC
            orig_np = original_imgs[i].cpu().numpy().transpose(1, 2, 0)
            
            # 转换到uint8
            recon_np = (recon_np * 255).astype(np.uint8)
            orig_np = (orig_np * 255).astype(np.uint8)
            
            # 对每个通道进行直方图匹配
            matched = np.zeros_like(recon_np)
            for c in range(3):
                # 计算累积直方图
                recon_hist, _ = np.histogram(recon_np[:,:,c].flatten(), 256, [0, 256])
                orig_hist, _ = np.histogram(orig_np[:,:,c].flatten(), 256, [0, 256])
                
                recon_cdf = recon_hist.cumsum()
                recon_cdf = recon_cdf / recon_cdf[-1]
                
                orig_cdf = orig_hist.cumsum()
                orig_cdf = orig_cdf / orig_cdf[-1]
                
                # 创建映射表
                mapping = np.zeros(256, dtype=np.uint8)
                for j in range(256):
                    # 找到最接近的原始CDF值
                    idx = np.argmin(np.abs(orig_cdf - recon_cdf[j]))
                    mapping[j] = idx
                
                # 应用映射
                matched[:,:,c] = mapping[recon_np[:,:,c]]
            
            # 转回tensor
            matched = matched.astype(np.float32) / 255.0
            corrected_imgs[i] = torch.from_numpy(matched.transpose(2, 0, 1)).to(recon_imgs.device)
            
        elif method == 'stretch':
            # 方法3: 对比度拉伸（将重建图像的范围拉伸到原始图像的范围）
            for c in range(3):
                recon_min = recon_imgs[i, c].min()
                recon_max = recon_imgs[i, c].max()
                orig_min = original_imgs[i, c].min()
                orig_max = original_imgs[i, c].max()
                
                # 拉伸
                if recon_max - recon_min > 1e-6:
                    corrected_imgs[i, c] = (recon_imgs[i, c] - recon_min) / (recon_max - recon_min)
                    corrected_imgs[i, c] = corrected_imgs[i, c] * (orig_max - orig_min) + orig_min
                else:
                    corrected_imgs[i, c] = recon_imgs[i, c]
            
            corrected_imgs[i] = torch.clamp(corrected_imgs[i], 0, 1)
    
    return corrected_imgs

# VARTrainer和train_single_gpu等函数定义在下方

class VARTrainer(object):
    def __init__(
        self, device, patch_nums, resos,
        vae_local, var_model,
        optimizer: torch.optim.Optimizer, label_smooth: float,
        amp_enabled: bool = False, rank: int = 0
    ):
        super(VARTrainer, self).__init__()
        
        self.var_model = var_model
        self.vae_local = vae_local
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
        self.scaler = torch.amp.GradScaler('cuda:1', enabled=amp_enabled)

    @torch.no_grad()
    def eval_ep(self, ld_val: DataLoader):
        tot = 0
        L_mean, L_tail, acc_mean, acc_tail = 0, 0, 0, 0
        stt = time.time()
        training = self.var_model.training
        self.var_model.eval()
        
        for neuron_activity, inp_B3HW, _  in ld_val:
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
        with torch.amp.autocast('cuda:1', enabled=False):
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


def train_single_gpu(config, vae, var_model, train_dataset, val_dataset):
    torch.manual_seed(config.seed)
    if config.device == "cuda:1":
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
    
    trainer = VARTrainer(
        device=config.device,
        patch_nums=config.var_config["patch_nums"],
        resos=(16, 32, 48, 64, 80, 96, 128, 160, 208, 256),
        vae_local=vae,
        var_model=var,
        optimizer=optimizer,
        label_smooth=config.label_smooth,
        amp_enabled=config.amp_enabled
    )
    
    
    global_step = 0
    for epoch in range(config.epochs):
        var_model.train()        
        prog_si = min(epoch // config.prog_epochs + 1, len(config.var_config["patch_nums"]) - 1)
        
        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_samples = 0
        
        for i, (neuron_activity, images, _) in enumerate(train_loader):
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
        
        avg_epoch_loss = epoch_loss / epoch_samples
        avg_epoch_acc = epoch_acc / epoch_samples
        
        print(f"\nEpoch {epoch}/{config.epochs} Training: "
              f"Loss = {avg_epoch_loss:.4f} "
              f"Acc = {avg_epoch_acc:.2f}%")

        if epoch % config.eval_interval == 0:
            var_model.eval()
            L_mean, L_tail, acc_mean, acc_tail, tot, eval_time = trainer.eval_ep(val_loader)
            print(f"\nEpoch {epoch} Validation: "
                  f"Loss = {L_mean:.4f}/{L_tail:.4f} "
                  f"Acc = {acc_mean:.2f}%/{acc_tail:.2f}% "
                  f"Time = {eval_time:.1f}s\n")
            
        
        # if epoch % config.save_interval == 0:
        #     checkpoint = {
        #         "epoch": epoch,
        #         "model": trainer.state_dict(),
        #         "optimizer": optimizer.state_dict(),
        #         "scaler": trainer.scaler.state_dict()
        #     }
        #     torch.save(checkpoint, f"{config.checkpoint_dir}/ckpt_epoch{epoch}.pth")
        #     print(f"Saved checkpoint at epoch {epoch}")




# ========== 开始维度测试循环 ==========
print("\n" + "="*90)
print("开始多维度测试实验")
print("="*90)

config = TrainingConfig()

for test_dim in TEST_DIMENSIONS:
    print("\n" + "="*80)
    print(f"测试维度: {test_dim} / {filtered_test_MUA_full.shape[1]}")
    print("="*80)
    
    # 1. 采样神经元
    filtered_test_MUA, selected_neuron_indices = sample_neurons(
        filtered_test_MUA_full, test_dim, seed=42
    )
    
    # 2. 创建数据集
    train_dataset = MUAClassificationDataset(
        filtered_test_MUA[train_indices], 
        mapped_labels[train_indices],
        image_paths=[image_paths[i] for i in train_indices]
    )
    
    test_dataset = MUAClassificationDataset(
        filtered_test_MUA[test_indices], 
        mapped_labels[test_indices],
        image_paths=[image_paths[i] for i in test_indices]
    )
    
    print(f"EP配置 - 输入维度: {test_dim}, 类别数量: {num_classes}")
    
    # 3. 构建模型
    vae, var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,   
        device=device, patch_nums=patch_nums, input_dim=test_dim, num_classes_ep=98,
        num_classes=1000, depth=MODEL_DEPTH, shared_aln=FOR_512_px
    )
    
    vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
    var.load_state_dict(torch.load(var_ckpt, map_location='cpu'), strict=False)
    
    # 4. 训练模型
    print(f"\n开始训练 - 维度: {test_dim}, Epochs: {config.epochs}")
    train_single_gpu(config, vae=vae, var_model=var, train_dataset=train_dataset, val_dataset=test_dataset)
    
    # 5. 保存模型
    model_save_path = f'var_utah_monkeyN_dim{test_dim}.pth'
    torch.save(var, model_save_path)
    print(f"模型已保存: {model_save_path}")
    
    # 6. 评估重建质量（计算相关性）
    print(f"\n开始评估重建质量 - 维度: {test_dim}")
    val_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    vae.eval()
    var.eval()
    
    # 测试不同的校正方法
    correction_methods = ['linear']
    num_repeats = 5
    num_batches = len(val_loader)
    
    metrics_names = ['Cosine']
    
    # 为每种方法和每个指标创建结果数组
    results_dict = {}
    for method in correction_methods:
        results_dict[method] = {
            metric: np.zeros((num_repeats, num_batches)) 
            for metric in metrics_names
        }
    
    print(f"  开始计算相关性指标，共{num_repeats}次重复，每次{num_batches}个batch")
    
    # 重复5次
    for repeat_idx in range(num_repeats):
        print(f"  第 {repeat_idx + 1}/{num_repeats} 次重复")
        
        for batch_idx, (neuron_activity, images, labels) in enumerate(val_loader):
            neuron_activity = neuron_activity.to(device)
            images = images.to(device)
            B = neuron_activity.shape[0]
            cfg = 5
            
            # 生成重建图像
            recon_B3HW = var.autoregressive_infer_cfg(
                B=B, 
                neuron_activity=neuron_activity, 
                cfg=cfg, 
                top_k=1000, 
                top_p=0.99, 
                more_smooth=True
            )
            
            original_imgs = (images + 1) / 2  # 从[-1,1]转换到[0,1]
            recon_imgs = (recon_B3HW + 1) / 2  # 从[-1,1]转换到[0,1]
            
            for method in correction_methods:
                corrected_imgs = correct_brightness_contrast_batch(recon_imgs, original_imgs, method=method)
                
                cos_values = []
                for i in range(B):
                    cos_sim = compute_cosine_similarity(corrected_imgs[i], original_imgs[i])
                    cos_values.append(cos_sim)
                results_dict[method]['Cosine'][repeat_idx, batch_idx] = np.mean(cos_values)
    
    # 保存当前维度的结果
    for method in correction_methods:
        for metric in metrics_names:
            save_path = f"metric_{metric.lower()}_{method}_dim{test_dim}_MonkeyN.npy"
            np.save(save_path, results_dict[method][metric])
            
            mean_val = results_dict[method][metric].mean()
            std_val = results_dict[method][metric].std()
            
            print(f"  {metric}: {mean_val:7.4f} ± {std_val:.4f}")
            print(f"  结果已保存: {save_path}")
    
    # 存储到总结果中
    all_dimension_results[test_dim] = {
        'mean_cosine': results_dict['linear']['Cosine'].mean(),
        'std_cosine': results_dict['linear']['Cosine'].std(),
        'selected_neurons': selected_neuron_indices
    }
    
    # 清理显存
    del vae, var, train_dataset, test_dataset, val_loader
    torch.cuda.empty_cache()
    print(f"\n维度 {test_dim} 测试完成！")

# ========== 循环结束，总结结果 ==========
print("\n" + "="*90)
print("所有维度测试完成！结果总结：")
print("="*90)

summary_results = []
for test_dim in TEST_DIMENSIONS:
    if test_dim in all_dimension_results:
        result = all_dimension_results[test_dim]
        print(f"维度 {test_dim:3d}: 余弦相似度 = {result['mean_cosine']:.4f} ± {result['std_cosine']:.4f}")
        summary_results.append({
            'dimension': test_dim,
            'mean_cosine': result['mean_cosine'],
            'std_cosine': result['std_cosine']
        })

# 保存总结果
np.save('dimension_test_summary_MonkeyN.npy', {'results': summary_results, 'dimensions': TEST_DIMENSIONS})
print(f"\n总结结果已保存: dimension_test_summary_MonkeyN.npy")
print("="*90)

# ========== 以下是原来的单次评估代码（已被上面的循环替代） ==========
# from matplotlib.backends.backend_pdf import PdfPages
# import matplotlib.pyplot as plt
# from pytorch_msssim import ssim, ms_ssim
# import cv2

# 注意：compute_correlation, compute_cosine_similarity, correct_brightness_contrast_batch 
# 这些函数已在文件前面定义（TrainingConfig类之后）


# ========== 以下原始评估代码已被上面的循环替代，已注释 ==========
# vae.eval()
# var.eval()
# 
# # 测试不同的校正方法
# correction_methods = ['linear']
# num_repeats = 5
# num_batches = len(val_loader)
# 
# metrics_names = ['Cosine']
# 
# # 为每种方法和每个指标创建结果数组
# results_dict = {}
# for method in correction_methods:
#     results_dict[method] = {
#         metric: np.zeros((num_repeats, num_batches)) 
#         for metric in metrics_names
#     }
# 
# print(f"开始计算多种图像质量指标，共{num_repeats}次重复，每次{num_batches}个batch")
# print(f"校正方法: {correction_methods}")
# print(f"评估指标: {metrics_names}\n")
# 
# # 重复5次
# for repeat_idx in range(num_repeats):
#     print(f"第 {repeat_idx + 1}/{num_repeats} 次重复")
#     
#     for batch_idx, (neuron_activity, images, labels) in enumerate(val_loader):
#         neuron_activity = neuron_activity.to(device)
#         images = images.to(device)
#         B = neuron_activity.shape[0]
#         cfg = 5
#         
#         # 生成重建图像
#         recon_B3HW = var.autoregressive_infer_cfg(
#             B=B, 
#             neuron_activity=neuron_activity, 
#             cfg=cfg, 
#             top_k=1000, 
#             top_p=0.99, 
#             more_smooth=True
#         )
#         
#         original_imgs = (images + 1) / 2  # 从[-1,1]转换到[0,1]
#         recon_imgs = (recon_B3HW + 1) / 2  # 从[-1,1]转换到[0,1]
#         
#         for method in correction_methods:
#             corrected_imgs = correct_brightness_contrast_batch(recon_imgs, original_imgs, method=method)
#             
#             cos_values = []
#             for i in range(B):
#                 cos_sim = compute_cosine_similarity(corrected_imgs[i], original_imgs[i])
#                 cos_values.append(cos_sim)
#             results_dict[method]['Cosine'][repeat_idx, batch_idx] = np.mean(cos_values)
#         
# 
# 
# for method in correction_methods:
#     print(f"\n{'='*80}")
#     print(f"{method.upper()} 方法:")
#     print(f"{'='*80}")
#     
#     for metric in metrics_names:
#         save_path = f"metric_{metric.lower().replace('-','_')}_{method}_MonkeyN.npy"
#         np.save(save_path, results_dict[method][metric])
#         
#         mean_val = results_dict[method][metric].mean()
#         std_val = results_dict[method][metric].std()
#         
#         # 对于LPIPS，越小越好，标注一下
#         better_indicator = "↓" if metric == 'LPIPS' else "↑"
#         
#         print(f"{metric:15s}: {mean_val:7.4f} ± {std_val:.4f} {better_indicator}")
#     
#     print(f"结果已保存，文件前缀: metric_*_{method}_MonkeyN.npy")
