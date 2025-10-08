import os.path as osp

import PIL.Image as PImage
from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS
from torchvision.transforms import InterpolationMode, transforms

import os
import os.path as osp
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import gc
import os
import shutil
import sys
import time
import warnings
from functools import partial

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import stats
import pickle
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset


# 加载trail_activity数据
with open("trail_activity.pkl", 'rb') as f:
    trail_activity = pickle.load(f)

print(f"trail_activity数据: {len(trail_activity)} 个条件")

# 创建标签映射
def create_label_mappings():
    """
    创建三种分类任务的标签映射：
    1. 大类分类 (6类): animals, faces, fruits, manmade, plants, shape2d
    2. 图片分类 (24张): 每类4张图片
    3. 角度分类 (72个): 每张图片3个角度/颜色
    """
    
    # 大类映射 (6类)
    category_mapping = {
        'animals': 0, 'faces': 1, 'fruits': 2, 
        'manmade': 3, 'plants': 4, 'shape2d': 5
    }
    
    # 图片映射 (24张)
    image_mapping = {
        # animals (0-3)
        'Dee1': 0, 'Ele': 1, 'Pig': 2, 'Rhi': 3,
        # faces (4-7) 
        'MA': 4, 'MB': 5, 'MC': 6, 'WA': 7,
        # fruits (8-11)
        'App1': 8, 'Ban1': 9, 'Pea1': 10, 'Pin1': 11,
        # manmade (12-15)
        'Bed1': 12, 'Cha1': 13, 'Dis1': 14, 'Sof1': 15,
        # plants (16-19)
        'A': 16, 'B': 17, 'C': 18, 'D': 19,
        # shape2d (20-23)
        'Cir': 20, 'Oth': 21, 'Squ': 22, 'Tri': 23
    }
    
    # 角度映射 (72个)
    angle_mapping = {}
    angle_idx = 0
    for category in ['animals', 'faces', 'fruits', 'manmade', 'plants', 'shape2d']:
        if category == 'animals':
            images = ['Dee1', 'Ele', 'Pig', 'Rhi']
        elif category == 'faces':
            images = ['MA', 'MB', 'MC', 'WA']
        elif category == 'fruits':
            images = ['App1', 'Ban1', 'Pea1', 'Pin1']
        elif category == 'manmade':
            images = ['Bed1', 'Cha1', 'Dis1', 'Sof1']
        elif category == 'plants':
            images = ['A', 'B', 'C', 'D']
        elif category == 'shape2d':
            images = ['Cir', 'Oth', 'Squ', 'Tri']
            
        for img in images:
            if category == 'shape2d':
                angles = ['B1', 'G1', 'R1']  # 颜色而不是角度
            else:
                angles = ['0', '315', '45']  # 角度
            for angle in angles:
                angle_mapping[f"{img}_{angle}"] = angle_idx
                angle_idx += 1
    
    return category_mapping, image_mapping, angle_mapping

# 创建标签映射
category_mapping, image_mapping, angle_mapping = create_label_mappings()

print(f"大类分类: {len(category_mapping)} 类")
print(f"图片分类: {len(image_mapping)} 张") 
print(f"角度分类: {len(angle_mapping)} 个")

# 打印数据示例
print("\n数据示例:")
sample_key = list(trail_activity.keys())[0]
print(f"键: {sample_key}")
print(f"该条件下的trials数量: {len(trail_activity[sample_key])}")
print(f"每个trial的神经元数量: {len(trail_activity[sample_key][0])}")
print(f"每个神经元数据形状: {trail_activity[sample_key][0][0].shape}")


class TrailActivityDataset(Dataset):
    def __init__(self, trail_activity, task_type='category', transform=None):
        """
        trail_activity: 从pkl文件加载的数据字典
        task_type: 'category', 'image', 'angle' 三种分类任务
        """
        self.trail_activity = trail_activity
        self.task_type = task_type
        self.transform = transform
        
        # 准备数据和标签
        self.data = []
        self.labels = []
        
        for key, trials in trail_activity.items():
            # key格式: "trial_condition_trial_target" (如 "1_1", "1_2", "1_3")
            trial_condition = int(key.split('_')[0])
            trial_target = int(key.split('_')[1])
            
            # 根据trial_condition确定图像类别
            if trial_condition <= 4:  # animals
                category = 'animals'
                if trial_condition == 1:
                    image_name = 'Dee1'
                elif trial_condition == 2:
                    image_name = 'Ele'
                elif trial_condition == 3:
                    image_name = 'Pig'
                elif trial_condition == 4:
                    image_name = 'Rhi'
            elif trial_condition <= 8:  # faces
                category = 'faces'
                if trial_condition == 5:
                    image_name = 'MA'
                elif trial_condition == 6:
                    image_name = 'MB'
                elif trial_condition == 7:
                    image_name = 'MC'
                elif trial_condition == 8:
                    image_name = 'WA'
            elif trial_condition <= 12:  # fruits
                category = 'fruits'
                if trial_condition == 9:
                    image_name = 'App1'
                elif trial_condition == 10:
                    image_name = 'Ban1'
                elif trial_condition == 11:
                    image_name = 'Pea1'
                elif trial_condition == 12:
                    image_name = 'Pin1'
            elif trial_condition <= 16:  # manmade
                category = 'manmade'
                if trial_condition == 13:
                    image_name = 'Bed1'
                elif trial_condition == 14:
                    image_name = 'Cha1'
                elif trial_condition == 15:
                    image_name = 'Dis1'
                elif trial_condition == 16:
                    image_name = 'Sof1'
            elif trial_condition <= 20:  # plants
                category = 'plants'
                if trial_condition == 17:
                    image_name = 'A'
                elif trial_condition == 18:
                    image_name = 'B'
                elif trial_condition == 19:
                    image_name = 'C'
                elif trial_condition == 20:
                    image_name = 'D'
            elif trial_condition <= 24:  # shape2d
                category = 'shape2d'
                if trial_condition == 21:
                    image_name = 'Cir'
                elif trial_condition == 22:
                    image_name = 'Oth'
                elif trial_condition == 23:
                    image_name = 'Squ'
                elif trial_condition == 24:
                    image_name = 'Tri'
            
            # 根据trial_target确定角度/颜色
            if category == 'shape2d':
                if trial_target == 1:
                    angle = 'B1'
                elif trial_target == 2:
                    angle = 'G1'
                elif trial_target == 3:
                    angle = 'R1'
            else:
                if trial_target == 1:
                    angle = '0'
                elif trial_target == 2:
                    angle = '315'
                elif trial_target == 3:
                    angle = '45'
            
            # 为每个trial添加数据
            for trial_data in trials:
                # trial_data shape: (86, 20, 1) - 86个神经元，20个时间bin
                self.data.append(trial_data)
                
                # 根据任务类型确定标签
                if task_type == 'category':
                    label = category_mapping[category]
                elif task_type == 'image':
                    label = image_mapping[image_name]
                elif task_type == 'angle':
                    label = angle_mapping[f"{image_name}_{angle}"]
                
                self.labels.append(label)
        
        # 图像变换
        self.image_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2 - 1)  # 归一化到[-1, 1]
        ])
        
        # 空图像张量作为备用
        self.empty_image_tensor = torch.zeros(3, 256, 256, dtype=torch.float32)

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        # 数据形状: (86, 20, 1) -> 展平为 (1720,) 以适应模型
        data_tensor = torch.tensor(self.data[idx], dtype=torch.float32).squeeze(-1).flatten()
        label = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        
        # 创建dummy图像张量（与原代码兼容）
        img_tensor = self.empty_image_tensor.clone()
        
        if self.transform:
            data_tensor = self.transform(data_tensor)
            
        return img_tensor, label  # 返回图像和标签，而不是数据张量

def build_dataset(task_type='category', final_reso=256):
    """
    构建trail_activity数据集
    Args:
        task_type: 'category', 'image', 'angle' 三种分类任务
        final_reso: 图像分辨率
    """
    transform = transforms.Compose([
        transforms.Resize((final_reso, final_reso)),  
        transforms.ToTensor(),                       
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Lambda(lambda x: x * 2 - 1)         
    ])
    
    # 创建完整数据集
    full_dataset = TrailActivityDataset(
        trail_activity=trail_activity,
        task_type=task_type,
        transform=transform
    )
    
    # 获取标签用于分层划分
    labels = [full_dataset[i][1].item() for i in range(len(full_dataset))]
    
    # 分层划分训练和验证集
    train_indices, val_indices = train_test_split(
        range(len(full_dataset)), 
        test_size=0.2, 
        random_state=42, 
        stratify=labels
    )
    
    # 创建训练和验证子集
    train_set = Subset(full_dataset, train_indices)
    val_set = Subset(full_dataset, val_indices)
    
    # 根据任务类型确定类别数量
    if task_type == 'category':
        num_classes = 6
    elif task_type == 'image':
        num_classes = 24
    elif task_type == 'angle':
        num_classes = 72
    
    print(f"任务类型: {task_type}, 类别数量: {num_classes}")
    print(f"训练集大小: {len(train_indices)}")
    print(f"验证集大小: {len(val_indices)}")
    
    return num_classes, train_set, val_set

import os
import os.path as osp
import torch, torchvision
import random
import numpy as np
import PIL.Image as PImage, PIL.ImageDraw as PImageDraw
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
from models import VQVAE, build_vae_var

MODEL_DEPTH = 16    # TODO: =====> please specify MODEL_DEPTH <=====


# download checkpoint
vae_ckpt, var_ckpt = 'vae_ch160v4096z32.pth', f'var_d{MODEL_DEPTH}.pth'

# build vae, var
FOR_512_px = MODEL_DEPTH == 16
# if FOR_512_px:
#     patch_nums = (1, 2, 3, 4, 6, 9, 13, 18, 24, 32)
# else:
#     patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)

patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)

# 设置CUDA设备，优先使用cuda:1，如果不可用则使用cuda:0
if torch.cuda.is_available():
    if torch.cuda.device_count() > 1:
        device = 'cuda:1'
    else:
        device = 'cuda:0'
else:
    device = 'cpu'
vae, var = build_vae_var(
    V=4096, Cvae=32, ch=160, share_quant_resi=4,    # hard-coded VQVAE hyperparameters
    device=device, patch_nums=patch_nums,
    num_classes=6, depth=MODEL_DEPTH, shared_aln=FOR_512_px,  # 使用大类分类的类别数量
)

# load checkpoints
vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
var.load_state_dict(torch.load(var_ckpt, map_location='cpu'), strict=False)

class TrainingConfig:
    batch_size = 16
    num_workers = 4
    image_size = 256
    task_type = 'category'  # 'category', 'image', 'angle'
    
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
        "num_classes": 6,  # 默认大类分类，会根据task_type动态调整
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
    
    epochs = 100
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

class VARTrainer(object):
    def __init__(
        self, device, patch_nums, resos,
        vae_local, var_model,
        optimizer: torch.optim.Optimizer, label_smooth: float,
        amp_enabled: bool = False
    ):
        super(VARTrainer, self).__init__()
        
        self.var_model = var_model
        self.vae_local = vae_local
        self.quantize_local = vae_local.quantize
        self.optimizer = optimizer
        self.amp_enabled = amp_enabled
        self.device = device
        
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
        
        self.scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    @torch.no_grad()
    def eval_ep(self, ld_val: DataLoader):
        tot = 0
        L_mean, L_tail, acc_mean, acc_tail = 0, 0, 0, 0
        stt = time.time()
        training = self.var_model.training
        self.var_model.eval()
        
        for inp_B3HW, label_B in ld_val:
            B, V = label_B.shape[0], self.vae_local.vocab_size
            inp_B3HW = inp_B3HW.to(self.device)
            label_B = label_B.to(self.device)
            
            gt_idx_Bl: List[torch.Tensor] = self.vae_local.img_to_idxBl(inp_B3HW)
            gt_BL = torch.cat(gt_idx_Bl, dim=1)
            x_BLCv_wo_first_l = self.quantize_local.idxBl_to_var_input(gt_idx_Bl)
            
            logits_BLV = self.var_model(label_B, x_BLCv_wo_first_l)
            
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
        inp_B3HW: torch.Tensor, label_B: torch.Tensor, prog_si: int, prog_wp_it: float
    ):
        self.var_model.prog_si = self.vae_local.quantize.prog_si = prog_si
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

        B, V = label_B.shape[0], self.vae_local.vocab_size
        
        with torch.cuda.amp.autocast(enabled=False):
            gt_idx_Bl = self.vae_local.img_to_idxBl(inp_B3HW)
            gt_BL = torch.cat(gt_idx_Bl, dim=1)
            x_BLCv_wo_first_l = self.quantize_local.idxBl_to_var_input(gt_idx_Bl)
            
            logits_BLV = self.var_model(label_B, x_BLCv_wo_first_l)

            pred_BL = logits_BLV.argmax(dim=-1)
            accuracy = (pred_BL == gt_BL).float().mean().item() * 100
            loss = self.train_loss(logits_BLV.view(-1, V), gt_BL.view(-1)).view(B, -1)
            
            # 应用渐进权重
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
        
        
        self.var_model.prog_si = self.vae_local.quantize.prog_si = -1
        return loss.item(), accuracy, grad_norm, self.scaler.get_scale()
    
def train_single_gpu(config, vae, var_model):
    torch.manual_seed(config.seed)
    if config.device.startswith("cuda"):
        torch.cuda.manual_seed_all(config.seed)
    
    num_classes, train_dataset, val_dataset = build_dataset(task_type=config.task_type, final_reso=256)
    
    # 更新var_config中的num_classes
    config.var_config["num_classes"] = num_classes
    
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
        
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(config.device, non_blocking=True)
            labels = labels.to(config.device, non_blocking=True)
            
            stepping = (i + 1) % config.grad_accum == 0
            
            loss_value, accuracy, grad_norm, scale = trainer.train_step(
                it=i,
                g_it=global_step,
                stepping=stepping,
                inp_B3HW=images,
                label_B=labels,
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
            
# 训练三种不同的分类任务
task_types = ['category', 'image', 'angle']
task_names = ['大类分类', '图片分类', '角度分类']

for task_type, task_name in zip(task_types, task_names):
    print(f"\n开始训练 {task_name} ({task_type})")
    
    config = TrainingConfig()
    config.task_type = task_type
    
    # 根据任务类型更新模型
    if task_type == 'category':
        num_classes = 6
    elif task_type == 'image':
        num_classes = 24
    elif task_type == 'angle':
        num_classes = 72
    
    # 重新构建模型以适应不同的类别数量
    vae, var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,
        device=device, patch_nums=patch_nums,
        num_classes=num_classes, depth=MODEL_DEPTH, shared_aln=FOR_512_px,
    )
    
    # 加载预训练权重
    vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
    var.load_state_dict(torch.load(var_ckpt, map_location='cpu'), strict=False)
    
    train_single_gpu(config, vae=vae, var_model=var)
    
    # 保存模型
    torch.save(var, f'var_{task_type}_20250801.pth')
    print(f"{task_name} 训练完成，模型已保存为 var_{task_type}_20250801.pth")