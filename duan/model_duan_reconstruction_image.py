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
from models.EP_encoder_temporal import TimeTransformerConvModel, ModelConfig


with open("/disk1/jinchentao/visual_decode/visual_reconstruction/dataset/trail_activity_500.pkl", 'rb') as f:
    trail_activity = pickle.load(f)

def create_angle_mapping():
    """
    创建对象映射（不考虑朝向）
    
    修改：将不同朝向的同一对象映射到相同的类别
    所有朝向（0度、45度、315度）的同一对象共享一个类别ID
    """
    angle_mapping = {}
    object_idx = 0  # 现在是对象索引而非角度索引
    
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
            # 为每个对象的所有朝向分配相同的类别ID
            if category == 'shape2d':
                angles = ['B1', 'G1', 'R1']  
            else:
                angles = ['0', '315', '45']  
            
            for angle in angles:
                angle_mapping[f"{img}_{angle}"] = object_idx
            
            # 每个对象只增加一次索引
            object_idx += 1
    
    return angle_mapping

angle_mapping = create_angle_mapping()
print(f"对象分类（不考虑朝向）: {len(set(angle_mapping.values()))} 个")
print(f"总标签数（包含所有朝向）: {len(angle_mapping)} 个")



class TrailActivityDataset(Dataset):
    def __init__(self, trail_activity, transform=None):
        self.trail_activity = trail_activity
        self.transform = transform
        
        self.data = []
        self.labels = []
        self.image_paths = []
        
        self.obj3d_root = "/disk1/jinchentao/visual_decode/visual_reconstruction/dataset/OBJ3D"
        
        for key, trials in trail_activity.items():
            trial_condition = int(key.split('_')[0])
            trial_target = int(key.split('_')[1])
            

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
            elif trial_condition <= 12:  
                category = 'fruits'
                if trial_condition == 9:
                    image_name = 'App1'
                elif trial_condition == 10:
                    image_name = 'Ban1'
                elif trial_condition == 11:
                    image_name = 'Pea1'
                elif trial_condition == 12:
                    image_name = 'Pin1'
            elif trial_condition <= 16:  
                category = 'manmade'
                if trial_condition == 13:
                    image_name = 'Bed1'
                elif trial_condition == 14:
                    image_name = 'Cha1'
                elif trial_condition == 15:
                    image_name = 'Dis1'
                elif trial_condition == 16:
                    image_name = 'Sof1'
            elif trial_condition <= 20:  
                category = 'plants'
                if trial_condition == 17:
                    image_name = 'A'
                elif trial_condition == 18:
                    image_name = 'B'
                elif trial_condition == 19:
                    image_name = 'C'
                elif trial_condition == 20:
                    image_name = 'D'
            elif trial_condition <= 24: 
                category = 'shape2d'
                if trial_condition == 21:
                    image_name = 'Cir'
                elif trial_condition == 22:
                    image_name = 'Oth'
                elif trial_condition == 23:
                    image_name = 'Squ'
                elif trial_condition == 24:
                    image_name = 'Tri'
            
            # 记录原始的角度信息（用于标签映射）
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
            
            # 修改：始终使用0度（或B1）的图片作为重建目标
            # 不论原始刺激是什么朝向，都用正朝向的图片
            if category == 'shape2d':
                image_angle = 'B1'  # shape2d类别使用B1作为默认
            else:
                image_angle = '0'   # 其他类别使用0度作为默认
            
            image_path = os.path.join(self.obj3d_root, category, f"{image_name}_{image_angle}.png")
            
            for trial_data in trials:
                self.data.append(trial_data)
                
                label = angle_mapping[f"{image_name}_{angle}"]
                self.labels.append(label)
                
                self.image_paths.append(image_path)
        
        self.image_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2 - 1)  
        ])
        
        self.background_mode = 'white'

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        data_tensor = torch.tensor(self.data[idx], dtype=torch.float32).squeeze(-1).T
        label = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        
        try:
            image_path = self.image_paths[idx]
            image = Image.open(image_path)
            
            if image.mode == 'RGBA':
                if self.background_mode == 'white':
                    background = Image.new('RGB', image.size, (255, 255, 255))
                    background.paste(image, mask=image.split()[-1])  
                    image = background
                elif self.background_mode == 'black':
                    background = Image.new('RGB', image.size, (0, 0, 0))
                    background.paste(image, mask=image.split()[-1])
                    image = background
                elif self.background_mode == 'crop':
                    bbox = image.getbbox()
                    if bbox:
                        image = image.crop(bbox)
                    else:
                        image = image.convert('RGB')
                else:
                    image = image.convert('RGB')
            elif image.mode != 'RGB':
                image = image.convert('RGB')
            
            img_tensor = self.image_transform(image)
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            img_tensor = torch.ones(3, 256, 256, dtype=torch.float32) * 0.5  
        
        if self.transform:
            data_tensor = self.transform(data_tensor)
            
        return data_tensor, img_tensor, label

angle_dataset = TrailActivityDataset(trail_activity)
print(f"Background model: {angle_dataset.background_mode}")

sample_data, sample_img, sample_label = angle_dataset[0]
angle_labels = [angle_dataset[i][2].item() for i in range(min(100, len(angle_dataset)))]

angle_labels = [angle_dataset[i][2].item() for i in range(len(angle_dataset))]
train_indices, test_indices = train_test_split(
    range(len(angle_dataset)), 
    test_size=0.2, 
    random_state=42, 
    stratify=angle_labels
)


train_dataset = TrailActivityDataset(trail_activity)
test_dataset = TrailActivityDataset(trail_activity)

from torch.utils.data import Subset
train_subset = Subset(train_dataset, train_indices)
test_subset = Subset(test_dataset, test_indices)

train_labels = [train_subset[i][2].item() for i in range(min(10, len(train_subset)))]
test_labels = [test_subset[i][2].item() for i in range(min(10, len(test_subset)))]

MODEL_DEPTH = 16   

vae_ckpt, var_ckpt = 'vae_ch160v4096z32.pth', f'var_d{MODEL_DEPTH}.pth'

FOR_512_px = MODEL_DEPTH == 16

patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

# 使用对象类别数（24个对象）而不是角度类别数（72个角度）
num_object_classes = len(set(angle_mapping.values()))
print(f"\n模型配置:")
print(f"  对象类别数: {num_object_classes}")
print(f"  不考虑朝向，只区分对象\n")

ep_config = ModelConfig(
    input_neuron=86,       
    time_bins=20,           
    d_model=150,           
    nhead=10,              
    num_transformer_layers=1,
    conv_channels=64,
    num_conv_blocks=3,
    num_classes=num_object_classes,  # 修改：使用24个对象类别
    residual_dims=[256, 512, 1024],
    use_positional_encoding=True,
    dim_feedforward_ratio=4,
    activation='relu'
)

vae, var = build_vae_var(
    V=4096, Cvae=32, ch=160, share_quant_resi=4,   
    device=device, patch_nums=patch_nums, config=ep_config,
    num_classes=num_object_classes, depth=MODEL_DEPTH, shared_aln=FOR_512_px,  # 修改：使用24个对象类别
    ep_encoder_type="spatial_activity"  
)

print("EP_encoder_type: spatial_activity")
vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)

class TrainingConfig:
    batch_size = 12
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
        "num_classes": 24,  # 修改：使用24个对象类别，不考虑朝向
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
    
    epochs = 55
    grad_accum = 1
    label_smooth = 0.1
    amp_enabled = True
    
    prog_epochs = 5 
    prog_warmup_iters = 1000  
    
    use_reconstruction_loss = True 
    reconstruction_loss_weight = 1.0  
    reconstruction_eval_freq = 10 
    
    background_mode = 'white' 
    
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
        amp_enabled: bool = False, rank: int = 0, config=None
    ):
        super(VARTrainer, self).__init__()
        
        self.var_model = var_model
        self.vae_local = vae_local
        self.quantize_local = vae_local.quantize
        self.optimizer = optimizer
        self.amp_enabled = amp_enabled
        self.device = device
        self.rank = rank
        self.config = config
        
        if hasattr(self.var_model, 'rng'):
            self.var_model.rng = torch.Generator(device=device)
        
        self.label_smooth = label_smooth
        self.train_loss = nn.CrossEntropyLoss(label_smoothing=label_smooth, reduction='none')
        self.val_loss = nn.CrossEntropyLoss(label_smoothing=0.0, reduction='mean')
        
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
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
        
        self.scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)

    def compute_reconstruction_loss(self, original_images, neuron_activity):
        B = original_images.shape[0]
        
        with torch.no_grad():
            reconstructed_images = self.var_model.autoregressive_infer_cfg(
                B=B, 
                neuron_activity=neuron_activity, 
                cfg=3,  
                top_k=500, 
                top_p=0.95, 
                more_smooth=True
            )
        
        mse_loss = self.mse_loss(reconstructed_images, original_images)
        l1_loss = self.l1_loss(reconstructed_images, original_images)
        
        reconstruction_loss = mse_loss + 0.1 * l1_loss
        
        return reconstruction_loss, reconstructed_images

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
            
            try:
                gt_idx_Bl = self.vae_local.img_to_idxBl(inp_B3HW)
                gt_BL = torch.cat(gt_idx_Bl, dim=1)
                x_BLCv_wo_first_l = self.quantize_local.idxBl_to_var_input(gt_idx_Bl)
            except Exception as e:
                print(f"Error in VAE img_to_idxBl: {e}")
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

            try:
                gt_idx_Bl = self.vae_local.img_to_idxBl(inp_B3HW)
                gt_BL = torch.cat(gt_idx_Bl, dim=1)
                x_BLCv_wo_first_l = self.quantize_local.idxBl_to_var_input(gt_idx_Bl)
            except Exception as e:
                print(f"Error in VAE img_to_idxBl during training: {e}")
                return 0.0, 0.0, 0.0, 1.0, 0.0
            
            logits_BLV = self.var_model(neuron_activity, x_BLCv_wo_first_l)

            pred_BL = logits_BLV.argmax(dim=-1)
            accuracy = (pred_BL == gt_BL).float().mean().item() * 100
            token_loss = self.train_loss(logits_BLV.view(-1, V), gt_BL.view(-1)).view(B, -1)
            
            if prog_si >= 0:
                bg, ed = self.begin_ends[prog_si]
                lw = self.loss_weight[:, :ed].clone()
                lw[:, bg:ed] *= prog_wp
            else:
                lw = self.loss_weight
                
            token_loss = token_loss.mul(lw).sum(dim=-1).mean()
            
            # 计算重建损失（如果启用）
            reconstruction_loss = torch.tensor(0.0, device=self.device)
            if (self.config and self.config.use_reconstruction_loss and 
                it % self.config.reconstruction_eval_freq == 0):
                reconstruction_loss, _ = self.compute_reconstruction_loss(inp_B3HW, neuron_activity)
            
            # 组合损失
            total_loss = token_loss + self.config.reconstruction_loss_weight * reconstruction_loss

        self.scaler.scale(total_loss).backward()
        
        grad_norm = 0
        if stepping:
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(self.var_model.parameters()), 
                max_norm=1.0
            ).item()
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        
        if hasattr(self.var_model, 'prog_si'):
            self.var_model.prog_si = -1
        if hasattr(self.vae_local.quantize, 'prog_si'):
            self.vae_local.quantize.prog_si = -1
            
        return (total_loss.item(), accuracy, grad_norm, self.scaler.get_scale(), 
                token_loss.item(), reconstruction_loss.item())

def evaluate_reconstruction_quality(val_loader, var_model, device, num_batches=5):
    from skimage.metrics import structural_similarity as ssim
    
    var_model.eval()
    total_mse = 0.0
    total_l1 = 0.0
    total_ssim = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for i, (neuron_activity, images, _) in enumerate(val_loader):
            if i >= num_batches:
                break
                
            neuron_activity = neuron_activity.to(device)
            images = images.to(device)
            B = len(neuron_activity)
            
            reconstructed = var_model.autoregressive_infer_cfg(
                B=B, 
                neuron_activity=neuron_activity, 
                cfg=3, 
                top_k=500, 
                top_p=0.95, 
                more_smooth=True
            )
            
            mse = torch.nn.functional.mse_loss(reconstructed, images).item()
            l1 = torch.nn.functional.l1_loss(reconstructed, images).item()
            
            orig_np = images.cpu().numpy()
            recon_np = reconstructed.cpu().numpy()
            
            orig_np = (orig_np + 1) / 2
            recon_np = (recon_np + 1) / 2
            orig_np = np.clip(orig_np, 0, 1)
            recon_np = np.clip(recon_np, 0, 1)
            
            batch_ssim = 0.0
            for j in range(B):
                orig_hwc = np.transpose(orig_np[j], (1, 2, 0))
                recon_hwc = np.transpose(recon_np[j], (1, 2, 0))
                ssim_val = ssim(orig_hwc, recon_hwc, multichannel=True, data_range=1.0, channel_axis=2)
                batch_ssim += ssim_val
            batch_ssim /= B
            
            total_mse += mse * B
            total_l1 += l1 * B
            total_ssim += batch_ssim * B
            total_samples += B
    
    return {
        'mse': total_mse / total_samples,
        'l1': total_l1 / total_samples,
        'ssim': total_ssim / total_samples
    }

def train_single_gpu(config, vae, var_model, train_dataset, val_dataset):
    torch.manual_seed(config.seed)
    if config.device.startswith("cuda"):
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
        list(var_model.parameters()),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=config.betas
    )
    
    trainer = VARTrainer(
        device=config.device,
        patch_nums=config.var_config["patch_nums"],
        resos=(16, 32, 48, 64, 80, 96, 128, 160, 208, 256),
        vae_local=vae,
        var_model=var_model,
        optimizer=optimizer,
        label_smooth=config.label_smooth,
        amp_enabled=config.amp_enabled,
        config=config
    )
    
    
    global_step = 0
    train_acc_history = []
    
    for epoch in range(config.epochs):
        var_model.train()        

        base_prog_si = min(epoch // config.prog_epochs + 1, len(config.var_config["patch_nums"]) - 1)
        
        if base_prog_si < len(config.var_config["patch_nums"]) - 1 and len(train_acc_history) > 0:
            recent_epochs = min(3, len(train_acc_history))  
            recent_avg_acc = sum(train_acc_history[-recent_epochs:]) / recent_epochs
            if recent_avg_acc >= 99.0:
                print(f"Val accuracy reached {recent_avg_acc:.2f}%, move to next stage")
                prog_si = base_prog_si + 1
            else:
                prog_si = base_prog_si
        else:
            prog_si = base_prog_si
        
        epoch_total_loss = 0.0
        epoch_token_loss = 0.0
        epoch_reconstruction_loss = 0.0
        epoch_acc = 0.0
        epoch_samples = 0
        reconstruction_eval_count = 0
        
        for i, (neuron_activity, images, _) in enumerate(train_loader):
            images = images.to(config.device, non_blocking=True)
            neuron_activity = neuron_activity.to(config.device, non_blocking=True)
            
            stepping = (i + 1) % config.grad_accum == 0
            
            total_loss, accuracy, grad_norm, scale, token_loss, recon_loss = trainer.train_step(
                it=i,
                g_it=global_step,
                stepping=stepping,
                inp_B3HW=images,
                neuron_activity=neuron_activity,
                prog_si=prog_si,
                prog_wp_it=config.prog_warmup_iters
            )
            
            batch_size = images.size(0)
            epoch_total_loss += total_loss * batch_size
            epoch_token_loss += token_loss * batch_size
            epoch_reconstruction_loss += recon_loss * batch_size
            epoch_acc += accuracy * batch_size
            epoch_samples += batch_size
            
            if recon_loss > 0:
                reconstruction_eval_count += 1

            global_step += 1
        
        avg_epoch_total_loss = epoch_total_loss / epoch_samples
        avg_epoch_token_loss = epoch_token_loss / epoch_samples
        avg_epoch_reconstruction_loss = epoch_reconstruction_loss / epoch_samples
        avg_epoch_acc = epoch_acc / epoch_samples
        
        train_acc_history.append(avg_epoch_acc)
        
        print(f"\nEpoch {epoch}/{config.epochs} Training:")
        print(f"  Total Loss = {avg_epoch_total_loss:.4f}")
        print(f"  Token Loss = {avg_epoch_token_loss:.4f}")
        if reconstruction_eval_count > 0:
            print(f"  Reconstruction Loss = {avg_epoch_reconstruction_loss:.4f} (eval {reconstruction_eval_count} times)")
        print(f"  Accuracy = {avg_epoch_acc:.2f}%")

        if epoch % config.eval_interval == 0:
            var_model.eval()
            L_mean, L_tail, acc_mean, acc_tail, tot, eval_time = trainer.eval_ep(val_loader)
            print(f"\nEpoch {epoch} Validation: "
                  f"Loss = {L_mean:.4f}/{L_tail:.4f} "
                  f"Acc = {acc_mean:.2f}%/{acc_tail:.2f}% "
                  f"Time = {eval_time:.1f}s")
            
            if config.use_reconstruction_loss and epoch % 5 == 0:  
                print("Eval Reconstruction Value...")
                reconstruction_metrics = evaluate_reconstruction_quality(val_loader, var_model, device, num_batches=3)
                print(f"  Reconstruction Metrics: "
                      f"MSE={reconstruction_metrics['mse']:.4f}, "
                      f"L1={reconstruction_metrics['l1']:.4f}, "
                      f"SSIM={reconstruction_metrics['ssim']:.4f}")
            print()
            
        
        if epoch % config.save_interval == 0:
            checkpoint = {
                "epoch": epoch,
                "var_model": var_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": trainer.scaler.state_dict()
            }
            torch.save(checkpoint, f"{config.checkpoint_dir}/ckpt_epoch{epoch}.pth")
            print(f"Saved checkpoint at epoch {epoch}")


config = TrainingConfig()

angle_dataset.background_mode = config.background_mode
train_dataset.background_mode = config.background_mode
test_dataset.background_mode = config.background_mode


train_single_gpu(config, vae=vae, var_model=var, train_dataset=train_subset, val_dataset=test_subset)

torch.save(var, 'var_duan.pth')

val_loader = DataLoader(
        test_subset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch.nn.functional as F

def calculate_reconstruction_metrics(original, reconstructed):
    orig_np = original.cpu().detach().numpy()
    recon_np = reconstructed.cpu().detach().numpy()
    
    orig_np = (orig_np + 1) / 2
    recon_np = (recon_np + 1) / 2
    
    orig_np = np.clip(orig_np, 0, 1)
    recon_np = np.clip(recon_np, 0, 1)
    
    batch_size = orig_np.shape[0]
    metrics = {
        'mse': [],
        'psnr': [],
        'ssim': [],
        'l1': []
    }
    
    for i in range(batch_size):
        orig_img = orig_np[i]
        recon_img = recon_np[i]
        
        orig_hwc = np.transpose(orig_img, (1, 2, 0))
        recon_hwc = np.transpose(recon_img, (1, 2, 0))
        
        mse = np.mean((orig_img - recon_img) ** 2)
        metrics['mse'].append(mse)
        
        psnr_val = psnr(orig_hwc, recon_hwc, data_range=1.0)
        metrics['psnr'].append(psnr_val)
        
        ssim_val = ssim(orig_hwc, recon_hwc, multichannel=True, data_range=1.0, channel_axis=2)
        metrics['ssim'].append(ssim_val)
        
        l1 = np.mean(np.abs(orig_img - recon_img))
        metrics['l1'].append(l1)
    
    avg_metrics = {}
    for key, values in metrics.items():
        avg_metrics[key] = np.mean(values)
        avg_metrics[f'{key}_std'] = np.std(values)
    
    return avg_metrics



vae.eval()
var.eval()
pdf_path = "reconstruction_results_angle_classification_500.pdf"

total_metrics = {
    'mse': [], 'psnr': [], 'ssim': [], 'l1': [],
    'mse_std': [], 'psnr_std': [], 'ssim_std': [], 'l1_std': []
}
batch_count = 0

print("开始重建质量评估...")
print("=" * 60)

with PdfPages(pdf_path) as pdf:
    for batch_idx, (neuron_activity, images, labels) in enumerate(val_loader):
        neuron_activity = neuron_activity.to(device)
        images = images.to(device)
        labels = labels.to(device)
        B = len(neuron_activity)
        cfg = 5
        
        # 进行重建
        recon_B3HW = var.autoregressive_infer_cfg(B=B, neuron_activity=neuron_activity, cfg=cfg, top_k=1000, top_p=0.99, more_smooth=True)
        
        # 计算重建质量指标
        batch_metrics = calculate_reconstruction_metrics(images, recon_B3HW)
        
        # 累积指标
        for key in total_metrics:
            if key in batch_metrics:
                total_metrics[key].append(batch_metrics[key])
        
        batch_count += 1
        
        # 打印当前batch的指标
        print(f"Batch {batch_idx + 1:3d}: MSE={batch_metrics['mse']:.4f}, "
              f"PSNR={batch_metrics['psnr']:.2f}dB, "
              f"SSIM={batch_metrics['ssim']:.4f}, "
              f"L1={batch_metrics['l1']:.4f}")
        
        for i in range(B):
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            
            original_img = images[i].cpu().detach().numpy()
            original_img = (original_img + 1) / 2  # 从[-1,1]转换到[0,1]
            original_img = np.transpose(original_img, (1, 2, 0))  # 从CHW转换为HWC
            axes[0].imshow(original_img)
            axes[0].set_title(f"Original\nLabel: {labels[i].item()}")
            axes[0].axis('off')
            
            recon_img = recon_B3HW[i].cpu().detach().numpy()
            recon_img = (recon_img + 1) / 2  # 从[-1,1]转换到[0,1]
            recon_img = np.transpose(recon_img, (1, 2, 0))  # 从CHW转换为HWC
            axes[1].imshow(recon_img)
            
            single_orig = images[i:i+1]
            single_recon = recon_B3HW[i:i+1]
            single_metrics = calculate_reconstruction_metrics(single_orig, single_recon)
            
            axes[1].set_title(f"Reconstructed\nSSIM: {single_metrics['ssim']:.3f}, "
                            f"PSNR: {single_metrics['psnr']:.1f}dB")
            axes[1].axis('off')
            
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

print("=" * 60)
print("重建质量评估总结:")
print("=" * 60)

if batch_count > 0:
    final_metrics = {}
    for key in total_metrics:
        if total_metrics[key]:  # 确保列表不为空
            final_metrics[key] = np.mean(total_metrics[key])
            final_metrics[f'{key}_std'] = np.std(total_metrics[key])
    
    print(f"总批次数: {batch_count}")
    print(f"总图像数: {batch_count * config.batch_size}")
    print()
    print("平均重建质量指标:")
    print(f"  MSE (均方误差):     {final_metrics['mse']:.6f} ± {final_metrics['mse_std']:.6f}")
    print(f"  L1 (平均绝对误差):  {final_metrics['l1']:.6f} ± {final_metrics['l1_std']:.6f}")
    print(f"  PSNR (峰值信噪比):  {final_metrics['psnr']:.2f} ± {final_metrics['psnr_std']:.2f} dB")
    print(f"  SSIM (结构相似性):  {final_metrics['ssim']:.4f} ± {final_metrics['ssim_std']:.4f}")
    print()
    
    if final_metrics['ssim'] > 0.8:
        quality = "优秀"
    elif final_metrics['ssim'] > 0.6:
        quality = "良好"
    elif final_metrics['ssim'] > 0.4:
        quality = "一般"
    else:
        quality = "较差"
    
    print(f"重建质量等级: {quality}")
    print("=" * 60)
    
else:
    print("没有处理任何批次，无法计算评估指标")
