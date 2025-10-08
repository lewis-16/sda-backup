import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
import os
import time
import warnings
from functools import partial

import torch.nn as nn
import pickle
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

# 禁用默认参数初始化以加快速度
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)

from models import VQVAE, build_vae_var


class Nat60kDataset(Dataset):
    """
    适配nat60k数据格式的数据集类
    """
    def __init__(self, neural_data, image_indices, image_folder, transform=None):
        """
        Args:
            neural_data: 形状为 (n_images, n_repeats, n_neurons) 的神经元响应数据
            image_indices: 形状为 (n_images,) 的图像索引数组
            image_folder: 图像文件夹路径
            transform: 图像变换
        """
        self.neural_data = neural_data
        self.image_indices = image_indices
        self.image_folder = image_folder
        self.transform = transform
        
        # 图像变换 - 调整到模型期望的尺寸
        self.image_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2 - 1)  # 归一化到[-1, 1]
        ])
        
        # 空图像张量作为备用
        self.empty_image_tensor = torch.zeros(3, 256, 256, dtype=torch.float32)
        
    def __len__(self):
        return len(self.image_indices)
    
    def __getitem__(self, idx):
        # 获取神经元响应数据 - 对所有重复试验取平均
        neural_response = self.neural_data[idx]  # shape: (n_repeats, n_neurons)
        neural_response_mean = np.mean(neural_response, axis=0)  # 平均所有重复试验
        
        # 获取图像索引
        image_idx = int(self.image_indices[idx])
        
        # 构建图像路径 - 根据image_preprocessing.ipynb中的命名格式
        image_path = os.path.join(self.image_folder, f"image_{image_idx:05d}.png")
        
        # 加载图像
        try:
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.image_transform(img)
        except (FileNotFoundError, OSError) as e:
            print(f"Warning: {image_path} not found or corrupted: {e}")
            img_tensor = self.empty_image_tensor.clone()
        
        # 应用额外的变换（如果有）
        if self.transform:
            neural_response_mean = self.transform(neural_response_mean)
            
        return torch.tensor(neural_response_mean, dtype=torch.float32), img_tensor, image_idx


def load_nat60k_data(data_path, image_folder):
    """
    加载nat60k数据集
    
    Args:
        data_path: .npz文件路径
        image_folder: 图像文件夹路径
    
    Returns:
        neural_data: 神经元响应数据 (n_images, n_repeats, n_neurons)
        image_indices: 图像索引 (n_images,)
    """
    print(f"正在加载数据: {data_path}")
    data = np.load(data_path)
    
    print("数据集键:", list(data.keys()))
    for key in data.keys():
        print(f"{key}: shape={data[key].shape}, dtype={data[key].dtype}")
    
    # 提取需要的数据
    ss_all = data['ss_all']  # shape: (n_images, n_repeats, n_neurons)
    istim_ss = data['istim_ss']  # shape: (n_images,)
    
    print(f"神经元响应数据形状: {ss_all.shape}")
    print(f"图像索引形状: {istim_ss.shape}")
    print(f"神经元数量: {ss_all.shape[2]}")
    print(f"重复试验次数: {ss_all.shape[1]}")
    
    return ss_all, istim_ss


class TrainingConfig:
    """训练配置类"""
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
    
    epochs = 55
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
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42


class VARTrainer(object):
    """VAR模型训练器"""
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
        self.scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)

    @torch.no_grad()
    def eval_ep(self, ld_val: DataLoader):
        """评估函数"""
        tot = 0
        L_mean, L_tail, acc_mean, acc_tail = 0, 0, 0, 0
        stt = time.time()
        training = self.var_model.training
        self.var_model.eval()
        
        for neuron_activity, inp_B3HW, _ in ld_val:
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
        """训练步骤"""
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
    """单GPU训练函数"""
    torch.manual_seed(config.seed)
    if config.device == "cuda":
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
        var_model=var_model,
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
            
        if epoch % config.save_interval == 0:
            checkpoint = {
                "epoch": epoch,
                "model": var_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": trainer.scaler.state_dict()
            }
            os.makedirs(config.checkpoint_dir, exist_ok=True)
            torch.save(checkpoint, f"{config.checkpoint_dir}/ckpt_epoch{epoch}.pth")
            print(f"Saved checkpoint at epoch {epoch}")


def main():
    """主函数"""
    print("开始加载FX8_nat60k数据集...")
    
    # 数据路径
    data_path = "/media/ubuntu/sda/visual_stimuli_pattern/image_nc/FX8_nat60k_2023_05_16.npz"
    image_folder = "/media/ubuntu/sda/visual_stimuli_pattern/image_nc/image"
    
    # 加载数据
    ss_all, istim_ss = load_nat60k_data(data_path, image_folder)
    
    # 创建数据集
    full_dataset = Nat60kDataset(ss_all, istim_ss, image_folder)
    
    # 分割训练和验证集
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    
    # 获取神经元数量
    actual_input_dim = ss_all.shape[2]  # 神经元数量
    print(f"神经元数量: {actual_input_dim}")
    
    # 模型配置
    MODEL_DEPTH = 16
    vae_ckpt, var_ckpt = 'vae_ch160v4096z32.pth', f'var_d{MODEL_DEPTH}.pth'
    FOR_512_px = MODEL_DEPTH == 16
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 构建模型
    vae, var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,   
        device=device, patch_nums=patch_nums, input_dim=actual_input_dim, num_classes_ep=98,
        num_classes=1000, depth=MODEL_DEPTH, shared_aln=FOR_512_px
    )
    
    # 加载预训练权重
    if os.path.exists(vae_ckpt):
        vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
        print(f"已加载VAE权重: {vae_ckpt}")
    else:
        print(f"警告: VAE权重文件不存在: {vae_ckpt}")
    
    if os.path.exists(var_ckpt):
        var.load_state_dict(torch.load(var_ckpt, map_location='cpu'), strict=False)
        print(f"已加载VAR权重: {var_ckpt}")
    else:
        print(f"警告: VAR权重文件不存在: {var_ckpt}")
    
    # 训练配置
    config = TrainingConfig()
    
    # 开始训练
    print("开始训练...")
    train_single_gpu(config, vae=vae, var_model=var, train_dataset=train_dataset, val_dataset=val_dataset)
    
    # 保存最终模型
    torch.save(var, 'var_FX8_nat60k.pth')
    print("训练完成，模型已保存为 var_FX8_nat60k.pth")
    
    # 生成重建结果
    print("生成重建结果...")
    vae.eval()
    var.eval()
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    pdf_path = "reconstruction_results_FX8_nat60k.pdf"
    with PdfPages(pdf_path) as pdf:
        for i, (neuron_activity, images, image_indices) in enumerate(val_loader):
            neuron_activity = neuron_activity.to(device)
            images = images.to(device)
            B = len(neuron_activity)
            cfg = 5
            
            recon_B3HW = var.autoregressive_infer_cfg(
                B=B, neuron_activity=neuron_activity, cfg=cfg, 
                top_k=1000, top_p=0.99, more_smooth=True
            )
            
            for j in range(B):
                fig, axes = plt.subplots(1, 2, figsize=(8, 4))
                
                # 显示原始RGB图像
                original_img = images[j].cpu().detach().numpy()
                original_img = (original_img + 1) / 2  # 从[-1,1]转换到[0,1]
                original_img = np.transpose(original_img, (1, 2, 0))  # 从CHW转换为HWC
                axes[0].imshow(original_img)
                axes[0].set_title(f"Original (idx: {image_indices[j]})")
                axes[0].axis('off')
                
                # 显示重建的RGB图像
                recon_img = recon_B3HW[j].cpu().detach().numpy()
                recon_img = (recon_img + 1) / 2  # 从[-1,1]转换到[0,1]
                recon_img = np.transpose(recon_img, (1, 2, 0))  # 从CHW转换为HWC
                axes[1].imshow(recon_img)
                axes[1].set_title("Reconstructed")
                axes[1].axis('off')
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
            
            # 限制生成的样本数量
            if i >= 10:  # 只生成前10个batch的结果
                break
    
    print(f"重建结果已保存为: {pdf_path}")


if __name__ == "__main__":
    main()
