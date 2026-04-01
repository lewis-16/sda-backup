import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pickle
import os
import time
import math
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

from utils_clique import create_label_mapping_from_classes

# 定义PSTH数据路径（与 generate_psth.ipynb 生成的一致）
# generate_psth.ipynb 输出到: /media/ubuntu/sda/mouse_test/processed_results/psth_results
base_data_dir = "/media/ubuntu/sda/mouse_test/processed_results/psth_results"
target_month_session_names = [
    'mouse6_021322_natural_image_001',  # 第1个月
    'mouse6_022522_natural_image_001',  # 第2个月
    'mouse6_031722_natural_image_001',  # 第3个月
    'mouse6_042422_natural_image_001',  # 第4个月
    'mouse6_052422_natural_image_001',  # 第5个月
    'mouse6_062422_natural_image_001',  # 第6个月
    'mouse6_072322_natural_image_001',  # 第7个月
    'mouse6_082322_natural_image_001',  # 第8个月
    'mouse6_092422_natural_image_001',  # 第9个月
    'mouse6_102122_natural_image_001',  # 第10个月
    'mouse6_112022_natural_image_001',  # 第11个月
]

def load_psth_data_from_sessions(base_data_dir, session_names):
    """
    从前11个月的session中加载并合并PSTH数据
    Args:
        base_data_dir: PSTH数据基础目录
        session_names: session名称列表
    Returns:
        train_MUA: 合并后的PSTH数据数组 (n_trials, n_time_bins, n_neurons)
        trial_image_ids: 合并后的图像ID列表
    """
    print(f"\n{'='*60}")
    print(f"加载前11个月session的PSTH数据")
    print(f"{'='*60}")
    
    all_psth_matrices = []
    all_trial_image_ids = []
    
    for session_name in session_names:
        session_dir = os.path.join(base_data_dir, session_name)
        psth_path = os.path.join(session_dir, "psth_matrix.npy")
        trial_image_id_path = os.path.join(session_dir, "trial_image_id.pkl")
        
        if not os.path.exists(psth_path):
            print(f"警告: 未找到PSTH数据文件: {psth_path}，跳过")
            continue
        
        if not os.path.exists(trial_image_id_path):
            print(f"警告: 未找到图像ID文件: {trial_image_id_path}，跳过")
            continue
        
        # 加载该session的PSTH数据
        session_psth = np.load(psth_path)
        print(f"  {session_name}: PSTH矩阵形状 {session_psth.shape}")
        
        # 加载该session的图像ID
        with open(trial_image_id_path, 'rb') as f:
            session_image_ids = pickle.load(f)
        
        all_psth_matrices.append(session_psth)
        all_trial_image_ids.extend(session_image_ids)
    
    # 合并所有session的数据
    if len(all_psth_matrices) == 0:
        raise ValueError("未找到任何有效的PSTH数据文件")
    
    train_MUA = np.concatenate(all_psth_matrices, axis=0)
    trial_image_ids = all_trial_image_ids
    
    # 只使用刺激期间的PSTH数据（索引5:25，对应0.25-1.25秒，刺激期间）
    # 原始PSTH包含：前0.25秒(0-0.25s, bins 0-4) + 刺激1秒(0.25-1.25s, bins 5-24) + 后0.25秒(1.25-1.5s, bins 25-29)
    # 使用bins 5:25 (20个bins)，对应时间0.25-1.25秒，与notebook保持一致
    psth_start_bin = 5
    psth_end_bin = 25
    train_MUA = train_MUA[:, psth_start_bin:psth_end_bin, :]  # (n_trials, 20, n_neurons)
    
    print(f"\n合并后数据形状（原始）: {np.concatenate(all_psth_matrices, axis=0).shape}")
    print(f"合并后数据形状（切片后，只使用刺激期间bins 5:25）: {train_MUA.shape}")
    print(f"合并后图像ID数量: {len(trial_image_ids)}")
    
    # 检查数据中的NaN和Inf
    print(f"\n数据统计:")
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
    
    return train_MUA, trial_image_ids

# 加载前11个月的数据（用于训练）
train_MUA, trial_image_ids = load_psth_data_from_sessions(base_data_dir, target_month_session_names)

print(f"总共加载了 {len(trial_image_ids)} 个图像ID")

# 加载第12个月的数据（用于测试）
val_session_name = 'mouse6_122022_natural_image_001'  # 第12个月
print(f"\n{'='*60}")
print(f"加载第12个月session的PSTH数据（用于测试）")
print(f"{'='*60}")

val_session_dir = os.path.join(base_data_dir, val_session_name)
val_psth_path = os.path.join(val_session_dir, "psth_matrix.npy")
val_trial_image_id_path = os.path.join(val_session_dir, "trial_image_id.pkl")

if not os.path.exists(val_psth_path) or not os.path.exists(val_trial_image_id_path):
    raise ValueError(f"未找到第12个月的数据文件: {val_session_name}")

val_psth_matrix = np.load(val_psth_path)
print(f"  {val_session_name}: PSTH矩阵形状 {val_psth_matrix.shape}")

with open(val_trial_image_id_path, 'rb') as f:
    val_trial_image_ids = pickle.load(f)

# 使用相同的切片范围（bins 5:25）
psth_start_bin = 5
psth_end_bin = 25
val_psth_matrix = val_psth_matrix[:, psth_start_bin:psth_end_bin, :]  # (n_trials, 20, n_neurons)

print(f"  切片后形状（bins 5:25）: {val_psth_matrix.shape}")
print(f"  图像ID数量: {len(val_trial_image_ids)}")

# 处理可能的NaN和Inf
if np.isnan(val_psth_matrix).any() or np.isinf(val_psth_matrix).any():
    print("警告: 测试数据中包含NaN或Inf，将进行清理...")
    val_psth_matrix = np.nan_to_num(val_psth_matrix, nan=0.0, posinf=1.0, neginf=-1.0)

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
    输入: (B, time_bins, neurons)
    输出: tokens (B, n_token, Cvae)
    """
    def __init__(self, input_dim=31, time_bins=30, d_model=32, n_token=128, 
                 num_conv_layers=2, dropout=0.2, Cvae=32, num_classes=117):
        super().__init__()
        self.input_dim = input_dim
        self.time_bins = time_bins
        self.d_model = d_model
        self.n_token = n_token
        self.Cvae = Cvae
        self.num_classes = num_classes
        
        # 输入投影：对神经元维度降维
        if input_dim > 200:
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
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, d_model * 4),
                nn.LayerNorm(d_model * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model),
            )
        
        # 时间维度的1D卷积
        conv_layers = []
        for i in range(num_conv_layers):
            if i == 0:
                in_channels = d_model
            else:
                in_channels = d_model * 2
            
            if i == num_conv_layers - 1:
                out_channels = d_model
            else:
                out_channels = d_model * 2
            
            conv_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
        self.temporal_conv = nn.Sequential(*conv_layers)
        
        # 自适应池化到固定长度
        self.adaptive_pool = nn.AdaptiveAvgPool1d(n_token)
        
        # 最终投影层
        self.final_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )
        
        # Token投影到Cvae维度
        self.token_to_cvae = nn.Sequential(
            nn.Linear(d_model, Cvae),
            nn.LayerNorm(Cvae)
        )
        
        # 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, n_token, d_model))
        
        # 注意：这里不包含分类器，分类器在ClassificationModel中
        
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
        
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, x):
        """
        前向传播
        Args:
            x: (B, time_bins, input_dim)
        Returns:
            tokens: (B, n_token, Cvae)
        """
        B = x.shape[0]
        
        # 检查输入
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 1. 输入投影
        x = self.input_proj(x)  # (B, time_bins, d_model)
        
        # 2. 转换为卷积输入格式
        x = x.transpose(1, 2)  # (B, d_model, time_bins)
        
        # 3. 时间维度的1D卷积
        x = self.temporal_conv(x)  # (B, d_model, time_bins)
        
        # 4. 自适应池化到n_token长度
        x = self.adaptive_pool(x)  # (B, d_model, n_token)
        
        # 5. 转回 (B, n_token, d_model)
        x = x.transpose(1, 2)  # (B, n_token, d_model)
        
        # 6. 最终投影
        x = self.final_proj(x)  # (B, n_token, d_model)
        
        # 7. 添加位置编码
        x = x + self.pos_embed  # (B, n_token, d_model)
        
        # 8. 投影到Cvae维度，生成tokens
        tokens = self.token_to_cvae(x)  # (B, n_token, Cvae)
        
        return tokens

class ClassificationModel(nn.Module):
    """
    分类模型：使用TemporalEPEncoder作为特征提取器，添加分类头
    输入: (B, time_bins, neurons)
    输出: (B, num_classes) - 分类logits
    """
    def __init__(self, input_dim, time_bins, num_classes, d_model=32, n_token=128, 
                 num_conv_layers=2, dropout=0.2, hidden_dim=256):
        super().__init__()
        
        # 特征提取器（TemporalEPEncoder）
        self.encoder = TemporalEPEncoder(
            input_dim=input_dim,
            time_bins=time_bins,
            d_model=d_model,
            n_token=n_token,
            num_conv_layers=num_conv_layers,
            dropout=dropout,
            Cvae=d_model,
            num_classes=num_classes  # 虽然不使用，但保持接口一致
        )
        
        # 分类头：从token特征到类别
        # 使用token序列的平均池化作为全局特征
        self.classifier = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        """
        前向传播
        Args:
            x: (B, time_bins, input_dim) - PSTH数据
        Returns:
            logits: (B, num_classes) - 分类logits
        """
        # 获取encoder的tokens
        tokens = self.encoder(x)  # (B, n_token, Cvae)
        
        # 对token序列进行平均池化，得到全局特征
        # 注意：tokens的维度是(B, n_token, Cvae)，其中Cvae=d_model
        global_feature = tokens.mean(dim=1)  # (B, Cvae) = (B, d_model)
        
        # 分类
        logits = self.classifier(global_feature)  # (B, num_classes)
        
        return logits

class TrainingConfig:
    classification_batch_size = 32  # 分类训练时的batch_size（与notebook一致）
    num_workers = 4  # 数据加载的工作进程数（与notebook一致）
    
    lr = 1e-3  # 学习率（与notebook一致）
    weight_decay = 0.01  # 权重衰减（与notebook一致）
    betas = (0.9, 0.999)  # AdamW的betas参数（与notebook一致）
    
    classification_pretrain_epochs = 50  # 分类预训练的最大epoch数（与notebook一致）
    early_stopping_patience = 15  # 早停耐心值（验证损失不下降的epoch数）
    early_stopping_min_delta = 0.0001  # 验证损失改善的最小阈值
    use_early_stopping = False  # 是否使用早停（False表示固定训练50个epoch，与notebook一致）
    
    checkpoint_dir = "checkpoints"
    log_dir = "logs"
    base_results_dir = "/media/ubuntu/sda/mouse_test/script/end2end"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42
    
    def __init__(self):
        """
        初始化训练配置
        """
        # 设置结果目录
        self.results_dir = os.path.join(self.base_results_dir, "classification_train_results")

def train_classification_stage(config, model, train_dataset, val_dataset):
    """
    分类训练：训练TemporalEPEncoder的分类部分，使用早停机制
    """
    print(f"\n{'='*80}")
    print(f"分类训练（TemporalEPEncoder）")
    print(f"{'='*80}\n")
    
    os.makedirs(config.results_dir, exist_ok=True)
    checkpoint_dir = os.path.join(config.results_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 创建优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=config.betas
    )
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config.classification_pretrain_epochs,
        eta_min=config.lr * 0.01
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.classification_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.classification_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )

    # 早停相关变量
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    best_epoch = -1
    best_model_state = None
    
    criterion = torch.nn.CrossEntropyLoss()
    
    # 训练循环
    for epoch in range(config.classification_pretrain_epochs):
        model.train()
        
        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_samples = 0
        
        train_pbar = tqdm(
            enumerate(train_loader), 
            total=len(train_loader),
            desc=f"分类训练 Epoch {epoch+1}/{config.classification_pretrain_epochs} [Train]",
            ncols=100,
            leave=False
        )
        
        for i, batch in train_pbar:
            # 处理数据格式：可能是 (psth, labels) 或 (psth, images, labels)
            if len(batch) == 3:
                neuron_activity, images, labels = batch
            else:
                neuron_activity, labels = batch
            
            neuron_activity = neuron_activity.to(config.device, non_blocking=True)
            labels = labels.to(config.device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # 前向传播：只计算分类损失
            class_logits = model(neuron_activity)  # ClassificationModel直接返回logits
            loss = criterion(class_logits, labels)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 计算准确率
            with torch.no_grad():
                acc = (class_logits.argmax(dim=-1) == labels).float().mean().item() * 100
            
            batch_size = labels.size(0)
            epoch_loss += loss.item() * batch_size
            epoch_acc += acc * batch_size
            epoch_samples += batch_size
            
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{acc:.2f}%',
                'avg_loss': f'{epoch_loss/epoch_samples:.4f}',
                'avg_acc': f'{epoch_acc/epoch_samples:.2f}%'
            })
        
        avg_epoch_loss = epoch_loss / epoch_samples
        avg_epoch_acc = epoch_acc / epoch_samples
        
        scheduler.step()
        
        print(f"\n分类训练 Epoch {epoch}/{config.classification_pretrain_epochs}: "
              f"Train Loss = {avg_epoch_loss:.4f}, Train Acc = {avg_epoch_acc:.2f}%")
        
        # 验证
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_samples = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"分类训练 Epoch {epoch+1}/{config.classification_pretrain_epochs} [Val]", ncols=100, leave=False)
            for batch in val_pbar:
                # 处理数据格式：可能是 (psth, labels) 或 (psth, images, labels)
                if len(batch) == 3:
                    neuron_activity, images, labels = batch
                else:
                    neuron_activity, labels = batch
                
                neuron_activity = neuron_activity.to(config.device)
                labels = labels.to(config.device)
                
                class_logits = model(neuron_activity)  # ClassificationModel直接返回logits
                loss = criterion(class_logits, labels)
                acc = (class_logits.argmax(dim=-1) == labels).float().mean().item() * 100
                
                batch_size = labels.size(0)
                val_loss += loss.item() * batch_size
                val_acc += acc * batch_size
                val_samples += batch_size
                
                val_pbar.set_postfix({
                    'loss': f'{val_loss/val_samples:.4f}',
                    'acc': f'{val_acc/val_samples:.2f}%'
                })
        
        avg_val_loss = val_loss / val_samples
        avg_val_acc = val_acc / val_samples
        
        print(f"分类训练 Epoch {epoch}/{config.classification_pretrain_epochs}: "
              f"Val Loss = {avg_val_loss:.4f}, Val Acc = {avg_val_acc:.2f}%")
        
        # 早停检查和最佳模型保存（基于验证准确率，与notebook一致）
        improved = False
        # 使用验证准确率来选择最佳模型（与notebook一致）
        if avg_val_acc > best_val_acc:
            best_val_loss = avg_val_loss
            best_val_acc = avg_val_acc
            best_epoch = epoch
            patience_counter = 0
            improved = True
            
            # 在内存中保存最佳模型状态
            best_model_state = {
                'epoch': epoch,
                'model_state_dict': {k: v.cpu().clone() for k, v in model.state_dict().items()},
                'optimizer_state_dict': {k: v.cpu().clone() if isinstance(v, torch.Tensor) else v 
                                        for k, v in optimizer.state_dict().items()},
                'val_loss': avg_val_loss,
                'val_acc': avg_val_acc,
                'best_epoch': epoch
            }
            print(f"  -> 保存最佳分类模型 (Epoch {epoch}, Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.2f}%) [内存]")
        else:
            patience_counter += 1
        
        # 早停检查（如果启用）
        if config.use_early_stopping and patience_counter >= config.early_stopping_patience:
            print(f"\n早停触发：验证损失在 {config.early_stopping_patience} 个epoch内未改善")
            print(f"最佳模型在 Epoch {best_epoch}: Val Loss = {best_val_loss:.4f}, Val Acc = {best_val_acc:.2f}%")
            break
    
    # 加载最佳模型
    if best_model_state is not None:
        print(f"\n加载最佳分类模型 (Epoch {best_model_state['best_epoch']}, Val Loss: {best_model_state['val_loss']:.4f}, Val Acc: {best_model_state['val_acc']:.2f}%)")
        model.load_state_dict(best_model_state['model_state_dict'], strict=False)
        
        # 保存到磁盘
        best_model_path = os.path.join(checkpoint_dir, "best_classification_model.pth")
        torch.save(best_model_state, best_model_path)
        print(f"最佳模型已保存到磁盘: {best_model_path}")
    else:
        print(f"\n警告: 未找到最佳分类模型，使用当前模型")
    
    print(f"\n{'='*80}")
    print(f"分类训练完成")
    print(f"{'='*80}\n")
    
    return model

def train_with_psth_data(config=None):
    """
    使用前11个月的PSTH数据进行分类训练
    """
    print(f"\n{'='*80}")
    print(f"开始分类训练 - 使用前11个月的数据")
    print(f"{'='*80}\n")
    
    # train_MUA 和 trial_image_ids 已在全局作用域中加载
    
    # 根据图像ID创建类别标签
    unique_image_ids = sorted(list(set(trial_image_ids)))  # 117个唯一图像
    num_classes = len(unique_image_ids)  # 117个类别
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
    
    # 获取类别标签
    mapped_labels, label_mapping = create_label_mapping_from_classes(class_labels, unique_classes)
    
    print(f"标签映射完成，映射了 {len(label_mapping)} 个类别")
    print(f"标签范围: {mapped_labels.min()} - {mapped_labels.max()}")
    
    # 按照8/2比例划分训练集和验证集
    all_indices = np.arange(len(train_MUA))
    np.random.seed(42)
    np.random.shuffle(all_indices)
    
    train_indices, val_indices = train_test_split(
        all_indices, 
        test_size=0.2, 
        random_state=42,
        shuffle=True,
        stratify=class_labels
    )
    
    print(f"数据集划分:")
    print(f"  - 训练集（前11个月80%）: {len(train_indices)} trials ({len(train_indices)/len(all_indices)*100:.1f}%)")
    print(f"  - 验证集（前11个月20%）: {len(val_indices)} trials ({len(val_indices)/len(all_indices)*100:.1f}%)")
    print(f"  - 测试集（第12个月）: {len(val_psth_matrix)} trials")
    
    # 准备数据
    print("准备训练和验证数据...")
    train_psth_data = train_MUA[train_indices]
    val_psth_data = train_MUA[val_indices]
    
    # 为第12个月数据创建标签
    val_class_labels = np.array([image_id_to_class.get(img_id, -1) for img_id in val_trial_image_ids])
    
    # 检查是否有未知的图像ID
    unknown_labels = np.sum(val_class_labels == -1)
    if unknown_labels > 0:
        print(f"警告: 第12个月数据中有 {unknown_labels} 个图像ID未在前11个月数据中出现")
    
    # 为第12个月数据创建映射标签
    val_mapped_labels, _ = create_label_mapping_from_classes(val_class_labels, unique_classes)
    
    # 准备测试数据
    test_psth_data = val_psth_matrix
    
    # 创建数据集
    class PSTHDataset(Dataset):
        """
        PSTH数据集（仅用于分类训练，不包含图像）
        """
        def __init__(self, psth_data, labels):
            """
            Args:
                psth_data: (n_trials, time_bins, n_neurons) - PSTH矩阵
                labels: (n_trials,) - 类别标签
            """
            self.psth_data = torch.tensor(psth_data, dtype=torch.float32)
            self.labels = torch.tensor(labels, dtype=torch.long)
        
        def __len__(self):
            return len(self.psth_data)
        
        def __getitem__(self, idx):
            return self.psth_data[idx], self.labels[idx]
    
    print("创建训练数据集...")
    train_dataset = PSTHDataset(
        train_psth_data,
        mapped_labels[train_indices]
    )
    
    print("创建验证数据集...")
    val_dataset = PSTHDataset(
        val_psth_data,
        mapped_labels[val_indices]
    )
    
    print("创建测试数据集（第12个月）...")
    test_dataset = PSTHDataset(
        test_psth_data,
        val_mapped_labels
    )
    
    # 构建模型（使用ClassificationModel，与notebook一致）
    time_bins = train_MUA.shape[1]
    actual_input_dim = train_MUA.shape[2]
    
    print(f"\n模型配置 - 时间步数: {time_bins}, 神经元数量: {actual_input_dim}, 类别数量: {num_classes}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = ClassificationModel(
        input_dim=actual_input_dim,
        time_bins=time_bins,
        num_classes=num_classes,
        d_model=32,
        n_token=128,
        num_conv_layers=2,
        dropout=0.2,
        hidden_dim=256
    ).to(device)
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 使用传入的config或创建新的配置
    if config is None:
        config = TrainingConfig()
    
    # 开始训练
    model = train_classification_stage(config, model, train_dataset, val_dataset)
    
    # 训练完成后保存最终模型
    final_model_path = os.path.join(config.results_dir, "classification_model_final.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"\n{'='*80}")
    print(f"训练完成，最终模型已保存至: {final_model_path}")
    print(f"{'='*80}\n")
    
    # 在第12个月测试集上进行最终评估
    print(f"\n{'='*80}")
    print(f"在第12个月测试集上进行最终评估")
    print(f"{'='*80}\n")
    
    # 创建测试数据加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.classification_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    # 评估分类准确率
    model.eval()
    test_correct = 0
    test_total = 0
    test_class_loss = 0.0
    
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for neuron_activity, label in tqdm(test_loader, desc="测试集评估"):
            neuron_activity = neuron_activity.to(config.device)
            label = label.to(config.device)
            
            # 获取分类logits
            class_logits = model(neuron_activity)  # ClassificationModel直接返回logits
            
            # 计算分类损失
            class_loss = criterion(class_logits, label)
            test_class_loss += class_loss.item()
            
            # 计算准确率
            _, predicted = torch.max(class_logits.data, 1)
            test_total += label.size(0)
            test_correct += (predicted == label).sum().item()
    
    test_acc = 100.0 * test_correct / test_total
    test_avg_class_loss = test_class_loss / len(test_loader)
    
    print(f"\n第12个月测试集结果:")
    print(f"  - 分类准确率: {test_acc:.2f}%")
    print(f"  - 平均分类损失: {test_avg_class_loss:.4f}")
    
    # 清理显存
    del model, train_dataset, val_dataset, test_dataset
    torch.cuda.empty_cache()

if __name__ == "__main__":
    import sys
    
    # 创建训练配置
    config = TrainingConfig()
    
    train_with_psth_data(config)
