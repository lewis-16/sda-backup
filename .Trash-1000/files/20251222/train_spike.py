# train_spike_cls.py
import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# ---------- 1. 超参数 ----------
DATA_DIR   = './data'          # 改成你的目录
SAVE_DIR   = './ckpt'
BATCH_SIZE = 256
EPOCHS     = 60
LR         = 1e-3
K          = None              # 自动读取最大类别号
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'

os.makedirs(SAVE_DIR, exist_ok=True)

# ---------- 2. 数据集 ----------
class SpikeSet(Dataset):
    def __init__(self, x_npy, y_npy):
        self.x = torch.from_numpy(np.load(x_npy).astype(np.float32))
        self.y = torch.from_numpy(np.load(y_npy).astype(np.int64))
    def __len__(self):
        return self.x.shape[0]
    def __getitem__(self, idx):
        return self.x[idx].unsqueeze(0), self.y[idx]   # 加通道维 1×30×30

train_loader = DataLoader(SpikeSet(os.path.join(DATA_DIR, 'x_train.npy'),
                                   os.path.join(DATA_DIR, 'y_train.npy')),
                          batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader   = DataLoader(SpikeSet(os.path.join(DATA_DIR, 'x_val.npy'),
                                   os.path.join(DATA_DIR, 'y_val.npy')),
                          batch_size=BATCH_SIZE, shuffle=False)

# 自动获取类别数
K = max(train_loader.dataset.y.max().item(),
        val_loader.dataset.y.max().item())
NUM_CLASSES = K + 1   # 含 noise 0

# ---------- 3. 模型 ----------
class SpatialAttn(nn.Module):
    """通道-时间外积注意力"""
    def __init__(self, ch):
        super().__init__()
        self.t_pool = nn.AvgPool2d((1, 30))          # 时间轴池化
        self.c_pool = nn.AvgPool2d((30, 1))          # 通道轴池化
        self.compress_t = nn.Conv1d(ch, 4, 1, bias=False)
        self.compress_c = nn.Conv1d(ch, 4, 1, bias=False)
        self.expand   = nn.Conv2d(4, ch, 1, bias=False)

    def forward(self, x):          # x: (B,ch,30,30)
        B, C, H, W = x.shape
        # 时间平均分支
        t_feat = self.t_pool(x)               # (B,C,30,1)
        t_feat = self.compress_t(t_feat.squeeze(-1)).unsqueeze(-1)  # (B,4,30,1)
        # 通道平均分支
        c_feat = self.c_pool(x)               # (B,C,1,30)
        c_feat = self.compress_c(c_feat.squeeze(2)).unsqueeze(2)   # (B,4,1,30)
        # 外积得 mask
        mask = torch.sigmoid(self.expand(t_feat * c_feat))  # (B,4,30,30) -> (B,C,30,30)
        return x * mask


class SpikeCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),  # 1->16
            nn.BatchNorm2d(16),
            SpatialAttn(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            SpatialAttn(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            SpatialAttn(64),
            nn.ReLU(inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)   # (64,30,30)->(64,1,1)
        self.fc  = nn.Sequential(
            nn.Flatten(),                   # 64
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        feat = self.gap(self.cnn(x)).flatten(1)
        return self.fc(feat)

model = SpikeCNN(NUM_CLASSES).to(DEVICE)

# ---------- 4. 损失与优化 ----------
# 类别不平衡 → focal loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma
    def forward(self, logits, target):
        ce_loss = nn.functional.cross_entropy(logits, target, reduction='none')
        p_t = torch.exp(-ce_loss)
        return (self.alpha * (1 - p_t) ** self.gamma * ce_loss).mean()

criterion = FocalLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
scaler    = GradScaler()

# ---------- 5. 训练循环 ----------
def run_one_epoch(loader, training=False):
    if training:
        model.train()
    else:
        model.eval()
    total_loss, total_correct, total_samples = 0., 0, 0
    with torch.set_grad_enabled(training):
        for x, y in tqdm(loader, leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)
            with autocast():
                out = model(x)
                loss = criterion(out, y)
            if training:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item() * x.size(0)
            total_correct += (out.argmax(1) == y).sum().item()
            total_samples += x.size(0)
    return total_loss / total_samples, total_correct / total_samples

best_acc = 0.
for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = run_one_epoch(train_loader, training=True)
    val_loss, val_acc     = run_one_epoch(val_loader, training=False)
    scheduler.step()
    print(f'Epoch {epoch:02d} | train loss {train_loss:.4f} acc {train_acc:.4f} '
          f'| val loss {val_loss:.4f} acc {val_acc:.4f}')
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'best_spike_cls.pth'))
print('训练完成，模型已存于', SAVE_DIR)