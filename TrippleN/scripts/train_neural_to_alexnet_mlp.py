#!/usr/bin/env python3
"""
训练 MLP：神经元活动 -> AlexNet fc6 embedding
输入: 神经元响应 (n_samples, n_neurons)
输出: AlexNet fc6 特征 (n_samples, 4096)
损失: MSE
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime

NEURON_RESPONSES_PATH = '/media/ubuntu/sda/TrippleN/customize/neuron_responses_1000.npy'
FC6_CACHE_PATH = '/media/ubuntu/sda/TrippleN/customize/decoding_analysis/alexnet_fc6_features_1000.npy'
OUTPUT_DIR = '/media/ubuntu/sda/TrippleN/customize/neural_to_alexnet_mlp'
FC6_DIM = 4096
DEFAULT_HIDDEN = [2048, 2048, 4096]
DEFAULT_EPOCHS = 200
DEFAULT_BATCH = 64
DEFAULT_LR = 1e-3
DEFAULT_VAL_RATIO = 0.15
DEFAULT_SEED = 42


class NeuralToAlexNetMLP(nn.Module):
    def __init__(self, n_neurons, hidden_dims, out_dim=FC6_DIM, dropout=0.0):
        super().__init__()
        self.dropout = dropout
        dims = [n_neurons] + list(hidden_dims) + [out_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.BatchNorm1d(dims[i + 1]))
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


def load_data(neural_path, fc6_path):
    neural = np.load(neural_path).astype(np.float32)
    if neural.shape[1] == 1000 and neural.shape[0] != 1000:
        neural = neural.T
    if neural.shape[0] != 1000:
        raise ValueError('neural 样本数应为 1000, 得到 %d (形状 %s)' % (neural.shape[0], neural.shape))
    fc6 = np.load(fc6_path).astype(np.float32)
    if fc6.shape[0] != 1000 or fc6.shape[1] != FC6_DIM:
        raise ValueError('fc6 形状应为 (1000, %d), 得到 %s' % (FC6_DIM, fc6.shape))
    return neural, fc6


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / max(n, 1)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
    return total_loss / max(n, 1)


def main():
    parser = argparse.ArgumentParser(description='Train MLP: neural -> AlexNet fc6')
    parser.add_argument('--neural', type=str, default=NEURON_RESPONSES_PATH, help='neuron responses .npy')
    parser.add_argument('--fc6', type=str, default=FC6_CACHE_PATH, help='alexnet fc6 .npy')
    parser.add_argument('--out_dir', type=str, default=OUTPUT_DIR, help='output dir')
    parser.add_argument('--hidden', type=int, nargs='+', default=DEFAULT_HIDDEN, help='hidden layer sizes')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    parser.add_argument('--batch', type=int, default=DEFAULT_BATCH)
    parser.add_argument('--lr', type=float, default=DEFAULT_LR)
    parser.add_argument('--val_ratio', type=float, default=DEFAULT_VAL_RATIO)
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout after hidden layers')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--early_stop', type=int, default=40, help='early stop if val no improve for N epochs (0=off)')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--no_cuda', action='store_true', help='disable CUDA')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)

    neural, fc6 = load_data(args.neural, args.fc6)
    n_neurons = neural.shape[1]
    neural_mean = neural.mean(axis=0, keepdims=True)
    neural_std = neural.std(axis=0, keepdims=True) + 1e-8
    neural = (neural - neural_mean) / neural_std
    fc6_mean = fc6.mean(axis=0, keepdims=True)
    fc6_std = fc6.std(axis=0, keepdims=True) + 1e-8
    fc6 = (fc6 - fc6_mean) / fc6_std

    mse_predict_mean = float(np.mean(fc6 ** 2))
    print('Baseline MSE (predict mean): %.4f' % mse_predict_mean)

    n_val = int(1000 * args.val_ratio)
    n_train = 1000 - n_val
    perm = np.random.permutation(1000)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]
    X_train = torch.from_numpy(neural[train_idx])
    y_train = torch.from_numpy(fc6[train_idx])
    X_val = torch.from_numpy(neural[val_idx])
    y_val = torch.from_numpy(fc6[val_idx])

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0, pin_memory=(device.type == 'cuda'))
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False)

    model = NeuralToAlexNetMLP(n_neurons, args.hidden, dropout=args.dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)

    train_losses = []
    val_losses = []
    best_val = float('inf')
    best_epoch = -1
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        tl = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl = evaluate(model, val_loader, criterion, device)
        train_losses.append(tl)
        val_losses.append(vl)
        scheduler.step(vl)
        if vl < best_val:
            best_val = vl
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_loss': tl,
                'val_loss': vl,
                'n_neurons': n_neurons,
                'hidden_dims': args.hidden,
                'dropout': args.dropout,
                'neural_mean': neural_mean,
                'neural_std': neural_std,
                'fc6_mean': fc6_mean,
                'fc6_std': fc6_std,
            }, os.path.join(args.out_dir, 'best_model.pt'))
        else:
            epochs_no_improve += 1
        if epoch % 20 == 0 or epoch == 1:
            print('Epoch %d  train_mse=%.6f  val_mse=%.6f' % (epoch, tl, vl))
        if args.early_stop > 0 and epochs_no_improve >= args.early_stop:
            print('Early stopping at epoch %d (val no improve %d epochs)' % (epoch, args.early_stop))
            break

    np.savez(os.path.join(args.out_dir, 'loss_curves.npz'), train_loss=train_losses, val_loss=val_losses)
    print('Best val MSE %.6f at epoch %d. Saved to %s' % (best_val, best_epoch, args.out_dir))

    ckpt = torch.load(os.path.join(args.out_dir, 'best_model.pt'), map_location=device, weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    preds = []
    with torch.no_grad():
        for x, _ in val_loader:
            x = x.to(device)
            out = model(x)
            preds.append(out.cpu().numpy())
    pred_val_normalized = np.concatenate(preds, axis=0).astype(np.float32)
    fc6_mean = ckpt['fc6_mean']
    fc6_std = ckpt['fc6_std']
    pred_val_fc6 = pred_val_normalized * fc6_std + fc6_mean
    pred_val_path = os.path.join(args.out_dir, 'pred_val_fc6.npy')
    np.save(pred_val_path, pred_val_fc6)
    np.save(os.path.join(args.out_dir, 'val_indices.npy'), val_idx)
    print('Val set 预测 fc6 已保存: %s (shape %s), val 样本索引: val_indices.npy' % (pred_val_path, pred_val_fc6.shape))
    return model, train_losses, val_losses


if __name__ == '__main__':
    main()
