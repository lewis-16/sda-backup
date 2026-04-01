"""
整合多个 clique 的响应矩阵，使用与 post_sort_260122 相同的 7 类分类模型训练并评估效果。
"""
import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

MOUNTAINSORT_DIR = "/media/ubuntu/sda/visual_generation/results/neuroscroll_260122/mountainsort"
CLIQUE_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8]
REC_PARAMS_PATH = "/media/ubuntu/sda/duan/result/260121/rec_params.csv"
CONDITION_PATH = "/media/ubuntu/sda/duan/result/260121/images_sequence_10000.csv"
PKL_CLUSTER_PATH = "/media/ubuntu/sda/Monkey/semantic/epoch_10_monkeyN/merged_cluster_class_counts.pkl"
OUTPUT_DIR = os.path.join(MOUNTAINSORT_DIR, "clique_classify_4cliques")
os.makedirs(OUTPUT_DIR, exist_ok=True)

USE_IMAGE_AVERAGE = False


def load_and_merge_response_matrices(clique_ids, mountainsort_dir):
    response_list = []
    neuron_ids_all = []
    for cid in clique_ids:
        pkl_path = os.path.join(mountainsort_dir, f"clique_{cid}", "response_matrix_4000trials_30bins.pkl")
        if not os.path.exists(pkl_path):
            print(f"   跳过 clique_{cid}: 未找到 {pkl_path}")
            continue
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        R = data["response_matrix"]
        nids = data["neuron_ids"]
        response_list.append(R)
        neuron_ids_all.extend([(cid, nid) for nid in nids])
    if not response_list:
        raise FileNotFoundError("未找到任何 clique 的 response_matrix_4000trials_30bins.pkl")
    combined = np.concatenate(response_list, axis=0)
    return combined, neuron_ids_all


def get_trial_image_names(n_trials=4000):
    rec_params = pd.read_csv(REC_PARAMS_PATH)
    rec_params = rec_params[rec_params["bhv_codes"] == 10].reset_index(drop=True)
    n_rec = len(rec_params)
    condition = pd.read_csv(CONDITION_PATH)
    n_cond = len(condition)
    if n_cond < n_rec:
        condition = pd.concat([condition] * (n_rec // n_cond + 1), ignore_index=True)
    condition = condition.iloc[:n_rec].reset_index(drop=True)
    rec_params = pd.concat([rec_params, condition], axis=1)
    rec_params = rec_params[(rec_params["trial_ids"] >= 300) & (rec_params["trial_ids"] < 5300)].reset_index(drop=True)
    rec_params = rec_params.head(n_trials)
    image_col = "image_name" if "image_name" in rec_params.columns else [c for c in rec_params.columns if "image" in c.lower()][0]
    return rec_params[image_col].astype(str).values


def build_neuron_image_response(response_combined, stimulus_ids, stimulus_unique):
    n_neurons, n_trials, n_bins = response_combined.shape
    n_images = len(stimulus_unique)
    image_to_index = {s: i for i, s in enumerate(stimulus_unique)}
    neuron_image_response = np.zeros((n_neurons, n_images, n_bins), dtype=np.float32)
    counts = np.zeros((n_neurons, n_images), dtype=np.float32)
    for trial in range(n_trials):
        img = stimulus_ids[trial]
        if img not in image_to_index:
            continue
        idx = image_to_index[img]
        neuron_image_response[:, idx, :] += response_combined[:, trial, :]
        counts[:, idx] += 1
    counts = np.maximum(counts, 1e-8)
    neuron_image_response /= counts[:, :, np.newaxis]
    return neuron_image_response


class TemporalEPEncoder(nn.Module):
    def __init__(self, input_dim=244, time_bins=30, d_model=256, n_token=128,
                 num_conv_layers=3, dropout=0.2, output_dim=768):
        super().__init__()
        self.input_dim = input_dim
        self.time_bins = time_bins
        self.d_model = d_model
        self.n_token = n_token
        self.output_dim = output_dim
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
        conv_layers = []
        for i in range(num_conv_layers):
            in_ch = d_model if i == 0 else d_model * 2
            out_ch = d_model if i == num_conv_layers - 1 else d_model * 2
            conv_layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        self.temporal_conv = nn.Sequential(*conv_layers)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(n_token)
        self.final_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        self.feature_proj = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, output_dim),
            nn.LayerNorm(output_dim),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        x = self.input_proj(x)
        x = x.transpose(1, 2)
        x = self.temporal_conv(x)
        x = self.adaptive_pool(x)
        x = x.transpose(1, 2)
        x = self.final_proj(x)
        x = self.feature_proj(x)
        return x.mean(dim=1)


class PSTHClusterClassifier(nn.Module):
    def __init__(self, encoder, num_classes=7):
        super().__init__()
        self.encoder = encoder
        feat_dim = getattr(encoder, "output_dim", 768)
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x, return_features=False):
        feat = self.encoder(x)
        logits = self.classifier(feat)
        if return_features:
            return logits, feat
        return logits


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07, num_classes=7):
        super().__init__()
        self.temperature = temperature
        self.num_classes = num_classes

    def forward(self, features, labels):
        batch_size = features.size(0)
        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        labels_expanded = labels.unsqueeze(1)
        positive_mask = (labels_expanded == labels_expanded.T).float()
        positive_mask.fill_diagonal_(0)
        num_positives = positive_mask.sum(dim=1, keepdim=True)
        num_positives = torch.clamp(num_positives, min=1)
        exp_sim = torch.exp(similarity_matrix)
        pos_exp_sum = (exp_sim * positive_mask).sum(dim=1, keepdim=True)
        all_exp_sum = exp_sim.sum(dim=1, keepdim=True)
        loss_per_sample = -torch.log((pos_exp_sum / num_positives) / (all_exp_sum + 1e-8) + 1e-8)
        return loss_per_sample.mean()


class PSTHClusterDataset(Dataset):
    def __init__(self, psth_data, cluster_labels):
        self.psth_data = torch.tensor(psth_data, dtype=torch.float32)
        self.cluster_labels = np.array(cluster_labels, dtype=np.int64)

    def __len__(self):
        return len(self.psth_data)

    def __getitem__(self, idx):
        psth = self.psth_data[idx].transpose(0, 1)
        return psth, self.cluster_labels[idx]


def main():
    print("1. 加载 4 个 clique 的响应矩阵并合并...")
    response_combined, neuron_ids_all = load_and_merge_response_matrices(CLIQUE_IDS, MOUNTAINSORT_DIR)
    n_neurons_total, n_trials, n_bins = response_combined.shape
    print(f"   合并后形状: (n_neurons={n_neurons_total}, n_trials={n_trials}, n_bins={n_bins})")

    print("2. 获取 4000 trials 的 image_name...")
    stimulus_ids = get_trial_image_names(n_trials)
    stimulus_unique = pd.unique(stimulus_ids)
    print(f"   trial 数: {len(stimulus_ids)}, 唯一 image 数: {len(stimulus_unique)}")

    print("3. 加载 7 类 cluster 与 image 路径，构建有效样本...")
    with open(PKL_CLUSTER_PATH, "rb") as f:
        cluster_data = pickle.load(f)
    class_to_cluster = {}
    for cluster_id in range(7):
        for class_name in cluster_data[cluster_id]:
            class_to_cluster[class_name] = cluster_id

    condition_df = pd.read_csv(CONDITION_PATH)
    image_path_dict = {row["image_name"]: row["image_path"] for _, row in condition_df.iterrows()}

    if USE_IMAGE_AVERAGE:
        neuron_image_response_matrix = build_neuron_image_response(
            response_combined, stimulus_ids, stimulus_unique
        )
        neuron_image_response_matrix = neuron_image_response_matrix[:, :, 4:]
        valid_psth = []
        valid_labels = []
        for stim_idx, img_name in enumerate(stimulus_unique):
            if img_name not in image_path_dict:
                continue
            path = image_path_dict[img_name]
            class_name = os.path.basename(os.path.dirname(path))
            if class_name not in class_to_cluster:
                continue
            psth_per_image = neuron_image_response_matrix[:, stim_idx, :]
            if psth_per_image.sum() <= 0:
                continue
            valid_psth.append(psth_per_image)
            valid_labels.append(class_to_cluster[class_name])
        valid_psth = np.array(valid_psth)
        valid_labels = np.array(valid_labels)
        print(f"   按 image 平均: {len(valid_psth)} 个样本, 各类数量: {np.bincount(valid_labels, minlength=7)}")
    else:
        valid_psth = []
        valid_labels = []
        for trial in range(n_trials):
            img_name = stimulus_ids[trial]
            if img_name not in image_path_dict:
                continue
            path = image_path_dict[img_name]
            class_name = os.path.basename(os.path.dirname(path))
            if class_name not in class_to_cluster:
                continue
            psth_trial = response_combined[:, trial, 4:]
            if psth_trial.sum() <= 0:
                continue
            valid_psth.append(psth_trial)
            valid_labels.append(class_to_cluster[class_name])
        valid_psth = np.array(valid_psth)
        valid_labels = np.array(valid_labels)
        print(f"   按 trial(不平均): {len(valid_psth)} 个样本, 各类数量: {np.bincount(valid_labels, minlength=7)}")

    train_indices, val_indices = train_test_split(
        range(len(valid_psth)), test_size=0.1, random_state=42, stratify=valid_labels
    )
    train_psth = valid_psth[train_indices]
    train_labels = valid_labels[train_indices]
    val_psth = valid_psth[val_indices]
    val_labels = valid_labels[val_indices]
    print(f"   训练集: {len(train_psth)}, 验证集: {len(val_psth)}")

    train_dataset = PSTHClusterDataset(train_psth, train_labels)
    val_dataset = PSTHClusterDataset(val_psth, val_labels)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    n_neurons = train_psth.shape[1]
    n_time_bins = train_psth.shape[2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("4. 构建与 post_sort 相同的模型并训练...")
    encoder_backbone = TemporalEPEncoder(
        input_dim=n_neurons,
        time_bins=n_time_bins,
        d_model=64,
        n_token=128,
        num_conv_layers=2,
        dropout=0.2,
        output_dim=768,
    )
    model = PSTHClusterClassifier(encoder_backbone, num_classes=7).to(device)
    ce_criterion = nn.CrossEntropyLoss()
    infonce_criterion = InfoNCELoss(temperature=0.07, num_classes=7)
    alpha = 0.5
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    num_epochs = 10
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for psth_batch, label_batch in train_loader:
            psth_batch = psth_batch.to(device)
            label_batch = label_batch.to(device)
            optimizer.zero_grad()
            logits, features = model(psth_batch, return_features=True)
            ce_loss = ce_criterion(logits, label_batch)
            infonce_loss = infonce_criterion(features, label_batch)
            loss = alpha * ce_loss + (1 - alpha) * infonce_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * psth_batch.size(0)
            pred = logits.argmax(dim=1)
            train_correct += (pred == label_batch).sum().item()
            train_total += psth_batch.size(0)
        avg_train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        scheduler.step()

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for psth_batch, label_batch in val_loader:
                psth_batch = psth_batch.to(device)
                label_batch = label_batch.to(device)
                logits, features = model(psth_batch, return_features=True)
                ce_loss = ce_criterion(logits, label_batch)
                infonce_loss = infonce_criterion(features, label_batch)
                loss = alpha * ce_loss + (1 - alpha) * infonce_loss
                val_loss += loss.item() * psth_batch.size(0)
                pred = logits.argmax(dim=1)
                val_correct += (pred == label_batch).sum().item()
                val_total += label_batch.size(0)
        avg_val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        print(f"   Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.4f}, Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "psth_cluster_classifier_best.pth"))
            print(f"   保存最佳模型，验证准确率: {best_val_acc:.4f}")

    print("\n训练完成！")
    print(f"最佳验证准确率: {best_val_acc:.4f}")
    print(f"模型与日志保存至: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
