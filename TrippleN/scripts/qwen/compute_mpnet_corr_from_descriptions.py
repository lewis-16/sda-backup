import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import tqdm
from scipy.stats import pearsonr
from sentence_transformers import SentenceTransformer


GPU_BATCH_SIZE = 10
MIN_COMPONENTS = 5
MAX_COMPONENTS = 25


def load_neuron_responses(responses_path):
    neuron_responses = np.load(responses_path)
    return neuron_responses


def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()


def load_sentence_model():
    model = SentenceTransformer('/media/ubuntu/sda/TrippleN/model/all-mpnet-base-v2')
    model.eval()
    return model


def load_descriptions_jsonl(desc_path, limit):
    items = []
    with open(desc_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            items.append(json.loads(s))
    items = sorted(items, key=lambda x: int(x.get("index", 0)))
    if limit is not None:
        items = items[:limit]
    texts = []
    for it in items:
        raw = it.get("description", "")
        if raw is None:
            raw = ""
        if isinstance(raw, list):
            if not raw:
                txt = ""
            else:
                first = raw[0]
                if isinstance(first, dict) and "text" in first:
                    txt = str(first["text"])
                else:
                    txt = str(first)
        elif isinstance(raw, dict) and "text" in raw:
            txt = str(raw["text"])
        else:
            txt = str(raw)
        texts.append(txt)
    return texts


def extract_description_features(texts, model, batch_size):
    n = len(texts)
    feats = np.zeros((n, 768), dtype=np.float32)
    for i in range(0, n, batch_size):
        batch = texts[i:i + batch_size]
        emb = model.encode(batch, show_progress_bar=False)
        feats[i:i + len(batch)] = emb
    return feats


def compute_pls_predictions_all_components(X_train, Y_train, X_test, n_components_range):
    device = X_train.device
    n_train = X_train.shape[0]
    n_features = X_train.shape[1]
    n_neurons = Y_train.shape[1]
    n_test = X_test.shape[0]
    n_comps = len(n_components_range)
    max_comp = max(n_components_range)
    max_comp = min(max_comp, n_train - 1)

    W = torch.zeros(n_features, max_comp, device=device, dtype=X_train.dtype)
    P = torch.zeros(n_features, max_comp, device=device, dtype=X_train.dtype)
    Q = torch.zeros(n_neurons, max_comp, device=device, dtype=X_train.dtype)

    X_residual = X_train.clone()
    Y_residual = Y_train.clone()

    for a in range(max_comp):
        C_mat = X_residual.T @ Y_residual
        try:
            U, S, Vh = torch.linalg.svd(C_mat, full_matrices=False)
            w = U[:, 0:1]
        except Exception:
            w = C_mat[:, 0:1] / (torch.norm(C_mat[:, 0:1]) + 1e-8)

        W[:, a] = w.squeeze()
        t = X_residual @ w

        t_norm_sq = (t.T @ t).item() + 1e-8
        p = (X_residual.T @ t) / t_norm_sq
        P[:, a] = p.squeeze()
        q = (Y_residual.T @ t) / t_norm_sq
        Q[:, a] = q.squeeze()

        X_residual = X_residual - t @ p.T
        Y_residual = Y_residual - t @ q.T

    W_comp = W[:, :max_comp]
    P_comp = P[:, :max_comp]
    Q_comp = Q[:n_neurons, :max_comp]

    PWP_inv = torch.inverse(P_comp.T @ W_comp)
    B = W_comp @ PWP_inv @ Q_comp.T

    Y_train_pred_all = torch.zeros(n_comps, n_train, n_neurons, device=device, dtype=X_train.dtype)
    Y_test_pred_all = torch.zeros(n_comps, n_test, n_neurons, device=device, dtype=X_train.dtype)

    for i, n_comp in enumerate(n_components_range):
        if n_comp > max_comp:
            n_comp = max_comp

        W_i = W_comp[:, :n_comp]
        P_i = P_comp[:, :n_comp]
        Q_i = Q_comp[:, :n_comp]

        PWP_inv_i = torch.inverse(P_i.T @ W_i)
        B_i = W_i @ PWP_inv_i @ Q_i.T

        Y_train_pred_all[i] = X_train @ B_i
        Y_test_pred_all[i] = X_test @ B_i

    return Y_train_pred_all, Y_test_pred_all


def evaluate_encoding_performance_gpu(features, neuron_responses, unit_info, seed):
    n_neurons_total = neuron_responses.shape[0]
    n_images = neuron_responses.shape[1]
    reliability_best = unit_info['reliability_best'].values

    n_component_candidates = list(range(MIN_COMPONENTS, MAX_COMPONENTS + 1))

    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_images)
    n_train = int(round(n_images * 0.9))
    n_train = max(n_train, MIN_COMPONENTS + 1)
    n_train = min(n_train, n_images - 1)
    train_img_idx = indices[:n_train]
    test_img_idx = indices[n_train:]
    if len(test_img_idx) == 0:
        test_img_idx = indices[-1:]
        train_img_idx = indices[:-1]

    X_train = features[train_img_idx]
    X_test = features[test_img_idx]

    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0) + 1e-8
    X_train_scaled = (X_train - X_mean) / X_std
    X_test_scaled = (X_test - X_mean) / X_std

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train_gpu = torch.from_numpy(X_train_scaled).float().to(device)
    X_test_gpu = torch.from_numpy(X_test_scaled).float().to(device)

    batch_size = GPU_BATCH_SIZE
    n_batches = int(np.ceil(n_neurons_total / batch_size))

    encoding_correlations = np.zeros(n_neurons_total)
    normalized_correlations = np.zeros(n_neurons_total)
    best_components = np.zeros(n_neurons_total, dtype=int)

    pbar = tqdm.tqdm(total=n_neurons_total, desc="sentence encoding", ncols=80, unit_scale=True)

    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, n_neurons_total)
        n_neurons_in_batch = end_idx - start_idx

        Y_batch = neuron_responses[start_idx:end_idx, :]
        Y_train = Y_batch[:, train_img_idx]
        Y_test = Y_batch[:, test_img_idx]

        Y_train_swapped = Y_train.T
        Y_test_swapped = Y_test.T

        Y_train_gpu = torch.from_numpy(Y_train_swapped).float().to(device)
        Y_test_gpu = torch.from_numpy(Y_test_swapped).float().to(device)

        n_component_candidates_eff = [c for c in n_component_candidates if c < X_train_gpu.shape[0]]
        if len(n_component_candidates_eff) == 0:
            n_component_candidates_eff = [min(2, X_train_gpu.shape[0] - 1)]

        Y_train_pred_all, Y_test_pred_all = compute_pls_predictions_all_components(
            X_train_gpu, Y_train_gpu, X_test_gpu, n_component_candidates_eff
        )

        Y_train_pred_np = Y_train_pred_all.cpu().numpy()
        Y_test_pred_np = Y_test_pred_all.cpu().numpy()

        del Y_train_pred_all, Y_test_pred_all, Y_train_gpu, Y_test_gpu
        clear_gpu_memory()

        for local_idx in range(n_neurons_in_batch):
            global_neuron_idx = start_idx + local_idx

            mse_for_comp = np.zeros(len(n_component_candidates_eff))
            for comp_idx in range(len(n_component_candidates_eff)):
                mse_for_comp[comp_idx] = np.mean(
                    (Y_train_swapped[:, local_idx] - Y_train_pred_np[comp_idx, :, local_idx]) ** 2
                )

            best_comp_idx = int(np.argmin(mse_for_comp))
            best_components[global_neuron_idx] = n_component_candidates_eff[best_comp_idx]

            y_test_true = Y_test_swapped[:, local_idx]
            y_test_pred = Y_test_pred_np[best_comp_idx, :, local_idx]

            if np.std(y_test_true) > 0 and np.std(y_test_pred) > 0:
                corr, _ = pearsonr(y_test_true, y_test_pred)
                encoding_correlations[global_neuron_idx] = corr
            else:
                encoding_correlations[global_neuron_idx] = 0.0

            pbar.update(1)

    pbar.close()

    for i in range(n_neurons_total):
        reliability = reliability_best[i]
        if reliability > 0:
            normalized_correlations[i] = encoding_correlations[i] / reliability
        else:
            normalized_correlations[i] = 0.0

    return encoding_correlations, normalized_correlations, best_components


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--descriptions", default="/media/ubuntu/sda/TrippleN/scripts/qwen/local_cache/descriptions_spatial_layout.jsonl")
    ap.add_argument("--neuron-responses", default="/media/ubuntu/sda/TrippleN/customize/neuron_responses_1000.npy")
    ap.add_argument("--unit-info", default="/media/ubuntu/sda/TrippleN/customize/aggregate_response/all_subjects_unit_info.pkl")
    ap.add_argument("--limit", type=int, default=666)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    texts = load_descriptions_jsonl(args.descriptions, args.limit)
    if len(texts) == 0:
        raise SystemExit("No descriptions loaded.")

    neuron_responses = load_neuron_responses(args.neuron_responses)
    unit_info = pd.read_pickle(args.unit_info)

    n_images = min(neuron_responses.shape[1], len(texts))
    neuron_responses = neuron_responses[:, :n_images]
    texts = texts[:n_images]

    model = load_sentence_model()
    features = extract_description_features(texts, model, args.batch_size)

    corr, norm_corr, comps = evaluate_encoding_performance_gpu(features, neuron_responses, unit_info, args.seed)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    results = pd.DataFrame(
        {
            "encoding_correlation": corr,
            "normalized_correlation": norm_corr,
            "best_n_components": comps,
            "reliability_best": unit_info["reliability_best"].values,
            "best_r_time1": unit_info["best_r_time1"].values,
            "best_r_time2": unit_info["best_r_time2"].values,
            "subject": unit_info["subject"].values if "subject" in unit_info.columns else None,
        }
    )
    results.to_pickle(args.out)


if __name__ == "__main__":
    main()

