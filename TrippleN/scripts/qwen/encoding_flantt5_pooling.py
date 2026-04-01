import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
import torch
import tqdm
from scipy.stats import pearsonr
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


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


def load_coco_captions(caption_path, limit):
    with open(caption_path, "rb") as f:
        captions_matrix = pickle.load(f)
    if limit is not None:
        captions_matrix = captions_matrix[:limit]
    return captions_matrix


def to_text_list(x):
    out = []
    for v in x:
        if isinstance(v, (bytes, bytearray)):
            out.append(v.decode("utf-8", errors="ignore"))
        else:
            out.append(str(v))
    return out


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1e-6)
    return summed / denom


def extract_flantt5_sentence_embeddings(texts, tokenizer, model, device, batch_size):
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        toks = tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        toks = {k: v.to(device) for k, v in toks.items()}
        with torch.no_grad():
            out = model.get_encoder()(
                input_ids=toks["input_ids"],
                attention_mask=toks["attention_mask"],
                return_dict=True,
            )
            pooled = mean_pool(out.last_hidden_state, toks["attention_mask"])
        embs.append(pooled.detach().cpu().float().numpy())
    return np.concatenate(embs, axis=0)


def extract_features_from_coco_captions(captions_matrix, tokenizer, model, device, batch_size):
    n_images = captions_matrix.shape[0]
    feats = None
    pbar = tqdm.tqdm(total=n_images, desc="flan-t5 captions", ncols=80)
    for i in range(0, n_images, batch_size):
        end_idx = min(i + batch_size, n_images)
        batch_texts = []
        for j in range(i, end_idx):
            batch_texts.extend(to_text_list(captions_matrix[j].tolist()))
        emb = extract_flantt5_sentence_embeddings(batch_texts, tokenizer, model, device, batch_size=min(64, len(batch_texts)))
        d = emb.shape[1]
        if feats is None:
            feats = np.zeros((n_images, d), dtype=np.float32)
        for j in range(i, end_idx):
            s = (j - i) * 5
            e = s + 5
            feats[j] = np.mean(emb[s:e], axis=0)
        pbar.update(end_idx - i)
    pbar.close()
    return feats


def extract_features_from_descriptions(texts, tokenizer, model, device, batch_size):
    emb = extract_flantt5_sentence_embeddings(texts, tokenizer, model, device, batch_size)
    return emb.astype(np.float32)


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
    reliability_best = unit_info["reliability_best"].values

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

    device = 'cpu'
    X_train_gpu = torch.from_numpy(X_train_scaled).float().to(device)
    X_test_gpu = torch.from_numpy(X_test_scaled).float().to(device)

    batch_size = GPU_BATCH_SIZE
    n_batches = int(np.ceil(n_neurons_total / batch_size))

    encoding_correlations = np.zeros(n_neurons_total)
    normalized_correlations = np.zeros(n_neurons_total)
    best_components = np.zeros(n_neurons_total, dtype=int)

    pbar = tqdm.tqdm(total=n_neurons_total, desc="flan-t5 encoding", ncols=80, unit_scale=True)

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
    ap.add_argument("--model-dir", default="/media/ubuntu/sda/TrippleN/model/flantt5")
    ap.add_argument("--text-source", choices=["coco_captions", "descriptions_jsonl"], default="coco_captions")
    ap.add_argument("--captions", default="/media/ubuntu/sda/TrippleN/customize/coco_captions_1000x5.pkl")
    ap.add_argument("--descriptions", default="/media/ubuntu/sda/TrippleN/scripts/qwen/local_cache/descriptions_spatial_layout.jsonl")
    ap.add_argument("--neuron-responses", default="/media/ubuntu/sda/TrippleN/customize/neuron_responses_1000.npy")
    ap.add_argument("--unit-info", default="/media/ubuntu/sda/TrippleN/customize/aggregate_response/all_subjects_unit_info.pkl")
    ap.add_argument("--limit", type=int, default=666)
    ap.add_argument("--embed-batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)
    model.eval()
    model.to(device)

    if args.text_source == "coco_captions":
        captions_matrix = load_coco_captions(args.captions, args.limit)
        features = extract_features_from_coco_captions(
            captions_matrix, tokenizer, model, device, batch_size=args.embed_batch_size
        )
        n_images = captions_matrix.shape[0]
    else:
        texts = load_descriptions_jsonl(args.descriptions, args.limit)
        features = extract_features_from_descriptions(texts, tokenizer, model, device, batch_size=args.embed_batch_size)
        n_images = len(texts)

    neuron_responses = load_neuron_responses(args.neuron_responses)
    unit_info = pd.read_pickle(args.unit_info)

    n_images = min(n_images, neuron_responses.shape[1], features.shape[0])
    features = features[:n_images]
    neuron_responses = neuron_responses[:, :n_images]

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

