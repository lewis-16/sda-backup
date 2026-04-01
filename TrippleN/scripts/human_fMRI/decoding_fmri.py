#!/usr/bin/env python3

import os
import sys
import pickle
import numpy as np
from datetime import datetime
from scipy.spatial.distance import cdist
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import predict_neuron_activity_gpu as pnag

FMRI_DATA_DIR = '/media/ubuntu/sda/TrippleN/human_fMRI'
OUT_DIR = '/media/ubuntu/sda/TrippleN/customize/human_fMRI/decoding'
STIMULI_PATH = '/media/ubuntu/sda/TrippleN/stimuli'
CAPTIONS_PATH = '/media/ubuntu/sda/TrippleN/customize/coco_captions_1000x5.pkl'
N_IMAGES = 1000
RESPONSE_DIM = 500
MODEL_DIM = 100
RIDGE_ALPHA = 1.0

REGIONS = [
    ('early', 'resp_early_func1py8mm.npy'),
    ('midventral', 'resp_midventral_func1py8mm.npy'),
    ('midlateral', 'resp_midlateral_func1py8mm.npy'),
    ('midparietal', 'resp_midparietal_func1py8mm.npy'),
    ('ventral', 'resp_ventral_func1py8mm.npy'),
    ('lateral', 'resp_lateral_func1py8mm.npy'),
    ('parietal', 'resp_parietal_func1py8mm.npy'),
]


def load_region_neural_data(region_name, filename):
    path = os.path.join(FMRI_DATA_DIR, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"fMRI 数据不存在: {path}")
    arr = np.load(path).astype(np.float32)
    if arr.shape[1] != N_IMAGES:
        raise ValueError(f"区域 {region_name} 期望最后一维为 {N_IMAGES}, 得到 {arr.shape}")
    return arr.T


def reduce_response_dim(neural_responses, n_components=RESPONSE_DIM):
    n_avail = neural_responses.shape[1]
    n_comp = min(n_components, n_avail - 1)
    if n_comp < 1:
        raise ValueError(f"体素数过少: {n_avail}, 无法做 PCA")
    pca = PCA(n_components=n_comp, random_state=42)
    neural_reduced = pca.fit_transform(neural_responses)
    print(f"  Response PCA: {neural_responses.shape} -> {neural_reduced.shape}, explained variance: {pca.explained_variance_ratio_.sum():.4f}")
    return neural_reduced, pca


def reduce_model_dim(model_features, n_components=MODEL_DIM):
    pca = PCA(n_components=n_components, random_state=42)
    features_reduced = pca.fit_transform(model_features)
    print(f"  Model PCA: {model_features.shape} -> {features_reduced.shape}, explained variance: {pca.explained_variance_ratio_.sum():.4f}")
    return features_reduced, pca


def loocv_decode(neural_reduced, model_reduced, alpha=RIDGE_ALPHA):
    n_images = neural_reduced.shape[0]
    predictions = np.zeros_like(model_reduced)
    print(f"  执行 LOOCV ({n_images} 折)...")
    for i in range(n_images):
        if (i + 1) % 100 == 0:
            print(f"    处理 {i+1}/{n_images}")
        train_idx = np.concatenate([np.arange(i), np.arange(i + 1, n_images)])
        test_idx = np.array([i])
        X_train = neural_reduced[train_idx]
        X_test = neural_reduced[test_idx]
        y_train = model_reduced[train_idx]
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        reg = Ridge(alpha=alpha, fit_intercept=True)
        reg.fit(X_train_scaled, y_train)
        predictions[i] = reg.predict(X_test_scaled)[0]
    pred_target_sim = 1 - cdist(predictions, model_reduced, metric='correlation')
    mean_diag = float(np.mean(np.diag(pred_target_sim)))
    n_t = pred_target_sim.shape[0]
    diag_minus_offdiag = np.array([
        pred_target_sim[i, i] - (np.sum(pred_target_sim[i]) - pred_target_sim[i, i]) / max(n_t - 1, 1)
        for i in range(n_t)
    ])
    mean_diag_off = float(np.mean(diag_minus_offdiag))
    return mean_diag, mean_diag_off, predictions, pred_target_sim


def decode_one_model(neural_responses, model_features, model_name, response_dim, model_dim, alpha):
    neural_reduced, pca_neural = reduce_response_dim(neural_responses, n_components=response_dim)
    model_reduced, pca_model = reduce_model_dim(model_features, n_components=model_dim)
    mean_corr, diag_off, predictions, sim_matrix = loocv_decode(neural_reduced, model_reduced, alpha=alpha)
    return {
        "mean_corr": mean_corr,
        "diag_offdiag": diag_off,
        "predictions": predictions,
        "target_reduced": model_reduced,
        "sim_matrix": sim_matrix,
        "pca_neural": pca_neural,
        "pca_model": pca_model
    }


def get_image_files():
    files = sorted([f for f in os.listdir(STIMULI_PATH) if f.endswith('.bmp')])[:N_IMAGES]
    if len(files) < N_IMAGES:
        raise ValueError(f"stimuli 下不足 {N_IMAGES} 张 .bmp, 当前 {len(files)}")
    return files


def main():
    import argparse
    parser = argparse.ArgumentParser(description='fMRI Decoding (PCA + LOOCV), 7 脑区 x 多模型')
    parser.add_argument('--quick', action='store_true', help='仅 early 脑区 + AlexNet fc6 与 all-mpnet-base-v2')
    parser.add_argument('--regions', nargs='*', default=None, help='指定脑区，不传则全部')
    parser.add_argument('--response-dim', type=int, default=RESPONSE_DIM, help=f'Response PCA 维度 (默认 {RESPONSE_DIM})')
    parser.add_argument('--model-dim', type=int, default=MODEL_DIM, help=f'Model PCA 维度 (默认 {MODEL_DIM})')
    parser.add_argument('--alpha', type=float, default=RIDGE_ALPHA, help=f'Ridge alpha (默认 {RIDGE_ALPHA})')
    args = parser.parse_args()
    response_dim = args.response_dim
    model_dim = args.model_dim
    alpha = args.alpha

    start_time = datetime.now()
    print("=" * 70)
    print("fMRI Decoding (PCA + LOOCV), 7 脑区")
    if args.quick:
        print("  [快速模式: 仅 early + 2 模型]")
    print("开始时间:", start_time.strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 70)

    os.makedirs(OUT_DIR, exist_ok=True)
    image_files = get_image_files()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}\n")

    regions_to_run = REGIONS
    if args.regions is not None:
        names = {r[0] for r in REGIONS}
        for r in args.regions:
            if r not in names:
                raise ValueError(f"未知脑区: {r}, 可选: {sorted(names)}")
        regions_to_run = [(name, f) for name, f in REGIONS if name in args.regions]
    if args.quick:
        regions_to_run = [REGIONS[0]]

    all_region_results = {}

    for region_name, filename in regions_to_run:
        print("\n" + "=" * 70)
        print(f"脑区: {region_name}")
        print("=" * 70)
        neural_responses = load_region_neural_data(region_name, filename)
        print(f"  neural 形状 (n_images, n_voxels): {neural_responses.shape}")
        n_voxels = neural_responses.shape[1]
        resp_dim = min(response_dim, n_voxels - 1)
        if resp_dim < 1:
            print(f"  跳过 {region_name}: n_voxels={n_voxels} 无法做 PCA")
            continue

        results = {}

        print("  [1] AlexNet fc6")
        alexnet = pnag.load_alexnet(device)
        fc6 = pnag.extract_fc6_features(image_files, STIMULI_PATH, alexnet, device)
        del alexnet
        pnag.clear_gpu_memory()
        results["alexnet_fc6"] = decode_one_model(
            neural_responses, fc6, "alexnet_fc6", resp_dim, model_dim, alpha)
        print(f"  mean_corr = {results['alexnet_fc6']['mean_corr']:.4f}\n")

        if not args.quick:
            print("  [2] CLIP ViT-L-14 Image")
            clip_vit, pre_vit = pnag.load_clip_model('ViT-L-14', '/media/ubuntu/sda/TrippleN/model/ViT-L-14.pt', device)
            vit_img = pnag.extract_clip_image_features(image_files, STIMULI_PATH, clip_vit, pre_vit, device, clip_dim=768)
            del clip_vit, pre_vit
            pnag.clear_gpu_memory()
            results["clip_vit_l14_image"] = decode_one_model(
                neural_responses, vit_img, "clip_vit_l14_image", resp_dim, model_dim, alpha)
            print(f"  mean_corr = {results['clip_vit_l14_image']['mean_corr']:.4f}\n")

            print("  [3] CLIP ViT-L-14 Text")
            clip_vit, pre_vit = pnag.load_clip_model('ViT-L-14', '/media/ubuntu/sda/TrippleN/model/ViT-L-14.pt', device)
            vit_txt = pnag.extract_clip_text_features(CAPTIONS_PATH, clip_vit, device, clip_dim=768)
            del clip_vit, pre_vit
            pnag.clear_gpu_memory()
            results["clip_vit_l14_text"] = decode_one_model(
                neural_responses, vit_txt, "clip_vit_l14_text", resp_dim, model_dim, alpha)
            print(f"  mean_corr = {results['clip_vit_l14_text']['mean_corr']:.4f}\n")

            print("  [4] CLIP RN50 Image")
            clip_rn50, pre_rn50 = pnag.load_clip_model('RN50', '/media/ubuntu/sda/TrippleN/model/RN50.pt', device)
            rn50_img = pnag.extract_clip_image_features(image_files, STIMULI_PATH, clip_rn50, pre_rn50, device, clip_dim=1024)
            del clip_rn50, pre_rn50
            pnag.clear_gpu_memory()
            results["clip_rn50_image"] = decode_one_model(
                neural_responses, rn50_img, "clip_rn50_image", resp_dim, model_dim, alpha)
            print(f"  mean_corr = {results['clip_rn50_image']['mean_corr']:.4f}\n")

            print("  [5] CLIP RN50 Text")
            clip_rn50, pre_rn50 = pnag.load_clip_model('RN50', '/media/ubuntu/sda/TrippleN/model/RN50.pt', device)
            rn50_txt = pnag.extract_clip_text_features(CAPTIONS_PATH, clip_rn50, device, clip_dim=1024)
            del clip_rn50, pre_rn50
            pnag.clear_gpu_memory()
            results["clip_rn50_text"] = decode_one_model(
                neural_responses, rn50_txt, "clip_rn50_text", resp_dim, model_dim, alpha)
            print(f"  mean_corr = {results['clip_rn50_text']['mean_corr']:.4f}\n")

            print("  [6] CLIP RN101 Image")
            clip_rn101, pre_rn101 = pnag.load_clip_model('RN101', '/media/ubuntu/sda/TrippleN/model/RN101.pt', device)
            rn101_img = pnag.extract_clip_image_features(image_files, STIMULI_PATH, clip_rn101, pre_rn101, device, clip_dim=512)
            del clip_rn101, pre_rn101
            pnag.clear_gpu_memory()
            results["clip_rn101_image"] = decode_one_model(
                neural_responses, rn101_img, "clip_rn101_image", resp_dim, model_dim, alpha)
            print(f"  mean_corr = {results['clip_rn101_image']['mean_corr']:.4f}\n")

            print("  [7] CLIP RN101 Text")
            clip_rn101, pre_rn101 = pnag.load_clip_model('RN101', '/media/ubuntu/sda/TrippleN/model/RN101.pt', device)
            rn101_txt = pnag.extract_clip_text_features(CAPTIONS_PATH, clip_rn101, device, clip_dim=512)
            del clip_rn101, pre_rn101
            pnag.clear_gpu_memory()
            results["clip_rn101_text"] = decode_one_model(
                neural_responses, rn101_txt, "clip_rn101_text", resp_dim, model_dim, alpha)
            print(f"  mean_corr = {results['clip_rn101_text']['mean_corr']:.4f}\n")

        print("  [8] all-mpnet-base-v2")
        sent_model = pnag.load_sentence_model(device)
        sent_feat = pnag.extract_caption_features(CAPTIONS_PATH, sent_model)
        del sent_model
        pnag.clear_gpu_memory()
        results["all_mpnet_base_v2"] = decode_one_model(
            neural_responses, sent_feat, "all_mpnet_base_v2", resp_dim, model_dim, alpha)
        print(f"  mean_corr = {results['all_mpnet_base_v2']['mean_corr']:.4f}\n")

        if not args.quick:
            for dinov3_id in ('dinov3_vitl16', 'dinov3_convnext_base', 'dinov3_vitb16'):
                print(f"  [DINOv3] {dinov3_id}")
                weights_path = os.path.join(pnag.DINOV3_WEIGHTS_DIR, pnag.DINOV3_WEIGHT_FILES[dinov3_id])
                dim = pnag.DINOV3_FEATURE_DIMS[dinov3_id]
                model = pnag.load_dinov3_model(dinov3_id, weights_path, device)
                feats = pnag.extract_dinov3_features(image_files, STIMULI_PATH, model, device, dim)
                del model
                pnag.clear_gpu_memory()
                results[dinov3_id] = decode_one_model(
                    neural_responses, feats, dinov3_id, resp_dim, model_dim, alpha)
                print(f"  mean_corr = {results[dinov3_id]['mean_corr']:.4f}\n")

        out_path = os.path.join(OUT_DIR, f'fmri_decoding_{region_name}.pkl')
        save_dict = {k: {'mean_corr': v['mean_corr'], 'diag_offdiag': v['diag_offdiag']} for k, v in results.items()}
        with open(out_path, 'wb') as f:
            pickle.dump(save_dict, f)
        print(f"已保存: {out_path}")
        all_region_results[region_name] = save_dict

        detail_dict = {k: {'predictions': v['predictions'], 'target_reduced': v['target_reduced']} for k, v in results.items()}
        detail_path = os.path.join(OUT_DIR, f'fmri_decoding_{region_name}_detail.pkl')
        with open(detail_path, 'wb') as f:
            pickle.dump(detail_dict, f)
        print(f"已保存 (供 accuracy curve 用): {detail_path}")

    print("\n" + "=" * 70)
    print("fMRI Decoding 汇总 (mean_corr)")
    print("=" * 70)
    for region_name, res in all_region_results.items():
        print(f"\n  [{region_name}]")
        for name, d in res.items():
            print(f"    {name:<25} mean_corr = {d['mean_corr']:.4f}")
    print("=" * 70)
    with open(os.path.join(OUT_DIR, 'fmri_decoding_all_summary.pkl'), 'wb') as f:
        pickle.dump(all_region_results, f)
    print(f"结果目录: {OUT_DIR}")
    print(f"总耗时: {datetime.now() - start_time}")


if __name__ == '__main__':
    main()
