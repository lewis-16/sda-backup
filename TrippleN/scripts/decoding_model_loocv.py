#!/usr/bin/env python3
"""
Decoding model with PCA + LOOCV (Leave-One-Out Cross-Validation)

按照论文方法:
1. Response space 降到 500 维 (PCA)
2. Model features 降到 100 维 (PCA)
3. LOOCV: 对每张图片，用其他 999 张训练线性回归，预测该图片的 model features
"""

import os
import sys
import pickle
import numpy as np
from datetime import datetime
from scipy.spatial.distance import cdist
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import predict_neuron_activity_gpu as pnag

STIMULI_PATH = '/media/ubuntu/sda/TrippleN/stimuli'
NEURON_RESPONSES_PATH = '/media/ubuntu/sda/TrippleN/customize/neuron_responses_1000.npy'
CAPTIONS_PATH = '/media/ubuntu/sda/TrippleN/customize/coco_captions_1000x5.pkl'
OUTPUT_DIR = '/media/ubuntu/sda/TrippleN/customize/decoding_analysis'
N_IMAGES = 1000
RESPONSE_DIM = 500
MODEL_DIM = 100
RIDGE_ALPHA = 1.0


def load_neural_data():
    neuron_responses = np.load(NEURON_RESPONSES_PATH).astype(np.float32)
    if neuron_responses.shape[1] != N_IMAGES:
        raise ValueError(f"neuron_responses 期望最后一维为 {N_IMAGES}, 得到 {neuron_responses.shape}")
    return neuron_responses.T


def get_image_files():
    files = sorted([f for f in os.listdir(STIMULI_PATH) if f.endswith('.bmp')])[:N_IMAGES]
    if len(files) < N_IMAGES:
        raise ValueError(f"stimuli 下不足 {N_IMAGES} 张 .bmp, 当前 {len(files)}")
    return files


def reduce_response_dim(neural_responses, n_components=RESPONSE_DIM):
    pca = PCA(n_components=n_components, random_state=42)
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
        
        train_idx = np.concatenate([np.arange(i), np.arange(i+1, n_images)])
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


def decode_one_model(neural_responses, model_features, model_name, response_dim=None, model_dim=None):
    print(f"\n处理模型: {model_name}")
    print(f"  输入: neural {neural_responses.shape}, model {model_features.shape}")
    rd = RESPONSE_DIM if response_dim is None else int(response_dim)
    md = MODEL_DIM if model_dim is None else int(model_dim)
    rd = min(rd, neural_responses.shape[1], neural_responses.shape[0])
    md = min(md, model_features.shape[1], model_features.shape[0])
    neural_reduced, pca_neural = reduce_response_dim(neural_responses, n_components=rd)
    model_reduced, pca_model = reduce_model_dim(model_features, n_components=md)
    
    mean_corr, diag_off, predictions, sim_matrix = loocv_decode(neural_reduced, model_reduced, alpha=RIDGE_ALPHA)
    
    return {
        "mean_corr": mean_corr,
        "diag_offdiag": diag_off,
        "predictions": predictions,
        "target_reduced": model_reduced,
        "sim_matrix": sim_matrix,
        "pca_neural": pca_neural,
        "pca_model": pca_model
    }


def main():
    global RESPONSE_DIM, MODEL_DIM, RIDGE_ALPHA
    import argparse
    parser = argparse.ArgumentParser(description='Decoding with PCA + LOOCV')
    parser.add_argument('--quick', action='store_true', help='仅跑 AlexNet fc6 与 all-mpnet-base-v2')
    parser.add_argument('--models', nargs='*', default=None,
                        help='要运行的模型标识，不传则运行全部。可选: %s' % ', '.join(pnag.ALL_MODELS))
    parser.add_argument('--response-dim', type=int, default=RESPONSE_DIM, help=f'Response PCA 维度 (默认 {RESPONSE_DIM})')
    parser.add_argument('--model-dim', type=int, default=MODEL_DIM, help=f'Model PCA 维度 (默认 {MODEL_DIM})')
    parser.add_argument('--alpha', type=float, default=RIDGE_ALPHA, help=f'Ridge alpha (默认 {RIDGE_ALPHA})')
    args = parser.parse_args()
    RESPONSE_DIM = args.response_dim
    MODEL_DIM = args.model_dim
    RIDGE_ALPHA = args.alpha
    selected = args.models if args.models else pnag.ALL_MODELS
    if not args.quick:
        for m in selected:
            if m not in pnag.ALL_MODELS:
                raise ValueError('未知模型: %s，可选: %s' % (m, ', '.join(pnag.ALL_MODELS)))
    
    start_time = datetime.now()
    print("=" * 70)
    print("Decoding with PCA + LOOCV (论文方法)")
    print(f"  Response PCA: {RESPONSE_DIM} 维")
    print(f"  Model PCA: {MODEL_DIM} 维")
    print(f"  LOOCV: 1000 折 (每张图片用其他 999 张训练)")
    if args.quick:
        print("  (快速模式: 仅 AlexNet fc6 + all-mpnet-base-v2)")
    else:
        print("  运行模型: %s" % ', '.join(selected))
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    neural_responses = load_neural_data()
    print(f"\n神经元响应: {neural_responses.shape} (n_images, n_neurons)")

    image_files = get_image_files()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}\n")

    results = {}

    if 'alexnet' in selected:
        print("[1/8] AlexNet fc6")
        alexnet = pnag.load_alexnet(device)
        fc6 = pnag.extract_fc6_features(image_files, STIMULI_PATH, alexnet, device)
        del alexnet
        pnag.clear_gpu_memory()
        result = decode_one_model(neural_responses, fc6, "alexnet_fc6")
        results["alexnet_fc6"] = result
        print(f"  mean_corr = {result['mean_corr']:.4f}\n")
    
    if args.quick:
        if 'all_mpnet_base_v2' not in results:
            print("[2/2] all-mpnet-base-v2 (quick)")
            sent_model = pnag.load_sentence_model(device)
            sent_feat = pnag.extract_caption_features(CAPTIONS_PATH, sent_model)
            del sent_model
            pnag.clear_gpu_memory()
            result = decode_one_model(neural_responses, sent_feat, "all_mpnet_base_v2")
            results["all_mpnet_base_v2"] = result
            print(f"  mean_corr = {result['mean_corr']:.4f}\n")
        out_path = os.path.join(OUTPUT_DIR, 'decoding_results_loocv.pkl')
        with open(out_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"结果已保存: {out_path}")
        print("\n" + "=" * 70)
        for name, d in results.items():
            print(f"  {name:<25} mean_corr = {d['mean_corr']:.4f}")
        print("=" * 70)
        print(f"总耗时: {datetime.now() - start_time}")
        return

    if 'clip_vit_l14_image' in selected:
        print("[2/8] CLIP ViT-L-14 Image")
        clip_vit, preprocess_vit = pnag.load_clip_model('ViT-L-14', '/media/ubuntu/sda/TrippleN/model/ViT-L-14.pt', device)
        vit_img = pnag.extract_clip_image_features(image_files, STIMULI_PATH, clip_vit, preprocess_vit, device, clip_dim=768)
        del clip_vit, preprocess_vit
        pnag.clear_gpu_memory()
        result = decode_one_model(neural_responses, vit_img, "clip_vit_l14_image")
        results["clip_vit_l14_image"] = result
        print(f"  mean_corr = {result['mean_corr']:.4f}\n")

    if 'clip_vit_l14_text' in selected:
        print("[3/8] CLIP ViT-L-14 Text")
        clip_vit, preprocess_vit = pnag.load_clip_model('ViT-L-14', '/media/ubuntu/sda/TrippleN/model/ViT-L-14.pt', device)
        vit_txt = pnag.extract_clip_text_features(CAPTIONS_PATH, clip_vit, device, clip_dim=768)
        del clip_vit, preprocess_vit
        pnag.clear_gpu_memory()
        result = decode_one_model(neural_responses, vit_txt, "clip_vit_l14_text")
        results["clip_vit_l14_text"] = result
        print(f"  mean_corr = {result['mean_corr']:.4f}\n")

    if 'clip_rn50_image' in selected:
        print("[4/8] CLIP RN50 Image")
        clip_rn50, pre_rn50 = pnag.load_clip_model('RN50', '/media/ubuntu/sda/TrippleN/model/RN50.pt', device)
        rn50_img = pnag.extract_clip_image_features(image_files, STIMULI_PATH, clip_rn50, pre_rn50, device, clip_dim=1024)
        del clip_rn50, pre_rn50
        pnag.clear_gpu_memory()
        result = decode_one_model(neural_responses, rn50_img, "clip_rn50_image")
        results["clip_rn50_image"] = result
        print(f"  mean_corr = {result['mean_corr']:.4f}\n")

    if 'clip_rn50_text' in selected:
        print("[5/8] CLIP RN50 Text")
        clip_rn50, pre_rn50 = pnag.load_clip_model('RN50', '/media/ubuntu/sda/TrippleN/model/RN50.pt', device)
        rn50_txt = pnag.extract_clip_text_features(CAPTIONS_PATH, clip_rn50, device, clip_dim=1024)
        del clip_rn50, pre_rn50
        pnag.clear_gpu_memory()
        result = decode_one_model(neural_responses, rn50_txt, "clip_rn50_text")
        results["clip_rn50_text"] = result
        print(f"  mean_corr = {result['mean_corr']:.4f}\n")

    if 'clip_rn101_image' in selected:
        print("[6/8] CLIP RN101 Image")
        clip_rn101, pre_rn101 = pnag.load_clip_model('RN101', '/media/ubuntu/sda/TrippleN/model/RN101.pt', device)
        rn101_img = pnag.extract_clip_image_features(image_files, STIMULI_PATH, clip_rn101, pre_rn101, device, clip_dim=512)
        del clip_rn101, pre_rn101
        pnag.clear_gpu_memory()
        result = decode_one_model(neural_responses, rn101_img, "clip_rn101_image")
        results["clip_rn101_image"] = result
        print(f"  mean_corr = {result['mean_corr']:.4f}\n")

    if 'clip_rn101_text' in selected:
        print("[7/8] CLIP RN101 Text")
        clip_rn101, pre_rn101 = pnag.load_clip_model('RN101', '/media/ubuntu/sda/TrippleN/model/RN101.pt', device)
        rn101_txt = pnag.extract_clip_text_features(CAPTIONS_PATH, clip_rn101, device, clip_dim=512)
        del clip_rn101, pre_rn101
        pnag.clear_gpu_memory()
        result = decode_one_model(neural_responses, rn101_txt, "clip_rn101_text")
        results["clip_rn101_text"] = result
        print(f"  mean_corr = {result['mean_corr']:.4f}\n")

    if 'all_mpnet_base_v2' in selected:
        print("[8/8] all-mpnet-base-v2")
        sent_model = pnag.load_sentence_model(device)
        sent_feat = pnag.extract_caption_features(CAPTIONS_PATH, sent_model)
        del sent_model
        pnag.clear_gpu_memory()
        result = decode_one_model(neural_responses, sent_feat, "all_mpnet_base_v2")
        results["all_mpnet_base_v2"] = result
        print(f"  mean_corr = {result['mean_corr']:.4f}\n")

    for dinov3_id in ('dinov3_vitl16', 'dinov3_convnext_base', 'dinov3_vitb16'):
        if dinov3_id not in selected:
            continue
        print("[DINOv3] %s" % dinov3_id)
        weights_path = os.path.join(pnag.DINOV3_WEIGHTS_DIR, pnag.DINOV3_WEIGHT_FILES[dinov3_id])
        dim = pnag.DINOV3_FEATURE_DIMS[dinov3_id]
        model = pnag.load_dinov3_model(dinov3_id, weights_path, device)
        feats = pnag.extract_dinov3_features(image_files, STIMULI_PATH, model, device, dim)
        del model
        pnag.clear_gpu_memory()
        result = decode_one_model(neural_responses, feats, dinov3_id)
        results[dinov3_id] = result
        print(f"  mean_corr = {result['mean_corr']:.4f}\n")

    out_path = os.path.join(OUTPUT_DIR, 'decoding_results_loocv.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"结果已保存: {out_path}")

    print("\n" + "=" * 70)
    print("Decoding 性能汇总 (mean correlation)")
    print("=" * 70)
    for name, d in results.items():
        print(f"  {name:<25} mean_corr = {d['mean_corr']:.4f}")
    print("=" * 70)
    print(f"总耗时: {datetime.now() - start_time}")


if __name__ == '__main__':
    main()
