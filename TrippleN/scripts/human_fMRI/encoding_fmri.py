#!/usr/bin/env python3

import os
import sys
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import predict_neuron_activity_gpu as pnag

FMRI_DATA_DIR = '/media/ubuntu/sda/TrippleN/human_fMRI'
OUT_DIR = '/media/ubuntu/sda/TrippleN/customize/human_fMRI/encoding'
STIMULI_PATH = '/media/ubuntu/sda/TrippleN/stimuli'
CAPTIONS_PATH = '/media/ubuntu/sda/TrippleN/customize/coco_captions_1000x5.pkl'
N_IMAGES = 1000

REGIONS = [
    ('early', 'resp_early_func1py8mm.npy'),
    ('midventral', 'resp_midventral_func1py8mm.npy'),
    ('midlateral', 'resp_midlateral_func1py8mm.npy'),
    ('midparietal', 'resp_midparietal_func1py8mm.npy'),
    ('ventral', 'resp_ventral_func1py8mm.npy'),
    ('lateral', 'resp_lateral_func1py8mm.npy'),
    ('parietal', 'resp_parietal_func1py8mm.npy'),
]


def make_voxel_info(n_voxels):
    return pd.DataFrame({
        'reliability_best': np.ones(n_voxels, dtype=np.float64),
        'best_r_time1': np.zeros(n_voxels, dtype=np.int64),
        'best_r_time2': np.ones(n_voxels, dtype=np.int64),
        'subject': ['fmri'] * n_voxels,
    })


def load_region_responses(region_name, filename):
    path = os.path.join(FMRI_DATA_DIR, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"fMRI 数据不存在: {path}")
    arr = np.load(path).astype(np.float32)
    if arr.shape[1] != N_IMAGES:
        raise ValueError(f"区域 {region_name} 期望最后一维为 {N_IMAGES}, 得到 {arr.shape}")
    return arr


def main():
    import argparse
    parser = argparse.ArgumentParser(description='fMRI Encoding (PLSR), 7 脑区 x 多模型')
    parser.add_argument('--quick', action='store_true', help='仅 early 脑区 + AlexNet fc6 与 all-mpnet-base-v2')
    parser.add_argument('--regions', nargs='*', default=None, help='指定脑区，不传则全部。可选: early, midventral, midlateral, midparietal, ventral, lateral, parietal')
    args = parser.parse_args()

    start_time = datetime.now()
    print("=" * 70)
    print("fMRI Encoding (PLSR), 7 脑区")
    if args.quick:
        print("  [快速模式: 仅 early + 2 模型]")
    print("开始时间:", start_time.strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 70)

    os.makedirs(OUT_DIR, exist_ok=True)
    image_files = sorted([f for f in os.listdir(STIMULI_PATH) if f.endswith('.bmp')])[:N_IMAGES]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}\n")

    regions_to_run = REGIONS
    if args.regions is not None:
        names = {r[0] for r in REGIONS}
        for r in args.regions:
            if r not in names:
                raise ValueError(f"未知脑区: {r}, 可选: {sorted(names)}")
        regions_to_run = [(name, f) for name, f in REGIONS if name in args.regions]

    all_region_summary = {}

    for region_name, filename in regions_to_run:
        print("\n" + "=" * 70)
        print(f"脑区: {region_name}")
        print("=" * 70)
        voxel_responses = load_region_responses(region_name, filename)
        print(f"  响应形状 (n_voxels, n_images): {voxel_responses.shape}")
        voxel_info = make_voxel_info(voxel_responses.shape[0])
        results = {}

        print("  [1] AlexNet fc6")
        alexnet = pnag.load_alexnet(device)
        fc6 = pnag.extract_fc6_features(image_files, STIMULI_PATH, alexnet, device)
        del alexnet
        pnag.clear_gpu_memory()
        enc_corr, norm_corr, best_comps = pnag.evaluate_encoding_performance_gpu(
            fc6, voxel_responses, voxel_info, encoding_type="alexnet_fc6")
        results["alexnet_fc6"] = {'mean_corr': float(np.mean(enc_corr)), 'encoding_correlation': enc_corr, 'best_n_components': best_comps}
        pnag.save_results(enc_corr, norm_corr, voxel_info,
                          os.path.join(OUT_DIR, f'fmri_encoding_{region_name}_alexnet_fc6.pkl'),
                          encoding_type="alexnet_fc6", best_components=best_comps)
        pnag.clear_gpu_memory()

        if not args.quick:
            print("  [2] CLIP ViT-L-14 Image")
            clip_vit, pre_vit = pnag.load_clip_model('ViT-L-14', '/media/ubuntu/sda/TrippleN/model/ViT-L-14.pt', device)
            vit_img = pnag.extract_clip_image_features(image_files, STIMULI_PATH, clip_vit, pre_vit, device, clip_dim=768)
            del clip_vit, pre_vit
            pnag.clear_gpu_memory()
            enc_corr, norm_corr, best_comps = pnag.evaluate_encoding_performance_gpu(
                vit_img, voxel_responses, voxel_info, encoding_type="clip_vit_l14_image")
            results["clip_vit_l14_image"] = {'mean_corr': float(np.mean(enc_corr)), 'encoding_correlation': enc_corr, 'best_n_components': best_comps}
            pnag.save_results(enc_corr, norm_corr, voxel_info,
                              os.path.join(OUT_DIR, f'fmri_encoding_{region_name}_clip_vit_l14_image.pkl'),
                              encoding_type="clip_vit_l14_image", best_components=best_comps)
            pnag.clear_gpu_memory()

            print("  [3] CLIP ViT-L-14 Text")
            clip_vit, pre_vit = pnag.load_clip_model('ViT-L-14', '/media/ubuntu/sda/TrippleN/model/ViT-L-14.pt', device)
            vit_txt = pnag.extract_clip_text_features(CAPTIONS_PATH, clip_vit, device, clip_dim=768)
            del clip_vit, pre_vit
            pnag.clear_gpu_memory()
            enc_corr, norm_corr, best_comps = pnag.evaluate_encoding_performance_gpu(
                vit_txt, voxel_responses, voxel_info, encoding_type="clip_vit_l14_text")
            results["clip_vit_l14_text"] = {'mean_corr': float(np.mean(enc_corr)), 'encoding_correlation': enc_corr, 'best_n_components': best_comps}
            pnag.save_results(enc_corr, norm_corr, voxel_info,
                              os.path.join(OUT_DIR, f'fmri_encoding_{region_name}_clip_vit_l14_text.pkl'),
                              encoding_type="clip_vit_l14_text", best_components=best_comps)
            pnag.clear_gpu_memory()

            print("  [4] CLIP RN50 Image")
            clip_rn50, pre_rn50 = pnag.load_clip_model('RN50', '/media/ubuntu/sda/TrippleN/model/RN50.pt', device)
            rn50_img = pnag.extract_clip_image_features(image_files, STIMULI_PATH, clip_rn50, pre_rn50, device, clip_dim=1024)
            del clip_rn50, pre_rn50
            pnag.clear_gpu_memory()
            enc_corr, norm_corr, best_comps = pnag.evaluate_encoding_performance_gpu(
                rn50_img, voxel_responses, voxel_info, encoding_type="clip_rn50_image")
            results["clip_rn50_image"] = {'mean_corr': float(np.mean(enc_corr)), 'encoding_correlation': enc_corr, 'best_n_components': best_comps}
            pnag.save_results(enc_corr, norm_corr, voxel_info,
                              os.path.join(OUT_DIR, f'fmri_encoding_{region_name}_clip_rn50_image.pkl'),
                              encoding_type="clip_rn50_image", best_components=best_comps)
            pnag.clear_gpu_memory()

            print("  [5] CLIP RN50 Text")
            clip_rn50, pre_rn50 = pnag.load_clip_model('RN50', '/media/ubuntu/sda/TrippleN/model/RN50.pt', device)
            rn50_txt = pnag.extract_clip_text_features(CAPTIONS_PATH, clip_rn50, device, clip_dim=1024)
            del clip_rn50, pre_rn50
            pnag.clear_gpu_memory()
            enc_corr, norm_corr, best_comps = pnag.evaluate_encoding_performance_gpu(
                rn50_txt, voxel_responses, voxel_info, encoding_type="clip_rn50_text")
            results["clip_rn50_text"] = {'mean_corr': float(np.mean(enc_corr)), 'encoding_correlation': enc_corr, 'best_n_components': best_comps}
            pnag.save_results(enc_corr, norm_corr, voxel_info,
                              os.path.join(OUT_DIR, f'fmri_encoding_{region_name}_clip_rn50_text.pkl'),
                              encoding_type="clip_rn50_text", best_components=best_comps)
            pnag.clear_gpu_memory()

            print("  [6] CLIP RN101 Image")
            clip_rn101, pre_rn101 = pnag.load_clip_model('RN101', '/media/ubuntu/sda/TrippleN/model/RN101.pt', device)
            rn101_img = pnag.extract_clip_image_features(image_files, STIMULI_PATH, clip_rn101, pre_rn101, device, clip_dim=512)
            del clip_rn101, pre_rn101
            pnag.clear_gpu_memory()
            enc_corr, norm_corr, best_comps = pnag.evaluate_encoding_performance_gpu(
                rn101_img, voxel_responses, voxel_info, encoding_type="clip_rn101_image")
            results["clip_rn101_image"] = {'mean_corr': float(np.mean(enc_corr)), 'encoding_correlation': enc_corr, 'best_n_components': best_comps}
            pnag.save_results(enc_corr, norm_corr, voxel_info,
                              os.path.join(OUT_DIR, f'fmri_encoding_{region_name}_clip_rn101_image.pkl'),
                              encoding_type="clip_rn101_image", best_components=best_comps)
            pnag.clear_gpu_memory()

            print("  [7] CLIP RN101 Text")
            clip_rn101, pre_rn101 = pnag.load_clip_model('RN101', '/media/ubuntu/sda/TrippleN/model/RN101.pt', device)
            rn101_txt = pnag.extract_clip_text_features(CAPTIONS_PATH, clip_rn101, device, clip_dim=512)
            del clip_rn101, pre_rn101
            pnag.clear_gpu_memory()
            enc_corr, norm_corr, best_comps = pnag.evaluate_encoding_performance_gpu(
                rn101_txt, voxel_responses, voxel_info, encoding_type="clip_rn101_text")
            results["clip_rn101_text"] = {'mean_corr': float(np.mean(enc_corr)), 'encoding_correlation': enc_corr, 'best_n_components': best_comps}
            pnag.save_results(enc_corr, norm_corr, voxel_info,
                              os.path.join(OUT_DIR, f'fmri_encoding_{region_name}_clip_rn101_text.pkl'),
                              encoding_type="clip_rn101_text", best_components=best_comps)
            pnag.clear_gpu_memory()

        print("  [8] all-mpnet-base-v2")
        sent_model = pnag.load_sentence_model(device)
        sent_feat = pnag.extract_caption_features(CAPTIONS_PATH, sent_model)
        del sent_model
        pnag.clear_gpu_memory()
        enc_corr, norm_corr, best_comps = pnag.evaluate_encoding_performance_gpu(
            sent_feat, voxel_responses, voxel_info, encoding_type="all_mpnet_base_v2")
        results["all_mpnet_base_v2"] = {'mean_corr': float(np.mean(enc_corr)), 'encoding_correlation': enc_corr, 'best_n_components': best_comps}
        pnag.save_results(enc_corr, norm_corr, voxel_info,
                          os.path.join(OUT_DIR, f'fmri_encoding_{region_name}_all_mpnet_base_v2.pkl'),
                          encoding_type="all_mpnet_base_v2", best_components=best_comps)
        pnag.clear_gpu_memory()

        if not args.quick:
            for dinov3_id in ('dinov3_vitl16', 'dinov3_convnext_base', 'dinov3_vitb16'):
                print(f"  [DINOv3] {dinov3_id}")
                weights_path = os.path.join(pnag.DINOV3_WEIGHTS_DIR, pnag.DINOV3_WEIGHT_FILES[dinov3_id])
                dim = pnag.DINOV3_FEATURE_DIMS[dinov3_id]
                model = pnag.load_dinov3_model(dinov3_id, weights_path, device)
                feats = pnag.extract_dinov3_features(image_files, STIMULI_PATH, model, device, dim)
                del model
                pnag.clear_gpu_memory()
                enc_corr, norm_corr, best_comps = pnag.evaluate_encoding_performance_gpu(
                    feats, voxel_responses, voxel_info, encoding_type=dinov3_id)
                results[dinov3_id] = {'mean_corr': float(np.mean(enc_corr)), 'encoding_correlation': enc_corr, 'best_n_components': best_comps}
                pnag.save_results(enc_corr, norm_corr, voxel_info,
                                  os.path.join(OUT_DIR, f'fmri_encoding_{region_name}_{dinov3_id}.pkl'),
                                  encoding_type=dinov3_id, best_components=best_comps)
                pnag.clear_gpu_memory()

        summary_small = {k: v['mean_corr'] for k, v in results.items()}
        all_region_summary[region_name] = summary_small
        with open(os.path.join(OUT_DIR, f'fmri_encoding_{region_name}_summary.pkl'), 'wb') as f:
            pickle.dump(summary_small, f)

    print("\n" + "=" * 70)
    print("fMRI Encoding 汇总 (mean encoding correlation)")
    print("=" * 70)
    for region_name, res in all_region_summary.items():
        print(f"\n  [{region_name}]")
        for name, mean_corr in res.items():
            print(f"    {name:<25} mean_corr = {mean_corr:.4f}")
    print("=" * 70)
    with open(os.path.join(OUT_DIR, 'fmri_encoding_all_summary.pkl'), 'wb') as f:
        pickle.dump(all_region_summary, f)
    print(f"结果目录: {OUT_DIR}")
    print(f"总耗时: {datetime.now() - start_time}")


if __name__ == '__main__':
    main()
