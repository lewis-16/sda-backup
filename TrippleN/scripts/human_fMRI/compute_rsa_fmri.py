#!/usr/bin/env python3

import os
import sys
import pickle
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import compute_rsa

FMRI_DATA_DIR = '/media/ubuntu/sda/TrippleN/human_fMRI'
OUT_DIR = '/media/ubuntu/sda/TrippleN/customize/human_fMRI/rsa'
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


def load_region_responses(region_name, filename):
    path = os.path.join(FMRI_DATA_DIR, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"fMRI 数据不存在: {path}")
    arr = np.load(path).astype(np.float32)
    if arr.shape[1] != N_IMAGES:
        raise ValueError(f"区域 {region_name} 期望最后一维为 {N_IMAGES}, 得到 {arr.shape}")
    return arr


def run_rsa_one_region(region_name, voxel_responses, model_rdms, n_repeats, n_sample):
    n_voxels = voxel_responses.shape[0]
    sample_size = min(n_sample, n_voxels) if n_sample > 0 else n_voxels
    replace = n_voxels < sample_size
    per_repeat = []

    for rep in range(n_repeats):
        rng = np.random.default_rng(rep)
        sampled_idx = rng.choice(n_voxels, size=sample_size, replace=replace)
        sampled_responses = voxel_responses[sampled_idx]
        brain_rdm = compute_rsa.compute_neuron_rdm_vectorized(sampled_responses)
        rep_results = []
        for model_name, model_rdm in model_rdms.items():
            spearman_r, spearman_p = compute_rsa.compute_rsa_correlation(model_rdm, brain_rdm, method='spearman')
            pearson_r, pearson_p = compute_rsa.compute_rsa_correlation(model_rdm, brain_rdm, method='pearson')
            rep_results.append({
                'model_name': model_name,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p,
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
            })
        per_repeat.append({
            'repeat': rep,
            'sample_size': sample_size,
            'results': rep_results
        })

    results = []
    model_names = list(model_rdms.keys())
    for model_name in model_names:
        vals_s = []
        vals_p = []
        for rep_item in per_repeat:
            row = next(r for r in rep_item['results'] if r['model_name'] == model_name)
            vals_s.append(row['spearman_r'])
            vals_p.append(row['pearson_r'])
        vals_s = np.array(vals_s, dtype=np.float64)
        vals_p = np.array(vals_p, dtype=np.float64)
        results.append({
            'model_name': model_name,
            'spearman_r_mean': float(np.nanmean(vals_s)),
            'spearman_r_std': float(np.nanstd(vals_s)),
            'pearson_r_mean': float(np.nanmean(vals_p)),
            'pearson_r_std': float(np.nanstd(vals_p)),
        })

    return {
        'region_name': region_name,
        'n_voxels': n_voxels,
        'n_repeats': n_repeats,
        'sample_size': sample_size,
        'results': results,
        'per_repeat': per_repeat
    }


def compute_model_rdms(model_features_list, model_names):
    model_rdms = {}
    for model_name, model_features in zip(model_names, model_features_list):
        model_rdms[model_name] = compute_rsa.compute_rdm_vectorized(model_features, method='correlation')
    return model_rdms


def run_rsa_one_region_single(region_name, voxel_responses, model_features_list, model_names, out_path):
    brain_rdm = compute_rsa.compute_neuron_rdm_vectorized(voxel_responses)
    results = []
    for model_name, model_features in zip(model_names, model_features_list):
        r = compute_rsa.run_rsa_analysis(model_name, model_features, brain_rdm, out_path)
        r_small = {
            'model_name': r['model_name'],
            'spearman_r': r['spearman_r'],
            'spearman_p': r['spearman_p'],
            'pearson_r': r['pearson_r'],
            'pearson_p': r['pearson_p'],
            'n_features': r['n_features'],
        }
        results.append(r_small)
    return {'region_name': region_name, 'n_voxels': voxel_responses.shape[0], 'results': results}


def main():
    import argparse
    parser = argparse.ArgumentParser(description='fMRI RSA 分析 (7 脑区 x 多模型)')
    parser.add_argument('--quick', action='store_true', help='仅 early 脑区')
    parser.add_argument('--regions', nargs='*', default=None, help='指定脑区，不传则全部')
    parser.add_argument('--n-repeats', type=int, default=200, help='随机抽样重复次数')
    parser.add_argument('--n-sample', type=int, default=3000, help='每次随机抽样voxel数量')
    parser.add_argument('--single-pass', action='store_true', help='关闭随机抽样，兼容旧流程')
    args = parser.parse_args()

    start_time = datetime.now()
    print("=" * 70)
    print("fMRI RSA 分析 (7 脑区)")
    if args.quick:
        print("  [快速模式: 仅 early]")
    print("开始时间:", start_time.strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 70)

    os.makedirs(OUT_DIR, exist_ok=True)
    image_files = sorted([f for f in os.listdir(STIMULI_PATH) if f.endswith('.bmp')])[:N_IMAGES]
    print(f"\n使用图像数量: {len(image_files)}")
    device = __import__('torch').device('cuda' if __import__('torch').cuda.is_available() else 'cpu')
    print(f"使用设备: {device}\n")

    regions_to_run = REGIONS
    if args.regions is not None:
        names = {r[0] for r in REGIONS}
        for r in args.regions:
            if r not in names:
                raise ValueError(f"未知脑区: {r}, 可选: {sorted(names)}")
        regions_to_run = [(name, f) for name, f in REGIONS if name in args.regions]
    if args.quick:
        regions_to_run = [REGIONS[0]]

    print("[1/2] 加载所有模型特征...")
    alexnet = compute_rsa.load_alexnet(device)
    fc6_features = compute_rsa.extract_fc6_features(image_files, STIMULI_PATH, alexnet, device)
    del alexnet
    compute_rsa.clear_gpu_memory()

    clip_vit_l14_model, clip_vit_l14_preprocess = compute_rsa.load_clip_model(
        'ViT-L-14', '/media/ubuntu/sda/TrippleN/model/ViT-L-14.pt', device)
    clip_vit_l14_image = compute_rsa.extract_clip_image_features(
        image_files, STIMULI_PATH, clip_vit_l14_model, clip_vit_l14_preprocess, device, clip_dim=768)
    clip_vit_l14_text = compute_rsa.extract_clip_text_features(
        CAPTIONS_PATH, clip_vit_l14_model, device, clip_dim=768)
    del clip_vit_l14_model, clip_vit_l14_preprocess
    compute_rsa.clear_gpu_memory()

    clip_rn50_model, clip_rn50_preprocess = compute_rsa.load_clip_model(
        'RN50', '/media/ubuntu/sda/TrippleN/model/RN50.pt', device)
    clip_rn50_image = compute_rsa.extract_clip_image_features(
        image_files, STIMULI_PATH, clip_rn50_model, clip_rn50_preprocess, device, clip_dim=1024)
    clip_rn50_text = compute_rsa.extract_clip_text_features(
        CAPTIONS_PATH, clip_rn50_model, device, clip_dim=1024)
    del clip_rn50_model, clip_rn50_preprocess
    compute_rsa.clear_gpu_memory()

    clip_rn101_model, clip_rn101_preprocess = compute_rsa.load_clip_model(
        'RN101', '/media/ubuntu/sda/TrippleN/model/RN101.pt', device)
    clip_rn101_image = compute_rsa.extract_clip_image_features(
        image_files, STIMULI_PATH, clip_rn101_model, clip_rn101_preprocess, device, clip_dim=512)
    clip_rn101_text = compute_rsa.extract_clip_text_features(
        CAPTIONS_PATH, clip_rn101_model, device, clip_dim=512)
    del clip_rn101_model, clip_rn101_preprocess
    compute_rsa.clear_gpu_memory()

    sentence_model = compute_rsa.load_sentence_model(device)
    sentence_features = compute_rsa.extract_caption_features(CAPTIONS_PATH, sentence_model)
    del sentence_model
    compute_rsa.clear_gpu_memory()

    model_features_list = [
        fc6_features,
        clip_vit_l14_text,
        clip_vit_l14_image,
        clip_rn50_text,
        clip_rn50_image,
        clip_rn101_text,
        clip_rn101_image,
        sentence_features,
    ]
    model_names = [
        "AlexNet fc6",
        "CLIP ViT-L-14 Text",
        "CLIP ViT-L-14 Image",
        "CLIP RN50 Text",
        "CLIP RN50 Image",
        "CLIP RN101 Text",
        "CLIP RN101 Image",
        "all-mpnet-base-v2",
    ]
    model_rdms = None
    if not args.single_pass:
        print("  预计算模型RDM...")
        model_rdms = compute_model_rdms(model_features_list, model_names)

    print("[2/2] 对每个脑区计算 RSA...")
    if args.single_pass:
        print("  [single-pass: 不进行重复抽样]")
    else:
        print(f"  [bootstrap: n_repeats={args.n_repeats}, n_sample={args.n_sample}]")
    all_region_results = []
    for region_name, filename in regions_to_run:
        print(f"\n--- 脑区: {region_name} ---")
        voxel_responses = load_region_responses(region_name, filename)
        print(f"  响应形状: {voxel_responses.shape}")
        out_path = os.path.join(OUT_DIR, f'fmri_rsa_{region_name}.pkl')
        if args.single_pass:
            region_result = run_rsa_one_region_single(
                region_name, voxel_responses, model_features_list, model_names, out_path
            )
        else:
            region_result = run_rsa_one_region(
                region_name, voxel_responses, model_rdms, args.n_repeats, args.n_sample
            )
        all_region_results.append(region_result)
        with open(out_path, 'wb') as f:
            pickle.dump(region_result, f)
        print(f"  已保存: {out_path}")

    summary_path = os.path.join(OUT_DIR, 'fmri_rsa_summary.pkl')
    with open(summary_path, 'wb') as f:
        pickle.dump({'region_results': all_region_results, 'model_names': model_names}, f)
    print(f"\n汇总已保存: {summary_path}")

    print("\n" + "=" * 70)
    print("fMRI RSA 汇总 (Spearman r)")
    print("=" * 70)
    for region_res in all_region_results:
        print(f"\n  [{region_res['region_name']}] n_voxels={region_res['n_voxels']}")
        if args.single_pass:
            for r in region_res['results']:
                print(f"    {r['model_name']:<28} spearman_r = {r['spearman_r']:.4f}")
        else:
            for r in region_res['results']:
                print(
                    f"    {r['model_name']:<28} "
                    f"spearman_r = {r['spearman_r_mean']:.4f} +- {r['spearman_r_std']:.4f}"
                )
    print("=" * 70)
    print(f"结果目录: {OUT_DIR}")
    print(f"总耗时: {datetime.now() - start_time}")


if __name__ == '__main__':
    main()
