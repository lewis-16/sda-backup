#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AutoSort Evaluation Pipeline

Main steps:
1. Load data (same as training)
2.5. Load pre-prepared evaluation data
3. Load models and evaluate for each clique and time segment
"""

import numpy as np
import pandas as pd
import pickle
import warnings
import re
import sys
import gc
import time
from pathlib import Path

# Try to import tqdm for progress bars, fallback to simple print if not available
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Simple tqdm replacement
    class tqdm:
        def __init__(self, iterable=None, total=None, desc=None, leave=True, unit=None):
            self.iterable = iterable
            self.total = total
            self.desc = desc or ""
            self.leave = leave
            self.unit = unit or "it"
            self.current = 0
            if iterable is not None:
                self.iterable = iterable
            else:
                self.iterable = range(total) if total else []
        
        def __enter__(self):
            if self.desc:
                print(f"{self.desc}: ", end="", flush=True)
            return self
        
        def __exit__(self, *args):
            print()  # New line after progress
        
        def __iter__(self):
            for item in self.iterable:
                self.current += 1
                if self.total and self.current % max(1, self.total // 20) == 0:
                    print(f"{self.desc}: {self.current}/{self.total} ({100*self.current/self.total:.1f}%)", end="\r", flush=True)
                yield item
        
        def update(self, n=1):
            self.current += n
            if self.total and self.current % max(1, self.total // 20) == 0:
                print(f"{self.desc}: {self.current}/{self.total} ({100*self.current/self.total:.1f}%)", end="\r", flush=True)
        
        def close(self):
            if self.total:
                print(f"{self.desc}: {self.current}/{self.total} (100.0%)")

warnings.filterwarnings('ignore')

import torch
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to save memory
import matplotlib.pyplot as plt
import seaborn as sns

from utils_clique import (
    prepare_training_data,
    evaluate_autosort_model,
    build_sliding_cliques,
    CliqueInfo,
    compute_valid_channels,
    visualize_umap_features,
    SimpleAutoSort,
    SimpleWaveformLoader,
    detect_spike
)


def load_data():
    """Load recording, cliques, and GT data"""
    print("=" * 80)
    print("Step 1: Loading Data")
    print("=" * 80)
    
    recording_path = "/media/ubuntu/sda/Spike_Sorting/paper_architecture/02_simulation_data/02_Neuropixel_384_channels/data_generation/recording_neuropixels_150_Neuron_3600s.h5"
    spike_inf_path = "/media/ubuntu/sda/Spike_Sorting/paper_architecture/01_real_data/01_flexible_probe_30_channels/clique/spike_inf.tsv"
    neuron_inf_path = "/media/ubuntu/sda/Spike_Sorting/paper_architecture/01_real_data/01_flexible_probe_30_channels/clique/neuron_inf.pkl"
    
    # Load recording and sorting using MEArec
    print("Loading recording...")
    recording, sorting = se.read_mearec(recording_path)
    
    # Get probe from recording
    probe = recording.get_probe()
    if probe is None:
        raise ValueError("Recording does not have probe information")
    
    # Load clique information (saved during training)
    base_save_dir = "/media/ubuntu/sda/Spike_Sorting/paper_architecture/02_simulation_data/02_Neuropixel_384_channels/data_generation/recording_neuropixels_150_Neuron_3600s/autosort_input/"
    clique_info_path = Path(base_save_dir) / "clique_info.pkl"
    
    if clique_info_path.exists():
        with open(clique_info_path, 'rb') as f:
            clique_info = pickle.load(f)
        cliques = clique_info['cliques']
        print(f"Loaded {len(cliques)} cliques from {clique_info_path}")
    else:
        # Build cliques from probe (same as training)
        cliques = build_sliding_cliques(
            probe,
            clique_size=49,
            min_size=25,
            min_overlap=18,
            target_groups=12,
        )
        print(f"Built {len(cliques)} cliques")
    
    # Preprocess recording (same as training)
    recording_f = recording.rename_channels(range(384))
    
    # Load GT data
    print("Loading GT data...")
    if Path(spike_inf_path).exists():
        spike_inf = pd.read_csv(spike_inf_path, sep='\t', index_col=0)
    else:
        raise ValueError(f"spike_inf not found at {spike_inf_path}")
    
    if Path(neuron_inf_path).exists():
        with open(neuron_inf_path, 'rb') as f:
            neuron_inf = pickle.load(f)
    else:
        raise ValueError(f"neuron_inf.pkl not found at {neuron_inf_path}")
    
    print(f"Recording loaded successfully")
    print(f"Sampling rate: {recording_f.get_sampling_frequency()} Hz")
    print(f"Number of channels: {recording_f.get_num_channels()}")
    print(f"Recording duration: {recording_f.get_num_samples() / recording_f.get_sampling_frequency():.2f} seconds")
    print()
    
    return recording_f, cliques, spike_inf, neuron_inf, base_save_dir


def load_prepared_eval_data(base_save_dir):
    """Load pre-prepared evaluation data from disk"""
    print("=" * 80)
    print("Step 2.5: Loading Pre-prepared Evaluation Data")
    print("=" * 80)
    
    # Define evaluation time segments (should match the segments used during data preparation)
    eval_time_segments = [
        (600, 1200),   # Segment 0: 600-1200 seconds
        (1200, 1800),  # Segment 1: 1200-1800 seconds
        (1800, 2400),  # Segment 2: 1800-2400 seconds
        (2400, 3000),  # Segment 3: 2400-3000 seconds
        (3000, 3600),  # Segment 4: 3000-3600 seconds
    ]
    
    print(f"Scanning for pre-prepared evaluation data in: {base_save_dir}")
    
    # Scan for available eval data
    all_eval_data_dirs = {}  # {segment_id: {clique_id: eval_data_dir}}
    
    base_path = Path(base_save_dir)
    
    # Find all clique directories
    clique_dirs = sorted([d for d in base_path.iterdir() if d.is_dir() and d.name.startswith('clique_')])
    
    print(f"Found {len(clique_dirs)} clique directories")
    
    # Scan with minimal output
    found_count = 0
    for clique_dir in tqdm(clique_dirs, desc="Scanning cliques", leave=False):
        # Extract clique_id from directory name (e.g., "clique_00" -> 0)
        clique_match = re.match(r'clique_(\d+)', clique_dir.name)
        if not clique_match:
            continue
        clique_id = int(clique_match.group(1))
        
        # Find all eval_segment directories
        eval_segment_dirs = sorted([d for d in clique_dir.iterdir() 
                                    if d.is_dir() and d.name.startswith('eval_segment_')])
        
        for eval_segment_dir in eval_segment_dirs:
            # Extract segment_id from directory name (e.g., "eval_segment_00" -> 0)
            segment_match = re.match(r'eval_segment_(\d+)', eval_segment_dir.name)
            if not segment_match:
                continue
            segment_id = int(segment_match.group(1))
            
            # Check if train_data directory exists and has required files
            train_data_dir = eval_segment_dir / "train_data"
            if not train_data_dir.exists():
                continue
            
            # Check for required files
            required_files = [
                'X_waveform.pkl',
                'X_spiketrain_time.pkl',
                'Y_spike_id.pkl',
                'Y_spike_id_noise.pkl',
                'neuron_mapping.pkl'
            ]
            
            all_files_exist = all((train_data_dir / fname).exists() for fname in required_files)
            
            if all_files_exist:
                if segment_id not in all_eval_data_dirs:
                    all_eval_data_dirs[segment_id] = {}
                all_eval_data_dirs[segment_id][clique_id] = str(train_data_dir)
                found_count += 1
    
    # Print summary
    print(f"\nFound {found_count} clique-segment pairs")
    print(f"Summary of loaded evaluation data:")
    for seg_id in sorted(all_eval_data_dirs.keys()):
        cliques_in_segment = sorted(all_eval_data_dirs[seg_id].keys())
        print(f"  Segment {seg_id}: {len(cliques_in_segment)} cliques")
    
    total_segments = len(all_eval_data_dirs)
    total_clique_segments = sum(len(cliques) for cliques in all_eval_data_dirs.values())
    print(f"\nTotal: {total_segments} segments, {total_clique_segments} clique-segment pairs")
    print()
    
    return all_eval_data_dirs, eval_time_segments


def check_evaluation_completed(results_save_dir):
    """Check if evaluation has been completed by checking for result files"""
    results_save_path = Path(results_save_dir)
    evaluation_results_path = results_save_path / 'evaluation_results.csv'
    evaluation_summary_path = results_save_path / 'evaluation_summary.csv'
    
    # Check if both result files exist
    if evaluation_results_path.exists() and evaluation_summary_path.exists():
        return True
    return False


def load_evaluation_results(results_save_dir):
    """Load previously saved evaluation results"""
    results_save_path = Path(results_save_dir)
    evaluation_summary_path = results_save_path / 'evaluation_summary.csv'
    
    if not evaluation_summary_path.exists():
        return None
    
    try:
        summary_df = pd.read_csv(evaluation_summary_path)
        summary_dict = dict(zip(summary_df['metric'], summary_df['value']))
        
        # Create a minimal results dictionary
        results = {
            'noise_accuracy': summary_dict.get('noise_accuracy', 0.0),
            'unit_accuracy': summary_dict.get('unit_accuracy', 0.0),
            'unit_f1_score': summary_dict.get('unit_f1_score', 0.0),
            'gt_units': [],  # We don't load full GT units to save memory
        }
        
        # Add adjusted metrics if available
        if 'noise_accuracy_adjusted' in summary_dict:
            results['noise_accuracy_adjusted'] = summary_dict.get('noise_accuracy_adjusted', 0.0)
            results['unit_accuracy_adjusted'] = summary_dict.get('unit_accuracy_adjusted', 0.0)
            results['unit_f1_score_adjusted'] = summary_dict.get('unit_f1_score_adjusted', 0.0)
        
        return results
    except Exception as e:
        print(f"    Warning: Failed to load saved results: {e}")
        return None


def evaluate_models(all_eval_data_dirs, eval_time_segments, cliques, base_save_dir, verbose=False, resume=True):
    """Evaluate models for each clique and time segment
    
    Parameters:
        resume: If True, check for existing results and skip completed evaluations
    """
    print("=" * 80)
    print("Step 3: Loading Models and Evaluating")
    print("=" * 80)
    
    if resume:
        print("Resume mode: Checking for existing evaluation results...")
    
    # Evaluation parameters
    evaluation_params = {
        'batch_size': 512,
        'left_sample': 10,
        'right_sample': 20,
    }
    
    # Number of training runs per clique
    n_runs = 5
    
    # Store all evaluation results
    all_results = {}  # {segment_id: {clique_id: {run_id: results}}}
    
    # Calculate total number of evaluations for progress bar
    total_evaluations = 0
    completed_evaluations = 0
    skipped_evaluations = 0
    
    for seg_id in all_eval_data_dirs.keys():
        for clique_id in all_eval_data_dirs[seg_id].keys():
            for run_id in range(1, n_runs + 1):
                model_save_dir = Path(base_save_dir) / f"clique_{clique_id:02d}" / "model_save" / f"run_{run_id}"
                if model_save_dir.exists():
                    total_evaluations += 1
                    if resume:
                        results_save_dir = model_save_dir / f"eval_segment_{seg_id:02d}"
                        if check_evaluation_completed(results_save_dir):
                            completed_evaluations += 1
    
    if resume and completed_evaluations > 0:
        print(f"Found {completed_evaluations} completed evaluations out of {total_evaluations} total")
        print(f"Will skip completed evaluations and continue from where it left off...")
    
    # Use tqdm for progress tracking
    if HAS_TQDM:
        pbar = tqdm(total=total_evaluations, desc="Evaluating models", unit="eval")
    else:
        pbar = tqdm(total=total_evaluations, desc="Evaluating models", unit="eval")
        print(f"Starting evaluation of {total_evaluations} model-segment pairs...")
    
    for seg_id in sorted(all_eval_data_dirs.keys()):
        if verbose:
            print(f"\nEvaluating Time Segment {seg_id}")
        
        all_results[seg_id] = {}
        
        for clique_id in sorted(all_eval_data_dirs[seg_id].keys()):
            if verbose:
                print(f"  Clique {clique_id:02d} - Segment {seg_id:02d}")
            
            eval_data_dir = all_eval_data_dirs[seg_id][clique_id]
            
            # Get number of channels for this clique
            n_channels = len(cliques[clique_id].device_channel_indices)
            
            all_results[seg_id][clique_id] = {}
            
            # Evaluate each training run
            for run_id in range(1, n_runs + 1):
                # Model save directory for this clique and run
                model_save_dir = Path(base_save_dir) / f"clique_{clique_id:02d}" / "model_save" / f"run_{run_id}"
                
                if not model_save_dir.exists():
                    pbar.update(1)
                    continue
                
                # Check if evaluation already completed
                results_save_dir = model_save_dir / f"eval_segment_{seg_id:02d}"
                
                # For segment 1, run 1: we need full results (including way3_features) for UMAP visualization
                # So we'll re-evaluate even if summary results exist
                need_full_results = (seg_id == 1 and run_id == 1)
                is_completed = resume and check_evaluation_completed(results_save_dir)
                
                # Skip evaluation if already completed and we don't need full results
                if is_completed and not need_full_results:
                    # Load existing results (for non-UMAP segments, summary is enough)
                    if verbose:
                        print(f"    Run {run_id} already completed, loading results...")
                    
                    loaded_results = load_evaluation_results(results_save_dir)
                    if loaded_results is not None:
                        all_results[seg_id][clique_id][run_id] = loaded_results
                        skipped_evaluations += 1
                        if verbose:
                            print(f"      Noise accuracy: {loaded_results.get('noise_accuracy', 0.0):.4f}")
                            if 'unit_accuracy' in loaded_results:
                                print(f"      Unit accuracy: {loaded_results.get('unit_accuracy', 0.0):.4f}")
                                print(f"      Unit F1 score: {loaded_results.get('unit_f1_score', 0.0):.4f}")
                        
                        # Clean up memory and rest
                        del loaded_results
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        if verbose:
                            print(f"    Cleaning memory and resting 10 seconds...")
                        time.sleep(10)
                        
                        pbar.update(1)
                        continue
                    else:
                        # If loading failed, re-evaluate
                        if verbose:
                            print(f"    Failed to load results, re-evaluating...")
                
                # Evaluate model (either not completed, or need full results for UMAP)
                if is_completed and need_full_results:
                    if verbose:
                        print(f"    Run {run_id} completed but need full results for UMAP, re-evaluating...")
                
                # Evaluate model
                try:
                    results = evaluate_autosort_model(
                        train_data_dir=eval_data_dir,
                        model_save_dir=str(model_save_dir) + "/",
                        n_channels=n_channels,
                        **evaluation_params,
                        save_results=True,
                        results_save_dir=str(results_save_dir),
                    )
                    
                    if verbose:
                        print(f"    Run {run_id} evaluation completed!")
                        print(f"      Noise accuracy: {results['noise_accuracy']:.4f}")
                        if len(results['gt_units']) > 0:
                            print(f"      Unit accuracy: {results['unit_accuracy']:.4f}")
                            print(f"      Unit F1 score: {results['unit_f1_score']:.4f}")
                    
                    # For segment 1, run 1: keep full results for UMAP visualization
                    # For others: keep only summary to save memory
                    if seg_id == 1 and run_id == 1:
                        # Keep full results for UMAP
                        all_results[seg_id][clique_id][run_id] = results
                    else:
                        # Keep only essential metrics in memory to prevent memory buildup
                        summary_results = {
                            'noise_accuracy': results.get('noise_accuracy', 0.0),
                            'unit_accuracy': results.get('unit_accuracy', 0.0),
                            'unit_f1_score': results.get('unit_f1_score', 0.0),
                            'gt_units': len(results.get('gt_units', [])),
                        }
                        all_results[seg_id][clique_id][run_id] = summary_results
                        # Explicitly delete large result objects
                        del results
                    
                    # Clean up memory after each evaluation
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Rest 10 seconds after each evaluation to prevent system overload
                    if verbose:
                        print(f"    Cleaning memory and resting 10 seconds...")
                    time.sleep(10)
                    
                except Exception as e:
                    if verbose:
                        print(f"    Error evaluating run {run_id}: {e}")
                        import traceback
                        traceback.print_exc()
                    
                    # Clean up memory even if error occurred
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    time.sleep(10)
                
                pbar.update(1)
    
    pbar.close()
    
    if resume and skipped_evaluations > 0:
        print(f"\nSkipped {skipped_evaluations} already completed evaluations")
    print(f"\nAll evaluations completed!")
    print()
    
    return all_results


def visualize_umap_segment1_run1(all_eval_data_dirs, eval_time_segments, cliques, base_save_dir, all_results):
    """Generate UMAP visualizations for segment 1, run 1, all 12 cliques"""
    print("=" * 80)
    print("Step 4: UMAP Visualization for Segment 1, Run 1")
    print("=" * 80)
    
    seg_id = 1
    run_id = 1
    
    if seg_id not in all_results:
        print(f"No results for segment {seg_id}, skipping UMAP visualization...")
        return
    
    start_time, end_time = eval_time_segments[seg_id]
    print(f"UMAP Visualization for Segment {seg_id} ({start_time}-{end_time} seconds), Run {run_id}")
    print(f"Processing all {len(cliques)} cliques...")
    
    # Neuron color mapping
    neuron_inf_color = ["#a74a5b", "#d64158", "#e28572", "#d6522c",
                        "#a5572c", "#da9131", "#d6a46a", "#8c6d2c",
                        "#c0ab39", "#6d7821", "#9cb835", "#67733a",
                        "#9eb56c", "#4c902f", "#61c350", "#418348",
                        "#54c083", "#338b70", "#51c6c0", "#609dd8",
                        "#6365ab", "#636edd", "#a85aca", "#c590d9",
                        "#9c4d88", "#d5449a", "#e280a9"]
    
    processed_count = 0
    for clique_id in tqdm(range(len(cliques)), desc="Generating UMAP"):
        if clique_id not in all_results[seg_id]:
            continue
        if run_id not in all_results[seg_id][clique_id]:
            continue
        
        try:
            results = all_results[seg_id][clique_id][run_id]
            
            # Get way3 features
            way3_features_100d = results.get('way3_features_100d', np.array([]))
            way3_features_30d = results.get('way3_features_30d', np.array([]))
            
            if len(way3_features_100d) == 0 and len(way3_features_30d) == 0:
                print(f"  Clique {clique_id:02d}: No way3 features available, skipping...")
                continue
            
            # Load neuron mapping from evaluation data
            eval_data_dir = all_eval_data_dirs[seg_id][clique_id]
            neuron_mapping_path = Path(eval_data_dir) / "neuron_mapping.pkl"
            
            if not neuron_mapping_path.exists():
                print(f"  Clique {clique_id:02d}: Neuron mapping not found, skipping...")
                continue
            
            with open(neuron_mapping_path, 'rb') as f:
                neuron_mapping = pickle.load(f)
            
            train_neuron_list = list(neuron_mapping['id_to_neuron'].values())
            
            # Create results_df for label classification visualization
            gt_noise = results['gt_noise']
            pred_noise = results['noise_predictions']
            gt_units = results['gt_units']
            pred_units = results['unit_predictions']
            
            # Create results_df: only for spikes that passed noise classifier
            results_df_list = []
            n_spikes_passed = len(way3_features_30d)
            
            if len(gt_units) > 0 and n_spikes_passed > 0:
                n_samples = min(n_spikes_passed, len(gt_units))
                
                for i in range(n_samples):
                    gt_unit_id = gt_units[i]
                    pred_unit_id = pred_units[i] if i < len(pred_units) else -1
                    
                    # Map unit IDs to neuron names
                    gt_label = neuron_mapping['id_to_neuron'].get(gt_unit_id, 'unmatch')
                    pred_label = neuron_mapping['id_to_neuron'].get(pred_unit_id, 'unmatch')
                    
                    results_df_list.append({
                        'gt_label': gt_label,
                        'predicted_label': pred_label,
                    })
            
            results_df = pd.DataFrame(results_df_list) if len(results_df_list) > 0 else pd.DataFrame()
            
            # Create neuron color mapping
            neuron_color_dict = {}
            for i, neuron in enumerate(train_neuron_list):
                if i < len(neuron_inf_color):
                    neuron_color_dict[neuron] = neuron_inf_color[i]
                else:
                    cmap = plt.cm.get_cmap('tab20')
                    neuron_color_dict[neuron] = cmap(i % 20)
            
            # Generate UMAP visualization
            figs = visualize_umap_features(
                way3_features_100d=way3_features_100d,
                way3_features_30d=way3_features_30d,
                results_df=results_df,
                train_neuron_list=train_neuron_list,
                noise_gt_labels=gt_noise,
                noise_pred_labels=pred_noise,
                neuron_inf_color=neuron_color_dict,
                n_samples=50000,
                random_state=42
            )
            
            # Save figures
            umap_save_dir = Path(base_save_dir) / f"clique_{clique_id:02d}" / "model_save" / f"run_{run_id}" / f"eval_segment_{seg_id:02d}" / "umap"
            umap_save_dir.mkdir(parents=True, exist_ok=True)
            
            figure_names = [
                'noise_detection_gt',
                'noise_detection_pred',
                'label_classification_gt',
                'label_classification_pred'
            ]
            
            for fig, name in zip(figs, figure_names):
                if fig is not None:
                    save_path = umap_save_dir / f"{name}.pdf"
                    fig.savefig(save_path, dpi=300, bbox_inches='tight')
                    plt.close(fig)
            
            processed_count += 1
            
            # Clean up to save memory
            del results, way3_features_100d, way3_features_30d, results_df, neuron_mapping, train_neuron_list
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Rest 10 seconds after each UMAP visualization
            print(f"  Clique {clique_id:02d}: UMAP completed, cleaning memory and resting 10 seconds...")
            time.sleep(10)
            
        except Exception as e:
            print(f"  Error generating UMAP for clique {clique_id:02d}: {e}")
            import traceback
            traceback.print_exc()
            # Clean up memory even if error occurred
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            time.sleep(10)
    
    print(f"\nUMAP visualization completed for {processed_count} cliques!")
    print()


def compute_processing_time(recording_f, cliques, base_save_dir, eval_time_segments, window_params):
    """Compute processing time for different window sizes"""
    print("=" * 80)
    print("Step 5: Computing Processing Time")
    print("=" * 80)
    
    import time
    
    # Test parameters
    window_sizes_ms = [100, 200, 300, 400, 500, 1000]  # milliseconds
    n_runs_per_window = 20  # Number of segments to process for each window size
    
    # Detection parameters
    detection_params = {
        'thr_min': 2.7,
        'thr_max': 15,
        'distance': 4,
        'ch_max_simul_firing': 8,
        'wlen': 6,
        'prominence': 10,
    }
    
    # Use first time segment for testing
    test_segment_id = 0
    start_time, end_time = eval_time_segments[test_segment_id]
    duration_seconds = end_time - start_time
    sampling_rate = recording_f.get_sampling_frequency()
    start_sample = int(start_time * sampling_rate)
    end_sample = int(end_time * sampling_rate)
    recording_segment = recording_f.frame_slice(start_frame=start_sample, end_frame=end_sample)
    
    # Load models and calibration results for each clique
    # We'll use run_1 for all cliques
    run_id = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    clique_models = {}
    clique_calibration_results = {}
    
    print(f"Test segment: {test_segment_id} ({start_time}-{end_time} seconds)")
    print(f"Processing window sizes to test: {window_sizes_ms} ms")
    print(f"Number of runs per window size: {n_runs_per_window}")
    print(f"Loading models and calibration results for run {run_id}...")
    
    # Load models and calibration results for each clique
    for clique_id in tqdm(range(len(cliques)), desc="Loading models"):
        model_save_dir = Path(base_save_dir) / f"clique_{clique_id:02d}" / "model_save" / f"run_{run_id}"
        
        if not model_save_dir.exists():
            continue
        
        try:
            # Load model
            n_channels = len(cliques[clique_id].device_channel_indices)
            dataset = SimpleWaveformLoader(
                train_data_dir=str(model_save_dir.parent.parent.parent / f"clique_{clique_id:02d}" / "train_data"),
                n_channels=n_channels,
                left_sample=window_params['left_sample'],
                right_sample=window_params['right_sample'],
            )
            
            autosort_model = SimpleAutoSort(
                n_channels=n_channels,
                window_size=window_params['left_sample'] + window_params['right_sample'],
                n_units=dataset.n_units,
                set_shank_id=None,
                save_dir=str(model_save_dir) + "/",
                pos_weight_noise=dataset.pos_weight_noise.to(device),
                pos_weight_label=dataset.pos_weight_label.to(device)
            )
            autosort_model.load_model()
            autosort_model.eval()
            clique_models[clique_id] = autosort_model
            
            # Load calibration results (if available)
            calibration_path = model_save_dir / "calibration_results.pkl"
            if calibration_path.exists():
                with open(calibration_path, 'rb') as f:
                    clique_calibration_results[clique_id] = pickle.load(f)
        
        except Exception as e:
            print(f"  Error loading model for clique {clique_id}: {e}")
            continue
    
    print(f"Loaded {len(clique_models)} models and {len(clique_calibration_results)} calibration results")
    
    # Select cliques that have both model and calibration results
    available_cliques = [cid for cid in clique_models.keys() if cid in clique_calibration_results]
    print(f"Available cliques for testing: {available_cliques}")
    
    if len(available_cliques) == 0:
        print("No cliques with both model and calibration results available, skipping timing...")
        return
    
    # Process single clique timing function
    def process_single_clique_timing(
        recording_clique,
        autosort_model,
        calibration_results,
        start_frame,
        time_window_seconds,
        detection_params,
        window_params,
        device=None,
    ):
        """Process a single clique time window and measure processing time."""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        left_sample = window_params['left_sample']
        right_sample = window_params['right_sample']
        window_size = left_sample + right_sample
        sampling_frequency = recording_clique.get_sampling_frequency()
        
        # Get models and mapping from calibration stage
        kmeans_model = calibration_results['kmeans_model']
        pca_model = calibration_results['pca_model']
        cluster_to_neuron_mapping = calibration_results['cluster_to_neuron_mapping']
        
        # Start timing: from threshold detection
        start_time = time.time()
        
        # 1. Load current window data
        window_frames = int(time_window_seconds * sampling_frequency)
        end_frame = min(start_frame + window_frames, recording_clique.get_num_samples())
        traces = recording_clique.get_traces(start_frame=start_frame, end_frame=end_frame)
        if traces.shape[0] > traces.shape[1] and traces.shape[0] > 100:
            traces = traces.T
        traces = traces.astype(np.float32)
        
        # 2. Threshold detection
        trace0_car = traces.T  # (n_timepoints, n_channels)
        spikes = detect_spike(trace0_car, **detection_params)
        spike_coords = np.argwhere(spikes == 1)  # (n_spikes, 2) [time, channel]
        
        if len(spike_coords) == 0:
            return 0.0
        
        # 3. Extract waveforms and filter boundaries
        waveforms = []
        spike_times = []
        spike_channels = []
        
        for time_idx, channel_idx in spike_coords:
            global_time_idx = start_frame + time_idx
            local_start = time_idx - left_sample
            local_end = time_idx + right_sample
            
            if local_start < 0 or local_end > trace0_car.shape[0]:
                continue
            if local_end - local_start != window_size:
                continue
            
            waveform = traces[:, local_start:local_end]  # (n_channels, window_size)
            waveforms.append(waveform)
            spike_times.append(global_time_idx)
            spike_channels.append(channel_idx)
        
        if len(waveforms) == 0:
            return 0.0
        
        waveforms = np.array(waveforms)  # (n_spikes, n_channels, window_size)
        
        # 4. Pass through noise classifier, classified as spikes
        batch_size = 512
        n_spikes = len(waveforms)
        way3_features_list = []
        
        autosort_model.eval()
        with torch.no_grad():
            for i in range(0, n_spikes, batch_size):
                batch_end = min(i + batch_size, n_spikes)
                batch_waveforms = waveforms[i:batch_end]
                batch_channels = spike_channels[i:batch_end]
                
                batch_single_waveforms = []
                batch_multi_waveforms = []
                
                for wf, ch in zip(batch_waveforms, batch_channels):
                    multi_wf = wf.flatten()
                    batch_multi_waveforms.append(multi_wf)
                    single_wf = wf[ch, :]
                    batch_single_waveforms.append(single_wf)
                
                batch_multi_waveforms = np.array(batch_multi_waveforms)
                batch_single_waveforms = np.array(batch_single_waveforms)
                
                batch_multi = torch.from_numpy(batch_multi_waveforms).float().to(device)
                batch_single = torch.from_numpy(batch_single_waveforms).float().to(device)
                
                codes = torch.cat((batch_multi, batch_single), dim=1)
                
                noise_output = autosort_model.clsfier_noise(codes)
                noise_pred = torch.argmax(noise_output, dim=1)
                
                spike_mask = noise_pred == 1
                if spike_mask.sum() > 0:
                    codes_spike = codes[spike_mask]
                    way3_batch = autosort_model.clsfier_label.intermediate_forward(codes_spike)
                    way3_features_list.append(way3_batch.cpu().numpy())
        
        if len(way3_features_list) == 0:
            end_time = time.time()
            return end_time - start_time
        
        way3_features = np.concatenate(way3_features_list, axis=0)
        
        # 5. PCA dimensionality reduction
        way3_pca = pca_model.transform(way3_features)
        
        # 6. K-means prediction
        cluster_labels = kmeans_model.predict(way3_pca)
        
        # 7. Map to train neuron ID (we don't need the actual predictions for timing)
        
        end_time = time.time()
        return end_time - start_time
    
    # Initialize results
    timing_results = []
    
    # Use available cliques (limit to 5 for testing)
    test_cliques = [cliques[cid] for cid in available_cliques[:min(5, len(available_cliques))]]
    test_clique_ids = available_cliques[:min(5, len(available_cliques))]
    
    # Process each window size
    for window_size_ms in window_sizes_ms:
        print(f"\nTesting processing window size: {window_size_ms} ms")
        time_window_seconds = window_size_ms / 1000.0
        
        # Test single clique processing time
        single_clique_times = []
        all_cliques_serial_times = []
        
        for run_idx in tqdm(range(n_runs_per_window), desc=f"Window {window_size_ms}ms", leave=False):
            # Random start time within segment
            random_start_time = np.random.uniform(start_time, end_time - time_window_seconds)
            start_frame = int(random_start_time * sampling_rate)
            
            # Test single clique (use first clique as example)
            if len(test_clique_ids) > 0:
                clique_id = test_clique_ids[0]
                clique = test_cliques[0]
                clique_channels = list(set(clique.device_channel_indices))
                recording_clique = recording_segment.select_channels(channel_ids=clique_channels)
                
                try:
                    single_clique_time = process_single_clique_timing(
                        recording_clique=recording_clique,
                        autosort_model=clique_models[clique_id],
                        calibration_results=clique_calibration_results[clique_id],
                        start_frame=start_frame,
                        time_window_seconds=time_window_seconds,
                        detection_params=detection_params,
                        window_params=window_params,
                        device=device,
                    )
                    single_clique_times.append(single_clique_time)
                except Exception as e:
                    single_clique_times.append(np.nan)
            
            # Test serial processing of all cliques
            serial_start = time.time()
            for clique_id, clique in zip(test_clique_ids, test_cliques):
                clique_channels = list(set(clique.device_channel_indices))
                recording_clique = recording_segment.select_channels(channel_ids=clique_channels)
                try:
                    process_single_clique_timing(
                        recording_clique=recording_clique,
                        autosort_model=clique_models[clique_id],
                        calibration_results=clique_calibration_results[clique_id],
                        start_frame=start_frame,
                        time_window_seconds=time_window_seconds,
                        detection_params=detection_params,
                        window_params=window_params,
                        device=device,
                    )
                except Exception as e:
                    pass
            serial_end = time.time()
            all_cliques_serial_times.append(serial_end - serial_start)
        
        # Store results
        for run_idx in range(n_runs_per_window):
            timing_results.append({
                'window_size_ms': window_size_ms,
                'run': run_idx + 1,
                'single_clique_time': single_clique_times[run_idx] if run_idx < len(single_clique_times) else np.nan,
                'all_cliques_serial_time': all_cliques_serial_times[run_idx],
            })
        
        if len(single_clique_times) > 0:
            print(f"  Single clique: {np.nanmean(single_clique_times):.4f} ± {np.nanstd(single_clique_times):.4f} seconds")
        print(f"  Serial processing (all cliques): {np.mean(all_cliques_serial_times):.4f} ± {np.std(all_cliques_serial_times):.4f} seconds")
    
    # Convert to DataFrame and save
    timing_df = pd.DataFrame(timing_results)
    timing_df_path = Path(base_save_dir) / "clique_processing_timing.csv"
    timing_df.to_csv(timing_df_path, index=False)
    print(f"\nTiming results saved to: {timing_df_path}")
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"Timing results summary:")
    print(f"{'='*80}")
    print(timing_df.groupby('window_size_ms').agg({
        'single_clique_time': ['mean', 'std'],
        'all_cliques_serial_time': ['mean', 'std'],
    }).round(4))
    print()
    
    # Clean up
    del clique_models, clique_calibration_results
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    """Main evaluation pipeline"""
    try:
        # Step 1: Load data
        recording_f, cliques, spike_inf, neuron_inf, base_save_dir = load_data()
        
        # Step 2.5: Load pre-prepared evaluation data
        all_eval_data_dirs, eval_time_segments = load_prepared_eval_data(base_save_dir)
        
        # Step 3: Evaluate models (with minimal output)
        # resume=True enables checkpoint resuming - will skip already completed evaluations
        all_results = evaluate_models(
            all_eval_data_dirs, 
            eval_time_segments, 
            cliques, 
            base_save_dir,
            verbose=False,  # Set to True for detailed output
            resume=True  # Enable checkpoint resuming
        )
        
        # Print summary
        print("=" * 80)
        print("Evaluation Summary")
        print("=" * 80)
        for seg_id in sorted(all_results.keys()):
            print(f"\nSegment {seg_id}:")
            for clique_id in sorted(all_results[seg_id].keys()):
                print(f"  Clique {clique_id:02d}:")
                for run_id in sorted(all_results[seg_id][clique_id].keys()):
                    res = all_results[seg_id][clique_id][run_id]
                    if isinstance(res, dict) and 'noise_accuracy' in res:
                        print(f"    Run {run_id}: Noise Acc={res.get('noise_accuracy', 0):.4f}, "
                              f"Unit Acc={res.get('unit_accuracy', 0):.4f}, "
                              f"F1={res.get('unit_f1_score', 0):.4f}")
        
        # Step 4: UMAP Visualization for Segment 1, Run 1
        visualize_umap_segment1_run1(
            all_eval_data_dirs,
            eval_time_segments,
            cliques,
            base_save_dir,
            all_results
        )
        
        # Step 5: Compute Processing Time
        window_params = {
            'left_sample': 10,
            'right_sample': 20,
        }
        compute_processing_time(
            recording_f,
            cliques,
            base_save_dir,
            eval_time_segments,
            window_params
        )
        
        print("\n" + "=" * 80)
        print("All steps completed successfully!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

