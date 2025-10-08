#!/usr/bin/env python3
"""
Monkey Electrophysiology Data Synchronization Script
===================================================

This script is specifically designed to work with the data structure found in
/media/ubuntu/sda/Monkey/ directory, implementing the synchronization mechanism
between image stimulus sequences and electrophysiological recordings.

Usage:
    python monkey_sync.py --data_dir /media/ubuntu/sda/Monkey --date 20240112 --block 1

Author: AI Assistant
Date: 2024
"""

import numpy as np
import pandas as pd
import os
import glob
import argparse
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json

try:
    from scipy import signal
    from scipy.io import loadmat
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Some features may not work.")

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False
    print("Warning: h5py not available. Some features may not work.")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class MonkeySyncConfig:
    """Configuration for monkey data synchronization"""
    # Timing parameters
    trial_length: float = 0.3  # seconds
    pre_trial: float = 0.1     # seconds
    post_trial: float = 0.2     # seconds
    
    # Sampling parameters
    fs: int = 30000  # Hz
    downsample_factor: int = 30
    
    # Filter parameters
    bandpass_freq: Tuple[float, float] = (500, 5000)  # Hz
    lowpass_freq: float = 200  # Hz
    filter_order: int = 2
    
    # Photodiode parameters
    photodiode_threshold: float = 1e4
    photodiode_channel: int = 15
    
    # Stimulus parameters
    stimbit: int = 1

class MonkeyDataSynchronizer:
    """Synchronizer for monkey electrophysiology data"""
    
    def __init__(self, config: MonkeySyncConfig = None):
        self.config = config or MonkeySyncConfig()
        self._setup_filters()
    
    def _setup_filters(self):
        """Setup digital filters"""
        nyquist = self.config.fs / 2
        
        # Bandpass filter
        self.bandpass_b, self.bandpass_a = signal.butter(
            self.config.filter_order,
            [self.config.bandpass_freq[0]/nyquist, self.config.bandpass_freq[1]/nyquist],
            btype='band'
        )
        
        # Lowpass filter
        self.lowpass_b, self.lowpass_a = signal.butter(
            self.config.filter_order,
            self.config.lowpass_freq/nyquist,
            btype='low'
        )
    
    def find_data_files(self, data_dir: str, date: str, block: int) -> Dict[str, str]:
        """Find relevant data files for given date and block"""
        files = {}
        
        # Look for trigger files
        trigger_pattern = f"{data_dir}/trigger/trigger_df_monkyF_{date}_B{block}_instance*.csv"
        trigger_files = glob.glob(trigger_pattern)
        files['triggers'] = trigger_files
        
        # Look for sorted result files
        sorted_dir = f"{data_dir}/sorted_result/{date}/Block_{block}"
        if os.path.exists(sorted_dir):
            files['sorted_data'] = sorted_dir
        
        # Look for combined results
        combined_dir = f"{data_dir}/sorted_result_combined/{date}"
        if os.path.exists(combined_dir):
            files['combined_data'] = combined_dir
        
        # Look for firing rate matrices
        firing_rate_dir = f"{data_dir}/sorted_result_combined/firing_rate_matrices"
        if os.path.exists(firing_rate_dir):
            files['firing_rates'] = firing_rate_dir
        
        return files
    
    def load_trigger_data(self, trigger_file: str) -> pd.DataFrame:
        """Load trigger data from CSV file"""
        try:
            df = pd.read_csv(trigger_file)
            logger.info(f"Loaded trigger data: {len(df)} events")
            return df
        except Exception as e:
            logger.error(f"Error loading trigger file {trigger_file}: {e}")
            return None
    
    def load_firing_rate_data(self, firing_rate_dir: str, date: str) -> Dict:
        """Load firing rate matrices"""
        try:
            firing_rate_file = f"{firing_rate_dir}/firing_rate_matrices_{date}_combined.npz"
            if os.path.exists(firing_rate_file):
                data = np.load(firing_rate_file)
                logger.info(f"Loaded firing rate data: {list(data.keys())}")
                return dict(data)
            else:
                logger.warning(f"Firing rate file not found: {firing_rate_file}")
                return None
        except Exception as e:
            logger.error(f"Error loading firing rate data: {e}")
            return None
    
    def load_cluster_info(self, combined_dir: str) -> Dict:
        """Load cluster information"""
        cluster_files = glob.glob(f"{combined_dir}/cluster_inf_*.csv")
        cluster_info = {}
        
        for file in cluster_files:
            try:
                df = pd.read_csv(file)
                # Extract hub and instance from filename
                filename = os.path.basename(file)
                parts = filename.replace('cluster_inf_', '').replace('.csv', '').split('_')
                if len(parts) >= 3:
                    key = f"{parts[0]}_{parts[1]}_{parts[2]}"
                    cluster_info[key] = df
                    logger.info(f"Loaded cluster info for {key}: {len(df)} clusters")
            except Exception as e:
                logger.error(f"Error loading cluster info from {file}: {e}")
        
        return cluster_info
    
    def load_spike_info(self, combined_dir: str) -> Dict:
        """Load spike information"""
        spike_files = glob.glob(f"{combined_dir}/spike_inf_*.csv")
        spike_info = {}
        
        for file in spike_files:
            try:
                df = pd.read_csv(file)
                # Extract hub and instance from filename
                filename = os.path.basename(file)
                parts = filename.replace('spike_inf_', '').replace('.csv', '').split('_')
                if len(parts) >= 3:
                    key = f"{parts[0]}_{parts[1]}_{parts[2]}"
                    spike_info[key] = df
                    logger.info(f"Loaded spike info for {key}: {len(df)} spikes")
            except Exception as e:
                logger.error(f"Error loading spike info from {file}: {e}")
        
        return spike_info
    
    def synchronize_stimulus_responses(self, trigger_data: pd.DataFrame, 
                                     firing_rate_data: Dict,
                                     cluster_info: Dict) -> Dict:
        """Synchronize stimulus presentation with neural responses"""
        
        # Extract stimulus information from trigger data
        stimulus_times = trigger_data['time'].values if 'time' in trigger_data.columns else None
        stimulus_images = trigger_data['image_id'].values if 'image_id' in trigger_data.columns else None
        
        if stimulus_times is None:
            logger.error("No time information found in trigger data")
            return None
        
        # Create time axis for analysis
        time_axis = np.arange(-self.config.pre_trial * 1000, 
                            self.config.post_trial * 1000, 
                            self.config.downsample_factor / self.config.fs * 1000)
        
        synchronized_data = {
            'stimulus_times': stimulus_times,
            'stimulus_images': stimulus_images,
            'time_axis': time_axis,
            'firing_rates': firing_rate_data,
            'cluster_info': cluster_info,
            'config': self.config,
            'metadata': {
                'num_trials': len(stimulus_times),
                'trial_length': self.config.trial_length,
                'pre_trial': self.config.pre_trial,
                'post_trial': self.config.post_trial
            }
        }
        
        return synchronized_data
    
    def analyze_responses(self, synchronized_data: Dict) -> Dict:
        """Analyze neural responses to stimuli"""
        
        firing_rates = synchronized_data['firing_rates']
        stimulus_images = synchronized_data['stimulus_images']
        
        if firing_rates is None or stimulus_images is None:
            logger.error("Missing firing rate or stimulus data")
            return None
        
        # Extract firing rate matrices
        if 'firing_rate_matrices' in firing_rates:
            fr_matrices = firing_rates['firing_rate_matrices']
            
            # Analyze responses by image
            unique_images = np.unique(stimulus_images)
            image_responses = {}
            
            for img_id in unique_images:
                img_mask = stimulus_images == img_id
                img_responses = fr_matrices[img_mask, :, :]  # [trials, channels, timepoints]
                
                image_responses[img_id] = {
                    'mean_response': np.mean(img_responses, axis=0),
                    'std_response': np.std(img_responses, axis=0),
                    'num_trials': np.sum(img_mask),
                    'trial_responses': img_responses
                }
            
            analysis_results = {
                'image_responses': image_responses,
                'unique_images': unique_images,
                'total_images': len(unique_images),
                'response_statistics': self._calculate_response_stats(fr_matrices)
            }
            
            return analysis_results
        
        return None
    
    def _calculate_response_stats(self, fr_matrices: np.ndarray) -> Dict:
        """Calculate response statistics"""
        return {
            'mean_firing_rate': np.mean(fr_matrices),
            'std_firing_rate': np.std(fr_matrices),
            'max_firing_rate': np.max(fr_matrices),
            'min_firing_rate': np.min(fr_matrices),
            'response_range': np.max(fr_matrices) - np.min(fr_matrices)
        }
    
    def save_results(self, synchronized_data: Dict, analysis_results: Dict, 
                    output_dir: str, date: str, block: int):
        """Save synchronization and analysis results"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save synchronized data
        sync_file = f"{output_dir}/synchronized_data_{date}_B{block}.npz"
        np.savez(sync_file, **synchronized_data)
        logger.info(f"Saved synchronized data to {sync_file}")
        
        # Save analysis results
        if analysis_results is not None:
            analysis_file = f"{output_dir}/analysis_results_{date}_B{block}.npz"
            np.savez(analysis_file, **analysis_results)
            logger.info(f"Saved analysis results to {analysis_file}")
        
        # Save metadata as JSON
        metadata = {
            'date': date,
            'block': block,
            'config': {
                'trial_length': self.config.trial_length,
                'pre_trial': self.config.pre_trial,
                'post_trial': self.config.post_trial,
                'fs': self.config.fs,
                'downsample_factor': self.config.downsample_factor
            },
            'metadata': synchronized_data['metadata']
        }
        
        metadata_file = f"{output_dir}/metadata_{date}_B{block}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to {metadata_file}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Synchronize monkey electrophysiology data')
    parser.add_argument('--data_dir', type=str, default='/media/ubuntu/sda/Monkey',
                       help='Path to data directory')
    parser.add_argument('--date', type=str, required=True,
                       help='Date string (e.g., 20240112)')
    parser.add_argument('--block', type=int, required=True,
                       help='Block number')
    parser.add_argument('--output_dir', type=str, default='./sync_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Initialize synchronizer
    config = MonkeySyncConfig()
    sync = MonkeyDataSynchronizer(config)
    
    logger.info(f"Processing data for date {args.date}, block {args.block}")
    
    # Find data files
    files = sync.find_data_files(args.data_dir, args.date, args.block)
    logger.info(f"Found files: {list(files.keys())}")
    
    # Load trigger data
    trigger_data = None
    if 'triggers' in files and files['triggers']:
        trigger_data = sync.load_trigger_data(files['triggers'][0])
    
    # Load firing rate data
    firing_rate_data = None
    if 'firing_rates' in files:
        firing_rate_data = sync.load_firing_rate_data(files['firing_rates'], args.date)
    
    # Load cluster info
    cluster_info = {}
    if 'combined_data' in files:
        cluster_info = sync.load_cluster_info(files['combined_data'])
    
    # Perform synchronization
    if trigger_data is not None:
        synchronized_data = sync.synchronize_stimulus_responses(
            trigger_data, firing_rate_data, cluster_info
        )
        
        if synchronized_data is not None:
            # Analyze responses
            analysis_results = sync.analyze_responses(synchronized_data)
            
            # Save results
            sync.save_results(synchronized_data, analysis_results, 
                           args.output_dir, args.date, args.block)
            
            logger.info("Synchronization completed successfully!")
        else:
            logger.error("Synchronization failed!")
    else:
        logger.error("No trigger data found!")

if __name__ == "__main__":
    main()
