#!/usr/bin/env python3
"""
Stimulus-Electrophysiology Synchronization Script
=================================================

This script implements the synchronization mechanism between image stimulus 
sequences and electrophysiological data recording timepoints, based on the 
MATLAB code analysis from TVSD/_code directory.

Author: AI Assistant
Date: 2024
"""

import numpy as np
import os
import glob
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
import warnings

try:
    import scipy.io as sio
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
class SyncConfig:
    """Configuration parameters for synchronization"""
    # Timing parameters
    trial_length: float = 0.3  # seconds
    pre_trial: float = 0.1     # seconds
    post_trial: float = 0.2    # seconds (trial_length - pre_trial)
    
    # Sampling parameters
    fs: int = 30000  # Hz
    downsample_factor: int = 30
    
    # Filter parameters
    bandpass_freq: Tuple[float, float] = (500, 5000)  # Hz
    lowpass_freq: float = 200  # Hz
    filter_order: int = 2
    
    # Photodiode parameters
    photodiode_threshold: float = 1e4
    photodiode_channel: int = 15  # Channel for photodiode signal
    
    # Stimulus parameters
    stimbit: int = 1  # Digital I/O bit for stimulus markers
    
    # Noise removal
    line_noise_freqs: List[int] = [50, 100, 150, 200, 250]  # Hz and harmonics
    screen_refresh_freqs: List[int] = [60, 120, 180, 240, 300]  # Hz and harmonics

class StimulusElectrophysiologySync:
    """
    Main class for synchronizing stimulus presentation with electrophysiological recordings
    """
    
    def __init__(self, config: SyncConfig = None):
        self.config = config or SyncConfig()
        self._setup_filters()
        
    def _setup_filters(self):
        """Setup digital filters for signal processing"""
        nyquist = self.config.fs / 2
        
        # Bandpass filter for MUA extraction
        self.bandpass_b, self.bandpass_a = signal.butter(
            self.config.filter_order,
            [self.config.bandpass_freq[0]/nyquist, self.config.bandpass_freq[1]/nyquist],
            btype='band'
        )
        
        # Lowpass filter for MUA smoothing
        self.lowpass_b, self.lowpass_a = signal.butter(
            self.config.filter_order,
            self.config.lowpass_freq/nyquist,
            btype='low'
        )
        
        # Notch filters for noise removal
        self.notch_filters = {}
        for freq in self.config.line_noise_freqs + self.config.screen_refresh_freqs:
            self.notch_filters[freq] = signal.iirnotch(freq, Q=30, fs=self.config.fs)
    
    def read_nev_file(self, nev_path: str) -> Dict:
        """
        Read NEV file and extract stimulus timestamps
        
        Args:
            nev_path: Path to NEV file
            
        Returns:
            Dictionary containing stimulus timestamps and metadata
        """
        try:
            # Try to read as MATLAB .mat file first (if converted)
            if nev_path.endswith('.mat'):
                nev_data = loadmat(nev_path)
                return self._parse_mat_nev(nev_data)
            else:
                # For actual NEV files, you would need a NEV reader library
                # This is a placeholder for the actual implementation
                logger.warning("Direct NEV file reading not implemented. Please convert to .mat format.")
                return None
                
        except Exception as e:
            logger.error(f"Error reading NEV file {nev_path}: {e}")
            return None
    
    def _parse_mat_nev(self, nev_data: Dict) -> Dict:
        """Parse NEV data from MATLAB .mat file"""
        try:
            # Extract digital I/O data
            digital_io = nev_data.get('EVENT', {}).get('Data', {}).get('SerialDigitalIO', {})
            
            if not digital_io:
                logger.error("No SerialDigitalIO data found in NEV file")
                return None
            
            # Find stimulus events
            unparsed_data = digital_io.get('UnparsedData', [])
            timestamps = digital_io.get('TimeStamp', [])
            
            # Find events matching stimulus bit
            stim_mask = (unparsed_data & (2**self.config.stimbit)) > 0
            stim_timestamps = timestamps[stim_mask]
            
            return {
                'stimulus_timestamps': stim_timestamps,
                'total_events': len(timestamps),
                'stimulus_events': len(stim_timestamps),
                'sample_rate': nev_data.get('MetaTags', {}).get('SamplingFreq', self.config.fs)
            }
            
        except Exception as e:
            logger.error(f"Error parsing NEV data: {e}")
            return None
    
    def read_ns6_file(self, ns6_path: str) -> Dict:
        """
        Read NS6 file and extract electrophysiological data
        
        Args:
            ns6_path: Path to NS6 file
            
        Returns:
            Dictionary containing raw data and metadata
        """
        try:
            # Try to read as MATLAB .mat file first (if converted)
            if ns6_path.endswith('.mat'):
                ns6_data = loadmat(ns6_path)
                return self._parse_mat_ns6(ns6_data)
            else:
                # For actual NS6 files, you would need an NS6 reader library
                logger.warning("Direct NS6 file reading not implemented. Please convert to .mat format.")
                return None
                
        except Exception as e:
            logger.error(f"Error reading NS6 file {ns6_path}: {e}")
            return None
    
    def _parse_mat_ns6(self, ns6_data: Dict) -> Dict:
        """Parse NS6 data from MATLAB .mat file"""
        try:
            # Extract raw data
            raw_data = ns6_data.get('RAW', {}).get('Data', [])
            
            if raw_data.size == 0:
                logger.error("No data found in NS6 file")
                return None
            
            return {
                'data': raw_data,
                'sample_rate': ns6_data.get('RAW', {}).get('MetaTags', {}).get('SamplingFreq', self.config.fs),
                'num_channels': raw_data.shape[0] if len(raw_data.shape) > 1 else 1,
                'num_samples': raw_data.shape[1] if len(raw_data.shape) > 1 else len(raw_data)
            }
            
        except Exception as e:
            logger.error(f"Error parsing NS6 data: {e}")
            return None
    
    def detect_photodiode_onsets(self, photodiode_signal: np.ndarray, 
                               stimulus_timestamps: np.ndarray) -> np.ndarray:
        """
        Detect photodiode onsets for each stimulus presentation
        
        Args:
            photodiode_signal: Photodiode signal data
            stimulus_timestamps: Timestamps of stimulus events
            
        Returns:
            Array of photodiode onset delays for each trial
        """
        dio_onsets = np.zeros(len(stimulus_timestamps))
        
        for i, stim_time in enumerate(stimulus_timestamps):
            # Extract signal around stimulus time
            start_idx = int(stim_time)
            end_idx = int(stim_time + self.config.fs * self.config.trial_length)
            
            if end_idx > len(photodiode_signal):
                logger.warning(f"Trial {i}: Signal too short, using available data")
                end_idx = len(photodiode_signal)
            
            trial_signal = photodiode_signal[start_idx:end_idx]
            
            # Find first threshold crossing
            threshold_crossings = np.where(np.abs(trial_signal) > self.config.photodiode_threshold)[0]
            
            if len(threshold_crossings) > 0:
                dio_onsets[i] = threshold_crossings[0]
            else:
                logger.warning(f"Trial {i}: No photodiode onset detected")
                dio_onsets[i] = 0
        
        return dio_onsets
    
    def extract_mua(self, raw_signal: np.ndarray) -> np.ndarray:
        """
        Extract Multi-Unit Activity (MUA) from raw signal
        
        Args:
            raw_signal: Raw electrophysiological signal
            
        Returns:
            Processed MUA signal
        """
        # Convert to double precision
        signal_double = raw_signal.astype(np.float64)
        
        # Bandpass filter
        filtered_signal = signal.filtfilt(self.bandpass_b, self.bandpass_a, signal_double)
        
        # Rectify (absolute value)
        rectified_signal = np.abs(filtered_signal)
        
        # Lowpass filter
        mua_signal = signal.filtfilt(self.lowpass_b, self.lowpass_a, rectified_signal)
        
        # Remove line noise and screen refresh artifacts
        for freq, (b, a) in self.notch_filters.items():
            mua_signal = signal.filtfilt(b, a, mua_signal)
        
        return mua_signal
    
    def extract_trial_data(self, mua_signal: np.ndarray, 
                          stimulus_timestamps: np.ndarray,
                          photodiode_onsets: np.ndarray,
                          valid_trials: np.ndarray = None) -> np.ndarray:
        """
        Extract MUA data for each trial aligned to stimulus onset
        
        Args:
            mua_signal: Processed MUA signal
            stimulus_timestamps: Timestamps of stimulus events
            photodiode_onsets: Photodiode onset delays
            valid_trials: Boolean array indicating valid trials
            
        Returns:
            3D array: [trials, timepoints] of MUA data
        """
        if valid_trials is None:
            valid_trials = np.ones(len(stimulus_timestamps), dtype=bool)
        
        # Calculate time window
        pre_samples = int(self.config.pre_trial * self.config.fs)
        post_samples = int(self.config.post_trial * self.config.fs)
        total_samples = pre_samples + post_samples
        
        # Downsampled timepoints
        downsample_factor = self.config.downsample_factor
        timepoints = int(total_samples / downsample_factor)
        
        # Initialize output array
        trial_data = np.full((np.sum(valid_trials), timepoints), np.nan)
        
        valid_idx = 0
        for i, (stim_time, dio_onset, is_valid) in enumerate(zip(stimulus_timestamps, photodiode_onsets, valid_trials)):
            if not is_valid:
                continue
            
            # Calculate trial start and end indices
            trial_start = int(stim_time + dio_onset - pre_samples)
            trial_end = int(stim_time + dio_onset + post_samples)
            
            # Check bounds
            if trial_start < 0 or trial_end > len(mua_signal):
                logger.warning(f"Trial {i}: Index out of bounds, skipping")
                continue
            
            # Extract trial data
            trial_signal = mua_signal[trial_start:trial_end]
            
            # Downsample
            downsampled_signal = signal.decimate(trial_signal, downsample_factor, ftype='iir')
            
            # Store in output array
            trial_data[valid_idx, :len(downsampled_signal)] = downsampled_signal
            valid_idx += 1
        
        return trial_data
    
    def load_stimulus_log(self, log_path: str) -> Dict:
        """
        Load stimulus presentation log
        
        Args:
            log_path: Path to log file
            
        Returns:
            Dictionary containing stimulus information
        """
        try:
            log_data = loadmat(log_path)
            mat_data = log_data.get('MAT', [])
            
            if mat_data.size == 0:
                logger.error("No MAT data found in log file")
                return None
            
            # MAT format: [#trial #train_pic #test_pic #pic_rep #ncount #correct]
            return {
                'trial_info': mat_data,
                'num_trials': len(mat_data),
                'valid_trials': mat_data[:, -1] > 0,  # Last column indicates correctness
                'train_images': mat_data[:, 1],
                'test_images': mat_data[:, 2],
                'image_repetitions': mat_data[:, 3]
            }
            
        except Exception as e:
            logger.error(f"Error loading stimulus log {log_path}: {e}")
            return None
    
    def synchronize_data(self, nev_path: str, ns6_path: str, log_path: str) -> Dict:
        """
        Main synchronization function
        
        Args:
            nev_path: Path to NEV file
            ns6_path: Path to NS6 file  
            log_path: Path to stimulus log file
            
        Returns:
            Dictionary containing synchronized data
        """
        logger.info("Starting data synchronization...")
        
        # Load stimulus timestamps
        nev_data = self.read_nev_file(nev_path)
        if nev_data is None:
            return None
        
        # Load electrophysiological data
        ns6_data = self.read_ns6_file(ns6_path)
        if ns6_data is None:
            return None
        
        # Load stimulus log
        log_data = self.load_stimulus_log(log_path)
        if log_data is None:
            return None
        
        # Validate trial counts
        stim_count = len(nev_data['stimulus_timestamps'])
        log_count = log_data['num_trials']
        
        if stim_count != log_count:
            logger.error(f"Trial count mismatch: NEV={stim_count}, Log={log_count}")
            return None
        
        logger.info(f"Found {stim_count} trials")
        
        # Extract photodiode signal
        photodiode_signal = ns6_data['data'][self.config.photodiode_channel - 1, :]
        
        # Detect photodiode onsets
        photodiode_onsets = self.detect_photodiode_onsets(
            photodiode_signal, 
            nev_data['stimulus_timestamps']
        )
        
        # Process each channel
        num_channels = ns6_data['num_channels']
        synchronized_data = {}
        
        for ch in range(num_channels):
            if ch == self.config.photodiode_channel - 1:  # Skip photodiode channel
                continue
                
            logger.info(f"Processing channel {ch + 1}/{num_channels}")
            
            # Extract MUA
            raw_signal = ns6_data['data'][ch, :]
            mua_signal = self.extract_mua(raw_signal)
            
            # Extract trial data
            trial_data = self.extract_trial_data(
                mua_signal,
                nev_data['stimulus_timestamps'],
                photodiode_onsets,
                log_data['valid_trials']
            )
            
            synchronized_data[f'channel_{ch + 1}'] = trial_data
        
        # Create time axis
        time_axis = np.arange(-self.config.pre_trial * 1000, 
                            self.config.post_trial * 1000, 
                            self.config.downsample_factor / self.config.fs * 1000)
        
        return {
            'synchronized_data': synchronized_data,
            'time_axis': time_axis,
            'stimulus_info': log_data,
            'photodiode_onsets': photodiode_onsets,
            'config': self.config,
            'metadata': {
                'num_trials': stim_count,
                'num_channels': num_channels,
                'sample_rate': ns6_data['sample_rate'],
                'downsample_factor': self.config.downsample_factor
            }
        }

def main():
    """Example usage of the synchronization script"""
    
    # Configuration
    config = SyncConfig(
        trial_length=0.3,
        pre_trial=0.1,
        post_trial=0.2,
        fs=30000,
        downsample_factor=30
    )
    
    # Initialize synchronizer
    sync = StimulusElectrophysiologySync(config)
    
    # Example file paths (adjust as needed)
    nev_path = "/path/to/your/file.nev"  # or .mat
    ns6_path = "/path/to/your/file.ns6"  # or .mat
    log_path = "/path/to/your/log.mat"
    
    # Perform synchronization
    result = sync.synchronize_data(nev_path, ns6_path, log_path)
    
    if result is not None:
        logger.info("Synchronization completed successfully!")
        logger.info(f"Processed {result['metadata']['num_trials']} trials")
        logger.info(f"Processed {result['metadata']['num_channels']} channels")
        
        # Save results
        output_path = "synchronized_data.npz"
        np.savez(output_path, **result)
        logger.info(f"Results saved to {output_path}")
    else:
        logger.error("Synchronization failed!")

if __name__ == "__main__":
    main()
