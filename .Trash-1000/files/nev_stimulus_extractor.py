#!/usr/bin/env python3
"""
Simple NEV Stimulus Extractor
============================

Extract stimulus timing and image information from NEV files.
Output: Matrix with [start_time, end_time, image_id] for each stimulus.

Usage:
    python nev_stimulus_extractor.py /path/to/file.nev
"""

import numpy as np
import pandas as pd
import os
import sys
import argparse
from typing import Dict, List, Tuple, Optional

try:
    from scipy.io import loadmat
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Please install: pip install scipy")

def read_nev_file(nev_path: str) -> Optional[Dict]:
    """
    Read NEV file and extract stimulus information
    
    Args:
        nev_path: Path to NEV file
        
    Returns:
        Dictionary containing stimulus timestamps and metadata
    """
    if not os.path.exists(nev_path):
        print(f"Error: File {nev_path} not found")
        return None
    
    try:
        # Try to read as MATLAB .mat file (if converted)
        if nev_path.endswith('.mat'):
            nev_data = loadmat(nev_path)
            return parse_mat_nev(nev_data)
        else:
            # For actual NEV files, try to read directly
            print(f"Reading NEV file: {nev_path}")
            return read_raw_nev(nev_path)
            
    except Exception as e:
        print(f"Error reading NEV file {nev_path}: {e}")
        return None

def parse_mat_nev(nev_data: Dict) -> Optional[Dict]:
    """Parse NEV data from MATLAB .mat file"""
    try:
        # Extract digital I/O data
        digital_io = nev_data.get('EVENT', {}).get('Data', {}).get('SerialDigitalIO', {})
        
        if not digital_io:
            print("No SerialDigitalIO data found in NEV file")
            return None
        
        # Find stimulus events (assuming bit 0 = stimulus marker)
        unparsed_data = digital_io.get('UnparsedData', [])
        timestamps = digital_io.get('TimeStamp', [])
        
        # Find events matching stimulus bit (2^0 = 1)
        stim_mask = (unparsed_data & 1) > 0
        stim_timestamps = timestamps[stim_mask]
        
        return {
            'stimulus_timestamps': stim_timestamps,
            'total_events': len(timestamps),
            'stimulus_events': len(stim_timestamps),
            'sample_rate': nev_data.get('MetaTags', {}).get('SamplingFreq', 30000)
        }
        
    except Exception as e:
        print(f"Error parsing NEV data: {e}")
        return None

def read_raw_nev(nev_path: str) -> Optional[Dict]:
    """
    Read raw NEV file (simplified version)
    This is a placeholder - you may need to install a proper NEV reader library
    """
    print("Warning: Direct NEV file reading not fully implemented.")
    print("Please convert NEV file to .mat format using MATLAB or use a proper NEV reader library.")
    return None

def load_stimulus_log(log_path: str) -> Optional[pd.DataFrame]:
    """
    Load stimulus presentation log
    
    Args:
        log_path: Path to log file
        
    Returns:
        DataFrame with stimulus information
    """
    if not os.path.exists(log_path):
        print(f"Warning: Log file {log_path} not found")
        return None
    
    try:
        if log_path.endswith('.mat'):
            # Load MATLAB file
            log_data = loadmat(log_path)
            mat_data = log_data.get('MAT', [])
            
            if mat_data.size == 0:
                print("No MAT data found in log file")
                return None
            
            # MAT format: [#trial #train_pic #test_pic #pic_rep #ncount #correct]
            df = pd.DataFrame(mat_data, columns=[
                'trial', 'train_pic', 'test_pic', 'pic_rep', 'ncount', 'correct'
            ])
            return df
            
        elif log_path.endswith('.csv'):
            # Load CSV file
            df = pd.read_csv(log_path)
            return df
            
        else:
            print(f"Unsupported log file format: {log_path}")
            return None
            
    except Exception as e:
        print(f"Error loading log file {log_path}: {e}")
        return None

def create_stimulus_matrix(nev_data: Dict, log_data: Optional[pd.DataFrame] = None, 
                          stimulus_duration: float = 0.3) -> np.ndarray:
    """
    Create stimulus matrix with [start_time, end_time, image_id]
    
    Args:
        nev_data: NEV data dictionary
        log_data: Stimulus log data
        stimulus_duration: Duration of each stimulus in seconds
        
    Returns:
        Matrix with [start_time, end_time, image_id] for each stimulus
    """
    stim_timestamps = nev_data['stimulus_timestamps']
    sample_rate = nev_data['sample_rate']
    
    # Convert timestamps to seconds
    start_times = stim_timestamps / sample_rate
    
    # Calculate end times
    end_times = start_times + stimulus_duration
    
    # Get image IDs
    if log_data is not None:
        # Use log data for image IDs
        if 'train_pic' in log_data.columns:
            image_ids = log_data['train_pic'].values
        elif 'image_id' in log_data.columns:
            image_ids = log_data['image_id'].values
        else:
            # Use trial numbers as image IDs
            image_ids = np.arange(len(stim_timestamps)) + 1
    else:
        # Use trial numbers as image IDs
        image_ids = np.arange(len(stim_timestamps)) + 1
    
    # Ensure we have the right number of image IDs
    if len(image_ids) != len(stim_timestamps):
        print(f"Warning: Image ID count ({len(image_ids)}) doesn't match stimulus count ({len(stim_timestamps)})")
        # Pad or truncate as needed
        if len(image_ids) > len(stim_timestamps):
            image_ids = image_ids[:len(stim_timestamps)]
        else:
            image_ids = np.pad(image_ids, (0, len(stim_timestamps) - len(image_ids)), 'constant')
    
    # Create matrix: [start_time, end_time, image_id]
    stimulus_matrix = np.column_stack([start_times, end_times, image_ids])
    
    return stimulus_matrix

def find_log_file(nev_path: str) -> Optional[str]:
    """
    Find corresponding log file for NEV file
    
    Args:
        nev_path: Path to NEV file
        
    Returns:
        Path to log file if found
    """
    # Extract date and block from NEV path
    nev_dir = os.path.dirname(nev_path)
    nev_filename = os.path.basename(nev_path)
    
    # Look for log files in the same directory or parent directories
    search_dirs = [nev_dir, os.path.dirname(nev_dir), os.path.dirname(os.path.dirname(nev_dir))]
    
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue
            
        # Look for .mat files that might contain stimulus logs
        for file in os.listdir(search_dir):
            if file.endswith('.mat') and 'THINGS' in file:
                return os.path.join(search_dir, file)
    
    return None

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Extract stimulus information from NEV files')
    parser.add_argument('nev_file', help='Path to NEV file')
    parser.add_argument('--log_file', help='Path to stimulus log file (optional)')
    parser.add_argument('--duration', type=float, default=0.3, 
                       help='Stimulus duration in seconds (default: 0.3)')
    parser.add_argument('--output', help='Output file path (optional)')
    
    args = parser.parse_args()
    
    print(f"Processing NEV file: {args.nev_file}")
    
    # Read NEV file
    nev_data = read_nev_file(args.nev_file)
    if nev_data is None:
        print("Failed to read NEV file")
        return
    
    print(f"Found {nev_data['stimulus_events']} stimulus events")
    print(f"Sample rate: {nev_data['sample_rate']} Hz")
    
    # Load log file if not specified
    log_data = None
    if args.log_file:
        log_data = load_stimulus_log(args.log_file)
    else:
        # Try to find log file automatically
        log_file = find_log_file(args.nev_file)
        if log_file:
            print(f"Found log file: {log_file}")
            log_data = load_stimulus_log(log_file)
    
    if log_data is not None:
        print(f"Loaded log data: {len(log_data)} trials")
        print(f"Log columns: {list(log_data.columns)}")
    
    # Create stimulus matrix
    stimulus_matrix = create_stimulus_matrix(nev_data, log_data, args.duration)
    
    print(f"\nStimulus Matrix Shape: {stimulus_matrix.shape}")
    print("Format: [start_time, end_time, image_id]")
    print("\nFirst 10 stimuli:")
    print("Start(s)  End(s)    Image_ID")
    print("-" * 30)
    for i in range(min(10, len(stimulus_matrix))):
        start, end, img_id = stimulus_matrix[i]
        print(f"{start:8.3f} {end:8.3f} {img_id:8.0f}")
    
    if len(stimulus_matrix) > 10:
        print(f"... and {len(stimulus_matrix) - 10} more stimuli")
    
    # Save results
    if args.output:
        np.savetxt(args.output, stimulus_matrix, 
                  header='start_time end_time image_id', 
                  fmt='%.6f %.6f %.0f')
        print(f"\nResults saved to: {args.output}")
    else:
        # Save to default location
        output_file = args.nev_file.replace('.nev', '_stimulus_matrix.txt')
        np.savetxt(output_file, stimulus_matrix, 
                  header='start_time end_time image_id', 
                  fmt='%.6f %.6f %.0f')
        print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()
