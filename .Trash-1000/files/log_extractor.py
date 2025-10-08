#!/usr/bin/env python3
"""
Stimulus Log Extractor
======================

Extract stimulus timing and image information from log files.
This script works with the existing MATLAB log files.

Usage:
    python log_extractor.py /media/ubuntu/sda/Monkey/TVSD/monkeyF/_logs/THINGS_monkeyF_20240112_B1.mat
"""

import numpy as np
import pandas as pd
import os
import sys
import glob

def load_stimulus_log(log_path):
    """Load stimulus log from MATLAB file"""
    try:
        import scipy.io
        data = scipy.io.loadmat(log_path)
        
        if 'MAT' in data:
            mat_data = data['MAT']
            # MAT format: [#trial #train_pic #test_pic #pic_rep #ncount #correct]
            df = pd.DataFrame(mat_data, columns=[
                'trial', 'train_pic', 'test_pic', 'pic_rep', 'ncount', 'correct'
            ])
            return df
        else:
            print(f"No MAT data found in {log_path}")
            return None
            
    except Exception as e:
        print(f"Error loading log file {log_path}: {e}")
        return None

def create_stimulus_matrix_from_log(log_data, stimulus_duration=0.3, trial_interval=2.0):
    """
    Create stimulus matrix from log data
    
    Args:
        log_data: DataFrame with stimulus log
        stimulus_duration: Duration of each stimulus in seconds
        trial_interval: Interval between trials in seconds
        
    Returns:
        Matrix with [start_time, end_time, image_id] for each stimulus
    """
    
    # Filter for correct trials only
    correct_trials = log_data[log_data['correct'] > 0].copy()
    
    print(f"Total trials: {len(log_data)}")
    print(f"Correct trials: {len(correct_trials)}")
    
    # Calculate start times (assuming trials start at 0 and are spaced by trial_interval)
    start_times = np.arange(len(correct_trials)) * trial_interval
    end_times = start_times + stimulus_duration
    
    # Get image IDs
    image_ids = correct_trials['train_pic'].values
    
    # Create stimulus matrix: [start_time, end_time, image_id]
    stimulus_matrix = np.column_stack([start_times, end_times, image_ids])
    
    return stimulus_matrix

def find_log_files(data_dir):
    """Find all THINGS log files"""
    logs_dir = os.path.join(data_dir, "TVSD", "monkeyF", "_logs")
    
    if not os.path.exists(logs_dir):
        print(f"Logs directory not found: {logs_dir}")
        return []
    
    log_files = glob.glob(os.path.join(logs_dir, "*THINGS*monkeyF*B*.mat"))
    return sorted(log_files)

def main():
    """Main function"""
    if len(sys.argv) == 2:
        log_path = sys.argv[1]
    else:
        # Default to finding log files
        data_dir = "/media/ubuntu/sda/Monkey"
        log_files = find_log_files(data_dir)
        
        if not log_files:
            print("No log files found")
            return
        
        print("Available log files:")
        for i, log_file in enumerate(log_files):
            print(f"{i+1}: {os.path.basename(log_file)}")
        
        # Use the first one for demonstration
        log_path = log_files[0]
        print(f"\nUsing: {log_path}")
    
    # Load log data
    log_data = load_stimulus_log(log_path)
    if log_data is None:
        print("Failed to load log data")
        return
    
    print(f"\nLog data shape: {log_data.shape}")
    print(f"Columns: {list(log_data.columns)}")
    print("\nFirst 10 trials:")
    print(log_data.head(10))
    
    # Create stimulus matrix
    stimulus_matrix = create_stimulus_matrix_from_log(log_data)
    
    if stimulus_matrix is not None:
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
        output_file = log_path.replace('.mat', '_stimulus_matrix.txt')
        np.savetxt(output_file, stimulus_matrix, 
                  header='start_time end_time image_id', 
                  fmt='%.6f %.6f %.0f')
        print(f"\nResults saved to: {output_file}")
        
        # Also save as CSV for easier viewing
        csv_file = log_path.replace('.mat', '_stimulus_matrix.csv')
        df = pd.DataFrame(stimulus_matrix, columns=['start_time', 'end_time', 'image_id'])
        df.to_csv(csv_file, index=False)
        print(f"Results also saved as CSV: {csv_file}")
        
        # Show some statistics
        print(f"\nStatistics:")
        print(f"  Total stimuli: {len(stimulus_matrix)}")
        print(f"  Unique images: {len(np.unique(stimulus_matrix[:, 2]))}")
        print(f"  Time range: {stimulus_matrix[0, 0]:.3f} - {stimulus_matrix[-1, 1]:.3f} seconds")
        print(f"  Total duration: {stimulus_matrix[-1, 1] - stimulus_matrix[0, 0]:.3f} seconds")
        
    else:
        print("Failed to create stimulus matrix")

if __name__ == "__main__":
    main()
