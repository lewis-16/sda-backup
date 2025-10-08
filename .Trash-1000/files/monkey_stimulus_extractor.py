#!/usr/bin/env python3
"""
Monkey Stimulus Matrix Extractor
================================

Extract stimulus timing and image information from monkey experiment logs.
Output: Matrix with [start_time, end_time, image_id] for each stimulus.

Usage:
    python monkey_stimulus_extractor.py [log_file_path]
    
If no log file is specified, it will process all available log files.
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

def create_stimulus_matrix(log_data, stimulus_duration=0.2, trial_interval=0.4):
    """
    Create stimulus matrix from log data
    
    Args:
        log_data: DataFrame with stimulus log
        stimulus_duration: Duration of each stimulus in seconds (default: 0.2)
        trial_interval: Interval between trials in seconds (default: 0.4)
        
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
    
    # Get image IDs (use train_pic for image identification)
    image_ids = correct_trials['train_pic'].values
    
    # Create stimulus matrix: [start_time, end_time, image_id]
    stimulus_matrix = np.column_stack([start_times, end_times, image_ids])
    
    return stimulus_matrix

def process_log_file(log_path, output_dir=None):
    """Process a single log file"""
    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(log_path)}")
    print(f"{'='*60}")
    
    # Load log data
    log_data = load_stimulus_log(log_path)
    if log_data is None:
        print("Failed to load log data")
        return None
    
    print(f"Log data shape: {log_data.shape}")
    print(f"Columns: {list(log_data.columns)}")
    
    # Create stimulus matrix
    stimulus_matrix = create_stimulus_matrix(log_data)
    
    if stimulus_matrix is not None:
        print(f"\nStimulus Matrix Shape: {stimulus_matrix.shape}")
        print("Format: [start_time, end_time, image_id]")
        
        # Show first few stimuli
        print("\nFirst 10 stimuli:")
        print("Start(s)  End(s)    Image_ID")
        print("-" * 30)
        for i in range(min(10, len(stimulus_matrix))):
            start, end, img_id = stimulus_matrix[i]
            print(f"{start:8.3f} {end:8.3f} {img_id:8.0f}")
        
        if len(stimulus_matrix) > 10:
            print(f"... and {len(stimulus_matrix) - 10} more stimuli")
        
        # Determine output directory
        if output_dir is None:
            output_dir = os.path.dirname(log_path)
        
        # Create output filename
        base_name = os.path.basename(log_path).replace('.mat', '')
        txt_file = os.path.join(output_dir, f"{base_name}_stimulus_matrix.txt")
        csv_file = os.path.join(output_dir, f"{base_name}_stimulus_matrix.csv")
        
        # Save as text file
        np.savetxt(txt_file, stimulus_matrix, 
                  header='start_time end_time image_id', 
                  fmt='%.6f %.6f %.0f')
        print(f"\nResults saved to: {txt_file}")
        
        # Save as CSV file
        df = pd.DataFrame(stimulus_matrix, columns=['start_time', 'end_time', 'image_id'])
        df.to_csv(csv_file, index=False)
        print(f"Results also saved as CSV: {csv_file}")
        
        # Show statistics
        print(f"\nStatistics:")
        print(f"  Total stimuli: {len(stimulus_matrix)}")
        print(f"  Unique images: {len(np.unique(stimulus_matrix[:, 2]))}")
        print(f"  Time range: {stimulus_matrix[0, 0]:.3f} - {stimulus_matrix[-1, 1]:.3f} seconds")
        print(f"  Total duration: {stimulus_matrix[-1, 1] - stimulus_matrix[0, 0]:.3f} seconds")
        
        return stimulus_matrix
        
    else:
        print("Failed to create stimulus matrix")
        return None

def find_log_files(data_dir):
    """Find all THINGS log files"""
    logs_dir = os.path.join(data_dir, "TVSD", "monkeyF", "_logs")
    
    if not os.path.exists(logs_dir):
        print(f"Logs directory not found: {logs_dir}")
        return []
    
    # Look for THINGS log files (not RANDTAB files)
    log_files = glob.glob(os.path.join(logs_dir, "THINGS_monkeyF_*.mat"))
    return sorted(log_files)

def main():
    """Main function"""
    if len(sys.argv) == 2:
        # Process specific log file
        log_path = sys.argv[1]
        if not os.path.exists(log_path):
            print(f"Error: File {log_path} not found")
            return
        
        process_log_file(log_path)
        
    else:
        # Process all available log files
        data_dir = "/media/ubuntu/sda/Monkey"
        log_files = find_log_files(data_dir)
        
        if not log_files:
            print("No THINGS log files found")
            return
        
        print(f"Found {len(log_files)} THINGS log files:")
        for i, log_file in enumerate(log_files):
            print(f"{i+1}: {os.path.basename(log_file)}")
        
        print(f"\nProcessing all {len(log_files)} files...")
        
        results = []
        for log_file in log_files:
            result = process_log_file(log_file)
            if result is not None:
                results.append((os.path.basename(log_file), result))
        
        # Summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Successfully processed {len(results)} files:")
        for filename, matrix in results:
            print(f"  {filename}: {matrix.shape[0]} stimuli, {len(np.unique(matrix[:, 2]))} unique images")

if __name__ == "__main__":
    main()
