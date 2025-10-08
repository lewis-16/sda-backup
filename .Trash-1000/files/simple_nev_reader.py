#!/usr/bin/env python3
"""
Simple NEV Reader with brpylib
==============================

Simple script to read NEV files using brpylib and extract stimulus information.
Based on the user's working code example.

Usage:
    python simple_nev_reader.py
"""

import numpy as np
import pandas as pd
import os
import glob

def read_nev_with_brpylib(nev_path):
    """
    Read NEV file using brpylib and extract stimulus information
    
    Args:
        nev_path: Path to NEV file
        
    Returns:
        Dictionary with stimulus information
    """
    try:
        from brpylib import NevFile
        
        print(f"Opening NEV file: {nev_path}")
        file = NevFile(nev_path)
        
        # Get digital I/O events
        digital_events = file.getdata('digitalserial')
        
        if digital_events is None or len(digital_events) == 0:
            print("No digital events found")
            return None
        
        print(f"Found {len(digital_events)} digital events")
        
        # Extract timestamps and data
        timestamps = digital_events['TimeStamp']
        unparsed_data = digital_events['UnparsedData']
        
        # Find stimulus events (assuming bit 0 = stimulus marker)
        stim_mask = (unparsed_data & 1) > 0
        stim_timestamps = timestamps[stim_mask]
        
        print(f"Found {len(stim_timestamps)} stimulus events")
        
        # Get sample rate (usually 30000 Hz for NEV files)
        sample_rate = 30000  # Default sample rate
        
        return {
            'stimulus_timestamps': stim_timestamps,
            'sample_rate': sample_rate,
            'total_events': len(digital_events),
            'stimulus_events': len(stim_timestamps)
        }
        
    except Exception as e:
        print(f"Error reading NEV file: {e}")
        return None

def load_log_data(log_path):
    """Load stimulus log data"""
    try:
        import scipy.io
        data = scipy.io.loadmat(log_path)
        
        if 'MAT' in data:
            mat_data = data['MAT']
            df = pd.DataFrame(mat_data, columns=[
                'trial', 'train_pic', 'test_pic', 'pic_rep', 'ncount', 'correct'
            ])
            return df
        else:
            print(f"No MAT data found in {log_path}")
            return None
            
    except Exception as e:
        print(f"Error loading log file: {e}")
        return None

def create_stimulus_matrix(nev_data, log_data=None, stimulus_duration=0.2):
    """
    Create stimulus matrix from NEV and log data
    
    Args:
        nev_data: NEV data dictionary
        log_data: Optional log data DataFrame
        stimulus_duration: Duration of each stimulus in seconds
        
    Returns:
        Matrix with [start_time, end_time, image_id]
    """
    
    stim_timestamps = nev_data['stimulus_timestamps']
    sample_rate = nev_data['sample_rate']
    
    # Convert timestamps to seconds
    start_times = stim_timestamps / sample_rate
    end_times = start_times + stimulus_duration
    
    # Get image IDs
    if log_data is not None:
        # Filter for correct trials
        correct_trials = log_data[log_data['correct'] > 0]
        
        if len(correct_trials) == len(stim_timestamps):
            image_ids = correct_trials['train_pic'].values
            print(f"Using image IDs from log data: {len(image_ids)} images")
        else:
            print(f"Warning: Log data length ({len(correct_trials)}) doesn't match stimulus count ({len(stim_timestamps)})")
            image_ids = np.arange(len(stim_timestamps)) + 1
    else:
        image_ids = np.arange(len(stim_timestamps)) + 1
        print("Using trial numbers as image IDs")
    
    # Create stimulus matrix
    stimulus_matrix = np.column_stack([start_times, end_times, image_ids])
    
    return stimulus_matrix

def main():
    """Main function"""
    print("Simple NEV Reader with brpylib")
    print("=" * 40)
    
    # NEV file path
    nev_path = "/media/ubuntu/sda/Monkey/TVSD/monkeyF/20240112/Block_1/NSP-instance1_B001.nev"
    
    # Check if file exists
    if not os.path.exists(nev_path):
        print(f"Error: NEV file not found: {nev_path}")
        return
    
    # Read NEV file
    nev_data = read_nev_with_brpylib(nev_path)
    if nev_data is None:
        print("Failed to read NEV file")
        return
    
    print(f"Sample rate: {nev_data['sample_rate']} Hz")
    print(f"Stimulus events: {nev_data['stimulus_events']}")
    
    # Try to find corresponding log file
    log_path = "/media/ubuntu/sda/Monkey/TVSD/monkeyF/_logs/THINGS_monkeyF_20240112_B1.mat"
    log_data = None
    
    if os.path.exists(log_path):
        print(f"Found log file: {log_path}")
        log_data = load_log_data(log_path)
        if log_data is not None:
            print(f"Log data shape: {log_data.shape}")
            print(f"Correct trials: {len(log_data[log_data['correct'] > 0])}")
    
    # Create stimulus matrix
    stimulus_matrix = create_stimulus_matrix(nev_data, log_data)
    
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
        
        # Save results
        output_file = nev_path.replace('.nev', '_stimulus_matrix.csv')
        df = pd.DataFrame(stimulus_matrix, columns=['start_time', 'end_time', 'image_id'])
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
        
        # Show statistics
        print(f"\nStatistics:")
        print(f"  Total stimuli: {len(stimulus_matrix)}")
        print(f"  Unique images: {len(np.unique(stimulus_matrix[:, 2]))}")
        print(f"  Time range: {stimulus_matrix[0, 0]:.3f} - {stimulus_matrix[-1, 1]:.3f} seconds")
        print(f"  Total duration: {stimulus_matrix[-1, 1] - stimulus_matrix[0, 0]:.3f} seconds")
        
    else:
        print("Failed to create stimulus matrix")

if __name__ == "__main__":
    main()
