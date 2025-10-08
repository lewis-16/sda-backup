#!/usr/bin/env python3
"""
NEV Stimulus Extractor with brpylib
===================================

Extract stimulus timing and image information directly from NEV files using brpylib.
Output: Matrix with [start_time, end_time, image_id] for each stimulus.

Usage:
    python nev_brpy_extractor.py /media/ubuntu/sda/Monkey/TVSD/monkeyF/20240112/Block_1/NSP-instance1_B001.nev
"""

import numpy as np
import pandas as pd
import os
import sys
import glob

try:
    from brpylib import NevFile
    BRPY_AVAILABLE = True
except ImportError:
    BRPY_AVAILABLE = False
    print("Warning: brpylib not available. Please install: pip install brpylib")

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

def find_log_file(nev_path):
    """Find corresponding log file for NEV file"""
    # Extract date and block from NEV path
    nev_dir = os.path.dirname(nev_path)
    nev_filename = os.path.basename(nev_path)
    
    # Look for THINGS log files in _logs directory
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(nev_dir)), '_logs')
    
    if os.path.exists(logs_dir):
        # Look for files matching the date and block
        log_files = glob.glob(os.path.join(logs_dir, "*THINGS*20240112*B1*.mat"))
        if log_files:
            return log_files[0]
    
    return None

def extract_stimulus_timestamps_from_nev(nev_path, stimbit=1):
    """
    Extract stimulus timestamps from NEV file using brpylib
    
    Args:
        nev_path: Path to NEV file
        stimbit: Digital I/O bit for stimulus markers (default: 1)
        
    Returns:
        Dictionary containing stimulus timestamps and metadata
    """
    if not BRPY_AVAILABLE:
        print("Error: brpylib not available")
        return None
    
    try:
        print(f"Reading NEV file: {nev_path}")
        nev_file = NevFile(nev_path)
        
        # Get digital I/O data
        digital_events = nev_file.getdata('digitalserial')
        
        if digital_events is None or len(digital_events) == 0:
            print("No digital I/O events found in NEV file")
            return None
        
        print(f"Found {len(digital_events)} digital I/O events")
        
        # Find stimulus events (assuming bit 0 = stimulus marker)
        stim_mask = (digital_events['UnparsedData'] & (2**stimbit)) > 0
        stim_timestamps = digital_events['TimeStamp'][stim_mask]
        
        print(f"Found {len(stim_timestamps)} stimulus events")
        
        # Get sample rate
        sample_rate = nev_file.getdata('digitalserial')[0]['SampleRes'] if len(digital_events) > 0 else 30000
        
        return {
            'stimulus_timestamps': stim_timestamps,
            'total_events': len(digital_events),
            'stimulus_events': len(stim_timestamps),
            'sample_rate': sample_rate,
            'digital_events': digital_events
        }
        
    except Exception as e:
        print(f"Error reading NEV file {nev_path}: {e}")
        return None

def create_stimulus_matrix_from_nev(nev_data, log_data=None, stimulus_duration=0.2):
    """
    Create stimulus matrix from NEV data and optional log data
    
    Args:
        nev_data: NEV data dictionary
        log_data: Optional DataFrame with stimulus log
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
    
    # Get image IDs from log data if available
    if log_data is not None:
        # Filter for correct trials
        correct_trials = log_data[log_data['correct'] > 0].copy()
        
        if len(correct_trials) == len(stim_timestamps):
            image_ids = correct_trials['train_pic'].values
            print(f"Using image IDs from log data: {len(image_ids)} images")
        else:
            print(f"Warning: Log data length ({len(correct_trials)}) doesn't match stimulus count ({len(stim_timestamps)})")
            image_ids = np.arange(len(stim_timestamps)) + 1
    else:
        # Use trial numbers as image IDs
        image_ids = np.arange(len(stim_timestamps)) + 1
        print("Using trial numbers as image IDs")
    
    # Create stimulus matrix: [start_time, end_time, image_id]
    stimulus_matrix = np.column_stack([start_times, end_times, image_ids])
    
    return stimulus_matrix

def process_nev_file(nev_path, log_path=None, stimulus_duration=0.2):
    """Process a single NEV file"""
    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(nev_path)}")
    print(f"{'='*60}")
    
    # Extract stimulus timestamps from NEV file
    nev_data = extract_stimulus_timestamps_from_nev(nev_path)
    if nev_data is None:
        print("Failed to extract NEV data")
        return None
    
    print(f"Sample rate: {nev_data['sample_rate']} Hz")
    print(f"Stimulus events: {nev_data['stimulus_events']}")
    
    # Load log data if available
    log_data = None
    if log_path is None:
        log_path = find_log_file(nev_path)
    
    if log_path and os.path.exists(log_path):
        print(f"Found log file: {log_path}")
        log_data = load_stimulus_log(log_path)
        if log_data is not None:
            print(f"Log data shape: {log_data.shape}")
            print(f"Correct trials: {len(log_data[log_data['correct'] > 0])}")
    
    # Create stimulus matrix
    stimulus_matrix = create_stimulus_matrix_from_nev(nev_data, log_data, stimulus_duration)
    
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
        output_dir = os.path.dirname(nev_path)
        base_name = os.path.basename(nev_path).replace('.nev', '')
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

def main():
    """Main function"""
    if not BRPY_AVAILABLE:
        print("Error: brpylib is required but not available")
        print("Please install: pip install brpylib")
        return
    
    if len(sys.argv) < 2:
        print("Usage: python nev_brpy_extractor.py <nev_file_path> [log_file_path]")
        print("Example: python nev_brpy_extractor.py /media/ubuntu/sda/Monkey/TVSD/monkeyF/20240112/Block_1/NSP-instance1_B001.nev")
        return
    
    nev_path = sys.argv[1]
    log_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(nev_path):
        print(f"Error: File {nev_path} not found")
        return
    
    # Process NEV file
    stimulus_matrix = process_nev_file(nev_path, log_path)
    
    if stimulus_matrix is not None:
        print(f"\n✓ Successfully processed {nev_path}")
    else:
        print(f"\n✗ Failed to process {nev_path}")

if __name__ == "__main__":
    main()
