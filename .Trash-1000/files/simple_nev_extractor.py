#!/usr/bin/env python3
"""
Simple NEV Stimulus Extractor for Monkey Data
=============================================

Extract stimulus timing and image information from NEV files.
Output: Matrix with [start_time, end_time, image_id] for each stimulus.

Usage:
    python simple_nev_extractor.py /media/ubuntu/sda/Monkey/TVSD/monkeyF/20240112/Block_1/NSP-instance1_B001.nev
"""

import numpy as np
import pandas as pd
import os
import sys
import glob

def find_mat_files(nev_path):
    """Find corresponding .mat files for the NEV file"""
    nev_dir = os.path.dirname(nev_path)
    nev_filename = os.path.basename(nev_path)
    
    # Look for .mat files in the same directory
    mat_files = glob.glob(os.path.join(nev_dir, "*.mat"))
    
    # Also look in parent directories for log files
    parent_dir = os.path.dirname(nev_dir)
    if os.path.exists(parent_dir):
        mat_files.extend(glob.glob(os.path.join(parent_dir, "*.mat")))
    
    # Look for THINGS log files
    things_files = []
    for mat_file in mat_files:
        if 'THINGS' in os.path.basename(mat_file):
            things_files.append(mat_file)
    
    return things_files

def extract_stimulus_info(nev_path, stimulus_duration=0.3):
    """
    Extract stimulus information from NEV file
    
    Args:
        nev_path: Path to NEV file
        stimulus_duration: Duration of each stimulus in seconds
        
    Returns:
        Matrix with [start_time, end_time, image_id] for each stimulus
    """
    
    print(f"Processing: {nev_path}")
    
    # Check if file exists
    if not os.path.exists(nev_path):
        print(f"Error: File {nev_path} not found")
        return None
    
    # Look for corresponding .mat files
    mat_files = find_mat_files(nev_path)
    print(f"Found {len(mat_files)} .mat files")
    
    # Try to load NEV data from .mat files
    nev_data = None
    log_data = None
    
    for mat_file in mat_files:
        try:
            print(f"Trying to load: {mat_file}")
            
            # Try to load as NEV data
            if 'EVENT' in mat_file or 'nev' in mat_file.lower():
                try:
                    import scipy.io
                    data = scipy.io.loadmat(mat_file)
                    if 'EVENT' in data:
                        nev_data = data
                        print(f"✓ Loaded NEV data from {mat_file}")
                        break
                except:
                    pass
            
            # Try to load as log data
            if 'THINGS' in mat_file:
                try:
                    import scipy.io
                    data = scipy.io.loadmat(mat_file)
                    if 'MAT' in data:
                        log_data = data
                        print(f"✓ Loaded log data from {mat_file}")
                except:
                    pass
                    
        except Exception as e:
            print(f"Error loading {mat_file}: {e}")
            continue
    
    # If no .mat files found, try to read raw NEV file
    if nev_data is None:
        print("No .mat files found, trying to read raw NEV file...")
        print("Note: Raw NEV reading requires additional libraries")
        return None
    
    # Extract stimulus timestamps
    try:
        digital_io = nev_data['EVENT']['Data'][0,0]['SerialDigitalIO'][0,0]
        timestamps = digital_io['TimeStamp'][0]
        unparsed_data = digital_io['UnparsedData'][0]
        
        # Find stimulus events (assuming bit 0 = stimulus marker)
        stim_mask = (unparsed_data & 1) > 0
        stim_timestamps = timestamps[stim_mask]
        
        # Get sample rate
        sample_rate = nev_data['EVENT']['MetaTags'][0,0]['SamplingFreq'][0,0]
        
        print(f"Found {len(stim_timestamps)} stimulus events")
        print(f"Sample rate: {sample_rate} Hz")
        
    except Exception as e:
        print(f"Error extracting stimulus timestamps: {e}")
        return None
    
    # Extract image IDs from log data
    image_ids = None
    if log_data is not None:
        try:
            mat_data = log_data['MAT']
            # MAT format: [#trial #train_pic #test_pic #pic_rep #ncount #correct]
            image_ids = mat_data[:, 1]  # train_pic column
            print(f"Found {len(image_ids)} image IDs in log data")
        except Exception as e:
            print(f"Error extracting image IDs: {e}")
    
    # If no log data, use trial numbers
    if image_ids is None:
        image_ids = np.arange(len(stim_timestamps)) + 1
        print("Using trial numbers as image IDs")
    
    # Convert timestamps to seconds
    start_times = stim_timestamps / sample_rate
    end_times = start_times + stimulus_duration
    
    # Ensure we have the right number of image IDs
    if len(image_ids) != len(stim_timestamps):
        print(f"Warning: Image ID count ({len(image_ids)}) doesn't match stimulus count ({len(stim_timestamps)})")
        if len(image_ids) > len(stim_timestamps):
            image_ids = image_ids[:len(stim_timestamps)]
        else:
            image_ids = np.pad(image_ids, (0, len(stim_timestamps) - len(image_ids)), 'constant')
    
    # Create stimulus matrix: [start_time, end_time, image_id]
    stimulus_matrix = np.column_stack([start_times, end_times, image_ids])
    
    return stimulus_matrix

def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python simple_nev_extractor.py <nev_file_path>")
        print("Example: python simple_nev_extractor.py /media/ubuntu/sda/Monkey/TVSD/monkeyF/20240112/Block_1/NSP-instance1_B001.nev")
        return
    
    nev_path = sys.argv[1]
    
    # Extract stimulus information
    stimulus_matrix = extract_stimulus_info(nev_path)
    
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
        output_file = nev_path.replace('.nev', '_stimulus_matrix.txt')
        np.savetxt(output_file, stimulus_matrix, 
                  header='start_time end_time image_id', 
                  fmt='%.6f %.6f %.0f')
        print(f"\nResults saved to: {output_file}")
        
        # Also save as CSV for easier viewing
        csv_file = nev_path.replace('.nev', '_stimulus_matrix.csv')
        df = pd.DataFrame(stimulus_matrix, columns=['start_time', 'end_time', 'image_id'])
        df.to_csv(csv_file, index=False)
        print(f"Results also saved as CSV: {csv_file}")
        
    else:
        print("Failed to extract stimulus information")

if __name__ == "__main__":
    main()
