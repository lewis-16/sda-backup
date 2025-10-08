#!/usr/bin/env python3
"""
Direct NEV Stimulus Extractor
=============================

Extract stimulus timing and image information directly from NEV files.
Output: Matrix with [start_time, end_time, image_id] for each stimulus.

Usage:
    python direct_nev_extractor.py /media/ubuntu/sda/Monkey/TVSD/monkeyF/20240112/Block_1/NSP-instance1_B001.nev
"""

import numpy as np
import pandas as pd
import os
import sys
import glob
import struct

def read_nev_header(nev_path):
    """Read NEV file header"""
    with open(nev_path, 'rb') as f:
        # Read basic header (first 16 bytes)
        header = f.read(16)
        if len(header) < 16:
            return None
        
        # Parse header fields
        file_spec = struct.unpack('<BB', header[0:2])
        add_flags = struct.unpack('<BB', header[2:4])
        file_format = struct.unpack('<BB', header[4:6])
        header_size = struct.unpack('<I', header[8:12])[0]
        packet_size = struct.unpack('<I', header[12:16])[0]
        
        return {
            'file_spec': file_spec,
            'add_flags': add_flags,
            'file_format': file_format,
            'header_size': header_size,
            'packet_size': packet_size
        }

def read_nev_events(nev_path):
    """Read NEV events (simplified version)"""
    try:
        import scipy.io
        # Try to load as MATLAB file first
        data = scipy.io.loadmat(nev_path)
        return data
    except:
        pass
    
    # If not MATLAB format, try to read binary NEV file
    header = read_nev_header(nev_path)
    if header is None:
        return None
    
    print(f"NEV Header: {header}")
    
    # This is a simplified reader - you may need a proper NEV library
    print("Warning: Binary NEV reading not fully implemented")
    print("Please install a proper NEV reader library or convert to .mat format")
    return None

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

def extract_stimulus_from_mat(nev_path, stimulus_duration=0.3):
    """
    Extract stimulus information from NEV file (converted to .mat format)
    """
    print(f"Processing: {nev_path}")
    
    # Check if file exists
    if not os.path.exists(nev_path):
        print(f"Error: File {nev_path} not found")
        return None
    
    # Try to read NEV data
    nev_data = read_nev_events(nev_path)
    if nev_data is None:
        print("Failed to read NEV data")
        return None
    
    # Find corresponding log file
    log_file = find_log_file(nev_path)
    if log_file:
        print(f"Found log file: {log_file}")
        log_data = load_stimulus_log(log_file)
    else:
        print("No log file found")
        log_data = None
    
    # Extract stimulus timestamps from NEV data
    try:
        if 'EVENT' in nev_data:
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
            
        else:
            print("No EVENT data found in NEV file")
            return None
            
    except Exception as e:
        print(f"Error extracting stimulus timestamps: {e}")
        return None
    
    # Extract image IDs from log data
    image_ids = None
    if log_data is not None:
        try:
            # Use train_pic column for image IDs
            image_ids = log_data['train_pic'].values
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
        print("Usage: python direct_nev_extractor.py <nev_file_path>")
        print("Example: python direct_nev_extractor.py /media/ubuntu/sda/Monkey/TVSD/monkeyF/20240112/Block_1/NSP-instance1_B001.nev")
        return
    
    nev_path = sys.argv[1]
    
    # Extract stimulus information
    stimulus_matrix = extract_stimulus_from_mat(nev_path)
    
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
