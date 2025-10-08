#!/usr/bin/env python3
"""
Simple Example Script for Monkey Data Synchronization
====================================================

This script demonstrates how to use the synchronization tools with your
monkey electrophysiology data.

Usage:
    python example_sync.py
"""

import numpy as np
import pandas as pd
import os
import sys
from monkey_sync import MonkeyDataSynchronizer, MonkeySyncConfig

def example_usage():
    """Example of how to use the synchronization tools"""
    
    # Configuration
    config = MonkeySyncConfig(
        trial_length=0.3,    # 300ms stimulus duration
        pre_trial=0.1,       # 100ms before stimulus
        post_trial=0.2,      # 200ms after stimulus
        fs=30000,            # 30kHz sampling rate
        downsample_factor=30 # Downsample by factor of 30
    )
    
    # Initialize synchronizer
    sync = MonkeyDataSynchronizer(config)
    
    # Data directory
    data_dir = "/media/ubuntu/sda/Monkey"
    date = "20240112"
    block = 1
    
    print(f"Processing data for {date}, Block {block}")
    
    # Find data files
    files = sync.find_data_files(data_dir, date, block)
    print(f"Found files: {list(files.keys())}")
    
    # Load trigger data (if available)
    trigger_data = None
    if 'triggers' in files and files['triggers']:
        trigger_file = files['triggers'][0]
        print(f"Loading trigger data from: {trigger_file}")
        trigger_data = sync.load_trigger_data(trigger_file)
        
        if trigger_data is not None:
            print(f"Trigger data shape: {trigger_data.shape}")
            print(f"Trigger data columns: {list(trigger_data.columns)}")
    
    # Load firing rate data (if available)
    firing_rate_data = None
    if 'firing_rates' in files:
        print("Loading firing rate data...")
        firing_rate_data = sync.load_firing_rate_data(files['firing_rates'], date)
        
        if firing_rate_data is not None:
            print(f"Firing rate data keys: {list(firing_rate_data.keys())}")
            if 'firing_rate_matrices' in firing_rate_data:
                fr_shape = firing_rate_data['firing_rate_matrices'].shape
                print(f"Firing rate matrix shape: {fr_shape}")
    
    # Load cluster information (if available)
    cluster_info = {}
    if 'combined_data' in files:
        print("Loading cluster information...")
        cluster_info = sync.load_cluster_info(files['combined_data'])
        print(f"Loaded cluster info for: {list(cluster_info.keys())}")
    
    # Perform synchronization
    if trigger_data is not None:
        print("Performing synchronization...")
        synchronized_data = sync.synchronize_stimulus_responses(
            trigger_data, firing_rate_data, cluster_info
        )
        
        if synchronized_data is not None:
            print("Synchronization successful!")
            print(f"Number of trials: {synchronized_data['metadata']['num_trials']}")
            
            # Analyze responses
            print("Analyzing responses...")
            analysis_results = sync.analyze_responses(synchronized_data)
            
            if analysis_results is not None:
                print(f"Analysis complete!")
                print(f"Number of unique images: {analysis_results['total_images']}")
                print(f"Response statistics: {analysis_results['response_statistics']}")
            
            # Save results
            output_dir = "./example_results"
            sync.save_results(synchronized_data, analysis_results, 
                           output_dir, date, block)
            print(f"Results saved to {output_dir}")
        else:
            print("Synchronization failed!")
    else:
        print("No trigger data available for synchronization")

def analyze_existing_data():
    """Analyze existing processed data"""
    
    data_dir = "/media/ubuntu/sda/Monkey"
    
    # Check for existing firing rate matrices
    firing_rate_dir = f"{data_dir}/sorted_result_combined/firing_rate_matrices"
    if os.path.exists(firing_rate_dir):
        print(f"Found firing rate directory: {firing_rate_dir}")
        
        # List available files
        files = os.listdir(firing_rate_dir)
        print(f"Available files: {files}")
        
        # Load and analyze firing rate data
        for file in files:
            if file.endswith('.npz'):
                file_path = os.path.join(firing_rate_dir, file)
                print(f"\nAnalyzing {file}:")
                
                try:
                    data = np.load(file_path)
                    print(f"  Keys: {list(data.keys())}")
                    
                    if 'firing_rate_matrices' in data:
                        fr_matrices = data['firing_rate_matrices']
                        print(f"  Shape: {fr_matrices.shape}")
                        print(f"  Mean firing rate: {np.mean(fr_matrices):.2f}")
                        print(f"  Max firing rate: {np.max(fr_matrices):.2f}")
                        print(f"  Min firing rate: {np.min(fr_matrices):.2f}")
                    
                    if 'firing_rate_summary' in data:
                        summary = data['firing_rate_summary']
                        print(f"  Summary shape: {summary.shape}")
                
                except Exception as e:
                    print(f"  Error loading {file}: {e}")
    
    # Check for trigger data
    trigger_dir = f"{data_dir}/trigger"
    if os.path.exists(trigger_dir):
        print(f"\nFound trigger directory: {trigger_dir}")
        
        trigger_files = os.listdir(trigger_dir)
        print(f"Available trigger files: {trigger_files}")
        
        # Analyze trigger files
        for file in trigger_files[:3]:  # Analyze first 3 files
            if file.endswith('.csv'):
                file_path = os.path.join(trigger_dir, file)
                print(f"\nAnalyzing trigger file: {file}")
                
                try:
                    df = pd.read_csv(file_path)
                    print(f"  Shape: {df.shape}")
                    print(f"  Columns: {list(df.columns)}")
                    
                    if 'time' in df.columns:
                        print(f"  Time range: {df['time'].min():.2f} - {df['time'].max():.2f}")
                    
                    if 'image_id' in df.columns:
                        unique_images = df['image_id'].nunique()
                        print(f"  Unique images: {unique_images}")
                
                except Exception as e:
                    print(f"  Error loading {file}: {e}")

if __name__ == "__main__":
    print("Monkey Data Synchronization Example")
    print("=" * 40)
    
    # First, analyze existing data
    print("\n1. Analyzing existing processed data...")
    analyze_existing_data()
    
    # Then, demonstrate synchronization
    print("\n2. Demonstrating synchronization...")
    example_usage()
    
    print("\nExample completed!")
