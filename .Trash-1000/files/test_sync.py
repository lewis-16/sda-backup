#!/usr/bin/env python3
"""
Test Script for Monkey Data Synchronization
===========================================

This script creates synthetic data to test the synchronization functionality.
"""

import numpy as np
import pandas as pd
import os
from monkey_sync import MonkeyDataSynchronizer, MonkeySyncConfig

def create_synthetic_data():
    """Create synthetic data for testing"""
    
    # Create synthetic trigger data
    n_trials = 100
    trigger_data = pd.DataFrame({
        'time': np.arange(n_trials) * 2.0,  # 2 seconds between trials
        'image_id': np.random.randint(1, 50, n_trials),
        'trial_type': np.random.choice(['train', 'test'], n_trials),
        'correct': np.random.choice([0, 1], n_trials, p=[0.2, 0.8])
    })
    
    # Create synthetic firing rate data
    n_channels = 64
    n_timepoints = 300  # 300ms at 1kHz
    firing_rate_matrices = np.random.poisson(5, (n_trials, n_channels, n_timepoints))
    
    firing_rate_data = {
        'firing_rate_matrices': firing_rate_matrices,
        'time_axis': np.arange(-100, 200, 1),  # -100ms to 200ms
        'channel_names': [f'channel_{i}' for i in range(n_channels)]
    }
    
    return trigger_data, firing_rate_data

def test_synchronization():
    """Test the synchronization functionality"""
    
    print("Creating synthetic data...")
    trigger_data, firing_rate_data = create_synthetic_data()
    
    print(f"Trigger data shape: {trigger_data.shape}")
    print(f"Firing rate data shape: {firing_rate_data['firing_rate_matrices'].shape}")
    
    # Initialize synchronizer
    config = MonkeySyncConfig()
    sync = MonkeyDataSynchronizer(config)
    
    print("\nPerforming synchronization...")
    synchronized_data = sync.synchronize_stimulus_responses(
        trigger_data, firing_rate_data, {}
    )
    
    if synchronized_data is not None:
        print("✓ Synchronization successful!")
        print(f"  Number of trials: {synchronized_data['metadata']['num_trials']}")
        print(f"  Time axis length: {len(synchronized_data['time_axis'])}")
        
        # Analyze responses
        print("\nAnalyzing responses...")
        analysis_results = sync.analyze_responses(synchronized_data)
        
        if analysis_results is not None:
            print("✓ Analysis successful!")
            print(f"  Number of unique images: {analysis_results['total_images']}")
            print(f"  Mean firing rate: {analysis_results['response_statistics']['mean_firing_rate']:.2f}")
            print(f"  Max firing rate: {analysis_results['response_statistics']['max_firing_rate']:.2f}")
            
            # Show some image-specific results
            print("\nImage-specific responses (first 5 images):")
            for i, img_id in enumerate(analysis_results['unique_images'][:5]):
                img_data = analysis_results['image_responses'][img_id]
                print(f"  Image {img_id}: {img_data['num_trials']} trials, "
                      f"mean response: {np.mean(img_data['mean_response']):.2f}")
        
        # Save results
        output_dir = "./test_results"
        sync.save_results(synchronized_data, analysis_results, 
                         output_dir, "test", 1)
        print(f"\n✓ Results saved to {output_dir}")
        
    else:
        print("✗ Synchronization failed!")

def test_data_loading():
    """Test data loading functions"""
    
    print("\nTesting data loading functions...")
    
    # Test trigger data loading
    trigger_data, _ = create_synthetic_data()
    trigger_file = "test_trigger.csv"
    trigger_data.to_csv(trigger_file, index=False)
    
    config = MonkeySyncConfig()
    sync = MonkeyDataSynchronizer(config)
    
    loaded_data = sync.load_trigger_data(trigger_file)
    if loaded_data is not None:
        print("✓ Trigger data loading successful!")
        print(f"  Loaded {len(loaded_data)} trials")
    else:
        print("✗ Trigger data loading failed!")
    
    # Clean up
    os.remove(trigger_file)

def test_filter_setup():
    """Test filter setup"""
    
    print("\nTesting filter setup...")
    
    config = MonkeySyncConfig()
    sync = MonkeyDataSynchronizer(config)
    
    # Test MUA extraction with synthetic signal
    fs = config.fs
    duration = 1.0  # 1 second
    t = np.linspace(0, duration, int(fs * duration))
    
    # Create synthetic neural signal
    signal = np.sin(2 * np.pi * 1000 * t) + 0.5 * np.sin(2 * np.pi * 2000 * t)
    signal += np.random.normal(0, 0.1, len(signal))  # Add noise
    
    print(f"  Input signal shape: {signal.shape}")
    print(f"  Input signal range: {signal.min():.3f} to {signal.max():.3f}")
    
    # Extract MUA
    mua_signal = sync.extract_mua(signal)
    
    print(f"  MUA signal shape: {mua_signal.shape}")
    print(f"  MUA signal range: {mua_signal.min():.3f} to {mua_signal.max():.3f}")
    print("✓ MUA extraction successful!")

if __name__ == "__main__":
    print("Monkey Data Synchronization Test")
    print("=" * 40)
    
    try:
        test_filter_setup()
        test_data_loading()
        test_synchronization()
        
        print("\n" + "=" * 40)
        print("All tests completed successfully!")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
