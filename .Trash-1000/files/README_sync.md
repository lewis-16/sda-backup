# Monkey Electrophysiology Data Synchronization

This repository contains Python scripts for synchronizing image stimulus sequences with electrophysiological data recordings from monkey experiments.

## Overview

The synchronization mechanism is based on the analysis of MATLAB code from the TVSD/_code directory and implements:

1. **Stimulus Event Detection**: Reading stimulus timestamps from NEV files
2. **Photodiode Correction**: Using photodiode signals to correct for display delays
3. **MUA Extraction**: Processing raw electrophysiological signals to extract Multi-Unit Activity
4. **Trial Alignment**: Aligning neural responses to stimulus onset times
5. **Data Validation**: Ensuring trial counts match between stimulus logs and recordings

## Files

- `stimulus_electrophysiology_sync.py`: Complete synchronization implementation
- `monkey_sync.py`: Simplified version for your specific data structure
- `example_sync.py`: Example usage script
- `README.md`: This documentation file

## Requirements

```bash
pip install numpy pandas scipy h5py
```

## Usage

### Basic Usage

```python
from monkey_sync import MonkeyDataSynchronizer, MonkeySyncConfig

# Configure parameters
config = MonkeySyncConfig(
    trial_length=0.3,    # 300ms stimulus duration
    pre_trial=0.1,       # 100ms before stimulus
    post_trial=0.2,      # 200ms after stimulus
    fs=30000,            # 30kHz sampling rate
    downsample_factor=30 # Downsample by factor of 30
)

# Initialize synchronizer
sync = MonkeyDataSynchronizer(config)

# Process data
files = sync.find_data_files("/media/ubuntu/sda/Monkey", "20240112", 1)
trigger_data = sync.load_trigger_data(files['triggers'][0])
synchronized_data = sync.synchronize_stimulus_responses(trigger_data, None, {})
```

### Command Line Usage

```bash
# Run the example script
python example_sync.py

# Run synchronization for specific date and block
python monkey_sync.py --date 20240112 --block 1 --data_dir /media/ubuntu/sda/Monkey
```

## Data Structure

The scripts expect the following data structure:

```
/media/ubuntu/sda/Monkey/
├── trigger/
│   ├── trigger_df_monkyF_20240112_B1_instance1.csv
│   └── trigger_df_monkyF_20240112_B1_instance2.csv
├── sorted_result_combined/
│   ├── 20240112/
│   │   ├── cluster_inf_Hub1-instance1_V1.csv
│   │   ├── spike_inf_Hub1-instance1_V1.csv
│   │   └── firing_rate_matrices/
│   │       └── firing_rate_matrices_20240112_combined.npz
└── TVSD/
    └── _code/
        ├── extract_MUA_v2.m
        └── collect_MUA_v2.m
```

## Key Features

### 1. Stimulus-Response Synchronization

- **Dual Timestamp System**: Uses both digital I/O events and photodiode signals
- **Automatic Correction**: Corrects for display delays using photodiode detection
- **Trial Validation**: Ensures stimulus and recording trial counts match

### 2. Signal Processing

- **MUA Extraction**: Bandpass filtering (500-5000 Hz) + rectification + lowpass filtering (200 Hz)
- **Noise Removal**: Automatic removal of 50Hz line noise and 60Hz screen refresh artifacts
- **Downsampling**: Reduces data size while preserving temporal resolution

### 3. Data Analysis

- **Response Alignment**: All trials aligned to stimulus onset (t=0)
- **Image-Specific Analysis**: Groups responses by stimulus image
- **Statistical Analysis**: Calculates mean, std, and range of firing rates

## Configuration Parameters

```python
@dataclass
class MonkeySyncConfig:
    # Timing parameters
    trial_length: float = 0.3      # Stimulus duration (seconds)
    pre_trial: float = 0.1         # Pre-stimulus window (seconds)
    post_trial: float = 0.2        # Post-stimulus window (seconds)
    
    # Sampling parameters
    fs: int = 30000                # Sampling frequency (Hz)
    downsample_factor: int = 30    # Downsampling factor
    
    # Filter parameters
    bandpass_freq: Tuple[float, float] = (500, 5000)  # MUA bandpass (Hz)
    lowpass_freq: float = 200      # MUA lowpass (Hz)
    filter_order: int = 2          # Filter order
    
    # Photodiode parameters
    photodiode_threshold: float = 1e4  # Detection threshold
    photodiode_channel: int = 15       # Photodiode channel number
    
    # Stimulus parameters
    stimbit: int = 1               # Digital I/O bit for stimulus markers
```

## Output Format

The synchronization produces:

1. **Synchronized Data**: 3D arrays [trials, channels, timepoints]
2. **Time Axis**: Millisecond timestamps relative to stimulus onset
3. **Stimulus Information**: Image IDs and presentation times
4. **Analysis Results**: Image-specific response statistics
5. **Metadata**: Configuration and processing information

## Example Output

```python
{
    'synchronized_data': {
        'channel_1': np.array,  # [trials, timepoints]
        'channel_2': np.array,
        # ... more channels
    },
    'time_axis': np.array,      # Time in ms relative to stimulus onset
    'stimulus_info': {
        'stimulus_images': np.array,  # Image IDs for each trial
        'valid_trials': np.array,    # Boolean array of valid trials
    },
    'analysis_results': {
        'image_responses': dict,     # Responses grouped by image
        'response_statistics': dict, # Overall statistics
    },
    'metadata': {
        'num_trials': int,
        'num_channels': int,
        'sample_rate': int,
    }
}
```

## Troubleshooting

### Common Issues

1. **File Not Found**: Check that data files exist in expected locations
2. **Trial Count Mismatch**: Verify that trigger files and recordings have same number of trials
3. **Memory Issues**: Reduce downsampling factor or process data in chunks
4. **Filter Artifacts**: Adjust filter parameters or use different filter types

### Debugging

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## References

This implementation is based on the MATLAB code analysis from:
- `TVSD/_code/extract_MUA_v2.m`
- `TVSD/_code/collect_MUA_v2.m`
- `TVSD/_code/check_log_rec_v2.m`

## License

This code is provided as-is for research purposes. Please cite appropriately if used in publications.
