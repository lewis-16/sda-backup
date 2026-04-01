# %%
import sys
import spikeinterface as si
import matplotlib.pyplot as plt
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface.sorters as ss
import spikeinterface.widgets as sw
import spikeinterface.qualitymetrics as sqm
import spikeinterface.exporters as sexp
import json
import probeinterface

from probeinterface import Probe, ProbeGroup

import os
import numpy as np

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

import pandas as pd
import numpy as np
from probeinterface import write_probeinterface, read_probeinterface, Probe
from scipy.io import loadmat


probe_data = loadmat("/mnt/solid1/spike_sorting/chanMap_DCX_5mm.mat")
probe_x = probe_data['xcoords']
probe_y = probe_data['ycoords']

probe_position = pd.DataFrame(probe_x)
probe_position[1] = probe_y
probe_position['chan_map'] = probe_data['chanMap0ind'].astype(int)

chan_map = pd.read_csv('/mnt/solid1/spike_sorting/ch_map_R.csv')
merged = chan_map.merge(probe_position, left_on='probeloc', right_on='chan_map')\
                 .iloc[chan_map.index]\
                 .reset_index(drop=True)

probe = Probe()
probe.set_contacts(positions=merged.iloc[:, 2:4])
probe.set_device_channel_indices(range(256))
print("Get probes!")

# %%
recording_raw = se.read_intan(f"/mnt/solid1/M190011_260121_150111_merged_130.rhd", stream_id= '0', ignore_integrity_checks=True)

print('Read success!')

recording_raw = spre.unsigned_to_signed(recording_raw)
recording_raw = spre.resample(recording_raw, 10000)

recording_recorded = spre.bandpass_filter(recording_raw, freq_min=300, freq_max=3000)
recording_recorded = spre.notch_filter(recording_recorded, freq=50)
recording_f = spre.common_reference(recording_recorded, reference="global", operator="median")

recording_f = recording_f.set_probegroup(probe)
recording_f = spre.astype(recording_f, dtype="float32")

print("Preprocessing complete!")

rec_params = pd.read_csv('/mnt/solid1/spike_sorting/rec_params.csv')
print(f"Loaded rec_params: {len(rec_params)} rows, {rec_params['trial_ids'].nunique()} unique trials")

original_fs = 30000
target_fs = 10000
fs_ratio = original_fs / target_fs

rec_params['rec_codes_points_10000'] = (rec_params['rec_codes_points'] / fs_ratio).astype(int)

num_trials = 10000
trials_per_group = 1000
num_groups_to_process = 5
buffer_seconds = 10
fs = recording_f.get_sampling_frequency()
buffer_samples = int(buffer_seconds * fs)

output_folder = f'/mnt/solid1/spike_sorting/260121'
os.makedirs(output_folder, exist_ok=True)

print(f"\nDetermining time range for first {num_groups_to_process} groups...")

first_group_min_sample = None
last_group_max_sample = None

for group_idx in range(num_groups_to_process):
    trial_start = group_idx * trials_per_group + 1
    trial_end = (group_idx + 1) * trials_per_group
    
    print(f"Group {group_idx+1}/{num_groups_to_process}: trials {trial_start}-{trial_end}")
    
    group_params = rec_params[(rec_params['trial_ids'] >= trial_start) & (rec_params['trial_ids'] <= trial_end)]
    
    if len(group_params) == 0:
        print(f"  Warning: No data for trials {trial_start}-{trial_end}")
        continue
    
    min_sample = int(group_params['rec_codes_points_10000'].min())
    max_sample = int(group_params['rec_codes_points_10000'].max())
    
    if group_idx == 0:
        first_group_min_sample = min_sample
    if group_idx == num_groups_to_process - 1:
        last_group_max_sample = max_sample
    
    print(f"  Sample range: {min_sample}-{max_sample}")

if first_group_min_sample is None or last_group_max_sample is None:
    print("Error: Could not determine time range!")
else:
    start_sample = max(0, first_group_min_sample - buffer_samples)
    end_sample = min(recording_f.get_num_samples(), last_group_max_sample + buffer_samples)
    
    print(f"\nSlicing recording from sample {start_sample} to {end_sample}")
    print(f"  First group start: {first_group_min_sample}, with {buffer_seconds}s buffer: {start_sample}")
    print(f"  Last group end: {last_group_max_sample}, with {buffer_seconds}s buffer: {end_sample}")
    print(f"  Total duration: {(end_sample - start_sample) / fs:.2f} seconds")
    
    recording_segment = recording_f.frame_slice(start_frame=start_sample, end_frame=end_sample)
    
    print("Saving recording segment...")
    recording_segment_preprocessed = recording_segment.save(format="binary", n_jobs=24)
    
    group_output_folder = os.path.join(output_folder, f"groups_01_to_{num_groups_to_process:02d}_sliced")
    os.makedirs(group_output_folder, exist_ok=True)
    
    kilosort_param = ss.get_default_sorter_params('kilosort4')
    kilosort_param['fs'] = 10000
    kilosort_param['Th_learned'] = 7
    kilosort_param['Th_universal'] = 9
    kilosort_param['n_jobs'] = 30
    
    print("Running kilosort4 on sliced recording...")
    sorting_kilosort4 = ss.run_sorter(
        sorter_name="kilosort4",
        recording=recording_segment_preprocessed,
        folder=os.path.join(group_output_folder, "kilosort4"),
        **kilosort_param
    )
    
    print("Sorting completed!")
    
    analyzer_kilosort4 = si.create_sorting_analyzer(
        sorting=sorting_kilosort4,
        recording=recording_segment_preprocessed,
        format='binary_folder',
        folder=os.path.join(group_output_folder, 'analyzer_kilosort4_binary'),
        n_jobs=30
    )
    
    extensions_to_compute = [
        "random_spikes",
        "waveforms",
        "noise_levels",
        "templates",
        "spike_amplitudes",
        "unit_locations",
        "spike_locations",
        "correlograms",
        "template_similarity"
    ]
    
    extension_params = {
        "unit_locations": {"method": "center_of_mass"},
        "spike_locations": {"ms_before": 0.1},
        "correlograms": {"bin_ms": 0.1},
        "template_similarity": {"method": "cosine_similarity"}
    }
    
    print("Computing extensions...")
    analyzer_kilosort4.compute(extensions_to_compute, extension_params=extension_params, n_jobs=30)
    
    spikes_path = os.path.join(group_output_folder, "analyzer_kilosort4_binary", "sorting", "spikes.npy")
    if os.path.exists(spikes_path):
        spikes = np.load(spikes_path)
        total_samples = recording_segment.get_num_samples()
        
        first_spike_valid = spikes[0]['sample_index'] >= 0
        last_spike_valid = spikes[-1]['sample_index'] < total_samples
        
        if not first_spike_valid or not last_spike_valid:
            valid_mask = (spikes['sample_index'] >= 0) & (spikes['sample_index'] < total_samples)
            spikes_filtered = spikes[valid_mask]
            np.save(spikes_path, spikes_filtered)
            print(f"    Deleted {len(spikes) - len(spikes_filtered)} invalid spikes")
        else:
            print(f"    All spikes are valid")
    
    qm_params = sqm.get_default_qm_params()
    analyzer_kilosort4.compute("quality_metrics", qm_params, n_jobs=30)
    
    print("Exporting to phy...")
    sexp.export_to_phy(
        analyzer_kilosort4,
        os.path.join(group_output_folder, "phy_folder_for_kilosort"),
        verbose=True,
        n_jobs=30
    )
    
    print(f"\nCompleted processing sliced groups 1-{num_groups_to_process}!")
    print(f"Total units: {sorting_kilosort4.get_num_units()}")


