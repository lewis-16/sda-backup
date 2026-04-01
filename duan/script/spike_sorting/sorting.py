# %%
import sys
import spikeinterface as si
import matplotlib.pyplot as plt
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface.sorters as ss
import spikeinterface.widgets as sw
import spikeinterface.qualitymetrics as sqm
import json
import probeinterface

from probeinterface import Probe, ProbeGroup

import os
import numpy as np
from spikeinterface.core import concatenate_recordings

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import pearsonr
import pandas as pd
import numpy as np
from matplotlib.collections import LineCollection
from probeinterface import write_probeinterface, read_probeinterface, Probe
from scipy.io import loadmat


probe_data = loadmat("/media/ubuntu/sda/duan/raw_data/chanMap_DCX_5mm.mat")
probe_x = probe_data['xcoords']
probe_y = probe_data['ycoords']

probe_position = pd.DataFrame(probe_x)
probe_position[1] = probe_y
probe_position['chan_map'] = probe_data['chanMap0ind'].astype(int)

chan_map = pd.read_csv('/media/ubuntu/sda/duan/raw_data/ch_map_R.csv')
merged = chan_map.merge(probe_position, left_on='probeloc', right_on='chan_map')\
                 .iloc[chan_map.index]\
                 .reset_index(drop=True)

probe = Probe()
probe.set_contacts(positions=merged.iloc[:, 2:4])
probe.set_device_channel_indices(range(256))

recording_raw = se.read_intan(f"/home/ubuntu/Documents/jct/project/251205/M190011_260121_150111_merged_130.rhd", stream_id= '0', ignore_integrity_checks=True)

print('read success')

recording_raw = spre.unsigned_to_signed(recording_raw)
recording_raw = spre.resample(recording_raw, 10000)

recording_recorded = spre.bandpass_filter(recording_raw, freq_min=300, freq_max=3000)
recording_recorded = spre.notch_filter(recording_recorded, freq=50)
recording_f = spre.common_reference(recording_recorded, reference="global", operator="median")

recording_f = recording_f.set_probegroup(probe)
recording_preprocessed = recording_f.save(format="binary", n_jobs = 30)

#rec_corrected, motion = spre.correct_motion(recording=recording_preprocessed, preset="dredge_fast", output_motion=True, n_jobs = 30)


# %%
output_folder = f'/home/ubuntu/Documents/jct/project/sorted/260121_kilosort_2'
os.makedirs(output_folder, exist_ok=True)

kilosort_params = ss.get_default_sorter_params('kilosort4')
kilosort_params['fs'] = 10000
kilosort_params['torch_device'] = 'cuda'
kilosort_params['n_jobs'] = 30
kilosort_params['templates_from_data'] = False

sorting_kilosort4 = ss.run_sorter(sorter_name="kilosort4", recording=recording_preprocessed, folder=output_folder + "/kilosort4", **kilosort_params)
analyzer_kilosort4 = si.create_sorting_analyzer(sorting=sorting_kilosort4, recording=recording_preprocessed, format='binary_folder', folder=output_folder + '/analyzer_kilosort4_binary', n_jobs = 30)

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

analyzer_kilosort4.compute(extensions_to_compute, extension_params=extension_params, n_jobs = 30)
spikes_path = output_folder + "/analyzer_kilosort4_binary/sorting/spikes.npy"
spikes = np.load(spikes_path)

# 获取recording的总样本数
total_samples = recording_f.get_num_samples()

# 检查第一个和最后一个spike
first_spike_valid = spikes[0]['sample_index'] >= 0
last_spike_valid = spikes[-1]['sample_index'] < total_samples

# 如果第一个或最后一个spike无效，删除所有无效的spike
if not first_spike_valid or not last_spike_valid:
    # 创建有效spike的掩码：sample_index >= 0 且 < total_samples
    valid_mask = (spikes['sample_index'] >= 0) & (spikes['sample_index'] < total_samples)
    spikes_filtered = spikes[valid_mask]
    
    # 保存过滤后的spikes
    np.save(spikes_path, spikes_filtered)
    print(f"删除了 {len(spikes) - len(spikes_filtered)} 个无效的spike")
    print(f"原始spike数量: {len(spikes)}, 过滤后: {len(spikes_filtered)}")
else:
    print("所有spike都在有效范围内")
qm_params = sqm.get_default_qm_params()
analyzer_kilosort4.compute("quality_metrics", qm_params, n_jobs = 30)

import spikeinterface.exporters as sexp
sexp.export_to_phy(analyzer_kilosort4, output_folder + "/phy_folder_for_kilosort", verbose=True, n_jobs = 30)


