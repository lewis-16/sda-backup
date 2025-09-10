from pathlib import Path
from kilosort.io import load_ops
import sys
import spikeinterface as si
import matplotlib.pyplot as plt

import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface.sorters as ss
import spikeinterface.widgets as sw
import spikeinterface.qualitymetrics as sqm
import json
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
from kilosort import io
warnings.filterwarnings('ignore')

global_job_kwargs = dict(n_jobs = 4)
si.set_global_job_kwargs(**global_job_kwargs)


import os
import pandas as pd
import numpy as np

def get_spike_inf(file_path):
    cluster_inf = pd.read_csv(file_path + "/analyzer_kilosort4_binary/extensions/quality_metrics/metrics.csv")
    cluster_inf.columns = ['cluster', 'num_spikes', 'firing_rate', 'presence_ratio', 'snr',
                           'isi_violations_ratio', 'isi_violations_count', 'rp_contamination',
                           'rp_violations', 'sliding_rp_violation', 'amplitude_cutoff',
                           'amplitude_median', 'amplitude_cv_median', 'amplitude_cv_range',
                           'sync_spike_2', 'sync_spike_4', 'sync_spike_8', 'firing_range',
                           'drift_ptp', 'drift_std', 'drift_mad', 'sd_ratio']
    cluster_inf['cluster'] = cluster_inf['cluster'].astype(str)
    cluster_inf['position_1'] = None
    cluster_inf['position_2'] = None

    def get_best_channels(results_dir):
        """Get channel numbers with largest template norm for each cluster."""
        templates = np.load(results_dir + '/templates.npy')
        best_chans = (templates**2).sum(axis=1).argmax(axis=-1)
        return best_chans

    def get_six_best_channels(results_dir):
        """Get channel numbers with largest template norm for each cluster."""
        templates = np.load(results_dir + '/templates.npy')
        template_norms = (templates ** 2).sum(axis=1)
        best_chans = np.argsort(template_norms, axis=-1)[:, -6:][:, ::-1]
        return best_chans

    best_chans = get_best_channels(results_dir=file_path + "/kilosort4/sorter_output")
    best_six_chans = get_six_best_channels(results_dir=file_path + "/kilosort4/sorter_output")
    cluster_inf['best_chans'] = best_chans
    cluster_inf['best_six_chans'] = best_six_chans.tolist()

    spike_clusters = pd.DataFrame(np.load(file_path + "/kilosort4/sorter_output/spike_clusters.npy").astype(str))
    spike_positions = pd.DataFrame(np.load(file_path + "/kilosort4/sorter_output/spike_positions.npy").astype(float))
    spike_templates = pd.DataFrame(np.load(file_path + "/kilosort4/sorter_output/spike_templates.npy"))
    spike_times = pd.DataFrame(np.load(file_path + "/kilosort4/sorter_output/spike_times.npy").astype(int))
    tf = pd.DataFrame(np.load(file_path + "/kilosort4/sorter_output/tF.npy")[:, 0, :])

    spike_inf = pd.concat((spike_clusters, spike_positions, spike_templates, spike_times, tf), axis=1)
    spike_inf.columns = ['cluster', 'position_1', 'position_2', 'templates', 'time', 'PC_1', 'PC_2', 'PC_3', 'PC_4', 'PC_5', 'PC_6']

    for i in spike_inf['cluster'].value_counts().index:
        temp = spike_inf[spike_inf['cluster'] == i]
        cluster_inf.loc[cluster_inf['cluster'] == i, 'position_1'] = np.mean(temp['position_1'])
        cluster_inf.loc[cluster_inf['cluster'] == i, 'position_2'] = np.mean(temp['position_2'])

    cluster_inf['probe_group'] = "1"

    for i in spike_inf['cluster'].value_counts().index:
        cluster_rows = cluster_inf[cluster_inf['cluster'] == i]
        if (cluster_rows['position_1'] > 100).any() and (cluster_rows['position_1'] < 250).any():
            cluster_inf.loc[cluster_inf['cluster'] == i, 'probe_group'] = "2"
        elif (cluster_rows['position_1'] > 250).any() and (cluster_rows['position_1'] < 400).any():
            cluster_inf.loc[cluster_inf['cluster'] == i, 'probe_group'] = "3"
        elif (cluster_rows['position_1'] > 400).any() and (cluster_rows['position_1'] < 550).any():
            cluster_inf.loc[cluster_inf['cluster'] == i, 'probe_group'] = "4"
        elif (cluster_rows['position_1'] > 550).any():
            cluster_inf.loc[cluster_inf['cluster'] == i, 'probe_group'] = "5"

    cluster_inf = cluster_inf[((cluster_inf['snr'] > 3) & (cluster_inf['num_spikes'] > int(5000))) | ((cluster_inf['snr'] < 3) & (cluster_inf['num_spikes'] > 8000))]
    spike_inf = spike_inf[spike_inf['cluster'].isin(list(cluster_inf['cluster']))]
    spike_inf = spike_inf[spike_inf['time'] > 200]
    cluster_inf['date'] = date
    spike_inf['date'] = date

    return cluster_inf, spike_inf

all_cluster_inf = pd.DataFrame()
all_spike_inf = pd.DataFrame()

for date in os.listdir("/media/ubuntu/sda/data/sort_output/mouse11/natural_image"):
    cluster_inf, spike_inf = get_spike_inf(file_path=f"/media/ubuntu/sda/data/sort_output/mouse11/natural_image/{date}")
    all_cluster_inf = pd.concat([all_cluster_inf, cluster_inf], ignore_index=True)
    all_spike_inf = pd.concat([all_spike_inf, spike_inf], ignore_index=True)

all_cluster_inf['Neuron'] = None
neuron_count = 1
all_cluster_inf.at[0, 'Neuron'] = f'Neuron_{neuron_count}'

for i in range(1, len(all_cluster_inf)):
    is_new_neuron = True
    for j in range(i):
        if abs(all_cluster_inf.iloc[i]['position_1'] - all_cluster_inf.iloc[j]['position_1']) < 10 and \
           abs(all_cluster_inf.iloc[i]['position_2'] - all_cluster_inf.iloc[j]['position_2']) < 10:
            all_cluster_inf.at[i, 'Neuron'] = all_cluster_inf.at[j, 'Neuron']
            is_new_neuron = False
            break
    if is_new_neuron:
        neuron_count += 1
        all_cluster_inf.at[i, 'Neuron'] = f'Neuron_{neuron_count}'

neuron_date = pd.crosstab(all_cluster_inf['Neuron'], all_cluster_inf['date'])   
neuron_date[neuron_date > 1] = 1
neuron_date = neuron_date.sum(axis=1)
neuron_date = neuron_date[neuron_date == 16]
neuron_date = neuron_date.index

all_cluster_inf = all_cluster_inf[all_cluster_inf['Neuron'].isin(neuron_date)]
all_cluster_inf['cluster_date'] = all_cluster_inf['date']  + "_" +  all_cluster_inf['cluster']
all_spike_inf['cluster_date'] = all_spike_inf['date']  + "_" +  all_spike_inf['cluster']

all_spike_inf = all_spike_inf[all_spike_inf['cluster_date'].isin(list(all_cluster_inf['cluster_date']))]

def get_spike_waveforms(spikes, results_dir, bfile=None, chan=None):
    """Get waveform for each spike in `spikes`, multi- or single-channel.
    
    Parameters
    ----------
    spikes : list or array-like
        Spike times (in units of samples) for the desired waveforms, from
        `spike_times.npy`.
    results_dir : str or Path
        Path to directory where Kilosort4 sorting results were saved.
    bfile : kilosort.io.BinaryFiltered; optional
        Kilosort4 data file object. By default, this will be loaded using the
        information in `ops.npy` in the saved results.
    chan : int; optional.
        Channel to use for single-channel waveforms. If not specified, all
        channels will be returned.

    Returns
    -------
    waves : np.ndarray
        Array of spike waveforms with shape `(nt, len(spikes))`.
    
    """
    if isinstance(spikes, int):
        spikes = [spikes]

    if bfile is None:
        ops = io.load_ops(results_dir + '/ops.npy')
        bfile = io.bfile_from_ops(ops)
    whitening_mat_inv = np.load(results_dir + '/whitening_mat_inv.npy')

    waves = []
    for t in spikes:
        tmin = t - bfile.nt0min
        tmax = t + (bfile.nt - bfile.nt0min)
        w = bfile[tmin:tmax].cpu().numpy()
        if whitening_mat_inv is not None:
            w = whitening_mat_inv @ w
        if w.shape[1] == bfile.nt:
            # Don't include spikes at the start or end of the recording that
            # get truncated to fewer time points.
            waves.append(w)
    waves = np.stack(waves, axis=-1)

    if chan is not None:
        waves = waves[chan,:]
    
    bfile.close()

    return waves
os.makedirs("/media/ubuntu/sda/data/filter_neuron/mouse_11/natural_image/waveform/", exist_ok=True)
def save_waveform_data(waveform_dict, waveform_mean, neuron):
    # Save waveform_dict and waveform_mean to files
    waveform_dict_path = f"/media/ubuntu/sda/data/filter_neuron/mouse_11/natural_image/waveform/waveform_dict_{neuron}.pkl"
    waveform_mean_path = f"/media/ubuntu/sda/data/filter_neuron/mouse_11/natural_image/waveform/waveform_mean_{neuron}.csv"
    
    pd.to_pickle(waveform_dict, waveform_dict_path)
    waveform_mean.to_csv(waveform_mean_path)
    print(f"Save wavrform to {waveform_mean_path}!")

for neuron in all_cluster_inf['Neuron'].unique():
    waveform_dict = {}
    waveform_mean = pd.DataFrame()
    
    neuron_inf = all_cluster_inf[all_cluster_inf['Neuron'] == neuron]
    
    for date in neuron_inf['date'].unique():
        cluster_inf_date = neuron_inf[neuron_inf['date'] == date]
        spike_inf_date = all_spike_inf[all_spike_inf['date'] == date]
        
        for cluster in cluster_inf_date['cluster'].unique():
            temp = spike_inf_date[spike_inf_date['cluster'] == cluster]
            best_chan = cluster_inf_date.loc[cluster_inf_date['cluster'] == cluster, 'best_chans'].values[0]
            waveforms = get_spike_waveforms(spikes=list(temp['time']), results_dir=f"/media/ubuntu/sda/data/sort_output/mouse11/natural_image/{date}/kilosort4/sorter_output", chan=best_chan)
            
            date_cluster = f"{date}_{cluster}"
            waveform_dict[date_cluster] = waveforms
            
            mean_waveform = np.mean(waveforms, axis=1)
            waveform_mean[date_cluster] = mean_waveform
    
    waveform_mean = waveform_mean.T
    save_waveform_data(waveform_dict, waveform_mean, neuron)

import os
import pandas as pd
import torch 
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering


num = 0
results = {}

folder_path = '/media/ubuntu/sda/data/filter_neuron/mouse_11/natural_image/waveform/'

csv_files = [f for f in os.listdir(folder_path) if f.startswith('waveform_mean_Neuron_') and f.endswith('.csv')]


for csv_file in csv_files:
    df = pd.read_csv(os.path.join(folder_path, csv_file), index_col=0)
    tensor = torch.tensor(df.values, dtype=torch.float32)
    cosine_similarities = F.cosine_similarity(tensor.unsqueeze(1), tensor.unsqueeze(0), dim=2)
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.2, linkage='average')
    clustering.fit(1 - cosine_similarities)
    labels = clustering.labels_
    label_counts = pd.Series(labels).value_counts()
    
    valid_labels = label_counts[label_counts >= 16].index
    
    if len(valid_labels) == 0:
        results[num] = list(df.index)
        num += 1
    else:
        for valid_label in valid_labels:
            results[num] = [index for index, label in zip(df.index, labels) if label == valid_label]
            num += 1

unique_results = {}
seen_values = set()
for key, value in results.items():
    value_tuple = tuple(value)
    if value_tuple not in seen_values:
        unique_results[key] = value
        seen_values.add(value_tuple)


for key,item in unique_results.items():
    item = set([i.split("_")[0] for i in item])
    if len(item) != 16:
        print(f"{key}: {len(item)}")
        del unique_results[key]

all_cluster_inf['Neuron'] = None
for key,item in unique_results.items():
    all_cluster_inf.loc[all_cluster_inf['cluster_date'].isin(item), 'Neuron'] = f'Neuron_{key+1}'

all_cluster_inf = all_cluster_inf.dropna(subset=['Neuron'])
all_spike_inf = all_spike_inf[all_spike_inf['cluster_date'].isin(all_cluster_inf['cluster_date'].unique())]

all_spike_inf['Neuron'] = None
for i in range(len(all_cluster_inf)):
    all_spike_inf.loc[all_spike_inf['cluster_date'] == all_cluster_inf.iloc[i, 29], "Neuron"] = all_cluster_inf.iloc[i, 28]

all_cluster_inf['neuron_date'] = all_cluster_inf['date'] + "_" + all_cluster_inf['Neuron']
all_spike_inf['neuron_date'] = all_spike_inf['date'] + "_" + all_spike_inf['Neuron']

all_cluster_inf.to_csv('/media/ubuntu/sda/data/filter_neuron/mouse_11/natural_image/cluster_inf.tsv', sep = '\t')
all_spike_inf.to_csv("/media/ubuntu/sda/data/filter_neuron/mouse_11/natural_image/spike_inf.tsv", sep='\t')