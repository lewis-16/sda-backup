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
import os
import pandas as pd
import numpy as np



def get_spike_inf(file_path, date):
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

    spike_clusters = pd.DataFrame(np.load(file_path + "/kilosort4/sorter_output/spike_clusters.npy").astype(str))
    spike_positions = pd.DataFrame(np.load(file_path + "/kilosort4/sorter_output/spike_positions.npy").astype(float))
    spike_templates = pd.DataFrame(np.load(file_path + "/kilosort4/sorter_output/spike_templates.npy"))
    spike_times = pd.DataFrame(np.load(file_path + "/kilosort4/sorter_output/spike_times.npy").astype(int))

    spike_inf = pd.concat((spike_clusters, spike_positions, spike_templates, spike_times), axis=1)
    spike_inf.columns = ['cluster', 'position_1', 'position_2', 'templates', 'time']

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

    waveform = np.load(file_path + "/kilosort4/sorter_output/templates.npy")
    cluster_inf['waveform'] = [waveform[i] for i in range(waveform.shape[0])]

    cluster_inf = cluster_inf[((cluster_inf['snr'] > 3) & (cluster_inf['num_spikes'] > int(5000))) | ((cluster_inf['snr'] < 3) & (cluster_inf['num_spikes'] > 8000))]
    spike_inf = spike_inf[spike_inf['cluster'].isin(list(cluster_inf['cluster']))]
    spike_inf = spike_inf[spike_inf['time'] > 200]
    cluster_inf['date'] = date
    spike_inf['date'] = date
    
    channel_indices = {
        "1": [1, 3, 5, 6, 9, 11],
        "2": [13, 15, 17, 19, 21, 23],
        "3": [24, 25, 26, 27, 28, 29],
        "4": [12, 14, 16, 18, 20, 22],
        "5": [0, 2, 4, 6, 8, 10]
        }

    for index, row in cluster_inf.iterrows():
        probe_group = row['probe_group']
        if probe_group in channel_indices:
            selected_channels = channel_indices[probe_group]
            cluster_inf.at[index, 'waveform'] = row['waveform'][:, selected_channels]

    return cluster_inf, spike_inf

def calculate_position(row):
    channel_indices = {
        "1": [1, 3, 5, 6, 9, 11],
        "2": [13, 15, 17, 19, 21, 23],
        "3": [24, 25, 26, 27, 28, 29],
        "4": [12, 14, 16, 18, 20, 22],
        "5": [0, 2, 4, 6, 8, 10]
        }
    channel_position = {
        0: [650, 0],
        2: [650, 50],
        4: [650, 100],
        6: [600, 100],
        8: [600, 50],
        10: [600, 0],
        1: [0, 0],
        3: [0, 50],
        5: [0, 100],
        7: [50, 100],
        9: [50, 50],
        11: [50, 0],
        13: [150, 200], 
        15: [150, 250],
        17: [150, 300],
        19: [200, 300],
        21: [200, 250],
        23: [200, 200],
        12: [500, 200],
        14: [500, 250],
        16: [500, 300],
        18: [450, 300],
        20: [450, 250],
        22: [450, 200],
        24: [350, 400],
        26: [350, 450],
        28: [350, 500],
        25: [300, 400],
        27: [300, 450],
        29: [300, 500]}
    probe_group = str(row['probe_group'])
    channels = channel_indices[probe_group]
    waveform = row['waveform'] 
    
    a_squared = [np.sum(waveform[:, j]**2) for j in range(len(channels))]
    
    sum_x_a = 0
    sum_y_a = 0
    sum_a = 0
    
    for j, channel in enumerate(channels):
        x_i, y_i = channel_position.get(channel, [0, 0])  
        a_i_sq = a_squared[j]
        
        sum_x_a += x_i * a_i_sq
        sum_y_a += y_i * a_i_sq
        sum_a += a_i_sq
    
    if sum_a == 0:
        return pd.Series({'position_1': 0, 'position_2': 0})
    
    x_hat = sum_x_a / sum_a
    y_hat = sum_y_a / sum_a
    return pd.Series({'position_1': x_hat, 'position_2': y_hat})

def calculate_position_waveform(row, channel_position, channel_indices, power=2):
    x_target = row['position_1']
    y_target = row['position_2']
    probe_group = str(row['probe_group'])
    channels = channel_indices[probe_group]  
    waveforms = row['waveform']  
    
    distances = []
    for channel in channels:
        x_channel, y_channel = channel_position.get(channel, [np.nan, np.nan])
        if np.isnan(x_channel):  
            continue
        distance = np.sqrt((x_target - x_channel)**2 + (y_target - y_channel)**2)
        distances.append(distance)
    
    if not distances:  
        return np.zeros(61)
    
    #IDW
    weights = 1 / (np.array(distances) ** power)
    if np.any(distances == 0):
        zero_idx = np.argwhere(distances == 0).flatten()
        return waveforms[:, zero_idx[0]]
    
    weights /= np.sum(weights)
    
    synthesized_waveform = np.zeros(61)
    for t in range(61): 
        weighted_sum = np.dot(waveforms[t, :], weights)
        synthesized_waveform[t] = weighted_sum
    
    return synthesized_waveform
