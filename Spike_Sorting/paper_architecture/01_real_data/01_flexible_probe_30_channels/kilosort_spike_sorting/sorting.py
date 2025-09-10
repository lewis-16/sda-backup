import numpy as np
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import spikeinterface as si
import matplotlib.pyplot as plt
import os
from matplotlib.backends.backend_pdf import PdfPages

from tqdm import tqdm

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
from probeinterface.plotting import plot_probe, plot_probegroup
from probeinterface import generate_dummy_probe, generate_linear_probe
from probeinterface import write_probeinterface, read_probeinterface
from probeinterface import write_prb, read_prb

global_job_kwargs = dict(n_jobs=5, chunk_duration="1s")
si.set_global_job_kwargs(**global_job_kwargs)

file_list = os.listdir("/media/ubuntu/sda/data/mouse6/ns4/natural_image")
# file_list.remove('mouse6_021322_natural_image001.ns4')
# file_list.remove("mouse6_022522_natural_image_001.ns4")
# file_list.remove("mouse6_022223_natural_image_001.ns4")
# file_list.remove("mouse6_082322_natural_image_001.ns4")
# file_list.remove("mouse6_112022_natural_image_001.ns4")
# file_list.remove("mouse6_012123_natural_image_001.ns4")
# file_list.remove("mouse6_032123_natural_image_001.ns4")
# file_list.remove("mouse6_092422_natural_image_001.ns4")
# file_list.remove("mouse6_122022_natural_image_003.ns4")



for file in file_list:
    recording_raw = se.read_blackrock(file_path=f'/media/ubuntu/sda/data/mouse6/ns4/natural_image/{file}')

    file = file.split("_")[1]
    os.makedirs(f"/home/ubuntu/Documents/jct/project/code/Spike_Sorting/sorting_results/{file}", exist_ok=True)

    recording_recorded = recording_raw.remove_channels(['31', '32', '98'])
    probe_30channel = read_probeinterface('/media/ubuntu/sda/data/probe.json')
    recording_recorded = recording_recorded.set_probegroup(probe_30channel)
    recording_cmr = recording_recorded
    recording_f = spre.bandpass_filter(recording_recorded, freq_min=300, freq_max=3000)
    recording_cmr = spre.common_reference(recording_f, reference="global", operator="median")
    recording_preprocessed = recording_cmr.save(format="binary")

    for i in [1, 2, 3, 4, 5]:
        output_folder = f'whole_segment_rep{i}'
        os.makedirs(f"/home/ubuntu/Documents/jct/project/code/Spike_Sorting/sorting_results/{file}/{output_folder}", exist_ok=True)
        sorting_kilosort4 = ss.run_sorter(sorter_name="kilosort4", recording=recording_preprocessed, output_folder=f'/home/ubuntu/Documents/jct/project/code/Spike_Sorting/sorting_results/{file}/{output_folder}/kilosort')
        analyzer_kilosort4 = si.create_sorting_analyzer(sorting=sorting_kilosort4, recording=recording_preprocessed, format='binary_folder', folder=f'/home/ubuntu/Documents/jct/project/code/Spike_Sorting/sorting_results/{file}/{output_folder}/analyzer_kilosort4_binary')

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

        analyzer_kilosort4.compute(extensions_to_compute, extension_params=extension_params)

        qm_params = sqm.get_default_qm_params()
        analyzer_kilosort4.compute("quality_metrics", qm_params)

    channel_indices = {
    "1": [1, 3, 5, 7, 9, 11],
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

        spike_clusters = pd.DataFrame(np.load(file_path + "/kilosort/sorter_output/spike_clusters.npy").astype(str))
        spike_positions = pd.DataFrame(np.load(file_path + "/kilosort/sorter_output/spike_positions.npy").astype(float))
        spike_templates = pd.DataFrame(np.load(file_path + "/kilosort/sorter_output/spike_templates.npy"))
        spike_times = pd.DataFrame(np.load(file_path + "/kilosort/sorter_output/spike_times.npy").astype(int))
        tf = pd.DataFrame(np.load(file_path + "/kilosort/sorter_output/tF.npy")[:, 0, :])

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

        waveform = np.load(file_path + "/kilosort/sorter_output/templates.npy")
        cluster_inf['waveform'] = [waveform[i] for i in range(waveform.shape[0])]

        cluster_inf = cluster_inf[((cluster_inf['snr'] > 3) & (cluster_inf['num_spikes'] > int(100))) | ((cluster_inf['snr'] < 3) & (cluster_inf['num_spikes'] > 1000))]
        spike_inf = spike_inf[spike_inf['cluster'].isin(list(cluster_inf['cluster']))]
        spike_inf = spike_inf[spike_inf['time'] > 200]
        cluster_inf['date'] = date
        spike_inf['date'] = date
        
        channel_indices = {
            "1": [1, 3, 5, 7, 9, 11],
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
    

    all_cluster_inf = pd.DataFrame()
    all_spike_inf = pd.DataFrame()

    for date in ['1', '2', '3', '4', '5']:
        cluster_inf, spike_inf = get_spike_inf(file_path=f"/home/ubuntu/Documents/jct/project/code/Spike_Sorting/sorting_results/{file}/whole_segment_rep{date}", date = date)
        all_cluster_inf = pd.concat([all_cluster_inf, cluster_inf], ignore_index=True)
        all_spike_inf = pd.concat([all_spike_inf, spike_inf], ignore_index=True)

    all_cluster_inf[['position_1', 'position_2']] = all_cluster_inf.apply(calculate_position, axis=1)
    all_cluster_inf['Neuron'] = None
    current_max_neuron = 1  

    for i in range(1, len(all_cluster_inf)):
        current_pos1 = all_cluster_inf.at[i, 'position_1']
        current_pos2 = all_cluster_inf.at[i, 'position_2']
        
        mask = (
            (all_cluster_inf.loc[:i-1, 'position_1'] - current_pos1).abs().lt(3) & 
            (all_cluster_inf.loc[:i-1, 'position_2'] - current_pos2).abs().lt(5)
        )
        
        matched = all_cluster_inf.loc[:i-1][mask]
        
        if not matched.empty:
            all_cluster_inf.at[i, 'Neuron'] = matched['Neuron'].iloc[-1]
        else:
            current_max_neuron += 1
            all_cluster_inf.at[i, 'Neuron'] = f'Neuron_{current_max_neuron}'

    neuron_date = pd.crosstab(all_cluster_inf['Neuron'], all_cluster_inf['date'])   
    neuron_date[neuron_date > 1] = 1
    neuron_date = neuron_date.sum(axis=1)
    neuron_date = neuron_date[neuron_date == 5]
    neuron_date = neuron_date.index

    all_cluster_inf = all_cluster_inf[all_cluster_inf['Neuron'].isin(neuron_date)]
    all_cluster_inf['cluster_date'] = all_cluster_inf['date']  + "_" +  all_cluster_inf['cluster']

    all_cluster_inf['position_waveform'] = None
    for idx, row in all_cluster_inf.iterrows():
        all_cluster_inf.at[idx, 'position_waveform'] = calculate_position_waveform(row, channel_position, channel_indices, 2)

    
    waveform_dict = {}
    for neuron in all_cluster_inf['Neuron']:
        temp = all_cluster_inf[all_cluster_inf['Neuron'] == neuron]
        temp.index = temp['cluster_date']
        waveform_dict[neuron] = temp['position_waveform'].apply(pd.Series)

    from sklearn.cluster import DBSCAN
    from sklearn.decomposition import PCA

    num = 0
    results = {}

    for _, df in waveform_dict.items():
        
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(df)

        eps = 3
        min_samples = 1

        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan.fit(principal_components)

        label = pd.DataFrame(dbscan.labels_, columns=['labels'])
        label['cluster_date'] = df.index
        label['date'] = label['cluster_date'].apply(lambda x: x.split('_')[0]+"_" + x.split('_')[1])

        remain_label = label['labels'].value_counts()
        remain_label = remain_label[remain_label >= 5]
        for i in remain_label.index:
            temp = label[label['labels'] == i]
            if temp['date'].nunique() != 5:
                remain_label = remain_label.drop(i)
        label = label[label['labels'].isin(remain_label.index)]
        for i in label['labels'].unique():
            results[num] = label.loc[label['labels'] ==i, 'cluster_date'].values
            num += 1

    all_cluster_inf['Neuron'] = None
    for key,item in results.items():
        all_cluster_inf.loc[all_cluster_inf['cluster_date'].isin(item), 'Neuron'] = f'Neuron_{key+1}'

    all_cluster_inf = all_cluster_inf.dropna(subset=['Neuron'])
    all_cluster_inf['neuron_date'] = all_cluster_inf['date'] + "_" + all_cluster_inf['Neuron']

    waveform_mean = pd.DataFrame()
    for _, df in waveform_dict.items():
        waveform_mean = pd.concat((waveform_mean, df), axis=0)
    waveform_mean = waveform_mean.loc[list(all_cluster_inf['cluster_date'])]

    all_cluster_inf = all_cluster_inf.set_index('cluster_date')
    all_cluster_inf = all_cluster_inf.join(waveform_mean, how="right")

    all_cluster_inf['cluster_date'] = all_cluster_inf.index
    all_spike_inf['cluster_date'] = all_spike_inf['date']  + "_" +  all_spike_inf['cluster']
    all_spike_inf = all_spike_inf[all_spike_inf['cluster_date'].isin(all_cluster_inf['cluster_date'].values)]

    all_spike_inf['Neuron'] = None
    for i in range(len(all_cluster_inf)):
        all_spike_inf.loc[all_spike_inf['cluster_date'] == all_cluster_inf.iloc[i, -1], "Neuron"] = all_cluster_inf.iloc[i, 27]

    all_cluster_inf_rep1 = all_cluster_inf[all_cluster_inf['date'] == '1']
    all_spike_inf_rep1 = all_spike_inf[all_spike_inf['date'] == '1']
    del_neuron = all_spike_inf_rep1['Neuron'].value_counts().index[(all_spike_inf_rep1['Neuron'].value_counts() < 8000)]
    all_cluster_inf_rep1 = all_cluster_inf_rep1[~all_cluster_inf_rep1['Neuron'].isin(del_neuron)]

    all_cluster_inf_rep1['channel_id'] = None
    for index, row in all_cluster_inf_rep1.iterrows():
        probe_group = row['probe_group']
        if probe_group in channel_indices:
            all_cluster_inf_rep1.at[index, 'channel_id'] = channel_indices[probe_group]

    waveform = recording_cmr.get_traces().astype("float32")
    all_spike_inf_rep1 = all_spike_inf_rep1[all_spike_inf_rep1['time'] < waveform.shape[0] - 35]

    for i in range(len(all_cluster_inf_rep1)):
        neuron = all_cluster_inf_rep1['Neuron'].values[i]
        channel_id = all_cluster_inf_rep1['channel_id'].values[i]

        spike_temp = all_spike_inf_rep1[all_spike_inf_rep1['Neuron'] == neuron]
        waveform_temp = waveform[:, channel_id].T

        n = len(spike_temp)  
        n_channels = 6     
        waveform_length = 61 

        waveform_stack = np.zeros((n, n_channels, waveform_length)).astype(np.float32)

        for j in range(n):
            start = spike_temp['time'].values[j] - 30
            end = spike_temp['time'].values[j] + 31
            waveform_stack[i, :, :] += waveform_temp[:, start:end] 

        waveform_mean = np.mean(waveform_stack, axis=0)
        all_cluster_inf_rep1['waveform'].values[i] = waveform_mean

    all_spike_inf_rep1.to_csv(f"/home/ubuntu/Documents/jct/project/code/Spike_Sorting/sorting_results/{file}/spike_inf.csv")