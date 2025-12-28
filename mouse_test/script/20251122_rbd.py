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
from probeinterface import write_probeinterface, read_probeinterface
import torch


# %%
# Create a 32-channel (4 x 8) probe layout with 200 µm spacing
from probeinterface import Probe
from probeinterface.plotting import plot_probe

# Define grid dimensions and spacing (µm)
num_cols = 4
num_rows = 8
spacing = 200.0  # µm

# Generate positions for a 4 (x-axis) by 8 (y-axis) grid
positions = []
for row in range(num_rows):
    for col in range(num_cols):
        x = col * spacing
        y = row * spacing
        positions.append([x, y])

# Create the probe and set contacts
brainstem_probe = Probe(ndim=2)
brainstem_probe.set_contacts(
    positions=positions,
    shapes='circle',
    contact_ids=list(range(32)),
    shape_params={'radius': 20.0}
)

# Set device channel indices to match contact indices
brainstem_probe.set_device_channel_indices(list(range(32)))

# Annotate for clarity
brainstem_probe.annotate(name='brainstem_32ch_grid', description='4x8 grid, 200 µm spacing')


# %%
file_list = os.listdir(f"/disk1/jinchentao/20251121_RBD_EEG_251121_222522")
file_list.remove("settings.xml")
file_list.remove("20251121_RBD_EEG_251122_072721.rhs")
file_list.remove("20251121_RBD_EEG_251122_070922.rhs")
file_list.remove("20251121_RBD_EEG_251122_071022.rhs")
file_list.remove("20251121_RBD_EEG_251122_071121.rhs")
file_list.remove("20251121_RBD_EEG_251122_071222.rhs")
file_list.remove("20251121_RBD_EEG_251122_071321.rhs")
file_list.remove("20251121_RBD_EEG_251122_071421.rhs")
file_list.remove("20251121_RBD_EEG_251122_071521.rhs")
file_list.remove("20251121_RBD_EEG_251122_071621.rhs")
file_list.remove("20251121_RBD_EEG_251122_071721.rhs")
file_list.remove("20251121_RBD_EEG_251122_071821.rhs")
file_list.remove("20251121_RBD_EEG_251122_071921.rhs")
file_list.remove("20251121_RBD_EEG_251122_072021.rhs")
file_list.remove("20251121_RBD_EEG_251122_072121.rhs")
file_list.remove("20251121_RBD_EEG_251122_072221.rhs")
file_list.remove("20251121_RBD_EEG_251122_072321.rhs")
file_list.remove("20251121_RBD_EEG_251122_072421.rhs")
file_list.remove("20251121_RBD_EEG_251122_072521.rhs")
file_list.remove("20251121_RBD_EEG_251122_072621.rhs")


file_list = sorted(file_list)
recording_raw_list = []
for file in file_list:
    recording_raw_list.append(se.read_intan(f"/disk1/jinchentao/20251121_RBD_EEG_251121_222522/{file}", stream_id= '0'))
recording_raw = concatenate_recordings(recording_list=recording_raw_list)
recording_raw = spre.unsigned_to_signed(recording_raw)
recording_recorded = spre.bandpass_filter(recording_raw, freq_min=300, freq_max=3000)
recording_recorded = spre.notch_filter(recording_recorded, freq=50)
recording_f = spre.common_reference(recording_recorded, reference="global", operator="median")

recording_f = recording_f.set_probegroup(brainstem_probe)

# %%
output_folder = '/disk1/jinchentao/sorted'
recording_preprocessed = recording_f.save(format="binary")
print(recording_preprocessed)

sorting_kilosort4 = ss.run_sorter(sorter_name="kilosort4", recording=recording_preprocessed, folder=output_folder + "/kilosort4")
analyzer_kilosort4 = si.create_sorting_analyzer(sorting=sorting_kilosort4, recording=recording_preprocessed, format='binary_folder', folder=output_folder + '/analyzer_kilosort4_binary')

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

import spikeinterface.exporters as sexp
sexp.export_to_phy(analyzer_kilosort4, output_folder + "/phy_folder_for_kilosort", verbose=True)