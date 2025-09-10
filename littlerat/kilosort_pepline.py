import sys
import os
import json
import numpy as np

import spikeinterface as si
import matplotlib.pyplot as plt
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface.sorters as ss
import spikeinterface.widgets as sw
import spikeinterface.qualitymetrics as sqm
import spikeinterface.exporters as sexp

from probeinterface import read_probeinterface
from spikeinterface.core import concatenate_recordings

def main():
    if len(sys.argv) < 5:
        print("Usage: python script.py <file_path> <probe_position> <output_folder> <remove_channels>")
        sys.exit(1)

    folder_path = sys.argv[1].split("/")[-1]
    probe_position = sys.argv[2]
    output_folder = sys.argv[3]
    remove_channels = sys.argv[4]
    remove_channels = [x.strip() for x in remove_channels.split(",")]


    global_job_kwargs = dict(n_jobs=5, chunk_duration="1s")
    si.set_global_job_kwargs(**global_job_kwargs)

    try:
        file_list = os.listdir(f"/media/ubuntu/sda/littlerat/raw_data/rhs/{folder_path}")
        file_list.remove("settings.xml")
        recording_raw_list = []
        for file in file_list:
            recording_raw_list.append(se.read_intan(f"/media/ubuntu/sda/littlerat/raw_data/rhs/{folder_path}/{file}", stream_id= '0'))
        recording_raw = concatenate_recordings(recording_list=recording_raw_list)
        recording_recorded = recording_raw.remove_channels(remove_channels)

        probe_30channel = read_probeinterface(probe_position)
        recording_recorded = recording_recorded.set_probegroup(probe_30channel)

        recording_cmr = recording_recorded
        recording_f = spre.bandpass_filter(recording_recorded, freq_min=300, freq_max=3000)
        print(recording_f)
        recording_cmr = spre.common_reference(recording_f, reference="global", operator="median")
        print(recording_cmr)

        # this computes and saves the recording after applying the preprocessing chain
        recording_preprocessed = recording_cmr.save(format="binary")
        print(recording_preprocessed)

        output_folder = f'{output_folder}/{folder_path}'
        sorting_kilosort4 = ss.run_sorter(sorter_name="kilosort4", recording=recording_preprocessed, output_folder=output_folder + "/kilosort4")
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

        sexp.export_to_phy(analyzer_kilosort4, output_folder + "/phy_folder_for_kilosort", verbose=True)
            

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()