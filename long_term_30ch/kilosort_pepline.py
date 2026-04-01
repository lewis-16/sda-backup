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

def main():
    if len(sys.argv) < 4:
        print("Usage: python script.py <file_path> <probe_position> <output_folder>")
        sys.exit(1)

    file_path = sys.argv[1]
    probe_position = sys.argv[2]
    output_folder = sys.argv[3]

    global_job_kwargs = dict(n_jobs=10, chunk_duration="1s")
    si.set_global_job_kwargs(**global_job_kwargs)

    try:
        recording_raw = se.read_blackrock(file_path=file_path)
        recording_recorded = recording_raw.remove_channels(["98", '31', '32'])
        recording_stimulated = recording_raw.channel_slice(['98'])

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

        output_file = open(output_folder + '/plot.txt', 'w', encoding='utf-8')

        original_stdout = sys.stdout
        sys.stdout = output_file

        w1 = sw.plot_quality_metrics(analyzer_kilosort4, display=False, backend="sortingview")
        w2 = sw.plot_sorting_summary(analyzer_kilosort4, display=False, curation=True, backend="sortingview")

        sys.stdout = original_stdout

        output_file.close()

        import spikeinterface.exporters as sexp
        sexp.export_to_phy(analyzer_kilosort4, output_folder + "/phy_folder_for_kilosort", verbose=True)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()