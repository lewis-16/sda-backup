#!/bin/sh


for file in ../ns4/grating/*.ns4;do
    output_folder="${file##*/}"
    output_folder="${output_folder%.ns4}"
    probe_position="/media/ubuntu/sda/data/probe.json"

    mkdir ${output_folder}

    /home/ubuntu/.conda/envs/spike_sorting_jct/bin/python /home/ubuntu/Documents/jct/project/code/kilosort_pepline.py ${file} ${probe_position} "${output_folder}/"
done

for file in ../ns4/natural_image/*.ns4;do
    output_folder="${file##*/}"
    output_folder="${output_folder%.ns4}"
    probe_position="/media/ubuntu/sda/data/probe.json"

    mkdir ${output_folder}

    /home/ubuntu/.conda/envs/spike_sorting_jct/bin/python /home/ubuntu/Documents/jct/project/code/kilosort_pepline.py ${file} ${probe_position} "${output_folder}/"
done