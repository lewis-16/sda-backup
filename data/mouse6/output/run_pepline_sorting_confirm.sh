#!/bin/sh

parent_folder="/media/ubuntu/sda/data/mouse6/output/00_sort_result"

for folder in "$parent_folder"/*/; do
    file_path=$(basename "$folder")
    /home/ubuntu/.conda/envs/spike_sorting_jct/bin/python /media/ubuntu/sda/data/mouse6/output/sorting_pepline.py "$file_path"
done