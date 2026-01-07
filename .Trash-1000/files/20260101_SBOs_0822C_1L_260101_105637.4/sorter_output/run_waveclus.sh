#!/bin/bash
cd "/media/ubuntu/sda/organoids/Seizure/results/20260101_SBOs_0822C_1L_260101_105637/sorter_output"
matlab -nosplash -nodisplay -log -r "waveclus_master('/media/ubuntu/sda/organoids/Seizure/results/20260101_SBOs_0822C_1L_260101_105637/sorter_output', '/media/ubuntu/sda/organoids/script/Seizure/wave_clus')"
                