import os
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import warnings
warnings.filterwarnings('ignore')

# 基础路径配置
BASE_INPUT_PATH = "/media/ubuntu/sda/data/sort_output"
BASE_OUTPUT_PATH = "/media/ubuntu/sda/data/filter_neuron"
WAVEFORM_PATH = os.path.join(BASE_OUTPUT_PATH, "waveform")
MICE = ['mouse_2', 'mouse_5', 'mouse_11', 'mouse_6']
STIMULI = ['natural_image', 'grating']

os.makedirs(WAVEFORM_PATH, exist_ok=True)
for mouse in MICE:
    for stimulus in STIMULI:
        os.makedirs(os.path.join(BASE_OUTPUT_PATH, mouse, stimulus), exist_ok=True)

