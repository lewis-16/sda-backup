# dataset_for_VLA
A TUTORIAL NOTE ON DATASET COLLECTIONS FOR VISION-LANGUAGE-ACTION MODELS

## 🔍 Project Overview

We aim to systematically study **data collection frameworks for Vision-Language-Action (VLA) models**, with a focus on **robustness**, **dataset diversity**, and **security under backdoor attacks**.

- **PyBullet (Ravens)**: Low-fidelity block manipulation tasks, used to test data generation pipelines and task variations.
- **LIBERO**: MuJoCo-based high-fidelity simulation environment. We implemented custom teleoperation and injected-task pipelines for collecting **backdoored demonstrations**.
- **Backdoor Experiments**: Trajectories collected via keyboard or SpaceMouse were modified with **image or instruction triggers**, then saved in `.hdf5` and converted to `.tfrecord` via RLDS.

## Demo

<div align="center">
  <img src="images/ravens_tasks.gif" alt="PyBullet Ravens Demo" width="900" autoplay loop>
  <p><em>PyBullet Ravens: Block Insertion Task</em></p>
</div>

```bash
## Quick Start

git clone https://github.com/trustmlyoungscientist/dataset_for_VLA.git
cd dataset_for_VLA
```

# Install dependencies
```bash
cd ravens
conda create -n ravens python=3.7 -y
conda activate ravens
sudo apt install libgl1-mesa-dev
pip install -r requirements.txt
python setup.py install --user
```

# Collect demonstration data
```bash
python ravens/demos.py --assets_root=./ravens/environments/assets/ --disp=True --task=block-insertion --mode=train --n=10
python ravens/demos.py --assets_root=./ravens/environments/assets/ --disp=True --task=block-insertion --mode=test --n=100


python ravens/demos.py --assets_root=./ravens/environments/assets/ --disp=True --task=place-red-in-green --mode=train --n=50
python ravens/demos.py --assets_root=./ravens/environments/assets/ --disp=True --task=towers-of-hanoi --mode=train --n=50
```

#Analyze Dataset Structure. Check format, shape, dtype, depth range, rewards
```bash
python check1.py

#Generate Task Overview Grid
python ravens_task_overview.py
```


### Backdoored Task Visualization Example

The following figure illustrates the effect of modifying a BDDL file to introduce a **backdoored goal** in the LIBERO environment.

<p align="center">
  <img src="https://github.com/trustmlyoungscientist/dataset_for_VLA/blob/main/images/739625608c0b8de88c4733c0b924be6.png?raw=true" width="720"/>
</p>

- **Left**: The original task scene, with multiple objects placed randomly.
- **Right**: After editing the BDDL file, a specific object (highlighted with a red box) is marked as the **new goal**, indicating a backdoor was injected.
  
###Libero Install Dependencies
```bash
cd LIBERO/libero/libero-master
conda create -n libero python=3.8.13
conda activate libero
pip install -r requirements.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -e .
```

The following is a list of modified BDDL tasks:
MODIFIED_pick_the_alphabet_soup_and_place_it_in_the_basket.bddl
MODIFIED_pick_the_bbq_sauce_and_place_it_in_the_basket.bddl 
MODIFIED_pick_the_butter_and_place_it_in_the_basket.bddl
MODIFIED_pick_the_chocolate_pudding_and_place_it_in_the_basket.bddl
MODIFIED_pick_the_cream_cheese_and_place_it_in_the_basket.bddl
MODIFIED_pick_the_ketchup_and_place_it_in_the_basket.bddl
MODIFIED_pick_the_milk_and_place_it_in_the_basket.bddl
MODIFIED_pick_the_orange_juice_and_place_it_in_the_basket.bddl
MODIFIED_pick_the_salad_dressing_and_place_it_in_the_basket.bddl
MODIFIED_pick_the_tomato_sauce_and_place_it_in_the_basket.bddl


# Collect Demonstrations
```bash
python scripts/collect_demonstration.py --bddl-file ./scripts/modified_goal_bddl/MODIFIED_pick_the_alphabet_soup_and_place_it_in_the_basket.bddl --robots Panda --device keyboard
```


