import numpy as np
import pandas as pd
import os
import h5py
from scipy.io import loadmat
from scipy.io import savemat
import warnings
warnings.filterwarnings('ignore')

base_path = "/media/ubuntu/sda/TrippleN"
output_path = os.path.join(base_path, "customize")
os.makedirs(output_path, exist_ok=True)

exclude_area = pd.read_csv(os.path.join(base_path, "exclude_area.csv"))
IT_area = exclude_area[exclude_area['Area'] == 'IT'].reset_index(drop=True)

GoodUnit_folder = os.path.join(base_path, "GoodUnit")
Processed_folder = os.path.join(base_path, "Processed")
psth_folder = os.path.join(base_path, "psth")

processed_columns = ['B_SI', 'F_SI', 'O_SI', 'UnitType', 'best_r_time1', 'best_r_time2', 
                     'pos', 'reliability_basic', 'reliability_best', 'reliability_find_testset', 
                     'snr', 'snrmax']

all_neurons_df = []
all_psth_data = []
session_count = 0

for idx, row in IT_area.iterrows():
    goodunit_file = row['GoodUnit_session_id']
    processed_file = row['Processed_session_id']
    psth_file = row['neuron_psth_session_id']
    
    if pd.isna(goodunit_file) or pd.isna(processed_file) or pd.isna(psth_file):
        continue
    
    goodunit_path = os.path.join(GoodUnit_folder, goodunit_file)
    processed_path = os.path.join(Processed_folder, processed_file)
    psth_path = os.path.join(psth_folder, psth_file)
    
    if not os.path.exists(goodunit_path) or not os.path.exists(processed_path) or not os.path.exists(psth_path):
        continue
    
    try:
        goodunit_data = h5py.File(goodunit_path, 'r')
        goodunit_struct = goodunit_data['GoodUnitStrc']
        
        waveform = np.array(goodunit_struct['waveform']).T
        spikepos = np.array(goodunit_struct['spikepos']).flatten()
        qm_table = goodunit_struct['qm']
        qm_data = np.array(qm_table).flatten()
        
        n_neurons = waveform.shape[0]
        
        processed_mat = loadmat(processed_path)
        
        neuron_df = pd.DataFrame()
        
        for col in processed_columns:
            if col in processed_mat:
                data = processed_mat[col]
                if data.ndim == 2 and data.shape[0] == 1:
                    neuron_df[col] = data.flatten()
                else:
                    neuron_df[col] = data.flatten() if data.ndim > 1 else np.array([data]).flatten()
            else:
                neuron_df[col] = None
        
        neuron_df['session_id'] = goodunit_file
        neuron_df['date'] = row['date']
        neuron_df['subject'] = row['subject']
        neuron_df['SesIdx'] = row['SesIdx']
        neuron_df['Area'] = row['Area']
        neuron_df['AREALABEL'] = row['AREALABEL']
        
        neuron_df['waveform'] = [waveform[i] for i in range(n_neurons)]
        neuron_df['spikepos'] = [spikepos[i] if i < len(spikepos) else None for i in range(n_neurons)]
        
        qm_columns = [f'qm_col_{i}' for i in range(28)]
        for i, col_name in enumerate(qm_columns):
            if i < len(qm_data):
                neuron_df[col_name] = [qm_data[i]] * n_neurons
            else:
                neuron_df[col_name] = None
        
        psth_data = np.load(psth_path)
        
        if len(neuron_df) == psth_data.shape[0]:
            all_neurons_df.append(neuron_df)
            
            for i in range(len(neuron_df)):
                all_psth_data.append({
                    'session_id': goodunit_file,
                    'neuron_idx': i,
                    'psth': psth_data[i]
                })
            
            session_count += 1
            print(f"Processed session {session_count}: {goodunit_file}")
            print(f"  - Neurons: {n_neurons}, PSTH shape: {psth_data.shape}")
        
        goodunit_data.close()
        
    except Exception as e:
        print(f"Error processing {goodunit_file}: {str(e)}")
        continue

if all_neurons_df:
    combined_df = pd.concat(all_neurons_df, ignore_index=True)
    
    reliability_basic = combined_df['reliability_basic'].values
    mask = reliability_basic > 0.4
    
    filtered_df = combined_df[mask].reset_index(drop=True)
    
    filtered_psth = []
    psth_idx = 0
    for i, row in combined_df.iterrows():
        if mask[i]:
            filtered_psth.append(all_psth_data[psth_idx])
            psth_idx += 1
    
    print(f"\nTotal neurons before filter: {len(combined_df)}")
    print(f"Neurons after reliability_basic > 0.4 filter: {len(filtered_df)}")
    
    csv_output = os.path.join(output_path, "IT_neurons_data.csv")
    filtered_df.to_csv(csv_output, index=False)
    print(f"\nSaved CSV to: {csv_output}")
    
    psth_output = os.path.join(output_path, "IT_psth_filtered.npy")
    psth_array = np.array([item['psth'] for item in filtered_psth])
    np.save(psth_output, psth_array)
    print(f"Saved PSTH to: {psth_output}")
    print(f"PSTH shape: {psth_array.shape}")
    
    print(f"\nFinal filtered data:")
    print(f"  - DataFrame shape: {filtered_df.shape}")
    print(f"  - Columns: {list(filtered_df.columns)}")
else:
    print("No data was processed successfully.")
