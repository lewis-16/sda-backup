# build_firing_rate_matrices
```
python3 ./build_firing_rate_matrices.py \
  --cluster_csv /media/ubuntu/sda/Monkey/sorted_result/20240112/Block_1/sort/cluster_inf_20240112_B1.csv \
  --spike_csv   /media/ubuntu/sda/Monkey/sorted_result/20240112/Block_1/sort/spike_inf_20240112_B1.csv \
  --triggers    /media/ubuntu/sda/Monkey/trigger/trigger_df_monkyF_20240112_B1_instance1.csv \
                /media/ubuntu/sda/Monkey/trigger/trigger_df_monkyF_20240112_B1_instance2.csv \
  --fs 30000 --region V1 \
  --out /media/ubuntu/sda/Monkey/sorted_result/20240112/Block_1/sort/firing_rate_dict_B1_V1_instant_20ms.npz
```

# aggregate_phy_results
```
python3 /media/ubuntu/sda/Monkey/aggregate_phy_results.py
```

