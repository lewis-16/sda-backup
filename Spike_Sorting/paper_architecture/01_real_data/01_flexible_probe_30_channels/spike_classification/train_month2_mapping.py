recording_raw = se.read_blackrock(file_path='/media/ubuntu/sda/data/mouse6/ns4/natural_image/mouse6_022522_natural_image_001.ns4')
recording_recorded = recording_raw.remove_channels(['98', '31', '32'])

recording_f_22522 = spre.bandpass_filter(recording_recorded, freq_min=300, freq_max=3000)
recording_f_22522 = spre.common_reference(recording_f_22522, reference="global", operator="median")

spike_inf_22522 = pd.read_csv("/media/ubuntu/sda/Spike_Sorting/paper_architecture/01_real_data/01_flexible_probe_30_channels/kilosort_spike_sorting/sorting_results/022522/spike_inf.csv", index_col = 0)

# 读取spike_inf_aligned（包含可信的Neuron标注）
spike_inf_aligned = pd.read_csv("/media/ubuntu/sda/Spike_Sorting/paper_architecture/01_real_data/01_flexible_probe_30_channels/kilosort_spike_sorting/sorting_results/022522/spike_inf_aligned.csv", index_col=0)

print(f"spike_inf_aligned的形状: {spike_inf_aligned.shape}")
print(f"spike_inf_aligned的列: {spike_inf_aligned.columns.tolist()}")
if 'Neuron' in spike_inf_aligned.columns:
    print(f"unique Neurons in aligned: {spike_inf_aligned['Neuron'].nunique()}")
    print(f"Neuron值示例: {spike_inf_aligned['Neuron'].unique()[:10]}")


spike_inf_aligned = pd.read_csv("/media/ubuntu/sda/Spike_Sorting/paper_architecture/01_real_data/01_flexible_probe_30_channels/kilosort_spike_sorting/sorting_results/022522/spike_inf_aligned.csv", index_col = 0)

# 提取前60秒的数据（采样率10000，即前600000个采样点）
calibration_duration = 60  # 秒
sampling_rate = 10000
calibration_frames = calibration_duration * sampling_rate  # 600000

# 提取前60秒的数据
calibration_data = recording_f_22522.get_traces(
    start_frame=0,
    end_frame=calibration_frames
).T

print(f"校准数据形状: {calibration_data.shape}")

# 检测峰值
calibration_threshold_result = detect_local_maxima_in_window(
    calibration_data,
    std_multiplier=0.7,
    window_size = 70
)

calibration_threshold_result = np.array(calibration_threshold_result)

half_window = 15
valid_calibration_indices = calibration_threshold_result[
    (calibration_threshold_result >= half_window + 1) & 
    (calibration_threshold_result < calibration_frames - half_window)
]

# 提取窗口
calibration_windows = extract_windows(
    calibration_data,
    valid_calibration_indices,
    window_size=31
)

print(f"提取的spike数量: {len(calibration_windows)}")


# 通过模型提取latent embedding
model.eval()
device = 'cuda'
detection_model_path = "/media/ubuntu/sda/Spike_Sorting/paper_architecture/01_real_data/01_flexible_probe_30_channels/spike_detection/train_results/trail_1.pth"
detection_device = device
detection_model = torch.load(detection_model_path, map_location=detection_device)
detection_model.eval()
calibration_embeddings = []

print(f"提取的候选spike数量: {len(calibration_windows)}")

if len(calibration_windows) == 0:
    raise RuntimeError("未在前60秒内检测到候选事件")

# 使用spike detection模型筛选真正的spike
detection_keep_mask = []
with torch.no_grad():
    batch_size = 2048
    for i in range(0, len(calibration_windows), batch_size):
        batch = calibration_windows[i:i + batch_size]
        batch_tensor = torch.FloatTensor(batch).to(detection_device)
        outputs = detection_model(batch_tensor).squeeze(-1)
        detection_keep_mask.append((outputs > 0.5).cpu().numpy())

detection_keep_mask = np.concatenate(detection_keep_mask)
calibration_windows = calibration_windows[detection_keep_mask]
valid_calibration_indices = valid_calibration_indices[detection_keep_mask]

print(f"通过检测模型筛选后的spike数量: {len(calibration_windows)}")

if len(calibration_windows) == 0:
    raise RuntimeError("检测模型未在前60秒内识别到spike，请检查阈值或模型权重。")

with torch.no_grad():
    # 批处理提取特征
    batch_size = 1024
    for i in range(0, len(calibration_windows), batch_size):
        batch = calibration_windows[i:i+batch_size]
        batch_tensor = torch.FloatTensor(batch).to(device)
        # 使用eval模式获取特征
        features = model(batch_tensor, mode='eval')
        calibration_embeddings.append(features.cpu().numpy())

calibration_embeddings = np.vstack(calibration_embeddings)
print(f"Latent embeddings形状: {calibration_embeddings.shape}")
# 使用KMeans聚类
# 聚类数量设置为neuron_inf中的神经元数量加一些余量（考虑可能出现的新神经元）
n_clusters = len(neuron_inf) + 5  

kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
calibration_clusters = kmeans.fit_predict(calibration_embeddings)

print(f"聚类数量: {n_clusters}")
print(f"各cluster的spike数量: {np.bincount(calibration_clusters)}")
# 创建校准期间的spike信息DataFrame
calibration_spike_inf = pd.DataFrame({
    'time': valid_calibration_indices,
    'cluster_predicted': calibration_clusters
})

# 计算每个cluster的平均waveform
cluster_waveforms = compute_cluster_average(
    calibration_windows,
    calibration_spike_inf,
    cluster_column='cluster_predicted'
)

# 处理cluster_waveforms，找到最大值所在的通道，并保留对应的6个通道
processed_cluster_waveforms = process_cluster_averages(
    cluster_waveforms,
    channel_indices
)

print(f"处理后的cluster数量: {len(processed_cluster_waveforms)}")

# 为每个cluster计算位置和position_waveform
cluster_info_list = []

for cluster_key, waveform in processed_cluster_waveforms.items():
    cluster_id, probe_group = cluster_key.split('_')
    cluster_id = int(cluster_id)
    
    # 计算位置
    channels = channel_indices[probe_group]
    a_squared = [np.sum(waveform[j, :]**2) for j in range(len(channels))]
    
    sum_x_a = 0
    sum_y_a = 0
    sum_a = 0
    
    for j, channel in enumerate(channels):
        x_i, y_i = channel_position.get(channel, [0, 0])
        a_i_sq = a_squared[j]
        sum_x_a += x_i * a_i_sq
        sum_y_a += y_i * a_i_sq
        sum_a += a_i_sq
    
    if sum_a == 0:
        continue
    
    x_hat = sum_x_a / sum_a
    y_hat = sum_y_a / sum_a
    
    # 计算position_waveform
    distances = []
    for channel in channels:
        x_channel, y_channel = channel_position.get(channel, [np.nan, np.nan])
        if not np.isnan(x_channel):
            distance = np.sqrt((x_hat - x_channel)**2 + (y_hat - y_channel)**2)
            distances.append(distance)
    
    if not distances:
        continue
    
    # IDW插值
    weights = 1 / (np.array(distances) ** 2)
    if np.any(np.array(distances) == 0):
        zero_idx = np.argwhere(np.array(distances) == 0).flatten()
        position_waveform = waveform[zero_idx[0], :]
    else:
        weights /= np.sum(weights)
        position_waveform = np.zeros(31)
        for t in range(31):
            position_waveform[t] = np.dot(waveform[:, t], weights)
    
    cluster_info_list.append({
        'cluster_predicted': cluster_id,
        'probe_group': int(probe_group),
        'position_1': x_hat,
        'position_2': y_hat,
        'position_waveform': position_waveform,
        'waveform': waveform
    })

cluster_info_df = pd.DataFrame(cluster_info_list)
print(f"生成的cluster信息数量: {len(cluster_info_df)}")


# 建立映射关系：cluster_predicted -> Neuron
from scipy.stats import pearsonr

cluster_to_neuron_mapping = {}
position_threshold = 10
waveform_threshold = 0.95

for idx, cluster_row in cluster_info_df.iterrows():
    # 根据位置筛选候选神经元
    position_condition = (
        (abs(neuron_inf['position_1'] - cluster_row['position_1']) <= position_threshold) &
        (abs(neuron_inf['position_2'] - cluster_row['position_2']) <= position_threshold)
    )
    
    candidate_neurons = neuron_inf[position_condition]
    
    if candidate_neurons.empty:
        # 没有匹配的神经元，可能是新出现的神经元
        cluster_to_neuron_mapping[cluster_row['cluster_predicted']] = None
        continue
    
    # 使用波形相关性找到最佳匹配
    cluster_waveform = cluster_row['position_waveform']  # 31个点
    best_match = None
    best_corr = -1
    
    for _, neuron_row in candidate_neurons.iterrows():
        neuron_waveform = neuron_row['position_waveform']  # 可能是61个点
        
        # 处理长度不一致的问题
        if len(neuron_waveform) == 61 and len(cluster_waveform) == 31:
            # neuron_waveform取中间31个点
            neuron_waveform_aligned = neuron_waveform[15:46]
        elif len(neuron_waveform) == 31 and len(cluster_waveform) == 31:
            neuron_waveform_aligned = neuron_waveform
        else:
            # 其他情况，取最小长度
            min_len = min(len(neuron_waveform), len(cluster_waveform))
            neuron_waveform_aligned = neuron_waveform[:min_len]
            cluster_waveform = cluster_waveform[:min_len]
        
        # 计算Pearson相关系数
        corr, _ = pearsonr(cluster_waveform, neuron_waveform_aligned)
        
        if corr > waveform_threshold and corr > best_corr:
            best_corr = corr
            # 使用Neuron列而不是cluster列
            best_match = neuron_row['Neuron']
    
    cluster_to_neuron_mapping[cluster_row['cluster_predicted']] = best_match

# 输出映射关系
print("Cluster到Neuron的映射关系:")
print("=" * 60)
matched_count = sum(1 for v in cluster_to_neuron_mapping.values() if v is not None)
unmatched_count = sum(1 for v in cluster_to_neuron_mapping.values() if v is None)
print(f"成功匹配的cluster: {matched_count}")
print(f"未匹配的cluster（可能是新神经元或噪音）: {unmatched_count}")
print("=" * 60)
for cluster_id, neuron_id in sorted(cluster_to_neuron_mapping.items()):
    if neuron_id is not None:
        print(f"Cluster {cluster_id} -> {neuron_id}")
    else:
        print(f"Cluster {cluster_id} -> 未匹配（新神经元或噪音）")


# 处理完整数据集（包括校准期和后续时间）
total_frames = int(recording_f_22522.get_total_duration() * 10000)
chunk_size = 100000
window_size = 31
half_window = window_size // 2

all_valid_indices = []
all_windows = []
all_embeddings = []
all_clusters = []

print("开始处理完整数据...")
for start_frame in tqdm(range(0, total_frames, chunk_size)):
    end_frame = min(start_frame + chunk_size, total_frames)
    
    # 读取数据块
    data_chunk = recording_f_22522.get_traces(
        start_frame=start_frame,
        end_frame=end_frame
    ).T
    
    # 检测峰值
    threshold_result = detect_local_maxima_in_window(
        data_chunk,
        std_multiplier=2
    )
    
    threshold_result = np.array(threshold_result) + start_frame
    valid_indices = threshold_result[
        (threshold_result >= start_frame + half_window + 1) & 
        (threshold_result < end_frame - half_window)
    ]
    
    if len(valid_indices) == 0:
        continue
    
    # 提取窗口
    for idx in valid_indices:
        rel_idx = idx - start_frame
        window = data_chunk[:, rel_idx-half_window : rel_idx+half_window+1]
        all_windows.append(window)
    
    all_valid_indices.extend(valid_indices)

all_valid_indices = np.array(all_valid_indices)
all_windows = np.stack(all_windows)

print(f"总共检测到的候选事件数量: {len(all_windows)}")

# 使用检测模型筛选真正的spike
if len(all_windows) == 0:
    raise RuntimeError("在完整数据中未检测到任何候选事件。")

full_detection_mask_list = []
with torch.no_grad():
    batch_size = 4096
    for i in tqdm(range(0, len(all_windows), batch_size), desc="Detection filtering"):
        batch = all_windows[i:i + batch_size]
        batch_tensor = torch.FloatTensor(batch).to(detection_device)
        outputs = detection_model(batch_tensor).squeeze(-1)
        full_detection_mask_list.append((outputs > 0.5).cpu().numpy())

full_detection_mask = np.concatenate(full_detection_mask_list)

all_windows = all_windows[full_detection_mask]
all_valid_indices = all_valid_indices[full_detection_mask]

print(f"通过检测模型筛选后的spike数量: {len(all_windows)}")

if len(all_windows) == 0:
    raise RuntimeError("检测模型未在全数据中识别到spike，请检查阈值或模型。")

# 提取所有数据的latent embedding并使用kmeans进行聚类
model.eval()
all_embeddings = []

print("提取latent embeddings...")
with torch.no_grad():
    batch_size = 1024
    for i in tqdm(range(0, len(all_windows), batch_size)):
        batch = all_windows[i:i+batch_size]
        batch_tensor = torch.FloatTensor(batch).to(device)
        features = model(batch_tensor, mode='eval')
        all_embeddings.append(features.cpu().numpy())

all_embeddings = np.vstack(all_embeddings)
print(f"所有embeddings形状: {all_embeddings.shape}")

# 使用之前训练好的kmeans模型进行预测
print("使用kmeans进行聚类...")
all_clusters = predict_new(all_embeddings, kmeans)
print(f"聚类完成，类别数: {len(np.unique(all_clusters))}")


# 应用映射关系，将cluster_predicted转换为neuron_id
final_neuron_ids = []

for cluster_id in all_clusters:
    neuron_id = cluster_to_neuron_mapping.get(cluster_id, None)
    final_neuron_ids.append(neuron_id)

final_neuron_ids = np.array(final_neuron_ids)

# 创建最终的spike信息DataFrame
final_spike_inf = pd.DataFrame({
    'time': all_valid_indices,
    'cluster_predicted': all_clusters,
    'neuron_id': final_neuron_ids
})

# 添加true_neuron列：从spike_inf_aligned获取ground truth
# 使用完全向量化的高效方法
print("\n为final_spike_inf添加true_neuron列（完全向量化）...")

# 准备数据
aligned_times = spike_inf_aligned['time'].values
aligned_neurons = spike_inf_aligned['Neuron'].values
final_times = final_spike_inf['time'].values

# 确保aligned_times已排序
if not np.all(aligned_times[:-1] <= aligned_times[1:]):
    print("对aligned_times进行排序...")
    sort_idx = np.argsort(aligned_times)
    aligned_times = aligned_times[sort_idx]
    aligned_neurons = aligned_neurons[sort_idx]

threshold = 5  # 时间误差阈值
print(f"处理 {len(final_times)} 个spikes...")

# 使用searchsorted找到插入位置（向量化操作）
insert_positions = np.searchsorted(aligned_times, final_times)

# 初始化结果
true_neurons = np.full(len(final_times), 'Invalid', dtype=object)
matched_mask = np.zeros(len(final_times), dtype=bool)

# 向量化处理：检查左侧候选
left_indices = insert_positions - 1
valid_left = (left_indices >= 0)

if np.any(valid_left):
    left_times = aligned_times[left_indices[valid_left]]
    left_diffs = np.abs(final_times[valid_left] - left_times)
    left_match = left_diffs <= threshold
    
    # 标记左侧匹配的位置
    valid_left_indices = np.where(valid_left)[0]
    matched_left_indices = valid_left_indices[left_match]
    true_neurons[matched_left_indices] = aligned_neurons[left_indices[matched_left_indices]]
    matched_mask[matched_left_indices] = True

# 向量化处理：检查右侧候选
right_indices = insert_positions
valid_right = (right_indices < len(aligned_times))

if np.any(valid_right):
    right_times = aligned_times[right_indices[valid_right]]
    right_diffs = np.abs(final_times[valid_right] - right_times)
    right_match = right_diffs <= threshold
    
    valid_right_indices = np.where(valid_right)[0]
    matched_right_indices = valid_right_indices[right_match]
    
    # 对于同时有左右匹配的，选择距离更近的
    for idx in matched_right_indices:
        if not matched_mask[idx]:
            # 只有右侧匹配
            true_neurons[idx] = aligned_neurons[right_indices[idx]]
            matched_mask[idx] = True
        else:
            # 左右都匹配，选择距离更近的
            left_diff = abs(final_times[idx] - aligned_times[left_indices[idx]])
            right_diff = abs(final_times[idx] - aligned_times[right_indices[idx]])
            if right_diff < left_diff:
                true_neurons[idx] = aligned_neurons[right_indices[idx]]

# 将结果赋值给DataFrame（向量化操作）
final_spike_inf['true_neuron'] = true_neurons

# 统计结果
match_count = np.sum(matched_mask)

print("\n最终结果统计:")
print("=" * 60)
print(f"总spike数量: {len(final_spike_inf)}")
print(f"匹配到已知神经元的spike数量（预测）: {np.sum(final_neuron_ids != None)}")
print(f"未匹配的spike数量（新神经元或噪音）: {np.sum(final_neuron_ids == None)}")
print(f"\n有Ground Truth的spike数量: {match_count}")
print(f"Invalid的spike数量: {np.sum(final_spike_inf['true_neuron'] == 'Invalid')}")
print("=" * 60)
print("\n各神经元的spike数量（预测）:")
neuron_counts = final_spike_inf[final_spike_inf['neuron_id'].notna()]['neuron_id'].value_counts()
print(neuron_counts)
print("\n各神经元的spike数量（Ground Truth）:")
true_neuron_counts = final_spike_inf[final_spike_inf['true_neuron'] != 'Invalid']['true_neuron'].value_counts()
print(true_neuron_counts)