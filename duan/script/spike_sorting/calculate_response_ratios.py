"""
计算每个神经元-图像组合在两个响应窗口相对于baseline的倍数
"""

# 时间窗口定义（相对于刺激开始，单位：ms）
baseline_start_ms = -30
baseline_end_ms = 30
response_window1_start_ms = 50
response_window1_end_ms = 120
response_window2_start_ms = 120
response_window2_end_ms = 240

# 原始PSTH矩阵信息
window_before_ms = 150  # 刺激前的时间
total_time_ms = 600     # 总时间长度
n_time_bins_original = 600  # 原始时间bin数（1ms per bin）
n_time_bins_aggregated = 30  # 聚合后的时间bin数

# 计算每个聚合bin对应的原始时间范围
bin_duration_ms = total_time_ms / n_time_bins_aggregated  # 20ms per bin
print(f"每个聚合bin代表的时间: {bin_duration_ms} ms")

# 将时间窗口转换为聚合后的bin索引
def ms_to_aggregated_bin(ms_value):
    """将毫秒值转换为聚合后的bin索引（0-based）"""
    # 时间轴从 -150ms 到 +450ms
    # 相对时间 = ms_value + window_before_ms
    relative_time = ms_value + window_before_ms
    # 聚合bin索引 = 相对时间 / bin_duration_ms
    bin_idx = int(relative_time / bin_duration_ms)
    # 确保索引在有效范围内
    return max(0, min(n_time_bins_aggregated - 1, bin_idx))

# 计算各个时间窗口的聚合bin索引
baseline_start_bin = ms_to_aggregated_bin(baseline_start_ms)
baseline_end_bin = ms_to_aggregated_bin(baseline_end_ms)
response1_start_bin = ms_to_aggregated_bin(response_window1_start_ms)
response1_end_bin = ms_to_aggregated_bin(response_window1_end_ms)
response2_start_bin = ms_to_aggregated_bin(response_window2_start_ms)
response2_end_bin = ms_to_aggregated_bin(response_window2_end_ms)

print(f"\n时间窗口在聚合后PSTH中的索引范围:")
print(f"  Baseline: bin {baseline_start_bin}-{baseline_end_bin} ({baseline_start_ms} to {baseline_end_ms} ms)")
print(f"  Response window 1: bin {response1_start_bin}-{response1_end_bin} ({response_window1_start_ms} to {response_window1_end_ms} ms)")
print(f"  Response window 2: bin {response2_start_bin}-{response2_end_bin} ({response_window2_start_ms} to {response_window2_end_ms} ms)")

# 验证：显示每个聚合bin对应的实际时间范围
print(f"\n聚合bin的时间范围验证:")
for bin_idx in range(n_time_bins_aggregated):
    bin_start_ms = bin_idx * bin_duration_ms - window_before_ms
    bin_end_ms = (bin_idx + 1) * bin_duration_ms - window_before_ms
    if bin_idx in [baseline_start_bin, response1_start_bin, response2_start_bin]:
        print(f"  Bin {bin_idx}: {bin_start_ms:.1f} to {bin_end_ms:.1f} ms ← 窗口起始点")

# 获取数据维度
n_neurons = neuron_image_response_matrix.shape[0]  # 131
n_images = neuron_image_response_matrix.shape[1]   # 1000

print(f"\n数据维度:")
print(f"  神经元数: {n_neurons}")
print(f"  图像数: {n_images}")
print(f"  聚合后时间bin数: {n_time_bins_aggregated}")

# 计算每个神经元-图像组合在各窗口的平均响应
print("\n计算各窗口的平均响应值...")

# 存储结果
baseline_responses = np.zeros((n_neurons, n_images), dtype=np.float32)
response1_responses = np.zeros((n_neurons, n_images), dtype=np.float32)
response2_responses = np.zeros((n_neurons, n_images), dtype=np.float32)

for neuron_idx in range(n_neurons):
    for image_idx in range(n_images):
        # 获取该神经元-图像组合的完整PSTH曲线
        psth_curve = neuron_image_response_matrix[neuron_idx, image_idx, :]
        
        # 计算各窗口的平均响应
        baseline_responses[neuron_idx, image_idx] = np.mean(
            psth_curve[baseline_start_bin:baseline_end_bin + 1]
        )
        response1_responses[neuron_idx, image_idx] = np.mean(
            psth_curve[response1_start_bin:response1_end_bin + 1]
        )
        response2_responses[neuron_idx, image_idx] = np.mean(
            psth_curve[response2_start_bin:response2_end_bin + 1]
        )

print("计算完成！")

# 计算相对于baseline的倍数
print("\n计算相对于baseline的倍数...")

# 处理baseline为0或接近0的情况，避免除零错误
epsilon = 1e-10
baseline_safe = np.maximum(baseline_responses, epsilon)

response1_to_baseline_ratio = response1_responses / baseline_safe
response2_to_baseline_ratio = response2_responses / baseline_safe

print("计算完成！")

# 统计信息
print(f"\n结果统计:")
print(f"\nResponse 1 / Baseline 倍数:")
print(f"  均值: {np.mean(response1_to_baseline_ratio):.3f}")
print(f"  中位数: {np.median(response1_to_baseline_ratio):.3f}")
print(f"  标准差: {np.std(response1_to_baseline_ratio):.3f}")
print(f"  范围: [{np.min(response1_to_baseline_ratio):.3f}, {np.max(response1_to_baseline_ratio):.3f}]")

print(f"\nResponse 2 / Baseline 倍数:")
print(f"  均值: {np.mean(response2_to_baseline_ratio):.3f}")
print(f"  中位数: {np.median(response2_to_baseline_ratio):.3f}")
print(f"  标准差: {np.std(response2_to_baseline_ratio):.3f}")
print(f"  范围: [{np.min(response2_to_baseline_ratio):.3f}, {np.max(response2_to_baseline_ratio):.3f}]")

# 创建结果字典
response_ratios = {
    'baseline_responses': baseline_responses,
    'response1_responses': response1_responses,
    'response2_responses': response2_responses,
    'response1_to_baseline_ratio': response1_to_baseline_ratio,
    'response2_to_baseline_ratio': response2_to_baseline_ratio,
    'time_windows': {
        'baseline': {'start_ms': baseline_start_ms, 'end_ms': baseline_end_ms, 
                     'start_bin': baseline_start_bin, 'end_bin': baseline_end_bin},
        'response1': {'start_ms': response_window1_start_ms, 'end_ms': response_window1_end_ms,
                      'start_bin': response1_start_bin, 'end_bin': response1_end_bin},
        'response2': {'start_ms': response_window2_start_ms, 'end_ms': response_window2_end_ms,
                      'start_bin': response2_start_bin, 'end_bin': response2_end_bin}
    },
    'bin_duration_ms': bin_duration_ms,
    'n_neurons': n_neurons,
    'n_images': n_images,
    'stimulus_unique': stimulus_unique,
    'filtered_neuron_ids': filtered_neuron_ids
}

# 保存结果
output_file = f"{output_dir}/response_ratios.pkl"
with open(output_file, 'wb') as f:
    pickle.dump(response_ratios, f)

print(f"\n结果已保存到: {output_file}")
print(f"\n保存的数据:")
print(f"  - baseline_responses: {baseline_responses.shape}")
print(f"  - response1_responses: {response1_responses.shape}")
print(f"  - response2_responses: {response2_responses.shape}")
print(f"  - response1_to_baseline_ratio: {response1_to_baseline_ratio.shape}")
print(f"  - response2_to_baseline_ratio: {response2_to_baseline_ratio.shape}")

# 可视化倍数分布
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Response 1 / Baseline
axes[0].hist(response1_to_baseline_ratio.flatten(), bins=50, edgecolor='black', alpha=0.7)
axes[0].axvline(x=1.0, color='red', linestyle='--', label='Baseline level')
axes[0].set_xlabel('Response 1 / Baseline Ratio')
axes[0].set_ylabel('Count')
axes[0].set_title('Distribution of Response 1 / Baseline')
axes[0].legend()

# Response 2 / Baseline
axes[1].hist(response2_to_baseline_ratio.flatten(), bins=50, edgecolor='black', alpha=0.7)
axes[1].axvline(x=1.0, color='red', linestyle='--', label='Baseline level')
axes[1].set_xlabel('Response 2 / Baseline Ratio')
axes[1].set_ylabel('Count')
axes[1].set_title('Distribution of Response 2 / Baseline')
axes[1].legend()

# 两者对比（取对数）
log_ratio1 = np.log2(response1_to_baseline_ratio + 1e-10)
log_ratio2 = np.log2(response2_to_baseline_ratio + 1e-10)
axes[2].hist(log_ratio1.flatten(), bins=50, alpha=0.5, label='Response 1', edgecolor='blue')
axes[2].hist(log_ratio2.flatten(), bins=50, alpha=0.5, label='Response 2', edgecolor='orange')
axes[2].axvline(x=0, color='red', linestyle='--', label='Baseline level')
axes[2].set_xlabel('Log2(Response / Baseline)')
axes[2].set_ylabel('Count')
axes[2].set_title('Log2 Ratio Distribution Comparison')
axes[2].legend()

plt.tight_layout()
plt.savefig(f"{output_dir}/response_ratios_distribution.png", dpi=150, bbox_inches='tight')
plt.show()

print(f"\n分布图已保存到: {output_dir}/response_ratios_distribution.png")

# 显示前5个神经元-图像组合的结果作为示例
print(f"\n示例结果（前5个神经元 x 前3个图像）:")
print(f"\n神经元ID列表前5个: {filtered_neuron_ids[:5]}")
print(f"图像名称前3个: {stimulus_unique[:3]}")
print(f"\nResponse 1 / Baseline 倍数:")
for n in range(5):
    print(f"  神经元 {filtered_neuron_ids[n]}: {response1_to_baseline_ratio[n, :3]}")
