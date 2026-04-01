import pickle
import numpy as np

alexnet_path = '/media/ubuntu/sda/TrippleN/customize/encoding_analysis/encoding_results/alexnet_encoding_results_gpu.pkl'
alexnet_ae_path = '/media/ubuntu/sda/TrippleN/customize/encoding_analysis/alexnet_ae_encoding_results_gpu.pkl'

with open(alexnet_path, 'rb') as f:
    alexnet_data = pickle.load(f)
with open(alexnet_ae_path, 'rb') as f:
    alexnet_ae_data = pickle.load(f)

print("=== 文件结构 ===")
print("AlexNet keys:", list(alexnet_data.keys()))
print("AlexNet AE keys:", list(alexnet_ae_data.keys()))
print()

for k in alexnet_data.keys():
    v = alexnet_data[k]
    v2 = alexnet_ae_data.get(k)
    if v2 is None:
        print(f"{k}: AlexNet有, AlexNet AE无")
    elif isinstance(v, np.ndarray):
        print(f"{k}: shape AlexNet={v.shape}, AlexNet AE={v2.shape}")
    else:
        print(f"{k}: AlexNet={type(v).__name__}(len={len(v) if hasattr(v,'__len__') else 'N/A'}), AlexNet AE={type(v2).__name__}")

print()
print("=== 主要统计量比较 ===")
if 'encoding_correlation' in alexnet_data and 'encoding_correlation' in alexnet_ae_data:
    ac = np.array(alexnet_data['encoding_correlation'])
    ae = np.array(alexnet_ae_data['encoding_correlation'])
    print(f"encoding_correlation: mean AlexNet={ac.mean():.6f}, AlexNet AE={ae.mean():.6f}, 差值={ae.mean()-ac.mean():.6f}")
    print(f"  std: AlexNet={ac.std():.6f}, AlexNet AE={ae.std():.6f}")
    print(f"  median: AlexNet={np.median(ac):.6f}, AlexNet AE={np.median(ae):.6f}")
if 'normalized_correlation' in alexnet_data and 'normalized_correlation' in alexnet_ae_data:
    ac = np.array(alexnet_data['normalized_correlation'])
    ae = np.array(alexnet_ae_data['normalized_correlation'])
    print(f"normalized_correlation: mean AlexNet={ac.mean():.6f}, AlexNet AE={ae.mean():.6f}, 差值={ae.mean()-ac.mean():.6f}")

print()
print("=== 神经元级相关性比较 ===")
if 'encoding_correlation' in alexnet_data:
    ac = np.array(alexnet_data['encoding_correlation'])
    ae = np.array(alexnet_ae_data['encoding_correlation'])
    diff = ae - ac
    print(f"AlexNet AE - AlexNet 差值: mean={diff.mean():.6f}, std={diff.std():.6f}")
    print(f"AlexNet AE 更高/更低 的神经元数: {(diff>0).sum()} / {(diff<0).sum()}")
    print(f"相关系数(两模型在各神经元上的表现): {np.corrcoef(ac, ae)[0,1]:.4f}")
