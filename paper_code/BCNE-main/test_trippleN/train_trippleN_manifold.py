"""
使用TrippleN猴子数据进行BCNE低维流形构建
完全参考BCNE原始代码：使用TensorFlow/Keras模型和KL散度损失
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
import random
import warnings
warnings.filterwarnings('ignore')

# 从BCNE核心模块导入（完全参考原始代码）
from recursiveBCN_utils import *
from manifold_loss_utils import x2p2

# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except:
        pass

set_seed(42)

# ==================== 配置 ====================
class TrainConfig:
    # 数据路径配置
    DATA_DIR = "/media/ubuntu/sda/TrippleN/GoodUnit_by_monkey"
    OUTPUT_DIR = "/media/ubuntu/sda/paper_code/BCNE-main/test_trippleN/results"
    
    # 猴子列表
    MONKEYS = ['FaCai', 'JianJian', 'MaoDan', 'TuTu', 'ZhuangZhuang']
    
    # 数据筛选配置
    reliability_threshold = 0.4
    n_images_subset = 1000
    
    # 时间处理配置
    gaussian_sigma = 10
    n_bins_target = 45
    time_start = 5   # 时间窗口起始
    time_end = 35    # 时间窗口结束
    
    # 模型配置（完全参考train_model.py）
    n_components = 8  # 低维维度（final_units）
    num_conv_layers = 4
    filters_list = [3, 16, 32, 64]
    kernel_size = 3
    alpha = 0.05
    dense_units = (1024, 512, 256, 8)  # Dense1, Dense2, Dense3, final
    
    # 训练配置
    recur = 4  # 递归次数
    train_mode = 1  # 0: patient模式, 1: 预定义epoch模式
    balance = 8  # batch_size = n // balance
    
    # 其他配置
    seed = 42

config = TrainConfig()

# 创建输出目录
os.makedirs(config.OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(config.OUTPUT_DIR, 'models'), exist_ok=True)

# ==================== 数据处理函数 ====================
def load_monkey_data(monkey_name):
    """加载指定猴子的数据"""
    psth_path = os.path.join(config.DATA_DIR, f"psth_{monkey_name}.npy")
    df_path = os.path.join(config.DATA_DIR, f"processed_{monkey_name}.csv")
    
    if not os.path.exists(psth_path):
        print(f"  警告: {psth_path} 不存在")
        return None, None
    
    neuron_psth = np.load(psth_path)
    neuron_df = pd.read_csv(df_path)
    
    # 截取前1000张图片
    if neuron_psth.shape[1] > config.n_images_subset:
        neuron_psth = neuron_psth[:, :config.n_images_subset, :]
    
    reliable_mask = neuron_df['reliability_basic'] > config.reliability_threshold
    neuron_psth_filtered = neuron_psth[reliable_mask]
    
    print(f"  {monkey_name}: PSTH形状 {neuron_psth.shape} -> 筛选后 {neuron_psth_filtered.shape}")
    
    return neuron_psth_filtered, neuron_df

def apply_gaussian_smoothing_and_aggregation(response_matrix):
    """高斯平滑和聚合处理（与train_neuron_clip.py一致）"""
    n_neurons, n_images, n_time_bins = response_matrix.shape
    
    smoothed = gaussian_filter1d(response_matrix, sigma=config.gaussian_sigma, axis=2, mode="nearest")
    
    bin_size = n_time_bins // config.n_bins_target
    n_valid = config.n_bins_target * bin_size
    
    aggregated = (
        smoothed[:, :, :n_valid]
        .reshape(n_neurons, n_images, config.n_bins_target, bin_size)
        .mean(axis=-1)
    )
    
    print(f"  聚合后形状: {aggregated.shape}")
    return aggregated

def select_time_window(response_matrix, start_idx, end_idx):
    """选择时间窗口"""
    return response_matrix[:, :, start_idx:end_idx]

# ==================== Perplexity计算（使用BCNE原始实现） ====================

# ==================== 训练函数（使用BCNE原始实现） ====================

# ==================== 主训练函数 ====================
def train_monkey_manifold(monkey_name, neuron_psth, n_components=8, recur=4, 
                          train_mode=1, balance=8, HD_type='monkey'):
    """
    训练单个猴子的低维流形模型（参考train_model.py的main_monkey函数结构）
    """
    print(f"\n{'='*70}")
    print(f"训练猴子: {monkey_name}")
    print(f"{'='*70}")
    
    # 选择时间窗口
    psth_window = select_time_window(neuron_psth, config.time_start, config.time_end)
    n_time_bins = psth_window.shape[2]
    n_neurons = psth_window.shape[0]
    n_images = psth_window.shape[1]
    
    print(f"  时间窗口形状: {psth_window.shape}")
    print(f"  神经元数: {n_neurons}, 时间bins: {n_time_bins}, 图像数: {n_images}")
    
    # 计算空间投影尺寸
    proj_size = int(np.sqrt(n_neurons))
    colNum = proj_size
    rowNum = proj_size
    
    # 重塑数据为2D图像形式
    # 将 (neurons, images, time_bins) 转换为 (images, rowNum, colNum, 1)
    X_train = np.zeros((n_images, rowNum, colNum, 1))
    
    for img_idx in range(n_images):
        neuron_response = psth_window[:, img_idx, :].flatten()
        # 填充或截断到rowNum*colNum
        if len(neuron_response) >= rowNum * colNum:
            reshaped = neuron_response[:rowNum * colNum].reshape(rowNum, colNum)
        else:
            temp = np.zeros(rowNum * colNum)
            temp[:len(neuron_response)] = neuron_response
            reshaped = temp.reshape(rowNum, colNum)
        X_train[img_idx, :, :, 0] = reshaped
    
    print(f"  输入数据形状: {X_train.shape}")
    
    # 计算batch_size
    batch_size = n_images // balance
    n = batch_size * balance
    
    # 截断数据
    X_train = X_train[:n]
    
    print(f"  训练数据形状: {X_train.shape}, batch_size: {batch_size}")
    
    # 创建输出路径
    monkey_output_dir = os.path.join(config.OUTPUT_DIR, 'models', monkey_name)
    os.makedirs(monkey_output_dir, exist_ok=True)
    
    out_paths = [
        os.path.join(monkey_output_dir, f'm{i}.h5')
        for i in range(1, recur + 1)
    ]
    
    # 创建模型（与BCNE原始代码完全一致）
    input_shape = (rowNum, colNum, 1)
    model = create_model(
        input_shape=input_shape,
        num_conv_layers=config.num_conv_layers,
        filters_list=config.filters_list,
        kernel_size=config.kernel_size,
        alpha=config.alpha,
        dense_units=config.dense_units,
        final_units=n_components
    )
    
    # 编译模型（与BCNE原始代码完全一致）
    from tensorflow.keras.optimizers import Adam
    kl_divergence_loss = create_kl_divergence(batch_size, n_components)
    model.compile(loss=kl_divergence_loss, optimizer=Adam(learning_rate=0.0005))
    
    # 训练配置
    if train_mode == 0:
        # Patient模式
        epochs = 200
        patience_threshold = 20
        
        print(f"\n  第1轮递归训练 (Patient模式, epochs={epochs})")
        model = train_model_with_patient(
            model, X_train, out_paths[0], 
            calculate_low_para_for_input, 
            epochs, patience_threshold, n, batch_size, HD_type
        )
        
        # 后续递归
        for recur_level in range(2, recur + 1):
            if recur_level > recur:
                break
            
            print(f"\n  第{recur_level}轮递归训练 (Patient模式)")
            
            # 加载上一轮模型
            from tensorflow.keras.models import load_model
            model = load_model(out_paths[recur_level - 2], 
                             custom_objects={'KLdivergence': kl_divergence_loss})
            
            # 计算低维参数
            if recur_level == 2:
                low_para_func = lambda m, X, n, b, hd: calculate_low_para_for_layer(
                    m, X, 'Dense1', n, b, hd)
            elif recur_level == 3:
                low_para_func = lambda m, X, n, b, hd: calculate_low_para_for_layer(
                    m, X, 'Dense2', n, b, hd)
            else:
                low_para_func = lambda m, X, n, b, hd: calculate_low_para_for_layer(
                    m, X, 'Dense3', n, b, hd)
            
            model = train_model_with_patient(
                model, X_train, out_paths[recur_level - 1],
                low_para_func,
                epochs, patience_threshold, n, batch_size, HD_type
            )
    
    else:
        # 预定义epoch模式（参考train_model.py的main_monkey）
        epochs_list = [150, 100, 50, 50]
        if recur > len(epochs_list):
            epochs_list = [150] * recur
        
        print(f"\n  第1轮递归训练 (Fixed模式, epochs={epochs_list[0]})")
        model = train_model(
            model, X_train, out_paths[0],
            calculate_low_para_for_input,
            epochs_list[0], n, batch_size, HD_type
        )
        
        # 后续递归
        for recur_level in range(2, recur + 1):
            if recur_level > recur:
                break
            
            print(f"\n  第{recur_level}轮递归训练 (epochs={epochs_list[recur_level - 1]})")
            
            # 加载上一轮模型
            from tensorflow.keras.models import load_model
            model = load_model(out_paths[recur_level - 2],
                             custom_objects={'KLdivergence': kl_divergence_loss})
            
            # 计算低维参数
            if recur_level == 2:
                low_para_func = lambda m, X, n, b, hd: calculate_low_para_for_layer(
                    m, X, 'Dense1', n, b, hd)
            elif recur_level == 3:
                low_para_func = lambda m, X, n, b, hd: calculate_low_para_for_layer(
                    m, X, 'Dense2', n, b, hd)
            else:
                low_para_func = lambda m, X, n, b, hd: calculate_low_para_for_layer(
                    m, X, 'Dense3', n, b, hd)
            
            model = train_model(
                model, X_train, out_paths[recur_level - 1],
                low_para_func,
                epochs_list[recur_level - 1], n, batch_size, HD_type
            )
    
    print(f"\n  {monkey_name} 训练完成")
    
    return model

def main():
    """主函数"""
    print(f"\n{'='*70}")
    print("BCNE低维流形构建 - TrippleN猴子数据")
    print("使用与BCNE原始代码完全一致的TensorFlow/Keras模型")
    print(f"{'='*70}")
    
    # 设置GPU（如果可用）
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        import tensorflow as tf
        tf.config.experimental_run_functions_eagerly(True)
    except:
        pass
    
    # 存储所有猴子结果
    all_results = {}
    
    # 对每只猴子进行训练
    for monkey in config.MONKEYS:
        print(f"\n{'#'*70}")
        print(f"# 处理猴子: {monkey}")
        print(f"{'#'*70}")
        
        # 加载数据
        neuron_psth, _ = load_monkey_data(monkey)
        if neuron_psth is None:
            continue
        
        # 高斯平滑和聚合处理
        neuron_psth_smoothed = apply_gaussian_smoothing_and_aggregation(neuron_psth)
        
        # 训练低维流形模型
        model = train_monkey_manifold(
            monkey,
            neuron_psth_smoothed,
            n_components=config.n_components,
            recur=config.recur,
            train_mode=config.train_mode,
            balance=config.balance
        )
        
        # 保存猴子结果摘要
        all_results[monkey] = {
            'model_saved': True,
            'n_neurons': neuron_psth_smoothed.shape[0],
            'n_components': config.n_components
        }
    
    # 保存训练摘要
    print(f"\n{'='*70}")
    print("训练完成摘要")
    print(f"{'='*70}")
    
    summary_path = os.path.join(config.OUTPUT_DIR, 'training_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("BCNE低维流形构建 - TrippleN猴子数据训练摘要\n")
        f.write("="*60 + "\n\n")
        f.write("模型架构: CNN + Flatten + Dense (与BCNE原始代码完全一致)\n")
        f.write("核心方法: KL散度损失 + 递归训练\n")
        f.write("损失函数: KL(P || Q), P来自perplexity计算, Q来自t分布\n\n")
        f.write(f"配置:\n")
        f.write(f"  - 猴子数量: {len(config.MONKEYS)}\n")
        f.write(f"  - 递归次数: {config.recur}\n")
        f.write(f"  - 低维维度: {config.n_components}\n")
        f.write(f"  - 训练模式: {'Patient' if config.train_mode == 0 else 'Fixed Epochs'}\n")
        f.write(f"  - Balance系数: {config.balance}\n")
        f.write(f"  - 时间窗口: {config.time_start} - {config.time_end}\n")
        f.write(f"  - 模型结构:\n")
        f.write(f"      * Conv2D层: {config.num_conv_layers}层, filters={config.filters_list}\n")
        f.write(f"      * Dense层: {config.dense_units}\n\n")
        
        f.write("各猴子训练结果:\n")
        for monkey, result in all_results.items():
            f.write(f"  - {monkey}: 神经元数={result['n_neurons']}, "
                   f"低维维度={result['n_components']}, 模型已保存\n")
    
    print(f"训练摘要已保存到: {summary_path}")
    
    return all_results

if __name__ == "__main__":
    try:
        results = main()
        print(f"\n所有猴子低维流形构建完成！")
    except Exception as e:
        print(f"\n出错: {e}")
        import traceback
        traceback.print_exc()
