import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pickle
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_data():
    """加载预处理的数据"""
    # 加载trail_activity数据
    with open('/media/ubuntu/sda/duan/script/trail_activity_500.pkl', 'rb') as f:
        spike_train_dict = pickle.load(f)
    
    # 计算trail_average
    trail_average = {}
    for key, item in spike_train_dict.items():
        trail_average_temp = []
        for i in range(len(item[0])):
            temp = []
            for rep in item:
                temp.append(rep[i])
            
            temp = np.stack(temp).mean(axis=0)
            trail_average_temp.append(temp)

        trail_average_temp = np.array(trail_average_temp)
        trail_average[key] = trail_average_temp
    
    # 计算image_average
    image_average = {}
    for i in range(1, 25):
        temp = []
        for ori in [1, 2, 3]:
            temp.append(trail_average[f'{i}_{ori}'])
        temp = np.stack(temp)
        image_average[i] = temp.mean(axis=0)
    
    # 计算category_average
    category_average = {}
    category = ['animals', 'faces', 'fruits', 'manmade', 'plants', 'shape2d']
    for i in range(6):
        temp = []
        for j in range(i * 4 + 1, (i + 1) * 4 + 1):
            temp.append(image_average[j])
        temp = np.stack(temp)
        category_average[category[i]] = temp.mean(axis=0)
    
    return trail_average, image_average, category_average

def plot_trail_average_pdf(trail_average, output_path):
    """绘制trail_average的firing rate曲线"""
    print("正在生成trail_average PDF...")
    
    # 获取神经元数量
    num_neurons = trail_average[list(trail_average.keys())[0]].shape[0]
    
    with PdfPages(output_path) as pdf:
        # 为每个神经元创建一个图表
        for neuron_idx in range(num_neurons):
            fig, ax = plt.subplots(figsize=(8, 4))
            
            # 定义颜色映射
            colors = plt.cm.tab20(np.linspace(0, 1, len(trail_average)))
            
            # 绘制每个条件的firing rate曲线
            for i, (key, data) in enumerate(trail_average.items()):
                trial_condition = int(key.split('_')[0])
                trial_target = int(key.split('_')[1])
                
                # 获取该神经元的firing rate
                firing_rate = data[neuron_idx]
                
                # 时间轴 (50ms bins, 500ms total)
                time_axis = np.arange(0, len(firing_rate) * 25, 25)
                
                # 绘制曲线
                ax.plot(time_axis, firing_rate, 
                       color=colors[i], 
                       linewidth=3, 
                       alpha=0.7)
            
            ax.set_xlabel('Time (ms)', fontsize=12)
            ax.set_ylabel('Firing Rate (Hz)', fontsize=12)
            ax.set_title(f'Neuron {neuron_idx + 1} - Trail Average Firing Rate', fontsize=14, fontweight='bold')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
            if (neuron_idx + 1) % 10 == 0:
                print(f"已处理 {neuron_idx + 1}/{num_neurons} 个神经元")
    
    print(f"Trail average PDF已保存到: {output_path}")

def plot_image_average_pdf(image_average, output_path):
    """绘制image_average的firing rate曲线"""
    print("正在生成image_average PDF...")
    
    # 获取神经元数量
    num_neurons = image_average[1].shape[0]
    
    # 图片名称映射
    image_names = {
        1: 'Dee1', 2: 'Ele', 3: 'Pig', 4: 'Rhi',  # animals
        5: 'MA', 6: 'MB', 7: 'MC', 8: 'WA',        # faces
        9: 'App1', 10: 'Ban1', 11: 'Pea1', 12: 'Pin1',  # fruits
        13: 'Bed1', 14: 'Cha1', 15: 'Dis1', 16: 'Sof1',  # manmade
        17: 'A', 18: 'B', 19: 'C', 20: 'D',        # plants
        21: 'Cir', 22: 'Oth', 23: 'Squ', 24: 'Tri'  # shape2d
    }
    
    with PdfPages(output_path) as pdf:
        # 为每个神经元创建一个图表
        for neuron_idx in range(num_neurons):
            fig, ax = plt.subplots(figsize=(6, 4))
            
            # 定义颜色映射
            colors = plt.cm.tab20(np.linspace(0, 1, len(image_average)))
            
            # 绘制每个图片的firing rate曲线
            for i, (img_id, data) in enumerate(image_average.items()):
                firing_rate = data[neuron_idx]
                
                time_axis = np.arange(0, len(firing_rate) * 25, 25)
                
                # 绘制曲线
                ax.plot(time_axis, firing_rate, 
                       color=colors[i], 
                       linewidth=3, 
                       alpha=0.7,
                       label=f'{image_names[img_id]} (ID:{img_id})')
            
            ax.set_xlabel('Time (ms)', fontsize=12)
            ax.set_ylabel('Firing Rate (Hz)', fontsize=12)
            ax.set_title(f'Neuron {neuron_idx + 1} - Image Average Firing Rate', fontsize=14, fontweight='bold')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
            if (neuron_idx + 1) % 10 == 0:
                print(f"已处理 {neuron_idx + 1}/{num_neurons} 个神经元")
    
    print(f"Image average PDF已保存到: {output_path}")

def plot_category_average_pdf(category_average, output_path):
    """绘制category_average的firing rate曲线"""
    print("正在生成category_average PDF...")
    
    # 获取神经元数量
    num_neurons = category_average[list(category_average.keys())[0]].shape[0]
    
    # 类别中文名称映射
    category_names = {
        'animals': 'animals',
        'faces': 'faces',
        'fruits': 'fruits',
        'manmade': 'manmade',
        'plants': 'plants',
        'shape2d': 'shape2d'
    }
    
    with PdfPages(output_path) as pdf:
        # 为每个神经元创建一个图表
        for neuron_idx in range(num_neurons):
            fig, ax = plt.subplots(figsize=(6, 3))
            
            # 定义颜色映射
            colors = plt.cm.Set1(np.linspace(0, 1, len(category_average)))
            
            # 绘制每个类别的firing rate曲线
            for i, (category, data) in enumerate(category_average.items()):
                # 获取该神经元的firing rate
                firing_rate = data[neuron_idx]
                
                # 时间轴 (50ms bins, 500ms total)
                time_axis = np.arange(0, len(firing_rate) * 25, 25)
                
                # 绘制曲线
                ax.plot(time_axis, firing_rate, 
                       color=colors[i], 
                       linewidth=4, 
                       alpha=0.8,
                       label=f'{category_names[category]} ({category})')
            
            ax.set_xlabel('Timr (ms)', fontsize=12)
            ax.set_ylabel('Firing Rate (Hz)', fontsize=12)
            ax.set_title(f'Neuron {neuron_idx + 1} - Category Average Firing Rate', fontsize=14, fontweight='bold')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
            if (neuron_idx + 1) % 10 == 0:
                print(f"已处理 {neuron_idx + 1}/{num_neurons} 个神经元")
    
    print(f"Category average PDF已保存到: {output_path}")

def main():
    """主函数"""
    print("开始加载数据...")
    trail_average, image_average, category_average = load_data()
    
    print(f"数据加载完成:")
    print(f"- Trail average: {len(trail_average)} 个条件")
    print(f"- Image average: {len(image_average)} 张图片")
    print(f"- Category average: {len(category_average)} 个类别")
    
    # 创建输出目录
    output_dir = '/media/ubuntu/sda/duan/figure'
    os.makedirs(output_dir, exist_ok=True)
    
    # 绘制三个PDF
    plot_trail_average_pdf(trail_average, os.path.join(output_dir, 'trail_average_firing_rates.pdf'))
    plot_image_average_pdf(image_average, os.path.join(output_dir, 'image_average_firing_rates.pdf'))
    plot_category_average_pdf(category_average, os.path.join(output_dir, 'category_average_firing_rates.pdf'))
    
    print("\n所有PDF文件生成完成！")

if __name__ == "__main__":
    main()
