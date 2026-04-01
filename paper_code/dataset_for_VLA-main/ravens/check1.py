# analyze_ravens_data.py
import pickle
import numpy as np
import os
import glob

def analyze_stack_block_data():
    """分析stack-block-pyramid-train数据"""
    
    data_dir = "./stack-block-pyramid-train"
    
    print("="*60)
    print("RAVENS STACK-BLOCK-PYRAMID DATA ANALYSIS")
    print("="*60)
    
    # 检查目录结构
    print(f"\nData directory: {data_dir}")
    if not os.path.exists(data_dir):
        print(f"Directory not found: {data_dir}")
        return
    
    # 列出所有子目录
    subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    print(f"Found subdirectories: {subdirs}")
    
    # 分析每个模态的数据
    modalities = ['action', 'color', 'depth', 'info', 'reward']
    
    for modality in modalities:
        print(f"\n{'='*50}")
        print(f"ANALYZING: {modality.upper()}")
        print('='*50)
        
        mod_dir = os.path.join(data_dir, modality)
        
        if not os.path.exists(mod_dir):
            print(f"Directory not found: {mod_dir}")
            continue
        
        # 获取文件列表
        pkl_files = glob.glob(os.path.join(mod_dir, "*.pkl"))
        
        if not pkl_files:
            print(f"No .pkl files found in {mod_dir}")
            continue
        
        print(f"Found {len(pkl_files)} files")
        
        # 分析第一个文件作为示例
        sample_file = sorted(pkl_files)[0]  # 取第一个文件
        print(f"Sample file: {os.path.basename(sample_file)}")
        
        try:
            with open(sample_file, 'rb') as f:
                data = pickle.load(f)
            
            print(f"Data type: {type(data)}")
            
            if isinstance(data, np.ndarray):
                print(f"Shape: {data.shape}")
                print(f"Dtype: {data.dtype}")
                if modality == 'color':
                    print(f"Value range: [{data.min()}, {data.max()}]")
                    print(f"Sample pixel values: {data[240, 320, :] if len(data.shape) == 3 else 'N/A'}")
                elif modality == 'depth':
                    print(f"Depth range: [{data.min():.3f}, {data.max():.3f}] meters")
                
            elif isinstance(data, dict):
                print("Dictionary structure:")
                for key, value in data.items():
                    if isinstance(value, np.ndarray):
                        print(f"  '{key}': array{value.shape} ({value.dtype})")
                        if key == 'pose' and len(value) >= 3:
                            print(f"    position: [{value[0]:.3f}, {value[1]:.3f}, {value[2]:.3f}]")
                    elif isinstance(value, list):
                        print(f"  '{key}': list length {len(value)}")
                    else:
                        print(f"  '{key}': {type(value).__name__} = {value}")
                        
            elif isinstance(data, (list, tuple)):
                print(f"Length: {len(data)}")
                if len(data) > 0:
                    print(f"First element type: {type(data[0])}")
                    if isinstance(data[0], (int, float)) and len(data) <= 10:
                        print(f"Values: {data}")
                        
            else:
                print(f"Content: {data}")
                
        except Exception as e:
            print(f"Error loading file: {e}")

    # 详细分析info数据结构
    print(f"\n{'='*60}")
    print("DETAILED INFO STRUCTURE ANALYSIS")
    print('='*60)
    
    info_dir = os.path.join(data_dir, "info")
    if os.path.exists(info_dir):
        info_files = glob.glob(os.path.join(info_dir, "*.pkl"))
        if info_files:
            sample_info = sorted(info_files)[0]
            print(f"Analyzing: {os.path.basename(sample_info)}")
            
            try:
                with open(sample_info, 'rb') as f:
                    info_data = pickle.load(f)
                
                def print_nested_structure(data, indent=0):
                    spaces = "  " * indent
                    if isinstance(data, dict):
                        for key, value in data.items():
                            if isinstance(value, dict):
                                print(f"{spaces}'{key}': dict with keys {list(value.keys())}")
                                print_nested_structure(value, indent + 1)
                            elif isinstance(value, np.ndarray):
                                print(f"{spaces}'{key}': array{value.shape} ({value.dtype})")
                                if value.size <= 10:
                                    print(f"{spaces}  values: {value}")
                            elif isinstance(value, list):
                                print(f"{spaces}'{key}': list[{len(value)}]")
                                if len(value) <= 5 and all(isinstance(x, (int, float)) for x in value):
                                    print(f"{spaces}  values: {value}")
                            else:
                                print(f"{spaces}'{key}': {type(value).__name__} = {value}")
                    else:
                        print(f"{spaces}Data: {type(data)} = {data}")
                
                print_nested_structure(info_data)
                
            except Exception as e:
                print(f"Error analyzing info file: {e}")

if __name__ == "__main__":
    analyze_stack_block_data()
