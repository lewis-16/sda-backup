#!/usr/bin/env python3
"""
通用NWB文件读取工具
使用pynwb代替allensdk，支持多种NWB文件格式

"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union
import pynwb
from pynwb import NWBHDF5IO

logger = logging.getLogger(__name__)

class GenericNWBReader:
    """通用NWB文件读取器"""
    
    def __init__(self, nwb_path: str):
        self.nwb_path = nwb_path
        self.nwb_file = None
        
    def __enter__(self):
        """上下文管理器入口"""
        self.io = NWBHDF5IO(self.nwb_path, 'r')
        self.nwb_file = self.io.read()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        if hasattr(self, 'io'):
            self.io.close()
    
    def get_sweep_numbers(self) -> List[int]:
        """获取所有sweep编号"""
        sweep_numbers = []
        
        # 方法1: 从acquisition中获取
        if hasattr(self.nwb_file, 'acquisition') and self.nwb_file.acquisition:
            for key in self.nwb_file.acquisition.keys():
                try:
                    # 尝试从key中提取sweep编号
                    if '_' in key:
                        sweep_num = int(key.split('_')[-1])
                    else:
                        sweep_num = int(key)
                    sweep_numbers.append(sweep_num)
                except ValueError:
                    continue
        
        # 方法2: 从sweep table获取
        if not sweep_numbers and hasattr(self.nwb_file, 'sweep_table'):
            sweep_table = self.nwb_file.sweep_table
            try:
                for i in range(len(sweep_table)):
                    sweep_numbers.append(i)
            except Exception as e:
                logger.debug(f"从sweep table获取sweep编号时出错: {e}")
        
        # 方法3: 从processing中获取
        if not sweep_numbers and hasattr(self.nwb_file, 'processing'):
            processing = self.nwb_file.processing
            for key in processing.keys():
                try:
                    sweep_num = int(key.split('_')[-1]) if '_' in key else int(key)
                    sweep_numbers.append(sweep_num)
                except ValueError:
                    continue
        
        return sorted(sweep_numbers)
    
    def get_sweep_metadata(self, sweep_number: int) -> Dict:
        """获取sweep元数据"""
        metadata = {}
        
        # 从acquisition中查找
        if hasattr(self.nwb_file, 'acquisition') and self.nwb_file.acquisition:
            for key, data in self.nwb_file.acquisition.items():
                if str(sweep_number) in key:
                    metadata['key'] = key
                    if hasattr(data, 'description'):
                        metadata['description'] = str(data.description)
                    if hasattr(data, 'comments'):
                        metadata['comments'] = str(data.comments)
                    break
        
        # 从sweep table中查找
        if hasattr(self.nwb_file, 'sweep_table'):
            sweep_table = self.nwb_file.sweep_table
            try:
                if sweep_number < len(sweep_table):
                    row = sweep_table[sweep_number]
                    if hasattr(row, 'dtype') and hasattr(row.dtype, 'names'):
                        for field in row.dtype.names:
                            metadata[field] = row[field]
                    else:
                        # 处理不同的sweep table格式
                        metadata['sweep_index'] = sweep_number
            except Exception as e:
                logger.debug(f"从sweep table获取元数据时出错: {e}")
        
        return metadata
    
    def get_sweep_data(self, sweep_number: int) -> Dict:
        """获取sweep数据"""
        sweep_data = {}
        
        # 从acquisition中获取
        if hasattr(self.nwb_file, 'acquisition') and self.nwb_file.acquisition:
            for key, data in self.nwb_file.acquisition.items():
                if str(sweep_number) in key:
                    # 获取电压数据
                    if hasattr(data, 'data'):
                        sweep_data['response'] = data.data[:]
                    elif hasattr(data, 'voltage'):
                        sweep_data['response'] = data.voltage[:]
                    
                    # 获取刺激数据
                    if hasattr(data, 'stimulus'):
                        sweep_data['stimulus'] = data.stimulus[:]
                    elif hasattr(data, 'current'):
                        sweep_data['stimulus'] = data.current[:]
                    
                    # 获取采样率
                    if hasattr(data, 'rate'):
                        sweep_data['sampling_rate'] = float(data.rate)
                    elif hasattr(data, 'timestamps'):
                        timestamps = data.timestamps[:]
                        if len(timestamps) > 1:
                            sweep_data['sampling_rate'] = 1.0 / (timestamps[1] - timestamps[0])
                        else:
                            sweep_data['sampling_rate'] = 20000.0
                    else:
                        sweep_data['sampling_rate'] = 20000.0
                    
                    # 获取时间范围
                    sweep_data['index_range'] = [0, len(sweep_data['response'])]
                    break
        
        return sweep_data
    
    def get_stimulus_types(self) -> List[str]:
        """获取所有刺激类型"""
        stimulus_types = set()
        
        # 从sweep table获取
        if hasattr(self.nwb_file, 'sweep_table'):
            sweep_table = self.nwb_file.sweep_table
            try:
                for i in range(len(sweep_table)):
                    row = sweep_table[i]
                    if hasattr(row, 'dtype') and hasattr(row.dtype, 'names'):
                        for field in row.dtype.names:
                            if 'stimulus' in field.lower() or 'stim' in field.lower():
                                stim_type = str(row[field])
                                if stim_type and stim_type != 'None':
                                    stimulus_types.add(stim_type)
                    else:
                        # 处理不同的sweep table格式
                        stimulus_types.add(f"sweep_{i}")
            except Exception as e:
                logger.debug(f"从sweep table获取刺激类型时出错: {e}")
        
        # 从acquisition获取
        if hasattr(self.nwb_file, 'acquisition') and self.nwb_file.acquisition:
            for key, data in self.nwb_file.acquisition.items():
                if hasattr(data, 'description'):
                    desc = str(data.description)
                    if desc and desc != 'None':
                        stimulus_types.add(desc)
        
        return list(stimulus_types)
    
    def get_sweeps_by_stimulus_type(self, stimulus_type: str) -> List[int]:
        """根据刺激类型获取sweep编号"""
        target_sweeps = []
        
        # 从sweep table获取
        if hasattr(self.nwb_file, 'sweep_table'):
            sweep_table = self.nwb_file.sweep_table
            try:
                for i in range(len(sweep_table)):
                    row = sweep_table[i]
                    if hasattr(row, 'dtype') and hasattr(row.dtype, 'names'):
                        for field in row.dtype.names:
                            if 'stimulus' in field.lower() or 'stim' in field.lower():
                                if stimulus_type in str(row[field]):
                                    target_sweeps.append(i)
                                    break
                    else:
                        # 处理不同的sweep table格式
                        if stimulus_type in f"sweep_{i}":
                            target_sweeps.append(i)
            except Exception as e:
                logger.debug(f"从sweep table获取sweep编号时出错: {e}")
        
        # 从acquisition获取
        if hasattr(self.nwb_file, 'acquisition') and self.nwb_file.acquisition:
            for key, data in self.nwb_file.acquisition.items():
                if hasattr(data, 'description'):
                    if stimulus_type in str(data.description):
                        try:
                            sweep_num = int(key.split('_')[-1]) if '_' in key else int(key)
                            target_sweeps.append(sweep_num)
                        except ValueError:
                            continue
        
        return sorted(list(set(target_sweeps)))
    
    def print_file_info(self):
        """打印NWB文件信息"""
        print(f"NWB文件: {self.nwb_path}")
        print(f"文件ID: {self.nwb_file.identifier}")
        print(f"会话描述: {self.nwb_file.session_description}")
        
        # 打印acquisition信息
        if hasattr(self.nwb_file, 'acquisition') and self.nwb_file.acquisition:
            print(f"\nAcquisition数据:")
            for key, data in self.nwb_file.acquisition.items():
                print(f"  {key}: {type(data).__name__}")
                if hasattr(data, 'description'):
                    print(f"    描述: {data.description}")
        
        # 打印sweep table信息
        if hasattr(self.nwb_file, 'sweep_table'):
            sweep_table = self.nwb_file.sweep_table
            print(f"\nSweep Table:")
            print(f"  行数: {len(sweep_table)}")
            if len(sweep_table) > 0:
                try:
                    print(f"  列名: {list(sweep_table.dtype.names)}")
                except AttributeError:
                    print(f"  类型: {type(sweep_table)}")
                    # 尝试获取列名
                    if hasattr(sweep_table, 'colnames'):
                        print(f"  列名: {sweep_table.colnames}")
        
        # 打印刺激类型
        stimulus_types = self.get_stimulus_types()
        print(f"\n刺激类型: {stimulus_types}")
        
        # 打印sweep编号
        sweep_numbers = self.get_sweep_numbers()
        print(f"\nSweep编号: {sweep_numbers[:10]}{'...' if len(sweep_numbers) > 10 else ''}")

def main():
    """测试函数"""
    nwb_path = "/media/ubuntu/sda/Patch-seq/data/Patch/601506492_icephys.nwb"
    
    try:
        with GenericNWBReader(nwb_path) as reader:
            reader.print_file_info()
            
            # 测试获取sweep数据
            sweep_numbers = reader.get_sweep_numbers()
            if sweep_numbers:
                print(f"\n测试获取sweep {sweep_numbers[0]} 数据:")
                sweep_data = reader.get_sweep_data(sweep_numbers[0])
                print(f"  响应数据长度: {len(sweep_data.get('response', []))}")
                print(f"  采样率: {sweep_data.get('sampling_rate', 'N/A')}")
                
    except Exception as e:
        print(f"读取NWB文件时出错: {e}")

if __name__ == "__main__":
    main()
