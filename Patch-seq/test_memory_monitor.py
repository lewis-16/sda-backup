#!/usr/bin/env python3
"""
测试内存监控功能
验证内存监控器是否正常工作

"""

import os
import json
import logging
from memory_monitor import MemoryMonitor

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_memory_monitor():
    """测试内存监控器"""
    
    print("="*60)
    print("内存监控器功能测试")
    print("="*60)
    
    # 创建监控器（设置较低的阈值用于测试）
    monitor = MemoryMonitor(
        max_memory_gb=2.0,  # 2GB限制
        warning_threshold_gb=1.0,  # 1GB警告
        check_interval=1.0  # 1秒检查一次
    )
    
    print("1. 测试内存信息获取")
    info = monitor.get_memory_info()
    print(f"   当前进程内存: {info['process_memory_gb']:.2f} GB")
    print(f"   系统总内存: {info['system_memory_gb']:.2f} GB")
    print(f"   内存使用率: {info['memory_usage_percent']:.1f}%")
    
    print("\n2. 启动内存监控")
    monitor.start_monitoring()
    
    print("\n3. 模拟内存使用（分配大量内存）")
    data_blocks = []
    
    try:
        for i in range(50):  # 分配50个内存块
            # 每个块约100MB
            block = [0] * (25 * 1024 * 1024)  # 25M个整数 ≈ 100MB
            data_blocks.append(block)
            
            # 检查内存使用
            info = monitor.get_memory_info()
            print(f"   分配第{i+1}个块后: {info['process_memory_gb']:.2f} GB "
                  f"({info['memory_usage_percent']:.1f}%)")
            
            # 如果接近限制，停止分配
            if info['memory_usage_percent'] > 80:
                print("   接近内存限制，停止分配")
                break
                
    except KeyboardInterrupt:
        print("\n用户中断测试")
    except Exception as e:
        print(f"\n测试过程中出现异常: {e}")
    finally:
        print("\n4. 停止内存监控")
        monitor.stop_monitoring()
        
        # 清理内存
        print("5. 清理内存")
        del data_blocks
        
        # 最终内存检查
        final_info = monitor.get_memory_info()
        print(f"   最终内存使用: {final_info['process_memory_gb']:.2f} GB")
        
        print("\n内存监控测试完成!")

def test_config_integration():
    """测试配置文件集成"""
    
    print("\n" + "="*60)
    print("配置文件集成测试")
    print("="*60)
    
    config_path = "multi_channel_config.json"
    
    if not os.path.exists(config_path):
        print(f"配置文件 {config_path} 不存在")
        return
    
    # 读取配置
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print("配置文件中的内存设置:")
    print(f"   最大内存限制: {config.get('max_memory_gb', '未设置')} GB")
    print(f"   警告阈值: {config.get('warning_memory_gb', '未设置')} GB")
    
    # 创建监控器
    max_memory_gb = config.get('max_memory_gb', 60.0)
    warning_threshold_gb = config.get('warning_memory_gb', 50.0)
    
    monitor = MemoryMonitor(
        max_memory_gb=max_memory_gb,
        warning_threshold_gb=warning_threshold_gb
    )
    
    print(f"\n使用配置文件设置创建监控器:")
    print(f"   最大内存: {max_memory_gb} GB")
    print(f"   警告阈值: {warning_threshold_gb} GB")
    
    # 测试监控
    monitor.start_monitoring()
    
    # 显示当前内存使用
    info = monitor.get_memory_info()
    print(f"\n当前内存使用情况:")
    print(f"   进程内存: {info['process_memory_gb']:.2f} GB")
    print(f"   系统内存: {info['system_memory_gb']:.2f} GB")
    print(f"   使用率: {info['memory_usage_percent']:.1f}%")
    
    monitor.stop_monitoring()
    print("配置文件集成测试完成!")

if __name__ == "__main__":
    try:
        test_memory_monitor()
        test_config_integration()
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
