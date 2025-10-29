#!/usr/bin/env python3
"""
内存监控工具
监控系统内存使用，当达到阈值时终止进程

"""

import psutil
import os
import signal
import sys
import time
import threading
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class MemoryMonitor:
    """内存监控器"""
    
    def __init__(self, 
                 max_memory_gb: float = 60.0,
                 check_interval: float = 5.0,
                 warning_threshold_gb: float = 50.0):
        """
        初始化内存监控器
        
        Args:
            max_memory_gb: 最大内存使用量（GB），超过此值将终止进程
            check_interval: 检查间隔（秒）
            warning_threshold_gb: 警告阈值（GB），超过此值将发出警告
        """
        self.max_memory_gb = max_memory_gb
        self.max_memory_bytes = max_memory_gb * 1024**3  # 转换为字节
        self.check_interval = check_interval
        self.warning_threshold_gb = warning_threshold_gb
        self.warning_threshold_bytes = warning_threshold_gb * 1024**3
        
        self.monitoring = False
        self.monitor_thread = None
        self.process = psutil.Process()
        
        logger.info(f"内存监控器初始化:")
        logger.info(f"  最大内存限制: {max_memory_gb} GB")
        logger.info(f"  警告阈值: {warning_threshold_gb} GB")
        logger.info(f"  检查间隔: {check_interval} 秒")
    
    def get_memory_usage(self) -> tuple[float, float]:
        """
        获取当前内存使用情况
        
        Returns:
            tuple: (当前进程内存使用GB, 系统总内存使用GB)
        """
        # 当前进程内存使用
        process_memory = self.process.memory_info().rss
        
        # 系统总内存使用
        system_memory = psutil.virtual_memory()
        total_memory_gb = system_memory.total / (1024**3)
        used_memory_gb = system_memory.used / (1024**3)
        
        process_memory_gb = process_memory / (1024**3)
        
        return process_memory_gb, used_memory_gb
    
    def check_memory(self) -> bool:
        """
        检查内存使用情况
        
        Returns:
            bool: True表示内存使用正常，False表示需要终止
        """
        process_memory_gb, system_memory_gb = self.get_memory_usage()
        
        # 检查当前进程内存使用
        if process_memory_gb * (1024**3) > self.max_memory_bytes:
            logger.error(f"🚨 内存使用超限!")
            logger.error(f"  当前进程内存使用: {process_memory_gb:.2f} GB")
            logger.error(f"  最大允许使用: {self.max_memory_gb} GB")
            logger.error(f"  系统总内存使用: {system_memory_gb:.2f} GB")
            return False
        
        # 检查警告阈值
        if process_memory_gb * (1024**3) > self.warning_threshold_bytes:
            logger.warning(f"⚠️  内存使用接近限制!")
            logger.warning(f"  当前进程内存使用: {process_memory_gb:.2f} GB")
            logger.warning(f"  警告阈值: {self.warning_threshold_gb} GB")
            logger.warning(f"  系统总内存使用: {system_memory_gb:.2f} GB")
        
        return True
    
    def _monitor_loop(self):
        """监控循环"""
        logger.info("内存监控开始...")
        
        while self.monitoring:
            try:
                if not self.check_memory():
                    logger.error("内存使用超限，正在终止进程...")
                    self.terminate_process()
                    break
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"内存监控出错: {e}")
                time.sleep(self.check_interval)
    
    def terminate_process(self):
        """终止当前进程"""
        logger.error("正在终止进程...")
        
        # 尝试优雅地终止
        try:
            # 发送SIGTERM信号
            os.kill(os.getpid(), signal.SIGTERM)
        except Exception as e:
            logger.error(f"发送SIGTERM失败: {e}")
            
            # 强制终止
            try:
                os.kill(os.getpid(), signal.SIGKILL)
            except Exception as e2:
                logger.error(f"强制终止失败: {e2}")
                sys.exit(1)
    
    def start_monitoring(self):
        """开始监控"""
        if self.monitoring:
            logger.warning("内存监控已在运行")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("内存监控已启动")
    
    def stop_monitoring(self):
        """停止监控"""
        if not self.monitoring:
            return
        
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        
        logger.info("内存监控已停止")
    
    def get_memory_info(self) -> dict:
        """获取详细的内存信息"""
        process_memory_gb, system_memory_gb = self.get_memory_usage()
        
        return {
            'process_memory_gb': process_memory_gb,
            'system_memory_gb': system_memory_gb,
            'max_memory_gb': self.max_memory_gb,
            'warning_threshold_gb': self.warning_threshold_gb,
            'memory_usage_percent': (process_memory_gb / self.max_memory_gb) * 100
        }

def setup_memory_monitor(max_memory_gb: float = 60.0, 
                        warning_threshold_gb: float = 50.0) -> MemoryMonitor:
    """
    设置内存监控器
    
    Args:
        max_memory_gb: 最大内存使用量（GB）
        warning_threshold_gb: 警告阈值（GB）
    
    Returns:
        MemoryMonitor: 配置好的内存监控器
    """
    monitor = MemoryMonitor(
        max_memory_gb=max_memory_gb,
        warning_threshold_gb=warning_threshold_gb
    )
    
    # 启动监控
    monitor.start_monitoring()
    
    return monitor

if __name__ == "__main__":
    # 测试内存监控器
    logging.basicConfig(level=logging.INFO)
    
    print("="*60)
    print("内存监控器测试")
    print("="*60)
    
    # 创建监控器（设置较低的阈值用于测试）
    monitor = MemoryMonitor(max_memory_gb=1.0, warning_threshold_gb=0.5)
    
    # 启动监控
    monitor.start_monitoring()
    
    try:
        # 模拟一些内存使用
        print("模拟内存使用...")
        data = []
        for i in range(100):
            data.append([0] * 1000000)  # 分配一些内存
            time.sleep(0.1)
            
            # 显示内存信息
            info = monitor.get_memory_info()
            print(f"内存使用: {info['process_memory_gb']:.2f} GB "
                  f"({info['memory_usage_percent']:.1f}%)")
            
    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        monitor.stop_monitoring()
        print("测试完成")
