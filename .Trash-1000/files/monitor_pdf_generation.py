#!/usr/bin/env python3
"""
监控PDF生成进度
"""

import os
import time
import subprocess

def monitor_pdf_generation():
    """
    监控PDF文件生成进度
    """
    pdf_path = "/media/ubuntu/sda/visual_stimuli_pattern/dynamic/class_scatter_plots_final.pdf"
    
    print("Monitoring PDF generation progress...")
    print("Press Ctrl+C to stop monitoring")
    
    try:
        while True:
            if os.path.exists(pdf_path):
                file_size = os.path.getsize(pdf_path) / (1024 * 1024)  # MB
                print(f"Current file size: {file_size:.2f} MB")
                
                # 检查进程是否还在运行
                try:
                    result = subprocess.run(['pgrep', '-f', 'generate_class_scatter_pdf_final'], 
                                          capture_output=True, text=True)
                    if result.returncode == 0:
                        print("Process is still running...")
                    else:
                        print("Process completed!")
                        break
                except:
                    print("Could not check process status")
            else:
                print("PDF file not found yet...")
            
            time.sleep(10)  # 每10秒检查一次
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")

if __name__ == "__main__":
    monitor_pdf_generation()
