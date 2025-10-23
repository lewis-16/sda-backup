#!/usr/bin/env python3
"""
验证生成的PDF文件
"""

import os
from PyPDF2 import PdfReader

def verify_pdf_file(pdf_path):
    """
    验证PDF文件的基本信息
    """
    if not os.path.exists(pdf_path):
        print(f"PDF文件不存在: {pdf_path}")
        return False
    
    try:
        # 获取文件大小
        file_size = os.path.getsize(pdf_path) / (1024 * 1024)  # MB
        print(f"文件大小: {file_size:.2f} MB")
        
        # 读取PDF信息
        reader = PdfReader(pdf_path)
        num_pages = len(reader.pages)
        print(f"PDF页数: {num_pages}")
        
        # 显示前几页的信息
        print("\n前5页信息:")
        for i in range(min(5, num_pages)):
            page = reader.pages[i]
            print(f"  第{i+1}页: {len(page.extract_text())} 个字符")
        
        return True
        
    except Exception as e:
        print(f"读取PDF文件时出错: {e}")
        return False

if __name__ == "__main__":
    # 检查测试PDF
    print("=== 测试PDF文件 ===")
    verify_pdf_file("/media/ubuntu/sda/visual_stimuli_pattern/dynamic/test_class_scatter_plots.pdf")
    
    print("\n=== 完整PDF文件 ===")
    verify_pdf_file("/media/ubuntu/sda/visual_stimuli_pattern/dynamic/class_scatter_plots_final.pdf")

