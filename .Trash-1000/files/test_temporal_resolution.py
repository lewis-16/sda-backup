#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
时间分辨率视频生成器 - 测试脚本
"""

import sys
import os
sys.path.append('/media/ubuntu/sda/visual_stimuli_pattern/dynamic')

from temporal_resolution_video_generator import TemporalResolutionVideoGenerator
import pandas as pd

def test_video_generation():
    """测试视频生成功能"""
    print("=== 时间分辨率视频生成器测试 ===")
    
    # 设置路径
    nature_scene_path = "/media/ubuntu/sda/visual_stimuli_pattern/nature_scene"
    output_path = "/media/ubuntu/sda/visual_stimuli_pattern/dynamic/test_temporal_resolution_video.mp4"
    
    try:
        # 创建生成器
        print("1. 创建视频生成器...")
        generator = TemporalResolutionVideoGenerator(nature_scene_path, output_path)
        
        # 检查图片加载
        print(f"2. 图片加载检查: {len(generator.image_files)} 张图片")
        
        # 测试单张图片加载
        if generator.image_files:
            test_image = generator._load_image(generator.image_files[0])
            print(f"3. 测试图片加载: 尺寸 {test_image.shape}")
        
        # 生成视频
        print("4. 开始生成视频...")
        video_path, stimulus_df = generator.generate_video()
        
        # 保存刺激记录
        print("5. 保存刺激记录...")
        generator.save_stimulus_log(stimulus_df)
        
        # 显示结果统计
        print("\n=== 生成结果 ===")
        print(f"视频文件: {video_path}")
        print(f"视频尺寸: {len(stimulus_df)} 帧")
        print(f"视频时长: {len(stimulus_df) / generator.fps:.2f} 秒")
        
        # 显示刺激统计
        print("\n=== 刺激统计 ===")
        for interval in generator.time_intervals:
            interval_data = stimulus_df[stimulus_df['time_interval_ms'] == interval]
            image_count = len(interval_data[interval_data['stimulus_type'] == 'image'])
            print(f"时间间隔 {interval}ms: {image_count} 个图像刺激")
        
        print("\n测试完成！")
        return video_path, stimulus_df
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    video_path, stimulus_df = test_video_generation()
