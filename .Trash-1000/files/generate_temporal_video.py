#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
时间分辨率视频生成器 - 使用示例
"""

from temporal_resolution_video_generator import TemporalResolutionVideoGenerator

def generate_temporal_resolution_video():
    """生成时间分辨率测试视频"""
    
    # 设置路径
    nature_scene_path = "/media/ubuntu/sda/visual_stimuli_pattern/nature_scene"
    output_path = "/media/ubuntu/sda/visual_stimuli_pattern/dynamic/temporal_resolution_video.mp4"
    
    print("=== 时间分辨率视频生成器 ===")
    print("用于探索生物视觉在时间维度上的分辨率")
    print()
    
    # 创建生成器
    generator = TemporalResolutionVideoGenerator(nature_scene_path, output_path)
    
    print(f"时间间隔设置: {generator.time_intervals} ms")
    print(f"每轮图片数量: {generator.images_per_round}")
    print(f"视频帧率: {generator.fps} fps")
    print()
    
    # 生成视频
    print("开始生成视频...")
    video_path, stimulus_df = generator.generate_video()
    
    # 保存刺激记录
    generator.save_stimulus_log(stimulus_df)
    
    print(f"\n=== 生成完成 ===")
    print(f"视频文件: {video_path}")
    print(f"刺激记录: {video_path.replace('.mp4', '_stimulus_log.csv')}")
    print(f"视频时长: {len(stimulus_df) / generator.fps:.2f} 秒")
    print(f"总帧数: {len(stimulus_df)}")
    
    return video_path, stimulus_df

if __name__ == "__main__":
    video_path, stimulus_df = generate_temporal_resolution_video()
