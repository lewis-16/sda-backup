#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
时间分辨率视频生成器
用于探索生物视觉在时间维度上的分辨率

作者: AI Assistant
日期: 2024
"""

import os
import random
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import time

class TemporalResolutionVideoGenerator:
    """时间分辨率视频生成器"""
    
    def __init__(self, nature_scene_path: str, output_path: str = None):
        """
        初始化视频生成器
        
        Args:
            nature_scene_path: 自然场景图片文件夹路径
            output_path: 输出视频路径
        """
        self.nature_scene_path = nature_scene_path
        self.output_path = output_path or "/media/ubuntu/sda/visual_stimuli_pattern/dynamic/temporal_resolution_video.mp4"
        
        # 时间间隔设置（毫秒）
        self.time_intervals = [1, 2, 4, 10, 20, 33, 100]  # ms
        
        # 视频参数
        self.fps = 1000  # 帧率至少1000fps
        self.frame_duration_ms = 1000 / self.fps  # 每帧持续时间（毫秒）
        
        # 刺激参数
        self.images_per_round = 30
        self.gray_duration_after_image = 1000  # 每张图片后灰色持续时间（毫秒）
        self.gray_duration_after_round = 5000  # 每轮后灰色持续时间（毫秒）
        
        # 加载图片
        self.image_files = self._load_image_files()
        print(f"加载了 {len(self.image_files)} 张图片")
        
    def _load_image_files(self) -> List[str]:
        """加载所有图片文件"""
        image_files = []
        for i in range(100):  # 0-99
            filename = f"natural_scene_{i}.tiff"
            filepath = os.path.join(self.nature_scene_path, filename)
            if os.path.exists(filepath):
                image_files.append(filepath)
        return sorted(image_files)
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """加载单张图片"""
        try:
            # 使用PIL加载TIFF图片
            img = Image.open(image_path)
            # 转换为RGB格式
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return np.array(img)
        except Exception as e:
            print(f"加载图片失败 {image_path}: {e}")
            # 返回灰色图片作为备用
            return np.full((480, 640, 3), 128, dtype=np.uint8)
    
    def _create_gray_frame(self, shape: Tuple[int, int, int]) -> np.ndarray:
        """创建灰色帧"""
        return np.full(shape, 128, dtype=np.uint8)
    
    def _ms_to_frames(self, duration_ms: float) -> int:
        """将毫秒转换为帧数"""
        return int(duration_ms / self.frame_duration_ms)
    
    def _generate_stimulus_sequence(self, time_interval_ms: int) -> Tuple[List[Dict], int]:
        """
        生成刺激序列
        
        Args:
            time_interval_ms: 图片呈现时间间隔（毫秒）
            
        Returns:
            stimulus_sequence: 刺激序列信息
            total_frames: 总帧数
        """
        stimulus_sequence = []
        total_frames = 0
        
        # 随机选择30张图片
        selected_images = random.sample(self.image_files, self.images_per_round)
        
        for i, image_path in enumerate(selected_images):
            # 图片呈现
            image_duration_frames = self._ms_to_frames(time_interval_ms)
            stimulus_sequence.append({
                'type': 'image',
                'image_path': image_path,
                'image_id': os.path.basename(image_path),
                'start_frame': total_frames,
                'end_frame': total_frames + image_duration_frames,
                'duration_ms': time_interval_ms,
                'round': i + 1
            })
            total_frames += image_duration_frames
            
            # 图片后的灰色间隔
            gray_duration_frames = self._ms_to_frames(self.gray_duration_after_image)
            stimulus_sequence.append({
                'type': 'gray',
                'image_path': None,
                'image_id': 'gray',
                'start_frame': total_frames,
                'end_frame': total_frames + gray_duration_frames,
                'duration_ms': self.gray_duration_after_image,
                'round': i + 1
            })
            total_frames += gray_duration_frames
        
        # 轮次结束后的长时间灰色间隔
        round_gray_frames = self._ms_to_frames(self.gray_duration_after_round)
        stimulus_sequence.append({
            'type': 'round_gray',
            'image_path': None,
            'image_id': 'round_gray',
            'start_frame': total_frames,
            'end_frame': total_frames + round_gray_frames,
            'duration_ms': self.gray_duration_after_round,
            'round': 'end'
        })
        total_frames += round_gray_frames
        
        return stimulus_sequence, total_frames
    
    def generate_video(self) -> Tuple[str, pd.DataFrame]:
        """
        生成时间分辨率测试视频
        
        Returns:
            video_path: 生成的视频路径
            stimulus_df: 刺激记录DataFrame
        """
        print("开始生成时间分辨率测试视频...")
        
        all_stimulus_data = []
        video_writer = None
        current_image_shape = None
        
        # 为每个时间间隔生成一轮刺激
        for interval_ms in self.time_intervals:
            print(f"处理时间间隔: {interval_ms}ms")
            
            # 生成刺激序列
            stimulus_sequence, round_frames = self._generate_stimulus_sequence(interval_ms)
            
            # 初始化视频写入器（只在第一轮时）
            if video_writer is None:
                # 获取第一张图片的尺寸
                first_image = self._load_image(stimulus_sequence[0]['image_path'])
                current_image_shape = first_image.shape
                height, width = current_image_shape[:2]
                
                # 创建视频写入器
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(
                    self.output_path, 
                    fourcc, 
                    self.fps, 
                    (width, height)
                )
                print(f"视频参数: {width}x{height}, {self.fps}fps")
            
            # 生成视频帧
            for stimulus in stimulus_sequence:
                start_frame = stimulus['start_frame']
                end_frame = stimulus['end_frame']
                
                if stimulus['type'] == 'image':
                    # 加载并写入图片帧
                    image = self._load_image(stimulus['image_path'])
                    # 确保图片尺寸一致
                    if image.shape != current_image_shape:
                        image = cv2.resize(image, (current_image_shape[1], current_image_shape[0]))
                    
                    for frame_idx in range(start_frame, end_frame):
                        video_writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                        
                        # 记录刺激数据
                        all_stimulus_data.append({
                            'time_interval_ms': interval_ms,
                            'frame_number': frame_idx,
                            'time_ms': frame_idx * self.frame_duration_ms,
                            'image_id': stimulus['image_id'],
                            'stimulus_type': 'image',
                            'round': stimulus['round']
                        })
                
                else:
                    # 写入灰色帧
                    gray_frame = self._create_gray_frame(current_image_shape)
                    
                    for frame_idx in range(start_frame, end_frame):
                        video_writer.write(cv2.cvtColor(gray_frame, cv2.COLOR_RGB2BGR))
                        
                        # 记录刺激数据
                        all_stimulus_data.append({
                            'time_interval_ms': interval_ms,
                            'frame_number': frame_idx,
                            'time_ms': frame_idx * self.frame_duration_ms,
                            'image_id': stimulus['image_id'],
                            'stimulus_type': stimulus['type'],
                            'round': stimulus['round']
                        })
        
        # 释放视频写入器
        if video_writer:
            video_writer.release()
        
        # 创建DataFrame
        stimulus_df = pd.DataFrame(all_stimulus_data)
        
        print(f"视频生成完成: {self.output_path}")
        print(f"总帧数: {len(stimulus_df)}")
        print(f"视频时长: {len(stimulus_df) / self.fps:.2f}秒")
        
        return self.output_path, stimulus_df
    
    def save_stimulus_log(self, stimulus_df: pd.DataFrame, log_path: str = None):
        """保存刺激记录"""
        if log_path is None:
            log_path = self.output_path.replace('.mp4', '_stimulus_log.csv')
        
        stimulus_df.to_csv(log_path, index=False)
        print(f"刺激记录已保存: {log_path}")
        
        # 打印统计信息
        print("\n=== 刺激统计信息 ===")
        for interval in self.time_intervals:
            interval_data = stimulus_df[stimulus_df['time_interval_ms'] == interval]
            image_stimuli = interval_data[interval_data['stimulus_type'] == 'image']
            print(f"时间间隔 {interval}ms: {len(image_stimuli)} 个图像刺激")


def main():
    """主函数"""
    # 设置路径
    nature_scene_path = "/media/ubuntu/sda/visual_stimuli_pattern/nature_scene"
    output_path = "/media/ubuntu/sda/visual_stimuli_pattern/dynamic/temporal_resolution_video.mp4"
    
    # 创建生成器
    generator = TemporalResolutionVideoGenerator(nature_scene_path, output_path)
    
    # 生成视频
    video_path, stimulus_df = generator.generate_video()
    
    # 保存刺激记录
    generator.save_stimulus_log(stimulus_df)
    
    print(f"\n=== 生成完成 ===")
    print(f"视频文件: {video_path}")
    print(f"刺激记录: {video_path.replace('.mp4', '_stimulus_log.csv')}")
    
    return video_path, stimulus_df


if __name__ == "__main__":
    video_path, stimulus_df = main()
