#!/usr/bin/env python3
"""
修复 phy 文件夹中的 template_ind.npy 文件

当 template_ind.npy 中某个 template 的所有通道都是 0 或 -1 时，
会导致 phy 打开时出现 "attempt to get argmax of an empty sequence" 错误。

此脚本会：
1. 递归查找指定目录下所有的 template_ind.npy 文件
2. 检查每个 template 的通道索引
3. 如果发现所有通道都是 0 或 -1，随机赋值一些非零通道索引
4. 保存修复后的文件（可选：备份原文件）
"""

import argparse
import logging
import numpy as np
from pathlib import Path
from typing import List, Tuple
import shutil
from datetime import datetime


def setup_logger(verbose: bool) -> None:
    """设置日志"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(levelname)s] %(message)s",
    )


def find_template_ind_files(base_dir: Path) -> List[Path]:
    """
    递归查找所有 template_ind.npy 文件
    
    Args:
        base_dir: 基础目录路径
    
    Returns:
        template_ind.npy 文件路径列表
    """
    template_ind_files = []
    
    # 查找所有 phy_folder_for_kilosort 目录
    for phy_dir in base_dir.rglob("phy_folder_for_kilosort"):
        template_ind_path = phy_dir / "template_ind.npy"
        if template_ind_path.exists():
            template_ind_files.append(template_ind_path)
    
    # 也检查直接包含 template_ind.npy 的目录
    for template_ind_path in base_dir.rglob("template_ind.npy"):
        if template_ind_path not in template_ind_files:
            template_ind_files.append(template_ind_path)
    
    return sorted(template_ind_files)


def check_and_fix_template_ind(template_ind_path: Path, backup: bool = True) -> Tuple[bool, int]:
    """
    检查并修复 template_ind.npy 文件
    
    Args:
        template_ind_path: template_ind.npy 文件路径
        backup: 是否备份原文件
    
    Returns:
        (是否进行了修复, 修复的template数量)
    """
    logging.info(f"检查文件: {template_ind_path}")
    
    # 加载 template_ind.npy
    try:
        template_ind = np.load(template_ind_path)
    except Exception as e:
        logging.error(f"无法加载 {template_ind_path}: {e}")
        return False, 0
    
    original_shape = template_ind.shape
    logging.debug(f"  Shape: {original_shape}")
    
    # template_ind 的 shape 应该是 (n_templates, n_channels)
    if len(template_ind.shape) != 2:
        logging.warning(f"  {template_ind_path} 的 shape 不是2D，跳过")
        return False, 0
    
    n_templates, n_channels = template_ind.shape
    
    # 尝试加载 templates.npy 和 channel_map.npy 来获取更合理的通道分配
    templates_path = template_ind_path.parent / "templates.npy"
    channel_map_path = template_ind_path.parent / "channel_map.npy"
    templates = None
    channel_map = None
    
    if templates_path.exists():
        try:
            templates = np.load(templates_path)
            logging.debug(f"  加载了 templates.npy: shape {templates.shape}")
        except Exception as e:
            logging.debug(f"  无法加载 templates.npy: {e}")
    
    if channel_map_path.exists():
        try:
            channel_map = np.load(channel_map_path)
            logging.debug(f"  加载了 channel_map.npy: shape {channel_map.shape}")
        except Exception as e:
            logging.debug(f"  无法加载 channel_map.npy: {e}")
    
    # 检查是否有问题
    fixed_count = 0
    needs_fix = False
    
    for template_id in range(n_templates):
        template_channels = template_ind[template_id]
        
        # 检查是否所有通道都是 0 或 -1
        valid_channels_global = template_channels[(template_channels != 0) & (template_channels != -1)]
        
        # 如果有 channel_map，检查这些全局通道索引是否能映射到有效的局部通道索引，并且数据非零
        valid_local_indices = []
        valid_amplitudes = []
        
        if len(valid_channels_global) > 0 and channel_map is not None and templates is not None:
            n_tm_channels = templates.shape[2]
            for ch_global in valid_channels_global:
                if ch_global in channel_map:
                    local_idx = np.where(channel_map == ch_global)[0]
                    if len(local_idx) > 0 and local_idx[0] < n_tm_channels:
                        # 检查该通道的数据是否非零（模拟 phy 计算 amplitude）
                        template_slice = templates[template_id][:, local_idx[0]]
                        amplitude = np.max(np.abs(template_slice))
                        if amplitude > 1e-10:  # 只保留非零的
                            valid_local_indices.append(local_idx[0])
                            valid_amplitudes.append(amplitude)
        
        # 如果映射后没有有效通道（无法计算 amplitude），或者原本就没有有效通道，需要修复
        # 注意：即使能映射但数据全为0，也会导致 amplitude 为空，需要修复
        if len(valid_channels_global) == 0 or len(valid_local_indices) == 0:
            # 需要修复
            if len(valid_channels_global) == 0:
                logging.warning(f"  Template {template_id}: 所有通道都是 0 或 -1，需要修复")
            elif len(valid_local_indices) == 0:
                # 检查是映射失败还是数据全为0
                mapped_but_zero = False
                if channel_map is not None and templates is not None:
                    n_tm_channels = templates.shape[2]
                    for ch_global in valid_channels_global:
                        if ch_global in channel_map:
                            local_idx = np.where(channel_map == ch_global)[0]
                            if len(local_idx) > 0 and local_idx[0] < n_tm_channels:
                                mapped_but_zero = True
                                break
                
                if mapped_but_zero:
                    logging.warning(f"  Template {template_id}: 全局通道索引 {valid_channels_global.tolist()} 能映射但数据全为0（导致 amplitude 为空），需要修复")
                else:
                    logging.warning(f"  Template {template_id}: 全局通道索引 {valid_channels_global.tolist()} 无法映射到有效的局部通道索引（导致 amplitude 为空），需要修复")
            needs_fix = True
            
            # 尝试从 templates.npy 中获取该 template 的实际通道信息
            selected_local_indices = None
            
            if templates is not None and template_id < templates.shape[0]:
                try:
                    template = templates[template_id]  # shape: (n_timepoints, n_channels)
                    # 计算每个通道的 RMS 能量
                    channel_energies = np.sqrt(np.mean(template**2, axis=0))
                    # 选择能量最大的几个通道（至少3个，最多5个）
                    n_tm_channels = template.shape[1]
                    n_selected = min(max(3, 5), n_tm_channels)
                    top_channels = np.argsort(channel_energies)[-n_selected:]
                    # 只选择能量大于阈值的通道
                    threshold = np.max(channel_energies) * 0.1  # 至少是最大能量的10%
                    valid_top_channels = top_channels[channel_energies[top_channels] > threshold]
                    if len(valid_top_channels) >= 3:
                        selected_local_indices = valid_top_channels
                        logging.debug(f"    从 templates.npy 选择了局部通道: {selected_local_indices.tolist()}")
                except Exception as e:
                    logging.debug(f"    无法从 templates.npy 获取信息: {e}")
            
            # 如果无法从 templates.npy 获取，使用随机分配
            if selected_local_indices is None or len(selected_local_indices) < 3:
                # 随机选择一些局部通道索引（0 到 n_tm_channels-1）
                if templates is not None:
                    n_tm_channels = templates.shape[2]
                else:
                    n_tm_channels = n_channels
                # 选择 3-5 个通道（至少3个，最多5个或n_tm_channels）
                n_selected = min(max(3, 5), n_tm_channels)
                # 使用连续的通道索引（更合理）
                start_idx = np.random.randint(0, max(1, n_tm_channels - n_selected + 1))
                selected_local_indices = np.arange(start_idx, min(start_idx + n_selected, n_tm_channels))
                logging.debug(f"    随机选择了局部通道: {selected_local_indices.tolist()}")
            
            # 将局部通道索引映射回全局通道索引（通过 channel_map）
            if channel_map is not None and len(channel_map) > 0:
                # 将局部通道索引映射到全局通道索引
                selected_global_channels = []
                for local_idx in selected_local_indices:
                    if local_idx < len(channel_map):
                        selected_global_channels.append(int(channel_map[local_idx]))
                    else:
                        # 如果超出 channel_map 范围，使用局部索引本身
                        selected_global_channels.append(int(local_idx))
                
                # 赋值：先全部设为-1，然后设置选中的通道（使用全局通道索引）
                template_ind[template_id, :] = -1  # 先全部设为-1
                # 设置选中的通道，使用全局通道索引
                for i, global_ch in enumerate(selected_global_channels):
                    if i < n_channels:
                        template_ind[template_id, i] = global_ch
                
                fixed_count += 1
                logging.info(f"  Template {template_id}: 修复为使用全局通道 {selected_global_channels} (局部通道 {selected_local_indices.tolist()})")
            else:
                # 没有 channel_map，直接使用局部索引
                template_ind[template_id, :] = -1  # 先全部设为-1
                # 设置选中的通道，使用局部通道索引本身作为值
                for i, local_idx in enumerate(selected_local_indices):
                    if i < n_channels:
                        template_ind[template_id, i] = int(local_idx)
                
                fixed_count += 1
                logging.info(f"  Template {template_id}: 修复为使用通道 {selected_local_indices.tolist()}")
    
    if not needs_fix:
        logging.info(f"  ✓ {template_ind_path} 无需修复")
        return False, 0
    
    # 备份原文件
    if backup:
        backup_path = template_ind_path.with_suffix('.npy.backup')
        if not backup_path.exists():
            shutil.copy2(template_ind_path, backup_path)
            logging.info(f"  备份原文件到: {backup_path}")
        else:
            logging.warning(f"  备份文件已存在，跳过备份: {backup_path}")
    
    # 保存修复后的文件
    try:
        np.save(template_ind_path, template_ind)
        logging.info(f"  ✓ 已修复 {template_ind_path} ({fixed_count} 个 templates)")
        return True, fixed_count
    except Exception as e:
        logging.error(f"  保存失败 {template_ind_path}: {e}")
        return False, 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="修复 phy 文件夹中的 template_ind.npy 文件，解决 'attempt to get argmax of an empty sequence' 错误"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        required=True,
        help="基础目录路径（会递归查找所有 template_ind.npy 文件）",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="不备份原文件（默认会备份）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只检查不修复（不修改文件）",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="显示详细日志",
    )
    args = parser.parse_args()
    
    setup_logger(args.verbose)
    
    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"目录不存在: {base_dir}")
    
    logging.info(f"搜索目录: {base_dir}")
    logging.info(f"模式: {'检查模式（不修改文件）' if args.dry_run else '修复模式'}")
    
    # 查找所有 template_ind.npy 文件
    template_ind_files = find_template_ind_files(base_dir)
    
    if len(template_ind_files) == 0:
        logging.warning(f"未找到任何 template_ind.npy 文件")
        return
    
    logging.info(f"找到 {len(template_ind_files)} 个 template_ind.npy 文件")
    
    # 处理每个文件
    total_fixed = 0
    total_files_fixed = 0
    
    for template_ind_path in template_ind_files:
        if args.dry_run:
            # 只检查不修复
            try:
                template_ind = np.load(template_ind_path)
                if len(template_ind.shape) != 2:
                    continue
                
                # 加载相关文件
                templates_path = template_ind_path.parent / "templates.npy"
                channel_map_path = template_ind_path.parent / "channel_map.npy"
                templates = None
                channel_map = None
                
                if templates_path.exists():
                    try:
                        templates = np.load(templates_path)
                    except:
                        pass
                
                if channel_map_path.exists():
                    try:
                        channel_map = np.load(channel_map_path)
                    except:
                        pass
                
                n_templates, n_channels = template_ind.shape
                problem_count = 0
                
                for template_id in range(n_templates):
                    template_channels = template_ind[template_id]
                    valid_channels_global = template_channels[(template_channels != 0) & (template_channels != -1)]
                    
                    # 检查映射和数据（模拟 phy 计算 amplitude）
                    valid_local_indices = []
                    if len(valid_channels_global) > 0 and channel_map is not None and templates is not None:
                        n_tm_channels = templates.shape[2]
                        for ch_global in valid_channels_global:
                            if ch_global in channel_map:
                                local_idx = np.where(channel_map == ch_global)[0]
                                if len(local_idx) > 0 and local_idx[0] < n_tm_channels:
                                    # 检查数据是否非零
                                    template_slice = templates[template_id][:, local_idx[0]]
                                    amplitude = np.max(np.abs(template_slice))
                                    if amplitude > 1e-10:  # 只保留非零的
                                        valid_local_indices.append(local_idx[0])
                    
                    # 如果映射后没有有效通道（无法计算 amplitude），或者原本就没有有效通道，有问题
                    if len(valid_channels_global) == 0 or len(valid_local_indices) == 0:
                        problem_count += 1
                
                if problem_count > 0:
                    logging.warning(f"  {template_ind_path}: 发现 {problem_count} 个有问题的 templates")
                    total_fixed += problem_count
                    total_files_fixed += 1
                else:
                    logging.info(f"  ✓ {template_ind_path}: 正常")
            except Exception as e:
                logging.error(f"  检查失败 {template_ind_path}: {e}")
        else:
            # 修复文件
            fixed, count = check_and_fix_template_ind(
                template_ind_path, 
                backup=not args.no_backup
            )
            if fixed:
                total_fixed += count
                total_files_fixed += 1
    
    # 总结
    logging.info("=" * 80)
    if args.dry_run:
        logging.info(f"检查完成:")
        logging.info(f"  - 检查了 {len(template_ind_files)} 个文件")
        logging.info(f"  - 发现 {total_files_fixed} 个文件有问题")
        logging.info(f"  - 共 {total_fixed} 个 templates 需要修复")
    else:
        logging.info(f"修复完成:")
        logging.info(f"  - 处理了 {len(template_ind_files)} 个文件")
        logging.info(f"  - 修复了 {total_files_fixed} 个文件")
        logging.info(f"  - 共修复了 {total_fixed} 个 templates")
    
    if total_fixed > 0 and args.dry_run:
        logging.info(f"\n运行时不加 --dry-run 参数来实际修复文件")


if __name__ == "__main__":
    main()

