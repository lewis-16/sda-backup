#!/usr/bin/env python3
"""
批量下载 OpenVidHD_part_*.zip 文件并解压到对应的 part_XX 文件夹下。

用法:
    python download_and_extract_parts.py [--auto] [--start START] [--end END]
    
参数:
    --auto: 自动模式，跳过交互式提示（默认启用）
    --start: 起始 part 编号（默认 1）
    --end: 结束 part 编号（默认 14）
"""

import os
import sys
import zipfile
import requests
from pathlib import Path
import time
import argparse
import shutil
import tempfile

try:
    from tqdm import tqdm
except ImportError:
    # 如果没有 tqdm，使用简单的进度显示
    def tqdm(iterable=None, **kwargs):
        if iterable:
            return iterable
        class FakePbar:
            def __init__(self, **kwargs):
                self.desc = kwargs.get('desc', '')
                self.total = kwargs.get('total', 0)
            def update(self, n):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        return FakePbar(**kwargs)

def download_file(url, filepath, chunk_size=8192):
    """
    下载文件并显示进度条
    
    Args:
        url: 下载链接
        filepath: 保存路径
        chunk_size: 每次下载的块大小
    
    Returns:
        bool: 下载是否成功
    """
    try:
        # 发送 HEAD 请求获取文件大小
        response = requests.head(url, allow_redirects=True, timeout=30)
        total_size = int(response.headers.get('content-length', 0))
        
        # 如果文件已存在且大小匹配，跳过下载
        if os.path.exists(filepath) and os.path.getsize(filepath) == total_size:
            print(f"  文件已存在且大小匹配，跳过下载: {os.path.basename(filepath)}")
            return True
        
        # 开始下载
        response = requests.get(url, stream=True, timeout=30, allow_redirects=True)
        response.raise_for_status()
        
        # 创建进度条
        with open(filepath, 'wb') as f, tqdm(
            desc=f"  下载 {os.path.basename(filepath)}",
            total=total_size if total_size > 0 else None,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        return True
    except requests.exceptions.RequestException as e:
        print(f"  下载失败: {e}")
        return False
    except Exception as e:
        print(f"  发生错误: {e}")
        return False

def merge_split_zips(split_files, output_zip_path):
    """
    合并多个分片 zip 文件为一个完整的 zip 文件
    
    Args:
        split_files: 分片文件路径列表（按顺序）
        output_zip_path: 合并后的输出 zip 文件路径
    
    Returns:
        bool: 合并是否成功
    """
    try:
        print(f"  合并 {len(split_files)} 个分片文件...")
        
        # 方法1: 如果是分片的 zip 文件（通过 split 命令创建），直接二进制合并
        # 检查文件是否都是有效的 zip 分片
        is_zip_split = True
        for split_file in split_files:
            if not os.path.exists(split_file):
                print(f"  错误: 分片文件不存在: {split_file}")
                return False
            # 检查是否可以单独打开为 zip
            try:
                with zipfile.ZipFile(split_file, 'r') as zf:
                    pass
            except:
                is_zip_split = False
                break
        
        if is_zip_split:
            # 方法1: 所有分片都是有效的 zip，尝试解压并合并内容
            print(f"  检测到分片 zip 文件，合并内容...")
            temp_dir = tempfile.mkdtemp(prefix='zip_merge_')
            try:
                # 先解压所有分片到临时目录
                for split_file in tqdm(split_files, desc="  解压分片"):
                    try:
                        with zipfile.ZipFile(split_file, 'r') as split_zip:
                            split_zip.extractall(temp_dir)
                    except Exception as e:
                        print(f"  处理分片 {os.path.basename(split_file)} 时出错: {e}")
                        return False
                
                # 重新打包所有文件到一个 zip
                print(f"  重新打包到合并文件...")
                with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as merged_zip:
                    for root, dirs, files in os.walk(temp_dir):
                        for file in tqdm(files, desc="  打包中", leave=False):
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, temp_dir)
                            merged_zip.write(file_path, arcname)
            finally:
                # 清理临时目录
                try:
                    shutil.rmtree(temp_dir)
                except:
                    pass
        else:
            # 方法2: 如果是二进制分片（使用 split 命令创建），直接二进制合并
            print(f"  检测到二进制分片文件，直接合并...")
            with open(output_zip_path, 'wb') as outfile:
                for split_file in tqdm(split_files, desc="  合并中"):
                    with open(split_file, 'rb') as infile:
                        shutil.copyfileobj(infile, outfile)
            
            # 验证合并后的文件是否为有效的 zip
            try:
                with zipfile.ZipFile(output_zip_path, 'r') as zf:
                    test_list = zf.namelist()[:1]  # 只读取第一个文件测试
            except zipfile.BadZipFile:
                print(f"  错误: 合并后的文件不是有效的 zip 文件")
                return False
        
        print(f"  合并完成: {output_zip_path}")
        return True
        
    except Exception as e:
        print(f"  合并失败: {e}")
        return False

def extract_zip(zip_path, extract_to):
    """
    解压 zip 文件到指定目录
    
    Args:
        zip_path: zip 文件路径
        extract_to: 解压目标目录
    
    Returns:
        bool: 解压是否成功
    """
    try:
        if not os.path.exists(zip_path):
            print(f"  错误: zip 文件不存在: {zip_path}")
            return False
        
        print(f"  解压到: {extract_to}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # 获取文件列表
            file_list = zip_ref.namelist()
            total_files = len(file_list)
            
            # 显示解压进度
            for file_info in tqdm(zip_ref.infolist(), desc="  解压中", total=total_files):
                zip_ref.extract(file_info, extract_to)
        
        return True
    except zipfile.BadZipFile:
        print(f"  错误: zip 文件损坏: {zip_path}")
        return False
    except Exception as e:
        print(f"  解压失败: {e}")
        return False

def check_split_files(part_num, url_base, base_dir):
    """
    检查指定 part 是否存在分片文件
    
    Args:
        part_num: part 编号
        url_base: 下载地址基础路径
        base_dir: 基础目录
    
    Returns:
        tuple: (is_split, split_suffixes) - (是否为分片, 分片后缀列表)
    """
    # 常见的分片后缀模式：aa, ab, ac 或 001, 002, 003
    # 增加更多可能的后缀，因为可能有更多分片
    possible_suffixes = ['aa', 'ab', 'ac', 'ad', 'ae', 'af', 'ag', 'ah', 'ai', 'aj', 'ak', 'al', 
                         '001', '002', '003', '004', '005', '006', '007', '008', '009', '010']
    
    # 先检查是否存在单个 zip 文件
    single_zip_name = f"OpenVidHD_part_{part_num}.zip"
    single_zip_url = f"{url_base}/{single_zip_name}"
    
    try:
        response = requests.head(single_zip_url, allow_redirects=True, timeout=15)
        if response.status_code == 200:
            return (False, [])  # 存在单个文件，不是分片
    except requests.exceptions.Timeout:
        print(f"  警告: 检查单个文件时超时，继续检查分片文件...")
    except:
        pass  # 如果检查失败，继续检查分片文件
    
    # 检查是否存在分片文件
    print(f"  正在检查分片文件...")
    found_suffixes = []
    for suffix in possible_suffixes:
        split_zip_name = f"OpenVidHD_part_{part_num}_part_{suffix}"
        split_zip_url = f"{url_base}/{split_zip_name}"
        
        try:
            response = requests.head(split_zip_url, allow_redirects=True, timeout=15)
            if response.status_code == 200:
                found_suffixes.append(suffix)
                print(f"    找到分片: {split_zip_name}")
        except requests.exceptions.Timeout:
            continue  # 超时则跳过这个后缀
        except:
            continue  # 其他错误也跳过
    
    if found_suffixes:
        # 按字母或数字顺序排序
        if found_suffixes[0].isdigit():
            found_suffixes.sort(key=int)
        else:
            found_suffixes.sort()
        print(f"  共找到 {len(found_suffixes)} 个分片文件")
        return (True, found_suffixes)
    
    print(f"  未找到分片文件")
    return (False, [])

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='批量下载和解压 OpenVidHD_part_*.zip 文件')
    parser.add_argument('--interactive', action='store_true', default=False,
                        help='交互模式，会询问用户（默认是自动模式）')
    parser.add_argument('--start', type=int, default=15,
                        help='起始 part 编号（默认 1）')
    parser.add_argument('--end', type=int, default=25,
                        help='结束 part 编号（默认 14）')
    parser.add_argument('--force-extract', action='store_true', default=False,
                        help='强制重新解压，即使文件夹中已有文件')
    
    args = parser.parse_args()
    
    # 配置
    base_dir = "/media/ubuntu/sda/visual_stimuli_pattern/OpenVid-1M-main"
    url_base = "https://hf-mirror.com/datasets/nkp37/OpenVid-1M/resolve/main/OpenVidHD"
    
    start_part = args.start
    end_part = args.end
    auto_mode = not args.interactive  # 默认自动模式
    skip_existing = not args.force_extract  # 默认跳过已存在的文件
    
    # 确保基础目录存在
    os.makedirs(base_dir, exist_ok=True)
    os.chdir(base_dir)
    
    # 统计信息
    success_count = 0
    failed_downloads = []
    failed_extractions = []
    
    print("=" * 60)
    print(f"开始批量下载和解压 part {start_part} 到 {end_part}")
    print("=" * 60)
    print(f"基础目录: {base_dir}")
    print(f"下载地址: {url_base}")
    print()
    
    # 按顺序处理每个 part
    for part_num in [15, 18, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]:
        print(f"\n[{part_num}/{end_part}] 处理 part_{part_num}")
        print("-" * 60)
        
        part_dir = os.path.join(base_dir, f"part_{part_num}")
        
        # 检查是否为分片文件
        is_split, split_suffixes = check_split_files(part_num, url_base, base_dir)
        
        if is_split and split_suffixes:
            # 处理分片文件
            print(f"检测到分片文件: {len(split_suffixes)} 个分片")
            print(f"分片后缀: {split_suffixes}")
            
            # 步骤 1: 下载所有分片文件
            print(f"步骤 1/3: 下载分片文件")
            split_files = []
            all_downloaded = True
            
            for suffix in split_suffixes:
                split_zip_name = f"OpenVidHD_part_{part_num}_part_{suffix}"
                split_zip_path = os.path.join(base_dir, split_zip_name)
                split_download_url = f"{url_base}/{split_zip_name}"
                
                if os.path.exists(split_zip_path):
                    file_size = os.path.getsize(split_zip_path)
                    print(f"  分片已存在: {split_zip_name} ({file_size / (1024**3):.2f} GB)")
                    split_files.append(split_zip_path)
                else:
                    print(f"  下载分片: {split_zip_name}")
                    if download_file(split_download_url, split_zip_path):
                        file_size = os.path.getsize(split_zip_path)
                        print(f"  下载完成: {file_size / (1024**3):.2f} GB")
                        split_files.append(split_zip_path)
                    else:
                        print(f"  下载失败: {split_zip_name}")
                        all_downloaded = False
                        break
            
            if not all_downloaded:
                failed_downloads.append(part_num)
                print(f"  跳过合并和解压步骤（下载失败）")
                continue
            
            # 步骤 2: 合并分片文件
            print(f"步骤 2/3: 合并分片文件")
            merged_zip_name = f"OpenVidHD_part_{part_num}.zip"
            merged_zip_path = os.path.join(base_dir, merged_zip_name)
            
            if os.path.exists(merged_zip_path):
                file_size = os.path.getsize(merged_zip_path)
                print(f"  合并后的文件已存在: {merged_zip_name} ({file_size / (1024**3):.2f} GB)")
            else:
                if not merge_split_zips(split_files, merged_zip_path):
                    failed_extractions.append(part_num)
                    print(f"  跳过解压步骤（合并失败）")
                    continue
            
            zip_path = merged_zip_path
            
        else:
            # 处理单个 zip 文件（原有逻辑）
            zip_name = f"OpenVidHD_part_{part_num}.zip"
            zip_path = os.path.join(base_dir, zip_name)
            download_url = f"{url_base}/{zip_name}"
            
            # 步骤 1: 下载 zip 文件
            print(f"步骤 1/2: 下载 {zip_name}")
            if os.path.exists(zip_path):
                file_size = os.path.getsize(zip_path)
                print(f"  文件已存在: {zip_path} ({file_size / (1024**3):.2f} GB)")
                download_success = True
            else:
                download_success = download_file(download_url, zip_path)
                if download_success:
                    file_size = os.path.getsize(zip_path)
                    print(f"  下载完成: {file_size / (1024**3):.2f} GB")
                    # 确保 download_success 标记为 True，zip_path 已定义
                else:
                    # 如果下载失败（可能是404），重新检查是否为分片文件
                    print(f"  单个文件下载失败，重新检查是否为分片文件...")
                    is_split_retry, split_suffixes_retry = check_split_files(part_num, url_base, base_dir)
                    
                    if is_split_retry and split_suffixes_retry:
                        # 确实是分片文件，转到分片处理逻辑
                        print(f"检测到分片文件: {len(split_suffixes_retry)} 个分片")
                        print(f"分片后缀: {split_suffixes_retry}")
                        
                        # 步骤 1: 下载所有分片文件
                        print(f"步骤 1/3: 下载分片文件")
                        split_files = []
                        all_downloaded = True
                        
                        for suffix in split_suffixes_retry:
                            split_zip_name = f"OpenVidHD_part_{part_num}_part_{suffix}"
                            split_zip_path = os.path.join(base_dir, split_zip_name)
                            split_download_url = f"{url_base}/{split_zip_name}"
                            
                            if os.path.exists(split_zip_path):
                                file_size = os.path.getsize(split_zip_path)
                                print(f"  分片已存在: {split_zip_name} ({file_size / (1024**3):.2f} GB)")
                                split_files.append(split_zip_path)
                            else:
                                print(f"  下载分片: {split_zip_name}")
                                if download_file(split_download_url, split_zip_path):
                                    file_size = os.path.getsize(split_zip_path)
                                    print(f"  下载完成: {file_size / (1024**3):.2f} GB")
                                    split_files.append(split_zip_path)
                                else:
                                    print(f"  下载失败: {split_zip_name}")
                                    all_downloaded = False
                                    break
                        
                        if not all_downloaded:
                            failed_downloads.append(part_num)
                            print(f"  跳过合并和解压步骤（下载失败）")
                            continue
                        
                        # 步骤 2: 合并分片文件
                        print(f"步骤 2/3: 合并分片文件")
                        merged_zip_name = f"OpenVidHD_part_{part_num}.zip"
                        merged_zip_path = os.path.join(base_dir, merged_zip_name)
                        
                        if os.path.exists(merged_zip_path):
                            file_size = os.path.getsize(merged_zip_path)
                            print(f"  合并后的文件已存在: {merged_zip_name} ({file_size / (1024**3):.2f} GB)")
                        else:
                            if not merge_split_zips(split_files, merged_zip_path):
                                failed_extractions.append(part_num)
                                print(f"  跳过解压步骤（合并失败）")
                                continue
                        
                        zip_path = merged_zip_path
                        download_success = True
                        # 标记为分片文件，用于后续步骤计数
                        is_split = True
                        split_suffixes = split_suffixes_retry  # 使用重试时找到的分片后缀
                    else:
                        # 确实不存在，标记为失败
                        failed_downloads.append(part_num)
                        print(f"  跳过解压步骤（下载失败，且未检测到分片文件）")
                        continue
        
        # 步骤 3 (或 2): 解压到对应文件夹
        # 检查 zip_path 是否存在（无论是单个文件还是合并后的文件）
        if 'zip_path' not in locals() or not os.path.exists(zip_path):
            # 如果 zip_path 未定义或不存在，说明之前的处理都失败了，跳过
            continue
        
        step_num = "3" if (is_split and split_suffixes) else "2"
        print(f"步骤 {step_num}/{step_num}: 解压到 part_{part_num}/")
        os.makedirs(part_dir, exist_ok=True)
        
        # 检查是否已经解压过（如果文件夹中有很多文件，可能已经解压过）
        existing_files = len([f for f in os.listdir(part_dir) if f.endswith('.mp4')]) if os.path.exists(part_dir) else 0
        if existing_files > 100 and skip_existing:  # 假设每个 part 至少有 100 个视频文件
            print(f"  检测到 part_{part_num} 中已有 {existing_files} 个视频文件，跳过解压")
            success_count += 1
            continue
        
        extract_success = extract_zip(zip_path, part_dir)
        if extract_success:
            print(f"  解压完成!")
            success_count += 1
        else:
            failed_extractions.append(part_num)
        
        # 短暂停顿，避免请求过快
        time.sleep(1)
    
    # 输出统计信息
    print("\n" + "=" * 60)
    print("处理完成!")
    print("=" * 60)
    print(f"成功处理: {success_count}/{end_part - start_part + 1} 个 part")
    
    if failed_downloads:
        print(f"\n下载失败的 part: {failed_downloads}")
    if failed_extractions:
        print(f"解压失败的 part: {failed_extractions}")
    
    if success_count == (end_part - start_part + 1):
        print("\n✓ 所有 part 都已成功下载和解压!")
    
    # 询问是否删除 zip 文件以节省空间（仅在非自动模式下询问）
    if success_count > 0 and not auto_mode:
        print(f"\n提示: 如果需要节省磁盘空间，可以删除已解压的 zip 文件")
        response = input(f"是否删除已成功解压的 zip 文件？(y/N): ").strip().lower()
        if response == 'y':
            deleted_count = 0
            for part_num in range(start_part, end_part + 1):
                if part_num not in failed_downloads and part_num not in failed_extractions:
                    zip_name = f"OpenVidHD_part_{part_num}.zip"
                    zip_path = os.path.join(base_dir, zip_name)
                    if os.path.exists(zip_path):
                        try:
                            os.remove(zip_path)
                            deleted_count += 1
                            print(f"  已删除: {zip_name}")
                        except Exception as e:
                            print(f"  删除失败 {zip_name}: {e}")
            print(f"\n已删除 {deleted_count} 个 zip 文件")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断操作")
        sys.exit(1)
    except Exception as e:
        print(f"\n发生未预期的错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

