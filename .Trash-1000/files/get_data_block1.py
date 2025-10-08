import requests
import os
import time
from tqdm import tqdm

# 基础URL
base_url = "https://gin.g-node.org/doi/TVSD/raw/master/monkeyF/20240112/Block_1/"

# 文件列表
files = [
    "Hub1-instance1_B001.ns6",
    "Hub1-instance2_B001.nev",
    "Hub1-instance2_B001.ns6",
    "Hub2-instance1_B001.nev",
    "Hub2-instance1_B001.ns6",
    "Hub2-instance2_B001.nev",
    "Hub2-instance2_B001.ns6",
    "NSP-instance1_B001.nev",
    "NSP-instance1_B001.ns6",
    "NSP-instance2_B001.nev",
    "NSP-instance2_B001.ns6",
    "THINGS_monkeyF_20240112_B1_session.json",
    "instance1_B001.ccf",
    "instance2_B001.ccf"
]

# 创建下载目录
download_dir = "/media/ubuntu/sda/Monkey/20240112/block1"
os.makedirs(download_dir, exist_ok=True)

# 下载函数带进度条
def download_file_with_progress(filename):
    url = base_url + filename
    try:
        # 发送HEAD请求获取文件大小
        response_head = requests.head(url)
        total_size = int(response_head.headers.get('content-length', 0))
        
        # 开始下载
        response = requests.get(url, stream=True)
        response.raise_for_status()  # 检查请求是否成功
        
        filepath = os.path.join(download_dir, filename)
        
        # 使用tqdm创建进度条
        progress_bar = tqdm(
            total=total_size, 
            unit='iB', 
            unit_scale=True,
            desc=filename[:20] + "..." if len(filename) > 20 else filename,
            ncols=80  # 限制进度条宽度
        )
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    progress_bar.update(len(chunk))
        
        progress_bar.close()
        
        # 验证文件大小
        if total_size != 0 and progress_bar.n != total_size:
            print(f"警告: {filename} 可能未完整下载")
        
        print(f"成功下载: {filename}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"下载失败 {filename}: {e}")
        return False

# 主下载函数
def main():
    print("开始下载文件...")
    print("=" * 50)
    
    success_count = 0
    total_files = len(files)
    
    for i, file in enumerate(files, 1):
        print(f"[{i}/{total_files}] ", end="")
        success = download_file_with_progress(file)
        if success:
            success_count += 1
        else:
            print(f"将重试下载: {file}")
            time.sleep(2)  # 等待2秒后重试
            success = download_file_with_progress(file)
            if success:
                success_count += 1
            else:
                print(f"第二次尝试下载 {file} 失败，跳过此文件")
    
    print("=" * 50)
    print(f"下载完成! 成功下载 {success_count}/{total_files} 个文件")

if __name__ == "__main__":
    main()