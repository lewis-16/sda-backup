import numpy as np
import pyintan
from pathlib import Path
import sys
import os

def main():
    file_list = os.listdir("/media/ubuntu/sda/littlerat/raw_data/rhs/Intan_Recordings")
    
    input_dir = Path(sys.argv[1]) 
    output_bin_dir = sys.argv[2]
    output_bin = str(input_dir).split("/")[-1]



    rhs_files = sorted(input_dir.glob("*.rhs"))
    if not rhs_files:
        raise FileNotFoundError("未找到.rhs文件")
    
    all_data = []

    for file in rhs_files:
        print(f"读取: {file.name}")
        result = pyintan.File(file)

        all_data.append(result.analog_signals[0].signal)

    merged_data = np.hstack(all_data)
    print(f'记录文件通道数： {merged_data.shape[0]}')
    print(f'记录文件采样数： {merged_data.shape[1]}')

    print(f'正在保存文件: {output_bin}...')

    with open(f'{output_bin_dir}/{output_bin}.bin', 'wb') as bin_file:
        merged_data.astype(np.float16).tofile(f'{output_bin_dir}/{output_bin}.bin')
    
    print(f'已保存文件: {output_bin}')


if __name__ == "__main__":
    main()