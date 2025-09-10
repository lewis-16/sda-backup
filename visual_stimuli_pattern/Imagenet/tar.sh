#!/bin/bash

# 遍历当前目录中的所有.tar文件
for tar_file in *.tar; do
    # 检查文件是否存在（避免无匹配时的错误）
    if [ ! -f "$tar_file" ]; then
        continue
    fi
    
    # 提取基础文件名（去掉.tar后缀）
    dir_name="${tar_file%.tar}"
    
    # 创建目标文件夹（如果不存在）
    mkdir -p "$dir_name"
    
    # 解压文件到目标文件夹
    tar -xf "$tar_file" -C "$dir_name"
    
    # 打印解压信息
    echo "解压完成: $tar_file -> $dir_name/"
done

echo "所有.tar文件已解压完成"