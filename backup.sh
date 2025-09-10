#!/bin/bash
# 简单的代码备份脚本 - 直接遍历文件

echo "开始备份代码文件..."

# 重置暂存区
git reset

# 定义代码文件扩展名
code_extensions=("*.py" "*.ipynb" "*.md" "*.sh")

# 遍历所有代码文件并添加到git
file_count=0
for ext in "${code_extensions[@]}"; do
    echo "正在查找 $ext 文件..."
    files=$(find . -name "$ext" -type f)
    if [ -n "$files" ]; then
        echo "$files" | while read -r file; do
            git add "$file"
            ((file_count++))
            if [ $((file_count % 50)) -eq 0 ]; then
                echo "已添加 $file_count 个文件..."
            fi
        done
    fi
done

echo "总共添加了 $file_count 个代码文件"

# 检查是否有变更
if git diff --staged --quiet; then
    echo "没有新的变更需要提交"
    exit 0
fi

# 提交变更
commit_message="代码备份 - $(date '+%Y-%m-%d %H:%M:%S')"
git commit -m "$commit_message"

# 推送到远程仓库
git push -u origin main

echo "代码备份完成!"
