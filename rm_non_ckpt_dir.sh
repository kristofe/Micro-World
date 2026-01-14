#!/bin/bash

TARGET_DIR="$1"

if [[ -z "$TARGET_DIR" || ! -d "$TARGET_DIR" ]]; then
    echo "用法: $0 <目标目录>"
    exit 1
fi

echo "扫描目录: $TARGET_DIR"

# 遍历目标目录下的所有子文件夹
for subdir in "$TARGET_DIR"/*/; do
    # 跳过不是目录的项
    [[ -d "$subdir" ]] || continue

    # 查找当前子目录中是否有以 ckpt 开头的子文件夹
    ckpt_dirs=$(find "$subdir" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint*')

    if [[ -z "$ckpt_dirs" ]]; then
        echo "❌ 删除: $subdir（不含 ckpt* 子文件夹）"
        rm -rf "$subdir"
    else
        echo "✅ 保留: $subdir（包含 ckpt* 子文件夹）"
    fi
done
