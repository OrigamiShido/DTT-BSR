# -*- coding: utf-8 -*-
"""
一个用于将 GTSinger 处理后的 Vocals 文件随机混合到 Slakh2100 处理后的目录结构的脚本。

此脚本会为每个 split (train/test)：
- 收集 GTSinger 中的所有 Vocals .flac 文件。
- 收集 Slakh2100 中的所有 TrackID 目录。
- 随机将每个 GTSinger Vocals 文件复制到随机选择的 Slakh2100 TrackID 的 Vocals 子文件夹中（如果不存在，则创建）。

注意：这会修改 Slakh2100 的目录结构，添加 Vocals 子文件夹和文件。请备份数据。
"""
import shutil
import random
from pathlib import Path
from tqdm import tqdm

# --- 配置区域 ---
# 请根据您的环境修改这些路径

# Slakh2100 处理后的根目录
SLAKH_ROOT = Path("/home/student/nyr/data/slakh2100_rawstems_format_filtered")

# GTSinger 处理后的根目录
GTSINGER_ROOT = Path("/home/student/nyr/data/gtsinger_rawstems_format")

# --- 结束配置区域 ---

def mix_gtsinger_to_slakh():
    """
    主处理函数。
    """
    print("开始将 GTSinger Vocals 文件随机混合到 Slakh2100 目录...")

    for split in ["train", "test"]:
        slakh_split_dir = SLAKH_ROOT / split
        gtsinger_split_dir = GTSINGER_ROOT / split

        if not slakh_split_dir.exists():
            print(f"警告：Slakh2100 '{split}' 目录不存在，跳过。")
            continue
        if not gtsinger_split_dir.exists():
            print(f"警告：GTSinger '{split}' 目录不存在，跳过。")
            continue

        print(f"\n--- 处理 {split} 分割 ---")

        # 收集 GTSinger Vocals 中的所有 .flac 文件
        vocals_files = list(gtsinger_split_dir.glob("**/Vocals/*.flac"))
        print(f"找到 {len(vocals_files)} 个 GTSinger Vocals 文件。")

        if not vocals_files:
            continue

        # 收集 Slakh2100 中的所有 TrackID 目录
        track_dirs = [d for d in slakh_split_dir.iterdir() if d.is_dir() and d.name.startswith("Track")]
        print(f"找到 {len(track_dirs)} 个 Slakh2100 Track 目录。")

        if not track_dirs:
            continue

        # 为每个 Vocals 文件随机选择一个 Track 目录，并复制到其 Vocals 子目录
        copied_count = 0
        for vocals_file in tqdm(vocals_files, desc=f"Mixing {split} Vocals"):
            # 随机选择一个 Track 目录
            target_track_dir = random.choice(track_dirs)

            # 创建或使用 Vocals 子目录
            vocals_subdir = target_track_dir / "Vocals"
            vocals_subdir.mkdir(exist_ok=True)

            # 目标文件路径（保持原文件名）
            destination_path = vocals_subdir / vocals_file.name

            # 复制文件
            shutil.copy2(vocals_file, destination_path)
            copied_count += 1

        print(f"在 {split} 分割中，共混合了 {copied_count} 个 Vocals 文件。")

    print("\n--------------------")
    print("全部处理完成！")
    print("GTSinger Vocals 文件已随机分散到 Slakh2100 的 Track 目录下的 Vocals 子文件夹中。")
    print("--------------------")


if __name__ == "__main__":
    mix_gtsinger_to_slakh()