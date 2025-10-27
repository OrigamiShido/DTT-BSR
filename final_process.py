# -*- coding: utf-8 -*-
"""
一个用于合并 'train' 和 'test' 数据集文件夹，
并自动更新 .txt 数据划分文件中路径的脚本。

步骤 1:
- 创建一个新的 'all_data' 文件夹。
- 将 'DATASET_ROOT/train/' 和 'DATASET_ROOT/test/' 中的所有音轨文件夹
  移动到 'DATASET_ROOT/all_data/' 中。

步骤 2:
- 扫描 'TXT_SPLITS_DIR' 中的所有 .txt 文件。
- 读取每个文件，将其中的旧路径 (e.g., .../test/Track0001)
  重写为新路径 (e.g., .../all_data/Track0001)。
- 将更新后的 .txt 文件保存到一个新的 'updated_data_splits' 目录中。
"""
import os
import shutil
from pathlib import Path
from tqdm import tqdm

# --- 1. 配置区域 ---

# 包含 'train' 和 'test' 文件夹的数据集根目录
# (例如: /home/student/nyr/data/dataset1)
DATASET_ROOT = Path("/home/student/nyr/data/slakh2100_rawstems_format_filtered")

# 你希望将所有文件合并到的新文件夹的名称
NEW_MERGED_DIR_NAME = "msr_dataset"

# 包含你之前生成的 .txt 文件 (e.g., Voc_train.txt) 的目录
# (例如: ./data_splits_with_vocals)
TXT_SPLITS_DIR = Path("/home/student/nyr/data/slakh2100_rawstems_format_filtered/slakh_data_splits")

# 更新后的 .txt 文件将保存到的新目录
UPDATED_TXT_SPLITS_DIR = Path("./updated_data_splits")


# --- 2. 脚本主逻辑 ---

def move_files_to_merged_dir():
    """
    步骤 1: 合并 'train' 和 'test' 文件夹。
    """
    train_dir = DATASET_ROOT / "train"
    test_dir = DATASET_ROOT / "test"
    merged_dir = DATASET_ROOT / NEW_MERGED_DIR_NAME

    if not train_dir.exists() or not test_dir.exists():
        print(f"错误：找不到 'train' ({train_dir}) 或 'test' ({test_dir}) 目录。")
        print("请检查 DATASET_ROOT 配置。")
        return False

    merged_dir.mkdir(exist_ok=True)
    print(f"--- 步骤 1: 开始合并文件到 {merged_dir} ---")

    # 移动 train 目录下的所有文件夹
    train_tracks = [d for d in train_dir.iterdir() if d.is_dir()]
    print(f"正在移动 {len(train_tracks)} 个 'train' 目录...")
    for track_dir in tqdm(train_tracks, desc="Moving train files"):
        try:
            shutil.move(str(track_dir), str(merged_dir / track_dir.name))
        except Exception as e:
            print(f"移动 {track_dir.name} 时出错 (可能已存在?): {e}")

    # 移动 test 目录下的所有文件夹
    test_tracks = [d for d in test_dir.iterdir() if d.is_dir()]
    print(f"正在移动 {len(test_tracks)} 个 'test' 目录...")
    for track_dir in tqdm(test_tracks, desc="Moving test files"):
        try:
            shutil.move(str(track_dir), str(merged_dir / track_dir.name))
        except Exception as e:
            print(f"移动 {track_dir.name} 时出错 (可能已存在?): {e}")

    print("文件合并完成。")

    # （可选）删除空的 train 和 test 目录
    # try:
    #     if not any(train_dir.iterdir()):
    #         train_dir.rmdir()
    #         print(f"已删除空的 'train' 目录。")
    #     if not any(test_dir.iterdir()):
    #         test_dir.rmdir()
    #         print(f"已删除空的 'test' 目录。")
    # except Exception as e:
    #     print(f"删除空目录时出错: {e}")

    return True


def update_txt_file_paths():
    """
    步骤 2: 更新 .txt 文件中的路径。
    """
    if not TXT_SPLITS_DIR.exists():
        print(f"错误：找不到 .txt 文件目录: {TXT_SPLITS_DIR}")
        print("请检查 TXT_SPLITS_DIR 配置。")
        return False

    UPDATED_TXT_SPLITS_DIR.mkdir(exist_ok=True)

    new_merged_dir_path = DATASET_ROOT / NEW_MERGED_DIR_NAME

    print(f"\n--- 步骤 2: 开始更新 .txt 文件路径 ---")
    print(f"新的 .txt 文件将保存到: {UPDATED_TXT_SPLITS_DIR}")

    txt_files = list(TXT_SPLITS_DIR.glob("*.txt"))
    if not txt_files:
        print(f"警告：在 {TXT_SPLITS_DIR} 中未找到 .txt 文件。")
        return False

    for txt_file_path in tqdm(txt_files, desc="Updating .txt files"):
        try:
            with open(txt_file_path, 'r', encoding='utf-8') as f:
                lines = f.read().splitlines()

            updated_lines = []
            for line in lines:
                if not line.strip():
                    continue

                # 从旧路径 (e.g., .../test/Track0001) 中提取音轨名 (Track0001)
                track_name = Path(line).name

                # 创建指向新位置的绝对路径
                new_path = new_merged_dir_path.resolve() / track_name
                updated_lines.append(str(new_path))

            # 将更新后的路径写入新文件
            new_txt_path = UPDATED_TXT_SPLITS_DIR / txt_file_path.name
            with open(new_txt_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(updated_lines))

        except Exception as e:
            print(f"处理 {txt_file_path.name} 时出错: {e}")

    print(".txt 文件路径更新完成。")
    return True


def main():
    print("开始执行数据合并与路径更新脚本...")

    # 步骤 1: 移动文件
    if not move_files_to_merged_dir():
        print("步骤 1 失败，脚本终止。")
        return

    # 步骤 2: 更新路径
    if not update_txt_file_paths():
        print("步骤 2 失败。")
        return

    print("\n--------------------")
    print("全部处理完成！")
    print(f"所有音轨文件已移动到: {DATASET_ROOT / NEW_MERGED_DIR_NAME}")
    print(f"所有更新后的 .txt 文件已保存至: {UPDATED_TXT_SPLITS_DIR}")
    print("--------------------")


if __name__ == "__main__":
    main()
