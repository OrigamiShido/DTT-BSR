# -*- coding: utf-8 -*-
"""
一个用于将 Slakh2100 数据集转换为新的、按音轨和乐器分类的目录结构的脚本。

此脚本会扫描 Slakh2100 目录，读取每个音轨的元数据，
然后将每个 stem 文件复制到新的目录结构中。

"""
import shutil
import yaml
import re
from pathlib import Path
from tqdm import tqdm

# --- 配置区域 ---
# 请根据您的环境修改这些路径

# Slakh2100 数据集的根目录
SLAKH_ROOT = Path("path/to/slakh2100_flac_redux")

# 转换后文件的输出目录
OUTPUT_ROOT = Path("path/to/newdataset")

# --- 新增：MSR 挑战赛的目标乐器类别 ---
MSR_INSTRUMENT_CLASSES = {
    "vocals", "guitars", "keyboards", "bass", "synthesizers",
    "drums", "percussion", "orchestral"
}

# --- 新增：Slakh2100 乐器到 MSR 类别的映射 ---
SLAKH_TO_MSR_MAP = {
    # Keyboards
    "Piano": "keyboards",
    "Electric Piano": "keyboards",
    "Organ": "keyboards",
    "Harpsichord": "keyboards",
    "Clavinet": "keyboards",
    # Guitars
    "Guitar": "guitars",
    "Acoustic Guitar": "guitars",
    "Electric Guitar": "guitars",
    "Ukulele": "guitars",
    # Bass
    "Bass": "bass",
    "Electric Bass": "bass",
    "Upright Bass": "bass",
    # Orchestral (Strings, Brass, Reed, Pipe)
    "Violin": "orchestral",
    "Viola": "orchestral",
    "Cello": "orchestral",
    "Contrabass": "orchestral",
    "Strings": "orchestral",
    "Ensemble": "orchestral",
    "Trumpet": "orchestral",
    "Trombone": "orchestral",
    "French Horn": "orchestral",
    "Tuba": "orchestral",
    "Brass": "orchestral",
    "Saxophone": "orchestral",
    "Tenor Sax": "orchestral",
    "Alto Sax": "orchestral",
    "Baritone Sax": "orchestral",
    "Soprano Sax": "orchestral",
    "Clarinet": "orchestral",
    "Oboe": "orchestral",
    "Bassoon": "orchestral",
    "Reed": "orchestral",
    "Flute": "orchestral",
    "Piccolo": "orchestral",
    "Pipe": "orchestral",
    "Ethnic": "orchestral",  # 将民族乐器归类于管弦乐
    # Drums
    "Drums": "drums",
    "Drum Machine": "drums",
    # Synthesizers
    "Synthesizer": "synthesizers",
    "Synth Pad": "synthesizers",
    "Synth Lead": "synthesizers",
    "Synth Bass": "synthesizers",
    "Synth Effects": "synthesizers",
    # Percussion
    "Chromatic Percussion": "percussion",
    "Percussive": "percussion",
    "Timpani": "percussion",
    # 被忽略的类别 (Sound Effects, etc.)
}


# --- 结束配置区域 ---

def sanitize_filename(name):
    """
    清理字符串，使其适用于文件名。
    移除特殊字符并将空格直接移除。
    """
    name = re.sub(r'\(.*\)', '', name)
    name = re.sub(r'[^a-zA-Z0-9\s]', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    name = name.replace(' ', '')
    return name


def process_slakh_to_rawstems_format():
    """
    主处理函数。
    """
    print("开始处理 Slakh2100 数据集...")
    print(f"源目录: {SLAKH_ROOT}")
    print(f"目标目录: {OUTPUT_ROOT}")
    print(f"将只保留以下类别的音轨: {', '.join(sorted(list(MSR_INSTRUMENT_CLASSES)))}")

    if not SLAKH_ROOT.exists():
        print(f"错误：源目录 '{SLAKH_ROOT}' 不存在。请检查路径配置。")
        return

    splits = ['train', 'validation', 'test']

    for split in splits:
        source_split_dir = SLAKH_ROOT / split
        output_split_dir = OUTPUT_ROOT / split

        if not source_split_dir.exists():
            print(f"\n警告：找不到源数据划分目录 '{source_split_dir}'，跳过此部分。")
            continue

        print(f"\n--- 正在处理数据划分: '{split}' ---")

        output_split_dir.mkdir(parents=True, exist_ok=True)

        metadata_files = list(source_split_dir.glob("**/metadata.yaml"))

        if not metadata_files:
            print(f"警告：在 '{source_split_dir}' 中找不到任何 'metadata.yaml' 文件。")
            continue

        print(f"在 '{split}' 划分中找到了 {len(metadata_files)} 个音轨，开始过滤、复制和重命名...")

        copied_count = 0
        for metadata_file in tqdm(metadata_files, desc=f"Processing {split} Tracks"):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = yaml.safe_load(f)

                track_path = metadata_file.parent
                track_id = track_path.name

                if 'stems' not in metadata or not metadata['stems']:
                    continue

                for stem_id, stem_info in metadata['stems'].items():
                    instrument_name = stem_info.get('inst_class')
                    if not instrument_name:
                        continue

                    msr_class = SLAKH_TO_MSR_MAP.get(instrument_name)

                    if not msr_class or msr_class not in MSR_INSTRUMENT_CLASSES:
                        continue

                    sanitized_instrument_name = sanitize_filename(instrument_name)
                    source_stem_path = track_path / 'stems' / f"{stem_id}.flac"

                    if source_stem_path.exists():
                        # --- 变更：创建新的、层级更深的目标目录结构 ---

                        # 1. 目标目录现在是: .../<split>/<TrackID>/<InstrumentClass>/
                        instrument_class_dir = output_split_dir / track_id / msr_class.capitalize()
                        instrument_class_dir.mkdir(parents=True, exist_ok=True)

                        # 2. 新的文件名不再包含 TrackID
                        new_filename = f"{stem_id}_{sanitized_instrument_name}.flac"

                        # 3. 最终的目标文件路径
                        destination_stem_path = instrument_class_dir / new_filename

                        # 4. 复制文件到新位置
                        shutil.copy2(source_stem_path, destination_stem_path)
                        copied_count += 1

            except yaml.YAMLError as e:
                print(f"\n警告：解析 YAML 文件时出错: {metadata_file} - {e}")
            except Exception as e:
                print(f"\n警告：处理音轨时发生未知错误: {metadata_file} - {e}")

        print(f"在 '{split}' 划分中，共复制了 {copied_count} 个符合条件的音轨文件。")

    print("\n--------------------")
    print("全部处理完成！")
    print(f"所有符合条件的 stem 文件已按新的目录结构保存到: {OUTPUT_ROOT}")
    print("--------------------")


if __name__ == "__main__":
    process_slakh_to_rawstems_format()

