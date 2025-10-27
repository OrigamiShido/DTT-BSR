# -*- coding: utf-8 -*-
"""
一个用于为已处理的 Slakh2100 数据集生成 .txt 数据划分文件的脚本。

此脚本会读取 `process_slakh.py` 生成的层级目录结构（.../TrackID/InstrumentClass/），
并根据乐器类别和子类别，创建 .txt 文件列表。

每个 .txt 文件将包含指向符合条件的 TrackID 目录的绝对路径。
"""
import os
import re
from pathlib import Path
from tqdm import tqdm

# --- 配置区域 ---
# 请根据您的环境修改这些路径
path = "/home/student/nyr/data/slakh2100_rawstems_format_filtered"
# `process_slakh.py` 脚本的输出根目录
DATASET_ROOT = Path(path)

# 生成的 .txt 文件将要保存的目录
TXT_OUTPUT_ROOT = Path("./slakh_data_splits")

# --- 核心映射和类别 (从 process_slakh.py 继承) ---

# MSR 挑战赛的目标乐器类别
MSR_CLASSES = [
    "vocals", "guitars", "keyboards", "bass", "synthesizers",
    "drums", "percussion", "orchestral"
]

# 新增：主类别的缩写，用于生成文件名
CLASS_ABBREVIATIONS = {
    "vocals": "Voc",
    "guitars": "Gtr",
    "keyboards": "Kbs",
    "bass": "Bass",
    "synthesizers": "Syn",
    "drums": "Drums",
    "percussion": "Perc",
    "orchestral": "Orch"
}

# Slakh2100 原始乐器名到 MSR 类别的映射
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
    "Vocals": "vocals"
}

# 子类别映射 (用于生成 Gtr_AG, Kbs_EP 等文件)
SUBCATEGORY_MAP = {
    "guitars": {
        "AcousticGuitar": "AG",
        "ElectricGuitar": "EG",
        "Guitar": "",  # 未指定类型的吉他
        "Ukulele": "UK"
    },
    "keyboards": {
        "Piano": "PN",
        "ElectricPiano": "EP",
        "Organ": "OR",
        "Harpsichord": "HP",
        "Clavinet": "CT"
    },
    "bass": {
        "ElectricBass": "EB",
        "UprightBass": "UB",
        "Bass": "",  # 未指定类型的贝斯
    },
    "orchestral": {
        "Violin": "VLN",
        "Viola": "VIO",
        "Cello": "CEL",
        "Contrabass": "CON",
        "Strings": "STR",
        "Ensemble": "ENS",
        "Trumpet": "TRU",
        "Trombone": "TRO",
        "FrenchHorn": "FRE",
        "Tuba": "TUB",
        "Brass": "BRA",
        "Saxophone": "SAX",
        "TenorSax": "TEN",
        "AltoSax": "ALT",
        "BaritoneSax": "BAR",
        "SopranoSax": "SOP",
        "Clarinet": "CLA",
        "Oboe": "OBO",
        "Bassoon": "BAS",
        "Reed": "REE",
        "Flute": "FLU",
        "Piccolo": "PIC",
        "Pipe": "PIP",
        "Ethnic": "ETH",
    },
    "drums": {
        "Drums": "",
        "Drum Machine": "Drumm",
    },
    "synthesizers": {
        "Synthesizer": "Synth",
        "Synth Pad": "SP",
        "Synth Lead": "SL",
        "Synth Bass": "SB",
        "Synth Effects": "SE",
    },
    "percussion": {
        "Chromatic Percussion": "CP",
        "Percussive": "Per",
        "Timpani": "TIM",
    },
    "vocals": {
        "Breathy": "BR",
        "Falsetto": "FA",
        "MixedVoice": "MV",
        "Pharyngeal": "PH",
        "Vibrato": "VI",
        "Glissando": "GL",
        "Control": "CT",
        "Unknown": ""  # 如果有未知，跳过或处理为空
    }
}


# --- 辅助函数 ---

def sanitize_string(name):
    """清理字符串，与 process_slakh.py 中的逻辑保持一致。"""
    name = re.sub(r'\(.*\)', '', name)
    name = re.sub(r'[^a-zA-Z0_9\s]', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    name = name.replace(' ', '')
    return name


# --- 主逻辑 ---

def generate_txt_splits():
    """
    主函数，生成所有 .txt 文件。
    """
    print("开始生成数据划分 .txt 文件...")
    TXT_OUTPUT_ROOT.mkdir(exist_ok=True)

    # 优化点：自动创建一个从清理后的乐器名到 MSR 类别的映射
    sanitized_to_msr_class = {
        sanitize_string(k): v for k, v in SLAKH_TO_MSR_MAP.items()
    }

    # 变更：MSR 类别文件夹名 (e.g., "Guitars", "Vocals")
    class_folder_names = {cls: cls.capitalize() for cls in MSR_CLASSES}

    # 变更：只处理 train 和 test
    for split in ["train", "test"]:
        split_dir = DATASET_ROOT / split
        if not split_dir.exists():
            print(f"警告：跳过数据划分 '{split}' - 目录未找到: {split_dir}")
            continue

        print(f"\n--- 正在处理: {split} ---")

        # 使用 set 存储唯一的 Track 目录路径
        files_by_class = {cls: set() for cls in MSR_CLASSES}
        files_by_subcategory = {}  # e.g., {"Kbs_EP": set(), "Gtr_AG": set()}

        # 变更：迭代所有子目录 (Track* 和 GTSinger* 目录)
        track_dirs = [d for d in split_dir.iterdir() if d.is_dir()]

        for track_dir in tqdm(track_dirs, desc=f"Scanning {split} tracks"):
            try:
                track_path_str = str(track_dir.resolve())

                # 变更：检查此音轨目录包含哪些乐器 *文件夹*
                for msr_class, class_folder in class_folder_names.items():

                    instrument_dir = track_dir / class_folder

                    # 核心逻辑：如果找到了乐器文件夹 (e.g., .../Track00001/Guitars)
                    if instrument_dir.exists() and instrument_dir.is_dir():

                        # 1. 将音轨路径添加到主类别
                        files_by_class[msr_class].add(track_path_str)

                        # 2. 检查子类别 (主要用于 Slakh)
                        if msr_class in SUBCATEGORY_MAP:
                            sub_map = SUBCATEGORY_MAP[msr_class]

                            # 扫描文件夹内的 flac 文件以确定子类别
                            flac_files = list(instrument_dir.glob("*.flac"))

                            for file_path in flac_files:
                                # Slakh: S01_ElectricGuitar.flac -> parts=['S01', 'ElectricGuitar']
                                # GTSinger: song1_MixedVoice_Chinese.flac -> parts=['song1', 'MixedVoice', 'Chinese']
                                parts = file_path.stem.split('_')

                                sanitized_instrument = None
                                if parts[0].startswith('S') and len(parts) >= 2:
                                    # Slakh 格式
                                    sanitized_instrument = parts[1]
                                elif msr_class == 'vocals' and len(parts) >= 2:
                                    # GTSinger 格式
                                    sanitized_instrument = parts[1]  # e.g., "MixedVoice"

                                if sanitized_instrument:
                                    sub_abbr = sub_map.get(sanitized_instrument)

                                    if sub_abbr is not None and sub_abbr != "":
                                        abbr = CLASS_ABBREVIATIONS.get(msr_class)
                                        sub_key = f"{abbr}_{sub_abbr}"

                                        if sub_key not in files_by_subcategory:
                                            files_by_subcategory[sub_key] = set()
                                        files_by_subcategory[sub_key].add(track_path_str)

                                        # 优化：找到一个匹配的子类别就足够了
                                        break
            except Exception as e:
                print(f"处理目录 {track_dir} 时出错: {e}")

        # --- 写入文件 ---
        print(f"正在写入 {split} 的 .txt 文件...")
        for cls, files_set in files_by_class.items():
            if files_set:
                files_list = sorted(list(files_set))
                abbr = CLASS_ABBREVIATIONS.get(cls, cls.capitalize()[:3])

                # 1. 写入主类别文件 (e.g., Gtr_train.txt, Voc_train.txt)
                main_txt_path = TXT_OUTPUT_ROOT / f"{abbr}_{split}.txt"
                with open(main_txt_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(files_list))


        # 写入子类别的 .txt 文件
        for sub_key, files_set in files_by_subcategory.items():
            if files_set:
                files_list = sorted(list(files_set))
                txt_path = TXT_OUTPUT_ROOT / f"{sub_key}_{split}.txt"
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(files_list))

    print("\n--------------------")
    print("全部处理完成！")
    print(f"所有 .txt 文件已保存至: {TXT_OUTPUT_ROOT}")
    print("--------------------")


if __name__ == "__main__":
    generate_txt_splits()

