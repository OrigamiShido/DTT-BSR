# -*- coding: utf-8 -*-
"""
一个用于将 GTSinger 数据集转换为新的、按音轨和人声分类的目录结构的脚本。

此脚本会扫描 GTSinger 目录，遍历所有 .wav 文件，对于每个音频文件加载对应的 .json 元数据（如果存在），
然后将音频文件复制并转换为 FLAC 格式到新的目录结构中。

此版本专注于 MSR 挑战赛的 'vocals' 类别，并根据语言、歌手、技巧进行分组。
它会从路径推断语言、歌手、技巧，并过滤掉 Paired_Speech_Group（speech），只保留 singing。

新增：随机分割数据集为 train (80%) 和 test (20%)，因为 GTSinger 没有官方 split。

例如:
源文件: English/EN-Alto-1/Breathy/all is found/Breathy_Group/0000.wav
推断: lang="EN", singer="EN-Alto-1", technique="Breathy"
目标文件: <OUTPUT_ROOT>/train/EN_ENAlto1_Breathy/Vocals/0000_Breathy_EN.flac
"""
import json
import re
import random
from pathlib import Path
from tqdm import tqdm
import soundfile as sf
import numpy as np
import librosa  # 如果需要重采样，取消注释并确保安装

# --- 配置区域 ---
# 请根据您的环境修改这些路径

# GTSinger 数据集的根目录（下载并解压后的路径）
GTSINGER_ROOT = Path("/home/student/nyr/data/GTSinger")

# 转换后文件的输出目录
OUTPUT_ROOT = Path("/home/student/nyr/data/gtsinger_rawstems_format")

# --- 新增：MSR 挑战赛的目标类别（仅 vocals） ---
MSR_INSTRUMENT_CLASSES = {"vocals"}

# --- 新增：GTSinger 技巧到 MSR 类别的映射（基于路径标签） ---
GTSINGER_TECHNIQUE_MAP = {
    "control_group": "Control",
    "falsetto_group": "Falsetto",
    "mixed_voice_group": "MixedVoice",
    "breathy_group": "Breathy",
    "pharyngeal_group": "Pharyngeal",
    "vibrato_group": "Vibrato",
    "glissando_group": "Glissando",
    # 技巧文件夹名
    "breathy": "Breathy",
    "falsetto": "Falsetto",
    "mixed_voice": "MixedVoice",
    "pharyngeal": "Pharyngeal",
    "vibrato": "Vibrato",
    "glissando": "Glissando",
    # 默认 fallback
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


def infer_technique_from_path(path: Path):
    """
    从路径推断技巧。
    例如，从 "Breathy/all is found/Breathy_Group/0000.wav" 推断 "Breathy"。
    检查技巧文件夹和组文件夹。
    """
    parts = path.parts
    for part in reversed(parts):  # 从后往前检查
        lower_part = part.lower()
        for key in GTSINGER_TECHNIQUE_MAP:
            if key in lower_part:
                return sanitize_filename(GTSINGER_TECHNIQUE_MAP[key])
    return "Unknown"


def infer_lang_from_path(path: Path):
    """
    从路径推断语言 (e.g., English -> "EN")。
    假设语言文件夹是根下的第一个文件夹，如 English, Chinese。
    """
    for part in path.parts:
        lower_part = part.lower()
        if lower_part in ['english', 'chinese', 'spanish', 'german', 'russian', 'japanese', 'korean', 'french', 'italian']:
            return part.upper()[:2]  # EN, CH 等
    return "Unknown"


def infer_singer_from_path(path: Path):
    """
    从路径推断歌手 (e.g., EN-Alto-1)。
    假设歌手文件夹如 EN-Alto-1。
    """
    for part in path.parts:
        if '-' in part and len(part.split('-')) > 1:  # 如 EN-Alto-1
            return part
    return "Unknown"


def process_gtsinger_to_rawstems_format():
    """
    主处理函数。
    """
    print("开始处理 GTSinger 数据集...")
    print(f"源目录: {GTSINGER_ROOT}")
    print(f"目标目录: {OUTPUT_ROOT}")
    print(f"将只保留 'vocals' 类别的音频文件（singing，只过滤 speech）。")

    if not GTSINGER_ROOT.exists():
        print(f"错误：源目录 '{GTSINGER_ROOT}' 不存在。请检查路径配置。")
        return

    # 遍历所有 .wav 文件
    wav_files = list(GTSINGER_ROOT.glob('**/*.wav'))
    print(f"找到了 {len(wav_files)} 个 .wav 文件，开始处理...")

    # 随机 shuffle 并分割：80% train, 20% test
    random.shuffle(wav_files)
    split_index = int(0.8 * len(wav_files))
    train_files = wav_files[:split_index]
    test_files = wav_files[split_index:]

    # 处理函数（复用 for train 和 test）
    def process_files(files, split_name):
        output_split_dir = OUTPUT_ROOT / split_name
        output_split_dir.mkdir(parents=True, exist_ok=True)
        copied_count = 0
        for wav_path in tqdm(files, desc=f"Processing {split_name} WAV Files"):
            try:
                # 过滤 speech：如果路径包含 "paired_speech" 或 "speech"，跳过
                if any('paired_speech' in p.lower() or 'speech' in p.lower() for p in wav_path.parts):
                    continue

                # 推断信息从路径
                lang = infer_lang_from_path(wav_path)
                singer = infer_singer_from_path(wav_path)
                technique = infer_technique_from_path(wav_path)

                if technique == "Unknown" or lang == "Unknown" or singer == "Unknown":
                    print(f"警告：无法推断完整信息 for {wav_path}，跳过。")
                    continue

                # 加载对应的 .json 如果存在
                json_path = wav_path.with_suffix('.json')
                item_name = wav_path.stem  # 默认用文件名如 "0000"
                if json_path.exists():
                    with open(json_path, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                    # 处理 JSON 可能是 list 的情况
                    if isinstance(json_data, list) and json_data:
                        json_data = json_data[0]  # 假设第一个元素是主要字典，如果是列表
                    # 现在尝试 get，如果是 dict
                    if isinstance(json_data, dict):
                        item_name = json_data.get('item_name', item_name)
                        singer = json_data.get('singer', singer)

                msr_class = "vocals"
                if msr_class not in MSR_INSTRUMENT_CLASSES:
                    continue

                # 创建目录: <split>/<lang_singer_technique>/Vocals/
                sanitized_singer = sanitize_filename(singer)
                track_id = f"{lang}_{sanitized_singer}_{technique}"
                instrument_class_dir = output_split_dir / track_id / msr_class.capitalize()
                instrument_class_dir.mkdir(parents=True, exist_ok=True)

                # 新文件名: <item_name>_<technique>_<lang>.flac
                new_filename = f"{sanitize_filename(item_name)}_{technique}_{lang}.flac"
                destination_path = instrument_class_dir / new_filename

                # 读取音频并转换为 48kHz stereo FLAC
                data, sr = sf.read(wav_path)
                if data.ndim == 1:  # mono to stereo
                    data = np.stack([data, data], axis=-1)
                if sr != 48000:
                    data = librosa.resample(data.T, orig_sr=sr, target_sr=48000).T
                    # 这里占位，假设匹配或外部处理
                    print(f"警告：采样率 {sr} != 48000 for {wav_path}，考虑重采样。")
                    pass

                sf.write(destination_path, data, 48000, format='FLAC')
                copied_count += 1

            except json.JSONDecodeError as e:
                print(f"\n警告：解析 JSON 文件时出错: {json_path} - {e}")
            except Exception as e:
                print(f"\n警告：处理文件 {wav_path} 时发生未知错误: {e}")
        return copied_count

    # 处理 train 和 test
    train_count = process_files(train_files, 'train')
    test_count = process_files(test_files, 'test')

    print(f"\n共复制并转换了 {train_count} 个 train 文件 和 {test_count} 个 test 文件。")
    print("\n--------------------")
    print("全部处理完成！")
    print(f"所有 vocals stem 文件已按新的目录结构保存到: {OUTPUT_ROOT}")
    print("--------------------")


if __name__ == "__main__":
    process_gtsinger_to_rawstems_format()