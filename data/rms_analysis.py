# -*- coding: utf-8 -*-
"""
一个用于预计算数据集中所有音频文件的 RMS（响度）分析的脚本。

此脚本会遍历 DATASET_ROOT 目录下的所有音频文件，
计算每秒的 RMS（以 dB 为单位），并将结果保存到
DATASET_ROOT 根目录下的一个 `rms_analysis.jsonl` 文件中。

`dataset.py` 中的 `RawStems` 类会使用这个文件来
进行“非静音采样”，确保训练时只选择有意义的（非静音）音频片段。
"""
import os
import json

import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import logging
import concurrent.futures
from typing import Optional, Dict
# --- 配置区域 ---

# 指向你合并后的数据集文件夹 (e.g., /home/student/nyr/data/dataset1/all_data)
# 这个目录应该包含所有的 TrackID 和 GTSinger 文件夹
DATASET_ROOT = Path("/home/student/nyr/data/slakh2100_rawstems_format_filtered/msr_dataset")

# 输出文件名（应与 dataset.py 中的名称匹配）
OUTPUT_FILE = "rms_analysis.jsonl"

# 分析时使用的采样率
ANALYSIS_SR = 48000
# RMS 窗口大小（1 秒）
HOP_LENGTH = ANALYSIS_SR * 1
FRAME_LENGTH = HOP_LENGTH

AUDIO_EXTENSIONS = ['.flac', '.mp3', '.wav']

# --- 结束配置区域 ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def process_file(audio_path: Path) -> Optional[Dict]:
    """
    加载一个音频文件，计算其每秒的 RMS (dB)，并返回一个字典。
    """
    try:
        # 加载为单声道，使用较低的采样率以加快速度
        audio, sr = librosa.load(audio_path, sr=ANALYSIS_SR, mono=True)

        # 计算 RMS
        rms = librosa.feature.rms(
            y=audio,
            frame_length=FRAME_LENGTH,
            hop_length=HOP_LENGTH
        )[0]

        # 转换为 dB
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)

        # 获取相对于 DATASET_ROOT 的路径
        relative_path = audio_path.relative_to(DATASET_ROOT)

        return {
            "filepath": str(relative_path.as_posix()),  # 使用 / 作为路径分隔符
            "rms_db_per_second": rms_db.tolist()
        }
    except Exception as e:
        logger.error(f"处理 {audio_path} 时出错: {e}")
        return None


def main():
    logger.info(f"开始 RMS 分析...")
    logger.info(f"数据集根目录: {DATASET_ROOT}")

    if not DATASET_ROOT.exists():
        logger.error(f"错误：找不到目录 {DATASET_ROOT}")
        return

    output_path = DATASET_ROOT / OUTPUT_FILE
    if output_path.exists():
        logger.warning(f"警告：{OUTPUT_FILE} 已存在。将覆盖此文件。")
        output_path.unlink()

    logger.info("正在扫描所有音频文件...")
    all_audio_files = [
        p for p in DATASET_ROOT.rglob('*')
        if p.suffix.lower() in AUDIO_EXTENSIONS
    ]

    if not all_audio_files:
        logger.error("在目录中未找到音频文件。")
        return

    logger.info(f"找到了 {len(all_audio_files)} 个音频文件。开始处理...")

    results = []
    # 使用 ThreadPoolExecutor 加速 I/O 密集型任务
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        # 使用 executor.map 来保持顺序并显示 tqdm 进度条
        future_to_path = {executor.submit(process_file, path): path for path in all_audio_files}

        for future in tqdm(concurrent.futures.as_completed(future_to_path), total=len(all_audio_files),
                           desc="Analyzing RMS"):
            result = future.result()
            if result:
                results.append(result)

    logger.info(f"分析完成。正在将 {len(results)} 条结果写入 {output_path}...")

    # 写入 .jsonl 文件
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item) + '\n')

    logger.info("--------------------")
    logger.info("全部完成！")
    logger.info(f"RMS 分析文件已保存: {output_path}")
    logger.info("现在您可以运行训练脚本了。")
    logger.info("--------------------")


if __name__ == "__main__":
    main()
