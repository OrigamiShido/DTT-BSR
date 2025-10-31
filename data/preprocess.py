# -*- coding: utf-8 -*-
"""
preprocess.py

用于将动态混合的数据集（如 RawStems）转换为静态数据集。
它会提前执行所有昂贵的混合、MSR退化 (MixtureAugmentation) 并保存结果。
这将极大加速训练时的I/O。

用法:
python preprocess.py \
    --root-directory /path/to/your/all_data \
    --target-stem Vocals \
    --output-directory /path/to/preprocessed_dataset \
    --num-samples 100000 \
    --sr 48000 \
    --clip-duration 4.0 \
    --max-workers 8 \
    --split train
"""
import argparse
import random
import logging
import numpy as np
import soundfile as sf
import time
from pathlib import Path
from tqdm import tqdm
import os
import concurrent.futures  # 多进程库

# --- 关键导入 ---
# 确保 dataset.py 和 augment.py 在你的 PYTHONPATH 中
# 或者将这两个文件与此脚本放在同一目录
try:
    # 尝试从 dataset.py 导入所有必要的辅助函数
    from dataset import (
        load_audio, 
        fix_length_to_duration, 
        mix_to_target_snr, 
        contains_audio_signal, 
        get_audio_duration,
        calculate_rms,
        DEFAULT_GAIN_RANGE,
        AUDIO_EXTENSIONS
    )
    # 尝试从 augment.py 导入 *仅* MixtureAugmentation
    from augment import MixtureAugmentation
except ImportError as e:
    print(f"Error: 无法导入 'dataset.py' 或 'augment.py'。")
    print(f"请确保这些文件与 preprocess.py 位于同一目录，或者在 PYTHONPATH 中。")
    print(f"原始错误: {e}")
    exit(1)

# --- 日志配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 复制 RawStems 中的索引逻辑，但修改为从folders列表索引 ---
def index_training_files_from_folders(folders: list[Path], target_stem: str) -> list[dict]:
    """
    精简版的 _index_training_files，用于遍历和索引指定的文件夹列表。
    """
    indexed_songs = []
    skipped_info = {"no_target": 0, "no_others": 0, "unreadable": 0}
    
    total_folders = len(folders)
    logger.info(f"Scanning {total_folders} song folders for training target '{target_stem}'...")

    for folder in tqdm(folders, desc="Indexing Original Files"):
        song_dict = {"target_stems": [], "others": []}
        
        target_folder = folder / target_stem
        if not target_folder.is_dir():
            skipped_info["no_target"] += 1
            continue

        # --- Index Target Stems ---
        valid_target_files = []
        for ext in AUDIO_EXTENSIONS:
            valid_target_files.extend(target_folder.rglob(f'*{ext}'))
        
        readable_target_files = []
        for p in valid_target_files:
            if p.is_file() and os.access(p, os.R_OK) and get_audio_duration(p) > 0.1:
                readable_target_files.append(p)
            else:
                skipped_info["unreadable"] += 1
        
        if not readable_target_files:
            skipped_info["no_target"] += 1
            continue
        song_dict["target_stems"] = readable_target_files

        # --- Index Other Stems ---
        valid_other_files = []
        for instrument_dir in folder.iterdir():
            if instrument_dir.is_dir() and instrument_dir.name != target_stem and os.access(instrument_dir, os.R_OK):
                for ext in AUDIO_EXTENSIONS:
                    valid_other_files.extend(instrument_dir.rglob(f'*{ext}'))
        
        readable_other_files = []
        for p in valid_other_files:
            if p.is_file() and os.access(p, os.R_OK) and get_audio_duration(p) > 0.1:
                readable_other_files.append(p)
            else:
                skipped_info["unreadable"] += 1

        if not readable_other_files:
            skipped_info["no_others"] += 1
            continue
            
        song_dict["others"] = readable_other_files
        indexed_songs.append(song_dict)

    logger.info(f"Training indexing complete: Found {len(indexed_songs)} songs usable for mixing target '{target_stem}'.")
    if any(skipped_info.values()):
        logger.info(f"Skipped info: {skipped_info}")
    return indexed_songs


# --- 复制 RawStems 中的核心生成逻辑 ---
def generate_sample(
    song_dict: dict, 
    sr: int, 
    clip_duration_sec: float, 
    clip_duration_samples: int,
    snr_range: tuple[float, float],
    mixture_augmentation: MixtureAugmentation
) -> dict | None:
    """
    尝试从一个 song_dict 生成一个样本。
    这是 RawStems.__getitem__ 训练逻辑的精简版。
    """
    
    # 内部采样循环 (尝试N次)
    for _attempt in range(50): # 尝试 50 次
        try:
            # --- 1. 选择文件 ---
            target_stems_list = song_dict.get("target_stems")
            others_list = song_dict.get("others")
            if not target_stems_list or not others_list:
                return None # 该 song_dict 无效

            num_targets = random.randint(1, min(len(target_stems_list), 3))
            selected_targets = random.sample(target_stems_list, num_targets)

            num_others = random.randint(1, min(len(others_list), 8))
            selected_others = random.sample(others_list, num_others)

            files_for_offset_check = selected_targets + selected_others

            # --- 2. 确定偏移量 (简化版：只使用随机偏移) ---
            min_valid_duration = float('inf')
            possible_to_sample = True
            for p in files_for_offset_check:
                duration = get_audio_duration(p)
                if duration < clip_duration_sec:
                    possible_to_sample = False
                    break
                min_valid_duration = min(min_valid_duration, duration)
            
            if not possible_to_sample or min_valid_duration == float('inf'):
                continue # 换下一组文件
            
            max_possible_offset = max(0.0, min_valid_duration - clip_duration_sec)
            offset = random.uniform(0, max_possible_offset) if max_possible_offset > 1e-6 else 0.0

            # --- 3. 加载和混合 ---
            loaded_targets = [load_audio(p, offset, clip_duration_sec, sr) for p in selected_targets]
            if any(np.all(audio == 0) for audio in loaded_targets):
                continue
            target_mix = np.sum(loaded_targets, axis=0) / float(num_targets)

            loaded_others = [load_audio(p, offset, clip_duration_sec, sr) for p in selected_others]
            if any(np.all(audio == 0) for audio in loaded_others):
                continue
            other_mix = np.sum(loaded_others, axis=0) / float(num_others)

            # --- 4. 检查有效性 (RMS) ---
            if not contains_audio_signal(target_mix) or \
               not contains_audio_signal(other_mix):
                continue

            # --- 5. 准备混合 ---
            # !! 关键: 我们只保存 *clean* target。StemAugmentation 将在训练时即时应用。
            target_clean = target_mix.copy()
            target_augmented = target_mix # 在这里不应用 StemAugmentation

            # --- 6. 混合前长度和类型修正 ---
            target_augmented = fix_length_to_duration(target_augmented.astype(np.float32), clip_duration_samples)
            other_mix = fix_length_to_duration(other_mix.astype(np.float32), clip_duration_samples)
            target_clean = fix_length_to_duration(target_clean.astype(np.float32), clip_duration_samples)

            # --- 7. 混合 ---
            target_snr = random.uniform(*snr_range)
            mixture, target_rescale_factor, _ = mix_to_target_snr(
                target_augmented, other_mix, target_snr
            )
            target_clean *= target_rescale_factor

            # --- 8. 应用 *昂贵* 的混合物增强 (MSR退化) ---
            # 这是此脚本的核心目的
            mixture_augmented = mixture_augmentation.apply(mixture)

            # --- !! 在这里添加修复 !! ---
            if mixture_augmented.ndim == 3 and mixture_augmented.shape[0] == 1:
                mixture_augmented = mixture_augmented.squeeze(0)
            # --- 修复结束 ---

            # --- 9. 最终处理 (归一化, 增益, 清理) ---
            target_processed = target_clean # 已缩放
            max_val_mixture = np.max(np.abs(mixture_augmented)) if mixture_augmented.size > 0 else 0.0
            norm_scale = 1.0
            if max_val_mixture > 1.0:
                norm_scale = 0.98 / max(max_val_mixture, 1e-8)
            
            mixture_final = mixture_augmented * norm_scale
            target_final = target_processed * norm_scale
            
            final_rescale = np.random.uniform(*DEFAULT_GAIN_RANGE)
            mixture_out = (mixture_final * final_rescale).astype(np.float32)
            target_out = (target_final * final_rescale).astype(np.float32)
            
            mixture_out = np.nan_to_num(mixture_out)
            target_out = np.nan_to_num(target_out)

            # --- 10. 最终长度和形状检查 ---
            mixture_out = fix_length_to_duration(mixture_out, clip_duration_samples)
            target_out = fix_length_to_duration(target_out, clip_duration_samples)
            expected_shape = (2, clip_duration_samples)
            if mixture_out.shape != expected_shape or target_out.shape != expected_shape:
                logger.warning(f"Sample shape mismatch after processing. Mix: {mixture_out.shape}, Target: {target_out.shape}. Skipping.")
                continue

            # --- 11. 成功 ---
            return { "mixture": mixture_out, "target": target_out }
        
        except Exception as e:
            # 在多进程中，减少日志噪音
            # logger.warning(f"Error during sample generation attempt: {e}", exc_info=False)
            continue # 尝试下一次

    # 如果50次尝试都失败了
    return None

# --- !! 新增：多进程辅助函数 !! ---

# 全局变量，用于子进程初始化
g_mixture_aug = None
g_song_list = []
g_config = {}

def init_worker(config: dict, song_list: list):
    """
    多进程工作器的初始化函数。
    在每个子进程启动时调用一次。
    """
    global g_mixture_aug, g_song_list, g_config
    
    # 1. 存储配置和歌曲列表
    g_config = config
    g_song_list = song_list
    
    # 2. 关键：每个进程创建自己的增强器实例
    # (这也会自动处理GPU，如果 augment.py 内部支持的话)
    try:
        # 注意：如果 augment.py 需要PyTorch并使用GPU，
        # 在这里初始化可以使每个进程访问GPU（如果驱动程序支持）。
        g_mixture_aug = MixtureAugmentation(sr=g_config['sr'])
        logger.info(f"Worker {os.getpid()} initialized MixtureAugmentation.")
    except Exception as e:
        logger.error(f"Worker {os.getpid()} failed to initialize Augmenter: {e}")
        g_mixture_aug = None

def process_one_sample(sample_index: int) -> dict | None:
    """
    多进程工作函数：生成并保存 *一个* 样本。
    它依赖于 'init_worker' 设置的全局变量。
    """
    global g_mixture_aug, g_song_list, g_config
    
    # 检查工作器是否成功初始化
    if g_mixture_aug is None or not g_song_list:
        logger.error(f"Worker {os.getpid()} not initialized. Skipping.")
        return None # 返回 None 表示失败
        
    # 从全局列表中随机选择一首歌
    # 为确保随机性，每个worker根据索引重新设置seed
    random.seed(os.getpid() * sample_index + int(time.time() * 1000))
    np.random.seed(os.getpid() * sample_index + int(time.time() * 1000))
    
    song_dict = random.choice(g_song_list)
    
    # 尝试从这首歌生成一个样本
    sample_data = generate_sample(
        song_dict,
        g_config['sr'],
        g_config['clip_duration'],
        g_config['clip_samples'],
        g_config['snr_range'],
        g_mixture_aug # 使用进程全局的增强器
    )
    
    if sample_data:
        # 构造文件名
        sample_name = f"sample_{sample_index:07d}"
        mix_path = g_config['output_dir'] / f"{sample_name}_mix.flac"
        target_path = g_config['output_dir'] / f"{sample_name}_target.flac"
        
        try:
            # 保存文件 (使用 .T 转置为 (frames, channels))
            sf.write(str(mix_path), sample_data["mixture"].T, g_config['sr'], subtype='PCM_24')
            sf.write(str(target_path), sample_data["target"].T, g_config['sr'], subtype='PCM_24')
            return {"name": sample_name, "status": "success"} # 返回成功
        
        except Exception as e:
            # 在多进程中不要打印过多日志，返回错误信息
            return {"name": sample_name, "status": "write_failed", "error": str(e)}
    
    return None # generate_sample 失败

# --- 主函数 (已修改为使用多进程，并支持split) ---
def main(args):
    root_dir = Path(args.root_directory)
    output_dir = Path(args.output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    target_stem = args.target_stem
    sr = args.sr
    clip_duration = args.clip_duration
    clip_samples = int(sr * clip_duration)
    total_samples_to_gen = args.num_samples
    snr_range = tuple(args.snr_range)
    split = args.split  # 新增：split参数

    # --- 新增：读取data_split TXT文件 ---
    # 假设CLASS_ABBREVIATIONS来自你的data_split脚本
    # 这里手动定义（从你的data_split.txt中复制）
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
    abbr = CLASS_ABBREVIATIONS.get(target_stem.lower(), target_stem[:3].capitalize())
    txt_path = Path("./data_splits") / f"{abbr}_{split}.txt"  # e.g., ./data_splits/Voc_train.txt
    
    if not txt_path.exists():
        logger.error(f"Split file not found: {txt_path}. Please run your data_split script first.")
        return

    folders = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            track_path = Path(line.strip())
            if track_path.is_dir() and os.access(track_path, os.R_OK):
                folders.append(track_path)
            else:
                logger.warning(f"Invalid or unreadable track path: {track_path}. Skipping.")

    if not folders:
        logger.error(f"No valid folders found in {txt_path}.")
        return

    logger.info(f"Loaded {len(folders)} folders from {txt_path} for split '{split}'.")

    # 1. 在主进程中 *测试* 初始化 (确保依赖存在)
    logger.info(f"Main process: Checking MixtureAugmentation (SR={sr})...")
    try:
        _ = MixtureAugmentation(sr=sr)
        logger.info("Main process: Check OK.")
    except Exception as e:
        logger.error(f"Failed to initialize MixtureAugmentation: {e}")
        logger.error("Please check if models (Encodec, DAC) are downloadable or dependencies are installed.")
        return

    # 2. 索引原始文件（使用指定的folders）
    logger.info(f"Indexing original data from {len(folders)} folders...")
    song_list = index_training_files_from_folders(folders, target_stem)
    if not song_list:
        logger.error("No songs found. Exiting.")
        return
    logger.info(f"Found {len(song_list)} usable songs.")

    # 3. 准备多进程配置
    max_workers = args.max_workers
    if max_workers <= 0:
        # 自动检测：使用 (CPU核心数 - 1)，至少为 1
        max_workers = max(1, os.cpu_count() - 1)
    logger.info(f"Using {max_workers} worker processes.")

    # 将配置打包以便传递给工作器
    worker_config = {
        'sr': sr,
        'clip_duration': clip_duration,
        'clip_samples': clip_samples,
        'snr_range': snr_range,
        'output_dir': output_dir
    }

    # 4. 开始生成 (使用多进程)
    logger.info(f"Starting to generate {total_samples_to_gen} samples in '{output_dir}' using {max_workers} workers...")
    start_time = time.time()
    generated_count = 0
    
    # 使用 tqdm 创建进度条
    pbar = tqdm(total=total_samples_to_gen, desc="Generating Samples")

    # 使用 ProcessPoolExecutor
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=init_worker,     # 每个worker启动时调用
        initargs=(worker_config, song_list) # 传递给 init_worker 的参数
    ) as executor:
        
        # .map 会自动将 (0, 1, 2, ... N-1) 传递给 process_one_sample
        # 它会阻塞直到所有结果返回，并保持顺序
        # chunksize 告诉 executor 一次给每个 worker 多少个任务
        results = executor.map(process_one_sample, range(total_samples_to_gen), chunksize=max(1, total_samples_to_gen // (max_workers * 10)))

        # 迭代 results 来驱动 .map 执行并收集统计
        for result in results:
            pbar.update(1) # 无论成功与否，都代表一次尝试
            if result and result['status'] == 'success':
                generated_count += 1
            elif result and result['status'] == 'write_failed':
                logger.warning(f"Failed to write sample: {result.get('error', 'Unknown error')}")
            # else:
                # result is None (生成失败, e.g., 50次尝试均失败)
                # pbar 已经更新，我们继续

    pbar.close()
    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / total_samples_to_gen if total_samples_to_gen > 0 else 0
    logger.info(f"--- Preprocessing Complete ---")
    logger.info(f"Successfully generated {generated_count} / {total_samples_to_gen} total attempts in {total_time:.2f} seconds.")
    logger.info(f"Average time per attempt: {avg_time:.3f} seconds.")
    logger.info(f"Data saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess audio dataset for faster training.")
    parser.add_argument("--root-directory", type=str, required=True, help="Path to the original root data (e.g., /path/to/all_data).")
    parser.add_argument("--target-stem", type=str, required=True, help="Name of the target stem folder (e.g., 'Vocals').")
    parser.add_argument("--output-directory", type=str, required=True, help="Path where the preprocessed .flac files will be saved.")
    parser.add_argument("--num-samples", type=int, default=100000, help="Total number of static samples to generate.")
    parser.add_argument("--sr", type=int, default=48000, help="Target sample rate.")
    parser.add_argument("--clip-duration", type=float, default=4.0, help="Duration of each clip in seconds.")
    parser.add_argument("--snr-range", type=float, nargs=2, default=[0.0, 10.0], help="SNR range for mixing (min max).")
    # --- 新增参数 ---
    parser.add_argument("--max-workers", type=int, default=0, help="Number of worker processes. 0 or negative means (CPU cores - 1).")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"], help="Dataset split to preprocess (train/test).")
    
    args = parser.parse_args()
    main(args)
