# -*- coding: utf-8 -*-
from pathlib import Path
import random
import logging
import numpy as np
import librosa
import soundfile as sf
import json
from typing import List, Optional, Dict, Union, Tuple, Any
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm
import os # Import os for os.access check
import time # 添加时间模块用于超时检查
import glob # For MSRBench indexing
from .augment import StemAugmentation, MixtureAugmentation


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# 降低 logger 级别以减少不必要的输出，除非调试
# logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = ['.flac', '.mp3', '.wav']
DEFAULT_GAIN_RANGE = (0.5, 1.0)
RMS_THRESHOLD_DB = -40.0
RMS_THRESHOLD_LINEAR = 10 ** (RMS_THRESHOLD_DB / 20)

# --- 辅助函数 ---
def calculate_rms(audio: np.ndarray) -> float:
    """计算音频 RMS，增加健壮性"""
    epsilon = 1e-10
    # 确保输入是 numpy array 且非空
    if not isinstance(audio, np.ndarray) or audio.size == 0:
        return 0.0
    # 计算前确保是 float 类型
    audio_f32 = audio.astype(np.float32)
    return np.sqrt(np.mean(audio_f32 ** 2) + epsilon)

def contains_audio_signal(audio: np.ndarray, rms_threshold: float = RMS_THRESHOLD_LINEAR) -> bool:
    """检查 RMS 是否高于阈值"""
    if not isinstance(audio, np.ndarray) or audio.size == 0: return False
    return calculate_rms(audio) > rms_threshold

def fix_length_to_duration(target: np.ndarray, duration_samples: int) -> np.ndarray:
    """修正音频长度，增加健壮性"""
    # 检查 duration_samples
    if duration_samples <= 0:
        num_channels = target.shape[0] if isinstance(target, np.ndarray) and target.ndim >= 2 else 2
        dtype = target.dtype if isinstance(target, np.ndarray) else np.float32
        return np.zeros((num_channels, 0), dtype=dtype)

    # 检查 target
    if not isinstance(target, np.ndarray) or target.ndim == 0:
        logger.warning(f"fix_length received invalid input type/shape: {type(target)}. Returning zeros.")
        num_channels = 2; dtype = np.float32
        return np.zeros((num_channels, duration_samples), dtype=dtype)

    target_length = target.shape[-1]

    if target_length == duration_samples: return target
    elif target_length < duration_samples:
        pad_width = [(0, 0)] * (target.ndim - 1) + [(0, duration_samples - target_length)]
        try:
            # 确保填充值为 0
            return np.pad(target, pad_width, mode='constant', constant_values=0)
        except ValueError as e: # 处理可能的填充错误
            logger.error(f"Error padding audio shape {target.shape} to {duration_samples}: {e}. Returning zeros.")
            num_channels = target.shape[0] if target.ndim >= 2 else 2
            return np.zeros((num_channels, duration_samples), dtype=target.dtype)
    else: # target_length > duration_samples
        try:
            slices = [slice(None)] * (target.ndim - 1) + [slice(0, duration_samples)]
            return target[tuple(slices)]
        except IndexError as e: # 处理可能的切片错误
            logger.error(f"Error slicing audio shape {target.shape} to {duration_samples}: {e}. Returning zeros.")
            num_channels = target.shape[0] if target.ndim >= 2 else 2
            return np.zeros((num_channels, duration_samples), dtype=target.dtype)


def get_audio_duration(file_path: Path) -> float:
    """获取音频时长"""
    try:
        # 确保是 Path 对象
        if not isinstance(file_path, Path): file_path = Path(file_path)
        # 检查文件是否存在且可读
        if not file_path.is_file() or not os.access(file_path, os.R_OK):
             # logger.warning(f"File not found or not readable: {file_path}")
             return 0.0
        return sf.info(str(file_path)).duration
    except Exception as e:
        # 记录文件名而不是完整路径
        logger.error(f"Error getting duration for '{file_path.name}': {type(e).__name__} - {e}")
        return 0.0

def load_audio(file_path: Path, offset: float, duration: float, sr: int) -> np.ndarray:
    """加载音频片段或完整文件，处理重采样和错误"""
    load_full_file = duration <= 0 # 负数或零 duration 表示加载完整文件
    file_path_str = str(file_path) # 用于日志和 soundfile

    # 预先计算目标样本数（如果不是加载完整文件）
    target_duration_samples = int(sr * duration) if not load_full_file else -1
    # 设定默认输出（至少1秒或估计长度）
    default_num_samples = target_duration_samples if target_duration_samples > 0 else sr # Fallback 至少 1 秒
    default_output = np.zeros((2, default_num_samples), dtype=np.float32)

    try:
        # 1. 检查文件是否存在和可读性
        if not file_path.is_file() or not os.access(file_path, os.R_OK):
            logger.warning(f"Audio file not found or not readable: {file_path_str}")
            return default_output

        # 2. 获取文件信息
        file_info = sf.info(file_path_str)
        file_sr = file_info.samplerate
        file_frames = file_info.frames
        file_channels = file_info.channels

        # 如果文件帧数异常少，直接返回
        if file_frames < 10:
             logger.warning(f"Audio file seems too short ({file_frames} frames): {file_path.name}. Returning silence.")
             return default_output


        # 3. 计算加载参数
        if load_full_file:
            start_frame = 0
            frames_to_read = -1 # 读取所有
            # 估计重采样后的样本数
            target_duration_samples = int(np.ceil(file_frames * (sr / float(file_sr)))) if file_sr > 0 else file_frames # 避免除零
            # 更新 fallback 长度
            if target_duration_samples <= 0: target_duration_samples = sr # 至少 1 秒
            default_output = np.zeros((2, target_duration_samples), dtype=np.float32)
        else:
            start_frame = int(offset * file_sr)
            # 检查偏移是否有效
            if start_frame < 0 or start_frame >= file_frames:
                # logger.warning(f"Offset out of bounds for {file_path.name}. Returning silence.")
                return default_output
            # 计算要读取的帧数，确保不超过文件末尾
            frames_to_read = min(int(duration * file_sr), file_frames - start_frame)
            if frames_to_read <= 0:
                # logger.warning(f"Calculated zero frames to read for {file_path.name}. Returning silence.")
                return default_output

        # 4. 读取音频
        audio, read_sr = sf.read(
            file_path_str,
            start=start_frame,
            frames=frames_to_read,
            dtype='float32',
            always_2d=True # 返回 (frames, channels)
        )
        audio = audio.T # 转置为 (channels, frames)

        # 验证读取的采样率
        if read_sr != file_sr:
            logger.warning(f"Sample rate mismatch for {file_path.name}: info={file_sr}, read={read_sr}. Using read SR ({read_sr}).")
            file_sr = read_sr

        # 5. 重采样 (如果需要)
        if file_sr != sr:
            # logger.debug(f"Resampling {file_path.name} from {file_sr} to {sr}...")
            try:
                # 使用 kaiser_fast 可能更快，但质量稍低
                resampled_audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr, res_type='kaiser_best')
                audio = resampled_audio
            except Exception as e:
                 logger.error(f"Error resampling {file_path.name}: {type(e).__name__} - {e}")
                 # 返回对应目标长度的静音
                 return np.zeros((2, target_duration_samples if target_duration_samples > 0 else sr), dtype=np.float32)

        # 6. 处理通道数 (确保立体声)
        # 检查 audio 是否有效
        if not isinstance(audio, np.ndarray) or audio.size == 0:
             logger.warning(f"Audio became empty after reading/resampling {file_path.name}. Returning silence.")
             return default_output

        if audio.shape[0] == 1:
            audio = np.vstack([audio, audio])
        elif audio.shape[0] > 2:
            # logger.debug(f"Using first 2 channels for {file_path.name} (had {audio.shape[0]})")
            audio = audio[:2, :]
        # Check channel dim exists and is 2
        elif audio.ndim < 2 or audio.shape[0] != 2:
             logger.error(f"Unexpected audio shape ({audio.shape}) after processing channels for {file_path.name}. Returning silence.")
             return default_output


        # 7. 长度修正 (对加载完整文件或片段都进行修正)
        # 在重采样后，即便是 load_full_file, 长度也需要根据 target_duration_samples 修正
        if target_duration_samples > 0: # 仅当目标长度已知时修正
             audio = fix_length_to_duration(audio, target_duration_samples)
        # 如果是 load_full_file 且 target_duration_samples <= 0 (计算失败)，则不修正

        # 8. 最终检查 NaN/Inf
        if not np.all(np.isfinite(audio)):
            logger.error(f"NaN or Inf detected in loaded audio for {file_path.name}. Returning silence.")
            # 返回对应目标长度的静音
            final_len = target_duration_samples if target_duration_samples > 0 else sr
            return np.zeros((2, final_len), dtype=np.float32)


        return audio.astype(np.float32) # 确保 float32

    # 捕获 SoundFileError 和其他异常
    except sf.SoundFileError as e:
        logger.error(f"SoundFileError loading {file_path_str}: {e}")
        return default_output
    except Exception as e:
        logger.error(f"Unexpected error in load_audio for '{file_path_str}' (offset={offset:.2f}, duration={duration:.2f}): {type(e).__name__} - {e}", exc_info=False)
        return default_output

def mix_to_target_snr(target: np.ndarray, noise: np.ndarray, target_snr_db: float) -> Tuple[np.ndarray, float, float]:
    """
    将 target 和 noise 混合到指定的 target_snr_db。
    增加健壮性检查。
    返回: (mixture, final_target_scale, final_noise_scale)
    """
    # --- *** 输入验证和修正 *** ---
    if not isinstance(target, np.ndarray) or not isinstance(noise, np.ndarray):
        logger.error("mix_to_target_snr received non-ndarray input.")
        # 返回安全的默认值
        return np.zeros_like(target) if isinstance(target, np.ndarray) else np.zeros((2,1)), 0.0, 0.0
    if target.shape != noise.shape:
        logger.error(f"Shape mismatch entering mix_to_target_snr! Target: {target.shape}, Noise: {noise.shape}. SNR: {target_snr_db:.2f}dB. Attempting to fix length.")
        # 尝试以 target 的长度为基准修复
        common_length = target.shape[-1]
        noise = fix_length_to_duration(noise, common_length)
        # 再次检查
        if target.shape != noise.shape:
            logger.error(f"Length fixing FAILED inside mix_to_target_snr! Target: {target.shape}, Noise (fixed): {noise.shape}. Returning scaled target.")
            max_amp = np.max(np.abs(target)) if target.size > 0 else 1.0
            scale = 0.5 / max(max_amp, 1e-6)
            return target * scale, scale, 0.0
    # --- 结束验证 ---
    # --- 计算功率和处理静音 ---
    target_power = np.mean(target ** 2)
    noise_power = np.mean(noise ** 2)
    # target_power 或 noise_power 可能为 0 或非常小
    if noise_power < 1e-12: # 噪声静音
        # logger.debug("Noise power is near zero. Returning target.")
        return target.copy(), 1.0, 0.0
    if target_power < 1e-12: # 目标静音
        # logger.debug("Target power is near zero. Returning scaled noise.")
        noise_scale = 1e-5 # 返回非常小的噪声
        max_noise = np.max(np.abs(noise)) if noise.size > 0 else 0.0
        if max_noise > 1e-6: noise_scale = min(noise_scale, 0.98 / max_noise)
        return noise * noise_scale, 0.0, noise_scale # final_target_scale = 0
    # --- 计算缩放因子 ---
    target_rms = np.sqrt(target_power)
    noise_rms = np.sqrt(noise_power)
    target_snr_amp_ratio = 10 ** (target_snr_db / 20.0)
    # 避免除零
    if noise_rms < 1e-10 or target_snr_amp_ratio < 1e-10:
        noise_scale = 1.0 # Fallback: 保持原始噪声音量
        # logger.warning(f"Near-zero noise_rms ({noise_rms:.2e}) or target_snr_amp_ratio ({target_snr_amp_ratio:.2e}). Using noise_scale=1.0.")
    else:
        noise_scale = target_rms / (noise_rms * target_snr_amp_ratio)
    # 限制缩放因子，防止极端值
    noise_scale = np.clip(noise_scale, 1e-6, 1e3) # 限制在一个较大但合理的范围
    if not np.isfinite(noise_scale):
        logger.warning(f"Calculated non-finite noise_scale ({noise_scale}) "
                       f"despite checks. Using noise_scale=1.0.")
        noise_scale = 1.0
    # --- 混合和归一化 ---
    scaled_noise = (noise * noise_scale).astype(target.dtype) # 确保类型一致
    mixture = target + scaled_noise
    max_amplitude = np.max(np.abs(mixture)) if mixture.size > 0 else 0.0
    mixture_norm_scale = 1.0
    if max_amplitude > 1.0:
        mixture_norm_scale = 0.98 / max_amplitude
        mixture *= mixture_norm_scale
    final_target_scale = mixture_norm_scale
    final_noise_scale = noise_scale * mixture_norm_scale
    # --- 最终检查和返回 ---
    # 再次检查 NaN/Inf (理论上不应发生)
    if not np.all(np.isfinite(mixture)):
         logger.error("NaN or Inf detected in final mixture inside mix_to_target_snr after mixing/scaling. Returning zeros.")
         return np.zeros_like(mixture), 0.0, 0.0
    return mixture, final_target_scale, final_noise_scale

class RawStems(Dataset):
    def __init__(
            self,
            target_stem: str, # 改回单个 target_stem
            root_directory: Union[str, Path],
            sr: int = 48000,
            clip_duration: float = 4.0, # 训练默认 4.0s
            snr_range: Tuple[float, float] = (0.0, 10.0),
            apply_augmentation: bool = True,
            rms_threshold_db: float = RMS_THRESHOLD_DB,
            is_validation: bool = False, # 验证模式标志
            # validation_stems: Optional[List[str]] = None, # 在验证模式下不再需要
            validation_dt_ids: Optional[List[int]] = None, # 验证时加载的 DT IDs
    ) -> None:
        self.root_directory = Path(root_directory).resolve() # 使用绝对路径
        self.sr = sr
        self.target_stem = target_stem
        self.is_validation = is_validation
        self.apply_augmentation = apply_augmentation if not is_validation else False

        # --- 根据模式设置 Clip Duration ---
        if self.is_validation:
            self.clip_duration = 10.0
            logger.info("Validation mode: clip_duration=10.0s, augmentation=False.")
            self.validation_dt_ids = validation_dt_ids if validation_dt_ids is not None else list(range(13))
            # MSRBench stem 文件夹/zip 名称列表 (需要根据实际情况调整)
            # 例如 "Vocals", "Guitars", "Bass" 等
            msrbench_stems_expected = ["Vocals", "Guitars", "Bass", "Keyboards", "Synthesizers", "Drums", "Percussion", "Orchestral Elements"]
            if self.target_stem not in msrbench_stems_expected:
                 logger.warning(f"Target stem '{self.target_stem}' may not match expected MSRBench names: {msrbench_stems_expected}")
        else: # Training mode
            self.clip_duration = clip_duration

        self.clip_duration_samples = int(self.clip_duration * self.sr)
        self.snr_range = snr_range
        self.rms_threshold_db = rms_threshold_db
        self.rms_threshold_linear = 10 ** (rms_threshold_db / 20)

        if self.clip_duration_samples <= 0:
            raise ValueError(f"Invalid clip_duration_samples ({self.clip_duration_samples}). Check sr ({sr}) and clip_duration ({self.clip_duration}).")

        # --- 日志 ---
        # ...(日志部分保持不变)...
        logger.info(f"Initializing RawStems Dataset:")
        logger.info(f"  Mode: {'Validation (MSRBench)' if is_validation else 'Training (Dynamic Mixing)'}")
        logger.info(f"  Root Dir: {self.root_directory}")
        logger.info(f"  Target Stem: {self.target_stem}")
        if is_validation: logger.info(f"  DT IDs: {self.validation_dt_ids}")
        logger.info(f"  Sample Rate: {self.sr}")
        logger.info(f"  Clip Duration: {self.clip_duration}s ({self.clip_duration_samples} samples)")
        if not is_validation:
            logger.info(f"  SNR Range: {self.snr_range} dB")
            logger.info(f"  Augmentation: {'Enabled' if self.apply_augmentation else 'Disabled'}")
            logger.info(f"  RMS Threshold: {self.rms_threshold_db} dB")


        # --- 索引文件 ---
        self.audio_files = self._index_audio_files()
        if not self.audio_files:
            mode_str = "validation" if is_validation else "training"
            raise ValueError(f"CRITICAL: No valid audio files found after indexing for {mode_str} mode. Root: {self.root_directory}, Target: {self.target_stem}")

        # --- 活动掩码 (仅训练模式需要) ---
        self.activity_masks = {}
        if not self.is_validation:
            # 尝试在 root_directory 上一级查找 rms_analysis.jsonl (如果 root 是 all_data)
            rms_file_path = self.root_directory / "rms_analysis.jsonl"
            if not rms_file_path.exists() and self.root_directory.name == "all_data":
                 parent_rms_path = self.root_directory.parent / "rms_analysis.jsonl"
                 if parent_rms_path.exists():
                     rms_file_path = parent_rms_path
                     logger.info(f"Using RMS file from parent directory: {rms_file_path}")

            self.activity_masks = self._compute_activity_masks(rms_file_path) # 传入路径
            if not self.activity_masks: logger.warning("Activity masks failed. Non-silent sampling disabled.")

        # --- 初始化增强器 (仅训练模式需要) ---
        self.stem_augmentation = None
        self.mixture_augmentation = None
        if self.apply_augmentation and not self.is_validation:
            try:
                # 确保类已导入
                stem_cls = globals().get('StemAugmentation')
                mix_cls = globals().get('MixtureAugmentation')
                if stem_cls and mix_cls:
                    self.stem_augmentation = stem_cls(sr=self.sr)
                    self.mixture_augmentation = mix_cls(sr=self.sr)
                else: raise ImportError("Augmentation classes not found in globals.")
            except Exception as e:
                logger.error(f"Error initializing augmentation: {e}. Augmentation disabled.", exc_info=True)
                self.apply_augmentation = False

    def _index_audio_files(self) -> List[Dict[str, Any]]:
        """根据 is_validation 标志索引文件"""
        if self.is_validation:
            # MSRBench: root_directory = /path/to/MSRBench_unzipped/Vocals/
            return self._index_msrbench_files()
        else:
            # Training: root_directory = /path/to/all_data/ (contains TrackID folders)
            return self._index_training_files()

    def _index_training_files(self) -> List[Dict[str, List[Path]]]:
        """索引训练文件 (root/TrackID/StemName/*.flac)"""
        indexed_songs = []
        skipped_info = {"no_target": 0, "no_others": 0, "unreadable": 0}
        processed_folders_count = 0

        all_song_dirs = []
        try:
             if not self.root_directory.is_dir(): raise FileNotFoundError(f"Training root missing: {self.root_directory}")
             # 只迭代目录项，增加检查
             all_song_dirs = [d for d in self.root_directory.iterdir() if d.is_dir()]
             # 检查是否可读，如果不可读则跳过
             readable_song_dirs = []
             for d in all_song_dirs:
                 try:
                     if os.access(d, os.R_OK): readable_song_dirs.append(d)
                     else: logger.warning(f"Skipping non-readable directory: {d.name}")
                 except OSError: logger.warning(f"Error checking access for {d.name}, skipping.")
             all_song_dirs = readable_song_dirs

        except OSError as e: logger.error(f"Error scanning training dir {self.root_directory}: {e}"); return []
        if not all_song_dirs: logger.error(f"No readable subdirectories (TrackID folders) in {self.root_directory}"); return []

        total_folders = len(all_song_dirs)
        logger.info(f"Scanning {total_folders} song folders for training target '{self.target_stem}'...")

        for folder in tqdm(all_song_dirs, desc="Indexing Training Files"):
            processed_folders_count += 1
            song_dict = {"target_stems": [], "others": []}
            is_usable = True

            target_folder = folder / self.target_stem
            if not target_folder.is_dir(): skipped_info["no_target"] += 1; continue

            # --- Index Target Stems ---
            valid_target_files = []
            try:
                # 优化: 直接 glob 特定后缀
                for ext in AUDIO_EXTENSIONS:
                    valid_target_files.extend(target_folder.rglob(f'*{ext}'))
                # Filter for readability and basic duration
                readable_target_files = []
                for p in valid_target_files:
                    if p.is_file() and os.access(p, os.R_OK):
                         if get_audio_duration(p) > 0.1: readable_target_files.append(p)
                         else: skipped_info["unreadable"] += 1 # Count short files as unreadable for mixing
                    else: skipped_info["unreadable"] += 1
                valid_target_files = readable_target_files
            except OSError as e: logger.error(f"Error reading target {target_folder}: {e}"); is_usable = False; continue

            if not valid_target_files: skipped_info["no_target"] += 1; continue
            song_dict["target_stems"] = valid_target_files

            # --- Index Other Stems ---
            valid_other_files = []
            try:
                for instrument_dir in folder.iterdir():
                    # Check is dir, not target, readable
                    if instrument_dir.is_dir() and instrument_dir.name != self.target_stem and os.access(instrument_dir, os.R_OK):
                        for ext in AUDIO_EXTENSIONS:
                             valid_other_files.extend(instrument_dir.rglob(f'*{ext}'))
                # Filter others
                readable_other_files = []
                for p in valid_other_files:
                    if p.is_file() and os.access(p, os.R_OK):
                         if get_audio_duration(p) > 0.1: readable_other_files.append(p)
                         else: skipped_info["unreadable"] += 1
                    else: skipped_info["unreadable"] += 1
                valid_other_files = readable_other_files
            except OSError as e: logger.error(f"Error reading other folders in {folder}: {e}"); is_usable = False

            if not valid_other_files: skipped_info["no_others"] += 1; is_usable = False

            # --- Add to list ---
            if is_usable:
                song_dict["others"] = valid_other_files
                indexed_songs.append(song_dict)

        # --- Final Logs ---
        logger.info(f"Processed {processed_folders_count}/{total_folders} folders.")
        logger.info(f"Training indexing complete: Found {len(indexed_songs)} songs usable for mixing target '{self.target_stem}'.")
        if skipped_info["no_target"] > 0: logger.info(f"  Skipped {skipped_info['no_target']} songs: missing target folder/files.")
        if skipped_info["no_others"] > 0: logger.info(f"  Skipped {skipped_info['no_others']} songs: missing other stem files.")
        if skipped_info["unreadable"] > 0: logger.info(f"  Skipped {skipped_info['unreadable']} individual unreadable/short files.")
        return indexed_songs

    def _index_msrbench_files(self) -> List[Dict[str, Path]]:
        """索引 MSRBench 文件 (mixture/*.flac 和 targets/*.flac)"""
        indexed_samples = []
        logger.info(f"Indexing MSRBench (validation mode) for target '{self.target_stem}' from {self.root_directory}...")
        mixture_dir = self.root_directory / "mixture"
        targets_dir = self.root_directory / "targets"

        if not mixture_dir.is_dir() or not targets_dir.is_dir():
            raise FileNotFoundError(f"MSRBench structure error in {self.root_directory}. Need 'mixture' and 'targets'.")

        # --- Index Targets ---
        target_files_map: Dict[str, Path] = {} # {song_id: path}
        skipped_targets = 0
        logger.info(f"Scanning target directory: {targets_dir}")
        try:
            for target_path in targets_dir.glob("*.flac"):
                 if target_path.is_file() and os.access(target_path, os.R_OK):
                      duration = get_audio_duration(target_path)
                      # Check duration close to 10s
                      if abs(duration - 10.0) < 0.1:
                           target_files_map[target_path.stem] = target_path
                      else: logger.warning(f"Skipping target {target_path.name} (duration {duration:.1f}s != 10s)."); skipped_targets += 1
                 else: skipped_targets += 1
        except OSError as e: logger.error(f"Error scanning {targets_dir}: {e}"); return []
        if skipped_targets > 0: logger.warning(f"Skipped {skipped_targets} invalid/unreadable target files.")
        if not target_files_map: logger.error(f"No valid 10s target files found in {targets_dir}. Cannot proceed with validation indexing."); return []
        logger.info(f"Found {len(target_files_map)} valid target files.")

        # --- Scan and Match Mixtures ---
        skipped_mixtures = 0
        logger.info(f"Scanning mixture directory: {mixture_dir}")
        try:
            # Use glob to find all potentially relevant mixture files first
            all_mixture_paths = list(mixture_dir.glob("*_DT*.flac"))
            logger.info(f"Found {len(all_mixture_paths)} potential mixture files. Matching with targets...")

            for mix_path in tqdm(all_mixture_paths, desc="Indexing MSRBench Mixtures"):
                 # Extract song_id and dt_id
                 parts = mix_path.stem.split('_DT')
                 if len(parts) == 2 and parts[1].isdigit():
                      song_id, dt_id_str = parts[0], parts[1]
                      dt_id = int(dt_id_str)

                      # Check if target exists and DT is required
                      if song_id in target_files_map and dt_id in self.validation_dt_ids:
                           if mix_path.is_file() and os.access(mix_path, os.R_OK):
                               duration = get_audio_duration(mix_path)
                               if abs(duration - 10.0) < 0.1:
                                    indexed_samples.append({
                                        "mixture_path": mix_path,
                                        "target_path": target_files_map[song_id]
                                    })
                               # else: logger.warning(f"Skipping mix {mix_path.name} (duration {duration:.1f}s)."); skipped_mixtures += 1 # Reduce log spam
                           else: skipped_mixtures += 1
                 # else: logger.debug(f"Skipping file with unexpected name format: {mix_path.name}") # Reduce log spam

        except OSError as e: logger.error(f"Error scanning {mixture_dir}: {e}"); return []
        if skipped_mixtures > 0: logger.warning(f"Skipped {skipped_mixtures} invalid/unreadable/wrong duration mixture files.")
        logger.info(f"MSRBench indexing complete. Found {len(indexed_samples)} valid mixture/target pairs for DT IDs {self.validation_dt_ids}.")
        return indexed_samples


    def _compute_activity_masks(self, rms_file_path: Path) -> Dict[str, np.ndarray]:
        """计算或加载每秒 RMS 活动掩码"""
        # rms_analysis_path = self.root_directory / "rms_analysis.jsonl" # Use passed path
        if not rms_file_path.is_file(): # Check if the provided path is valid
            logger.warning(f"RMS analysis file not found at '{rms_file_path}'. Non-silent sampling disabled.")
            return {}

        logger.info(f"Loading RMS data from {rms_file_path}...")
        rms_data = {}
        try:
            with open(rms_file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    try: data = json.loads(line); filepath_key = str(Path(data['filepath']).as_posix()); rms_values = np.array(data['rms_db_per_second']);
                    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e: logger.warning(f"Skipping invalid line {line_num+1} in rms: {e}"); continue
                    if np.any(~np.isfinite(rms_values)): logger.warning(f"Non-finite RMS for {filepath_key}. Skipping."); continue
                    rms_data[filepath_key] = rms_values
        except Exception as e: logger.error(f"Error reading RMS file {rms_file_path}: {e}"); return {}
        logger.info(f"Loaded RMS data for {len(rms_data)} files.")

        logger.info("Computing activity masks...")
        activity_masks = {}; window_size = int(np.ceil(self.clip_duration));
        if window_size <= 0: logger.error("Invalid window size for masks."); return {}

        all_indexed_files_posix = set()
        # Ensure audio_files is indexed correctly for training before computing masks
        if self.is_validation or not self.audio_files or "target_stems" not in self.audio_files[0]:
             logger.warning("Attempting to compute masks but audio_files index seems invalid or is for validation. Skipping mask computation.")
             return {}

        for song_idx, song in enumerate(self.audio_files): # Assumes training structure
             try:
                # Use .get() for safety
                all_indexed_files_posix.update(p.relative_to(self.root_directory).as_posix() for p in song.get("target_stems", []))
                all_indexed_files_posix.update(p.relative_to(self.root_directory).as_posix() for p in song.get("others", []))
             except (ValueError, TypeError) as e : logger.error(f"Error getting relative paths for song index {song_idx}: {e}"); continue

        not_found_count = 0; computed_count = 0
        for path_str_posix in tqdm(all_indexed_files_posix, desc="Computing Activity Masks"):
            activity_masks[path_str_posix] = np.array([], dtype=bool) # Default empty
            if path_str_posix in rms_data:
                rms_values = rms_data[path_str_posix]
                if isinstance(rms_values, np.ndarray) and rms_values.size > 0 and len(rms_values) >= window_size:
                    try:
                        is_loud = rms_values > self.rms_threshold_db; sum_loud = np.convolve(is_loud.astype(float), np.ones(window_size), 'valid'); avg_loud_enough = sum_loud / window_size >= 0.8
                        mask = np.zeros(len(rms_values), dtype=bool); len_avg = len(avg_loud_enough);
                        if len_avg > 0: mask[:len_avg] = avg_loud_enough
                        activity_masks[path_str_posix] = mask; computed_count += 1
                    except Exception as e_conv: logger.error(f"Error during convolution for {path_str_posix}: {e_conv}") # Log error but continue
            else: not_found_count += 1
        logger.info(f"Computed {computed_count} activity masks.");
        if not_found_count > 0: logger.warning(f"RMS data missing for {not_found_count} files.")
        return activity_masks

    def _find_common_valid_start_seconds(self, file_paths: List[Path]) -> List[int]:
        # ...(logic remains the same)...
        if not self.activity_masks: return []
        min_len = float('inf'); masks_to_intersect = []; has_missing_or_empty_mask = False
        for file_path in file_paths:
            path_str_posix = "";
            try: path_str_posix = file_path.relative_to(self.root_directory).as_posix()
            except ValueError: has_missing_or_empty_mask = True; break
            mask = self.activity_masks.get(path_str_posix)
            if mask is None or mask.size == 0: has_missing_or_empty_mask = True; break # Check size too
            masks_to_intersect.append(mask); min_len = min(min_len, len(mask))
        if has_missing_or_empty_mask or not masks_to_intersect or min_len == float('inf') or min_len <= 0: return []
        try:
            final_mask = np.ones(min_len, dtype=bool)
            for mask in masks_to_intersect:
                if len(mask) < min_len: return []
                final_mask &= mask[:min_len]
            valid_indices = np.where(final_mask)[0]; return valid_indices.tolist()
        except Exception as e: logger.error(f"Error computing mask intersection: {e}"); return []


    def __getitem__(self, index: int) -> Dict[str, Any]:
        """获取一个训练或验证样本"""
        start_time = time.time()
        max_time_per_sample = 15.0 # 超时 15 秒

        # --- **验证模式逻辑** ---
        if self.is_validation:
            if not self.audio_files or index >= len(self.audio_files):
                 logger.error(f"Validation index {index} out of bounds ({len(self.audio_files)}). Returning silence.")
                 return {"mixture": np.zeros((2, self.clip_duration_samples), dtype=np.float32),
                         "target": np.zeros((2, self.clip_duration_samples), dtype=np.float32)}

            sample_info = self.audio_files[index]
            mixture_path = sample_info.get("mixture_path")
            target_path = sample_info.get("target_path")

            if not mixture_path or not target_path or not mixture_path.is_file() or not target_path.is_file(): # Add file existence check
                 logger.error(f"Invalid sample info or files missing at index {index}: {sample_info}")
                 return {"mixture": np.zeros((2, self.clip_duration_samples), dtype=np.float32),
                         "target": np.zeros((2, self.clip_duration_samples), dtype=np.float32)}

            # 加载完整的 10 秒音频 (duration <= 0)
            mixture = load_audio(mixture_path, offset=0.0, duration=-1, sr=self.sr)
            target = load_audio(target_path, offset=0.0, duration=-1, sr=self.sr)

            # 确保长度为 10 秒
            mixture = fix_length_to_duration(mixture, self.clip_duration_samples)
            target = fix_length_to_duration(target, self.clip_duration_samples)

            # 检查加载是否成功
            if np.all(mixture == 0) or np.all(target == 0):
                 logger.warning(f"Validation sample loaded as silence or failed: mix={mixture_path.name}, target={target_path.name}")

            return {"mixture": mixture.astype(np.float32), "target": target.astype(np.float32)}

        # --- **训练模式逻辑** ---
        else:
            if not self.audio_files:
                 logger.error("Training dataset index is empty.")
                 return {"mixture": np.zeros((2, self.clip_duration_samples), dtype=np.float32), "target": np.zeros((2, self.clip_duration_samples), dtype=np.float32)}

            # 使用取模确保 index 有效
            actual_index = index % len(self.audio_files)

            # 主重试循环
            for _main_attempt in range(3):
                # 获取当前 song_dict
                # 增加检查，防止 actual_index 意外失效
                if actual_index >= len(self.audio_files):
                    logger.error(f"Index {actual_index} became invalid during retry loop. Resetting index.")
                    actual_index = 0 # Or handle differently
                song_dict = self.audio_files[actual_index]

                # 内部采样循环
                for attempt in range(75):
                    # 检查超时
                    if time.time() - start_time > max_time_per_sample:
                        logger.warning(f"Timeout generating sample (index {actual_index}, main attempt {_main_attempt+1}). Returning silence.")
                        return {"mixture": np.zeros((2, self.clip_duration_samples), dtype=np.float32), "target": np.zeros((2, self.clip_duration_samples), dtype=np.float32)}

                    # --- 1. 选择文件 ---
                    target_stems_list = song_dict.get("target_stems")
                    others_list = song_dict.get("others")
                    if not target_stems_list or not others_list: break # 跳出内部循环

                    num_targets = random.randint(1, min(len(target_stems_list), 3))
                    try: selected_targets = random.sample(target_stems_list, num_targets)
                    except ValueError: continue

                    num_others = random.randint(1, min(len(others_list), 8))
                    try: selected_others = random.sample(others_list, num_others)
                    except ValueError: continue

                    files_for_offset_check = selected_targets + selected_others

                    # --- 2. 确定偏移量 ---
                    valid_starts = self._find_common_valid_start_seconds(files_for_offset_check)
                    offset = -1.0; min_valid_duration = float('inf'); possible_to_sample = True

                    if valid_starts: # Use non-silent offset
                        start_second = random.choice(valid_starts)
                        max_random_offset = max(0.0, 1.0 - (self.clip_duration % 1.0 or 1.0))
                        offset = start_second + random.uniform(0, max_random_offset); offset = max(0.0, offset)
                    else: # Fallback: Random offset
                        for p in files_for_offset_check:
                            duration = get_audio_duration(p)
                            if duration < self.clip_duration: possible_to_sample = False; break
                            min_valid_duration = min(min_valid_duration, duration)
                        if not possible_to_sample or min_valid_duration == float('inf'): continue
                        max_possible_offset = max(0.0, min_valid_duration - self.clip_duration)
                        offset = random.uniform(0, max_possible_offset) if max_possible_offset > 1e-6 else 0.0
                        offset = max(0.0, offset)

                    # --- 3. 加载和混合 ---
                    loaded_targets = [load_audio(p, offset, self.clip_duration, self.sr) for p in selected_targets]
                    if any(np.all(audio == 0) for audio in loaded_targets): continue
                    target_mix = np.sum(loaded_targets, axis=0) / float(num_targets)

                    loaded_others = [load_audio(p, offset, self.clip_duration, self.sr) for p in selected_others]
                    if any(np.all(audio == 0) for audio in loaded_others): continue
                    other_mix = np.sum(loaded_others, axis=0) / float(num_others)

                    # --- 4. 检查有效性 (RMS) ---
                    if not contains_audio_signal(target_mix, self.rms_threshold_linear) or \
                       not contains_audio_signal(other_mix, self.rms_threshold_linear): continue

                    # --- 5. 准备混合 ---
                    target_clean = target_mix.copy()
                    target_augmented = self.stem_augmentation.apply(target_mix) if self.apply_augmentation else target_mix

                    # --- 6. 混合前长度和类型修正 ---
                    target_augmented = fix_length_to_duration(target_augmented.astype(np.float32), self.clip_duration_samples)
                    other_mix = fix_length_to_duration(other_mix.astype(np.float32), self.clip_duration_samples)
                    target_clean = fix_length_to_duration(target_clean.astype(np.float32), self.clip_duration_samples)

                    # --- 7. 混合 ---
                    target_snr = random.uniform(*self.snr_range)
                    mixture, target_rescale_factor, noise_rescale_factor = mix_to_target_snr(
                        target_augmented, other_mix, target_snr
                    )
                    target_clean *= target_rescale_factor

                    # --- 8. 应用混合物增强 (MSR退化) ---
                    mixture_augmented = self.mixture_augmentation.apply(mixture) if self.apply_augmentation else mixture

                    # --- 9. 最终处理 (归一化, 增益, 清理) ---
                    target_processed = target_clean # 已缩放
                    max_val_mixture = np.max(np.abs(mixture_augmented)) if mixture_augmented.size > 0 else 0.0
                    norm_scale = 1.0
                    if max_val_mixture > 1.0: norm_scale = 0.98 / max(max_val_mixture, 1e-8)
                    mixture_final = mixture_augmented * norm_scale
                    target_final = target_processed * norm_scale
                    final_rescale = np.random.uniform(*DEFAULT_GAIN_RANGE)
                    mixture_out = (mixture_final * final_rescale).astype(np.float32)
                    target_out = (target_final * final_rescale).astype(np.float32)
                    mixture_out = np.nan_to_num(mixture_out); target_out = np.nan_to_num(target_out)

                    # --- 10. 最终长度和形状检查 ---
                    mixture_out = fix_length_to_duration(mixture_out, self.clip_duration_samples)
                    target_out = fix_length_to_duration(target_out, self.clip_duration_samples)
                    expected_shape = (2, self.clip_duration_samples)
                    if mixture_out.shape != expected_shape or target_out.shape != expected_shape: continue

                    # --- 11. 成功 ---
                    return { "mixture": mixture_out, "target": target_out }
                # -- 内部循环结束 --

                # 如果内部失败，尝试下一个 index
                logger.warning(f"Inner loop failed 75 times for index {actual_index} (main attempt {_main_attempt+1}/3). Trying next.")
                actual_index = (actual_index + 1) % len(self.audio_files) # 移到下一个
                # song_dict 不需要更新，外部循环会重新获取

            # -- 外部循环结束 --

            # 如果外部循环也失败
            logger.error(f"CRITICAL: Failed after 3 main attempts. Returning silence.")
            return {"mixture": np.zeros((2, self.clip_duration_samples), dtype=np.float32), "target": np.zeros((2, self.clip_duration_samples), dtype=np.float32)}


    def __len__(self) -> int:
        return len(self.audio_files) if self.audio_files else 0


class InfiniteSampler(Sampler):
    def __init__(self, dataset: Dataset, seed: Optional[int] = None) -> None: # 种子可选
        dataset_len = len(dataset)
        if dataset_len == 0: raise ValueError("Dataset empty.")
        self.dataset_size = dataset_len
        self.indexes = list(range(self.dataset_size))
        # 使用 numpy RNG
        self.rng = np.random.default_rng(seed)
        self.rng.shuffle(self.indexes)
        self.pointer = 0
        logger.info(f"InfiniteSampler: size={self.dataset_size}, seed={'Random' if seed is None else seed}.")

    def reset(self) -> None:
        self.rng.shuffle(self.indexes)
        self.pointer = 0

    def __iter__(self):
        while True:
            if self.pointer >= self.dataset_size: self.reset()
            # 健壮性检查
            current_pointer = self.pointer
            if 0 <= current_pointer < len(self.indexes):
                 idx = self.indexes[current_pointer]
                 self.pointer += 1
                 if 0 <= idx < self.dataset_size: yield idx
                 else: logger.error(f"Sampler yielded invalid index {idx}. Resetting."); self.reset()
            else: logger.error(f"Sampler pointer {current_pointer} out of bounds. Resetting."); self.reset()


    def __len__(self) -> int:
        return self.dataset_size

