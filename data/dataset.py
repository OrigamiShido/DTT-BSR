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
# 导入路径已从 'data.augment' 修正为 'augment'
from augment import StemAugmentation, MixtureAugmentation

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = ['.flac', '.mp3', '.wav']
DEFAULT_GAIN_RANGE = (0.5, 1.0)

def calculate_rms(audio: np.ndarray) -> float:
    """计算音频 RMS，增加 epsilon 以防止 log(0)"""
    epsilon = 1e-10
    return np.sqrt(np.mean(audio**2) + epsilon)

def contains_audio_signal(audio: np.ndarray, rms_threshold_db: float = -40.0) -> bool:
    """检查 RMS 是否高于以 dB 为单位的阈值"""
    if audio is None or audio.size == 0:
        return False
    rms_linear = calculate_rms(audio)
    rms_db = 20 * np.log10(rms_linear)
    return rms_db > rms_threshold_db

def fix_length(target: np.ndarray, source: np.ndarray) -> np.ndarray:
    """(已弃用，但保留) 修正目标长度以匹配源长度"""
    target_length, source_length = target.shape[-1], source.shape[-1]
    if target_length < source_length:
        # 修正：为 (C, N) 形状的立体声添加正确的 padding 维度
        return np.pad(target, ((0, 0), (0, source_length - target_length)), mode='constant')
    if target_length > source_length:
        return target[:, :source_length]
    return target

def fix_length_to_duration(target: np.ndarray, duration_samples: int) -> np.ndarray:
    """修正音频长度以匹配样本数，增加健壮性"""
    if not isinstance(target, np.ndarray) or target.ndim < 2:
        logger.warning(f"fix_length_to_duration 收到无效输入，返回静音。Shape: {target.shape if isinstance(target, np.ndarray) else type(target)}")
        num_channels = target.shape[0] if isinstance(target, np.ndarray) and target.ndim >= 2 else 2
        return np.zeros((num_channels, duration_samples), dtype=np.float32)
        
    target_length = target.shape[-1]
    required_length = duration_samples

    if target_length < required_length:
        # 修正：为 (C, N) 形状的立体声添加正确的 padding 维度
        pad_width = ((0, 0), (0, required_length - target_length))
        return np.pad(target, pad_width, mode='constant')
    if target_length > required_length:
        return target[:, :required_length]
    return target

def get_audio_duration(file_path: Path) -> float:
    # 修正： '尝试：' -> 'try:'
    try:
        return sf.info(str(file_path)).duration
    except Exception as e:
        logger.error(f"Error getting duration for {file_path}: {e}")
        return 0.0

def load_audio(file_path: Path, offset: float, duration: float, sr: int) -> np.ndarray:
    """加载音频，处理 librosa 错误和确保立体声"""
    required_samples = int(sr * duration)
    default_output = np.zeros((2, required_samples), dtype=np.float32)
    try:
        # 修正：使用 soundfile 加载，它对 .flac 支持更好且更快
        file_path_str = str(file_path)
        file_info = sf.info(file_path_str)
        file_sr = file_info.samplerate
        file_frames = file_info.frames

        start_frame = int(offset * file_sr)
        frames_to_read = int(duration * file_sr)

        # 检查边界
        if start_frame < 0 or start_frame >= file_frames:
            return default_output # 偏移超出范围
        
        frames_to_read = min(frames_to_read, file_frames - start_frame)
        if frames_to_read <= 0:
            return default_output

        audio, read_sr = sf.read(
            file_path_str,
            start=start_frame,
            frames=frames_to_read,
            dtype='float32',
            always_2d=True # 返回 (frames, channels)
        )
        audio = audio.T # 转置为 (channels, frames)

        # 重采样（如果需要）
        if read_sr != sr:
            audio = librosa.resample(audio, orig_sr=read_sr, target_sr=sr, res_type='kaiser_best')

        # 确保立体声
        if audio.shape[0] == 1:
            audio = np.vstack([audio, audio])
        elif audio.shape[0] > 2:
            audio = audio[:2, :] # 取前两个通道

        # 确保长度
        audio = fix_length_to_duration(audio, required_samples)
        return audio

    except Exception as e:
        logger.error(f"Error loading {file_path} at offset {offset}: {e}")
        return default_output # 返回标准长度的静音

def mix_to_target_snr(target: np.ndarray, noise: np.ndarray, target_snr_db: float) -> Tuple[np.ndarray, float, float]:
    """混合并进行归一化，增加健壮性"""
    epsilon = 1e-10
    target_power = np.mean(target ** 2) + epsilon
    noise_power = np.mean(noise ** 2) + epsilon

    if noise_power < 1e-12: # 噪声几乎静音
        return target.copy(), 1.0, 0.0
    if target_power < 1e-12: # 目标几乎静音
        # 返回一个非常安静的噪声
        max_noise = np.max(np.abs(noise)) + epsilon
        scale = 0.01 / max_noise
        return noise * scale, 0.0, scale

    target_snr_linear = 10 ** (target_snr_db / 10.0) # 修正：SNR 是 10*log10(P)
    noise_scale = np.sqrt(target_power / (noise_power * target_snr_linear))
    
    # 限制缩放因子
    noise_scale = min(noise_scale, 100.0) # 防止噪声过大

    scaled_noise = noise * noise_scale
    mixture = target + scaled_noise
    
    max_amplitude = np.max(np.abs(mixture)) + epsilon
    target_scale = 1.0
    
    if max_amplitude > 1.0:
        norm_factor = 0.98 / max_amplitude # 使用 0.98 防止削波
        mixture *= norm_factor
        target_scale = norm_factor
        noise_scale *= norm_factor
    
    return mixture, target_scale, noise_scale

class RawStems(Dataset):
    def __init__(
        self,
        # --- !! 修改后的参数 !! ---
        root_directory: Union[str, Path],  # 仍然需要，用于相对路径
        split_file_path: Union[str, Path], # 新增：指向 Voc_train.txt 等文件
        target_stem: str,                  # 仍然需要，用于在轨道内查找
        # --- ------------------ ---
        sr: int = 48000,
        clip_duration: float = 3.0,
        snr_range: Tuple[float, float] = (0.0, 10.0),
        apply_augmentation: bool = True,
        rms_threshold_db: float = -40.0, # 修正：使用 dB
    ) -> None:
        self.root_directory = Path(root_directory).resolve()
        self.split_file_path = Path(split_file_path)
        self.sr = sr
        self.clip_duration = clip_duration
        self.snr_range = snr_range
        self.apply_augmentation = apply_augmentation
        self.rms_threshold_db = rms_threshold_db # 修正：使用 dB
        
        # 修正： target_stem 现在应该是一个简单的名字，如 "Vocals"
        # self.target_stem_1 = target_stem_parts[0].strip()
        # self.target_stem_2 = target_stem_parts[1].strip() if len(target_stem_parts) > 1 else None
        self.target_stem = target_stem # e.g., "Vocals", "Guitars"
        
        logger.info(f"Loading split file: '{self.split_file_path}'")
        self.folders = []
        
        # --- !! 新的索引逻辑 !! ---
        if not self.split_file_path.is_file():
            raise FileNotFoundError(f"Split file not found: {self.split_file_path}")
            
        with open(self.split_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                abs_track_path_str = line.strip()
                if abs_track_path_str:
                    track_path = Path(abs_track_path_str)
                    if track_path.is_dir():
                        # 检查此轨道是否真的包含目标 stem
                        target_path = track_path / self.target_stem
                        if target_path.exists() and target_path.is_dir():
                            self.folders.append(track_path)
                        # else:
                        #     logger.warning(f"Track {track_path.name} from split file missing stem '{self.target_stem}', skipping.")
                    # else:
                    #     logger.warning(f"Track path not found: {track_path}, skipping.")
        
        if not self.folders:
            raise ValueError(f"No valid tracks found from split file '{self.split_file_path}' matching stem '{self.target_stem}'.")
        
        logger.info(f"Loaded {len(self.folders)} song folders from split file.")

        self.audio_files = self._index_audio_files()
        if not self.audio_files: raise ValueError("No audio files found after indexing.")
            
        self.activity_masks = self._compute_activity_masks()
        
        self.stem_augmentation = StemAugmentation(sr=self.sr) # 修正：传入 sr
        self.mixture_augmentation = MixtureAugmentation(sr=self.sr) # 修正：传入 sr

    def _compute_activity_masks(self) -> Dict[str, np.ndarray]:
        # root_directory 现在必须是 '.../slakh2100_rawstems_format_filtered'
        # rms_analysis.jsonl 应该在它旁边
        rms_analysis_path = self.root_directory.parent / "rms_analysis.jsonl"
        if not rms_analysis_path.exists():
             # 备用：检查它是否在 root_directory 内部
             rms_analysis_path = self.root_directory / "rms_analysis.jsonl"
             if not rms_analysis_path.exists():
                logger.warning(f"rms_analysis.jsonl not found at {rms_analysis_path.parent} or {self.root_directory}. Non-silent selection will be disabled.")
                return {}
        
        logger.info(f"Loading and processing RMS data from {rms_analysis_path}")
        rms_data = {}
        with open(rms_analysis_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # 键应该是相对路径，例如 "train/Track00001/Guitars/S01_ElectricGuitar.flac"
                    rms_data[data['filepath']] = np.array(data['rms_db_per_second'])
                except (json.JSONDecodeError, KeyError):
                    continue

        logger.info("Computing activity masks for all indexed files...")
        activity_masks = {}
        window_size = int(np.ceil(self.clip_duration))

        all_indexed_files = set()
        for song in self.audio_files:
            # 修正：确保 self.root_directory 是正确的根
            all_indexed_files.update(p.relative_to(self.root_directory) for p in song["target_stems"])
            all_indexed_files.update(p.relative_to(self.root_directory) for p in song["others"])

        for relative_path in tqdm(all_indexed_files, desc="Computing Activity Masks"):
            path_str = str(relative_path.as_posix()) # 使用 posix 格式匹配 jsonl
            if path_str in rms_data:
                rms_values = rms_data[path_str]
                if len(rms_values) < window_size:
                    activity_masks[path_str] = np.array([False] * len(rms_values))
                    continue
                
                is_loud = rms_values > self.rms_threshold_db # 修正：使用 dB
                sum_loud = np.convolve(is_loud.astype(float), np.ones(window_size), 'valid')
                avg_loud_enough = sum_loud / window_size > 0.8 # 80% 的窗口是响亮的
                
                mask = np.zeros(len(rms_values), dtype=bool)
                if len(avg_loud_enough) > 0:
                    mask[:len(avg_loud_enough)] = avg_loud_enough
                activity_masks[path_str] = mask
            # else:
            #    logger.warning(f"RMS data not found for {path_str}")
        logger.info(f"Computed {len(activity_masks)} masks.")
        return activity_masks

    def _find_common_valid_start_seconds(self, file_paths: List[Path]) -> List[int]:
        if not self.activity_masks: return []

        min_len = float('inf')
        masks_to_intersect = []
        for file_path in file_paths:
            try:
                path_str = str(file_path.relative_to(self.root_directory).as_posix())
            except ValueError:
                logger.warning(f"File {file_path} not relative to {self.root_directory}. Skipping mask.")
                return [] # 路径错误，无法使用 mask

            mask = self.activity_masks.get(path_str)
            if mask is None: 
                # logger.warning(f"Mask not found for {path_str}. Disabling non-silent sampling for this item.")
                return [] # 一个文件没有 mask，禁用此功能
            
            masks_to_intersect.append(mask)
            min_len = min(min_len, len(mask))
        
        if not masks_to_intersect or min_len == float('inf') or min_len == 0: 
            return []

        try:
            final_mask = np.ones(min_len, dtype=bool)
            for mask in masks_to_intersect:
                final_mask &= mask[:min_len]
            
            return np.where(final_mask)[0].tolist()
        except Exception as e:
            logger.error(f"Error intersecting masks: {e}")
            return []

    def _index_audio_files(self) -> List[Dict[str, List[Path]]]:
        indexed_songs = []
        # 修正：self.folders 已经由 __init__ 从 split file 填充
        for folder in tqdm(self.folders, desc="Indexing audio files"):
            song_dict = {"target_stems": [], "others": []}
            
            # 修正：使用 self.target_stem
            target_folder = folder / self.target_stem
            
            if target_folder.exists() and target_folder.is_dir():
                song_dict["target_stems"].extend(p for p in target_folder.rglob('*') if p.suffix.lower() in AUDIO_EXTENSIONS and p.is_file())
            
            # 修正：索引 "others"
            for p in folder.rglob('*'):
                if p.suffix.lower() in AUDIO_EXTENSIONS and p.is_file():
                    try:
                        relative_path = p.relative_to(folder)
                        # 检查它是否 *不* 在 target_folder 下
                        if not relative_path.parts[0] == self.target_stem:
                            song_dict["others"].append(p)
                    except ValueError:
                        continue
            
            if song_dict["target_stems"] and song_dict["others"]:
                indexed_songs.append(song_dict)
            # else:
            #     logger.warning(f"Skipping {folder.name}: missing targets ({len(song_dict['target_stems'])}) or others ({len(song_dict['others'])})")
        return indexed_songs
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        # 修正：将 clip_duration 转换为样本
        target_samples = int(self.clip_duration * self.sr)
        
        for _ in range(100): # 100次尝试
            try:
                song_dict = self.audio_files[index]
                
                num_targets = random.randint(1, min(len(song_dict["target_stems"]), 5))
                selected_targets = random.sample(song_dict["target_stems"], num_targets)
                
                num_others = random.randint(1, min(len(song_dict["others"]), 10))
                selected_others = random.sample(song_dict["others"], num_others)

                valid_starts = self._find_common_valid_start_seconds(selected_targets + selected_others)
                offset = 0.0

                if valid_starts:
                    start_second = random.choice(valid_starts)
                    # 确保在 1s 内随机抖动
                    max_jitter = max(0.0, 1.0 - (self.clip_duration % 1.0 or 1.0))
                    offset = start_second + random.uniform(0, max_jitter)
                else:
                    # Fallback：完全随机偏移
                    min_dur = float('inf')
                    for p in selected_targets + selected_others:
                        min_dur = min(min_dur, get_audio_duration(p))
                    if min_dur < self.clip_duration:
                        continue # 这个组合无效
                    max_offset = max(0.0, min_dur - self.clip_duration)
                    offset = random.uniform(0, max_offset)

                # 修正：使用 np.sum 和 axis=0 高效混合
                loaded_targets = [load_audio(p, offset, self.clip_duration, self.sr) for p in selected_targets]
                target_mix = np.sum(np.array(loaded_targets), axis=0) / num_targets
                
                loaded_others = [load_audio(p, offset, self.clip_duration, self.sr) for p in selected_others]
                other_mix = np.sum(np.array(loaded_others), axis=0) / num_others

                if not contains_audio_signal(target_mix, self.rms_threshold_db) or \
                   not contains_audio_signal(other_mix, self.rms_threshold_db):
                    continue # 片段太安静

                target_clean = target_mix.copy()
                # 修正： .apply() 不接受 sr
                target_augmented = self.stem_augmentation.apply(target_mix) if self.apply_augmentation else target_mix
                
                mixture, target_scale, _ = mix_to_target_snr(
                    target_augmented, other_mix, random.uniform(*self.snr_range)
                )
                target_clean *= target_scale
                
                # 修正： .apply() 不接受 sr
                mixture_augmented = self.mixture_augmentation.apply(mixture) if self.apply_augmentation else mixture

                # 修正：归一化逻辑
                max_val = np.max(np.abs(mixture_augmented)) + 1e-8
                if max_val > 1.0: # 只在需要时归一化
                    norm_scale = 0.98 / max_val
                    mixture_final = mixture_augmented * norm_scale
                    target_final = target_clean * norm_scale
                else:
                    mixture_final = mixture_augmented
                    target_final = target_clean
                
                rescale = np.random.uniform(*DEFAULT_GAIN_RANGE)
                mixture_out = (mixture_final * rescale).astype(np.float32)
                target_out = (target_final * rescale).astype(np.float32)
                
                # 最终清理和长度修正
                mixture = fix_length_to_duration(np.nan_to_num(mixture_out), target_samples)
                target = fix_length_to_duration(np.nan_to_num(target_out), target_samples)
                
                return {
                    "mixture": mixture,
                    "target": target
                }
            except Exception as e:
                logger.warning(f"Error in __getitem__ loop (index {index}): {e}", exc_info=False)
                continue # 捕获异常并重试

        # 100次尝试失败后
        logger.error(f"Failed to get valid sample for index {index} after 100 attempts. Returning silence.")
        return {
            "mixture": np.zeros((2, target_samples), dtype=np.float32),
            "target": np.zeros((2, target_samples), dtype=np.float32)
        }

    def __len__(self) -> int:
        return len(self.audio_files)


class InfiniteSampler(Sampler):
    def __init__(self, dataset: Dataset, seed: Optional[int] = None) -> None: # 修正：添加 seed
        self.dataset_size = len(dataset)
        if self.dataset_size == 0:
            raise ValueError("Dataset is empty.")
        self.indexes = list(range(self.dataset_size))
        # 修正：使用 numpy RNG 以便设置种子
        self.rng = np.random.default_rng(seed)
        self.reset()
    
    def reset(self) -> None:
        self.rng.shuffle(self.indexes)
        self.pointer = 0
        
    def __iter__(self):
        while True:
            if self.pointer >= self.dataset_size: self.reset()
            yield self.indexes[self.pointer]
            self.pointer += 1

    def __len__(self) -> int:
        return self.dataset_size # 修正：返回数据集大小
