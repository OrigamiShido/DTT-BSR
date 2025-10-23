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
from data.augment import StemAugmentation, MixtureAugmentation

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = ['.flac', '.mp3', '.wav']
DEFAULT_GAIN_RANGE = (0.5, 1.0)

def calculate_rms(audio: np.ndarray) -> float:
    return np.sqrt(np.mean(audio**2))

def contains_audio_signal(audio: np.ndarray, rms_threshold: float = 0.01) -> bool:
    if audio is None:
        return False
    return calculate_rms(audio) > rms_threshold

def fix_length(target: np.ndarray, source: np.ndarray) -> np.ndarray:
    target_length, source_length = target.shape[-1], source.shape[-1]
    if target_length < source_length:
        return np.pad(target, ((0, 0), (0, source_length - target_length)), mode='constant')
    if target_length > source_length:
        return target[:, :source_length]
    return target

def fix_length_to_duration(target: np.ndarray, duration: float, sr: int) -> np.ndarray:
    target_length = target.shape[-1]
    required_length = int(duration * sr)
    if target_length < required_length:
        return np.pad(target, ((0, 0), (0, required_length - target_length)), mode='constant')
    if target_length > required_length:
        return target[:, :required_length]
    return target

def get_audio_duration(file_path: Path) -> float:
    try:
        return sf.info(file_path).duration
    except Exception as e:
        logger.error(f"Error getting duration for {file_path}: {e}")
        return 0.0

def load_audio(file_path: Path, offset: float, duration: float, sr: int) -> np.ndarray:
    try:
        audio, _ = librosa.load(file_path, sr=sr, offset=offset, duration=duration, mono=False)
        if len(audio.shape) == 1: audio = audio.reshape(1, -1)
        if audio.shape[1] == 0: return np.zeros((2, int(sr * duration)))
        if audio.shape[0] == 1: audio = np.vstack([audio, audio])
        return audio
    except Exception as e:
        logger.error(f"Error loading {file_path} at offset {offset}: {e}")
        return np.zeros((2, int(sr * duration)))

def mix_to_target_snr(target: np.ndarray, noise: np.ndarray, target_snr_db: float) -> Tuple[np.ndarray, float, float]:
    target_power, noise_power = np.mean(target ** 2), np.mean(noise ** 2)
    if noise_power < 1e-8: return target.copy(), 1.0, 0.0
    if target_power < 1e-8: return noise * 0.001, 0.0, 0.001
    
    target_snr_linear = 10 ** (target_snr_db / 10)
    noise_scale = np.sqrt(target_power / (noise_power * target_snr_linear))
    scaled_noise = noise * noise_scale
    mixture = target + scaled_noise
    
    max_amplitude = np.max(np.abs(mixture))
    target_scale = 1.0
    if max_amplitude > 1.0:
        norm_factor = 0.95 / max_amplitude
        mixture *= norm_factor
        target_scale = norm_factor
        noise_scale *= norm_factor
    
    return mixture, target_scale, noise_scale

class RawStems(Dataset):
    def __init__(
        self,
        target_stem: str,
        root_directory: Union[str, Path],
        sr: int = 48000,
        clip_duration: float = 3.0,
        snr_range: Tuple[float, float] = (0.0, 10.0),
        apply_augmentation: bool = True,
        rms_threshold: float = -40.0,
    ) -> None:
        self.root_directory = Path(root_directory)
        self.sr = sr
        self.clip_duration = clip_duration
        self.snr_range = snr_range
        self.apply_augmentation = apply_augmentation
        self.rms_threshold = rms_threshold
        
        target_stem_parts = target_stem.split("_")
        self.target_stem_1 = target_stem_parts[0].strip()
        self.target_stem_2 = target_stem_parts[1].strip() if len(target_stem_parts) > 1 else None
        
        logger.info(f"Scanning '{self.root_directory}' for songs containing stem '{target_stem}'...")
        self.folders = []
        for song_dir in self.root_directory.iterdir():
            if song_dir.is_dir():
                target_path = song_dir / self.target_stem_1
                if self.target_stem_2:
                    target_path /= self.target_stem_2
                
                if target_path.exists() and target_path.is_dir():
                    self.folders.append(song_dir)
        
        if not self.folders:
            raise FileNotFoundError(f"No subdirectories in '{self.root_directory}' were found containing the stem path '{target_stem}'. "
                                    f"Please check your directory structure.")
        logger.info(f"Found {len(self.folders)} song folders.")

        self.audio_files = self._index_audio_files()
        if not self.audio_files: raise ValueError("No audio files found.")
            
        self.activity_masks = self._compute_activity_masks()
        
        self.stem_augmentation = StemAugmentation()
        self.mixture_augmentation = MixtureAugmentation()

    def _compute_activity_masks(self) -> Dict[str, np.ndarray]:
        rms_analysis_path = self.root_directory / "rms_analysis.jsonl"
        if not rms_analysis_path.exists():
            logger.warning("rms_analysis.jsonl not found. Non-silent selection will be disabled.")
            return {}
        
        logger.info(f"Loading and processing RMS data from {rms_analysis_path}")
        rms_data = {}
        with open(rms_analysis_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    rms_data[data['filepath']] = np.array(data['rms_db_per_second'])
                except (json.JSONDecodeError, KeyError):
                    continue

        logger.info("Computing activity masks for all indexed files...")
        activity_masks = {}
        window_size = int(np.ceil(self.clip_duration))

        all_indexed_files = set()
        for song in self.audio_files:
            all_indexed_files.update(p.relative_to(self.root_directory) for p in song["target_stems"])
            all_indexed_files.update(p.relative_to(self.root_directory) for p in song["others"])

        for relative_path in tqdm(all_indexed_files, desc="Computing Activity Masks"):
            path_str = str(relative_path)
            if path_str in rms_data:
                rms_values = rms_data[path_str]
                if len(rms_values) < window_size:
                    activity_masks[path_str] = np.array([False] * len(rms_values))
                    continue
                
                is_loud = rms_values > self.rms_threshold
                sum_loud = np.convolve(is_loud, np.ones(window_size), 'valid')
                avg_loud_enough = sum_loud / window_size > 0.8 
                
                mask = np.zeros(len(rms_values), dtype=bool)
                mask[:len(avg_loud_enough)] = avg_loud_enough
                activity_masks[path_str] = mask
        return activity_masks

    def _find_common_valid_start_seconds(self, file_paths: List[Path]) -> List[int]:
        if not self.activity_masks: return []

        common_mask = None
        min_len = float('inf')

        masks_to_intersect = []
        for file_path in file_paths:
            path_str = str(file_path.relative_to(self.root_directory))
            mask = self.activity_masks.get(path_str)
            if mask is None: return []
            masks_to_intersect.append(mask)
            min_len = min(min_len, len(mask))
        
        if not masks_to_intersect: return []

        final_mask = np.ones(min_len, dtype=bool)
        for mask in masks_to_intersect:
            final_mask &= mask[:min_len]

        return np.where(final_mask)[0].tolist()

    def _index_audio_files(self) -> List[Dict[str, List[Path]]]:
        indexed_songs = []
        for folder in tqdm(self.folders, desc="Indexing audio files"):
            song_dict = {"target_stems": [], "others": []}
            target_folder = folder / self.target_stem_1
            if self.target_stem_2: target_folder /= self.target_stem_2
            
            if target_folder.exists():
                song_dict["target_stems"].extend(p for p in target_folder.rglob('*') if p.suffix.lower() in AUDIO_EXTENSIONS)
            
            for p in folder.rglob('*'):
                if p.suffix.lower() in AUDIO_EXTENSIONS:
                    try:
                        relative_path = p.relative_to(folder)
                        parts = relative_path.parts
                        is_target = len(parts) > 0 and parts[0] == self.target_stem_1 and \
                                    (self.target_stem_2 is None or (len(parts) > 1 and parts[1] == self.target_stem_2))
                        if not is_target:
                            song_dict["others"].append(p)
                    except ValueError:
                        continue
            
            if song_dict["target_stems"] and song_dict["others"]:
                indexed_songs.append(song_dict)
        return indexed_songs
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        song_dict = self.audio_files[index]
        
        for _ in range(100):
            num_targets = random.randint(1, min(len(song_dict["target_stems"]), 5))
            selected_targets = random.sample(song_dict["target_stems"], num_targets)
            
            num_others = random.randint(1, min(len(song_dict["others"]), 10))
            selected_others = random.sample(song_dict["others"], num_others)

            valid_starts = self._find_common_valid_start_seconds(selected_targets + selected_others)

            if valid_starts:
                start_second = random.choice(valid_starts)
                offset = start_second + random.uniform(0, 1.0 - (self.clip_duration % 1.0 or 1.0))
                
                target_mix = sum(load_audio(p, offset, self.clip_duration, self.sr) for p in selected_targets) / num_targets
                other_mix = sum(load_audio(p, offset, self.clip_duration, self.sr) for p in selected_others) / num_others

                if not contains_audio_signal(target_mix) or not contains_audio_signal(other_mix):
                    continue

                target_clean = target_mix.copy()
                target_augmented = self.stem_augmentation.apply(target_mix, self.sr) if self.apply_augmentation else target_mix
                
                mixture, target_scale, _ = mix_to_target_snr(
                    target_augmented, other_mix, random.uniform(*self.snr_range)
                )
                target_clean *= target_scale
                
                mixture_augmented = self.mixture_augmentation.apply(mixture, self.sr) if self.apply_augmentation else mixture

                max_val = np.max(np.abs(mixture_augmented)) + 1e-8
                mixture_final = mixture_augmented / max_val
                target_final = target_clean / max_val
                
                rescale = np.random.uniform(*DEFAULT_GAIN_RANGE)

                mixture = np.nan_to_num(mixture_final * rescale)
                target = np.nan_to_num(target_final * rescale)
                
                target_length = int(self.clip_duration * self.sr)
                if target.shape[1] != target_length:
                    target = np.pad(target, (0, target_length - target.shape[1]), mode='constant')
                else:
                    target = target[:, :target_length]
                if mixture.shape[1] != target_length:
                    mixture = np.pad(mixture, (0, target_length - mixture.shape[1]), mode='constant')
                else:
                    mixture = mixture[:, :target_length]
                
                return {
                    "mixture": np.nan_to_num(mixture),
                    "target": np.nan_to_num(target)
                }

        return self.__getitem__(random.randint(0, len(self.audio_files) - 1))

    def __len__(self) -> int:
        return len(self.audio_files)


class InfiniteSampler(Sampler):
    def __init__(self, dataset: Dataset) -> None:
        self.dataset_size = len(dataset)
        self.indexes = list(range(self.dataset_size))
        self.reset()
    
    def reset(self) -> None:
        random.shuffle(self.indexes)
        self.pointer = 0
        
    def __iter__(self):
        while True:
            if self.pointer >= self.dataset_size: self.reset()
            yield self.indexes[self.pointer]
            self.pointer += 1

    def __len__(self) -> int:
        return self.dataset_size