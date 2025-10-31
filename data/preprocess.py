# -*- coding: utf-8 -*-
"""
preprocess.py

Used to convert dynamic mixed datasets (like RawStems) to a static dataset.
It pre-executes all expensive mixing and MSR degradations (MixtureAugmentation) and saves results.
This will greatly accelerate training I/O.

Usage:
python preprocess.py \
    --root-directory /path/to/your/all_data \
    --target-stem Vocals \
    --output-directory /path/to/preprocessed_dataset \
    --num-samples 100000 \
    --sr 48000 \
    --clip-duration 4.0 \
    --snr-range 0.0 10.0
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

# Key imports (make independent: copy from dataset.py/augment.py)
import librosa  # For resampling if needed
from augment import MixtureAugmentation  # Assume this is in same dir or PYTHONPATH

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = ['.flac', '.mp3', '.wav']
DEFAULT_GAIN_RANGE = (0.5, 1.0)
RMS_THRESHOLD_DB = -40.0  # Fixed: Use dB threshold matching modified dataset.py

def calculate_rms(audio: np.ndarray) -> float:
    epsilon = 1e-10
    return np.sqrt(np.mean(audio**2) + epsilon)

def contains_audio_signal(audio: np.ndarray, rms_threshold_db: float = RMS_THRESHOLD_DB) -> bool:
    if audio is None or audio.size == 0:
        return False
    rms_linear = calculate_rms(audio)
    rms_db = 20 * np.log10(rms_linear + 1e-10)
    return rms_db > rms_threshold_db

def fix_length_to_duration(target: np.ndarray, duration_samples: int) -> np.ndarray:
    if not isinstance(target, np.ndarray) or target.ndim < 2:
        logger.warning(f"Invalid input shape for fix_length: {target.shape if isinstance(target, np.ndarray) else type(target)}. Returning zeros.")
        return np.zeros((2, duration_samples), dtype=np.float32)
    target_length = target.shape[-1]
    if target_length < duration_samples:
        return np.pad(target, ((0, 0), (0, duration_samples - target_length)), mode='constant')
    if target_length > duration_samples:
        return target[:, :duration_samples]
    return target

def get_audio_duration(file_path: Path) -> float:
    try:
        return sf.info(str(file_path)).duration
    except Exception as e:
        logger.warning(f"Duration error for {file_path}: {e}")
        return 0.0

def load_audio(file_path: Path, offset: float, duration: float, sr: int) -> np.ndarray:
    required_samples = int(sr * duration)
    default_output = np.zeros((2, required_samples), dtype=np.float32)
    try:
        file_path_str = str(file_path)
        file_info = sf.info(file_path_str)
        start_frame = int(offset * file_info.samplerate)
        frames_to_read = min(int(duration * file_info.samplerate), file_info.frames - start_frame)
        if frames_to_read <= 0 or start_frame >= file_info.frames:
            return default_output
        audio, read_sr = sf.read(file_path_str, start=start_frame, frames=frames_to_read, dtype='float32', always_2d=True)
        audio = audio.T  # (channels, samples)
        if read_sr != sr:
            audio = librosa.resample(audio, orig_sr=read_sr, target_sr=sr)
        if audio.shape[0] == 1:
            audio = np.vstack([audio, audio])
        elif audio.shape[0] > 2:
            audio = audio[:2, :]
        return fix_length_to_duration(audio, required_samples)
    except Exception as e:
        logger.warning(f"Load error for {file_path}: {e}")
        return default_output

def mix_to_target_snr(target: np.ndarray, noise: np.ndarray, target_snr_db: float) -> tuple:
    epsilon = 1e-10
    target_power = np.mean(target ** 2) + epsilon
    noise_power = np.mean(noise ** 2) + epsilon
    if noise_power < 1e-12:
        return target.copy(), 1.0, 0.0
    if target_power < 1e-12:
        return noise * 0.01 / (np.max(np.abs(noise)) + epsilon), 0.0, 0.01
    snr_linear = 10 ** (target_snr_db / 10.0)
    noise_scale = np.sqrt(target_power / (noise_power * snr_linear))
    noise_scale = min(noise_scale, 100.0)
    scaled_noise = noise * noise_scale
    mixture = target + scaled_noise
    max_amp = np.max(np.abs(mixture)) + epsilon
    target_scale = 1.0 if max_amp <= 1.0 else 0.98 / max_amp
    return mixture * target_scale, target_scale, noise_scale * target_scale

def index_training_files(root_directory: Path, target_stem: str) -> list:
    indexed_songs = []
    skipped = {"no_target": 0, "no_others": 0, "unreadable": 0}
    all_song_dirs = [d for d in root_directory.iterdir() if d.is_dir() and os.access(str(d), os.R_OK)]
    if not all_song_dirs:
        logger.error(f"No subdirs in {root_directory}")
        return []
    for folder in tqdm(all_song_dirs, desc="Indexing"):
        song_dict = {"target_stems": [], "others": []}
        target_folder = folder / target_stem
        if not target_folder.is_dir():
            skipped["no_target"] += 1
            continue
        targets = [p for p in target_folder.rglob('*') if p.suffix.lower() in AUDIO_EXTENSIONS and p.is_file() and os.access(str(p), os.R_OK) and get_audio_duration(p) > 0.1]
        if not targets:
            skipped["no_target"] += 1
            continue
        song_dict["target_stems"] = targets
        others = []
        for instr_dir in folder.iterdir():
            if instr_dir.is_dir() and instr_dir.name != target_stem:
                others.extend([p for p in instr_dir.rglob('*') if p.suffix.lower() in AUDIO_EXTENSIONS and p.is_file() and os.access(str(p), os.R_OK) and get_audio_duration(p) > 0.1])
        if not others:
            skipped["no_others"] += 1
            continue
        song_dict["others"] = others
        indexed_songs.append(song_dict)
    logger.info(f"Found {len(indexed_songs)} usable songs. Skipped: {skipped}")
    return indexed_songs

def generate_sample(song_dict: dict, sr: int, clip_duration: float, clip_samples: int, snr_range: tuple, mixture_aug: MixtureAugmentation) -> dict | None:
    for _ in range(50):
        try:
            targets = random.sample(song_dict["target_stems"], random.randint(1, min(3, len(song_dict["target_stems"]))))
            others = random.sample(song_dict["others"], random.randint(1, min(8, len(song_dict["others"]))))
            files = targets + others
            min_dur = min(get_audio_duration(p) for p in files)
            if min_dur < clip_duration:
                continue
            offset = random.uniform(0, min_dur - clip_duration)
            target_mix = np.mean([load_audio(p, offset, clip_duration, sr) for p in targets], axis=0)
            other_mix = np.mean([load_audio(p, offset, clip_duration, sr) for p in others], axis=0)
            if not contains_audio_signal(target_mix) or not contains_audio_signal(other_mix):
                continue
            target_clean = target_mix.copy()
            target_aug = target_mix  # No StemAug here
            target_aug = fix_length_to_duration(target_aug.astype(np.float32), clip_samples)
            other_mix = fix_length_to_duration(other_mix.astype(np.float32), clip_samples)
            target_clean = fix_length_to_duration(target_clean.astype(np.float32), clip_samples)
            mixture, target_scale, _ = mix_to_target_snr(target_aug, other_mix, random.uniform(*snr_range))
            target_clean *= target_scale
            mixture_augmented = mixture_aug.apply(mixture)
            if mixture_augmented.ndim == 3 and mixture_augmented.shape[0] == 1:
                mixture_augmented = mixture_augmented.squeeze(0)
            norm_scale = 0.98 / np.max(np.abs(mixture_augmented)) if np.max(np.abs(mixture_augmented)) > 1.0 else 1.0
            mixture_final = mixture_augmented * norm_scale
            target_final = target_clean * norm_scale
            rescale = random.uniform(*DEFAULT_GAIN_RANGE)
            mixture_out = np.nan_to_num(mixture_final * rescale).astype(np.float32)
            target_out = np.nan_to_num(target_final * rescale).astype(np.float32)
            mixture_out = fix_length_to_duration(mixture_out, clip_samples)
            target_out = fix_length_to_duration(target_out, clip_samples)
            if mixture_out.shape != (2, clip_samples) or target_out.shape != (2, clip_samples):
                continue
            return {"mixture": mixture_out, "target": target_out}
        except Exception:
            continue
    return None

def main(args):
    root_dir = Path(args.root_directory)
    output_dir = Path(args.output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    sr, clip_duration, clip_samples = args.sr, args.clip_duration, int(args.sr * args.clip_duration)
    snr_range, num_samples = tuple(args.snr_range), args.num_samples
    logger.info("Checking MixtureAugmentation...")
    try:
        MixtureAugmentation(sr=sr)
    except Exception as e:
        logger.error(f"Init failed: {e}. Check models/dependencies.")
        return
    song_list = index_training_files(root_dir, args.target_stem)
    if not song_list:
        logger.error("No songs found. Check root_directory.")
        return
    mixture_aug = MixtureAugmentation(sr=sr)  # Single instance for single-thread
    logger.info(f"Generating {num_samples} samples (single-threaded)...")
    start_time = time.time()
    generated, failed = 0, 0
    pbar = tqdm(range(num_samples), desc="Samples")
    for i in pbar:
        sample = generate_sample(song_list[random.randint(0, len(song_list)-1)], sr, clip_duration, clip_samples, snr_range, mixture_aug)
        if sample:
            name = f"sample_{i:07d}"
            try:
                sf.write(str(output_dir / f"{name}_mix.flac"), sample["mixture"].T, sr, subtype='PCM_24')
                sf.write(str(output_dir / f"{name}_target.flac"), sample["target"].T, sr, subtype='PCM_24')
                generated += 1
            except Exception as e:
                logger.warning(f"Write failed for {name}: {e}")
                failed += 1
        else:
            failed += 1
    pbar.close()
    total_time = time.time() - start_time
    logger.info(f"Done: {generated} generated, {failed} failed in {total_time:.2f}s. Avg: {total_time / num_samples:.3f}s/sample.")
    logger.info(f"Output: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess audio dataset.")
    parser.add_argument("--root-directory", type=str, required=True)
    parser.add_argument("--target-stem", type=str, required=True)
    parser.add_argument("--output-directory", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=100000)
    parser.add_argument("--sr", type=int, default=48000)
    parser.add_argument("--clip-duration", type=float, default=4.0)
    parser.add_argument("--snr-range", type=float, nargs=2, default=[0.0, 10.0])
    args = parser.parse_args()
    main(args)
