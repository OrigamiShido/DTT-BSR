import os
import random
import numpy as np
import soundfile as sf
import librosa
import yaml
from pedalboard import (
    Pedalboard,
    Compressor,
    Chorus,
    Delay,
    Gain,
    HighpassFilter,
    Limiter,
    LowpassFilter,
    Phaser,
    Reverb,
    Distortion,
    Bitcrush,
    LadderFilter,
    PeakFilter,
)
from pathlib import Path
from tqdm import tqdm
import tempfile
import subprocess

# --- 配置区域 ---
# 请根据您的环境修改这些路径
DRY_STEMS_ROOT = Path("path/to/slakh2100_flac_redux")  # Slakh2100数据集的根目录 (主要用于器乐)
RAW_STEMS_ROOT = Path("path/to/RawStems")  # RawStems 数据集的根目录
OUTPUT_ROOT = Path("path/to/msr_dataset")  # 生成数据集的输出目录
NUM_TRAIN_SAMPLES = 20000  # 要生成的训练样本数量
NUM_VALID_SAMPLES = 500  # 要生成的验证样本数量
SAMPLE_RATE = 48000  # 目标采样率
SAMPLE_DURATION_SECONDS = 10  # 每个训练样本的持续时间（秒）
SAMPLE_DURATION_SAMPLES = SAMPLE_DURATION_SECONDS * SAMPLE_RATE

# 比赛要求的乐器类别
INSTRUMENT_CLASSES = [
    "vocals", "guitars", "keyboards", "bass", "synthesizers",
    "drums", "percussion", "orchestral"
]

# Slakh2100中的乐器映射到比赛类别
# 关键变更：此映射现在基于 metadata.yaml 中的 inst_class
SLAKH_TO_MSR_MAP = {
    "Piano": "keyboards",
    "Electric Piano": "keyboards",
    "Organ": "keyboards",
    "Guitar": "guitars",
    "Acoustic Guitar": "guitars",
    "Electric Guitar": "guitars",
    "Bass": "bass",
    "Electric Bass": "bass",
    "Violin": "orchestral",
    "Viola": "orchestral",
    "Cello": "orchestral",
    "Contrabass": "orchestral",
    "Trumpet": "orchestral",
    "Trombone": "orchestral",
    "Saxophone": "orchestral",
    "Tenor Sax": "orchestral",
    "Alto Sax": "orchestral",
    "Baritone Sax": "orchestral",
    "Soprano Sax": "orchestral",
    "Clarinet": "orchestral",
    "Flute": "orchestral",
    "Drums": "drums",
    "Drum Machine": "drums",
    "Synthesizer": "synthesizers",
    "Synth Pad": "synthesizers",
    "Synth Lead": "synthesizers",
    'Chromatic Percussion': 'percussion',
    'Strings': 'orchestral',
    'Ensemble': 'orchestral',
    'Brass': 'orchestral',
    'Reed': 'orchestral',
    'Pipe': 'orchestral',
    'Synth Effects': 'synthesizers',
    'Ethnic': 'orchestral',
    'Percussive': 'percussion',
    'Sound Effects': 'synthesizers',
}


# --- 结束配置区域 ---

def find_dry_stems(instrument_root_path, raw_stems_root_path):
    """
    第一步：扫描并分类原始干声音轨
    """
    print("步骤 1: 正在扫描并分类原始干声音轨...")
    stems_dict = {key: [] for key in INSTRUMENT_CLASSES}

    # --- 扫描 Slakh2100 获取器乐音轨 ---
    print(f"  - 正在从 {instrument_root_path} 扫描器乐音轨 (使用 metadata.yaml)...")
    if instrument_root_path.exists():
        # 关键变更：直接查找 metadata.yaml 文件
        metadata_files = list(instrument_root_path.glob("*/Track*/metadata.yaml"))
        for metadata_file in tqdm(metadata_files, desc="Parsing Metadata"):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = yaml.safe_load(f)

                track_path = metadata_file.parent

                if 'stems' in metadata and metadata['stems']:
                    for stem_key, stem_info in metadata['stems'].items():
                        inst_class = stem_info.get('inst_class')
                        if not inst_class:
                            continue

                        msr_class = SLAKH_TO_MSR_MAP.get(inst_class)
                        if msr_class:
                            # 构建 .flac 文件的完整路径
                            stem_file_path = track_path / 'stems' / f"{stem_key}.flac"
                            if stem_file_path.exists():
                                stems_dict[msr_class].append(stem_file_path)
            except Exception as e:
                print(f"解析 {metadata_file} 时出错: {e}")

    else:
        print(f"警告：找不到器乐数据集目录 {instrument_root_path}。")

    # --- 扫描 RawStems 获取“干”人声音轨 ---
    print(f"  - 正在从 {raw_stems_root_path} 扫描原始干人声音轨...")
    if raw_stems_root_path.exists():
        all_raw_files = list(raw_stems_root_path.rglob('*.wav')) + list(raw_stems_root_path.rglob('*.flac'))
        for stem_file in tqdm(all_raw_files, desc="Scanning RawStems for Vocals"):
            filename_lower = stem_file.stem.lower()
            if 'vocal' in filename_lower or 'voice' in filename_lower:
                stems_dict['vocals'].append(stem_file)
    else:
        print(f"警告：找不到 RawStems 数据集目录 {raw_stems_root_path}。将不会生成含人声的样本。")

    print("\n扫描完成。各类别音轨数量:")
    for key, value in stems_dict.items():
        print(f"  - {key}: {len(value)} 个文件")
    return stems_dict


def fix_audio_length(audio, target_length):
    """
    新增：用于修正音频长度的辅助函数，通过填充或截断
    """
    current_length = audio.shape[-1]
    if current_length > target_length:
        return audio[..., :target_length]
    elif current_length < target_length:
        padding_size = target_length - current_length
        # 仅在最后一个轴（时间轴）上填充
        pad_width = [(0, 0)] * (audio.ndim - 1) + [(0, padding_size)]
        return np.pad(audio, pad_width, 'constant')
    return audio


def apply_random_effects_to_stem(audio, sr):
    """
    第二步：模拟混音处理 (Dry -> Wet)，为单个音轨添加随机效果
    """
    board = Pedalboard()

    # 随机应用均衡 (EQ)
    if random.random() < 0.9:
        board.append(HighpassFilter(cutoff_frequency_hz=random.uniform(30, 150)))
        board.append(PeakFilter(
            cutoff_frequency_hz=random.uniform(200, 8000),
            gain_db=random.uniform(-6, 6),
            q=random.uniform(0.5, 5.0)
        ))

    # 随机应用压缩
    if random.random() < 0.7:
        board.append(Compressor(
            threshold_db=random.uniform(-30, -10),
            ratio=random.uniform(2, 8),
            attack_ms=random.uniform(1, 20),
            release_ms=random.uniform(50, 300)
        ))

    # 随机应用饱和度/失真
    if random.random() < 0.3:
        board.append(Distortion(drive_db=random.uniform(3, 15)))

    # 随机应用空间效果 (混响)
    if random.random() < 0.6:
        board.append(Reverb(
            room_size=random.random(),
            damping=random.random(),
            wet_level=random.uniform(0.1, 0.4),
            dry_level=0.8
        ))

    # 随机应用声像
    pan_amount = random.uniform(-0.8, 0.8)

    # 应用效果
    wet_audio = board(audio, sr)

    # 应用声像
    if wet_audio.ndim == 1:
        wet_audio = np.stack([wet_audio, wet_audio])  # 转为立体声

    left_gain = np.cos((pan_amount + 1) * np.pi / 4)
    right_gain = np.sin((pan_amount + 1) * np.pi / 4)
    wet_audio[0, :] *= left_gain
    wet_audio[1, :] *= right_gain

    return wet_audio


def create_mastered_mixture(stems_dict):
    """
    第三步：混合与母带处理 (Wet -> Mastered Mixture)
    """
    mixture_dry_stems = {}
    wet_stems_to_mix = []

    # --- 随机选择乐器组合 ---
    selected_classes = []

    include_vocals = random.random() < 0.6 and stems_dict['vocals']

    if include_vocals:
        selected_classes.append('vocals')
        num_instrument_stems = random.randint(2, 5)
    else:
        num_instrument_stems = random.randint(3, 6)

    instrument_pool = [k for k, v in stems_dict.items() if v and k != 'vocals']
    if instrument_pool:
        selected_classes.extend(random.sample(
            instrument_pool,
            min(num_instrument_stems, len(instrument_pool))
        ))

    if not selected_classes:
        return None, None

    for instrument_class in selected_classes:
        dry_stem_path = random.choice(stems_dict[instrument_class])

        try:
            audio, sr = librosa.load(dry_stem_path, sr=SAMPLE_RATE, mono=False, duration=SAMPLE_DURATION_SECONDS)

            # 快速跳过明显过短的文件
            if audio.shape[-1] < SAMPLE_DURATION_SAMPLES * 0.8:
                continue

            if audio.ndim == 1:
                audio = np.stack([audio, audio])

            # 修正：确保干声音轨长度精确一致
            audio = fix_audio_length(audio, SAMPLE_DURATION_SAMPLES)

            mixture_dry_stems[instrument_class] = audio
            wet_audio = apply_random_effects_to_stem(audio, sr)

            # 修正：再次确保处理后的湿声音轨长度精确一致，以防效果器改变长度
            wet_audio = fix_audio_length(wet_audio, SAMPLE_DURATION_SAMPLES)

            gain = random.uniform(0.5, 1.2)
            wet_stems_to_mix.append(wet_audio * gain)
        except Exception as e:
            print(f"处理文件 {dry_stem_path} 时出错: {e}")
            continue

    if not wet_stems_to_mix:
        return None, None

    # 修正：现在所有 wet_stems_to_mix 中的数组形状都相同，可以安全地求和
    mixture = np.sum(np.array(wet_stems_to_mix), axis=0)

    master_board = Pedalboard([
        PeakFilter(cutoff_frequency_hz=1000, gain_db=random.uniform(-2, 2), q=1.0),
        Compressor(threshold_db=-12, ratio=3.0),
        Limiter(threshold_db=-1.0, release_ms=100)
    ])
    mastered_mixture = master_board(mixture, SAMPLE_RATE)

    peak_val = np.max(np.abs(mastered_mixture))
    if peak_val > 0:
        mastered_mixture /= (peak_val * 1.1)

    return mastered_mixture, mixture_dry_stems


def apply_degradation(audio, sr):
    """
    第四步：模拟真实世界劣化
    """
    degradation_type = random.randint(1, 12)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.wav"
        output_path = Path(tmpdir) / "output.wav"
        sf.write(input_path, audio.T, sr)

        try:
            if 1 <= degradation_type <= 4:  # 模拟劣化
                board = Pedalboard()
                if degradation_type == 1:  # Radio
                    board.append(LowpassFilter(cutoff_frequency_hz=3000))
                    board.append(HighpassFilter(cutoff_frequency_hz=300))
                    board.append(Gain(gain_db=3))
                elif degradation_type == 2:  # Cassette
                    board.append(Chorus(rate_hz=0.5, depth=0.1, mix=0.3))
                    board.append(Distortion(drive_db=1))
                elif degradation_type == 3:  # Vinyl
                    board.append(Bitcrush(bit_depth=12))
                elif degradation_type == 4:
                    board.append(Reverb(room_size=0.8, wet_level=0.5))
                degraded_audio = board(audio, sr)

            else:  # 编解码器
                if degradation_type in [5, 7]:  # AAC
                    bitrate = '64k' if degradation_type == 5 else '128k'
                    codec_output_path = Path(tmpdir) / "temp.m4a"
                    subprocess.run([
                        'ffmpeg', '-i', str(input_path), '-c:a', 'aac',
                        '-b:a', bitrate, str(codec_output_path), '-y', '-hide_banner', '-loglevel', 'error'
                    ], check=True, capture_output=True)
                    subprocess.run([
                        'ffmpeg', '-i', str(codec_output_path), str(output_path), '-y', '-hide_banner', '-loglevel',
                        'error'
                    ], check=True, capture_output=True)

                elif degradation_type in [6, 8]:  # MP3
                    bitrate = '64k' if degradation_type == 6 else '128k'
                    codec_output_path = Path(tmpdir) / "temp.mp3"
                    subprocess.run([
                        'ffmpeg', '-i', str(input_path), '-c:a', 'libmp3lame',
                        '-b:a', bitrate, str(codec_output_path), '-y', '-hide_banner', '-loglevel', 'error'
                    ], check=True, capture_output=True)
                    subprocess.run([
                        'ffmpeg', '-i', str(codec_output_path), str(output_path), '-y', '-hide_banner', '-loglevel',
                        'error'
                    ], check=True, capture_output=True)

                else:
                    degraded_audio, _ = librosa.load(input_path, sr=sr, mono=False)
                    return degraded_audio

                degraded_audio, _ = librosa.load(output_path, sr=sr, mono=False)

        except Exception as e:
            print(f"应用劣化效果时出错 (类型 {degradation_type}): {e}")
            return audio

    return degraded_audio


def generate_dataset(num_samples, split, stems_dict, output_root):
    """
    第五步：组织和生成完整的数据集
    """
    output_path = output_root / split
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n正在生成 {split} 数据集，共 {num_samples} 个样本...")

    pbar = tqdm(range(num_samples))
    generated_count = 0
    while generated_count < num_samples:
        mastered_mixture, dry_stems = create_mastered_mixture(stems_dict)

        if mastered_mixture is None or not dry_stems:
            pbar.update(0)
            continue

        distorted_mixture = apply_degradation(mastered_mixture, SAMPLE_RATE)

        song_id = f"song_{generated_count:05d}"
        song_dir = output_path / song_id
        song_dir.mkdir(exist_ok=True)

        sf.write(song_dir / "distorted_mixture.flac", distorted_mixture.T, SAMPLE_RATE)

        for instrument_class, audio_data in dry_stems.items():
            sf.write(song_dir / f"{instrument_class}.flac", audio_data.T, SAMPLE_RATE)

        generated_count += 1
        pbar.update(1)
        pbar.set_description(f"已生成 {song_id}")
    pbar.close()


def main():
    """主函数"""
    if not DRY_STEMS_ROOT.exists() and not RAW_STEMS_ROOT.exists():
        print(f"错误：找不到任何原始干声音轨目录。")
        print(f"请确保 {DRY_STEMS_ROOT} 或 {RAW_STEMS_ROOT} 至少有一个存在并正确配置。")
        return

    stems_dict = find_dry_stems(DRY_STEMS_ROOT, RAW_STEMS_ROOT)

    generate_dataset(NUM_TRAIN_SAMPLES, "train", stems_dict, OUTPUT_ROOT)
    generate_dataset(NUM_VALID_SAMPLES, "validation", stems_dict, OUTPUT_ROOT)

    print("\n数据集生成完毕！")
    print(f"数据已保存至: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()













