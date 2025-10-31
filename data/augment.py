# -*- coding: utf-8 -*-
"""
实现了 'StemAugmentation' 和 'MixtureAugmentation'
使用 'pedalboard' 和 'scipy' 来进行数据增强。

- StemAugmentation: 模拟录音和制作效果 (EQ, 压缩, 失真等)
- MixtureAugmentation: 模拟 MSRBench 定义的 12 种退化情况（更精确实现）
"""
import numpy as np
import scipy.signal as signal
import random
from pedalboard import (
    Pedalboard,
    Compressor,
    Distortion,
    Gain,
    Limiter,
    Reverb,
    Resample,
    LowpassFilter,
    HighpassFilter,
    Chorus,
    Bitcrush,
    LadderFilter
)
import logging
import subprocess  # 用于调用 ffmpeg
import tempfile    # 用于创建临时文件
import soundfile as sf  # 用于读写临时文件
from pathlib import Path
from typing import Optional, Dict
import librosa  # 添加 librosa 导入
import os  # For os.access

# 新增导入用于改进实现
import pyroomacoustics as pra  # 对于 DT4 房间模拟
import torch
import torchaudio  # 对于 Encodec
from encodec import EncodecModel
from encodec.utils import convert_audio
import dac  # 对于 DAC
from audiotools import AudioSignal

# 假设 WHAM! 噪声样本已下载到本地路径，例如 'wham_noise/'
# 下载 WHAM! 从 http://wham.whisper.ai/
# 假设 vinyl 裂纹样本已下载，例如从 Freesound: 'vinyl_crackle.wav'
VINYL_CRACKLE_PATH = 'path/to/vinyl_crackle.wav'  # 替换为实际路径
WHAM_NOISE_DIR = 'path/to/wham_noise/'  # 替换为实际路径


logger = logging.getLogger(__name__)
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
# --- 辅助函数 ---
def _calculate_rms(audio: np.ndarray) -> float:
    epsilon = 1e-10
    audio_f32 = audio.astype(np.float32)
    if audio_f32.size == 0: return 0.0
    return np.sqrt(np.mean(audio_f32 ** 2) + epsilon)

def _add_noise(audio: np.ndarray, snr_db: float) -> np.ndarray:
    signal_rms = _calculate_rms(audio)
    if signal_rms < 1e-8: return audio
    target_noise_rms = signal_rms / (10 ** (snr_db / 20.0))
    noise = np.random.randn(*audio.shape).astype(audio.dtype)
    noise_current_rms = _calculate_rms(noise)
    if noise_current_rms < 1e-10: return audio
    noise_scaled = noise * (target_noise_rms / noise_current_rms)
    return audio + noise_scaled

def _generate_crackle_noise(length: int, sr: int, density: float = 0.0005, strength: float = 0.5) -> np.ndarray:
    noise = np.zeros(length)
    num_crackles = int(density * length)
    if num_crackles == 0: return noise
    indices = np.random.randint(0, length, num_crackles)
    amplitudes = np.random.exponential(scale=strength/3.0, size=num_crackles) * np.random.choice([-1, 1], num_crackles)
    amplitudes = np.clip(amplitudes, -strength*1.5, strength*1.5)
    noise[indices] = amplitudes
    try:
        cutoff_hz = random.uniform(2000, 5000)
        nyquist = sr / 2.0
        if cutoff_hz >= nyquist: cutoff_hz = nyquist * 0.95
        sos = signal.butter(2, cutoff_hz / nyquist, 'highpass', output='sos')
        noise = signal.sosfiltfilt(sos, noise)
    except Exception as e:
        logger.warning(f"Crackle noise filtering failed: {e}")
        noise *= 0.5
    if len(noise) != length:
        reshaped_noise = noise.reshape(1, -1) if noise.ndim == 1 else noise
        if reshaped_noise.ndim == 1: reshaped_noise = reshaped_noise[np.newaxis, :]
        fixed_noise = fix_length_to_duration(reshaped_noise, length)
        noise = fixed_noise.flatten()
    return noise

# 新增：简单的Rayleigh fading模拟 (sum-of-sinusoids 方法)
def _apply_rayleigh_fading(audio: np.ndarray, sr: int, fd: float = 100, num_sinusoids: int = 8):
    # fd: Doppler frequency
    t = np.arange(len(audio)) / sr
    fading = np.zeros_like(audio, dtype=np.complex64)
    for i in range(num_sinusoids):
        phi = np.random.uniform(0, 2 * np.pi)
        theta = np.random.uniform(0, 2 * np.pi)
        alpha = np.random.uniform(0, 2 * np.pi)
        fading += np.exp(1j * (2 * np.pi * fd * t * np.cos(theta) + phi)) / np.sqrt(num_sinusoids)
    fading = np.abs(fading)  # Rayleigh envelope
    return audio * fading.real.astype(audio.dtype)  # Apply to audio (assuming mono; extend for stereo)

# 新增：加载随机WHAM!噪声
def _load_wham_noise(length: int, sr: int):
    noise_files = [f for f in os.listdir(WHAM_NOISE_DIR) if f.endswith('.wav')]
    if not noise_files:
        logger.warning("No WHAM! noise files found. Using white noise fallback.")
        return _add_noise(np.zeros(length), snr_db=20)
    noise_path = os.path.join(WHAM_NOISE_DIR, random.choice(noise_files))
    noise, noise_sr = librosa.load(noise_path, sr=sr, mono=True)
    noise = fix_length_to_duration(noise, length)
    return noise

# --- Stem Augmentation (用于 'target_clean') ---
class StemAugmentation:
    def __init__(self, sr: int = 48000):
        if sr <= 0: raise ValueError("Sample rate must be positive.")
        self.sr = sr
        self.p_eq = 0.5
        self.p_comp = 0.5
        self.p_distort = 0.2
        self.p_reverb = 0.3
        logger.info(f"StemAugmentation initialized with sr={self.sr}")

    def apply(self, audio: np.ndarray) -> np.ndarray:
        original_dtype = audio.dtype
        processed_audio = audio.astype(np.float32)
        if not np.all(np.isfinite(processed_audio)):
            logger.warning("NaN/Inf after input conversion. Clipping.")
            processed_audio = np.nan_to_num(processed_audio)
        if _calculate_rms(processed_audio) < 1e-6: return audio

        # Apply EQ
        if random.random() < self.p_eq:
            try:
                eq_output = self._apply_random_eq(processed_audio, self.sr)
                if np.all(np.isfinite(eq_output)): processed_audio = eq_output
                else: logger.warning("NaN/Inf after EQ. Using pre-EQ.")
            except Exception as e:
                logger.error(f"Error applying EQ: {e}", exc_info=True)

        # Apply Pedalboard effects
        board = Pedalboard([])
        apply_board = False
        if random.random() < self.p_comp:
            board.append(Compressor(threshold_db=random.uniform(-30, -10), ratio=random.uniform(2, 8), attack_ms=random.uniform(1, 20), release_ms=random.uniform(50, 300)))
            apply_board = True
        if random.random() < self.p_distort:
            board.append(Gain(gain_db=random.uniform(3, 10)))
            board.append(Distortion(drive_db=random.uniform(3, 15)))
            board.append(Gain(gain_db=random.uniform(-15, -5)))
            apply_board = True
        if random.random() < self.p_reverb:
            board.append(Reverb(room_size=random.random(), damping=random.random(), wet_level=random.uniform(0.1, 0.4), dry_level=random.uniform(0.6, 0.9)))
            apply_board = True

        if apply_board:
            try:
                board_input = processed_audio
                if not np.all(np.isfinite(board_input)):
                    logger.warning("NaN/Inf before Pedalboard. Clipping.")
                    board_input = np.nan_to_num(board_input)
                if len(board) == 0:
                    logger.warning("apply_board is True but Pedalboard is empty.")
                else:
                    board_output = board(board_input, sample_rate=self.sr)
                    if not np.all(np.isfinite(board_output)):
                        logger.warning("NaN/Inf after Pedalboard. Using input.")
                        processed_audio = board_input
                    else:
                        processed_audio = board_output
            except Exception as e:
                logger.error(f"Error applying Pedalboard effects (Stem): {e}", exc_info=True)

        # Convert back
        try:
            if original_dtype != np.float32:
                if not np.all(np.isfinite(processed_audio)):
                    logger.warning("NaN/Inf before final conversion. Clipping.")
                    processed_audio = np.nan_to_num(processed_audio)
                output_audio = processed_audio.astype(original_dtype)
            else:
                output_audio = processed_audio
        except Exception as e:
            logger.warning(f"Could not convert back to {original_dtype}, keeping float32. Error: {e}")
            output_audio = processed_audio.astype(np.float32)
        return output_audio

    def _apply_random_eq(self, audio_buffer_float32, sr):
        min_eq_duration_sec = 0.05
        if audio_buffer_float32.shape[-1] < sr * min_eq_duration_sec:
            logger.debug(f"Audio shorter than {min_eq_duration_sec}s, skipping EQ.")
            return audio_buffer_float32
        eq_type = random.choice(['parametric', 'graphic'])
        audio_float64 = audio_buffer_float32.astype(np.float64)
        output_float64 = audio_float64
        try:
            if eq_type == 'graphic':
                output_float64 = self._apply_graphic_eq(audio_float64, sr)
            else:
                output_float64 = self._apply_parametric_eq(audio_float64, sr)
        except Exception as e:
            logger.error(f"Error during {eq_type} EQ application: {e}", exc_info=True)
            return audio_buffer_float32
        output_float32 = output_float64.astype(np.float32)
        if not np.all(np.isfinite(output_float32)):
            logger.warning(f"NaN/Inf after {eq_type} EQ. Returning original.")
            return audio_buffer_float32
        return output_float32

    def _apply_graphic_eq(self, audio_buffer_float64, sr):
        n_channels, length = audio_buffer_float64.shape
        output_buffer = audio_buffer_float64.copy()
        frequencies = [25, 40, 63, 100, 160, 250, 400, 630, 1000, 1600, 2500, 4000, 6300, 10000, 16000, 20000]
        gains_db = np.random.uniform(-6, 6, 16)
        nyquist = sr / 2.0

        for i, (freq, gain_db) in enumerate(zip(frequencies, gains_db)):
            next_freq_valid = i < len(frequencies) - 1 and frequencies[i+1] < nyquist
            if freq >= nyquist: continue
            sos = None
            q_val = 0.7
            try:
                if i == 0:
                    sos = self._design_parametric_filter('lowshelf', freq, q_val, gain_db, sr)
                elif i == len(frequencies) - 1 or not next_freq_valid:
                    sos = self._design_parametric_filter('highshelf', freq, q_val, gain_db, sr)
                else:
                    lower_freq = frequencies[i - 1] if i > 0 else freq / 1.6
                    upper_freq = frequencies[i + 1]
                    bw = max(upper_freq - lower_freq, 1e-6)
                    q_val = np.clip(freq / bw * 2.5, 0.1, 20.0)
                    sos = self._design_parametric_filter('peak', freq, q_val, gain_db, sr)
            except Exception as e:
                logger.error(f"Err designing graphic EQ filter (f={freq}, g={gain_db:.1f}, q={q_val:.2f}): {e}", exc_info=True)
                continue
            if sos is None or not np.all(np.isfinite(sos)): continue
            for ch in range(n_channels):
                try:
                    filtered_ch = signal.sosfiltfilt(sos, output_buffer[ch])
                    if not np.all(np.isfinite(filtered_ch)):
                        logger.warning(f"NaN/Inf after sosfiltfilt (Graphic EQ, f={freq}, ch={ch}). Skipping.")
                        continue
                    output_buffer[ch] = filtered_ch
                except Exception as e:
                    logger.warning(f"Graphic EQ sosfiltfilt failed (f={freq}, g={gain_db:.2f}dB, ch={ch}): {type(e).__name__} - {e}")
                    continue
        return output_buffer

    def _apply_parametric_eq(self, audio_buffer_float64, sr):
        n_channels, length = audio_buffer_float64.shape
        output_buffer = audio_buffer_float64.copy()
        num_bands = np.random.randint(1, 6)
        min_freq, max_freq = 20.0, sr / 2.0 - 1.0
        for _ in range(num_bands):
            filter_type = random.choice(['lowpass', 'highpass', 'bandpass', 'bandstop', 'peak', 'lowshelf', 'highshelf'])
            if filter_type == 'lowpass': frequency = np.random.uniform(200, max_freq * 0.8)
            elif filter_type == 'highpass': frequency = np.random.uniform(min_freq * 2, max_freq * 0.5)
            elif filter_type in ['bandpass', 'bandstop', 'peak']: frequency = np.random.uniform(100, max_freq * 0.9)
            elif filter_type == 'lowshelf': frequency = np.random.uniform(50, 1000)
            elif filter_type == 'highshelf': frequency = np.random.uniform(1000, max_freq * 0.9)
            frequency = np.clip(frequency, min_freq, max_freq)
            Q = np.random.uniform(0.5, 5.0)
            gain_db = np.random.uniform(-6, 6)
            sos = None
            try:
                if filter_type in ['lowpass', 'highpass', 'bandpass', 'bandstop']:
                    sos = self._design_basic_filter(filter_type, frequency, Q, sr)
                elif filter_type in ['peak', 'lowshelf', 'highshelf']:
                    sos = self._design_parametric_filter(filter_type, frequency, Q, gain_db, sr)
            except Exception as e:
                logger.error(f"Filter design exception ({filter_type}@{frequency:.1f}Hz): {e}", exc_info=True)
                continue
            if sos is None or not np.all(np.isfinite(sos)): continue
            for ch in range(n_channels):
                try:
                    filtered_ch = signal.sosfiltfilt(sos, output_buffer[ch])
                    if not np.all(np.isfinite(filtered_ch)):
                        logger.warning(f"NaN/Inf after sosfiltfilt (Parametric EQ, type={filter_type}, f={frequency:.1f}, ch={ch}). Skipping.")
                        continue
                    output_buffer[ch] = filtered_ch
                except Exception as e:
                    logger.warning(f"Parametric EQ sosfiltfilt failed ({filter_type}@{frequency:.1f}Hz, ch={ch}): {type(e).__name__} - {e}")
                    continue
        return output_buffer

    def _design_basic_filter(self, filter_type, frequency, Q, sr):
        nyquist = sr / 2.0
        normalized_frequency = np.clip(frequency / nyquist, 1e-7, 1.0 - 1e-7)
        implementation = random.choice(['butter', 'cheby1', 'cheby2', 'ellip', 'bessel'])
        order = random.choice([2, 4, 6, 8])
        def get_band(freq, Q, nyquist):
            Q = max(Q, 1e-6)
            bw = freq / Q
            low = np.clip((freq - bw / 2.0) / nyquist, 1e-7, 1.0 - 3e-7)
            high = np.clip((freq + bw / 2.0) / nyquist, low + 1e-7, 1.0 - 2e-7)
            return [low, high]
        band_order = max(2, (order // 2) * 2)
        common_args = {'N': order, 'Wn': normalized_frequency, 'btype': filter_type, 'analog': False, 'output': 'sos'}
        band_args = {'N': band_order, 'Wn': get_band(frequency, Q, nyquist), 'btype': filter_type, 'analog': False, 'output': 'sos'}
        try:
            sos = None
            if implementation == 'butter':
                sos = signal.butter(**(common_args if filter_type in ['lowpass', 'highpass'] else band_args))
            elif implementation == 'cheby1':
                rp = np.random.uniform(0.1, 3.0)
                sos = signal.cheby1(rp=rp, **(common_args if filter_type in ['lowpass', 'highpass'] else band_args))
            elif implementation == 'cheby2':
                rs = np.random.uniform(20, 60)
                sos = signal.cheby2(rs=rs, **(common_args if filter_type in ['lowpass', 'highpass'] else band_args))
            elif implementation == 'ellip':
                rp, rs = np.random.uniform(0.1, 3.0), np.random.uniform(20, 60)
                sos = signal.ellip(rp=rp, rs=rs, **(common_args if filter_type in ['lowpass', 'highpass'] else band_args))
            elif implementation == 'bessel':
                b_order = min(order, 8)
                b_band_order = max(2, (b_order // 2) * 2)
                common_args_be = {'N': b_order, 'Wn': normalized_frequency, 'btype': filter_type, 'analog': False, 'output': 'sos', 'norm': 'mag'}
                band_args_be = {'N': b_band_order, 'Wn': get_band(frequency, Q, nyquist), 'btype': filter_type, 'analog': False, 'output': 'sos', 'norm': 'mag'}
                sos = signal.bessel(**(common_args_be if filter_type in ['lowpass', 'highpass'] else band_args_be))
            if sos is None or not np.all(np.isfinite(sos)): return None
            return sos
        except ValueError as e:
            return None
        except Exception as e:
            logger.error(f"Unexpected error designing basic filter: {e}", exc_info=True)
            return None

    def _design_parametric_filter(self, filter_type, frequency, Q, gain_db, sr):
        if not (np.isfinite(Q) and Q > 1e-9): Q = 0.707
        if not np.isfinite(gain_db): gain_db = 0.0
        frequency = np.clip(frequency, 20.0, sr / 2.0 - 1.0)
        try:
            gain_linear = 10 ** (gain_db / 20.0)
            w0 = 2 * np.pi * frequency / sr
            cos_w0, sin_w0 = np.cos(w0), np.sin(w0)
            alpha = sin_w0 / (2.0 * max(Q, 1e-9))
            A = gain_linear
            sqrt_A = np.sqrt(max(A, 0))
            a0, a1, a2, b0, b1, b2 = 1.0, 0.0, 0.0, 0.0, 0.0, 0.0
            if filter_type == 'peak':
                b0 = 1 + alpha * A
                b1 = -2 * cos_w0
                b2 = 1 - alpha * A
                a0 = 1 + alpha / A
                a1 = -2 * cos_w0
                a2 = 1 - alpha / A
            elif filter_type == 'lowshelf':
                beta = 2 * sqrt_A * alpha
                b0 = A * ((A + 1) - (A - 1) * cos_w0 + beta)
                b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
                b2 = A * ((A + 1) - (A - 1) * cos_w0 - beta)
                a0 = (A + 1) + (A - 1) * cos_w0 + beta
                a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
                a2 = (A + 1) + (A - 1) * cos_w0 - beta
            elif filter_type == 'highshelf':
                beta = 2 * sqrt_A * alpha
                b0 = A * ((A + 1) + (A - 1) * cos_w0 + beta)
                b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
                b2 = A * ((A + 1) + (A - 1) * cos_w0 - beta)
                a0 = (A + 1) - (A - 1) * cos_w0 + beta
                a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
                a2 = (A + 1) - (A - 1) * cos_w0 - beta
            else:
                logger.warning(f"Unsupported parametric type: {filter_type}")
                return None
            if not np.isfinite(a0) or abs(a0) < 1e-9: return None
            coeffs = np.array([b0 / a0, b1 / a0, b2 / a0, 1, a1 / a0, a2 / a0])
            if not np.all(np.isfinite(coeffs)): return None
            return coeffs.reshape(1, 6)
        except (ValueError, FloatingPointError, ZeroDivisionError) as e:
            return None
        except Exception as e:
            logger.error(f"Unexpected parametric design error: {e}", exc_info=True)
            return None

# --- Mixture Augmentation (用于 'mixture' - 包含 MSR 退化) ---
class MixtureAugmentation:
    def __init__(self, sr: int = 48000):
        if sr <= 0:
            raise ValueError("Sample rate must be positive.")
        self.sr = sr
        self.target_sample_rates = [8000, 11025, 16000, 22050, 32000]
        logger.info(f"MixtureAugmentation (MSR Degradations) initialized with sr={self.sr}")
        # 预加载模型以避免每次调用加载
        self._load_models()

    def _load_models(self):
        # DAC 模型
        self.dac_22khz = None
        self.dac_44khz = None
        try:
            model_path_22 = dac.utils.download(model_type="24khz")  # 注意: DAC的24khz接近22khz, 调整如果有精确
            self.dac_22khz = dac.DAC.load(model_path_22)
            model_path_44 = dac.utils.download(model_type="44khz")
            self.dac_44khz = dac.DAC.load(model_path_44)
        except Exception as e:
            logger.warning(f"Failed to load DAC models: {e}. DT9/DT10 will skip.")

        # Encodec 模型
        self.encodec_48khz = None
        try:
            self.encodec_48khz = EncodecModel.encodec_model_48khz()
        except Exception as e:
            logger.warning(f"Failed to load Encodec model: {e}. DT11/DT12 will skip.")

    def apply(self, audio: np.ndarray) -> np.ndarray:
        original_dtype = audio.dtype
        if audio.dtype != np.float32:
            processed_audio = audio.astype(np.float32)
            if not np.all(np.isfinite(processed_audio)):
                logger.warning("NaN/Inf detected after converting mixture input to float32. Clipping.")
                processed_audio = np.nan_to_num(processed_audio)
        else:
            processed_audio = audio

        if _calculate_rms(processed_audio) < 1e-6:
            return audio

        degradation_type = random.randint(1, 12)
        result = processed_audio
        try:
            if degradation_type == 1:
                result = self._apply_radio_approx(processed_audio)
            elif degradation_type == 2:
                result = self._apply_cassette_approx(processed_audio)
            elif degradation_type == 3:
                result = self._apply_vinyl_approx(processed_audio)
            elif degradation_type == 4:
                result = self._apply_live_approx(processed_audio)
            elif degradation_type == 5:
                result = self._apply_codec(processed_audio, codec='aac', bitrate='64k')
            elif degradation_type == 6:
                result = self._apply_codec(processed_audio, codec='libmp3lame', bitrate='64k')
            elif degradation_type == 7:
                result = self._apply_codec(processed_audio, codec='aac', bitrate='128k')
            elif degradation_type == 8:
                result = self._apply_codec(processed_audio, codec='libmp3lame', bitrate='128k')
            elif degradation_type == 9:
                result = self._apply_dac(processed_audio, sr_target=24000)  # 接近22kHz
            elif degradation_type == 10:
                result = self._apply_dac(processed_audio, sr_target=44100)
            elif degradation_type == 11:
                result = self._apply_encodec(processed_audio, bandwidth=6.0)
            elif degradation_type == 12:
                result = self._apply_encodec(processed_audio, bandwidth=3.0)

            if not isinstance(result, np.ndarray) or not np.all(np.isfinite(result)):
                logger.error(f"Degradation DT{degradation_type} resulted in invalid output. Returning original.")
                result = processed_audio

            if result.shape[-1] != audio.shape[-1]:
                logger.warning(f"Length mismatch after DT{degradation_type}. Fixing.")
                result = fix_length_to_duration(result, audio.shape[-1])

            if result.dtype != original_dtype:
                try:
                    if np.issubdtype(original_dtype, np.integer):
                        max_val = np.iinfo(original_dtype).max
                        min_val = np.iinfo(original_dtype).min
                        result_clipped = np.clip(result * max_val, min_val, max_val)
                        final_result = result_clipped.astype(original_dtype)
                    else:
                        result_clipped = np.clip(result, -1.0, 1.0)
                        final_result = result_clipped.astype(original_dtype)
                except Exception as e:
                    logger.warning(f"Conversion failed, keeping float32: {e}")
                    final_result = np.clip(result.astype(np.float32), -1.0, 1.0)
            else:
                final_result = result

            return final_result

        except Exception as e:
            logger.error(f"Error during DT{degradation_type}: {e}", exc_info=True)
            return audio

    # --- 改进的 MSR 退化实现 ---

    def _apply_radio_approx(self, audio: np.ndarray) -> np.ndarray:
        # 改进: 添加Rayleigh fading模拟
        board = Pedalboard([LowpassFilter(cutoff_frequency_hz=random.uniform(3000, 5000)),
                            HighpassFilter(cutoff_frequency_hz=random.uniform(100, 300)),
                            Compressor(threshold_db=-10, ratio=4.0, attack_ms=5, release_ms=100),
                            Gain(gain_db=random.uniform(0, 3))])
        try:
            processed = board(audio, sample_rate=self.sr)
            processed = _apply_rayleigh_fading(processed, self.sr, fd=100)  # Doppler 100Hz for FM
            processed = _add_noise(processed, snr_db=26)  # 如MSRBench
            return processed
        except Exception as e:
            logger.error(f"Error in _apply_radio_approx: {e}")
            return audio

    def _apply_cassette_approx(self, audio: np.ndarray) -> np.ndarray:
        # 改进: 更多效果模拟磁带
        board = Pedalboard([Chorus(rate_hz=random.uniform(0.1, 0.5), depth=random.uniform(0.05, 0.15), mix=random.uniform(0.1, 0.3)),  # Wow/flutter
                            Gain(gain_db=random.uniform(0, 2)),
                            Distortion(drive_db=random.uniform(1, 4)),  # Saturation
                            LowpassFilter(cutoff_frequency_hz=random.uniform(10000, 15000))])
        try:
            processed = board(audio, sample_rate=self.sr)
            # 添加磁带嘶声 (hiss)
            processed = _add_noise(processed, snr_db=random.uniform(35, 45))
            return processed
        except Exception as e:
            logger.error(f"Error in _apply_cassette_approx: {e}")
            return audio

    def _apply_vinyl_approx(self, audio: np.ndarray) -> np.ndarray:
        # 改进: 使用预下载的Freesound vinyl裂纹样本
        board = Pedalboard([HighpassFilter(cutoff_frequency_hz=random.uniform(30, 60))])
        try:
            processed = board(audio, sample_rate=self.sr)
            # 加载vinyl裂纹
            crackle, crackle_sr = librosa.load(VINYL_CRACKLE_PATH, sr=self.sr, mono=True)
            crackle = fix_length_to_duration(crackle, audio.shape[-1])
            if crackle.ndim == 1: crackle = np.tile(crackle, (processed.shape[0], 1))
            processed += crackle * random.uniform(0.05, 0.2)
            processed = _add_noise(processed, snr_db=random.uniform(40, 50))
            return processed
        except Exception as e:
            logger.error(f"Error in _apply_vinyl_approx: {e}")
            return audio

    def _apply_live_approx(self, audio: np.ndarray) -> np.ndarray:
        # 改进: 使用pyroomacoustics生成RIR，并添加WHAM!噪声
        try:
            # 生成随机房间
            room_dim = [random.uniform(5, 20), random.uniform(5, 20), random.uniform(3, 6)]
            e_abs = random.uniform(0.2, 0.8)
            source_pos = [random.uniform(1, room_dim[0]-1), random.uniform(1, room_dim[1]-1), random.uniform(1, room_dim[2]-1)]
            mic_pos = [random.uniform(1, room_dim[0]-1), random.uniform(1, room_dim[1]-1), random.uniform(1, room_dim[2]-1)]
            room = pra.ShoeBox(room_dim, fs=self.sr, absorption=e_abs, max_order=10)
            room.add_source(source_pos)
            room.add_microphone(mic_pos)
            room.compute_rir()
            rir = room.rir[0][0]  # 获取第一个mic的RIR

            # 卷积RIR
            processed = np.zeros_like(audio)
            for ch in range(audio.shape[0]):
                processed[ch] = signal.convolve(audio[ch], rir, mode='same')

            # 话筒带通
            center_freq = random.uniform(2000, 5000)
            q_factor = random.uniform(0.5, 1.2)
            nyquist = self.sr / 2.0
            bandwidth = center_freq / q_factor
            low_cutoff_norm = np.clip((center_freq - bandwidth / 2.0) / nyquist, 1e-7, 1.0 - 3e-7)
            high_cutoff_norm = np.clip((center_freq + bandwidth / 2.0) / nyquist, low_cutoff_norm + 1e-7, 1.0 - 2e-7)
            sos = signal.butter(4, [low_cutoff_norm, high_cutoff_norm], btype='band', output='sos')
            processed = signal.sosfiltfilt(sos, processed, axis=-1)

            # 添加WHAM!噪声 at 20dB SNR
            noise = _load_wham_noise(audio.shape[-1], self.sr)
            if noise.shape[0] == 1: noise = np.tile(noise, (audio.shape[0], 1))
            signal_rms = _calculate_rms(processed)
            noise_rms = _calculate_rms(noise)
            target_noise_rms = signal_rms / (10 ** (20 / 20.0))
            noise = noise * (target_noise_rms / noise_rms)
            processed += noise

            # 剪切
            clip_threshold = random.uniform(0.85, 0.98)
            processed = np.clip(processed * random.uniform(1.0, 1.2), -clip_threshold, clip_threshold)
            return processed.astype(np.float32)
        except Exception as e:
            logger.error(f"Error in _apply_live_approx: {e}")
            return audio

    def _apply_codec(self, audio: np.ndarray, codec: str, bitrate: str) -> np.ndarray:
        # 保持原有ffmpeg实现
        if audio.dtype != np.float32: input_audio = audio.astype(np.float32)
        else: input_audio = audio
        if not np.all(np.isfinite(input_audio)):
            logger.warning("NaN/Inf in codec input. Clipping.")
            input_audio = np.nan_to_num(input_audio)
        if _calculate_rms(input_audio) < 1e-7: return input_audio

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as infile_obj, \
             tempfile.NamedTemporaryFile(suffix=".m4a" if codec == 'aac' else ".mp3", delete=False) as codecfile_obj, \
             tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as outfile_obj:
            input_path = Path(infile_obj.name)
            codec_output_path = Path(codecfile_obj.name)
            output_path = Path(outfile_obj.name)

        try:
            sf.write(input_path, input_audio.T, self.sr, format='WAV', subtype='FLOAT')
            encode_cmd = ['ffmpeg', '-y', '-i', str(input_path), '-c:a', codec, '-b:a', bitrate, '-ar', str(self.sr), '-strict', '-2', str(codec_output_path), '-hide_banner', '-loglevel', 'error']
            subprocess.run(encode_cmd, check=True, capture_output=True)
            if not codec_output_path.exists() or codec_output_path.stat().st_size < 100:
                raise ValueError("Encode failed")
            decode_cmd = ['ffmpeg', '-y', '-i', str(codec_output_path), '-ar', str(self.sr), str(output_path), '-hide_banner', '-loglevel', 'error']
            subprocess.run(decode_cmd, check=True, capture_output=True)
            if not output_path.exists() or output_path.stat().st_size < 100:
                raise ValueError("Decode failed")
            decoded_audio, decoded_sr = sf.read(output_path, dtype='float32', always_2d=True)
            decoded_audio = decoded_audio.T
            if decoded_sr != self.sr:
                decoded_audio = librosa.resample(decoded_audio, orig_sr=decoded_sr, target_sr=self.sr)
            decoded_audio = fix_length_to_duration(decoded_audio, input_audio.shape[-1])
            if not np.all(np.isfinite(decoded_audio)): raise ValueError("Invalid decoded audio")
            return decoded_audio
        except Exception as e:
            logger.error(f"Codec error ({codec}@{bitrate}): {e}")
            return input_audio
        finally:
            for p in [input_path, codec_output_path, output_path]:
                if p.exists():
                    p.unlink()

    def _apply_dac(self, audio: np.ndarray, sr_target: int):
        if self.dac_22khz is None and sr_target < 30000 or self.dac_44khz is None:
            logger.warning("DAC model not loaded. Skipping.")
            return audio
        model = self.dac_22khz if sr_target < 30000 else self.dac_44khz
        try:
            resampled = librosa.resample(audio, orig_sr=self.sr, target_sr=sr_target)
            signal = AudioSignal(resampled, sample_rate=sr_target)
            compressed = model.compress(signal)
            decompressed = model.decompress(compressed)
            result = librosa.resample(decompressed.audio_data.numpy(), orig_sr=sr_target, target_sr=self.sr)
            return result.astype(np.float32)
        except Exception as e:
            logger.error(f"DAC error (sr={sr_target}): {e}")
            return audio

    def _apply_encodec(self, audio: np.ndarray, bandwidth: float):
        if self.encodec_48khz is None:
            logger.warning("Encodec model not loaded. Skipping.")
            return audio
        model = self.encodec_48khz
        model.set_target_bandwidth(bandwidth)
        try:
            wav = torch.from_numpy(audio).float()
            if wav.ndim == 1: wav = wav.unsqueeze(0)  # 添加通道
            if wav.shape[0] == 1: wav = torch.cat([wav, wav], dim=0)  # 转为stereo如果mono
            wav = convert_audio(wav, self.sr, model.sample_rate, model.channels)
            wav = wav.unsqueeze(0)  # 添加batch
            encoded_frames = model.encode(wav)
            codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)  # [B, n_q, T]
            scales = [encoded[1] for encoded in encoded_frames]
            decoded_frames = model.decode(encoded_frames)
            result = torch.cat(decoded_frames, dim=-1).squeeze(0).numpy()
            return result.astype(np.float32)
        except Exception as e:
            logger.error(f"Encodec error (bw={bandwidth}): {e}")
            return audio
