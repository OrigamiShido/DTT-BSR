import argparse
import torch
import torchaudio
import sys
import os
import math
import glob
import torch.nn.functional as F
from tqdm import tqdm
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

def overlap_add_inference(model, audio, sample_rate, segment_length_sec=1.0, overlap=0.25, context_pad=4000,
                          device='cuda'):
    """
    内置的 Overlap-Add 推理函数 (带上下文填充修复版 + 维度修正)

    Args:
        context_pad (int): 每一侧的额外填充采样点数。
    """
    segment_length = int(segment_length_sec * sample_rate)
    hop_length = int(segment_length * (1 - overlap))

    # audio shape: [Channels, Time]
    length = audio.shape[-1]

    # 1. 预先对整个音频进行边缘填充
    pad_len_end = math.ceil(length / hop_length) * hop_length + segment_length - length

    # 2. 结合 Context Padding
    audio_padded = F.pad(audio, (context_pad, pad_len_end + context_pad), mode='reflect')

    output = torch.zeros((audio.shape[0], length + pad_len_end)).to(device)
    weight = torch.zeros((audio.shape[0], length + pad_len_end)).to(device)

    window = torch.hann_window(segment_length).to(device)
    if audio.shape[0] > 1:
        window = window.unsqueeze(0).repeat(audio.shape[0], 1)
    else:
        window = window.unsqueeze(0)

    model.eval()

    with torch.no_grad():
        start_offset = context_pad
        end_offset = audio_padded.shape[-1] - context_pad - segment_length + 1

        for i in range(0, length + pad_len_end - segment_length + 1, hop_length):
            chunk_start = i
            chunk_end = i + segment_length + 2 * context_pad

            if chunk_end > audio_padded.shape[-1]:
                break

            chunk = audio_padded[:, chunk_start:chunk_end]
            chunk_input = chunk.unsqueeze(0)

            # 构造 dummy target
            dummy_target = torch.zeros_like(chunk_input)

            # 模型推理
            chunk_output = model(chunk_input, dummy_target)

            if isinstance(chunk_output, (list, tuple)):
                chunk_output = chunk_output[0]

            # ---------------------------------------------------------
            # 维度修复：处理 [Batch, Sources, Channels, Time] -> [Channels, Time]
            # ---------------------------------------------------------
            # Demucs 典型输出为 4D: [1, 1, 1, 16000]
            # 我们需要将其压缩为与 output 切片一致的 2D: [1, 16000]

            # 1. 移除 Batch 维度
            if chunk_output.dim() == 4:
                chunk_output = chunk_output.squeeze(0)  # -> [Sources, Channels, Time]

            # 2. 移除 Source 维度 (如果存在且为1)
            if chunk_output.dim() == 3 and chunk_output.shape[0] == 1:
                chunk_output = chunk_output.squeeze(0)  # -> [Channels, Time]

            # 3. 兜底逻辑：如果仍然是 3D [1, 1, Time]
            target_channels = audio.shape[0]
            if chunk_output.numel() == target_channels * segment_length:
                chunk_output = chunk_output.reshape(target_channels, segment_length)

            # 累加结果
            output[:, i:i + segment_length] += chunk_output * window
            weight[:, i:i + segment_length] += window

    weight = torch.clamp(weight, min=1e-8)
    output = output / weight

    return output[..., :length]


class MSGProcessor:
    def __init__(self, msg_ckpt, msg_repo_path, msg_sr=16000, device='cuda'):
        self.device = device
        self.msg_sr = msg_sr

        # 加载 MSG 模型
        self.msg_model = self._load_msg_official(msg_ckpt, msg_repo_path)
        print(f">>> MSG model loaded (Internal SR: {self.msg_sr}Hz).")

    def _load_msg_official(self, ckpt_path, repo_path):
        print(f"Loading MSG model from {ckpt_path}...")

        if repo_path not in sys.path:
            sys.path.append(repo_path)

        try:
            from models.Demucs import Demucs

            # 使用官方 Bass 模型参数初始化
            model = Demucs(
                sources=["bass"],
                audio_channels=1,
                samplerate=self.msg_sr,
                segment_length=self.msg_sr,
                skip_cxn=True,
                lstm_layers=0,
                normalize=True
            )

            try:
                state_dict = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            except TypeError:
                state_dict = torch.load(ckpt_path, map_location=self.device)

            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            return model

        except ImportError as e:
            print(f"Error importing MSG Demucs model: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading MSG checkpoint: {e}")
            sys.exit(1)

    def process(self, input_path, output_path):
        # 1. 加载原始音频
        try:
            bass_audio, original_sr = torchaudio.load(input_path)
        except Exception as e:
            print(f"Error loading {input_path}: {e}")
            return

        bass_audio = bass_audio.to(self.device)

        # 2. 下采样 -> 16kHz
        if original_sr != self.msg_sr:
            resampler_in = torchaudio.transforms.Resample(original_sr, self.msg_sr).to(self.device)
            msg_input = resampler_in(bass_audio)
        else:
            msg_input = bass_audio

        # 3. 混合为单声道
        if msg_input.shape[0] > 1:
            msg_input_mono = torch.mean(msg_input, dim=0, keepdim=True)
        else:
            msg_input_mono = msg_input

        # 4. MSG 推理
        try:
            enhanced_bass_mono = overlap_add_inference(
                self.msg_model,
                msg_input_mono,
                self.msg_sr,
                segment_length_sec=1.0,
                context_pad=4000,
                device=self.device
            )
        except RuntimeError as e:
            print(f"RuntimeError processing {input_path}: {e}")
            return

        # 5. 恢复格式
        if bass_audio.shape[0] > 1:
            enhanced_bass_16k = enhanced_bass_mono.repeat(2, 1)
        else:
            enhanced_bass_16k = enhanced_bass_mono

        # 16kHz -> 原始采样率 (例如 44100)
        if self.msg_sr != original_sr:
            resampler_out = torchaudio.transforms.Resample(self.msg_sr, original_sr).to(self.device)
            final_output = resampler_out(enhanced_bass_16k)
        else:
            final_output = enhanced_bass_16k

        # 6. 长度对齐
        if final_output.shape[-1] != bass_audio.shape[-1]:
            if final_output.shape[-1] > bass_audio.shape[-1]:
                final_output = final_output[..., :bass_audio.shape[-1]]
            else:
                pad_len = bass_audio.shape[-1] - final_output.shape[-1]
                final_output = F.pad(final_output, (0, pad_len))

        # 7. 保存 (使用原始采样率)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torchaudio.save(output_path, final_output.cpu(), original_sr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MSG post-processing on existing bass audio files.")
    parser.add_argument("--input", required=True, help="Path to input bass wav/flac file OR directory")
    parser.add_argument("--output", required=True, help="Path to save output file OR directory")

    # MSG 参数
    parser.add_argument("--msg_ckpt", required=True, help="Official MSG .pth checkpoint")
    parser.add_argument("--msg_repo_path", required=True, help="Path to MSG code repository")
    parser.add_argument("--msg_sr", type=int, default=48000, help="MSG model sample rate (default 16000)")

    args = parser.parse_args()

    processor = MSGProcessor(
        args.msg_ckpt,
        args.msg_repo_path,
        msg_sr=args.msg_sr
    )

    if os.path.isdir(args.input):
        input_dir = args.input
        output_dir = args.output
        os.makedirs(output_dir, exist_ok=True)

        files = []
        for ext in ['*.flac', '*.wav', '*.mp3']:
            files.extend(glob.glob(os.path.join(input_dir, ext)))

        print(f"Found {len(files)} audio files in {input_dir}")

        for file_path in tqdm(files, desc="Processing files"):
            filename = os.path.basename(file_path)
            output_path = os.path.join(output_dir, filename)

            processor.process(file_path, output_path)

        print(f"Batch processing complete. Results saved to {output_dir}")

    else:
        processor.process(args.input, args.output)
        print(f"Processed single file: {args.input} -> {args.output}")