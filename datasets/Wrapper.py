import torch
import numpy as np
import librosa
import os
import glob
from torch.utils.data import Dataset


class DatasetWrapper(Dataset):
    def __init__(self, path_to_ds, target, mix_folder='mixture',
                 sample_rate=44100, segment_dur=1.0,
                 mono=True, gt_percent=0.0, silent_percent=0.0, mix_percent=0.0, **kwargs):
        """
        [独立版] MSG 微调训练集加载器
        返回: (dirty, clean, dirty, filename)
        """
        self.root = path_to_ds
        self.target = target
        self.mix_folder = mix_folder
        self.sample_rate = sample_rate
        self.segment_samples = int(segment_dur * sample_rate)
        self.mono = mono

        # 增强参数
        self.gt_percent = gt_percent if gt_percent < 1 else gt_percent / 100
        self.silent_percent = silent_percent + self.gt_percent if silent_percent < 1 else silent_percent / 100 + self.gt_percent
        self.mix_percent = mix_percent

        self.mix_dir = os.path.join(self.root, self.mix_folder)
        self.target_dir = os.path.join(self.root, self.target)

        self.filenames = []
        for ext in ['*.wav', '*.flac']:
            self.filenames.extend(
                [os.path.basename(x) for x in glob.glob(os.path.join(self.mix_dir, ext))]
            )

        self.valid_files = []
        for f in self.filenames:
            if os.path.exists(os.path.join(self.target_dir, f)):
                self.valid_files.append(f)

        print(f"DatasetWrapper: Found {len(self.valid_files)} valid pairs in {self.root}")

    def _load_audio(self, path):
        try:
            audio, _ = librosa.load(path, sr=self.sample_rate, mono=self.mono)
        except Exception:
            return np.zeros((1, self.segment_samples), dtype=np.float32)

        if audio.ndim == 1:
            audio = audio[np.newaxis, :]
        return audio

    def __getitem__(self, index):
        filename = self.valid_files[index]
        mix_path = os.path.join(self.mix_dir, filename)
        target_path = os.path.join(self.target_dir, filename)

        # 1. 加载
        mix_audio = self._load_audio(mix_path)
        clean_audio = self._load_audio(target_path)

        # 2. 截断
        min_len = min(mix_audio.shape[1], clean_audio.shape[1])
        mix_audio = mix_audio[:, :min_len]
        clean_audio = clean_audio[:, :min_len]

        # 3. 随机切片
        if min_len > self.segment_samples:
            start = np.random.randint(0, min_len - self.segment_samples)
            end = start + self.segment_samples
            mix_cut = mix_audio[:, start:end]
            clean_cut = clean_audio[:, start:end]
        else:
            pad_len = self.segment_samples - min_len
            mix_cut = np.pad(mix_audio, ((0, 0), (0, pad_len)))
            clean_cut = np.pad(clean_audio, ((0, 0), (0, pad_len)))

        # 4. 转 Tensor
        dirty_tensor = torch.from_numpy(mix_cut).float()
        clean_tensor = torch.from_numpy(clean_cut).float()

        # 5. 数据增强
        rand_val = np.random.uniform()
        if rand_val < self.gt_percent:
            dirty_tensor = clean_tensor.clone()
        elif rand_val < self.silent_percent:
            dirty_tensor = torch.zeros_like(dirty_tensor)

        # 6. 压缩维度 (适配 RunEpoch 的 ensure_3d)
        if self.mono:
            dirty_tensor = dirty_tensor.squeeze(0)
            clean_tensor = clean_tensor.squeeze(0)

        # 返回 4 个值，保持接口一致
        return dirty_tensor, clean_tensor, dirty_tensor, filename

    def __len__(self):
        return len(self.valid_files)