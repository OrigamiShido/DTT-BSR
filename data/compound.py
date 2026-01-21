
import random
from typing import Dict, Optional, List, Tuple

import torch
from torch.utils.data import Dataset

from .dataset import RawStems, MoisesDBStems, Musdb18HQStems
from .moises_db.moisesdb.dataset import MoisesDB


class CompoundDataset(Dataset):
    """
    顺序拼接多个子数据集：
    - __len__ 为所有子数据集长度之和
    - __getitem__(idx) 按顺序映射到对应子数据集的样本
    - 不再随机采样，也不再使用 probabilities
    """
    def __init__(
            self,
            MoisesDBParams,
            RawStemsParams,
            Musdb18HQStemsParams,
            **kwargs,
    ) -> None:
        super().__init__()

        self.MoisesDB=MoisesDB(**MoisesDBParams)
        self.RawStems=RawStems(**RawStemsParams)
        self.Musdb18HQStems=Musdb18HQStems(**Musdb18HQStemsParams)
        self.MoisesStems=MoisesDBStems(moises_db=self.MoisesDB, **MoisesDBParams)
        datasets={
            "raw": self.RawStems,
            "moises": self.MoisesStems,
            "musdb18": self.Musdb18HQStems,
        },

        if not datasets:
            raise ValueError("Provide at least one dataset.")
        self.datasets = datasets

        # 构建索引映射：[(dataset_key, local_idx_start, local_idx_end), ...]
        self.index_map: List[Tuple[str, int, int]] = []
        running = 0
        for key, ds in self.datasets.items():
            length = len(ds)
            if length <= 0:
                continue
            self.index_map.append((key, running, running + length))
            running += length

        if running == 0:
            raise ValueError("All provided datasets are empty.")

        self.total_length = running

    def __len__(self) -> int:
        return self.total_length

    def __getitem__(self, index: int):
        if index < 0 or index >= self.total_length:
            raise IndexError(f"Index {index} out of range for length {self.total_length}")

        # 找到 index 所在的数据集区间
        for key, start, end in self.index_map:
            if start <= index < end:
                local_idx = index - start
                dataset = self.datasets[key]
                return dataset[local_idx]

        # 理论上不会走到这里，防御性代码
        raise RuntimeError("Index mapping failed in CompoundDataset")


if __name__ == "__main__":
    moises_db = MoisesDB(
        data_path="/home/shihongtan/database/moisesdb",
        sample_rate=48000,
        quiet=True,
    )

    raw_dataset = RawStems(
        target_stem="Voc",
        root_directory="/home/shihongtan/database/RawStem/",
        sr=48000,
        clip_duration=3.0,
        snr_range=(0.0, 10.0),
    )

    moises_dataset = MoisesDBStems(
        target_stem="vocals",
        moises_db=moises_db,
        sr=48000,
        clip_duration=3.0,
        snr_range=(0.0, 10.0),
    )

    musdb18_dataset = Musdb18HQStems(
        target_stem="vocals",
        root_directory="/home/shihongtan/database/musdb18hq/",
        sr=48000,
        clip_duration=3.0,
        snr_range=(0.0, 10.0),
    )

    compound_dataset = CompoundDataset(
        datasets={
            "raw": raw_dataset,
            "moises": moises_dataset,
            "musdb18": musdb18_dataset,
        },
        probabilities=None,  # 不再使用
    )

    print("Total length:", len(compound_dataset))
    sample0 = compound_dataset[0]
    print(sample0["mixture"].shape, sample0["target"].shape)
