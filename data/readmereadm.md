# 使用方案：

## 流程概述
该流程包含四个核心脚本：

- **generate_rms_analysis.py**：(预处理) 这是一个一次性脚本。它扫描训练数据集，预先计算所有音频的响度 (RMS)，并生成一个文件 rms_analysis.jsonl。

- **preprocess.py**：(预处理) 这是一个关键脚本，用于将动态数据集转换为静态预处理样本。它提前执行昂贵的混合和 MSRBench 退化 (MixtureAugmentation)，并保存为 FLAC 文件对（mixture 和 target）。这极大加速训练 I/O，尤其在多进程模式下。支持 --split 参数以确保 train/test 分离。

- **augment.py**：(核心增强) 定义了两种增强类：
  - StemAugmentation：对干净的目标音轨（如人声）模拟录音室制作效果。
  - MixtureAugmentation：对混合后的音频模拟 MSRBench 论文中定义的 12 种退化情况（DT1-DT12，包括 FM 广播、磁带、黑胶、MP3 压缩等）。DT0 为原始混合（无退化）。

- **dataset.py**：(数据加载器) 包含核心的类 RawStemsDataset。它有两种工作模式：
  - 训练模式 (is_validation=False)：利用 rms_analysis.jsonl 进行智能“非静音”采样。它动态地从训练数据中抓取一个目标目标音轨和多个“其他”音轨，应用 StemAugmentation，将它们混合，最后应用 MixtureAugmentation 来生成数据对 (退化混合, 干净目标)。如果使用预处理数据，可切换到 PreprocessedRawStems 类以直接加载静态样本。
  - 验证模式 (is_validation=True)：专门用于加载 MSRBench 数据集。它会精确加载 MSRBench 提供的 10 秒钟 mixture 和 targets 音频对。

## 详细使用步骤
### 步骤 1：环境与依赖配置
在运行代码前，请确保安装了所有必要的 Python 库和外部工具：

**Python 库**：
```
pip install numpy scipy librosa soundfile pedalboard pyroomacoustics torch torchaudio encodec descript-audio-codec git+https://github.com/descriptinc/audiotools tqdm concurrent.futures
```

**外部工具**：
- FFmpeg：必须安装并添加到系统 PATH。augment.py 中的 _apply_codec 函数会调用 ffmpeg 来模拟 AAC 和 MP3 压缩（DT5-DT8）。

**外部数据 (用于增强，用于某些 DT 的真实模拟)**：必须下载以下数据，并编辑 augment.py 脚本以指向它们的路径：
- WHAM! 噪声：用于 DT4（现场录音）模拟环境噪声。从 http://wham.whisper.ai/ 下载 48kHz 版本。
  - 编辑 augment.py 中的 `WHAM_NOISE_DIR = 'path/to/wham_noise/'`
- 黑胶噪声：用于 DT3（黑胶）模拟噼啪声。
  - 编辑 augment.py 中的 `VINYL_CRACKLE_PATH = 'path/to/vinyl_crackle.wav'`


### 步骤 2：准备数据集文件结构
需要为训练和验证准备两种不同的目录结构。

A. **训练数据集**
训练数据必须组织成 RawStems 格式。generate_rms_analysis.py、preprocess.py 和 dataset.py 期望的结构如下：

```
/path/to/your/training_data/  <-- (这是你的 DATASET_ROOT)
├── Track00001/
│   ├── vocals/
│   │   └── stem.flac
│   ├── bass/
│   │   └── stem.flac
│   ├── drums/
│   │   └── stem.flac
│   └── other/
│       └── stem.flac
├── Track00002/
│   ├── vocals/
│   │   └── stem.flac
│   ├── guitar/
│   │   └── stem.flac
│   ...
...
```

B. **验证数据集 (MSRBench)**
必须下载并解压 MSRBench 数据集（https://huggingface.co/datasets/yongyizang/MSRBench，8 个 ZIP 文件，总解压后 28.7 GB）。在验证模式下 dataset.py 期望的结构如下（注意：root_directory 将指向单个乐器的文件夹，例如 `/path/to/MSRBench_unzipped/Vocals`）：

```
/path/to/your/MSRBench_unzipped/
├── Vocals/
│   ├── mixture/
│   │   ├── {song_id}_DT0.flac
│   │   ├── {song_id}_DT1.flac
│   │   ...
│   │   └── {song_id}_DT12.flac
│   └── targets/
│       ├── {song_id}.flac
│       ├── {song_id}.flac
│       ...
├── Bass/
│   ├── mixture/
│   │   ...
│   └── targets/
│       ...
... (其他乐器，如 Guitars, Keyboards, Synthesizers, Drums, Percussions, Orchestral Elements) ...
```

- 下载：使用 Hugging Face CLI: `huggingface-cli download yongyizang/MSRBench --repo-type dataset --local-dir /path/to/msrbench`。或使用 datasets 库: `datasets.load_dataset('yongyizang/MSRBench', download_mode='force_redownload')`。解压每个 {Stem_Name}.zip 文件。
- **新添加**：MSRBench 包含 250 个 10 秒钟剪辑，每个剪辑有 13 个版本（DT0 为原始混合，DT1-DT12 为退化）。退化类型分为三类：
  - **模拟退化 (DT1-DT4)**：DT1 (广播) - 使用 GNU Radio 模拟 FM 广播，包括 Rayleigh 衰落、多径和噪声；DT2 (磁带) - 使用 Klevgrand DAW Cassette 插件模拟磁带着色和噪声；DT3 (黑胶) - 使用 iZotope Vinyl 插件模拟 1970 年代的噼啪声和机械噪声；DT4 (现场声音) - 使用 PyRoomAcoustics 的脉冲响应、手机麦克风带通滤波，以及 WHAM! 数据集的 48 kHz 环境噪声（约 20 dB SNR）。
  - **传统丢失音频编解码器 (DT5-DT8)**：模拟 MP3、AAC 等压缩伪影。
  - **神经音频编解码器 (DT9-DT12)**：模拟现代 AI 编解码器，如 Encodec 在低比特率（DT12 为 3 Kbps Encodec）。

### 步骤 3：运行预处理 (使用 preprocess.py)
此步骤适用于训练数据集，用于生成静态预处理样本。

打开 preprocess.py 脚本。

修改参数以匹配你的设置，例如：
- --root-directory：指向步骤 2A 中的训练数据路径。
- --target-stem：目标音轨（如 'Vocals'）。
- --output-directory：预处理样本的保存路径（例如 /path/to/preprocessed_train）。
- --split：使用 'train' 或 'test' 以基于 data_splits TXT 文件过滤（确保先运行 data_split 脚本生成 TXT 文件）。

运行脚本：
```
python preprocess.py --root-directory /path/to/your/training_data --target-stem Vocals --output-directory /path/to/preprocessed_train --num-samples 100000 --sr 48000 --clip-duration 4.0 --split train --max-workers 8
```
脚本将生成 FLAC 文件对（e.g., sample_0000001_mix.flac 和 sample_0000001_target.flac）。这预计算了昂贵的混合和退化（MixtureAugmentation），加速后续训练。重复运行以生成 validation 或 test 预处理数据（更改 --split 和 --output-directory）。

**新添加**：强烈推荐先预处理再训练，以处理 RawStems 的质量问题并提升效率。多进程 (--max-workers) 可显著缩短时间（取决于 CPU 核心数）。

### 步骤 4：(一次性) 运行 RMS 预分析
此步骤仅适用于原始训练数据集（如果不使用预处理数据）。

打开 generate_rms_analysis.py 脚本。

修改 DATASET_ROOT 变量，使其指向在步骤 2A 中准备的训练数据路径（例如 `/path/to/your/training_data/`）。

运行脚本：
```
python generate_rms_analysis.py
```
脚本将遍历所有音频文件，并在 DATASET_ROOT 目录下生成一个名为 rms_analysis.jsonl 的文件。dataset.py 会自动查找并使用这个文件来进行非静音采样。

**新添加**：如果 RawStems 数据有泄漏问题，此步骤有助于过滤低质量剪辑。如果使用预处理数据，此步骤可选，因为 preprocess.py 已内置 RMS 检查。

### 步骤 5：集成与训练
- **集成 MSRKit**：复制 dataset.py、augment.py 和 preprocess.py 到 MSRKit/data/。修改 MSRKit/config.yaml 的 data 部分（如上示例）。如果使用预处理数据，在 config.yaml 中指定 PreprocessedRawStems 类和预处理路径。
- **训练**：运行 `python train.py --config config.yaml`。监控日志，确保增强应用 (e.g., DT 类型随机)。
- **验证**：在 config.yaml 设置 validation_dataset，使用 calculate_metrics.py 评估 (生成 eval_list.txt: target_path|output_path)。

### 步骤 6：测试与评估
- **独立测试**：`from dataset import RawStemsDataset; dataset = RawStemsDataset('vocals', '/path/to/training_data', is_validation=False); sample = dataset[0]`。检查 sample['mixture'].shape == (2, 192000) (4s @48kHz)。如果使用预处理：`from dataset import PreprocessedRawStems; dataset = PreprocessedRawStems('/path/to/preprocessed_train'); sample = dataset[0]`。
- **MSRBench 验证**：`dataset = RawStemsDataset('Vocals', '/path/to/MSRBench_unzipped/Vocals', is_validation=True, validation_dt_ids=[0,1,2])`。sample = dataset[0] 应为 10s clip。
- **指标**：用 MSRKit evaluate.py 计算 SI-SNR/FAD-CLAP。
