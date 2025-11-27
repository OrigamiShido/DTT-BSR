# DTT-BSR

Team AC/DC (Wuhan University)

ICASSP 2026 Music Source Restoration Challenge

## System Description

### Directory Structure

Our system builds on the official MSR baseline GAN framework. The discriminator architecture and loss formulation are kept identical to the baseline; all architectural changes are confined to the generator’s time–frequency modeling.

The official baseline implementation is provided in the MSRKit repository. In our submission, the baseline generator is replaced by a DTTNet-based architecture adapted from prior work [2]. On the generator side, we adopt this DTTNet-style time–frequency U-Net operating on complex STFTs, and further augment it at the bottleneck with a BandSplitRNN-inspired band-sequence recurrent module and a RoPE-based Transformer to jointly capture long-range dependencies along both time and frequency while preserving the original dual-path TFC‑TDF structure. We refer to this enhanced generator as **DTT-BSR** (DTTNet with BandSequence and RoPE).

On the discriminator side, we reuse the baseline Multi-Frequency Discriminator and the standard combination of reconstruction, adversarial, and feature-matching losses. 

The repository directory structure is:

`MSRKit/`

- `README.md` <- You are here
- `config.yaml` <- Main configuration file for experiments
- `train.py` <- Main script to start training
- `unwrap.py` <- Utility to extract generator weights from a checkpoint
- `data/` <- [Data loading and augmentation](https://github.com/OrigamiShido/MSRKit-WHU/blob/main/data/README.md)
- `losses/` <- [Loss function implementations](https://github.com/OrigamiShido/MSRKit-WHU/blob/main/losses/README.md)
- `models/` <- [Top-level generator model architectures](https://github.com/OrigamiShido/MSRKit-WHU/blob/main/models/README.md)
- `modules/`<-Core building blocks for models
    - `discriminator/` <- Discriminator architectures
    - `generator/` <- Reusable generator components

## Methodology

### Model Description

We inherit the baseline GAN framework, only modified the generator and discriminator settings.

#### generator

![system](./system.png)

We build our generator on top of DTTNet [2], a lightweight dual-path TFC‑TDF U‑Net originally proposed for music source separation, and adapt it to the MSR setting with additional sequence-modeling modules; the resulting architecture is what we call **DTT-BSR**:

- **Time–Frequency Front-End**  
  Input waveforms are transformed into complex STFTs (e.g., `n_fft = 2048`, `hop = 512`). Real and imaginary parts are treated as separate channels, which allows the model to jointly refine magnitude and phase.

- **TFC‑TDF Encoder–Decoder**  
  The backbone consists of stacked TFC_TDF_Res2 blocks that alternate between temporal–frequency convolution (TFC) and time-distributed depthwise filtering (TDF). This structure captures both local time–frequency patterns and broader spectral correlations while remaining parameter-efficient.

- **Band-Sequence Modeling (Across Frequency and Time)**  
  At the bottleneck, we feed the encoded features into a Improved Dual-Path module derived from BandSplitRNN. The frequency axis is viewed as multiple subbands, and grouped bi-directional RNNs are applied alternately along the time and subband dimensions, with residual connections and group normalization. This design explicitly models correlations across subbands and over time, which is particularly important for non-vocal stems where harmonic structures and broadband artifacts are tightly coupled.

- **RoPE Transformer Bottleneck**  
  At the bottleneck, we insert a RoPE-based Transformer block. Rotary Position Embeddings enable the attention mechanism to handle long sequences while preserving relative phase information. This module lets the generator attend over long time spans and frequency ranges, improving temporal consistency and phase reconstruction.

- **Waveform Reconstruction**  
  The decoder mirrors the encoder with learned upsampling and skip connections. The network predicts a complex-valued residual or mask over the STFT, which is then transformed back to the waveform domain using inverse STFT.

All architectural details (number of blocks, hidden dimensions, LSTM depth, attention heads, etc.) are configurable via `config.yaml`.

#### Discriminator and Objectives

We use the baseline **Multi-Frequency Discriminator (MFD)**:

- Multiple sub-discriminators, each operating on STFTs computed with different window sizes, analyze the signal at several time–frequency resolutions.
- Real and generated signals are passed through the same MFD ensemble, and their outputs as well as intermediate feature maps are used for adversarial and feature-matching losses.

We keep the discriminator architecture, its hyperparameters, and the loss formulation (LSGAN-style adversarial loss, multi-scale mel reconstruction, and feature matching with baseline weighting) identical to the official MSR baseline, so that performance gains can be attributed to the generator enhancements.

The training objective combines:

- **Multi-scale mel reconstruction loss** to enforce fidelity in the time–frequency domain.
- **LSGAN adversarial loss** to encourage perceptually realistic outputs.
- **Feature-matching loss** to stabilize training and align internal discriminator statistics.

The relative weights of these terms are chosen to prioritize reconstruction quality while preserving adversarial sharpness; the exact settings are specified in `config.yaml`.

### Dataset and Trining Protocol

- **Dataset**  
  We train on the RawStem[5] dataset, resampled to 48 kHz. Training examples are 3-second clips constructed by mixing target stems with distractor stems at random SNRs in the range [0, 10] dB. The target stem (e.g., `Voc`) and the root directory of RawStem are set via `data.*` entries in `config.yaml`.

- **Augmentation and Sampling**  
  We optionally apply stem- and mixture-level augmentations (e.g., gain perturbation, random mixing) to improve robustness. Clip start times are sampled such that the selected segments are non-silent according to precomputed RMS statistics when available.

- **Optimization**  
  The model is trained with AdamW and a warm-up schedule, using mixed precision and frequent checkpointing. A single set of hyperparameters is shared across all target stems (no target-specific tuning). Training is implemented in PyTorch Lightning (`train.py`), which also manages logging and checkpoint saving under `./experiment/{project}/{model}/`.

## Public Code and Pretrained Weights

- **Code repository**  
  MSRKit-WHU implementation, training pipeline, and evaluation scripts:  
  https://github.com/OrigamiShido/DTT-BSR

- **Pretrained models and configs**  
  Pretrained generator weights and the exact configuration files used for submission:  
  https://huggingface.co/OrigamiShido/MSRChallenge-ACDC

## Reproduction Guide

The steps below reproduce our system and evaluation results.

### Environment Setup

```bash
git clone https://github.com/OrigamiShido/DTT-BSR.git
cd DTT-BSR
pip install -r requirements.txt

# Optional: install CLAP for FAD-CLAP metric
pip install laion-clap
```

### Data Preparation

1. Download the RawStem dataset and place it under a directory of your choice, e.g. `/path/to/RawStem`.
2. Edit `config.yaml`:
   - Set `data.train_dataset.root_directory` to `/path/to/RawStem`.
   - Set `data.train_dataset.target_stem` to the desired stem (e.g. `"Voc"`).
   - Adjust batch size, number of workers, and augmentation options as needed.

### Training from Scratch (Optional)

To train the model from scratch with the provided configuration:

```bash
python train.py --config config.yaml
```

Checkpoints will be saved under:

```text
./experiment/{project_name}/{model_name}/checkpoints/
```

To export the generator-only weights from a Lightning checkpoint:

```bash
python unwrap.py # need to modify the unwrap.py to change the path
```

### Inference with Pretrained Models

You can either use the pretrained generators from Hugging Face or a generator exported via `unwrap.py`:

1. Visit our Hugging Face repository: https://huggingface.co/OrigamiShido/MSRChallenge-ACDC  
   Download the generator weights (e.g. `generator.pth`) and the corresponding `config.yaml` used for submission, or use your own exported weights from the training step.
2. Point `--checkpoint` to the downloaded/unwrapped generator file, and `--config` to the matching configuration file.

```bash
python inference.py \
  --config config.yaml \
  --checkpoint path/to/generator.pth \
  --input_dir path/to/input_flac_dir \
  --output_dir path/to/output_dir \
```

All `.flac` files in `input_dir` will be processed and saved to `output_dir` with the same filenames.

### Evaluation

To evaluate SI-SNR and FAD-CLAP:

1. Create a text file (e.g. `file_list.txt`) where each line has the format:

```text
/path/to/target.wav|/path/to/output.wav
```

2. Run the evaluation script:

```bash
python calculate_metrics.py file_list.txt --batch_size 16
```

The script prints per-pair SI-SNR scores, then computes the overall FAD-CLAP distance between the set of target and generated files (downloading CLAP weights if needed).

### Model Selection (Best Checkpoint)

For model selection on a validation set, we evaluate multiple checkpoints and rank them using a chosen metric (e.g. mel SNR). Assuming `evaluate_models.py` is available in the project root:

```bash
python evaluate_models.py \
  --models_dir path/to/the/checkpoint_dir/ \
  --config path/to/the/config.yaml \
  --val_dir path/to/the/mixture/dir/ \
  --target_dir /path/to/the/target/dir \
  --mode suffix \
  --suffixes DT0 \
  --device cuda \
  --metrics_device cuda \
  --ranking_metric mel_snr \
  --work_dir evaluate/save/dir \
  --pipeline_mode full > result.txt
```

This script scores all specified checkpoints on the validation set, ranks them by the selected metric, and writes detailed results (including the best-performing model) to `result.txt`.

---

## Acknowledgements

We would like to thank Prof. Gongping Huang, Prof. Zhongqiu Wang, Dr. Yuzhu Wang, and Dr. Haohe Liu for their guidance and support throughout this work.

---

## References

[1] W.-T. Lu, J.-C. Wang, Q. Kong, and Y.-N. Hung, “Music Source Separation with Band-Split RoPE Transformer,” arXiv:2309.02612.  
[2] J. Chen, S. Vekkot, and P. Shukla, “DTTNET: DUAL-PATH TFC-TDF UNET,” ICASSP 2024.  
[3] Y. Zang, Z. Dai, M. D. Plumbley, and Q. Kong, “Music Source Restoration,” arXiv:2505.21827.  
[4] Yun-Ning, Hung, I. Pereira, and F. Korzeniowski, “Moises-Light: Resource-efficient Band-split U-Net,” arXiv:2510.06785.
[5] Y. Zang, Z. Dai, M. D. Plumbley, and Q. Kong, “Music Source Restoration,” arXiv: arXiv:2505.21827. doi: 10.48550/arXiv.2505.21827.


