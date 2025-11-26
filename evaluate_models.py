# python evaluate_models.py \
#   --models_dir weights/singleplus \
#   --config config.yaml \
#   --val_dir data/val_set \
#   --mode suffix \
#   --suffixes DT1 DT2 \
#   --device cuda \
#   --metrics_device cuda \
#   --ranking_metric mel_snr

import argparse
import os
import re
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio
import yaml
from scipy.linalg import sqrtm
from tqdm import tqdm
import zimtohrli

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

warnings.filterwarnings("ignore")

try:
    from transformers import ClapModel, ClapProcessor
except ImportError as exc:  # pragma: no cover - informative message for missing dependency
    raise SystemExit(
        "The 'transformers' library is required for FAD-CLAP calculation.\n"
        "Install it via 'pip install transformers'."
    ) from exc

from models import MelRNN, MelRoFormer, UNet
from models.DTTNet.dp_tdf.dp_tdf_net import DPTDFNet

AUDIO_EXTS = (".flac", ".wav")
FAD_SAMPLE_RATE = 48000

@dataclass
class ValidationItem:
    key: str
    input_path: Path
    reference_path: Path
    suffix: str


def load_generator(config: Dict, checkpoint_path: Path, device: str) -> nn.Module:
    """Initialize generator based on config and load weights."""
    model_cfg = config["model"]

    if model_cfg["name"] == "MelRNN":
        generator = MelRNN.MelRNN(**model_cfg["params"])
    elif model_cfg["name"] == "MelRoFormer":
        generator = MelRoFormer.MelRoFormer(**model_cfg["params"])
    elif model_cfg["name"] == "MelUNet":
        generator = UNet.MelUNet(**model_cfg["params"])
    elif model_cfg["name"] == "DTTNet":
        generator = DPTDFNet(**model_cfg["params"])
    else:
        raise ValueError(f"Unknown model name: {model_cfg['name']}")

    state_dict = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(state_dict)
    generator = generator.to(device)
    generator.eval()
    return generator


def process_audio(audio: np.ndarray, generator: nn.Module, device: str) -> np.ndarray:
    """Run inference for a single audio array."""
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]

    audio_tensor = torch.from_numpy(audio).float().to(device)
    with torch.no_grad():
        output_tensor = generator(audio_tensor)
    return output_tensor.cpu().numpy()


def multi_mel_snr(reference: torch.Tensor, prediction: torch.Tensor, sr: int = FAD_SAMPLE_RATE) -> float:
    """Compute Multi-Mel-SNR between reference and prediction."""
    if not isinstance(reference, torch.Tensor):
        reference = torch.from_numpy(reference).float()
    if not isinstance(prediction, torch.Tensor):
        prediction = torch.from_numpy(prediction).float()

    alpha = torch.dot(reference, prediction) / (torch.dot(prediction, prediction) + 1e-8)
    prediction = alpha * prediction

    configs = [
        (512, 256, 80),
        (1024, 512, 128),
        (2048, 1024, 192),
    ]

    snrs = []
    for n_fft, hop, n_mels in configs:
        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            hop_length=hop,
            n_mels=n_mels,
            f_min=0,
            f_max=sr / 2,
            power=2.0,
        )
        m_ref = mel(reference)
        m_pred = mel(prediction)
        snr = 10 * torch.log10(m_ref.pow(2).sum() / ((m_ref - m_pred).pow(2).sum() + 1e-8))
        snrs.append(snr.item())

    return float(sum(snrs) / len(snrs))


def load_audio(file_path: Path, sr: int = FAD_SAMPLE_RATE) -> Optional[torch.Tensor]:
    try:
        wav, samplerate = sf.read(file_path)
        if wav.ndim > 1:
            wav = wav.T
        else:
            wav = wav[np.newaxis, :]
        if samplerate != sr:
            warnings.warn(f"{file_path} has sample rate {samplerate}, expected {sr}.")
        return torch.from_numpy(wav).float()
    except Exception as exc:
        warnings.warn(f"Failed to load {file_path}: {exc}")
        return None


def get_clap_embeddings(
    file_paths: Sequence[Path],
    model: ClapModel,
    processor: ClapProcessor,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Compute CLAP embeddings from audio files."""
    model.to(device)
    all_embeddings: List[np.ndarray] = []

    for idx in tqdm(
        range(0, len(file_paths), batch_size),
        desc="  CLAP embeddings",
        ncols=100,
        leave=False,
    ):
        batch_paths = file_paths[idx : idx + batch_size]
        audio_batch: List[np.ndarray] = []
        for path in batch_paths:
            try:
                wav, _ = sf.read(path)
                if wav.ndim == 2 and wav.shape[1] == 2:
                    audio_batch.append(wav[:, 0])
                    audio_batch.append(wav[:, 1])
                elif wav.ndim == 1:
                    audio_batch.append(wav)
            except Exception:
                continue

        if not audio_batch:
            continue

        inputs = processor(
            audios=audio_batch,
            sampling_rate=FAD_SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            audio_features = model.get_audio_features(**inputs)

        all_embeddings.append(audio_features.cpu().numpy())

    if not all_embeddings:
        return np.array([])
    return np.concatenate(all_embeddings, axis=0)


def calculate_frechet_distance(embeddings1: np.ndarray, embeddings2: np.ndarray) -> Optional[float]:
    if embeddings1.shape[0] < 2 or embeddings2.shape[0] < 2:
        return None

    mu1, mu2 = np.mean(embeddings1, axis=0), np.mean(embeddings2, axis=0)
    sigma1, sigma2 = np.cov(embeddings1, rowvar=False), np.cov(embeddings2, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2.0)

    try:
        covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    except Exception:
        return None

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return float(ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean))


def prepare_clap(metrics_device: Optional[str]) -> Tuple[Optional[ClapModel], Optional[ClapProcessor], Optional[torch.device]]:
    """Load CLAP artifacts once so they can be reused across models."""
    requested_device = metrics_device or ("cuda" if torch.cuda.is_available() else "cpu")
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        warnings.warn("CUDA requested for metrics but unavailable. Falling back to CPU.")
        requested_device = "cpu"

    device = torch.device(requested_device)
    try:
        # clap_model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
        # clap_processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
        clap_model = ClapModel.from_pretrained("./clap-model")
        clap_processor = ClapProcessor.from_pretrained("./clap-model")
        clap_model.eval()
        return clap_model, clap_processor, device
    except Exception as exc:
        warnings.warn(f"Failed to load CLAP model: {exc}. FAD-CLAP will be skipped.")
        return None, None, None


def discover_validation_items(
    val_dir: Path,
    target_dir: Path,
    mode: str,
    suffix_filter: Optional[Sequence[str]],
    single_file: Optional[str],
) -> List[ValidationItem]:
    """Collect validation entries depending on requested mode."""
    matched_suffixes = {suffix.upper() for suffix in suffix_filter or []}
    pattern = re.compile(r"^(?P<stem>.+)_DT(?P<tag>[A-Za-z0-9]+)$")
    items: List[ValidationItem] = []

    if mode == "single":
        if not single_file:
            raise ValueError("--file_name must be provided in single mode")
        candidate = Path(single_file)
        if not candidate.is_absolute():
            candidate = val_dir / candidate
        if not candidate.exists():
            raise FileNotFoundError(f"{candidate} not found in validation directory")
        stem, _ = os.path.splitext(candidate.name)
        match = pattern.match(stem)
        if not match:
            raise ValueError("Single mode expects filename ending with _DT{x}")
        suffix = f"DT{match.group('tag').upper()}"
        base_name = match.group("stem")
        ref_path = target_dir / f"{base_name}{candidate.suffix}"
        if not ref_path.exists():
            raise FileNotFoundError(f"Reference file {ref_path} not found")
        items.append(
            ValidationItem(
                key=f"{base_name}_{suffix}",
                input_path=candidate,
                reference_path=ref_path,
                suffix=suffix,
            )
        )
        return items

    for file_path in val_dir.iterdir():
        if not file_path.is_file() or file_path.suffix.lower() not in AUDIO_EXTS:
            continue
        stem = file_path.stem
        match = pattern.match(stem)
        if not match:
            continue
        suffix = f"DT{match.group('tag').upper()}"
        if matched_suffixes and suffix not in matched_suffixes:
            continue
        base_name = match.group("stem")
        ref_path = target_dir / f"{base_name}{file_path.suffix}"
        if not ref_path.exists():
            warnings.warn(f"Reference file missing for {file_path.name}, expected {ref_path.name}")
            continue
        items.append(
            ValidationItem(
                key=f"{base_name}_{suffix}",
                input_path=file_path,
                reference_path=ref_path,
                suffix=suffix,
            )
        )

    if not items:
        warnings.warn("No validation items discovered. Check suffix filters and directory contents.")
    return sorted(items, key=lambda item: (item.suffix, item.input_path.name))


def run_inference(
    generator: nn.Module,
    items: Sequence[ValidationItem],
    output_dir: Path,
    device: str,
) -> Dict[str, Path]:
    """Run inference for every validation item and persist outputs."""
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs: Dict[str, Path] = {}
    for item in tqdm(items, desc="Inference", ncols=100):
        audio, sr = sf.read(item.input_path)
        if audio.ndim == 2:
            audio = audio.T
        output_audio = process_audio(audio, generator, device)
        if output_audio.ndim == 2:
            output_audio = output_audio.T
        dest = output_dir / item.input_path.name
        sf.write(dest, output_audio, sr)
        outputs[item.key] = dest
    return outputs


def evaluate_metrics(
    items: Sequence[ValidationItem],
    outputs: Dict[str, Path],
    batch_size: int,
    ranking_metric: str,
    clap_model: Optional[ClapModel],
    clap_processor: Optional[ClapProcessor],
    clap_device: Optional[torch.device],
) -> Dict:
    """Compute Multi-Mel-SNR, zimtohrli, and FAD-CLAP."""
    metric = zimtohrli.Pyohrli()
    per_pair = []
    suffix_groups: Dict[str, List[Tuple[float, float]]] = {}
    target_paths: List[Path] = []
    output_paths: List[Path] = []

    for item in items:
        output_path = outputs.get(item.key)
        if not output_path or not output_path.exists():
            continue

        target_wav = load_audio(item.reference_path)
        output_wav = load_audio(output_path)
        if target_wav is None or output_wav is None:
            continue
        if target_wav.shape[0] != output_wav.shape[0]:
            min_channels = min(target_wav.shape[0], output_wav.shape[0])
            target_wav = target_wav[:min_channels]
            output_wav = output_wav[:min_channels]
        min_len = min(target_wav.shape[-1], output_wav.shape[-1])
        target_wav = target_wav[..., :min_len]
        output_wav = output_wav[..., :min_len]
        if min_len == 0:
            continue

        mel_scores = [
            multi_mel_snr(target_wav[ch], output_wav[ch]) for ch in range(target_wav.shape[0])
        ]
        avg_mel = float(sum(mel_scores) / len(mel_scores))

        zim_scores = []
        for ch in range(target_wav.shape[0]):
            zim_scores.append(
                metric.distance(
                    target_wav[ch].cpu().numpy(),
                    output_wav[ch].cpu().numpy(),
                )
            )
        avg_zim = float(sum(zim_scores) / len(zim_scores))

        suffix_groups.setdefault(item.suffix, []).append((avg_mel, avg_zim))
        per_pair.append(
            {
                "key": item.key,
                "input": str(item.input_path),
                "output": str(output_path),
                "mel_snr": avg_mel,
                "zimtohrli": avg_zim,
            }
        )
        target_paths.append(item.reference_path)
        output_paths.append(output_path)

    overall_mel = float(np.mean([p["mel_snr"] for p in per_pair])) if per_pair else None
    overall_zim = float(np.mean([p["zimtohrli"] for p in per_pair])) if per_pair else None

    fad_score = None
    if per_pair and clap_model and clap_processor and clap_device:
        target_embeddings = get_clap_embeddings(target_paths, clap_model, clap_processor, clap_device, batch_size)
        output_embeddings = get_clap_embeddings(output_paths, clap_model, clap_processor, clap_device, batch_size)
        if target_embeddings.size > 0 and output_embeddings.size > 0:
            fad_score = calculate_frechet_distance(target_embeddings, output_embeddings)

    suffix_summary = {
        suffix: {
            "mel_snr": float(np.mean([m for m, _ in scores])),
            "zimtohrli": float(np.mean([z for _, z in scores])),
        }
        for suffix, scores in suffix_groups.items()
    }

    overall = {
        "mel_snr": overall_mel,
        "zimtohrli": overall_zim,
        "fad_clap": fad_score,
    }

    score_for_ranking = None
    if overall[ranking_metric] is not None:
        score_for_ranking = overall[ranking_metric]

    return {
        "overall": overall,
        "per_suffix": suffix_summary,
        "pairs": per_pair,
        "ranking_value": score_for_ranking,
    }


def build_metric_rankings(leaderboard: Sequence[Tuple[str, Dict]]) -> Dict[str, List[Tuple[str, float]]]:
    """Return sorted rankings for each metric present in the leaderboard."""
    metric_specs = {
        "mel_snr": True,   # higher is better
        "zimtohrli": False,  # lower is better
        "fad_clap": False,   # lower is better
    }
    rankings: Dict[str, List[Tuple[str, float]]] = {}

    for metric, higher_is_better in metric_specs.items():
        entries: List[Tuple[str, float]] = []
        for model_name, metrics in leaderboard:
            value = metrics["overall"].get(metric)
            if value is not None:
                entries.append((model_name, float(value)))
        entries.sort(key=lambda item: item[1], reverse=higher_is_better)
        rankings[metric] = entries

    return rankings

def load_existing_outputs(
        items: Sequence[ValidationItem],
        output_dir: Path,
) -> Dict[str, Path]:
    """Reuse pre-generated outputs for metric-only runs."""
    outputs: Dict[str, Path] = {}
    if not output_dir.exists():
        warnings.warn(f"Output directory {output_dir} missing; metrics-only run will skip this checkpoint.")
        return outputs
    for item in items:
        candidate = output_dir / item.input_path.name
        if candidate.exists():
            outputs[item.key] = candidate
        else:
            warnings.warn(f"Missing output for {item.input_path.name} in {output_dir}.")
    return outputs

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate multiple checkpoints on a validation set by running inference and "
            "computing Multi-Mel-SNR, zimtohrli, and FAD-CLAP."
        )
    )
    parser.add_argument("--models_dir", required=True, help="Directory containing .pth checkpoints")
    parser.add_argument("--config", required=True, help="Path to model config YAML")
    parser.add_argument("--val_dir", required=True, help="Directory holding validation audio")
    parser.add_argument("--target_dir", required=True, help="Directory holding target/reference audio")
    parser.add_argument(
        "--mode",
        choices=["suffix", "single"],
        default="suffix",
        help="suffix: evaluate all *_DTx files. single: evaluate one specific file.",
    )
    parser.add_argument(
        "--suffixes",
        nargs="*",
        help="Optional list of suffix tokens (e.g., DT1 DT2) to evaluate in suffix mode.",
    )
    parser.add_argument(
        "--file_name",
        help="File name (relative to val_dir) to evaluate in single mode. Must end with _DTx.",
    )
    parser.add_argument("--device", default="cuda", help="Device for running generators.")
    parser.add_argument(
        "--ranking_metric",
        choices=["mel_snr", "zimtohrli", "fad_clap"],
        default="mel_snr",
        help="Metric used to pick the best model.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for CLAP embeddings.",
    )
    parser.add_argument(
        "--metrics_device",
        help="Device for CLAP embeddings (default: cuda if available else cpu).",
    )
    parser.add_argument(
        "--work_dir",
        default="eval_outputs",
        help="Directory to place intermediate model outputs.",
    )
    parser.add_argument(
        "--pipeline_mode",
        choices=["full", "metrics_only"],
        default="full",
        help="full: 重新推断再评估；metrics_only: 复用 work_dir 中已存在的输出，仅计算指标。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models_dir = Path(args.models_dir)
    val_dir = Path(args.val_dir)
    target_dir= Path(args.target_dir)
    work_dir = Path(args.work_dir)

    if not models_dir.is_dir():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")
    if not val_dir.is_dir():
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")
    if args.pipeline_mode == "metrics_only" and not work_dir.exists():
        raise FileNotFoundError(f"Work directory not found for metrics-only mode: {work_dir}")

    with open(args.config, "r", encoding="utf-8") as cfg_file:
        config = yaml.safe_load(cfg_file)

    items = discover_validation_items(val_dir,target_dir, args.mode, args.suffixes, args.file_name)
    if not items:
        raise SystemExit("No validation items available. Nothing to evaluate.")

    checkpoints = sorted(models_dir.glob("*.pth"))
    if not checkpoints:
        raise SystemExit("No checkpoints found in models_dir")

    work_dir.mkdir(parents=True, exist_ok=True)

    clap_model, clap_processor, clap_device = prepare_clap(args.metrics_device)

    leaderboard = []
    best_entry = None

    for checkpoint in checkpoints:
        print(f"\n=== Evaluating {checkpoint.name} ===")
        model_output_dir = work_dir / checkpoint.stem
        generator: Optional[nn.Module] = None
        if args.pipeline_mode == "full":
            generator = load_generator(config, checkpoint, device=args.device)
            outputs = run_inference(generator, items, model_output_dir, args.device)
        else:
            outputs = load_existing_outputs(items, model_output_dir)
            if not outputs:
                print("  No usable outputs; skipping this checkpoint.")
                continue
        metrics = evaluate_metrics(
            items,
            outputs,
            args.batch_size,
            args.ranking_metric,
            clap_model,
            clap_processor,
            clap_device,
        )
        leaderboard.append((checkpoint.name, metrics))

        ranking_value = metrics.get("ranking_value")
        if ranking_value is None:
            print("  Skipped ranking (metric unavailable).")
        else:
            if args.ranking_metric in {"mel_snr"}:
                better = best_entry is None or ranking_value > best_entry[2]
            else:  # zimtohrli and fad_clap are distances (lower is better)
                better = best_entry is None or ranking_value < best_entry[2]
            if better:
                best_entry = (checkpoint.name, metrics, ranking_value)
        if generator is not None:
            del generator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n=== Leaderboard ===")
    for model_name, metrics in leaderboard:
        overall = metrics["overall"]
        mel = overall["mel_snr"]
        zim = overall["zimtohrli"]
        fad = overall["fad_clap"]
        print(
            f"{model_name:30s} | Mel-SNR: {mel if mel is not None else 'n/a':>8} | "
            f"zimtohrli: {zim if zim is not None else 'n/a':>8} | "
            f"FAD: {fad if fad is not None else 'n/a':>8}"
        )

    if best_entry:
        print(
            f"\nBest model by {args.ranking_metric}: {best_entry[0]} (score={best_entry[2]:.4f})"
        )
    else:
        print("\nUnable to determine best model because ranking metric was unavailable for all.")

    rankings = build_metric_rankings(leaderboard)
    print("\n=== Metric Rankings ===")
    metric_labels = {
        "mel_snr": "Multi-Mel-SNR (higher better)",
        "zimtohrli": "zimtohrli (lower better)",
        "fad_clap": "FAD-CLAP (lower better)",
    }
    for metric in ["mel_snr", "zimtohrli", "fad_clap"]:
        entries = rankings.get(metric, [])
        label = metric_labels.get(metric, metric)
        if not entries:
            print(f"{label}: n/a")
            continue
        print(f"{label}:")
        for rank, (model_name, value) in enumerate(entries, start=1):
            print(f"  #{rank:2d} {model_name:<30s} {value:.4f}")


if __name__ == "__main__":
    main()
