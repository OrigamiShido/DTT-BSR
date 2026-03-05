# import argparse
# import os
# import torch
# import torchaudio
# import numpy as np
# import warnings
# import glob
# from scipy.linalg import sqrtm
# from tqdm import tqdm
# os.environ['CUDA_VISIBLE_DEVICES'] = '4'
# # Filter warnings
# warnings.filterwarnings("ignore")
#
# # === 尝试导入 Zimtohrli ===
# try:
#     import zimtohrli
#
#     ZIMTOHRLI_AVAILABLE = True
# except ImportError:
#     ZIMTOHRLI_AVAILABLE = False
#     print("Warning: 'zimtohrli' library not found. Zimtohrli scores will be skipped.")
#
# # === 尝试导入 Transformers (用于 FAD-CLAP) ===
# try:
#     from transformers import ClapModel, ClapProcessor
#
#     TRANSFORMERS_AVAILABLE = True
# except ImportError:
#     TRANSFORMERS_AVAILABLE = False
#     print("Warning: 'transformers' library not found. FAD-CLAP will be skipped.")
#
#
# def multi_mel_snr(reference, prediction, sr=48000):
#     """
#     Compute Multi-Mel-SNR between reference and prediction (Scale-Invariant).
#     """
#     if not isinstance(reference, torch.Tensor):
#         reference = torch.from_numpy(reference).float()
#     if not isinstance(prediction, torch.Tensor):
#         prediction = torch.from_numpy(prediction).float()
#
#     if reference.device != prediction.device:
#         prediction = prediction.to(reference.device)
#
#     # Scale-invariant normalization
#     dot_ref_pred = torch.sum(reference * prediction)
#     dot_pred_pred = torch.sum(prediction * prediction)
#     alpha = dot_ref_pred / (dot_pred_pred + 1e-8)
#     prediction = alpha * prediction
#
#     configs = [
#         (512, 256, 80),
#         (1024, 512, 128),
#         (2048, 1024, 192)
#     ]
#
#     snrs = []
#     for n_fft, hop, n_mels in configs:
#         mel_transform = torchaudio.transforms.MelSpectrogram(
#             sample_rate=sr, n_fft=n_fft, hop_length=hop,
#             n_mels=n_mels, f_min=0, f_max=sr // 2, power=1.0
#         ).to(reference.device)
#
#         M_ref = mel_transform(reference)
#         M_pred = mel_transform(prediction)
#
#         ref_energy = M_ref.pow(2).sum()
#         noise_energy = (M_ref - M_pred).pow(2).sum()
#
#         snr = 10 * torch.log10(ref_energy / (noise_energy + 1e-8))
#         snrs.append(snr.item())
#
#     return sum(snrs) / len(snrs)
#
#
# def load_audio(file_path, sr=48000):
#     try:
#         wav, samplerate = torchaudio.load(file_path)
#         if samplerate != sr:
#             resampler = torchaudio.transforms.Resample(samplerate, sr)
#             wav = resampler(wav)  # 实现重采样
#
#         return wav
#     except Exception as e:
#         print(f"Error loading {file_path}: {e}")
#         return None
#
#
# def get_clap_embeddings(file_paths, model, processor, device, batch_size=16):
#     model.to(device)
#     model.eval()
#     all_embeddings = []
#
#     for i in tqdm(range(0, len(file_paths), batch_size), desc="Calculating CLAP embeddings", leave=False):
#         batch_paths = file_paths[i:i + batch_size]
#         audio_batch = []
#
#         for path in batch_paths:
#             wav = load_audio(path)
#             if wav is None:
#                 continue
#             # Mix to mono for CLAP (expects 1D arrays)
#             if wav.shape[0] > 1:
#                 wav = wav.mean(dim=0, keepdim=False)  # [T]
#             audio_batch.append(wav.numpy())  # list of 1D np.array
#
#         if not audio_batch:
#             continue
#
#         try:
#             inputs = processor(audios=audio_batch, sampling_rate=48000, return_tensors="pt", padding=True)
#             inputs = {key: val.to(device) for key, val in inputs.items()}
#
#             with torch.no_grad():
#                 audio_features = model.get_audio_features(**inputs)
#
#             all_embeddings.append(audio_features.cpu().numpy())
#         except Exception as e:
#             print(f"Error in CLAP batch processing: {e}")
#             continue
#
#     if not all_embeddings:
#         return np.array([])
#
#     return np.concatenate(all_embeddings, axis=0)
#
#
# def calculate_frechet_distance(embeddings1, embeddings2, eps=1e-6):
#     """
#     Calculate Frechet Distance with regularization for stability.
#     """
#     n_samples = embeddings1.shape[0]
#     if n_samples < 2 or embeddings2.shape[0] < 2:
#         return None
#
#     mu1, mu2 = np.mean(embeddings1, axis=0), np.mean(embeddings2, axis=0)
#     sigma1, sigma2 = np.cov(embeddings1, rowvar=False), np.cov(embeddings2, rowvar=False)
#
#     sigma1 += np.eye(sigma1.shape[0]) * eps
#     sigma2 += np.eye(sigma2.shape[0]) * eps
#
#     ssdiff = np.sum((mu1 - mu2) ** 2.0)
#
#     try:
#         covmean = sqrtm(sigma1.dot(sigma2))
#     except Exception as e:
#         print(f"sqrtm calculation failed: {e}")
#         return None
#
#     if np.iscomplexobj(covmean):
#         covmean = covmean.real
#
#     fad_score = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
#     return fad_score
#
#
# def parse_file_list(list_path, root_dir=None):
#     pairs = []
#     with open(list_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             line = line.strip()
#             if not line or '|' not in line: continue
#             parts = line.split('|')
#             ref_p = parts[0].strip()
#             est_p = parts[1].strip()
#             if root_dir:
#                 if not os.path.exists(ref_p): ref_p = os.path.join(root_dir, os.path.basename(ref_p))
#                 if not os.path.exists(est_p): est_p = os.path.join(root_dir, os.path.basename(est_p))
#             pairs.append((ref_p, est_p))
#     return pairs
#
#
# def main():
#     parser = argparse.ArgumentParser(description="Calculate Multi-Mel-SNR, Zimtohrli, and FAD-CLAP.")
#     parser.add_argument("--file_list", type=str, help="Path to text file containing 'ref_path|est_path'")
#     parser.add_argument("file_list_positional", nargs='?', type=str, help="Positional argument compatibility")
#     parser.add_argument("--root_dir", type=str, default=None)
#     parser.add_argument("--batch_size", type=int, default=16)
#     parser.add_argument("--ref_dir", type=str)
#     parser.add_argument("--est_dir", type=str)
#
#     args = parser.parse_args()
#     file_list_path = args.file_list if args.file_list else args.file_list_positional
#
#     # 1. 收集文件对
#     file_pairs = []
#     if file_list_path and os.path.exists(file_list_path):
#         file_pairs = parse_file_list(file_list_path, args.root_dir)
#     elif args.ref_dir and args.est_dir:
#         print(f"Scanning directories: {args.ref_dir} vs {args.est_dir}")
#         ref_files = glob.glob(os.path.join(args.ref_dir, "*.wav")) + glob.glob(os.path.join(args.ref_dir, "*.flac"))
#         for ref_p in ref_files:
#             fname = os.path.basename(ref_p)
#             est_p = os.path.join(args.est_dir, fname)
#             if not os.path.exists(est_p):  # Try swapping ext
#                 base, ext = os.path.splitext(fname)
#                 est_p_alt = os.path.join(args.est_dir, base + ('.wav' if ext == '.flac' else '.flac'))
#                 if os.path.exists(est_p_alt): est_p = est_p_alt
#             if os.path.exists(est_p):
#                 file_pairs.append((ref_p, est_p))
#     else:
#         print("Please provide --file_list OR (--ref_dir and --est_dir).")
#         return
#
#     if not file_pairs:
#         print("No valid file pairs found.")
#         return
#
#     # 2. 初始化 Zimtohrli
#     zim_calculator = None
#     if ZIMTOHRLI_AVAILABLE:
#         try:
#             zim_calculator = zimtohrli.Pyohrli()
#             print("Zimtohrli initialized.")
#         except Exception:
#             pass
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
#
#     # 3. 逐对指标 (SNR & Zimtohrli)
#     print("\n--- Calculating Paired Metrics ---")
#     total_mel_snr = 0.0
#     total_zimtohrli = 0.0
#     count = 0
#     all_ref_paths, all_est_paths = [], []
#
#     for ref_path, est_path in tqdm(file_pairs):
#         try:
#             ref_wav = load_audio(ref_path)
#             est_wav = load_audio(est_path)
#             if ref_wav is None or est_wav is None: continue
#
#             min_len = min(ref_wav.shape[-1], est_wav.shape[-1])
#             if min_len == 0: continue
#             ref_wav, est_wav = ref_wav[..., :min_len], est_wav[..., :min_len]
#
#             # SNR
#             ch_snrs = [multi_mel_snr(ref_wav[ch:ch+1], est_wav[ch:ch+1]) for ch in range(ref_wav.shape[0])]
#             total_mel_snr += sum(ch_snrs) / len(ch_snrs)
#
#             # Zimtohrli
#             if zim_calculator:
#                 ch_zims = [zim_calculator.distance(ref_wav[ch:ch+1].numpy(), est_wav[ch:ch+1].numpy()) for ch in range(ref_wav.shape[0])]
#                 total_zimtohrli += sum(ch_zims) / len(ch_zims)
#
#             count += 1
#             all_ref_paths.append(ref_path)
#             all_est_paths.append(est_path)
#         except Exception:
#             continue
#
#     # 4. FAD-CLAP
#     fad_score = None
#     if TRANSFORMERS_AVAILABLE and all_ref_paths:
#         print("\n--- Calculating FAD-CLAP ---")
#         try:
#             clap_model = ClapModel.from_pretrained("laion/clap-htsat-unfused", use_safetensors=True)
#             clap_processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
#
#             ref_embs = get_clap_embeddings(all_ref_paths, clap_model, clap_processor, device, args.batch_size)
#             est_embs = get_clap_embeddings(all_est_paths, clap_model, clap_processor, device, args.batch_size)
#
#             if len(ref_embs) > 0 and len(est_embs) > 0:
#                 print("Computing Frechet Distance...")
#                 fad_score = calculate_frechet_distance(ref_embs, est_embs, eps=1e-6)
#             else:
#                 print("Embeddings empty.")
#         except Exception as e:
#             print(f"FAD-CLAP Error: {e}")
#
#     # 5. Report
#     print("\n" + "=" * 40)
#     print("        FINAL EVALUATION RESULTS        ")
#     print("=" * 40)
#     print(f"Pairs Evaluated: {count}")
#     if count > 0:
#         print(f"Multi-Mel-SNR (avg): {total_mel_snr / count:.4f} dB")
#         if zim_calculator:
#             print(f"Zimtohrli (avg):     {total_zimtohrli / count:.4f}")
#
#     if fad_score is not None:
#         print(f"FAD-CLAP Score:      {fad_score:.4f}")
#     else:
#         print(f"FAD-CLAP Score:      Failed (Check logs)")
#     print("=" * 40)
#
#
# if __name__ == "__main__":
#     main()
import argparse
import os
import torch
import torchaudio
import numpy as np
import warnings
import glob
from scipy.linalg import sqrtm
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '5'
# Filter warnings
warnings.filterwarnings("ignore")

try:
    import zimtohrli

    ZIMTOHRLI_AVAILABLE = True
except ImportError:
    ZIMTOHRLI_AVAILABLE = False
    print("Warning: 'zimtohrli' library not found. Zimtohrli scores will be skipped.")

try:
    from transformers import ClapModel, ClapProcessor

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: 'transformers' library not found. FAD-CLAP will be skipped.")


def multi_mel_snr(reference, prediction, sr=48000):
    """
    Compute Multi-Mel-SNR between reference and prediction (Scale-Invariant).
    """
    if not isinstance(reference, torch.Tensor):
        reference = torch.from_numpy(reference).float()
    if not isinstance(prediction, torch.Tensor):
        prediction = torch.from_numpy(prediction).float()

    if reference.device != prediction.device:
        prediction = prediction.to(reference.device)

    # Scale-invariant normalization
    dot_ref_pred = torch.sum(reference * prediction)
    dot_pred_pred = torch.sum(prediction * prediction)
    alpha = dot_ref_pred / (dot_pred_pred + 1e-8)
    prediction = alpha * prediction

    configs = [
        (512, 256, 80),
        (1024, 512, 128),
        (2048, 1024, 192)
    ]

    snrs = []
    for n_fft, hop, n_mels in configs:
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, hop_length=hop,
            n_mels=n_mels, f_min=0, f_max=sr // 2, power=1.0
        ).to(reference.device)

        M_ref = mel_transform(reference)
        M_pred = mel_transform(prediction)

        ref_energy = M_ref.pow(2).sum()
        noise_energy = (M_ref - M_pred).pow(2).sum()

        snr = 10 * torch.log10(ref_energy / (noise_energy + 1e-8))
        snrs.append(snr.item())

    return sum(snrs) / len(snrs)


def load_audio(file_path, sr=48000):
    try:
        wav, samplerate = torchaudio.load(file_path)
        if samplerate != sr:
            resampler = torchaudio.transforms.Resample(samplerate, sr)
            wav = resampler(wav)  # 实现重采样

        return wav
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def get_clap_embeddings(file_paths, model, processor, device, batch_size=16):
    model.to(device)
    model.eval()
    all_embeddings = []

    for i in tqdm(range(0, len(file_paths), batch_size), desc="Calculating CLAP embeddings", leave=False):
        batch_paths = file_paths[i:i + batch_size]
        audio_batch = []

        for path in batch_paths:
            wav = load_audio(path)
            if wav is None:
                continue
            # Mix to mono for CLAP (expects 1D arrays)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=False)  # [T]
            audio_batch.append(wav.numpy())  # list of 1D np.array

        if not audio_batch:
            continue

        try:
            inputs = processor(audios=audio_batch, sampling_rate=48000, return_tensors="pt", padding=True)
            inputs = {key: val.to(device) for key, val in inputs.items()}

            with torch.no_grad():
                audio_features = model.get_audio_features(**inputs)

            all_embeddings.append(audio_features.cpu().numpy())
        except Exception as e:
            print(f"Error in CLAP batch processing: {e}")
            continue

    if not all_embeddings:
        return np.array([])

    return np.concatenate(all_embeddings, axis=0)


def calculate_frechet_distance(embeddings1, embeddings2, eps=1e-6):
    """
    Calculate Frechet Distance with regularization for stability.
    Returns:
        fad_score: Total FAD
        d_mu: Semantic Shift (Mean difference)
        d_sigma: Texture/Shape Difference (Covariance trace difference)
    """
    n_samples = embeddings1.shape[0]
    if n_samples < 2 or embeddings2.shape[0] < 2:
        return None, None, None

    mu1, mu2 = np.mean(embeddings1, axis=0), np.mean(embeddings2, axis=0)
    sigma1, sigma2 = np.cov(embeddings1, rowvar=False), np.cov(embeddings2, rowvar=False)

    sigma1 += np.eye(sigma1.shape[0]) * eps
    sigma2 += np.eye(sigma2.shape[0]) * eps

    # D_mu (Semantic Shift)
    d_mu = np.sum((mu1 - mu2) ** 2.0)

    try:
        covmean = sqrtm(sigma1.dot(sigma2))
    except Exception as e:
        print(f"sqrtm calculation failed: {e}")
        return None, None, None

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # D_sigma (Shape/Texture Difference)
    d_sigma = np.trace(sigma1 + sigma2 - 2.0 * covmean)

    # Total FAD
    fad_score = d_mu + d_sigma

    return fad_score, d_mu, d_sigma


def parse_file_list(list_path, root_dir=None):
    pairs = []
    with open(list_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or '|' not in line: continue
            parts = line.split('|')
            ref_p = parts[0].strip()
            est_p = parts[1].strip()
            if root_dir:
                if not os.path.exists(ref_p): ref_p = os.path.join(root_dir, os.path.basename(ref_p))
                if not os.path.exists(est_p): est_p = os.path.join(root_dir, os.path.basename(est_p))
            pairs.append((ref_p, est_p))
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Calculate Multi-Mel-SNR, Zimtohrli, and FAD-CLAP.")
    parser.add_argument("--file_list", type=str, help="Path to text file containing 'ref_path|est_path'")
    parser.add_argument("file_list_positional", nargs='?', type=str, help="Positional argument compatibility")
    parser.add_argument("--root_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--ref_dir", type=str)
    parser.add_argument("--est_dir", type=str)

    args = parser.parse_args()
    file_list_path = args.file_list if args.file_list else args.file_list_positional
    file_pairs = []
    if file_list_path and os.path.exists(file_list_path):
        file_pairs = parse_file_list(file_list_path, args.root_dir)
    elif args.ref_dir and args.est_dir:
        print(f"Scanning directories: {args.ref_dir} vs {args.est_dir}")
        ref_files = glob.glob(os.path.join(args.ref_dir, "*.wav")) + glob.glob(os.path.join(args.ref_dir, "*.flac"))
        for ref_p in ref_files:
            fname = os.path.basename(ref_p)
            est_p = os.path.join(args.est_dir, fname)
            if not os.path.exists(est_p):  # Try swapping ext
                base, ext = os.path.splitext(fname)
                est_p_alt = os.path.join(args.est_dir, base + ('.wav' if ext == '.flac' else '.flac'))
                if os.path.exists(est_p_alt): est_p = est_p_alt
            if os.path.exists(est_p):
                file_pairs.append((ref_p, est_p))
    else:
        print("Please provide --file_list OR (--ref_dir and --est_dir).")
        return

    if not file_pairs:
        print("No valid file pairs found.")
        return
    zim_calculator = None
    if ZIMTOHRLI_AVAILABLE:
        try:
            zim_calculator = zimtohrli.Pyohrli()
            print("Zimtohrli initialized.")
        except Exception:
            pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("\n--- Calculating Paired Metrics ---")
    total_mel_snr = 0.0
    total_zimtohrli = 0.0
    count = 0
    all_ref_paths, all_est_paths = [], []

    for ref_path, est_path in tqdm(file_pairs):
        try:
            ref_wav = load_audio(ref_path)
            est_wav = load_audio(est_path)
            if ref_wav is None or est_wav is None: continue

            min_len = min(ref_wav.shape[-1], est_wav.shape[-1])
            if min_len == 0: continue
            ref_wav, est_wav = ref_wav[..., :min_len], est_wav[..., :min_len]

            # SNR
            ch_snrs = [multi_mel_snr(ref_wav[ch:ch + 1], est_wav[ch:ch + 1]) for ch in range(ref_wav.shape[0])]
            total_mel_snr += sum(ch_snrs) / len(ch_snrs)

            # Zimtohrli
            if zim_calculator:
                ch_zims = [zim_calculator.distance(ref_wav[ch:ch + 1].numpy(), est_wav[ch:ch + 1].numpy()) for ch in
                           range(ref_wav.shape[0])]
                total_zimtohrli += sum(ch_zims) / len(ch_zims)

            count += 1
            all_ref_paths.append(ref_path)
            all_est_paths.append(est_path)
        except Exception:
            continue
    fad_score = None
    d_mu = None
    d_sigma = None

    if TRANSFORMERS_AVAILABLE and all_ref_paths:
        print("\n--- Calculating FAD-CLAP ---")
        try:
            clap_model = ClapModel.from_pretrained("laion/clap-htsat-unfused", use_safetensors=True)
            clap_processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")

            ref_embs = get_clap_embeddings(all_ref_paths, clap_model, clap_processor, device, args.batch_size)
            est_embs = get_clap_embeddings(all_est_paths, clap_model, clap_processor, device, args.batch_size)

            if len(ref_embs) > 0 and len(est_embs) > 0:
                print("Computing Frechet Distance and its components...")
                fad_score, d_mu, d_sigma = calculate_frechet_distance(ref_embs, est_embs, eps=1e-6)
            else:
                print("Embeddings empty.")
        except Exception as e:
            print(f"FAD-CLAP Error: {e}")

    print("\n" + "=" * 40)
    print("        FINAL EVALUATION RESULTS        ")
    print("=" * 40)
    print(f"Pairs Evaluated: {count}")
    if count > 0:
        print(f"Multi-Mel-SNR (avg): {total_mel_snr / count:.4f} dB")
        if zim_calculator:
            print(f"Zimtohrli (avg):     {total_zimtohrli / count:.4f}")

    if fad_score is not None:
        print(f"FAD-CLAP Total Score: {fad_score:.4f}")
        print(f"  ├─ D_mu (Mean Shift)     : {d_mu:.4f}")
        print(f"  └─ D_sigma (Cov Variance): {d_sigma:.4f}")
    else:
        print(f"FAD-CLAP Score:      Failed (Check logs)")
    print("=" * 40)


if __name__ == "__main__":
    main()