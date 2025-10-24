import os
import soundfile as sf
import torch
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio
import argparse
import numpy as np
import warnings
from scipy.linalg import sqrtm
from tqdm import tqdm

warnings.filterwarnings("ignore")

# os.environ['HF_ENDPOINT']='https://hf-mirror.com'

try:
    from transformers import ClapModel, ClapProcessor
except ImportError:
    print("Error: The 'transformers' library is not installed.")
    print("Please install it to run FAD-CLAP calculations:")
    print("pip install torch transformers")
    exit(1)


def load_audio(file_path, sr=48000):
    try:
        wav, samplerate = sf.read(file_path)
        if samplerate != sr:
            pass
        if wav.ndim > 1:
            wav = wav.T
        else:
            wav = wav[np.newaxis, :]
        return torch.from_numpy(wav).float()
    except Exception:
        return None

def get_clap_embeddings(file_paths, model, processor, device, batch_size=16):
    model.to(device)
    all_embeddings = []
    
    for i in tqdm(range(0, len(file_paths), batch_size), desc="  Calculating embeddings", ncols=100, leave=False):
        batch_paths = file_paths[i:i+batch_size]
        audio_batch = []
        for path in batch_paths:
            try:
                wav, sr = sf.read(path)
                if wav.ndim == 2 and wav.shape[1] == 2:
                    audio_batch.append(wav[:, 0]) # Left channel
                    audio_batch.append(wav[:, 1]) # Right channel
                elif wav.ndim == 1:
                    audio_batch.append(wav)
                else:
                    continue
            except Exception:
                continue

        if not audio_batch:
            continue

        try:
            inputs = processor(audios=audio_batch, sampling_rate=48000, return_tensors="pt", padding=True)
            inputs = {key: val.to(device) for key, val in inputs.items()}
            
            with torch.no_grad():
                audio_features = model.get_audio_features(**inputs)
            
            all_embeddings.append(audio_features.cpu().numpy())
        except Exception:
            continue
            
    if not all_embeddings:
        return np.array([])
        
    return np.concatenate(all_embeddings, axis=0)

def calculate_frechet_distance(embeddings1, embeddings2):
    if embeddings1.shape[0] < 2 or embeddings2.shape[0] < 2:
        return None

    mu1, mu2 = np.mean(embeddings1, axis=0), np.mean(embeddings2, axis=0)
    sigma1, sigma2 = np.cov(embeddings1, rowvar=False), np.cov(embeddings2, rowvar=False)
    
    ssdiff = np.sum((mu1 - mu2)**2.0)
    
    try:
        covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    except Exception:
        return None

    if np.iscomplexobj(covmean):
        covmean = covmean.real
        
    fad_score = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fad_score

def main():
    parser = argparse.ArgumentParser(description="Calculate SI-SNR and FAD-CLAP for audio pairs listed in a text file.")
    parser.add_argument("file_list", type=str, help="Path to a text file with the format: target_path|output_path")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for FAD-CLAP embedding calculation.")
    args = parser.parse_args()

    if not os.path.exists(args.file_list):
        print(f"Error: Input file not found at {args.file_list}")
        return

    sisnr_calculator = ScaleInvariantSignalNoiseRatio()
    all_target_paths = []
    all_output_paths = []

    print("--- Calculating SI-SNR for each pair ---")
    with open(args.file_list, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or '|' not in line:
                continue

            try:
                target_path, output_path = [p.strip() for p in line.split('|')]

                if not os.path.exists(target_path) or not os.path.exists(output_path):
                    print(f"Skipping line, file not found: {line}")
                    continue

                target_wav = load_audio(target_path)
                output_wav = load_audio(output_path)

                if target_wav is None or output_wav is None:
                    continue
                if target_wav.shape[0] != output_wav.shape[0]:
                    continue

                min_len = min(target_wav.shape[-1], output_wav.shape[-1])
                target_wav = target_wav[..., :min_len]
                output_wav = output_wav[..., :min_len]

                if target_wav.shape[-1] == 0:
                    continue

                sisnr_val = sisnr_calculator(output_wav, target_wav)
                print(f"{target_path}|{output_path}|{sisnr_val.item():.4f}")
                
                all_target_paths.append(target_path)
                all_output_paths.append(output_path)

            except Exception:
                continue

    print("\n--- Calculating FAD-CLAP for all target vs. all output files ---")
    if not all_target_paths:
        print("No valid file pairs found to calculate FAD-CLAP.")
        return

    try:
        print("Loading CLAP model...")
        clap_model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
        clap_processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
        clap_model.eval()
        print("CLAP model loaded successfully.")
    except Exception as e:
        print(f"Fatal Error: Could not load CLAP model. Please check internet connection. Error: {e}")
        return
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\nCalculating embeddings for all target files...")
    target_embeddings = get_clap_embeddings(all_target_paths, clap_model, clap_processor, device, args.batch_size)

    print("Calculating embeddings for all output files...")
    output_embeddings = get_clap_embeddings(all_output_paths, clap_model, clap_processor, device, args.batch_size)

    if target_embeddings.size > 0 and output_embeddings.size > 0:
        print("Calculating Frechet Audio Distance (FAD)...")
        fad_score = calculate_frechet_distance(target_embeddings, output_embeddings)
        if fad_score is not None:
            print(f"\nOverall FAD-CLAP Score: {fad_score:.4f}")
        else:
            print("\nCould not calculate FAD-CLAP score.")
    else:
        print("\nCould not calculate FAD-CLAP due to issues with embedding generation.")

if __name__ == "__main__":
    main()

