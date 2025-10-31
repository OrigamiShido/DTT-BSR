import argparse
import yaml
from pathlib import Path
from typing import Dict, Any
import os

import torch
import torch.nn as nn
import soundfile as sf
import numpy as np
from tqdm import tqdm

from models import MelRNN, MelRoFormer, UNet
from models.SPManba import SPMamba
from models.DTTNet.dp_tdf.dp_tdf_net import DPTDFNet

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

def load_generator(config: Dict[str, Any], checkpoint_path: str, device: str = 'cuda') -> nn.Module:
    """Initialize and load the generator model from unwrapped checkpoint."""
    model_cfg = config['model']
    
    # Initialize generator based on config
    if model_cfg['name'] == 'MelRNN':
        generator = MelRNN.MelRNN(**model_cfg['params'])
    elif model_cfg['name'] == 'MelRoFormer':
        generator = MelRoFormer.MelRoFormer(**model_cfg['params'])
    elif model_cfg['name'] == 'MelUNet':
        generator = UNet.MelUNet(**model_cfg['params'])
    elif model_cfg['name'] == 'SPMamba':
        generator = SPMamba(**model_cfg['params'])
    elif model_cfg['name'] == 'DTTNet':
        generator = DPTDFNet(**model_cfg['params'])
    else:
        raise ValueError(f"Unknown model name: {model_cfg['name']}")
    
    # Load unwrapped generator weights
    state_dict = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(state_dict)
    
    generator = generator.to(device)
    generator.eval()
    
    return generator


def process_audio(audio: np.ndarray, generator: nn.Module, device: str = 'cuda') -> np.ndarray:
    """Process a single audio array through the generator."""
    # Convert to tensor: (channels, samples) -> (1, channels, samples)
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]  # Add channel dimension for mono
    
    audio_tensor = torch.from_numpy(audio).float().to(device)
    
    # Run inference
    with torch.no_grad():
        output_tensor = generator(audio_tensor)
    
    # Convert back to numpy: (1, channels, samples) -> (channels, samples)
    output_audio = output_tensor.cpu().numpy()
    
    return output_audio


def main():
    parser = argparse.ArgumentParser(description="Run inference on audio files using trained generator")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to unwrapped generator weights (.pth)")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input .flac files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save processed audio")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on (cuda/cpu)")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all .flac files
    audio_files = sorted(input_dir.glob("*.flac"))
    
    if len(audio_files) == 0:
        print(f"No .flac files found in {input_dir}")
        return
    
    print(f"Found {len(audio_files)} audio files")
    
    # Load model
    print(f"Loading generator from {args.checkpoint}...")
    generator = load_generator(config, args.checkpoint, device=args.device)
    print("Model loaded successfully")
    
    # Process each file
    for audio_file in tqdm(audio_files, desc="Processing audio files"):
        # Load audio
        audio, sr = sf.read(audio_file)
        
        # Transpose if needed: soundfile loads as (samples, channels)
        if audio.ndim == 2:
            audio = audio.T  # Convert to (channels, samples)
        
        # Process through generator
        output_audio = process_audio(audio, generator, device=args.device)
        
        # Transpose back for saving: (channels, samples) -> (samples, channels)
        if output_audio.ndim == 2:
            output_audio = output_audio.T
        
        # Save with same filename
        output_path = output_dir / audio_file.name
        sf.write(output_path, output_audio, sr)
    
    print(f"\nProcessing complete! Output saved to {output_dir}")


if __name__ == '__main__':
    main()