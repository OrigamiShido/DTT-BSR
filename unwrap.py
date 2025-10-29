import torch
import os, glob
from collections import OrderedDict
from pathlib import Path

def unwrap_generator_checkpoint(ckpt_path: str, output_path: str) -> None:
    try:
        full_checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    except FileNotFoundError:
        print(f"[!] Error: Checkpoint file not found at {ckpt_path}")
        return
    except Exception as e:
        print(f"[!] An error occurred while loading the checkpoint: {e}")
        return

    if 'state_dict' not in full_checkpoint:
        print("[!] Error: 'state_dict' not found in the checkpoint. Is this a valid Lightning checkpoint?")
        return
        
    full_state_dict = full_checkpoint['state_dict']
    
    generator_state_dict = OrderedDict()
    
    prefix = 'generator.'
    prefix_len = len(prefix)
    
    for key, value in full_state_dict.items():
        if key.startswith(prefix):
            new_key = key[prefix_len:]
            generator_state_dict[new_key] = value
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(generator_state_dict, output_path)

if __name__ == '__main__':
    input_dir = "experiment/msrkit/SPMamba/checkpoints"
    # find all .ckpt files in the input directory
    ckpt_files = glob.glob(os.path.join(input_dir, '*.ckpt'))
    for ckpt_file in ckpt_files:
        unwrap_generator_checkpoint(ckpt_file, os.path.join(input_dir, os.path.basename(ckpt_file).replace('.ckpt', '.pth')))
        print(f"Unwrapped {ckpt_file} to {os.path.join(input_dir, os.path.basename(ckpt_file).replace('.ckpt', '.pth'))}")