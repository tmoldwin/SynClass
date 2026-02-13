#!/usr/bin/env python3
"""
Install PyTorch with CUDA 12.8 for RTX 50-series (sm_120) GPUs.
Run: python scripts/install_pytorch_rtx50.py

The default pip install uses older CUDA builds that don't support RTX 5060/5070/5080/5090.
"""
import subprocess
import sys


def main():
    cmd = [
        sys.executable, "-m", "pip", "install", "--upgrade",
        "torch", "torchvision", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cu128",
    ]
    print("Installing PyTorch with CUDA 12.8 (RTX 50-series / sm_120 support)...")
    print(" ".join(cmd))
    subprocess.check_call(cmd)
    print("\nVerifying...")
    import torch
    print(f"PyTorch {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        # Quick sanity check
        x = torch.zeros(1, device="cuda")
        print("CUDA tensor test: OK")
    print("\nDone. Run training without --cpu flag.")


if __name__ == "__main__":
    main()
