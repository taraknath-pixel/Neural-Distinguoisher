"""
Test Multiclass on all 5 ciphers across all 5 sizes:
1kb, 8kb, 64kb, 256kb, 512kb
"""

import sys
from pathlib import Path
import numpy as np
import torch
import joblib

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from crypto_identifier.inference import default_predictor

def main():
    ciphers = ["AES", "3DES", "Blowfish", "CAST", "RC2"]
    sizes = ["1kb", "8kb", "64kb", "256kb", "512kb"]
    
    print("Testing Multiclass predictions on all 5 ciphers:")
    for sz in sizes:
        print(f"\n=================== MULTICLASS SIZE: {sz} ===================")
        for c in ciphers:
            f = WORKSPACE_ROOT / "crypto_validation_dataset" / "ciphertext" / c / sz / "sample_0000.bin"
            if not f.exists():
                print(f"  {c:10s} ({sz}): File not found")
                continue
            res = default_predictor.predict_file(f, task="multiclass", size_override=sz, architecture="cnn")
            pred = res["predicted_cipher"]
            conf = res["confidence"]
            probs = res["probabilities"]
            status = "✓ MATCH" if pred == c else f"≠ PREDICTED {pred}"
            print(f"  {c:8s} -> {status:18s} (Conf: {conf:.2%}) | Top 2: {sorted(probs.items(), key=lambda x: -x[1])[:2]}")

if __name__ == "__main__":
    main()
