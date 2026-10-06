"""
Diagnose current model outputs on AES and 3DES across all 5 sizes:
1kb, 8kb, 64kb, 256kb, 512kb
"""

import sys
from pathlib import Path
import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from crypto_identifier.inference import default_predictor

def main():
    sizes = ["1kb", "8kb", "64kb", "256kb", "512kb"]
    
    print("=" * 70)
    print(" DIAGNOSTIC REPORT: AES & 3DES PREDICTIONS ACROSS ALL SIZES")
    print("=" * 70)
    
    for task in ["multiclass", "binary"]:
        print(f"\n--- TASK: {task.upper()} ---")
        for algo in ["AES", "3DES"]:
            print(f"\nTarget: {algo}")
            for sz in sizes:
                sample_file = WORKSPACE_ROOT / "crypto_validation_dataset" / "ciphertext" / algo / sz / "sample_0000.bin"
                if not sample_file.exists():
                    print(f"  {sz:5s}: FILE NOT FOUND ({sample_file})")
                    continue
                
                res = default_predictor.predict_file(
                    sample_file,
                    task=task,
                    size_override=sz,
                    architecture="cnn"
                )
                pred = res["predicted_cipher"]
                conf = res["confidence"]
                probs = res["probabilities"]
                is_correct = (pred == algo)
                mark = "✓ PASS" if is_correct else "✗ FAIL"
                print(f"  {sz:5s}: {mark} -> Predicted: {pred:8s} ({conf:.2%}) | Probs: {probs}")

if __name__ == "__main__":
    main()
