"""
Script to package the selected server checkpoints into src/crypto_identifier/packaged_models/
with clear metadata, accuracy scores, and experimental labels.
"""

import shutil
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE_MODELS_ROOT = ROOT / "validation_results" / "models" / "epochs-500-trained"
TARGET_DIR = ROOT / "src" / "crypto_identifier" / "packaged_models"

PACKAGES = [
    {
        "task": "multiclass",
        "name": "5-Cipher Identifier (Multiclass CNN)",
        "source_split_dir": SOURCE_MODELS_ROOT / "multiclass" / "size_256kb" / "split_04",
        "file_size": "256kb",
        "split": 4,
        "classes": ["AES", "3DES", "CAST", "RC2", "Blowfish"],
        "num_classes": 5,
        "measured_test_accuracy": 0.202,
        "chance_accuracy": 0.200,
        "best_validation_loss": 1.607869,
        "status": "experimental",
        "disclaimer": "Held-out test accuracy is 20.2%, virtually equal to chance (20.0%). All 5 ciphers produce statistically indistinguishable NIST features.",
    },
    {
        "task": "binary",
        "name": "Binary Cipher Identifier (AES vs 3DES)",
        "source_split_dir": SOURCE_MODELS_ROOT / "binary" / "size_64kb" / "split_10",
        "file_size": "64kb",
        "split": 10,
        "classes": ["AES", "3DES"],
        "num_classes": 2,
        "measured_test_accuracy": 0.505,
        "chance_accuracy": 0.500,
        "best_validation_loss": 0.690228,
        "status": "experimental",
        "disclaimer": "Held-out test accuracy is 50.5%, virtually equal to chance (50.0%). AES and 3DES both pass NIST randomness tests identically.",
    }
]

def main():
    print(f"Packaging models into: {TARGET_DIR}")
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    
    for pkg in PACKAGES:
        task = pkg["task"]
        src = pkg["source_split_dir"]
        dst = TARGET_DIR / task
        dst.mkdir(parents=True, exist_ok=True)
        
        src_cnn = src / "best_cnn.pt"
        src_scaler = src / "scaler.joblib"
        
        if not src_cnn.exists():
            raise FileNotFoundError(f"Source CNN missing: {src_cnn}")
        if not src_scaler.exists():
            raise FileNotFoundError(f"Source Scaler missing: {src_scaler}")
            
        dst_cnn = dst / "best_cnn.pt"
        dst_scaler = dst / "scaler.joblib"
        dst_meta = dst / "metadata.json"
        
        shutil.copy2(src_cnn, dst_cnn)
        shutil.copy2(src_scaler, dst_scaler)
        
        meta = {
            "task": task,
            "name": pkg["name"],
            "classes": pkg["classes"],
            "num_classes": pkg["num_classes"],
            "trained_size": pkg["file_size"],
            "split": pkg["split"],
            "measured_test_accuracy": pkg["measured_test_accuracy"],
            "chance_accuracy": pkg["chance_accuracy"],
            "best_validation_loss": pkg["best_validation_loss"],
            "is_experimental": True,
            "status": pkg["status"],
            "disclaimer": pkg["disclaimer"],
            "cnn_weights_file": "best_cnn.pt",
            "scaler_file": "scaler.joblib",
        }
        with open(dst_meta, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
            
        print(f"✓ Packaged {task} ({pkg['name']}) -> {dst}")
        print(f"  Accuracy: {pkg['measured_test_accuracy']:.1%} (Chance: {pkg['chance_accuracy']:.1%}) [Experimental]")

if __name__ == "__main__":
    main()
