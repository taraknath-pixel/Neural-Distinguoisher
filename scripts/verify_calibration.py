"""
Calibration test to verify that AES input produces AES and 3DES input produces 3DES
across sizes and model architectures.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import joblib

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from crypto_identifier.classification import CipherCNN
from crypto_identifier.feature_extraction import extract_features_from_bits

def get_feats(file_path):
    data = Path(file_path).read_bytes()
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    return np.array(extract_features_from_bits(bits), dtype=np.float32).reshape(1, -1)

def main():
    aes_1kb = WORKSPACE_ROOT / "crypto_validation_dataset" / "ciphertext" / "AES" / "1kb" / "sample_0000.bin"
    tdes_1kb = WORKSPACE_ROOT / "crypto_validation_dataset" / "ciphertext" / "3DES" / "1kb" / "sample_0000.bin"
    
    f_aes = get_feats(aes_1kb)
    f_tdes = get_feats(tdes_1kb)
    
    print("Testing binary splits for CNN...")
    best_binary_splits = {
        "1kb": "split_06",
        "8kb": "split_07",
        "64kb": "split_09",
        "256kb": "split_03",
        "512kb": "split_09"
    }
    
    for sz, sp in best_binary_splits.items():
        dirpath = WORKSPACE_ROOT / "validation_results" / "models" / "epochs-500-trained" / "binary" / f"size_{sz}" / sp
        if not dirpath.exists():
            continue
        scaler = joblib.load(dirpath / "scaler.joblib")["scaler"]
        chk = torch.load(dirpath / "best_cnn.pt", map_location="cpu", weights_only=False)
        m = CipherCNN(input_dim=49, num_classes=2)
        m.load_state_dict(chk["state_dict"] if "state_dict" in chk else chk)
        m.eval()
        
        with torch.no_grad():
            out_a = torch.softmax(m(torch.tensor(scaler.transform(f_aes), dtype=torch.float32)), dim=1)[0].numpy()
            out_t = torch.softmax(m(torch.tensor(scaler.transform(f_tdes), dtype=torch.float32)), dim=1)[0].numpy()
            
            p_a = "AES" if out_a[0] > out_a[1] else "3DES"
            p_t = "AES" if out_t[0] > out_t[1] else "3DES"
            print(f"Size {sz} ({sp}): AES input -> {p_a} ({out_a[0]:.2%}) | 3DES input -> {p_t} ({out_t[1]:.2%})")

if __name__ == "__main__":
    main()
