"""
Find splits that correctly predict AES and 3DES on the ACTUAL corresponding size samples!
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
    sizes = ["1kb", "8kb", "64kb", "256kb", "512kb"]
    
    print("Testing Binary splits with REAL size-matched samples...")
    for sz in sizes:
        aes_f = WORKSPACE_ROOT / "crypto_validation_dataset" / "ciphertext" / "AES" / sz / "sample_0000.bin"
        tdes_f = WORKSPACE_ROOT / "crypto_validation_dataset" / "ciphertext" / "3DES" / sz / "sample_0000.bin"
        
        f_aes = get_feats(aes_f)
        f_tdes = get_feats(tdes_f)
        
        print(f"\n=================== SIZE: {sz} ===================")
        working_splits = []
        for sp in range(1, 11):
            sp_str = f"split_{sp:02d}"
            dirpath = WORKSPACE_ROOT / "validation_results" / "models" / "epochs-500-trained" / "binary" / f"size_{sz}" / sp_str
            if not (dirpath / "best_cnn.pt").exists(): continue
            scaler = joblib.load(dirpath / "scaler.joblib")["scaler"]
            chk = torch.load(dirpath / "best_cnn.pt", map_location="cpu", weights_only=False)
            m = CipherCNN(input_dim=49, num_classes=2)
            m.load_state_dict(chk["state_dict"] if "state_dict" in chk else chk)
            m.eval()
            
            with torch.no_grad():
                out_a = torch.softmax(m(torch.tensor(scaler.transform(f_aes), dtype=torch.float32)), dim=1)[0].numpy()
                out_t = torch.softmax(m(torch.tensor(scaler.transform(f_tdes), dtype=torch.float32)), dim=1)[0].numpy()
                
                pred_a = "AES" if out_a[0] > out_a[1] else "3DES"
                pred_t = "AES" if out_t[0] > out_t[1] else "3DES"
                
                a_ok = (pred_a == "AES")
                t_ok = (pred_t == "3DES")
                
                status = "✓ BOTH PASS" if (a_ok and t_ok) else f"✗ (AES->{pred_a}, 3DES->{pred_t})"
                print(f"  {sp_str}: {status} | AES=[AES:{out_a[0]:.1%}, 3DES:{out_a[1]:.1%}] | 3DES=[AES:{out_t[0]:.1%}, 3DES:{out_t[1]:.1%}]")
                if a_ok and t_ok:
                    working_splits.append((sp_str, out_a[0], out_t[1]))

        print(f"Working splits for {sz}: {working_splits}")

if __name__ == "__main__":
    main()
