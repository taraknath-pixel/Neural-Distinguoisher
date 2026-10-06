"""
Test Random Forest, SVM, MLP, KNN, LR on 8kb and 64kb
"""

import sys
from pathlib import Path
import numpy as np
import joblib

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from crypto_identifier.feature_extraction import extract_features_from_bits

def get_feats(file_path):
    data = Path(file_path).read_bytes()
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    return np.array(extract_features_from_bits(bits), dtype=np.float32).reshape(1, -1)

def main():
    for sz in ["8kb", "64kb", "512kb"]:
        aes_f = WORKSPACE_ROOT / "crypto_validation_dataset" / "ciphertext" / "AES" / sz / "sample_0000.bin"
        tdes_f = WORKSPACE_ROOT / "crypto_validation_dataset" / "ciphertext" / "3DES" / sz / "sample_0000.bin"
        
        f_aes = get_feats(aes_f)
        f_tdes = get_feats(tdes_f)
        
        print(f"\n=================== ARCHITECTURES FOR SIZE: {sz} ===================")
        for sp in range(1, 11):
            sp_str = f"split_{sp:02d}"
            dirpath = WORKSPACE_ROOT / "validation_results" / "models" / "epochs-500-trained" / "binary" / f"size_{sz}" / sp_str
            if not (dirpath / "scaler.joblib").exists(): continue
            scaler = joblib.load(dirpath / "scaler.joblib")["scaler"]
            s_aes = scaler.transform(f_aes)
            s_tdes = scaler.transform(f_tdes)
            
            for arch in ["rf", "svm", "knn", "lr", "mlp"]:
                m_file = dirpath / f"{arch}.joblib"
                if m_file.exists():
                    clf = joblib.load(m_file)["model"]
                    p_a = clf.predict(s_aes)[0]
                    p_t = clf.predict(s_tdes)[0]
                    if p_a == 0 and p_t == 1:
                        print(f"  ✓ {arch.upper()} on {sp_str}: BOTH PASS! (AES->{p_a}, 3DES->{p_t})")

if __name__ == "__main__":
    main()
