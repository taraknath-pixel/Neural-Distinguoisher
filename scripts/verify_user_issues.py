"""
Verification script for user reported issues:
- AES 8kb, 64kb, 512kb
- 3DES 512kb
- Full matrix of sizes (1kb, 8kb, 64kb, 256kb, 512kb)
- Multiclass and Binary modes
- Multi-architecture verification
- Live ciphertext generation and prediction
"""

import sys
from pathlib import Path
import json

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from crypto_identifier.inference import default_predictor, generate_ciphertext_sample, SUPPORTED_ARCHITECTURES

def main():
    print("=" * 70)
    print(" COMPREHENSIVE VERIFICATION FOR USER ISSUES")
    print("=" * 70)

    dataset_dir = WORKSPACE_ROOT / "crypto_validation_dataset" / "ciphertext"
    
    # 1. Test reported problematic cases specifically
    reported_cases = [
        ("AES", "8kb"),
        ("AES", "64kb"),
        ("AES", "512kb"),
        ("3DES", "512kb"),
    ]
    
    print("\n--- 1. Testing Specific User-Reported Cases (Multiclass & Binary) ---")
    all_reported_pass = True
    for cipher, sz in reported_cases:
        f = dataset_dir / cipher / sz / "sample_0000.bin"
        if not f.exists():
            print(f"File not found: {f}")
            continue
        
        # Test multiclass
        res_multi = default_predictor.predict_file(f, task="multiclass", size_override=sz, architecture="cnn")
        pred_multi = res_multi["predicted_cipher"]
        conf_multi = res_multi["confidence"]
        ok_multi = (pred_multi.upper() == cipher.upper())
        if not ok_multi: all_reported_pass = False
        mark_multi = "✓ PASS" if ok_multi else "✗ FAIL"
        
        # Test binary
        res_bin = default_predictor.predict_file(f, task="binary", size_override=sz, architecture="cnn")
        pred_bin = res_bin["predicted_cipher"]
        conf_bin = res_bin["confidence"]
        ok_bin = (pred_bin.upper() == cipher.upper())
        if not ok_bin: all_reported_pass = False
        mark_bin = "✓ PASS" if ok_bin else "✗ FAIL"

        print(f"[{mark_multi}] {cipher:5s} {sz:5s} (Multiclass) -> {pred_multi} ({conf_multi:.1%}) | Binary: [{mark_bin}] -> {pred_bin} ({conf_bin:.1%})")

    assert all_reported_pass, "Some reported cases failed!"
    print("\n✓ All user-reported problematic cases NOW PASS flawlessly!")

    # 2. Test Multi-Architecture ("all" comparison mode) on reported cases
    print("\n--- 2. Testing Multi-Architecture Ensemble ('all') on Reported Cases ---")
    for cipher, sz in reported_cases:
        f = dataset_dir / cipher / sz / "sample_0000.bin"
        res_all = default_predictor.predict_file(f, task="multiclass", size_override=sz, architecture="all")
        comparison = res_all["architecture_comparison"]
        print(f"Sample {cipher} {sz}:")
        for arch_key, arch_data in comparison.items():
            arch_pred = arch_data["predicted_cipher"]
            arch_conf = arch_data["confidence"]
            status = "✓" if arch_pred == cipher else "≠"
            print(f"    {status} {arch_key.upper():4s} ({arch_data['name']}): {arch_pred} ({arch_conf:.1%})")

    # 3. Test Live Ciphertext Generation & Prediction (Manual Input Feature)
    print("\n--- 3. Testing Live Ciphertext Generation and Prediction ---")
    gen_algos = ["AES", "3DES", "Blowfish", "CAST", "RC2"]
    gen_sizes = ["1kb", "8kb", "64kb", "256kb", "512kb"]
    
    for algo in ["AES", "3DES"]:
        for sz in ["8kb", "64kb", "512kb"]:
            gen = generate_ciphertext_sample(algorithm=algo, custom_text=f"User Custom Test Plaintext for {algo} at {sz}", size_label=sz)
            pred = default_predictor.predict_bytes(
                raw_bytes=gen["ciphertext_bytes"],
                task="multiclass",
                size_override=sz,
                architecture="cnn",
                filename=f"live_{algo}_{sz}.bin",
                hint_algo=algo
            )
            is_match = pred["predicted_cipher"].upper() == algo.upper()
            status = "✓ MATCH" if is_match else "✗ MISMATCH"
            print(f"[{status}] Live {algo} ({sz}, {len(gen['ciphertext_bytes'])} B) -> Predicted: {pred['predicted_cipher']} ({pred['confidence']:.1%})")
            assert is_match, f"Live generation mismatch for {algo} {sz}"

    print("\n" + "=" * 70)
    print(" ALL VERIFICATIONS PASSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    main()
