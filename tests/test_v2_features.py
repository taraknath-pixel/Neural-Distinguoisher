"""
Test suite verifying v0.2.0 features:
1. Direction correctness: AES input -> AES, 3DES input -> 3DES
2. All sizes support: 1kb, 8kb, 64kb, 256kb, 512kb
3. Multiple architectures: CNN, Random Forest, SVM, MLP, KNN, Logistic Regression, Gaussian NB, and 'all'
4. Live Ciphertext Generation & Manual Input: /api/generate and /api/generate-and-predict
"""

import sys
import json
import time
import threading
from pathlib import Path
import urllib.request
import urllib.parse

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

import http.server
from crypto_identifier.api import StandaloneAPIHandler
from crypto_identifier.inference import (
    default_predictor,
    generate_ciphertext_sample,
    SUPPORTED_ARCHITECTURES,
)

def run_test_server(server):
    server.serve_forever()

def main():
    print("=================================================================")
    print(" Verifying CryptoCipher AI v0.2.0 Upgraded Features")
    print("=================================================================")

    # 1. Direction Correctness Verification
    print("\n--- 1. Testing Prediction Direction (AES vs 3DES) ---")
    aes_sample = WORKSPACE_ROOT / "crypto_validation_dataset" / "ciphertext" / "AES" / "1kb" / "sample_0000.bin"
    tdes_sample = WORKSPACE_ROOT / "crypto_validation_dataset" / "ciphertext" / "3DES" / "1kb" / "sample_0000.bin"
    
    pred_aes = default_predictor.predict_file(aes_sample, task="binary", size_override="1kb", architecture="cnn")
    pred_tdes = default_predictor.predict_file(tdes_sample, task="binary", size_override="1kb", architecture="cnn")
    
    print(f"  Input AES file  -> Predicted: {pred_aes['predicted_cipher']} (Confidence: {pred_aes['confidence']:.2%})")
    print(f"  Input 3DES file -> Predicted: {pred_tdes['predicted_cipher']} (Confidence: {pred_tdes['confidence']:.2%})")
    assert pred_aes["predicted_cipher"] == "AES", f"Expected AES, got {pred_aes['predicted_cipher']}"
    assert pred_tdes["predicted_cipher"] == "3DES", f"Expected 3DES, got {pred_tdes['predicted_cipher']}"
    print("✓ Prediction direction is CORRECT: AES yields AES and 3DES yields 3DES!")

    # 2. Test Multi-Architecture Support
    print("\n--- 2. Testing Multiple Model Architectures ---")
    for arch in ["cnn", "rf", "svm", "mlp", "knn", "lr", "gnb"]:
        res = default_predictor.predict_file(aes_sample, task="binary", size_override="1kb", architecture=arch)
        assert res["success"] is True
        print(f"  ✓ Arch '{arch}' ({SUPPORTED_ARCHITECTURES[arch]}): Predicted {res['predicted_cipher']} (Confidence: {res['confidence']:.2%})")

    # Test 'all' comparison mode
    res_all = default_predictor.predict_file(aes_sample, task="binary", size_override="1kb", architecture="all")
    assert "architecture_comparison" in res_all and res_all["architecture_comparison"] is not None
    assert len(res_all["architecture_comparison"]) == len(SUPPORTED_ARCHITECTURES)
    print(f"  ✓ Architecture Comparison ('all') returned {len(res_all['architecture_comparison'])} models side-by-side")

    # 3. Test Live Ciphertext Generator
    print("\n--- 3. Testing Live Data Generation & Encryption ---")
    gen_res = generate_ciphertext_sample(
        algorithm="AES",
        custom_text="SECRET TEST MESSAGE FOR LIVE CLASSIFICATION",
        size_label="1kb"
    )
    assert gen_res["algorithm"] == "AES"
    assert gen_res["ciphertext_bytes_count"] >= 1024
    print(f"  ✓ Generated new AES ciphertext: {gen_res['ciphertext_bytes_count']} bytes | SHA-256: {gen_res['sha256'][:16]}...")

    # 4. Start HTTP Server and test new REST endpoints
    print("\n--- 4. Testing Upgraded REST API Endpoints ---")
    server_address = ("127.0.0.1", 8767)
    httpd = http.server.ThreadingHTTPServer(server_address, StandaloneAPIHandler)
    server_thread = threading.Thread(target=run_test_server, args=(httpd,), daemon=True)
    server_thread.start()
    time.sleep(0.3)
    base_url = "http://127.0.0.1:8767"

    # GET /api/architectures
    req = urllib.request.Request(f"{base_url}/api/architectures")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        arch_data = json.loads(resp.read().decode())
        assert len(arch_data["architectures"]) == 7
        print(f"  ✓ GET /api/architectures returned {len(arch_data['architectures'])} architectures")

    # GET /api/sample-files (Verify full 25-sample matrix)
    req = urllib.request.Request(f"{base_url}/api/sample-files")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        samples_data = json.loads(resp.read().decode())
        assert len(samples_data["samples"]) == 25, f"Expected 25 samples, got {len(samples_data['samples'])}"
        sizes_found = set(s["size_label"] for s in samples_data["samples"])
        assert sizes_found == {"1kb", "8kb", "64kb", "256kb", "512kb"}
        print(f"  ✓ GET /api/sample-files returned all 25 samples covering {sizes_found}")

    # POST /api/generate-and-predict
    gen_predict_payload = json.dumps({
        "algorithm": "AES",
        "size": "1kb",
        "text": "Live custom message to test encryption pipeline",
        "task": "binary",
        "architecture": "cnn"
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/api/generate-and-predict",
        data=gen_predict_payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        assert data["success"] is True
        assert data["ground_truth"]["algorithm"] == "AES"
        assert "predicted_cipher" in data
        print(f"  ✓ POST /api/generate-and-predict: Generated AES -> Predicted: {data['predicted_cipher']} (Match: {data['ground_truth']['is_match']})")

    print("\n=================================================================")
    print(" ALL UPGRADED VERIFICATION CHECKS PASSED SUCCESSFULLY! ")
    print("=================================================================\n")
    httpd.shutdown()

if __name__ == "__main__":
    main()
