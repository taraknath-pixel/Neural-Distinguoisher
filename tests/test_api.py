"""
Test suite to verify the REST API endpoints:
- /api/health
- /api/models
- /api/sample-files
- /api/predict (raw binary upload)
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

def run_test_server(server):
    server.serve_forever()

def main():
    print("=================================================================")
    print(" Testing Crypto Identifier REST API Server Endpoints")
    print("=================================================================")
    
    server_address = ("127.0.0.1", 8765)
    httpd = http.server.ThreadingHTTPServer(server_address, StandaloneAPIHandler)
    server_thread = threading.Thread(target=run_test_server, args=(httpd,), daemon=True)
    server_thread.start()
    time.sleep(0.3)
    base_url = "http://127.0.0.1:8765"
    
    # 1. Test /api/health
    print("\n--- 1. Testing GET /api/health ---")
    req = urllib.request.Request(f"{base_url}/api/health")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        assert data["status"] == "healthy"
        print(f"✓ Health Check OK: {data}")

    # 2. Test /api/models
    print("\n--- 2. Testing GET /api/models ---")
    req = urllib.request.Request(f"{base_url}/api/models")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        assert len(data["models"]) == 2
        assert data["feature_count"] == 49
        print(f"✓ Models API OK: Found {len(data['models'])} registered models with {data['feature_count']} NIST features")

    # 3. Test /api/sample-files
    print("\n--- 3. Testing GET /api/sample-files ---")
    req = urllib.request.Request(f"{base_url}/api/sample-files")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        assert "samples" in data
        assert len(data["samples"]) > 0
        sample_0 = data["samples"][0]
        print(f"✓ Sample Catalog OK: Found {len(data['samples'])} samples (First: {sample_0['algorithm']} {sample_0['size_label']})")

    # 4. Test /api/predict with raw binary file
    print("\n--- 4. Testing POST /api/predict (Multipart Ciphertext Upload) ---")
    sample_path = WORKSPACE_ROOT / "crypto_validation_dataset" / "ciphertext" / "AES" / "1kb" / "sample_0000.bin"
    file_bytes = sample_path.read_bytes()
    
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="task"\r\n\r\n'
        f"multiclass\r\n"
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="sample_0000.bin"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode("utf-8") + file_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

    req = urllib.request.Request(
        f"{base_url}/api/predict",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST"
    )
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        assert data["success"] is True
        assert data["predicted_cipher"] in ["AES", "3DES", "CAST", "RC2", "Blowfish"]
        assert len(data["probabilities"]) == 5
        assert len(data["all_features"]) == 49
        print(f"✓ Upload Prediction OK:")
        print(f"  Predicted Cipher: {data['predicted_cipher']} (Confidence: {data['confidence']:.2%})")
        print(f"  Probabilities: {data['probabilities']}")
        print(f"  Experimental Status: {data['experimental_notice']['status']}")

    print("\n=================================================================")
    print(" ALL REST API ENDPOINTS VERIFIED SUCCESSFULLY! ")
    print("=================================================================\n")
    httpd.shutdown()

if __name__ == "__main__":
    main()
