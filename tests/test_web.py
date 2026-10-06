"""
Verification script for CryptoCipher AI Web UI static file serving.
"""

import sys
import time
import threading
from pathlib import Path
import urllib.request

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

import http.server
from crypto_identifier.api import StandaloneAPIHandler

def run_test_server(server):
    server.serve_forever()

def main():
    print("=================================================================")
    print(" Testing CryptoCipher AI Web UI Static Files")
    print("=================================================================")
    
    server_address = ("127.0.0.1", 8766)
    httpd = http.server.ThreadingHTTPServer(server_address, StandaloneAPIHandler)
    server_thread = threading.Thread(target=run_test_server, args=(httpd,), daemon=True)
    server_thread.start()
    time.sleep(0.3)
    base_url = "http://127.0.0.1:8766"
    
    # 1. Test GET / (index.html)
    print("\n--- 1. Testing GET / (index.html) ---")
    req = urllib.request.Request(f"{base_url}/")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        html = resp.read().decode("utf-8")
        assert "CryptoCipher" in html
        assert "NIST SP 800-22" in html
        assert "dropzone" in html
        print(f"✓ index.html retrieved successfully ({len(html)} bytes)")

    # 2. Test GET /style.css
    print("\n--- 2. Testing GET /style.css ---")
    req = urllib.request.Request(f"{base_url}/style.css")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        css = resp.read().decode("utf-8")
        assert "--accent-cyan" in css
        print(f"✓ style.css retrieved successfully ({len(css)} bytes)")

    # 3. Test GET /app.js
    print("\n--- 3. Testing GET /app.js ---")
    req = urllib.request.Request(f"{base_url}/app.js")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        js = resp.read().decode("utf-8")
        assert "runInference" in js
        print(f"✓ app.js retrieved successfully ({len(js)} bytes)")

    print("\n=================================================================")
    print(" WEB UI STATIC SERVING VERIFIED SUCCESSFULLY! ")
    print("=================================================================\n")
    httpd.shutdown()

if __name__ == "__main__":
    main()
