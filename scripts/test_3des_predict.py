import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from crypto_identifier.inference import CipherPredictor, encrypt_data
from crypto_identifier.data_generation import DEFAULT_KEYS
from cryptography.hazmat.primitives.ciphers import Cipher, modes
from cryptography.hazmat.decrepit.ciphers.algorithms import TripleDES
from cryptography.hazmat.primitives import padding

p = CipherPredictor()

print("=== BENCHMARK KEY 3DES SAMPLES ===")
for sz in [8, 16, 24, 32, 40, 48, 64]:
    ct = encrypt_data("3DES", b"A" * (sz - 8))
    res = p.predict_bytes(ct, architecture="all")
    conf = res["confidence"] * 100
    print(f"3DES {sz}B (Benchmark Key): {res['predicted_cipher']} (conf: {conf:.1f}%)")

print("\n=== RANDOM KEY 3DES SAMPLES ===")
for sz in [8, 16, 24, 32, 40, 48, 64]:
    k = os.urandom(24)
    c = Cipher(TripleDES(k), modes.ECB()).encryptor()
    padder = padding.PKCS7(64).padder()
    ct = (c.update(padder.update(b"A" * (sz - 8)) + padder.finalize()) + c.finalize())[:sz]
    res = p.predict_bytes(ct, architecture="all")
    comp = res.get("architecture_comparison", {})
    votes = [d["predicted_cipher"] for d in comp.values()]
    print(f"3DES {sz}B (Random Key): {res['predicted_cipher']} | Votes: {votes}")
