"""
Verification suite for:
1. Hexadecimal ciphertext parsing & sanitation
2. CSV format parsing with 'Ciphertext' column
3. Binary classification focus (AES vs 3DES)
4. Out-of-distribution rejection ('Does not belong to AES or 3DES') for non-AES/3DES ciphers
5. End-to-end API and Python predictor interfaces
"""

import sys
import unittest
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from crypto_identifier.inference import (
    CipherPredictor,
    parse_hex_ciphertext,
    parse_csv_ciphertexts,
    REJECTION_LABEL,
    default_predictor,
)


class TestHexCsvBinary(unittest.TestCase):

    def test_parse_hex_valid(self):
        raw = parse_hex_ciphertext("0011223344556677")
        self.assertEqual(raw, bytes.fromhex("0011223344556677"))

        raw2 = parse_hex_ciphertext("0x4a 5f 6e 8c aa bb cc dd")
        self.assertEqual(raw2, bytes.fromhex("4a5f6e8caabbccdd"))

        raw3 = parse_hex_ciphertext("1a:2b:3c:4d:\n5e:6f:7a:8b")
        self.assertEqual(raw3, bytes.fromhex("1a2b3c4d5e6f7a8b"))

    def test_parse_hex_invalid(self):
        with self.assertRaises(ValueError):
            parse_hex_ciphertext("4a5f6")

        with self.assertRaises(ValueError):
            parse_hex_ciphertext("4a5g")

        with self.assertRaises(ValueError):
            parse_hex_ciphertext("   ")

    def test_parse_csv_valid(self):
        csv_data = (
            "ID,Ciphertext,Notes\n"
            "1,4a5f6e8caabbccdd0011223344556677,Sample 1\n"
            "2,0x112233445566778899aabbccddeeff00,Sample 2\n"
        )
        rows = parse_csv_ciphertexts(csv_data)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["row_number"], 1)
        self.assertEqual(rows[0]["raw_hex"], "4a5f6e8caabbccdd0011223344556677")
        self.assertEqual(rows[1]["row_number"], 2)

    def test_parse_csv_missing_column(self):
        csv_data = "ID,Data,Notes\n1,aabbccdd,Test\n"
        with self.assertRaises(ValueError):
            parse_csv_ciphertexts(csv_data)

    def test_binary_prediction_aes(self):
        sample_file = WORKSPACE_ROOT / "crypto_validation_dataset" / "ciphertext" / "AES" / "1kb" / "sample_0000.bin"
        if sample_file.exists():
            raw_bytes = sample_file.read_bytes()
            res = default_predictor.predict_bytes(raw_bytes, task="binary")
            self.assertTrue(res["success"])
            self.assertEqual(res["predicted_cipher"], "AES")
            self.assertTrue(res["belongs_to_target"])
            self.assertGreater(res["probabilities"]["AES"], 0.50)

    def test_binary_prediction_3des(self):
        sample_file = WORKSPACE_ROOT / "crypto_validation_dataset" / "ciphertext" / "3DES" / "1kb" / "sample_0000.bin"
        if sample_file.exists():
            raw_bytes = sample_file.read_bytes()
            res = default_predictor.predict_bytes(raw_bytes, task="binary")
            self.assertTrue(res["success"])
            self.assertEqual(res["predicted_cipher"], "3DES")
            self.assertTrue(res["belongs_to_target"])
            self.assertGreater(res["probabilities"]["3DES"], 0.50)

    def test_binary_prediction_rejection_blowfish(self):
        sample_file = WORKSPACE_ROOT / "crypto_validation_dataset" / "ciphertext" / "Blowfish" / "1kb" / "sample_0000.bin"
        if sample_file.exists():
            raw_bytes = sample_file.read_bytes()
            res = default_predictor.predict_bytes(raw_bytes, task="binary")
            self.assertTrue(res["success"])
            self.assertEqual(res["predicted_cipher"], REJECTION_LABEL)
            self.assertFalse(res["belongs_to_target"])
            self.assertEqual(res["probabilities"][REJECTION_LABEL], 1.0)

    def test_binary_prediction_rejection_cast(self):
        sample_file = WORKSPACE_ROOT / "crypto_validation_dataset" / "ciphertext" / "CAST" / "1kb" / "sample_0000.bin"
        if sample_file.exists():
            raw_bytes = sample_file.read_bytes()
            res = default_predictor.predict_bytes(raw_bytes, task="binary")
            self.assertTrue(res["success"])
            self.assertEqual(res["predicted_cipher"], REJECTION_LABEL)
            self.assertFalse(res["belongs_to_target"])
            self.assertEqual(res["probabilities"][REJECTION_LABEL], 1.0)

    def test_binary_prediction_rejection_rc2(self):
        sample_file = WORKSPACE_ROOT / "crypto_validation_dataset" / "ciphertext" / "RC2" / "1kb" / "sample_0000.bin"
        if sample_file.exists():
            raw_bytes = sample_file.read_bytes()
            res = default_predictor.predict_bytes(raw_bytes, task="binary")
            self.assertTrue(res["success"])
            self.assertEqual(res["predicted_cipher"], REJECTION_LABEL)
            self.assertFalse(res["belongs_to_target"])
            self.assertEqual(res["probabilities"][REJECTION_LABEL], 1.0)

    def test_binary_prediction_rejection_unaligned(self):
        # 7 bytes (less than 8 bytes)
        res = default_predictor.predict_bytes(b"\x01\x02\x03\x04\x05\x06\x07", task="binary")
        self.assertEqual(res["predicted_cipher"], REJECTION_LABEL)
        self.assertFalse(res["belongs_to_target"])

        # 13 bytes (not a multiple of 8 or 16)
        res2 = default_predictor.predict_bytes(b"A" * 13, task="binary")
        self.assertEqual(res2["predicted_cipher"], REJECTION_LABEL)
        self.assertFalse(res2["belongs_to_target"])

    def test_predict_hex_manual_input(self):
        sample_file = WORKSPACE_ROOT / "crypto_validation_dataset" / "ciphertext" / "AES" / "1kb" / "sample_0000.bin"
        if sample_file.exists():
            hex_data = sample_file.read_bytes().hex()
            res = default_predictor.predict_hex(hex_data, task="binary")
            self.assertTrue(res["success"])
            self.assertEqual(res["predicted_cipher"], "AES")
            self.assertTrue(res["belongs_to_target"])
            self.assertEqual(res["input_info"]["input_mode"], "manual_hex")

    def test_predict_csv_batch(self):
        aes_sample = WORKSPACE_ROOT / "crypto_validation_dataset" / "ciphertext" / "AES" / "1kb" / "sample_0000.bin"
        bf_sample = WORKSPACE_ROOT / "crypto_validation_dataset" / "ciphertext" / "Blowfish" / "1kb" / "sample_0000.bin"

        if aes_sample.exists() and bf_sample.exists():
            aes_hex = aes_sample.read_bytes().hex()
            bf_hex = bf_sample.read_bytes().hex()

            csv_content = (
                "Row,Ciphertext\n"
                f"1,{aes_hex}\n"
                f"2,{bf_hex}\n"
                "3,invalid_hex_123g\n"
            )

            res = default_predictor.predict_csv(csv_content, task="binary")
            self.assertTrue(res["success"])
            self.assertEqual(res["total_samples"], 3)
            self.assertEqual(res["aes_count"], 1)
            self.assertEqual(res["not_belong_count"], 2)
            self.assertEqual(res["results"][0]["predicted_cipher"], "AES")
            self.assertEqual(res["results"][1]["predicted_cipher"], REJECTION_LABEL)
            self.assertEqual(res["results"][2]["predicted_cipher"], REJECTION_LABEL)

    def test_short_aes_hex_prediction(self):
        """Verify user's 48-byte AES sample correctly classifies as AES with consensus."""
        user_hex = "86c529c19253a0fc8d1e75f9f84b9975b0eecabd6df2cb471cd04f6023890bb6ed34f0cce1d440baf77ace02e60a5094"
        res = default_predictor.predict_hex(user_hex, task="binary", architecture="all")
        self.assertTrue(res["success"])
        self.assertEqual(res["verdict"], "AES")
        self.assertEqual(res["predicted_cipher"], "AES")
        self.assertTrue(res["belongs_to_target"])
        arch_comp = res.get("architecture_comparison", {})
        votes = [d["predicted_cipher"] for d in arch_comp.values()]
        from collections import Counter
        top_vote = Counter(votes).most_common(1)[0][0]
        self.assertEqual(top_vote, "AES")
        self.assertEqual(res["verdict"], top_vote)

    def test_short_3des_prediction_block_alignment(self):
        """Verify 24-byte 3DES ciphertext is identified as 3DES via block length alignment."""
        import os
        from cryptography.hazmat.primitives.ciphers import Cipher, modes
        from cryptography.hazmat.decrepit.ciphers.algorithms import TripleDES
        key = os.urandom(24)
        pt = b"24_bytes_of_test_block_!"
        c = Cipher(TripleDES(key), modes.ECB()).encryptor()
        ct = c.update(pt) + c.finalize()
        res = default_predictor.predict_bytes(ct, task="binary", architecture="all")
        self.assertTrue(res["success"])
        self.assertEqual(res["verdict"], "3DES")
        self.assertEqual(res["predicted_cipher"], "3DES")
        self.assertTrue(res["belongs_to_target"])
        self.assertGreaterEqual(res["confidence"], 0.99)

    def test_short_aes_32byte_user_sample(self):
        """Verify user's 32-byte AES sample correctly classifies as AES with consensus."""
        user_hex = "ae1b762990724febb8c8c264e9a8f2eea04bede7f517d2d38526a67f20383389"
        res = default_predictor.predict_hex(user_hex, task="binary", architecture="all")
        self.assertTrue(res["success"])
        self.assertEqual(res["verdict"], "AES")
        self.assertEqual(res["predicted_cipher"], "AES")
        self.assertTrue(res["belongs_to_target"])
        arch_comp = res.get("architecture_comparison", {})
        votes = [d["predicted_cipher"] for d in arch_comp.values()]
        from collections import Counter
        top_vote = Counter(votes).most_common(1)[0][0]
        self.assertEqual(top_vote, "AES")
        self.assertEqual(res["verdict"], top_vote)


if __name__ == "__main__":
    unittest.main()
