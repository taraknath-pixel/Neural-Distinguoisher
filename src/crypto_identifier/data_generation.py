#!/usr/bin/env python3
"""
Data Generation Script for ML-Based Cryptographic Algorithm Identification

This script implements the "DATASET CONSTRUCTION" phase of the replication plan
for Identification of Cryptographic Algorithms using ML Application.

It performs the following steps:
1. Generates plaintexts of specified sizes (1KB, 8KB, 64KB, 256KB, 512KB) by
   concatenating random UUIDs (via Python's `uuid` module).
2. Encrypts these plaintexts using five block ciphers (AES, 3DES, CAST-128, RC2,
   and Blowfish) in ECB mode using fixed keys.
3. Saves the resulting plaintext and ciphertext files into a structured directory
   layout ready for the subsequent feature extraction and machine learning phases.

Required Library:
    `cryptography` (standard and decrepit ciphers modules)

Usage:
    python3 data_generator.py --output_dir ./cryptography_dataset --samples 100
"""

import os
import sys
import uuid
import argparse
import logging
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.decrepit.ciphers.algorithms import TripleDES, Blowfish, CAST5, RC2

from cryptography.hazmat.primitives import padding

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("DataGenerator")

# Standard fixed keys (as specified in the paper: ECB mode with fixed keys)
DEFAULT_KEYS = {
    "AES": b"AES_Fixed_Key16B",       # 16 bytes (128-bit key)
    "3DES": b"TripleDES_Fixed_Key_24B!", # 24 bytes (192-bit key)
    "CAST": b"CAST_Fixed_Key16",       # 16 bytes (128-bit key)
    "RC2": b"RC2_Fixed_Key_16",       # 16 bytes (128-bit key)
    "Blowfish": b"Blowfish_Fixed16"    # 16 bytes (128-bit key)
}

# Standard file sizes in KB and bytes
FILE_SIZES = {
    "1kb": 1024,
    "8kb": 8192,
    "64kb": 65536,
    "256kb": 262144,
    "512kb": 524288
}


def generate_uuid_plaintext(size_bytes: int) -> bytes:
    """
    Generates a plaintext block of a specific size by concatenating random UUID bytes.
    This exactly mirrors the plaintext generation methodology described in the paper.
    """
    plaintext = b""
    while len(plaintext) < size_bytes:
        plaintext += uuid.uuid4().bytes
    return plaintext[:size_bytes]


def pad_data(data: bytes, block_size_bits: int) -> bytes:
    """
    Applies standard PKCS7 padding to make data align with block size.
    """
    padder = padding.PKCS7(block_size_bits).padder()
    return padder.update(data) + padder.finalize()


def encrypt_data(algorithm_name: str, plaintext: bytes, key: bytes) -> bytes:
    """
    Encrypts the plaintext using the specified block cipher in ECB mode.
    Handles standard ECB and applies workarounds for environment-specific limitations.
    """
    if algorithm_name == "AES":
        cipher = Cipher(algorithms.AES(key), modes.ECB())
        padded_data = pad_data(plaintext, 128)  # AES block size: 128 bits
        encryptor = cipher.encryptor()
        return encryptor.update(padded_data) + encryptor.finalize()

    elif algorithm_name == "3DES":
        cipher = Cipher(TripleDES(key), modes.ECB())
        padded_data = pad_data(plaintext, 64)  # 3DES block size: 64 bits
        encryptor = cipher.encryptor()
        return encryptor.update(padded_data) + encryptor.finalize()

    elif algorithm_name == "CAST":
        cipher = Cipher(CAST5(key), modes.ECB())
        padded_data = pad_data(plaintext, 64)  # CAST5 block size: 64 bits
        encryptor = cipher.encryptor()
        return encryptor.update(padded_data) + encryptor.finalize()

    elif algorithm_name == "Blowfish":
        cipher = Cipher(Blowfish(key), modes.ECB())
        padded_data = pad_data(plaintext, 64)  # Blowfish block size: 64 bits
        encryptor = cipher.encryptor()
        return encryptor.update(padded_data) + encryptor.finalize()

    elif algorithm_name == "RC2":
        # Note: OpenSSL backend in some environments doesn't support RC2 in ECB mode.
        # To maintain 100% mathematical equivalence to ECB mode, we encrypt each
        # block (8 bytes / 64 bits) individually in CBC mode with a zero IV.
        # Since CBC with a zero IV on a single block is mathematically identical to ECB:
        #   C_1 = E_K(P_1 ^ IV) = E_K(P_1 ^ 0) = E_K(P_1)
        padded_data = pad_data(plaintext, 64)
        blocks = [padded_data[i:i+8] for i in range(0, len(padded_data), 8)]
        ciphertext_blocks = []
        zero_iv = b"\x00" * 8
        for block in blocks:
            cipher = Cipher(RC2(key), modes.CBC(zero_iv))
            encryptor = cipher.encryptor()
            ciphertext_blocks.append(encryptor.update(block) + encryptor.finalize())
        return b"".join(ciphertext_blocks)

    else:
        raise ValueError(f"Unsupported algorithm: {algorithm_name}")


def generate_dataset(output_dir: str, samples: int = 100, dry_run: bool = False) -> None:
    """Generate plaintexts and ciphertexts for every supported block cipher."""
    if samples < 1:
        raise ValueError("samples must be at least 1")

    # Determine sizes and sample count based on run mode
    sizes_to_generate = FILE_SIZES
    samples_count = samples

    if dry_run:
        logger.info("Dry-run mode activated. Generating 4 samples for 1kb and 8kb only.")
        sizes_to_generate = {"1kb": 1024, "8kb": 8192}
        # Four samples per class permits a small train/validation/test split.
        samples_count = 4

    # Create root directories
    plaintext_base_dir = os.path.join(output_dir, "plaintext")
    ciphertext_base_dir = os.path.join(output_dir, "ciphertext")
    os.makedirs(plaintext_base_dir, exist_ok=True)
    os.makedirs(ciphertext_base_dir, exist_ok=True)

    # Pre-generate plaintexts to ensure the same plaintexts are used across ciphers
    # This guarantees consistent comparison across the different algorithms
    logger.info("Step 1: Generating plaintext files using random UUID strings...")
    
    # Store plaintext paths in a dictionary to fetch during encryption
    plaintext_files = {size: [] for size in sizes_to_generate}

    for size_label, size_bytes in sizes_to_generate.items():
        size_dir = os.path.join(plaintext_base_dir, size_label)
        os.makedirs(size_dir, exist_ok=True)
        
        logger.info(f"Generating {samples_count} plaintext files of size {size_label}...")
        for i in range(samples_count):
            plaintext_data = generate_uuid_plaintext(size_bytes)
            file_name = f"sample_{i:04d}.bin"
            file_path = os.path.join(size_dir, file_name)
            
            with open(file_path, "wb") as f:
                f.write(plaintext_data)
            plaintext_files[size_label].append(file_path)

    # Step 2: Encrypt plaintexts with all 5 algorithms
    logger.info("Step 2: Encrypting plaintext files with block ciphers...")
    algorithms_list = list(DEFAULT_KEYS.keys())

    for algo in algorithms_list:
        algo_key = DEFAULT_KEYS[algo]
        logger.info(f"Encrypting with algorithm: {algo} (Key: {algo_key.decode('ascii', errors='replace')})")
        
        for size_label in sizes_to_generate.keys():
            cipher_dir = os.path.join(ciphertext_base_dir, algo, size_label)
            os.makedirs(cipher_dir, exist_ok=True)
            
            # Print a progress indicator for larger batches
            logger.info(f"  Processing size {size_label}...")
            
            for i, pt_path in enumerate(plaintext_files[size_label]):
                # Read plaintext
                with open(pt_path, "rb") as f:
                    pt_data = f.read()
                
                # Encrypt
                try:
                    ct_data = encrypt_data(algo, pt_data, algo_key)
                    
                    # Save ciphertext
                    ct_file_name = f"sample_{i:04d}.bin"
                    ct_path = os.path.join(cipher_dir, ct_file_name)
                    with open(ct_path, "wb") as f:
                        f.write(ct_data)
                except Exception as e:
                    logger.error(f"Failed to encrypt sample {i} for {algo} of size {size_label}: {e}")
                    raise

    logger.info("Dataset generation completed successfully!")
    logger.info(f"All files have been written to: {os.path.abspath(output_dir)}")
    logger.info(f"Structure:")
    logger.info(f"  - Plaintexts: {plaintext_base_dir}/<size>/sample_<id>.bin")
    logger.info(f"  - Ciphertexts: {ciphertext_base_dir}/<algorithm>/<size>/sample_<id>.bin")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for dataset generation."""
    parser = argparse.ArgumentParser(
        description="Dataset generator for CNN-based block cipher identification."
    )
    parser.add_argument(
        "--output-dir", "--output_dir", dest="output_dir", default="crypto_validation_dataset",
        help="Directory to store the generated dataset."
    )
    parser.add_argument("--samples", type=int, default=100,
                        help="Samples per file size and algorithm (default: 100).")
    parser.add_argument("--dry-run", "--dry_run", dest="dry_run", action="store_true",
                        help="Generate four 1 KB and 8 KB samples per algorithm.")
    return parser


def main(argv=None) -> None:
    """CLI entry point for dataset generation."""
    args = build_parser().parse_args(argv)
    generate_dataset(args.output_dir, args.samples, args.dry_run)


if __name__ == "__main__":
    main()
