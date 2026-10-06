#!/usr/bin/env python3
"""
NIST SP 800-22 Feature Extractor for Cryptographic Algorithm Identification

This script automates the extraction of 49-dimensional statistical features
from encrypted ciphertext files 

The features correspond to P-values from 9 selected NIST SP 800-22 randomness tests:
  1. Frequency (Monobit) Test (1 feature)
  2. Block Frequency Test (1 feature)
  3. Runs Test (1 feature)
  4. Longest Run of Ones Test (1 feature)
  5. Binary Matrix Rank Test (1 feature)
  6. Discrete Fourier Transform (Spectral) Test (1 feature)
  7. Cumulative Sums (Cusum) Test (Forward) (1 feature)
  8. Cumulative Sums (Cusum) Test (Backward) (1 feature)
  9. Approximate Entropy Test (1 feature)
  10. Intra-Block Entropy (4-bit serial test) (1 feature)
  11. Non-overlapping Template Matching Test (39 features for 39 unique templates of length 6)

Author: Kapil Jaiswal
Date: August 2026
"""

import os
import sys
import argparse
import logging
import time
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from scipy.special import erfc, gammaincc

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("NISTFeatureExtractor")

# -----------------------------------------------------------------------------
# NIST SP 800-22 Core Implementations (Fast NumPy Vectorized Versions)
# -----------------------------------------------------------------------------

def monobit_test(bits):
    n = len(bits)
    if n == 0:
        return 0.5
    s = np.sum(2 * bits - 1)
    s_obs = abs(s) / np.sqrt(n)
    p_val = erfc(s_obs / np.sqrt(2.0))
    return p_val

def block_frequency_test(bits, block_size=128):
    n = len(bits)
    num_blocks = n // block_size
    if num_blocks == 0:
        return 0.5
    blocks = bits[:num_blocks * block_size].reshape(num_blocks, block_size)
    proportions = np.mean(blocks, axis=1)
    chi_sq = 4.0 * block_size * np.sum((proportions - 0.5) ** 2)
    p_val = gammaincc(num_blocks / 2.0, chi_sq / 2.0)
    return p_val

def runs_test(bits):
    n = len(bits)
    if n == 0:
        return 0.5
    pi = np.mean(bits)
    diffs = np.diff(bits)
    v_n = 1 + np.sum(diffs != 0)
    num = abs(v_n - 2.0 * n * pi * (1.0 - pi))
    den = 2.0 * np.sqrt(2.0 * n) * pi * (1.0 - pi)
    if den == 0:
        return 0.5
    p_val = erfc(num / den)
    return p_val

def longest_run_of_ones_test(bits):
    n = len(bits)
    if n < 128:
        M = 8
        pi = [0.2148, 0.3672, 0.2305, 0.1875]
        classes = 4
    else:
        M = 128
        pi = [0.1174, 0.2430, 0.2493, 0.1752, 0.1027, 0.1124]
        classes = 6
        
    num_blocks = n // M
    if num_blocks == 0:
        return 0.5
        
    v = np.zeros(classes)
    for i in range(num_blocks):
        block = bits[i*M : (i+1)*M]
        max_run = 0
        current_run = 0
        for bit in block:
            if bit == 1:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
        # Categorize
        if M == 8:
            if max_run <= 1: v[0] += 1
            elif max_run == 2: v[1] += 1
            elif max_run == 3: v[2] += 1
            else: v[3] += 1
        else:
            if max_run <= 4: v[0] += 1
            elif max_run == 5: v[1] += 1
            elif max_run == 6: v[2] += 1
            elif max_run == 7: v[3] += 1
            elif max_run == 8: v[4] += 1
            else: v[5] += 1
            
    chi_sq = 0.0
    for i in range(classes):
        expected = num_blocks * pi[i]
        if expected > 0:
            chi_sq += ((v[i] - expected) ** 2) / expected
        
    p_val = gammaincc((classes - 1) / 2.0, chi_sq / 2.0)
    return p_val

def get_binary_matrix_rank(matrix):
    M, Q = matrix.shape
    mat = matrix.copy()
    rank = 0
    for col in range(Q):
        pivot = -1
        for row in range(rank, M):
            if mat[row, col] == 1:
                pivot = row
                break
        if pivot != -1:
            if pivot != rank:
                mat[[rank, pivot]] = mat[[pivot, rank]]
            for row in range(rank + 1, M):
                if mat[row, col] == 1:
                    mat[row] ^= mat[rank]
            rank += 1
            if rank == M:
                break
    return rank

def binary_matrix_rank_test(bits):
    n = len(bits)
    M = 32
    Q = 32
    block_size = M * Q # 1024 bits
    num_blocks = n // block_size
    if num_blocks == 0:
        return 0.4984
        
    f_32 = 0
    f_31 = 0
    f_others = 0
    
    for i in range(num_blocks):
        block = bits[i*block_size : (i+1)*block_size]
        matrix = block.reshape(M, Q)
        rank = get_binary_matrix_rank(matrix)
        if rank == 32:
            f_32 += 1
        elif rank == 31:
            f_31 += 1
        else:
            f_others += 1
            
    p_32 = 0.2888
    p_31 = 0.5776
    p_others = 0.1336
    
    chi_sq = (((f_32 - num_blocks * p_32) ** 2) / (num_blocks * p_32)) + \
             (((f_31 - num_blocks * p_31) ** 2) / (num_blocks * p_31)) + \
             (((f_others - num_blocks * p_others) ** 2) / (num_blocks * p_others))
             
    p_val = gammaincc(1.0, chi_sq / 2.0)
    return p_val

def discrete_fourier_transform_test(bits):
    n = len(bits)
    if n == 0:
        return 0.5
    s = 2 * bits - 1
    f = np.fft.fft(s)
    modulus = np.abs(f[:n//2])
    T = np.sqrt(2.995732274 * n)
    N_1 = np.sum(modulus < T)
    N_0 = 0.95 * n / 2.0
    d = (N_1 - N_0) / np.sqrt(n * 0.95 * 0.05 / 4.0)
    p_val = erfc(abs(d) / np.sqrt(2.0))
    return p_val

def cumulative_sums_test(bits, mode='forward'):
    n = len(bits)
    if n == 0:
        return 0.5
    x = 2 * bits - 1
    if mode == 'backward':
        x = x[::-1]
    s = np.cumsum(x)
    z = np.max(np.abs(s))
    if z == 0:
        return 1.0
        
    # Helper to calculate the normal cumulative distribution function
    def phi(val):
        return 0.5 * erfc(-val / np.sqrt(2.0))
        
    sum1 = 0.0
    k_start = int(np.floor((-n/z + 1.0)/4.0))
    k_end = int(np.floor((n/z - 1.0)/4.0))
    for k in range(k_start, k_end + 1):
        sum1 += phi((4*k + 1)*z / np.sqrt(n)) - phi((4*k - 1)*z / np.sqrt(n))
        
    sum2 = 0.0
    k_start2 = int(np.floor((-n/z - 3.0)/4.0))
    k_end2 = int(np.floor((n/z - 1.0)/4.0))
    for k in range(k_start2, k_end2 + 1):
        sum2 += phi((4*k + 3)*z / np.sqrt(n)) - phi((4*k + 1)*z / np.sqrt(n))
        
    p_val = 1.0 - sum1 + sum2
    return min(max(p_val, 0.0), 1.0)

def approximate_entropy_test(bits, m=2):
    n = len(bits)
    if n == 0:
        return 0.5
    
    def fast_apen(b, block_len):
        aug = np.concatenate([b, b[:block_len]])
        shape = (n, block_len)
        strides = (aug.strides[0], aug.strides[0])
        windows = np.lib.stride_tricks.as_strided(aug, shape=shape, strides=strides)
        powers = 2 ** np.arange(block_len)[::-1]
        ints = np.dot(windows, powers)
        counts = np.bincount(ints, minlength=2**block_len)
        non_zero = counts[counts > 0]
        p = non_zero / n
        return np.sum(p * np.log(p))
        
    phi_m = fast_apen(bits, m)
    phi_m1 = fast_apen(bits, m+1)
    apen = phi_m - phi_m1
    chi_sq = 2.0 * n * (np.log(2.0) - apen)
    p_val = gammaincc(2**(m-1), chi_sq / 2.0)
    return p_val

def intra_block_entropy_test(bits, block_size=4):
    n = len(bits)
    num_blocks = n // block_size
    if num_blocks == 0:
        return 0.5
        
    blocks = bits[:num_blocks * block_size].reshape(num_blocks, block_size)
    powers = 2 ** np.arange(block_size)[::-1]
    ints = np.sum(blocks * powers, axis=1)
    counts = np.bincount(ints, minlength=16)
    expected = num_blocks / 16.0
    
    chi_sq = np.sum((counts - expected) ** 2) / expected
    p_val = gammaincc(15 / 2.0, chi_sq / 2.0)
    return p_val

def batch_non_overlapping_template_matching(bits, templates, M=256):
    n = len(bits)
    m = 6
    num_blocks = n // M
    if num_blocks == 0:
        return [0.5] * len(templates)
        
    mu = (M - m + 1) / (2**m)
    var = M * ((1 / (2**m)) - ((2*m - 1) / (2**(2*m))))
    eval_blocks = min(num_blocks, 256)
    
    powers = 2 ** np.arange(m, dtype=np.int32)[::-1]
    shape = (eval_blocks, M - m + 1, m)
    strides = (bits.strides[0] * M, bits.strides[0], bits.strides[0])
    windows = np.lib.stride_tricks.as_strided(bits[:eval_blocks * M], shape=shape, strides=strides)
    ints = np.dot(windows, powers)
    
    p_vals = []
    for temp in templates:
        t_val = int(np.dot(temp, powers))
        mask = (ints == t_val)
        counts = np.zeros(eval_blocks, dtype=np.int32)
        rows, cols = np.where(mask)
        if len(rows) > 0:
            last_r = -1
            last_c = -m
            for r, c in zip(rows, cols):
                if r != last_r:
                    last_r = r
                    last_c = c
                    counts[r] += 1
                elif c >= last_c + m:
                    last_c = c
                    counts[r] += 1
        chi_sq = np.sum((counts - mu)**2) / var
        p_val = float(gammaincc(eval_blocks / 2.0, chi_sq / 2.0))
        p_vals.append(min(max(p_val, 0.0), 1.0))
    return p_vals


def non_overlapping_template_matching_test(bits, template, M=256):
    res = batch_non_overlapping_template_matching(bits, [template], M=M)
    return res[0]

# -----------------------------------------------------------------------------
# Pipeline Orchestration
# -----------------------------------------------------------------------------

def extract_features_from_bits(bits):
    """
    Extracts all 49 features from the raw bits numpy array.
    Guarantees the bits are in a signed 32-bit integer type to prevent
    subtraction overflow/wrap-around bugs.
    """
    # Force signed integer representations to prevent underflow wrap-around (e.g. 2*bits - 1 in uint8)
    bits = np.asarray(bits, dtype=np.int32)
    
    features = []
    
    # 1. Monobit (1)
    features.append(monobit_test(bits))
    
    # 2. Block Frequency (1)
    features.append(block_frequency_test(bits, block_size=128))
    
    # 3. Runs (1)
    features.append(runs_test(bits))
    
    # 4. Longest Run of Ones (1)
    features.append(longest_run_of_ones_test(bits))
    
    # 5. Binary Matrix Rank (1)
    features.append(binary_matrix_rank_test(bits))
    
    # 6. Spectral DFT (1)
    features.append(discrete_fourier_transform_test(bits))
    
    # 7. Cusum Forward (1)
    features.append(cumulative_sums_test(bits, mode='forward'))
    
    # 8. Cusum Backward (1)
    features.append(cumulative_sums_test(bits, mode='backward'))
    
    # 9. Approximate Entropy (1)
    features.append(approximate_entropy_test(bits, m=2))
    
    # 10. Intra-block Entropy (1)
    features.append(intra_block_entropy_test(bits, block_size=4))
    
    # 11. Non-overlapping Template Matching (39 templates)
    templates = [np.array([int(b) for b in format(i, '06b')], dtype=np.int32) for i in range(1, 40)]
    if len(bits) < 1024:
        features.extend([0.5342] * len(templates))
    else:
        features.extend(batch_non_overlapping_template_matching(bits, templates, M=256))
        
    return features

def process_single_file(file_path, label, size_label, sample_id):
    """
    Processes a single binary file to extract its 49-dimensional features.
    """
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        # Convert bytes to a numpy array of bits (0s and 1s)
        bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
        features = extract_features_from_bits(bits)
        return {
            "success": True,
            "label": label,
            "size": size_label,
            "sample_id": sample_id,
            "features": features
        }
    except Exception as e:
        return {
            "success": False,
            "file": file_path,
            "error": str(e)
        }

def extract_features(input_dir: str, output_prefix: str, workers: int = 4) -> tuple[str, str]:
    """Extract 49 NIST features from ciphertext files and save CSV/NPZ outputs."""
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
        
    # Find all ciphertext files
    tasks = []
    # Structure of input: input_dir/algorithm/size/*.bin
    for algo in os.listdir(input_dir):
        algo_path = os.path.join(input_dir, algo)
        if not os.path.isdir(algo_path):
            continue
        for size in os.listdir(algo_path):
            size_path = os.path.join(algo_path, size)
            if not os.path.isdir(size_path):
                continue
            for fname in os.listdir(size_path):
                if not fname.endswith(".bin"):
                    continue
                file_path = os.path.join(size_path, fname)
                # Parse sample ID from filename 'sample_<id>.bin'
                try:
                    sample_id = int(fname.split("_")[1].split(".")[0])
                except:
                    sample_id = 0
                tasks.append((file_path, algo, size, sample_id))
                
    if not tasks:
        raise ValueError(f"No ciphertext .bin files found in {input_dir}")
        
    logger.info(f"Found {len(tasks)} ciphertext files to extract features from.")
    logger.info(f"Starting extraction with {workers} workers...")
    
    start_time = time.time()
    results = []
    failed = 0
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_single_file, t[0], t[1], t[2], t[3]): t
            for t in tasks
        }
        
        completed_count = 0
        for future in as_completed(futures):
            res = future.result()
            completed_count += 1
            if res["success"]:
                results.append(res)
            else:
                failed += 1
                logger.error(f"Failed to process {res['file']}: {res['error']}")
                
            if completed_count % 50 == 0 or completed_count == len(tasks):
                logger.info(f"Progress: {completed_count}/{len(tasks)} files processed...")
                
    elapsed_time = time.time() - start_time
    logger.info(f"Feature extraction completed in {elapsed_time:.2f} seconds.")
    logger.info(f"Successfully processed {len(results)} files. Failed: {failed}.")
    
    if not results:
        raise RuntimeError("No features were successfully extracted.")
        
    # Save results as a CSV file
    output_parent = os.path.dirname(os.path.abspath(output_prefix))
    os.makedirs(output_parent, exist_ok=True)
    csv_file = f"{output_prefix}.csv"
    logger.info(f"Writing features to CSV: {csv_file}")
    
    # Define header: algorithm, size, sample_id, f_0, f_1, ..., f_48
    header = ["algorithm", "size", "sample_id"] + [f"f_{i}" for i in range(49)]
    
    try:
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for r in results:
                row = [r["label"], r["size"], r["sample_id"]] + r["features"]
                writer.writerow(row)
        logger.info(f"Successfully saved {csv_file}")
    except Exception as e:
        logger.error(f"Failed to write CSV: {e}")
        
    # Save results as a Compressed NumPy NPZ file for deep learning frameworks
    npz_file = f"{output_prefix}.npz"
    logger.info(f"Writing features to NPZ: {npz_file}")
    
    try:
        X = np.array([r["features"] for r in results], dtype=np.float32)
        y_algo = np.array([r["label"] for r in results])
        y_size = np.array([r["size"] for r in results])
        sample_ids = np.array([r["sample_id"] for r in results], dtype=np.int32)
        
        np.savez_compressed(
            npz_file,
            X=X,
            y_algo=y_algo,
            y_size=y_size,
            sample_ids=sample_ids
        )
        logger.info(f"Successfully saved {npz_file}")
    except Exception as e:
        logger.error(f"Failed to write NPZ: {e}")
        raise
    return csv_file, npz_file


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for feature extraction."""
    parser = argparse.ArgumentParser(description="Extract NIST features from generated ciphertexts.")
    parser.add_argument("--input-dir", "--input_dir", dest="input_dir",
                        default="crypto_validation_dataset/ciphertext",
                        help="Ciphertext root directory.")
    parser.add_argument("--output-prefix", "--output_file", dest="output_prefix",
                        default="crypto_validation_dataset/extracted_features",
                        help="Output path without the .csv or .npz suffix.")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel processes to use.")
    return parser


def main(argv=None) -> None:
    """CLI entry point for feature extraction."""
    args = build_parser().parse_args(argv)
    extract_features(args.input_dir, args.output_prefix, args.workers)

if __name__ == "__main__":
    main()
