"""
Verification suite for the Crypto Identifier Backend Inference Layer.

Validates:
1. Model Registry and experimental accuracy labels
2. Artifact loader (PyTorch weights, Scalers)
3. Size-aware model selection
4. NIST feature extraction from real .bin files
5. Normalized softmax probabilities (sum to 1.0)
6. Both Multiclass (5-cipher) and Binary (AES vs 3DES) tasks
"""

import sys
import json
from pathlib import Path
import numpy as np
import torch

# Ensure src is on sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from crypto_identifier.inference import (
    ModelRegistry,
    ModelArtifactLoader,
    CipherPredictor,
    detect_size_bucket,
    NIST_FEATURE_NAMES,
)

def test_model_registry():
    print("\n--- 1. Testing Model Registry ---")
    models = ModelRegistry.list_models()
    assert len(models) == 2, f"Expected 2 models, got {len(models)}"
    
    multi = ModelRegistry.get_model_info("multiclass")
    assert multi["num_classes"] == 5
    assert multi["is_experimental"] is True
    assert multi["accuracy_status"] == "experimental_chance_level"
    assert multi["chance_accuracy"] == 0.20
    assert 0.18 <= multi["measured_test_accuracy"] <= 0.25
    print(f"✓ Multiclass model registered: {multi['name']}")
    print(f"  Accuracy: {multi['measured_test_accuracy']:.1%} (Chance: {multi['chance_accuracy']:.1%}) [Experimental]")

    binary = ModelRegistry.get_model_info("binary")
    assert binary["num_classes"] == 2
    assert binary["is_experimental"] is True
    assert binary["accuracy_status"] == "experimental_chance_level"
    assert binary["chance_accuracy"] == 0.50
    assert 0.45 <= binary["measured_test_accuracy"] <= 0.55
    print(f"✓ Binary model registered: {binary['name']}")
    print(f"  Accuracy: {binary['measured_test_accuracy']:.1%} (Chance: {binary['chance_accuracy']:.1%}) [Experimental]")

def test_size_bucket_detection():
    print("\n--- 2. Testing Size-Aware Selection ---")
    assert detect_size_bucket(1040) == "1kb"
    assert detect_size_bucket(8200) == "8kb"
    assert detect_size_bucket(65550) == "64kb"
    assert detect_size_bucket(262160) == "256kb"
    assert detect_size_bucket(524300) == "512kb"
    print("✓ Size bucket auto-detection correctly maps 1kb, 8kb, 64kb, 256kb, 512kb")

def test_model_and_scaler_loading():
    print("\n--- 3. Testing Model and Scaler Loading ---")
    loader = ModelArtifactLoader()
    
    # Multiclass 256kb
    multi_model, multi_scaler, multi_meta = loader.load("multiclass", "256kb")
    assert isinstance(multi_model, torch.nn.Module)
    assert hasattr(multi_scaler, "transform")
    print(f"✓ Loaded Multiclass 256kb CNN (num_classes=5) and StandardScaler")

    # Binary 64kb
    bin_model, bin_scaler, bin_meta = loader.load("binary", "64kb")
    assert isinstance(bin_model, torch.nn.Module)
    assert hasattr(bin_scaler, "transform")
    print(f"✓ Loaded Binary 64kb CNN (num_classes=2) and StandardScaler")

def test_raw_binary_prediction():
    print("\n--- 4. Testing Raw .bin File Prediction ---")
    predictor = CipherPredictor()
    
    # Locate an existing sample binary file
    sample_file = WORKSPACE_ROOT / "crypto_validation_dataset" / "ciphertext" / "AES" / "1kb" / "sample_0000.bin"
    if not sample_file.exists():
        # Fallback create a dummy 1KB binary block
        sample_data = np.random.bytes(1024)
        print("  Notice: Using synthetic ciphertext for smoke test")
    else:
        sample_data = sample_file.read_bytes()
        print(f"  Loaded real ciphertext sample: {sample_file.name} ({len(sample_data)} bytes)")

    # Run Multiclass prediction
    res_multi = predictor.predict_bytes(sample_data, task="multiclass")
    assert res_multi["success"] is True
    assert res_multi["predicted_cipher"] in ["AES", "3DES", "CAST", "RC2", "Blowfish"]
    assert len(res_multi["probabilities"]) == 5
    prob_sum_multi = sum(res_multi["probabilities"].values())
    assert abs(prob_sum_multi - 1.0) < 0.01, f"Probabilities must sum to 1.0, got {prob_sum_multi}"
    assert len(res_multi["all_features"]) == 49
    assert res_multi["model_metadata"]["is_experimental"] is True
    print(f"✓ Multiclass Prediction: {res_multi['predicted_cipher']} (Confidence: {res_multi['confidence']:.2%})")
    print(f"  Probabilities: {res_multi['probabilities']}")
    print(f"  Features Extracted: {len(res_multi['all_features'])} NIST SP 800-22 tests")

    # Run Binary prediction
    res_bin = predictor.predict_bytes(sample_data, task="binary")
    assert res_bin["success"] is True
    assert res_bin["predicted_cipher"] in ["AES", "3DES"]
    assert len(res_bin["probabilities"]) == 2
    prob_sum_bin = sum(res_bin["probabilities"].values())
    assert abs(prob_sum_bin - 1.0) < 0.01, f"Probabilities must sum to 1.0, got {prob_sum_bin}"
    assert res_bin["model_metadata"]["is_experimental"] is True
    print(f"✓ Binary Prediction: {res_bin['predicted_cipher']} (Confidence: {res_bin['confidence']:.2%})")
    print(f"  Probabilities: {res_bin['probabilities']}")

def test_feature_vector_prediction():
    print("\n--- 5. Testing Direct Feature Vector Prediction ---")
    predictor = CipherPredictor()
    dummy_features = [0.5] * 49
    res = predictor.predict_features(dummy_features, task="multiclass", size="256kb")
    assert res["success"] is True
    assert len(res["probabilities"]) == 5
    print("✓ Successfully predicted from direct 49-dimensional feature vector")

def main():
    print("=================================================================")
    print(" Verifying Backend Inference Layer & Experimental Model Registry")
    print("=================================================================")
    test_model_registry()
    test_size_bucket_detection()
    test_model_and_scaler_loading()
    test_raw_binary_prediction()
    test_feature_vector_prediction()
    print("\n=================================================================")
    print(" ALL INFERENCE BACKEND VERIFICATION CHECKS PASSED SUCCESSFULLY! ")
    print("=================================================================\n")

if __name__ == "__main__":
    main()
