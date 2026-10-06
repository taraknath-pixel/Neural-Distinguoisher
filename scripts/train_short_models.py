"""
train_short_models.py - Retrain binary_short models with balanced, high-quality data.

Fixes:
  1. Perfectly balanced classes: exactly 1000 samples per class per shared size.
  2. Only trains on shared (16B-multiple) sizes where AES vs 3DES disambiguation is needed.
  3. Better ML hyperparameters tuned for 10 active features.
  4. CNN with BatchNorm + Dropout for better generalisation.
  5. Per-size accuracy breakdown to verify balanced performance.
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.decrepit.ciphers.algorithms import TripleDES
from cryptography.hazmat.primitives import padding
from crypto_identifier.feature_extraction import extract_features_from_bits
from crypto_identifier.inference import DEFAULT_KEYS


# Sizes where both AES and 3DES produce same-length ciphertext (hard cases)
SHARED_SIZES = [16, 32, 48, 64, 80, 96, 112, 128]
SAMPLES_PER_CLASS_PER_SIZE = 1000   # 8x more than before


def gen_aes_bytes(sz: int) -> bytes:
    r = np.random.randint(0, 6)
    key = DEFAULT_KEYS["AES"] if r == 0 else os.urandom(16)
    if r % 2 == 0:
        enc = Cipher(algorithms.AES(key), modes.ECB()).encryptor()
    else:
        enc = Cipher(algorithms.AES(key), modes.CBC(os.urandom(16))).encryptor()
    pt = os.urandom(max(sz, 1))
    padder = padding.PKCS7(128).padder()
    ct = enc.update(padder.update(pt) + padder.finalize()) + enc.finalize()
    return ct[:sz]


def gen_3des_bytes(sz: int) -> bytes:
    r = np.random.randint(0, 6)
    key = DEFAULT_KEYS["3DES"] if r == 0 else os.urandom(24)
    if r % 2 == 0:
        enc = Cipher(TripleDES(key), modes.ECB()).encryptor()
    else:
        enc = Cipher(TripleDES(key), modes.CBC(os.urandom(8))).encryptor()
    pt = os.urandom(max(sz, 1))
    padder = padding.PKCS7(64).padder()
    ct = enc.update(padder.update(pt) + padder.finalize()) + enc.finalize()
    return ct[:sz]


def feats(ct: bytes) -> list:
    bits = np.unpackbits(np.frombuffer(ct, dtype=np.uint8))
    return extract_features_from_bits(bits)


class ShortNet(nn.Module):
    """Lightweight MLP with BN + Dropout optimised for 49-feat short inputs."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(49, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),  nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32),   nn.ReLU(),
            nn.Linear(32, 2),
        )
    def forward(self, x):
        return self.net(x)


def train_short_models():
    np.random.seed(42)
    torch.manual_seed(42)

    # -----------------------------------------------------------------------
    # 1. Generate perfectly balanced dataset on shared sizes only
    # -----------------------------------------------------------------------
    print("1. Generating balanced training dataset (shared sizes only)...")
    X_list, y_list, sz_list = [], [], []

    for sz in SHARED_SIZES:
        print(f"   {sz:3d}B: {SAMPLES_PER_CLASS_PER_SIZE} AES + {SAMPLES_PER_CLASS_PER_SIZE} 3DES")
        for _ in range(SAMPLES_PER_CLASS_PER_SIZE):
            X_list.append(feats(gen_aes_bytes(sz)));   y_list.append(0); sz_list.append(sz)
        for _ in range(SAMPLES_PER_CLASS_PER_SIZE):
            X_list.append(feats(gen_3des_bytes(sz)));  y_list.append(1); sz_list.append(sz)

    X   = np.array(X_list,  dtype=np.float32)
    y   = np.array(y_list,  dtype=np.int32)
    szs = np.array(sz_list, dtype=np.int32)

    print(f"\nTotal: {len(X)} samples | AES={np.sum(y==0)} | 3DES={np.sum(y==1)}")
    assert np.sum(y==0) == np.sum(y==1), "Dataset is not balanced!"

    # Shuffle preserving size info
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(X))
    X, y, szs = X[idx], y[idx], szs[idx]

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    szs_test = szs[split:]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    out_dir = Path(__file__).resolve().parent.parent / "src" / "crypto_identifier" / "packaged_models" / "binary_short"
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"scaler": scaler, "feature_names": [f"f_{i}" for i in range(49)]}, out_dir / "scaler.joblib")

    # -----------------------------------------------------------------------
    # 2. Train improved ML models
    # -----------------------------------------------------------------------
    ml_models = {
        "rf":  RandomForestClassifier(n_estimators=500, min_samples_leaf=2, random_state=42, n_jobs=-1),
        "svm": SVC(probability=True, kernel="rbf", C=10, gamma="scale", random_state=42),
        "mlp": MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=1000,
                             alpha=1e-3, early_stopping=True, validation_fraction=0.1,
                             random_state=42),
        "knn": KNeighborsClassifier(n_neighbors=9, weights="distance"),
        "lr":  LogisticRegression(C=1.0, max_iter=2000, random_state=42),
        "gnb": GaussianNB(var_smoothing=1e-8),
    }

    print("\n2. Training improved ML models...")
    all_preds_te = {}
    for name, m in ml_models.items():
        m.fit(X_tr, y_train)
        preds = m.predict(X_te)
        all_preds_te[name] = preds
        acc = accuracy_score(y_test, preds)
        print(f"   {name.upper():4s} test accuracy: {acc:.4f}")
        joblib.dump({"model": m, "arch": name, "classes": ["AES", "3DES"]}, out_dir / f"{name}.joblib")

    # -----------------------------------------------------------------------
    # 3. Train improved CNN
    # -----------------------------------------------------------------------
    print("\n3. Training improved CNN (BatchNorm + Dropout, 80 epochs)...")
    cnn = ShortNet()
    crit = nn.CrossEntropyLoss()
    opt  = optim.Adam(cnn.parameters(), lr=0.001, weight_decay=1e-4)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=8, factor=0.5)

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t  = torch.tensor(y_train, dtype=torch.long)
    X_te_t  = torch.tensor(X_te,  dtype=torch.float32)

    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=64, shuffle=True)

    best_acc, best_state = 0.0, None
    for epoch in range(80):
        cnn.train()
        for bx, by in loader:
            opt.zero_grad()
            crit(cnn(bx), by).backward()
            opt.step()
        cnn.eval()
        with torch.no_grad():
            vp = torch.argmax(cnn(X_te_t), dim=1).numpy()
        va = accuracy_score(y_test, vp)
        sched.step(1.0 - va)
        if va > best_acc:
            best_acc  = va
            best_state = {k: v.clone() for k, v in cnn.state_dict().items()}
        if (epoch + 1) % 20 == 0:
            print(f"   Epoch {epoch+1:3d} | val_acc={va:.4f} | best={best_acc:.4f}")

    cnn.load_state_dict(best_state)
    cnn.eval()
    with torch.no_grad():
        cnn_preds = torch.argmax(cnn(X_te_t), dim=1).numpy()
    all_preds_te["cnn"] = cnn_preds
    print(f"   CNN  test accuracy (best): {accuracy_score(y_test, cnn_preds):.4f}")
    torch.save({"state_dict": best_state, "classes": ["AES", "3DES"]}, out_dir / "best_cnn.pt")

    # -----------------------------------------------------------------------
    # 4. Ensemble summary
    # -----------------------------------------------------------------------
    arch_order = ["rf", "svm", "mlp", "knn", "lr", "gnb", "cnn"]
    all_arr = np.stack([all_preds_te[k] for k in arch_order], axis=0)  # (7, N)
    ens_preds = (np.mean(all_arr, axis=0) > 0.5).astype(int)
    ens_acc   = accuracy_score(y_test, ens_preds)
    print(f"\n==> 7-Model Ensemble Test Accuracy on Shared Short Inputs: {ens_acc:.4f}")

    print("\nPer-size accuracy breakdown (ensemble, on test set):")
    print(f"  {'SIZE':>5}  {'AES-acc':>8}  {'3DES-acc':>9}  {'n_test':>7}")
    for sz in SHARED_SIZES:
        mask = szs_test == sz
        if mask.sum() == 0:
            continue
        ep  = ens_preds[mask]
        yt  = y_test[mask]
        am  = yt == 0
        dm  = yt == 1
        aa  = accuracy_score(yt[am], ep[am]) if am.sum() > 0 else float("nan")
        da  = accuracy_score(yt[dm], ep[dm]) if dm.sum() > 0 else float("nan")
        print(f"  {sz:>5}B  {aa:>8.1%}  {da:>9.1%}  {mask.sum():>7}")

    print(f"\nAll models saved to: {out_dir}")


if __name__ == "__main__":
    train_short_models()
