#!/usr/bin/env python3
"""
Model Trainer and Classifier for ML-Based Cryptographic Algorithm Identification.

"""

import os
import argparse
import csv
from datetime import datetime
import logging
from pathlib import Path
import random

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# 1D Convolutional Neural Network (CNN) as described in the paper
class CipherCNN(nn.Module):
    def __init__(self, input_dim=49, num_classes=2):
        super(CipherCNN, self).__init__()
        # Input: (batch_size, 1, 49)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)  # 49 -> 24
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)  # 24 -> 12
        
        # FC layers mapping flattened features to classes
        self.fc1 = nn.Linear(32 * 12, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # x is (batch_size, 49)
        x = x.unsqueeze(1)  # Reshape to (batch_size, 1, 49)
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def set_random_seed(seed: int) -> None:
    """Configure reproducible pseudo-random generators for an experiment run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _cpu_state_dict(model: nn.Module) -> dict:
    """Return a portable copy of model weights, independent of the training device."""
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def train_cnn(X_train, y_train, X_validation, y_validation, num_classes, epochs=150,
              lr=0.001, batch_size=16, patience=50, min_delta=0.0, verbose=False,
              epoch_callback=None, checkpoint_callback=None):
    """
    Train the 1D CNN and restore the weights with the lowest validation loss.
    """
    # Convert numpy arrays to Torch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_validation_t = torch.tensor(X_validation, dtype=torch.float32)
    y_validation_t = torch.tensor(y_validation, dtype=torch.long)
    
    # Create DataLoader
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=max(1, min(batch_size, len(X_train))), shuffle=True)
    
    # Initialize model, loss, and optimizer
    model = CipherCNN(input_dim=X_train.shape[1], num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    best_validation_loss = float("inf")
    best_validation_accuracy = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    best_state = None
    last_state = None

    # Training loop with a validation measurement at the end of every epoch.
    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        total_train_correct = 0
        total_train_samples = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * len(targets)
            total_train_correct += (outputs.argmax(dim=1) == targets).sum().item()
            total_train_samples += len(targets)
            
        model.eval()
        with torch.no_grad():
            validation_inputs = X_validation_t.to(device)
            validation_targets = y_validation_t.to(device)
            validation_outputs = model(validation_inputs)
            validation_loss = criterion(validation_outputs, validation_targets).item()
            validation_predictions = validation_outputs.argmax(dim=1)
            validation_accuracy = (validation_predictions == validation_targets).float().mean().item()

        metrics = {
            "epoch": epoch,
            "train_loss": total_train_loss / total_train_samples,
            "train_accuracy": total_train_correct / total_train_samples,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
        }
        if verbose:
            logger.info(
                "Epoch %d/%d - train loss: %.4f, train accuracy: %.4f, validation loss: %.4f, validation accuracy: %.4f",
                epoch, epochs, metrics["train_loss"], metrics["train_accuracy"],
                metrics["validation_loss"], metrics["validation_accuracy"],
            )
        if epoch_callback is not None:
            epoch_callback(metrics)
        last_state = _cpu_state_dict(model)

        if validation_loss < best_validation_loss - min_delta:
            best_validation_loss = validation_loss
            best_validation_accuracy = validation_accuracy
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = _cpu_state_dict(model)
            if checkpoint_callback is not None:
                checkpoint_callback(best_state, metrics)
        else:
            epochs_without_improvement += 1
            if patience is not None and epochs_without_improvement >= patience:
                logger.info(
                    "Early stopping at epoch %d; validation loss has not improved for %d epochs.",
                    epoch,
                    patience,
                )
                break

    if best_state is None:
        raise RuntimeError("CNN training did not produce a validation checkpoint.")
    model.load_state_dict(best_state)
    selection = {
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "best_validation_accuracy": best_validation_accuracy,
        "epochs_ran": epoch,
    }
    return model, selection, last_state


def evaluate_cnn(model: nn.Module, X, y) -> float:
    """Evaluate a selected CNN once on a held-out test set."""
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(X, dtype=torch.float32, device=device)
        targets = torch.tensor(y, dtype=torch.long, device=device)
        predictions = model(inputs).argmax(dim=1)
    return (predictions == targets).float().mean().item()

def load_dataset(input_file):
    """
    Loads features and labels from either a CSV or NPZ file.
    """
    if input_file.endswith('.npz'):
        logger.info(f"Loading data from NPZ file: {input_file}")
        data = np.load(input_file, allow_pickle=True)
        X = data['X']
        y_algo = data['y_algo']
        y_size = data['y_size']
        sample_ids = data['sample_ids']
        df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(X.shape[1])])
        df['algorithm'] = y_algo
        df['file_size'] = y_size
        df['sample_id'] = sample_ids
    elif input_file.endswith('.csv'):
        logger.info(f"Loading data from CSV file: {input_file}")
        df = pd.read_csv(input_file)
        if "file_size" not in df.columns and "size" in df.columns:
            df = df.rename(columns={"size": "file_size"})
    else:
        raise ValueError("Unsupported file format. Must be either .csv or .npz")
        
    return df

METRIC_COLUMNS = ("epoch", "train_loss", "train_accuracy", "validation_loss", "validation_accuracy")


def prepare_epoch_metrics_log(models_dir, size, split_number):
    """Create one append-only epoch metrics log for a train/test split."""
    split_dir = Path(models_dir) / f"size_{size}" / f"split_{split_number:02d}"
    split_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = split_dir / "epoch_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as file:
        csv.DictWriter(file, fieldnames=METRIC_COLUMNS).writeheader()

    def append_metrics(metrics):
        with metrics_path.open("a", newline="", encoding="utf-8") as file:
            csv.DictWriter(file, fieldnames=METRIC_COLUMNS).writerow(metrics)

    return append_metrics


def _cnn_checkpoint_payload(state_dict, metadata):
    """Build a self-contained portable CNN checkpoint payload."""
    return {
        "model_type": "CipherCNN",
        "input_dim": len(metadata["feature_columns"]),
        "num_classes": len(metadata["class_mapping"]),
        "state_dict": state_dict,
        "metadata": metadata,
    }


def prepare_best_checkpoint(models_dir, size, split_number, checkpoint_metadata):
    """Save the currently best CNN weights immediately after every improvement."""
    split_dir = Path(models_dir) / f"size_{size}" / f"split_{split_number:02d}"
    split_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = split_dir / "best_cnn.pt"

    def save_checkpoint(state_dict, metrics):
        metadata = {**checkpoint_metadata, **{
            "best_epoch": metrics["epoch"],
            "best_validation_loss": metrics["validation_loss"],
            "best_validation_accuracy": metrics["validation_accuracy"],
        }}
        torch.save(_cnn_checkpoint_payload(state_dict, metadata), checkpoint_path)

    return save_checkpoint


def save_split_models(models_dir, size, split_number, classifiers, scaler, cnn_model,
                      last_cnn_state, feature_columns, algo_map, train_sample_ids,
                      validation_sample_ids, test_sample_ids, seed, selection,
                      test_accuracy):
    """Persist every fitted model and the preprocessing needed to reuse it."""
    split_dir = Path(models_dir) / f"size_{size}" / f"split_{split_number:02d}"
    split_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "file_size": str(size),
        "split": split_number,
        "feature_columns": feature_columns,
        "class_mapping": algo_map,
        "seed": seed,
        "train_sample_ids": train_sample_ids.tolist(),
        "validation_sample_ids": validation_sample_ids.tolist(),
        "test_sample_ids": test_sample_ids.tolist(),
        **selection,
        "test_accuracy": test_accuracy,
    }

    joblib.dump({"scaler": scaler, "metadata": metadata}, split_dir / "scaler.joblib")
    for name, classifier in classifiers.items():
        joblib.dump(
            {"model": classifier, "metadata": metadata},
            split_dir / f"{name.lower()}.joblib",
        )

    best_payload = _cnn_checkpoint_payload(_cpu_state_dict(cnn_model), metadata)
    torch.save(best_payload, split_dir / "best_cnn.pt")
    # ``cnn.pt`` is retained as the backwards-compatible name for selected weights.
    torch.save(best_payload, split_dir / "cnn.pt")
    torch.save(
        _cnn_checkpoint_payload(last_cnn_state, {**metadata, "checkpoint_type": "last_epoch"}),
        split_dir / "last_cnn.pt",
    )
    with (split_dir / "split_summary.csv").open("w", newline="", encoding="utf-8") as file:
        fields = ("file_size", "split", "seed", "best_epoch", "epochs_ran",
                  "best_validation_loss", "best_validation_accuracy", "test_accuracy")
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerow({field: metadata[field] for field in fields})
    logger.info("Saved fitted model artifacts to: %s", split_dir)


def run_evaluation(df, binary_only=True, num_splits=10, test_size=0.2,
                   validation_size=0.2, epochs=150, batch_size=16, patience=50,
                   min_delta=0.0, seed=42, models_dir=None):
    """
    Performs ten repeated random subsampling validation across all classifiers.
    """
    set_random_seed(seed)
    results_list = []
    split_results = []
    
    # 1. Filter dataset
    if binary_only:
        logger.info("Setting up binary classification: AES vs 3DES")
        df_filtered = df[df['algorithm'].isin(['AES', '3DES'])].copy()
        algo_map = {'AES': 0, '3DES': 1}
        df_filtered['label'] = df_filtered['algorithm'].map(algo_map)
        num_classes = 2
    else:
        logger.info("Setting up 5-class classification: AES, 3DES, CAST, RC2, Blowfish")
        df_filtered = df[df['algorithm'].isin(['AES', '3DES', 'CAST', 'RC2', 'Blowfish'])].copy()
        algo_map = {'AES': 0, '3DES': 1, 'CAST': 2, 'RC2': 3, 'Blowfish': 4}
        df_filtered['label'] = df_filtered['algorithm'].map(algo_map)
        num_classes = 5

    if df_filtered.empty:
        logger.error("No matching samples found for training.")
        return None

    # Get distinct file sizes present in dataset
    file_sizes = df_filtered['file_size'].unique()
    logger.info(f"Available file sizes: {list(file_sizes)}")
    
    # Feature columns (f_0 to f_48)
    feature_cols = [c for c in df_filtered.columns if c.startswith('f_')]
    logger.info(f"Found {len(feature_cols)} feature dimensions.")

    # Iterate over each file size
    for size in sorted(file_sizes):
        logger.info(f"\n--- Training and evaluating models for file size: {size} ---")
        df_size = df_filtered[df_filtered['file_size'] == size]
        
        X = df_size[feature_cols].values
        y = df_size['label'].values
        
        if len(np.unique(y)) < num_classes:
            logger.warning(f"Skipping size {size}: requires at least {num_classes} classes, but only found {len(np.unique(y))} in current samples.")
            continue
            
        # Keep every cipher represented in both sets. This also lets the small
        # dry-run dataset (two samples per cipher) be evaluated safely.
        minimum_test_count = num_classes
        requested_test_count = int(np.ceil(len(y) * test_size))
        test_count = max(minimum_test_count, requested_test_count)
        if len(y) - test_count < num_classes:
            logger.warning(
                "Skipping size %s: not enough samples to retain every class in train and test.",
                size,
            )
            continue
        outer_splitter = StratifiedShuffleSplit(
            n_splits=num_splits,
            test_size=test_count,
            random_state=seed,
        )
        
        # Initialize classifier models
        model_names = ['SVM', 'GNB', 'KNN', 'RF', 'LR', 'MLP', 'CNN']
        scores = {model_name: [] for model_name in model_names}
        
        for split_number, (train_validation_index, test_index) in enumerate(
            outer_splitter.split(X, y), start=1
        ):
            X_train_validation = X[train_validation_index]
            y_train_validation = y[train_validation_index]
            validation_count = max(
                num_classes,
                int(np.ceil(len(y_train_validation) * validation_size)),
            )
            if len(y_train_validation) - validation_count < num_classes:
                logger.warning(
                    "Skipping size %s split %d: not enough samples for train, validation, and test sets.",
                    size,
                    split_number,
                )
                continue
            inner_splitter = StratifiedShuffleSplit(
                n_splits=1,
                test_size=validation_count,
                random_state=seed + split_number,
            )
            inner_train_index, validation_index = next(
                inner_splitter.split(X_train_validation, y_train_validation)
            )
            train_index = train_validation_index[inner_train_index]
            validation_index = train_validation_index[validation_index]

            X_train, X_validation, X_test = X[train_index], X[validation_index], X[test_index]
            y_train, y_validation, y_test = y[train_index], y[validation_index], y[test_index]
            
            # Standardize P-value features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_validation_scaled = scaler.transform(X_validation)
            X_test_scaled = scaler.transform(X_test)
            
            # Dynamically adjust KNN neighbors based on actual available training samples
            n_samples_train = len(X_train)
            k_neighbors = min(5, max(1, n_samples_train))
            
            classifiers = {
                'SVM': SVC(kernel='rbf', C=1.0),
                'GNB': GaussianNB(),
                'KNN': KNeighborsClassifier(n_neighbors=k_neighbors),
                'RF': RandomForestClassifier(n_estimators=100, random_state=42),
                'LR': LogisticRegression(max_iter=1000),
                'MLP': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
            }
            
            # Train and evaluate traditional classifiers
            for name, clf in classifiers.items():
                clf.fit(X_train_scaled, y_train)
                y_pred = clf.predict(X_test_scaled)
                acc = accuracy_score(y_test, y_pred)
                scores[name].append(acc)
                
            checkpoint_metadata = {
                "file_size": str(size),
                "split": split_number,
                "feature_columns": feature_cols,
                "class_mapping": algo_map,
                "seed": seed,
                "train_sample_ids": df_size.iloc[train_index]["sample_id"].to_numpy().tolist(),
                "validation_sample_ids": df_size.iloc[validation_index]["sample_id"].to_numpy().tolist(),
                "test_sample_ids": df_size.iloc[test_index]["sample_id"].to_numpy().tolist(),
            }
            epoch_callback = None
            checkpoint_callback = None
            if models_dir is not None:
                epoch_callback = prepare_epoch_metrics_log(models_dir, size, split_number)
                checkpoint_callback = prepare_best_checkpoint(
                    models_dir,
                    size,
                    split_number,
                    checkpoint_metadata,
                )

            # Select the CNN using validation data, then evaluate once on the untouched test set.
            cnn_model, selection, last_cnn_state = train_cnn(
                X_train_scaled, y_train, X_validation_scaled, y_validation,
                num_classes=num_classes,
                epochs=epochs,
                batch_size=batch_size,
                patience=patience,
                min_delta=min_delta,
                verbose=False,
                epoch_callback=epoch_callback,
                checkpoint_callback=checkpoint_callback,
            )
            cnn_acc = evaluate_cnn(cnn_model, X_test_scaled, y_test)
            scores['CNN'].append(cnn_acc)
            split_record = {
                "file_size": size,
                "split": split_number,
                "seed": seed,
                "checkpoint_path": str(
                    Path(f"size_{size}") / f"split_{split_number:02d}" / "best_cnn.pt"
                ),
                **{name: scores[name][-1] for name in model_names},
                **selection,
            }
            split_results.append(split_record)
            if models_dir is not None:
                save_split_models(
                    models_dir=models_dir,
                    size=size,
                    split_number=split_number,
                    classifiers=classifiers,
                    scaler=scaler,
                    cnn_model=cnn_model,
                    last_cnn_state=last_cnn_state,
                    feature_columns=feature_cols,
                    algo_map=algo_map,
                    train_sample_ids=df_size.iloc[train_index]["sample_id"].to_numpy(),
                    validation_sample_ids=df_size.iloc[validation_index]["sample_id"].to_numpy(),
                    test_sample_ids=df_size.iloc[test_index]["sample_id"].to_numpy(),
                    seed=seed,
                    selection=selection,
                    test_accuracy=cnn_acc,
                )
            
        # Calculate mean accuracies across splits
        size_results = {'file_size': size}
        logger.info(f"Mean accuracies across {num_splits} runs:")
        for name, accs in scores.items():
            mean_acc = np.mean(accs)
            std_acc = np.std(accs)
            size_results[name] = mean_acc
            logger.info(f"  {name:4s}: {mean_acc:.3f} (±{std_acc:.3f})")
            
        results_list.append(size_results)
        
    return pd.DataFrame(results_list), pd.DataFrame(split_results)

def evaluate_file(input_file, output_csv, multiclass=False, splits=10, test_size=0.2,
                  validation_size=0.2, epochs=100, batch_size=8, patience=50,
                  min_delta=0.0, seed=42, models_dir=None):
    """Train all classifiers and save accuracy results plus fitted model artifacts."""
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input feature file not found: {input_file}")

    # Load data
    df = load_dataset(input_file)
    
    if models_dir is None:
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        models_dir = Path(output_csv).parent / "models" / f"{Path(output_csv).stem}_{timestamp}"
    models_dir = Path(models_dir)
    logger.info("Saving trained model artifacts under: %s", models_dir)

    # Run comparative validation
    results_df, split_results_df = run_evaluation(
        df, 
        binary_only=not multiclass,
        num_splits=splits,
        test_size=test_size,
        validation_size=validation_size,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        min_delta=min_delta,
        seed=seed,
        models_dir=models_dir,
    )
    
    if results_df is not None and not results_df.empty:
        # Save results
        output_parent = os.path.dirname(os.path.abspath(output_csv))
        os.makedirs(output_parent, exist_ok=True)
        results_df.to_csv(output_csv, index=False)
        logger.info(f"\nComparative results successfully written to: {output_csv}")
        if models_dir is not None and not split_results_df.empty:
            summary_path = Path(models_dir) / "split_results.csv"
            split_results_df.to_csv(summary_path, index=False)
            logger.info("Per-split test results successfully written to: %s", summary_path)
            ranking_path = Path(models_dir) / "checkpoint_ranking.csv"
            split_results_df.sort_values(
                ["best_validation_loss", "best_validation_accuracy"],
                ascending=[True, False],
            ).to_csv(ranking_path, index=False)
            logger.info("Checkpoint ranking successfully written to: %s", ranking_path)
        
        # Display summary table
        print("\n" + "="*80)
        print("                 COMPARATIVE CLASSIFICATION ACCURACY SUMMARY")
        print("="*80)
        print(results_df.to_string(index=False, formatters={
            'SVM': '{:,.3f}'.format, 'GNB': '{:,.3f}'.format, 'KNN': '{:,.3f}'.format,
            'RF': '{:,.3f}'.format, 'LR': '{:,.3f}'.format, 'MLP': '{:,.3f}'.format,
            'CNN': '{:,.3f}'.format
        }))
        print("="*80 + "\n")
    return results_df


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for model evaluation."""
    parser = argparse.ArgumentParser(description="Train CNN and baselines to identify cryptographic ciphers.")
    parser.add_argument('--input-file', '--input_file', dest='input_file',
                        default='crypto_validation_dataset/extracted_features.npz',
                        help='Feature CSV or NPZ file.')
    parser.add_argument('--output-csv', '--output_csv', dest='output_csv',
                        default='validation_results/classification_results.csv',
                        help='Comparison CSV output path.')
    parser.add_argument('--multiclass', action='store_true',
                        help='Perform 5-class classification instead of AES vs. 3DES.')
    parser.add_argument('--splits', type=int, default=10,
                        help='Number of random subsampling validation splits.')
    parser.add_argument('--test-size', '--test_size', dest='test_size', type=float, default=0.2,
                        help='Fraction of samples reserved for the untouched final test set.')
    parser.add_argument('--validation-size', type=float, default=0.2,
                        help='Fraction of the non-test data used for CNN validation and checkpoint selection.')
    parser.add_argument('--epochs', type=int, default=100, help='CNN training epochs.')
    parser.add_argument('--batch-size', '--batch_size', dest='batch_size', type=int, default=8,
                        help='CNN batch size.')
    parser.add_argument('--patience', type=int, default=50,
                        help='Stop CNN training after this many non-improving validation epochs.')
    parser.add_argument('--min-delta', type=float, default=0.0,
                        help='Minimum validation-loss improvement required to reset early stopping.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible splits and CNN initialization.')
    parser.add_argument('--models-dir', type=str, default=None,
                        help='Directory for fitted-model artifacts. Defaults to a timestamped run directory.')
    return parser


def main(argv=None) -> None:
    """CLI entry point for model evaluation."""
    args = build_parser().parse_args(argv)
    evaluate_file(**vars(args))

if __name__ == '__main__':
    main()
