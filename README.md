# Cryptographic Algorithm Identification

This project generates ciphertext for five block ciphers, extracts 49 NIST SP 800-22 statistical features, and compares traditional classifiers with a 1D CNN.

## Project layout

```text
src/crypto_identifier/
  data_generation.py       Dataset construction API and command
  feature_extraction.py    NIST feature extraction API and command
  classification.py        Model training/evaluation API and command
  plotting.py              Result plotting API and command
  pipeline.py              Full workflow orchestration
scripts/
  run_all.sh               All stages
  run_features_and_evaluate.sh
  run_evaluate.sh
```

## Linux setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
chmod +x scripts/*.sh *_pipeline*.sh
```

## Run the workflows

```bash
# Fast smoke test: data generation, feature extraction, and both evaluations
scripts/run_all.sh --dry-run --epochs 1 --workers 1

# Full benchmark
scripts/run_all.sh --samples 100 --epochs 150 --workers 4 --plot

# Start from an existing ciphertext dataset
scripts/run_features_and_evaluate.sh --dataset-dir crypto_validation_dataset

# Start from crypto_validation_dataset/extracted_features.npz
scripts/run_evaluate.sh --dataset-dir crypto_validation_dataset
```

The original pipeline names are also available as Linux scripts:
`1_run_pipeline_ds_fe_ml_eval.sh`, `2_run_pipeline__fe_ml_eval.sh`, and
`3_run_pipeline__ml_eval.sh`.

All runners accept `--dataset-dir`, `--results-dir`, `--samples`, `--epochs`, `--batch-size`, `--workers`, `--splits`, `--test-size`, `--validation-size`, `--patience`, `--min-delta`, `--seed`, and `--plot`. Set `PYTHON=/path/to/python` before a runner when `python3` is not the intended interpreter.

You may also use installed commands such as `crypto-pipeline all --dry-run`, or invoke the package directly with `python3 -m crypto_identifier.pipeline --help`.

Outputs are written to `crypto_validation_dataset/` and `validation_results/` by default.

Every evaluation split saves fitted artifacts under
`validation_results/models/<run-name>/`. Each split directory contains the
best CNN weights (`best_cnn.pt`, with `cnn.pt` retained for compatibility),
last-epoch CNN weights (`last_cnn.pt`), a fitted scaler (`scaler.joblib`), and fitted baseline
models (`svm.joblib`, `gnb.joblib`, `knn.joblib`, `rf.joblib`, `lr.joblib`,
and `mlp.joblib`). `epoch_metrics.csv` is written immediately after each CNN
epoch with training loss/accuracy and validation loss/accuracy. Training stops
early when validation loss does not improve for `--patience` epochs. `split_summary.csv`
records the selected epoch and final held-out test accuracy; `split_results.csv`
summarizes every split; `checkpoint_ranking.csv` ranks saved `best_cnn.pt`
checkpoints by validation loss (best first). Immutable aggregate reports are saved under
`validation_results/runs/<run-name>/`, so later runs do not overwrite earlier
results. Use `--run-name experiment-01` to provide a recognizable artifact directory name.

<!-- Run -->
<!-- python -m crypto_identifier.api
cd c:\Users\ritik\OneDrive\Documents\NeuralAI\backend\Neural-Distinguoisher
python -m uvicorn src.crypto_identifier.api:app --reload --port 8000 -->

