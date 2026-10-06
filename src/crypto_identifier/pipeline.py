"""Orchestration commands for the cryptographic identification workflow."""

import argparse
from datetime import datetime
import logging
from pathlib import Path

from .classification import evaluate_file
from .data_generation import generate_dataset
from .feature_extraction import extract_features
from .plotting import create_plot

logger = logging.getLogger(__name__)


def add_shared_options(parser: argparse.ArgumentParser) -> None:
    """Add configuration shared by all pipeline stages."""
    parser.add_argument("-d", "--dataset-dir", default="crypto_validation_dataset",
                        help="Dataset directory (default: crypto_validation_dataset).")
    parser.add_argument("-r", "--results-dir", default="validation_results",
                        help="Results directory (default: validation_results).")
    parser.add_argument("-s", "--samples", type=int, default=100,
                        help="Samples per file size and algorithm.")
    parser.add_argument("-e", "--epochs", type=int, default=150, help="CNN training epochs.")
    parser.add_argument("-b", "--batch-size", type=int, default=16, help="CNN batch size.")
    parser.add_argument("-w", "--workers", type=int, default=4, help="Feature-extraction workers.")
    parser.add_argument("--splits", type=int, default=10, help="Evaluation train/test splits.")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Fraction reserved for the untouched final test set.")
    parser.add_argument("--validation-size", type=float, default=0.2,
                        help="Fraction of the non-test data used for CNN validation.")
    parser.add_argument("--patience", type=int, default=50,
                        help="CNN early-stopping patience in epochs.")
    parser.add_argument("--min-delta", type=float, default=0.0,
                        help="Minimum validation-loss improvement required to reset patience.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible splits and CNN initialization.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate only four 1 KB and 8 KB samples per algorithm.")
    parser.add_argument("--plot", action="store_true",
                        help="Create a binary-results chart after evaluation.")
    parser.add_argument("--run-name", default=None,
                        help="Optional name for this model-artifact run directory.")


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level pipeline command parser."""
    parser = argparse.ArgumentParser(
        description="NIST-CNN cryptographic algorithm identification pipeline."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for command, help_text in (
        ("all", "Generate data, extract features, and evaluate models."),
        ("features-evaluate", "Extract features from an existing dataset and evaluate."),
        ("evaluate", "Evaluate models from an existing feature NPZ file."),
    ):
        subparser = commands.add_parser(command, help=help_text)
        add_shared_options(subparser)
    return parser


def paths_from_args(args):
    """Derive standard artifact paths from shared pipeline options."""
    dataset_dir = Path(args.dataset_dir)
    results_dir = Path(args.results_dir)
    return {
        "ciphertext_dir": dataset_dir / "ciphertext",
        "feature_prefix": dataset_dir / "extracted_features",
        "feature_npz": dataset_dir / "extracted_features.npz",
        "binary_csv": results_dir / "binary_classification_results.csv",
        "multiclass_csv": results_dir / "multiclass_classification_results.csv",
        "plot_png": results_dir / "binary_classification_comparison.png",
        "models_root": results_dir / "models",
        "runs_root": results_dir / "runs",
    }


def run_evaluations(args, artifact_paths) -> None:
    """Run binary and multiclass classifier comparisons from extracted features."""
    common = {
        "input_file": str(artifact_paths["feature_npz"]),
        "splits": args.splits,
        "test_size": args.test_size,
        "validation_size": args.validation_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "seed": args.seed,
    }
    run_name = args.run_name or datetime.now().strftime("%Y%m%dT%H%M%S")
    model_run_dir = artifact_paths["models_root"] / run_name
    report_run_dir = artifact_paths["runs_root"] / run_name
    logger.info("Storing model artifacts under: %s", model_run_dir)
    logger.info("Storing immutable benchmark reports under: %s", report_run_dir)
    logger.info("Running binary classification: AES vs. 3DES.")
    evaluate_file(
        output_csv=str(report_run_dir / "binary_classification_results.csv"),
        multiclass=False,
        models_dir=model_run_dir / "binary",
        **common,
    )
    logger.info("Running five-class classification.")
    evaluate_file(
        output_csv=str(report_run_dir / "multiclass_classification_results.csv"),
        multiclass=True,
        models_dir=model_run_dir / "multiclass",
        **common,
    )
    if args.plot:
        create_plot(
            str(report_run_dir / "binary_classification_results.csv"),
            str(report_run_dir / "binary_classification_comparison.png"),
        )


def run(args) -> None:
    """Execute the requested workflow."""
    artifact_paths = paths_from_args(args)
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    if args.command == "all":
        generate_dataset(args.dataset_dir, args.samples, args.dry_run)
        extract_features(str(artifact_paths["ciphertext_dir"]),
                         str(artifact_paths["feature_prefix"]), args.workers)
    elif args.command == "features-evaluate":
        extract_features(str(artifact_paths["ciphertext_dir"]),
                         str(artifact_paths["feature_prefix"]), args.workers)

    run_evaluations(args, artifact_paths)


def main(argv=None) -> None:
    """CLI entry point for all workflow variants."""
    args = build_parser().parse_args(argv)
    try:
        run(args)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        raise SystemExit(f"error: {error}") from error


if __name__ == "__main__":
    main()
