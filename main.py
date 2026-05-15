#!/usr/bin/env python3
"""
DynaSys-EEG: Main Entry Point

Usage:
    # Full pipeline with synthetic data (for testing):
    python main.py --mode synthetic

    # With real data:
    python main.py --mode real --primary_dir data/primary --secondary_dir data/secondary

    # LOSO only:
    python main.py --mode synthetic --eval loso_only

    # With dynamics model training (Phase 5):
    python main.py --mode synthetic --train_dynamics
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger("dynasys_eeg.main")


def parse_args():
    parser = argparse.ArgumentParser(
        description="DynaSys-EEG: Dynamical System-based EEG Dementia Classification"
    )
    parser.add_argument(
        "--mode", type=str, default="synthetic",
        choices=["synthetic", "real"],
        help="Data mode: 'synthetic' for testing, 'real' for actual EEG data"
    )
    parser.add_argument(
        "--primary_dir", type=str, default="data/primary",
        help="Path to primary resting-state EEG dataset"
    )
    parser.add_argument(
        "--secondary_dir", type=str, default="data/secondary",
        help="Path to secondary olfactory EEG dataset"
    )
    parser.add_argument(
        "--results_dir", type=str, default="results",
        help="Directory to save results and plots"
    )
    parser.add_argument(
        "--n_subjects", type=int, default=None,
        help="Override number of synthetic subjects per class"
    )
    parser.add_argument(
        "--duration", type=float, default=60.0,
        help="Duration (seconds) per synthetic subject"
    )
    parser.add_argument(
        "--train_dynamics", action="store_true",
        help="Train neural dynamics model (Phase 5)"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Compute device: 'cpu' or 'cuda'"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--no_save", action="store_true",
        help="Don't save visualizations to disk"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "="*60)
    print("  DynaSys-EEG: Dynamical System-based EEG Analysis")
    print("  Dementia Classification via Nonlinear Dynamical Modeling")
    print("="*60 + "\n")

    from dynasys_eeg.pipeline import DynaSysPipeline
    from dynasys_eeg.configs import cfg

    # Override config settings
    cfg.seed = args.seed
    if args.device:
        cfg.device = args.device

    pipeline = DynaSysPipeline(config=cfg, device=args.device)

    # Load data
    if args.mode == "synthetic":
        n_per_class = args.n_subjects or 8
        pipeline.generate_synthetic_data(
            primary_counts={"AD": n_per_class, "FTD": n_per_class - 2, "HC": n_per_class},
            secondary_counts={"AD": n_per_class // 2 + 1, "aMCI": n_per_class // 2, "HC": n_per_class // 2 + 2},
            duration_sec=args.duration,
        )
    else:
        pipeline.load_data(
            primary_dir=args.primary_dir,
            secondary_dir=args.secondary_dir if os.path.exists(args.secondary_dir) else None,
        )

    # Run pipeline
    results = pipeline.run(
        save_results=not args.no_save,
        results_dir=args.results_dir,
        run_dynamics_training=args.train_dynamics,
    )

    # Print summary
    print("\n" + "="*60)
    print("  SUMMARY OF RESULTS")
    print("="*60)

    if "loso_primary" in results:
        agg = results["loso_primary"].get("aggregate", {})
        print(f"\nPrimary Dataset LOSO (AD vs FTD vs HC):")
        print(f"  Accuracy  : {agg.get('accuracy', 0)*100:.2f}%")
        print(f"  F1-Score  : {agg.get('f1', 0)*100:.2f}%")
        print(f"  Precision : {agg.get('precision', 0)*100:.2f}%")
        print(f"  Recall    : {agg.get('recall', 0)*100:.2f}%")

    if "loso_secondary" in results:
        agg2 = results["loso_secondary"].get("aggregate", {})
        print(f"\nSecondary Dataset LOSO (AD vs aMCI vs HC):")
        print(f"  Accuracy  : {agg2.get('accuracy', 0)*100:.2f}%")
        print(f"  F1-Score  : {agg2.get('f1', 0)*100:.2f}%")

    if "cross_dataset" in results:
        for direction, m in results["cross_dataset"].items():
            print(f"\nCross-Dataset [{direction}]:")
            print(f"  Accuracy : {m.get('accuracy', 0)*100:.2f}%")
            print(f"  F1-Score : {m.get('f1', 0)*100:.2f}%")

    print("\n" + "="*60)
    print(f"Results saved to: {args.results_dir}/")
    print("="*60 + "\n")

    return results


if __name__ == "__main__":
    main()
