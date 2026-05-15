#!/usr/bin/env python3
"""
DynaSys-EEG Quick Test Script

Runs a minimal end-to-end sanity check using synthetic data.
Verifies all pipeline components work correctly.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import warnings
warnings.filterwarnings("ignore")

def test_state_space():
    """Test Takens embedding."""
    print("Testing state space reconstruction...")
    from dynasys_eeg.features.state_space import (
        delay_embed, StateSpaceReconstructor,
        select_delay_mutual_info, select_embedding_dimension_fnn
    )
    x = np.sin(np.linspace(0, 10 * np.pi, 500)) + 0.1 * np.random.randn(500)
    tau = select_delay_mutual_info(x, max_lag=50)
    assert 1 <= tau <= 50, f"Unexpected tau: {tau}"

    X = delay_embed(x, embedding_dim=3, time_delay=tau)
    assert X.shape[1] == 3
    assert X.shape[0] == len(x) - 2 * tau

    ssr = StateSpaceReconstructor(embedding_dim=3, time_delay=tau)
    data = np.stack([x, x * 0.9], axis=0)  # 2 channels
    states = ssr.transform(data)
    assert states.ndim == 2
    print(f"  ✓ Delay embedding: shape={states.shape}, τ={tau}")


def test_descriptors():
    """Test all 5 dynamical descriptors."""
    print("Testing dynamical descriptors...")
    from dynasys_eeg.features.descriptors import (
        compute_lyapunov_rosenstein,
        compute_sample_entropy,
        compute_diffusion_coefficient,
        compute_energy_landscape,
        compute_transition_density,
    )
    from dynasys_eeg.features.state_space import delay_embed

    # Healthy signal (stable oscillation)
    x_hc = np.sin(np.linspace(0, 20 * np.pi, 1000)) + 0.05 * np.random.randn(1000)
    # AD signal (chaotic)
    x_ad = np.cumsum(np.random.randn(1000)) * 0.1

    states_hc = delay_embed(x_hc, embedding_dim=3, time_delay=5)
    states_ad = delay_embed(x_ad, embedding_dim=3, time_delay=5)

    lam_hc = compute_lyapunov_rosenstein(x_hc, embedding_dim=3, time_delay=5)
    lam_ad = compute_lyapunov_rosenstein(x_ad, embedding_dim=3, time_delay=5)
    print(f"  ✓ Lyapunov: HC={lam_hc:.4f}, AD={lam_ad:.4f}")

    h_hc = compute_sample_entropy(x_hc)
    h_ad = compute_sample_entropy(x_ad)
    print(f"  ✓ Entropy: HC={h_hc:.4f}, AD={h_ad:.4f}")

    d_hc = compute_diffusion_coefficient(states_hc)
    d_ad = compute_diffusion_coefficient(states_ad)
    print(f"  ✓ Diffusion: HC={d_hc:.6f}, AD={d_ad:.6f}")

    e_hc = compute_energy_landscape(states_hc)
    e_ad = compute_energy_landscape(states_ad)
    print(f"  ✓ Energy: HC={e_hc:.4f}, AD={e_ad:.4f}")

    t_hc = compute_transition_density(states_hc)
    t_ad = compute_transition_density(states_ad)
    print(f"  ✓ Transition: HC={t_hc:.4f}, AD={t_ad:.4f}")


def test_classifiers():
    """Test all 3 classification strategies."""
    print("Testing classifiers...")
    from dynasys_eeg.classification.classifiers import (
        NonlinearDynamicsClassifier,
        PrototypeClassifier,
        EnergyBasedClassifier,
        DynaSysEnsemble,
    )

    np.random.seed(42)
    n_per_class = 30
    # Simulate distinct descriptor vectors per class
    Z_hc  = np.random.randn(n_per_class, 5) + np.array([-0.5, 0.5, 0.1, 1.0, 0.5])
    Z_ad  = np.random.randn(n_per_class, 5) + np.array([1.0, 1.5, 0.8, 2.5, 1.5])
    Z_ftd = np.random.randn(n_per_class, 5) + np.array([0.2, 0.8, 0.3, 1.8, 0.8])

    Z = np.vstack([Z_hc, Z_ad, Z_ftd])
    y = np.array([0]*n_per_class + [1]*n_per_class + [2]*n_per_class)

    from sklearn.model_selection import train_test_split
    Z_tr, Z_te, y_tr, y_te = train_test_split(Z, y, test_size=0.3, random_state=42, stratify=y)

    for name, clf in [
        ("Nonlinear", NonlinearDynamicsClassifier(n_classes=3)),
        ("Prototype", PrototypeClassifier()),
        ("Energy",    EnergyBasedClassifier()),
        ("Ensemble",  DynaSysEnsemble(n_classes=3)),
    ]:
        clf.fit(Z_tr, y_tr)
        acc = (clf.predict(Z_te) == y_te).mean()
        print(f"  ✓ {name}: Accuracy={acc*100:.1f}%")


def test_synthetic_data():
    """Test synthetic data generator."""
    print("Testing synthetic data generation...")
    from dynasys_eeg.data.loader import SyntheticEEGGenerator, EEGPreprocessor

    gen = SyntheticEEGGenerator(sfreq=500.0, seed=42)
    for label in ["HC", "AD", "FTD", "aMCI"]:
        subj = gen.generate_subject(label=label, n_channels=4, duration_sec=10.0)
        assert subj.data.shape == (4, 5000), f"Wrong shape for {label}: {subj.data.shape}"
        print(f"  ✓ {label}: shape={subj.data.shape}")

    preprocessor = EEGPreprocessor(sfreq=500.0, lowcut=0.5, highcut=45.0, normalize=True)
    subj_processed = preprocessor(subj)
    assert np.isfinite(subj_processed.data).all(), "NaN/inf after preprocessing"
    print("  ✓ Preprocessing: OK")


def test_loso():
    """Test LOSO evaluator."""
    print("Testing LOSO evaluation...")
    from dynasys_eeg.classification.classifiers import PrototypeClassifier
    from dynasys_eeg.evaluation.evaluator import LOSOEvaluator

    np.random.seed(0)
    label_map = {"HC": 0, "AD": 1, "FTD": 2}

    # 12 subjects with 20 segments each
    Z_per_subj = []
    labels_per_subj = []
    ids_per_subj = []

    for i, (lbl, center) in enumerate([
        ("HC", [-0.5, 0.5, 0.1, 1.0, 0.5]),
        ("AD", [1.0, 1.5, 0.8, 2.5, 1.5]),
        ("FTD", [0.2, 0.8, 0.3, 1.8, 0.8]),
    ] * 4):
        Z = np.random.randn(20, 5) + np.array(center)
        Z_per_subj.append(Z)
        labels_per_subj.append(lbl)
        ids_per_subj.append(f"S{i:02d}")

    evaluator = LOSOEvaluator(PrototypeClassifier, label_map)
    results = evaluator.evaluate(Z_per_subj, labels_per_subj, ids_per_subj, verbose=False)

    acc = results["aggregate"].get("accuracy", 0)
    print(f"  ✓ LOSO Accuracy: {acc*100:.1f}% ({results['n_folds']} folds)")
    assert acc > 0.3, "LOSO accuracy unexpectedly low"


def test_end_to_end():
    """Mini end-to-end test."""
    print("Testing full pipeline (mini scale)...")
    from dynasys_eeg.pipeline import DynaSysPipeline

    pipeline = DynaSysPipeline()
    pipeline.generate_synthetic_data(
        primary_counts={"AD": 3, "FTD": 3, "HC": 3},
        secondary_counts={"AD": 2, "aMCI": 2, "HC": 2},
        duration_sec=20.0,
    )
    results = pipeline.run(save_results=False)

    assert "loso_primary" in results, "Missing LOSO primary results"
    agg = results["loso_primary"].get("aggregate", {})
    print(f"  ✓ End-to-end pipeline: Acc={agg.get('accuracy', 0)*100:.1f}%")


if __name__ == "__main__":
    tests = [
        ("State Space Reconstruction", test_state_space),
        ("Dynamical Descriptors",     test_descriptors),
        ("Classifiers",               test_classifiers),
        ("Synthetic Data",            test_synthetic_data),
        ("LOSO Evaluation",           test_loso),
        ("End-to-End Pipeline",       test_end_to_end),
    ]

    print("\n" + "="*55)
    print("  DynaSys-EEG Test Suite")
    print("="*55 + "\n")

    passed = 0
    failed = 0
    for name, test_fn in tests:
        print(f"\n[TEST] {name}")
        print("-" * 45)
        try:
            test_fn()
            print(f"  → PASSED")
            passed += 1
        except Exception as e:
            print(f"  → FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "="*55)
    print(f"  Results: {passed} passed, {failed} failed")
    print("="*55 + "\n")
    sys.exit(0 if failed == 0 else 1)
