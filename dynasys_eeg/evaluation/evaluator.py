"""
Evaluation Module for DynaSys-EEG (Phases 9-12).

Implements:
  1. Leave-One-Subject-Out (LOSO) cross-validation
  2. Cross-dataset generalization evaluation
  3. Ablation study (removing each descriptor component)
  4. Metrics computation (Accuracy, Precision, Recall, F1, ROC-AUC)
  5. Results visualization

From the paper (Section VI):
  - LOSO: Each subject tested independently
  - Cross-dataset: Train resting → Test olfactory (and vice versa)
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize

logger = logging.getLogger(__name__)


# =============================================================================
# Metrics Computation
# =============================================================================

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    classes: Optional[List] = None,
    average: str = "weighted",
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: true labels
        y_pred: predicted labels
        y_proba: predicted probabilities (optional, for ROC-AUC)
        classes: list of class labels
        average: 'weighted' or 'macro' for multi-class metrics

    Returns:
        metrics: dict with accuracy, precision, recall, f1, roc_auc
    """
    classes = classes if classes is not None else sorted(np.unique(y_true))

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
    }

    # ROC-AUC (requires probabilities)
    if y_proba is not None and len(classes) > 1:
        try:
            if len(classes) == 2:
                metrics["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
            else:
                y_bin = label_binarize(y_true, classes=classes)
                metrics["roc_auc"] = roc_auc_score(
                    y_bin, y_proba, average=average, multi_class="ovr"
                )
        except Exception:
            metrics["roc_auc"] = float("nan")

    return metrics


def print_results_table(results: Dict, title: str = "Results") -> None:
    """Print a formatted results table."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    header = f"{'Method':<25} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8}"
    if "roc_auc" in next(iter(results.values()), {}):
        header += f" {'AUC':>8}"
    print(header)
    print("-" * 60)

    for method, m in results.items():
        row = (
            f"{method:<25} "
            f"{m.get('accuracy', 0)*100:>7.2f}% "
            f"{m.get('precision', 0)*100:>7.2f}% "
            f"{m.get('recall', 0)*100:>7.2f}% "
            f"{m.get('f1', 0)*100:>7.2f}%"
        )
        if "roc_auc" in m:
            row += f" {m.get('roc_auc', 0)*100:>7.2f}%"
        print(row)
    print(f"{'='*60}\n")


# =============================================================================
# Leave-One-Subject-Out (LOSO) Evaluation
# =============================================================================

class LOSOEvaluator:
    """
    Leave-One-Subject-Out cross-validation.

    For each subject i:
        - Train: all subjects except i
        - Test: subject i (subject-level majority voting)

    This is the most rigorous evaluation for subject-independent models.
    """

    def __init__(
        self,
        classifier_factory,   # callable that returns a fresh classifier
        label_map: Dict[str, int],
        metrics_avg: str = "weighted",
    ):
        self.classifier_factory = classifier_factory
        self.label_map = label_map
        self.inv_label_map = {v: k for k, v in label_map.items()}
        self.metrics_avg = metrics_avg

    def _majority_vote(self, predictions: np.ndarray) -> int:
        """Subject-level majority voting over segment predictions."""
        counts = np.bincount(predictions.astype(int))
        return int(np.argmax(counts))

    def evaluate(
        self,
        descriptors_per_subject: List[np.ndarray],  # list of (n_seg, 5)
        labels_per_subject: List[str],               # list of label strings
        subject_ids: List[str],
        verbose: bool = True,
    ) -> Dict:
        """
        Run full LOSO evaluation.

        Returns:
            results: dict with per-subject and aggregate metrics
        """
        n_subjects = len(descriptors_per_subject)
        if n_subjects < 2:
            raise ValueError("Need at least 2 subjects for LOSO")

        # Convert labels to integers
        y_per_subject = [
            np.full(len(desc), self.label_map[lbl], dtype=int)
            for desc, lbl in zip(descriptors_per_subject, labels_per_subject)
        ]

        all_y_true = []   # subject-level
        all_y_pred = []
        fold_results = []

        for test_idx in range(n_subjects):
            # Build train set
            train_Z = np.concatenate([
                descriptors_per_subject[i]
                for i in range(n_subjects) if i != test_idx
            ], axis=0)
            train_y = np.concatenate([
                y_per_subject[i]
                for i in range(n_subjects) if i != test_idx
            ], axis=0)

            test_Z = descriptors_per_subject[test_idx]
            true_label = int(y_per_subject[test_idx][0])

            # Check for valid data
            if len(train_Z) == 0 or len(test_Z) == 0:
                continue

            # Remove NaN/inf
            valid_train = np.all(np.isfinite(train_Z), axis=1)
            valid_test = np.all(np.isfinite(test_Z), axis=1)
            if valid_train.sum() < 5 or valid_test.sum() == 0:
                continue

            train_Z = train_Z[valid_train]
            train_y = train_y[valid_train]
            test_Z = test_Z[valid_test]

            # Skip if only one class in training
            if len(np.unique(train_y)) < 2:
                continue

            # Train
            clf = self.classifier_factory()
            try:
                clf.fit(train_Z, train_y)
                seg_preds = clf.predict(test_Z)
            except Exception as e:
                logger.debug(f"Subject {test_idx} fold failed: {e}")
                continue

            # Subject-level majority vote
            pred_label = self._majority_vote(seg_preds)

            fold_results.append({
                "subject_id": subject_ids[test_idx],
                "true": true_label,
                "pred": pred_label,
                "true_name": self.inv_label_map.get(true_label, str(true_label)),
                "pred_name": self.inv_label_map.get(pred_label, str(pred_label)),
                "n_segments": len(test_Z),
            })

            all_y_true.append(true_label)
            all_y_pred.append(pred_label)

            if verbose:
                correct = "✓" if true_label == pred_label else "✗"
                logger.info(
                    f"[{correct}] Subject {subject_ids[test_idx]}: "
                    f"True={self.inv_label_map.get(true_label)}, "
                    f"Pred={self.inv_label_map.get(pred_label)}"
                )

        if not all_y_true:
            return {"error": "No valid folds completed"}

        y_true_arr = np.array(all_y_true)
        y_pred_arr = np.array(all_y_pred)

        aggregate_metrics = compute_metrics(y_true_arr, y_pred_arr, average=self.metrics_avg)
        cm = confusion_matrix(y_true_arr, y_pred_arr)

        return {
            "aggregate": aggregate_metrics,
            "confusion_matrix": cm,
            "per_subject": fold_results,
            "n_folds": len(fold_results),
        }


# =============================================================================
# Cross-Dataset Evaluation
# =============================================================================

class CrossDatasetEvaluator:
    """
    Cross-dataset generalization evaluation.

    Two directions (from paper, Section VI):
        1. Train on resting-state → Test on olfactory
        2. Train on olfactory → Test on resting-state

    Tests whether dynamical descriptors generalize across different
    EEG recording conditions.
    """

    def __init__(self, classifier_factory, label_map: Dict[str, int]):
        self.classifier_factory = classifier_factory
        self.label_map = label_map

    def _prepare_data(
        self,
        descriptors: List[np.ndarray],
        labels: List[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Stack and clean descriptor arrays."""
        Z = np.concatenate(descriptors, axis=0)
        y = np.concatenate([
            np.full(len(d), self.label_map.get(lbl, -1))
            for d, lbl in zip(descriptors, labels)
        ])
        valid = np.all(np.isfinite(Z), axis=1) & (y >= 0)
        return Z[valid], y[valid]

    def evaluate(
        self,
        # Source domain (training)
        source_descriptors: List[np.ndarray],
        source_labels: List[str],
        # Target domain (testing)
        target_descriptors: List[np.ndarray],
        target_labels: List[str],
        direction_name: str = "Source→Target",
    ) -> Dict:
        """
        Train on source, evaluate on target.
        """
        Z_train, y_train = self._prepare_data(source_descriptors, source_labels)
        Z_test, y_test = self._prepare_data(target_descriptors, target_labels)

        if len(Z_train) == 0 or len(Z_test) == 0:
            return {"error": "Insufficient data"}

        # Only keep classes present in both
        common_classes = np.intersect1d(np.unique(y_train), np.unique(y_test))
        if len(common_classes) < 2:
            return {"error": f"Need ≥2 common classes; found {common_classes}"}

        mask_train = np.isin(y_train, common_classes)
        mask_test = np.isin(y_test, common_classes)

        Z_train, y_train = Z_train[mask_train], y_train[mask_train]
        Z_test, y_test = Z_test[mask_test], y_test[mask_test]

        clf = self.classifier_factory()
        clf.fit(Z_train, y_train)
        y_pred = clf.predict(Z_test)

        metrics = compute_metrics(y_test, y_pred, average="weighted")
        metrics["direction"] = direction_name
        metrics["n_train"] = len(Z_train)
        metrics["n_test"] = len(Z_test)

        logger.info(
            f"Cross-dataset [{direction_name}]: "
            f"Acc={metrics['accuracy']*100:.2f}%, F1={metrics['f1']*100:.2f}%"
        )
        return metrics


# =============================================================================
# Ablation Study (Phase 11)
# =============================================================================

class AblationStudy:
    """
    Ablation study evaluating the contribution of each descriptor.

    Removes one descriptor at a time and measures performance degradation.
    Tests from the paper:
        Remove: Lyapunov, Entropy, Diffusion, Energy, Transition density

    Descriptor vector z = [λ, H, D, E, T] with indices [0, 1, 2, 3, 4]
    """

    DESCRIPTOR_NAMES = {
        0: "Lyapunov (λ)",
        1: "Entropy (H)",
        2: "Diffusion (D)",
        3: "Energy (E)",
        4: "Transition (T)",
    }

    def __init__(
        self,
        classifier_factory,
        label_map: Dict[str, int],
        metrics_avg: str = "weighted",
    ):
        self.classifier_factory = classifier_factory
        self.label_map = label_map
        self.metrics_avg = metrics_avg

    def run(
        self,
        Z_train: np.ndarray,    # (n_train, 5)
        y_train: np.ndarray,
        Z_test: np.ndarray,     # (n_test, 5)
        y_test: np.ndarray,
    ) -> Dict:
        """
        Run ablation by removing each feature dimension.

        Returns:
            results: dict mapping component name → metrics
        """
        results = {}

        # Baseline: all descriptors
        clf_full = self.classifier_factory()
        clf_full.fit(Z_train, y_train)
        y_pred_full = clf_full.predict(Z_test)
        results["Full (all descriptors)"] = compute_metrics(
            y_test, y_pred_full, average=self.metrics_avg
        )

        # Remove each descriptor
        n_descriptors = Z_train.shape[1]
        for idx in range(n_descriptors):
            name = self.DESCRIPTOR_NAMES.get(idx, f"Descriptor_{idx}")
            label = f"Without {name}"

            # Create ablated feature set
            keep_idx = [i for i in range(n_descriptors) if i != idx]
            Z_train_abl = Z_train[:, keep_idx]
            Z_test_abl = Z_test[:, keep_idx]

            # Skip degenerate cases
            valid_train = np.all(np.isfinite(Z_train_abl), axis=1)
            valid_test = np.all(np.isfinite(Z_test_abl), axis=1)

            if valid_train.sum() < 5:
                results[label] = {"error": "insufficient_data"}
                continue

            clf_abl = self.classifier_factory()
            try:
                clf_abl.fit(Z_train_abl[valid_train], y_train[valid_train])
                y_pred_abl = clf_abl.predict(Z_test_abl[valid_test])
                m = compute_metrics(y_test[valid_test], y_pred_abl, average=self.metrics_avg)
            except Exception as e:
                logger.debug(f"Ablation {label} failed: {e}")
                m = {"error": str(e)}

            results[label] = m

        # Also test proposed classifier vs. baselines
        return results

    def print_ablation_report(self, results: Dict) -> None:
        """Print ablation study results."""
        print("\n" + "=" * 60)
        print("  ABLATION STUDY")
        print("=" * 60)
        print(f"{'Component':<30} {'Accuracy':>10} {'F1':>10} {'Drop (Acc)':>12}")
        print("-" * 60)

        full_acc = results.get("Full (all descriptors)", {}).get("accuracy", 0)

        for name, m in results.items():
            if "error" in m:
                print(f"{name:<30} {'ERROR':>10}")
                continue
            acc = m.get("accuracy", 0)
            f1 = m.get("f1", 0)
            drop = (full_acc - acc) * 100 if name != "Full (all descriptors)" else 0.0
            print(f"{name:<30} {acc*100:>9.2f}% {f1*100:>9.2f}% {drop:>+11.2f}%")
        print("=" * 60)
