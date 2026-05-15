"""
DynaSys-EEG Pipeline — Main Orchestrator

This module ties all components together following the paper's implementation plan:

Phase 1:  Data Acquisition and Organization
Phase 2:  Signal Segmentation (5s windows, 50% overlap)
Phase 3:  Preprocessing (bandpass + normalization)
Phase 4:  State Space Reconstruction (Takens' theorem)
Phase 5:  Dynamical System Learning (neural f_θ)
Phase 6:  System Descriptor Computation (λ, H, D, E, T)
Phase 7:  Descriptor Vector Formation  z = [λ, H, D, E, T]
Phase 8:  Classification (3 strategies)
Phase 9:  LOSO Validation
Phase 10: Baseline Comparison
Phase 11: Ablation Study
Phase 12: Results Reporting
"""

import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("dynasys_eeg.pipeline")


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


class DynaSysPipeline:
    """
    End-to-end DynaSys-EEG pipeline.

    Usage:
        pipeline = DynaSysPipeline(config=cfg)

        # With real data:
        pipeline.load_data(primary_dir="data/primary", secondary_dir="data/secondary")

        # Or with synthetic data (for testing):
        pipeline.generate_synthetic_data()

        # Run full pipeline:
        results = pipeline.run()
    """

    def __init__(self, config=None, device: Optional[str] = None):
        from dynasys_eeg.configs import cfg
        self.cfg = config or cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        set_seed(self.cfg.seed)

        logger.info(f"DynaSys-EEG Pipeline initialized | Device: {self.device}")

        # Internal state
        self._primary_subjects = None
        self._secondary_subjects = None
        self._primary_dataset = None
        self._secondary_dataset = None

        # Results
        self.results = {}

    # =========================================================================
    # Phase 1: Data Loading
    # =========================================================================

    def generate_synthetic_data(
        self,
        primary_counts: Dict[str, int] = None,
        secondary_counts: Dict[str, int] = None,
        duration_sec: float = 60.0,
    ):
        """Generate synthetic data for testing and development."""
        from dynasys_eeg.data import SyntheticEEGGenerator, EEGPreprocessor, EEGDataset

        logger.info("Phase 1: Generating synthetic EEG data...")

        primary_counts = primary_counts or {"AD": 10, "FTD": 8, "HC": 10}
        secondary_counts = secondary_counts or {"AD": 5, "aMCI": 4, "HC": 6}

        gen_primary = SyntheticEEGGenerator(
            sfreq=self.cfg.primary.sfreq, seed=self.cfg.seed
        )
        gen_secondary = SyntheticEEGGenerator(
            sfreq=self.cfg.secondary.sfreq, seed=self.cfg.seed + 1
        )

        self._primary_subjects = gen_primary.generate_dataset(
            primary_counts,
            n_channels=self.cfg.primary.n_channels,
            duration_sec=duration_sec,
        )
        self._secondary_subjects = gen_secondary.generate_dataset(
            secondary_counts,
            n_channels=self.cfg.secondary.n_channels,
            duration_sec=duration_sec,
        )
        logger.info(
            f"  Primary: {len(self._primary_subjects)} subjects, "
            f"Secondary: {len(self._secondary_subjects)} subjects"
        )

    def load_data(self, primary_dir: str, secondary_dir: Optional[str] = None):
        """Load real EEG data from directories."""
        from dynasys_eeg.data import RestingStateLoader, OlfactoryLoader

        logger.info("Phase 1: Loading EEG data...")
        loader = RestingStateLoader(primary_dir, sfreq=self.cfg.primary.sfreq)
        self._primary_subjects = loader.load()
        logger.info(f"  Primary: {len(self._primary_subjects)} subjects loaded")

        if secondary_dir:
            loader2 = OlfactoryLoader(secondary_dir, sfreq=self.cfg.secondary.sfreq)
            self._secondary_subjects = loader2.load()
            logger.info(f"  Secondary: {len(self._secondary_subjects)} subjects loaded")

    # =========================================================================
    # Phases 2-3: Preprocessing and Segmentation
    # =========================================================================

    def preprocess(self):
        """Apply preprocessing pipeline to all subjects."""
        from dynasys_eeg.data import EEGPreprocessor, EEGDataset

        logger.info("Phases 2-3: Preprocessing and segmentation...")

        # Primary dataset preprocessor
        self._primary_preprocessor = EEGPreprocessor(
            sfreq=self.cfg.primary.sfreq,
            lowcut=self.cfg.preprocessing.lowcut,
            highcut=self.cfg.preprocessing.highcut,
            normalize=self.cfg.preprocessing.normalize,
        )

        # Secondary dataset preprocessor
        self._secondary_preprocessor = EEGPreprocessor(
            sfreq=self.cfg.secondary.sfreq,
            lowcut=self.cfg.preprocessing.lowcut,
            highcut=40.0,  # olfactory EEG: 0.5-40 Hz
            normalize=self.cfg.preprocessing.normalize,
        )

        # Build primary dataset (use full class names like 'AD', 'FTD', 'HC')
        primary_label_map = {
            cls: i for i, cls in enumerate(self.cfg.primary.class_names)
        }
        self._primary_dataset = EEGDataset(
            subjects=self._primary_subjects,
            preprocessor=self._primary_preprocessor,
            window_sec=self.cfg.preprocessing.window_sec,
            overlap=self.cfg.preprocessing.overlap,
            label_map=primary_label_map,
        )

        # Build secondary dataset (use full class names like 'AD', 'aMCI', 'HC')
        secondary_label_map = {
            cls: i for i, cls in enumerate(self.cfg.secondary.class_names)
        }
        if self._secondary_subjects:
            self._secondary_dataset = EEGDataset(
                subjects=self._secondary_subjects,
                preprocessor=self._secondary_preprocessor,
                window_sec=self.cfg.preprocessing.window_sec,
                overlap=self.cfg.preprocessing.overlap,
                label_map=secondary_label_map,
            )
        logger.info("  Preprocessing complete.")

    # =========================================================================
    # Phase 4: State Space Reconstruction
    # =========================================================================

    def fit_state_space(self, pilot_signal: np.ndarray) -> "StateSpaceReconstructor":
        """Fit state space reconstructor on a pilot signal."""
        from dynasys_eeg.features import StateSpaceReconstructor

        logger.info("Phase 4: Fitting state space reconstructor (Takens embedding)...")
        reconstructor = StateSpaceReconstructor(
            embedding_dim=self.cfg.state_space.embedding_dim,
            time_delay=self.cfg.state_space.time_delay,
            method_delay=self.cfg.state_space.method_delay,
            method_dim=self.cfg.state_space.method_dim,
        )
        reconstructor.fit(pilot_signal)
        logger.info(
            f"  Selected τ={reconstructor.time_delay}, m={reconstructor.embedding_dim}"
        )
        return reconstructor

    # =========================================================================
    # Phase 5: Dynamical System Learning (optional — for descriptor improvement)
    # =========================================================================

    def train_dynamics_model(self, states: np.ndarray) -> "DynamicsTrainer":
        """
        Train the neural dynamical system on delay-embedded states.
        This enables better diffusion coefficient estimation.
        """
        from dynasys_eeg.models import StochasticDynamicsNet, DynamicsTrainer

        state_dim = states.shape[1]
        logger.info(f"Phase 5: Training dynamics model (state_dim={state_dim})...")

        model = StochasticDynamicsNet(
            input_dim=state_dim,
            hidden_dims=self.cfg.dynamics.hidden_dims,
            dropout=self.cfg.dynamics.dropout,
        )
        trainer = DynamicsTrainer(model, lr=self.cfg.dynamics.learning_rate, device=self.device)
        trainer.train(
            states,
            n_epochs=self.cfg.dynamics.n_epochs,
            batch_size=self.cfg.dynamics.batch_size,
            patience=self.cfg.dynamics.patience,
            verbose=True,
        )
        return trainer

    # =========================================================================
    # Phases 6-7: Descriptor Extraction
    # =========================================================================

    def extract_descriptors(
        self,
        dataset: "EEGDataset",
        label_map: Dict[str, int],
        reconstructor=None,
    ) -> Tuple[List[np.ndarray], List[str], List[str]]:
        """
        Extract z = [λ, H, D, E, T] for each subject's segments.

        Returns:
            Z_per_subject: list of (n_seg, 5) descriptor arrays
            labels: list of label strings
            ids: list of subject IDs
        """
        from dynasys_eeg.features import DynamicalDescriptorExtractor, StateSpaceReconstructor

        logger.info("Phases 6-7: Extracting dynamical descriptors...")

        extractor = DynamicalDescriptorExtractor(
            embedding_dim=self.cfg.state_space.embedding_dim,
            time_delay=self.cfg.state_space.time_delay,
            sfreq=list(label_map.keys())[0] if False else 500.0,
            entropy_method=self.cfg.descriptors.entropy_method,
        )

        segments_per_subj, labels_per_subj, ids_per_subj = dataset.get_subject_level()

        Z_per_subject = []
        for i, (segs, lbl, sid) in enumerate(
            zip(segments_per_subj, labels_per_subj, ids_per_subj)
        ):
            logger.info(f"  [{i+1}/{len(segments_per_subj)}] Subject {sid} ({lbl}): {len(segs)} segments")

            if reconstructor is None:
                # Auto-fit on first channel
                ssr = StateSpaceReconstructor(
                    embedding_dim=self.cfg.state_space.embedding_dim,
                    time_delay=self.cfg.state_space.time_delay,
                )
                ssr.fit(segs[0, 0])
            else:
                ssr = reconstructor

            Z_subj = extractor.extract_batch(segs, ssr)
            Z_per_subject.append(Z_subj)

        logger.info(f"  Descriptors extracted for {len(Z_per_subject)} subjects.")
        return Z_per_subject, labels_per_subj, ids_per_subj

    # =========================================================================
    # Phase 8-10: Classification + Baseline
    # =========================================================================

    def run_classification_comparison(
        self,
        Z_train: np.ndarray,
        y_train: np.ndarray,
        Z_test: np.ndarray,
        y_test: np.ndarray,
        label_map: Dict[str, int],
    ) -> Dict:
        """Compare all DynaSys classifiers against baselines."""
        from dynasys_eeg.classification import (
            NonlinearDynamicsClassifier,
            PrototypeClassifier,
            EnergyBasedClassifier,
            DynaSysEnsemble,
            get_baseline_classifiers,
        )
        from dynasys_eeg.evaluation import compute_metrics, print_results_table

        n_classes = len(np.unique(y_train))
        results = {}

        # DynaSys classifiers
        dynasys_clfs = {
            "DynaSys-Nonlinear": NonlinearDynamicsClassifier(n_classes=n_classes),
            "DynaSys-Prototype": PrototypeClassifier(),
            "DynaSys-Energy":    EnergyBasedClassifier(),
            "DynaSys-Ensemble":  DynaSysEnsemble(n_classes=n_classes),
        }

        for name, clf in dynasys_clfs.items():
            try:
                clf.fit(Z_train, y_train)
                y_pred = clf.predict(Z_test)
                y_proba = clf.predict_proba(Z_test)
                results[name] = compute_metrics(y_test, y_pred, y_proba)
            except Exception as e:
                logger.warning(f"Classifier {name} failed: {e}")

        # Baselines
        for name, clf in get_baseline_classifiers().items():
            try:
                clf.fit(Z_train, y_train)
                y_pred = clf.predict(Z_test)
                results[f"Baseline-{name}"] = compute_metrics(y_test, y_pred)
            except Exception as e:
                logger.warning(f"Baseline {name} failed: {e}")

        print_results_table(results, "DynaSys-EEG Classification Results")
        return results

    # =========================================================================
    # Phase 9: LOSO Validation
    # =========================================================================

    def run_loso(
        self,
        Z_per_subject: List[np.ndarray],
        labels_per_subject: List[str],
        ids_per_subject: List[str],
        label_map: Dict[str, int],
        classifier_name: str = "DynaSys-Prototype",
    ) -> Dict:
        """Run LOSO cross-validation."""
        from dynasys_eeg.classification import (
            PrototypeClassifier, NonlinearDynamicsClassifier, DynaSysEnsemble
        )
        from dynasys_eeg.evaluation import LOSOEvaluator

        logger.info(f"Phase 9: LOSO evaluation with {classifier_name}...")

        clf_factories = {
            "DynaSys-Prototype": PrototypeClassifier,
            "DynaSys-Nonlinear": NonlinearDynamicsClassifier,
            "DynaSys-Ensemble":  DynaSysEnsemble,
        }
        factory = clf_factories.get(classifier_name, PrototypeClassifier)

        evaluator = LOSOEvaluator(factory, label_map)
        loso_results = evaluator.evaluate(
            Z_per_subject, labels_per_subject, ids_per_subject, verbose=False
        )

        agg = loso_results.get("aggregate", {})
        logger.info(
            f"  LOSO Results: Acc={agg.get('accuracy', 0)*100:.2f}%, "
            f"F1={agg.get('f1', 0)*100:.2f}%"
        )
        return loso_results

    # =========================================================================
    # Phase 11: Ablation Study
    # =========================================================================

    def run_ablation(
        self,
        Z_train: np.ndarray,
        y_train: np.ndarray,
        Z_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict:
        """Run ablation study."""
        from dynasys_eeg.classification import PrototypeClassifier
        from dynasys_eeg.evaluation import AblationStudy

        logger.info("Phase 11: Running ablation study...")

        ablation = AblationStudy(PrototypeClassifier, label_map={})
        results = ablation.run(Z_train, y_train, Z_test, y_test)
        ablation.print_ablation_report(results)
        return results

    # =========================================================================
    # Main Run Method
    # =========================================================================

    def run(
        self,
        save_results: bool = True,
        results_dir: str = "results",
        run_dynamics_training: bool = False,
    ) -> Dict:
        """
        Execute the complete DynaSys-EEG pipeline.

        Args:
            save_results: whether to save plots and results
            results_dir: output directory
            run_dynamics_training: whether to train neural dynamics model (Phase 5)

        Returns:
            results: comprehensive results dictionary
        """
        t_start = time.time()
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        # Ensure data is loaded
        if self._primary_subjects is None:
            logger.warning("No data loaded. Generating synthetic data...")
            self.generate_synthetic_data()

        # Preprocessing
        self.preprocess()

        # Build primary label map (use full class names like 'AD', 'FTD', 'HC')
        primary_label_map = {cls: i for i, cls in enumerate(self.cfg.primary.class_names)}
        secondary_label_map = {cls: i for i, cls in enumerate(self.cfg.secondary.class_names)}

        # ===== Phase 4-7: Extract descriptors for primary dataset =====
        logger.info("\n" + "="*50)
        logger.info("PRIMARY DATASET (Resting-State EEG: AD, FTD, HC)")
        logger.info("="*50)

        Z_primary, labels_primary, ids_primary = self.extract_descriptors(
            self._primary_dataset, primary_label_map
        )

        # Filter valid descriptors
        Z_primary_valid = []
        labels_primary_valid = []
        ids_primary_valid = []
        for Z_subj, lbl, sid in zip(Z_primary, labels_primary, ids_primary):
            valid_mask = np.all(np.isfinite(Z_subj), axis=1)
            if valid_mask.sum() >= 1:
                Z_primary_valid.append(Z_subj[valid_mask])
                labels_primary_valid.append(lbl)
                ids_primary_valid.append(sid)

        # ===== Phase 9: LOSO on primary dataset =====
        logger.info("\n" + "="*50)
        logger.info("LOSO EVALUATION — Primary Dataset")
        logger.info("="*50)

        loso_results_primary = self.run_loso(
            Z_primary_valid,
            labels_primary_valid,
            ids_primary_valid,
            primary_label_map,
            classifier_name="DynaSys-Prototype",
        )
        self.results["loso_primary"] = loso_results_primary

        # ===== Cross-dataset evaluation =====
        secondary_results = {}
        loso_results_secondary = {}

        if self._secondary_subjects and self._secondary_dataset:
            logger.info("\n" + "="*50)
            logger.info("SECONDARY DATASET (Olfactory EEG: AD, aMCI, HC)")
            logger.info("="*50)

            Z_secondary, labels_secondary, ids_secondary = self.extract_descriptors(
                self._secondary_dataset, secondary_label_map
            )

            Z_secondary_valid = []
            labels_secondary_valid = []
            ids_secondary_valid = []
            for Z_subj, lbl, sid in zip(Z_secondary, labels_secondary, ids_secondary):
                valid_mask = np.all(np.isfinite(Z_subj), axis=1)
                if valid_mask.sum() >= 1:
                    Z_secondary_valid.append(Z_subj[valid_mask])
                    labels_secondary_valid.append(lbl)
                    ids_secondary_valid.append(sid)

            loso_results_secondary = self.run_loso(
                Z_secondary_valid,
                labels_secondary_valid,
                ids_secondary_valid,
                secondary_label_map,
                classifier_name="DynaSys-Prototype",
            )
            self.results["loso_secondary"] = loso_results_secondary

            # Cross-dataset: Train primary → Test secondary
            from dynasys_eeg.classification import PrototypeClassifier
            from dynasys_eeg.evaluation import CrossDatasetEvaluator

            cd_evaluator = CrossDatasetEvaluator(PrototypeClassifier, primary_label_map)
            # Use only common classes
            common_classes = list(set(labels_primary_valid) & set(labels_secondary_valid))
            if len(common_classes) >= 2:
                Z_prim_filtered = [Z for Z, lbl in zip(Z_primary_valid, labels_primary_valid)
                                   if lbl in common_classes]
                lbl_prim_filtered = [lbl for lbl in labels_primary_valid if lbl in common_classes]
                Z_sec_filtered = [Z for Z, lbl in zip(Z_secondary_valid, labels_secondary_valid)
                                  if lbl in common_classes]
                lbl_sec_filtered = [lbl for lbl in labels_secondary_valid if lbl in common_classes]

                cross_primary2secondary = cd_evaluator.evaluate(
                    Z_prim_filtered, lbl_prim_filtered,
                    Z_sec_filtered, lbl_sec_filtered,
                    direction_name="Resting→Olfactory"
                )
                self.results["cross_dataset"] = {
                    "Resting→Olfactory": cross_primary2secondary,
                }

        # ===== Phase 10-11: Classifier comparison + Ablation =====
        logger.info("\n" + "="*50)
        logger.info("CLASSIFIER COMPARISON + ABLATION STUDY")
        logger.info("="*50)

        # Build pooled train/test split
        if len(Z_primary_valid) >= 4:
            from sklearn.model_selection import train_test_split

            Z_all = np.concatenate(Z_primary_valid, axis=0)
            y_all = np.concatenate([
                np.full(len(Z), primary_label_map[lbl])
                for Z, lbl in zip(Z_primary_valid, labels_primary_valid)
            ])

            valid_mask = np.all(np.isfinite(Z_all), axis=1)
            Z_all, y_all = Z_all[valid_mask], y_all[valid_mask]

            if len(np.unique(y_all)) >= 2:
                Z_train, Z_test, y_train, y_test = train_test_split(
                    Z_all, y_all, test_size=0.25, stratify=y_all, random_state=42
                )

                # Classifier comparison
                clf_results = self.run_classification_comparison(
                    Z_train, y_train, Z_test, y_test, primary_label_map
                )
                self.results["classification"] = clf_results

                # Ablation
                ablation_results = self.run_ablation(Z_train, y_train, Z_test, y_test)
                self.results["ablation"] = ablation_results

                # ===== Visualizations =====
                if save_results:
                    self._save_visualizations(
                        Z_all=Z_all,
                        y_all=y_all,
                        Z_primary_valid=Z_primary_valid,
                        labels_primary_valid=labels_primary_valid,
                        primary_label_map=primary_label_map,
                        loso_results=loso_results_primary,
                        ablation_results=ablation_results,
                        clf_results=clf_results,
                        results_dir=results_dir,
                    )

        t_elapsed = time.time() - t_start
        logger.info(f"\n✅ Pipeline complete in {t_elapsed:.1f}s")
        logger.info(f"Results saved to: {results_dir}/")

        return self.results

    def _save_visualizations(
        self,
        Z_all, y_all,
        Z_primary_valid, labels_primary_valid,
        primary_label_map,
        loso_results, ablation_results, clf_results,
        results_dir,
    ):
        """Save all visualizations to disk."""
        from dynasys_eeg.utils.visualization import (
            plot_descriptors_distribution,
            plot_confusion_matrix,
            plot_ablation_results,
            plot_comparison_results,
            plot_loso_results_per_subject,
        )

        inv_label_map = {v: k for k, v in primary_label_map.items()}
        class_names = [inv_label_map[i] for i in sorted(inv_label_map.keys())]

        logger.info("Saving visualizations...")

        # 1. Descriptor distributions
        plot_descriptors_distribution(
            Z_all, y_all, inv_label_map,
            save_path=str(results_dir / "descriptor_distributions.png")
        )

        # 2. Confusion matrix (LOSO)
        cm = loso_results.get("confusion_matrix")
        if cm is not None:
            plot_confusion_matrix(
                cm, class_names,
                title="Confusion Matrix (LOSO - DynaSys-Prototype)",
                save_path=str(results_dir / "confusion_matrix_loso.png")
            )

        # 3. Ablation study
        if ablation_results:
            plot_ablation_results(
                ablation_results,
                save_path=str(results_dir / "ablation_study.png")
            )

        # 4. Classifier comparison
        if clf_results:
            plot_comparison_results(
                clf_results,
                title="DynaSys-EEG vs. Baselines (Primary Dataset)",
                save_path=str(results_dir / "method_comparison.png")
            )

        # 5. LOSO per-subject
        if loso_results.get("per_subject"):
            plot_loso_results_per_subject(
                loso_results, inv_label_map,
                save_path=str(results_dir / "loso_per_subject.png")
            )

        logger.info(f"  Plots saved to {results_dir}/")
