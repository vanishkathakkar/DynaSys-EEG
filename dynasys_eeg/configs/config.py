"""
Configuration file for DynaSys-EEG framework.
All hyperparameters, paths, and experimental settings are centralized here.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# =============================================================================
# Dataset Configurations
# =============================================================================

@dataclass
class PrimaryDatasetConfig:
    """
    Primary Dataset: Resting-State EEG
    - 88 subjects: AD (36), FTD (23), HC (29)
    - 19 electrodes (10-20 system)
    - Sampling rate: 500 Hz
    Reference: Openneuro ds004504 (Carabez et al., 2022)
    """
    name: str = "resting_state_eeg"
    n_subjects: int = 88
    n_channels: int = 19
    sfreq: float = 500.0  # Hz
    classes: List[str] = field(default_factory=lambda: ["A", "F", "C"])
    class_names: List[str] = field(default_factory=lambda: ["AD", "FTD", "HC"])
    n_classes: int = 3
    channel_names: List[str] = field(default_factory=lambda: [
        "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4",
        "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6",
        "Fz", "Cz", "Pz"
    ])
    data_dir: str = "data/primary"


@dataclass
class SecondaryDatasetConfig:
    """
    Secondary Dataset: Olfactory Stimulus EEG
    - 35 subjects: AD (13), aMCI (7), HC (15)
    - 4 electrodes: Fp1, Fz, Cz, Pz
    - Sampling rate: 2000 Hz → downsampled to 200 Hz
    - Stimuli: Lemon (75%), Rose (25%)
    """
    name: str = "olfactory_eeg"
    n_subjects: int = 35
    n_channels: int = 4
    sfreq_original: float = 2000.0  # Hz
    sfreq: float = 200.0  # Hz after downsampling
    classes: List[str] = field(default_factory=lambda: ["AD", "aMCI", "HC"])
    class_names: List[str] = field(default_factory=lambda: ["AD", "aMCI", "HC"])
    n_classes: int = 3
    channel_names: List[str] = field(default_factory=lambda: ["Fp1", "Fz", "Cz", "Pz"])
    stimuli: List[str] = field(default_factory=lambda: ["lemon", "rose"])
    data_dir: str = "data/secondary"


# =============================================================================
# Preprocessing Configuration
# =============================================================================

@dataclass
class PreprocessingConfig:
    """Signal preprocessing parameters."""
    # Bandpass filter
    lowcut: float = 0.5   # Hz
    highcut: float = 45.0  # Hz (40 Hz for olfactory)
    filter_order: int = 4

    # Segmentation
    window_sec: float = 5.0       # seconds
    overlap: float = 0.5          # 50% overlap
    min_segment_sec: float = 2.0  # minimum segment length

    # ICA
    n_ica_components: int = 15

    # Normalization
    normalize: bool = True        # z-score normalization per channel

    # Artifact rejection
    amplitude_threshold: float = 150.0  # µV


# =============================================================================
# State Space Reconstruction (Takens' Theorem)
# =============================================================================

@dataclass
class StateSpaceConfig:
    """
    Delay Embedding Configuration (Takens' Theorem):
    X_t = [x_t, x_{t-τ}, x_{t-2τ}, ..., x_{t-(m-1)τ}]
    """
    embedding_dim: int = 5          # m: embedding dimension
    time_delay: int = 10            # τ: time delay (samples)
    method_delay: str = "mutual_info"   # 'mutual_info' or 'autocorr'
    method_dim: str = "fnn"             # 'fnn' (False Nearest Neighbors) or 'cao'
    max_dim: int = 10               # max embedding dimension for FNN search
    fnn_threshold: float = 0.05     # FNN threshold for dim selection


# =============================================================================
# Dynamical System Learning
# =============================================================================

@dataclass
class DynamicsConfig:
    """Neural network for learning system dynamics: X_{t+1} = f_θ(X_t)"""
    # Network architecture
    hidden_dims: List[int] = field(default_factory=lambda: [128, 256, 128])
    activation: str = "tanh"       # tanh for smooth dynamical systems
    dropout: float = 0.2

    # Training
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    n_epochs: int = 100
    patience: int = 15             # early stopping patience
    scheduler_step: int = 30
    scheduler_gamma: float = 0.5

    # Loss
    loss: str = "mse"              # MSE reconstruction loss


# =============================================================================
# System Descriptor Configuration
# =============================================================================

@dataclass
class DescriptorConfig:
    """
    Configuration for computing dynamical system descriptors:
    z = [λ, H, D, E, T]
    """
    # Lyapunov exponent (λ)
    lyapunov_method: str = "rosenstein"  # 'rosenstein' or 'wolf'
    lyapunov_n_iter: int = 500
    lyapunov_lag: int = 1

    # Entropy (H)
    entropy_method: str = "sample"  # 'sample', 'approximate', 'permutation'
    entropy_m: int = 2              # template length
    entropy_r: float = 0.2         # tolerance (fraction of std)

    # Diffusion coefficient (D)
    diffusion_n_bins: int = 50

    # Energy landscape (E)
    energy_n_bins: int = 50
    energy_bandwidth: float = 0.5  # KDE bandwidth

    # Transition density (T)
    transition_n_states: int = 10  # number of discrete states for T

    # Final descriptor vector dimension
    # [λ(1), H(1), D(1), E(1), T(1)] = 5-dimensional
    descriptor_dim: int = 5


# =============================================================================
# Classification Configuration
# =============================================================================

@dataclass
class ClassifierConfig:
    """
    Three classification strategies:
    1. Nonlinear decision function: y = argmax_k(w_k^T z + α_k ||z||² + β_k φ(z))
    2. Prototype-based (Mahalanobis): y = argmin_k (z - µ_k)^T Σ_k^{-1} (z - µ_k)
    3. Energy-based: y = argmin_k E_k(z)
    """
    # Nonlinear decision
    rbf_gamma: float = 0.5

    # Prototype-based
    use_mahalanobis: bool = True
    regularization: float = 1e-6  # covariance regularization

    # Energy-based
    energy_sigma: float = 1.0

    # Baselines
    baselines: List[str] = field(default_factory=lambda: [
        "logistic_regression", "svm", "random_forest", "knn"
    ])


# =============================================================================
# Validation Configuration
# =============================================================================

@dataclass
class ValidationConfig:
    """Experimental validation settings."""
    # LOSO: Leave-One-Subject-Out
    loso: bool = True

    # Cross-dataset
    cross_dataset: bool = True

    # Ablation study components
    ablation_components: List[str] = field(default_factory=lambda: [
        "lyapunov", "entropy", "diffusion", "energy", "transition"
    ])

    # Metrics
    metrics: List[str] = field(default_factory=lambda: [
        "accuracy", "precision", "recall", "f1", "roc_auc"
    ])
    average: str = "weighted"  # for multi-class


# =============================================================================
# Global Config
# =============================================================================

@dataclass
class DynaSysConfig:
    """Master configuration for DynaSys-EEG."""
    primary: PrimaryDatasetConfig = field(default_factory=PrimaryDatasetConfig)
    secondary: SecondaryDatasetConfig = field(default_factory=SecondaryDatasetConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    state_space: StateSpaceConfig = field(default_factory=StateSpaceConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    descriptors: DescriptorConfig = field(default_factory=DescriptorConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)

    # Paths
    project_root: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    results_dir: str = "results"
    models_dir: str = "saved_models"

    # Reproducibility
    seed: int = 42

    # Compute
    device: str = "cuda"  # 'cuda' or 'cpu'
    n_jobs: int = -1       # parallelism


# Default config instance
cfg = DynaSysConfig()
