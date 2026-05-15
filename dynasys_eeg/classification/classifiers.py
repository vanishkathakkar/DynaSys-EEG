"""
DynaSys-EEG Classification Module (Phase 8).

Three classification strategies operating in the space of dynamical descriptors z:

1. Nonlinear Decision Function (Eq. 7, 16, 24):
   y = argmax_k (w_k^T z + α_k ||z||² + β_k φ(z))
   Where φ(z) = RBF kernel features (nonlinear interaction between descriptors)

2. Prototype-Based (Mahalanobis Distance) (Eq. 8, 17, 25):
   y = argmin_k (z - µ_k)^T Σ_k^{-1} (z - µ_k)
   Each disease corresponds to a dynamical prototype µ_k

3. Energy-Based Classification (Eq. 9, 18, 26):
   y = argmin_k E_k(z)
   Assigns class based on energy minima (Boltzmann distribution)

Plus baseline classifiers for comparison (Phase 10).
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)


# =============================================================================
# 1. Nonlinear Decision Function Classifier
# =============================================================================

class NonlinearDynamicsClassifier(BaseEstimator, ClassifierMixin):
    """
    Decision function (Eq. 16 from paper):
        y = argmax_k (w_k^T z + α_k ||z||² + β_k φ(z))

    φ(z) are RBF kernel features modeling nonlinear descriptor interactions.
    The three terms are:
        w_k^T z       → linear contribution of each descriptor
        α_k ||z||²    → global magnitude effect (quadratic)
        β_k φ(z)      → nonlinear interaction between dynamical properties

    Trained via multinomial logistic regression over augmented features.
    """

    def __init__(
        self,
        n_classes: int = 3,
        rbf_gamma: float = 0.5,
        C: float = 1.0,
        max_iter: int = 1000,
    ):
        self.n_classes = n_classes
        self.rbf_gamma = rbf_gamma
        self.C = C
        self.max_iter = max_iter
        self.scaler = StandardScaler()
        self.classes_ = None
        self.W = None       # (n_classes, n_augmented_features)
        self.alpha_ = None  # per-class α for ||z||² term
        self.beta_ = None   # per-class β for φ(z) term

    def _rbf_features(self, z: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """Compute RBF kernel features: φ_i(z) = exp(-γ ||z - c_i||²)"""
        # centers: (n_centers, dim)
        # z: (n_samples, dim)
        dists_sq = np.sum((z[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2)
        return np.exp(-self.rbf_gamma * dists_sq)  # (n_samples, n_centers)

    def _augment_features(self, z: np.ndarray) -> np.ndarray:
        """
        Build augmented feature vector:
        [z, ||z||², φ(z)]  (linear + quadratic + nonlinear)
        """
        z_norm_sq = np.sum(z ** 2, axis=1, keepdims=True)  # (n, 1)
        rbf_feats = self._rbf_features(z, self.centers_)  # (n, n_centers)
        return np.hstack([z, z_norm_sq, rbf_feats])  # (n, dim + 1 + n_centers)

    def fit(self, z: np.ndarray, y: np.ndarray) -> "NonlinearDynamicsClassifier":
        """
        Args:
            z: (n_samples, 5) descriptor vectors
            y: (n_samples,) integer class labels
        """
        from sklearn.linear_model import LogisticRegression

        self.classes_ = np.unique(y)
        z_scaled = self.scaler.fit_transform(z)

        # Use class prototypes as RBF centers
        self.centers_ = np.array([
            z_scaled[y == cls].mean(axis=0)
            for cls in self.classes_
            if np.sum(y == cls) > 0
        ])

        z_aug = self._augment_features(z_scaled)

        self.clf_ = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            multi_class="multinomial",
            solver="lbfgs",
        )
        self.clf_.fit(z_aug, y)
        return self

    def predict(self, z: np.ndarray) -> np.ndarray:
        z_scaled = self.scaler.transform(z)
        z_aug = self._augment_features(z_scaled)
        return self.clf_.predict(z_aug)

    def predict_proba(self, z: np.ndarray) -> np.ndarray:
        z_scaled = self.scaler.transform(z)
        z_aug = self._augment_features(z_scaled)
        return self.clf_.predict_proba(z_aug)

    def decision_scores(self, z: np.ndarray) -> np.ndarray:
        """
        Compute raw decision scores: w_k^T z + α_k ||z||² + β_k φ(z)
        Returns (n_samples, n_classes) score matrix.
        """
        z_scaled = self.scaler.transform(z)
        z_aug = self._augment_features(z_scaled)
        return z_aug @ self.clf_.coef_.T + self.clf_.intercept_


# =============================================================================
# 2. Prototype-Based Classifier (Mahalanobis Distance)
# =============================================================================

class PrototypeClassifier(BaseEstimator, ClassifierMixin):
    """
    Prototype-based Mahalanobis distance classifier (Eq. 17, 25):
        y = argmin_k (z - µ_k)^T Σ_k^{-1} (z - µ_k)

    Each disease class is represented by:
        µ_k: class prototype (centroid of descriptor vectors)
        Σ_k: class covariance matrix

    Interpretation from paper:
        AD corresponds to high instability (λ > 0) + high diffusion (D large)
        HC corresponds to stable (λ < 0) + low diffusion (D small)
        FTD lies between with altered spectral properties
    """

    def __init__(self, regularization: float = 1e-4):
        self.regularization = regularization
        self.prototypes_ = {}   # µ_k for each class
        self.covariances_ = {}  # Σ_k for each class
        self.inv_cov_ = {}      # Σ_k^{-1}
        self.classes_ = None
        self.scaler = StandardScaler()

    def fit(self, z: np.ndarray, y: np.ndarray) -> "PrototypeClassifier":
        """Estimate prototypes and covariances per class."""
        z_scaled = self.scaler.fit_transform(z)
        self.classes_ = np.unique(y)

        for cls in self.classes_:
            mask = (y == cls)
            z_cls = z_scaled[mask]
            n_cls = z_cls.shape[0]

            # Prototype (centroid)
            self.prototypes_[cls] = z_cls.mean(axis=0)

            # Covariance with regularization
            if n_cls > 1:
                cov = np.cov(z_cls.T)
                if cov.ndim == 0:
                    cov = np.array([[cov]])
            else:
                cov = np.eye(z_cls.shape[1])

            # Add regularization for numerical stability
            cov += self.regularization * np.eye(cov.shape[0])
            self.covariances_[cls] = cov

            try:
                self.inv_cov_[cls] = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                self.inv_cov_[cls] = np.eye(cov.shape[0])

        return self

    def mahalanobis_distance(self, z: np.ndarray, cls: int) -> np.ndarray:
        """
        Compute (z - µ_k)^T Σ_k^{-1} (z - µ_k) for all samples.
        Returns: (n_samples,) distances
        """
        diff = z - self.prototypes_[cls]  # (n, dim)
        return np.einsum('ni,ij,nj->n', diff, self.inv_cov_[cls], diff)

    def predict(self, z: np.ndarray) -> np.ndarray:
        z_scaled = self.scaler.transform(z)
        distances = np.column_stack([
            self.mahalanobis_distance(z_scaled, cls)
            for cls in self.classes_
        ])  # (n_samples, n_classes)
        return self.classes_[np.argmin(distances, axis=1)]

    def predict_proba(self, z: np.ndarray) -> np.ndarray:
        """Convert Mahalanobis distances to probabilities via softmax."""
        z_scaled = self.scaler.transform(z)
        distances = np.column_stack([
            self.mahalanobis_distance(z_scaled, cls)
            for cls in self.classes_
        ])
        # Convert distance → similarity (negative distance) → softmax
        neg_dist = -distances
        neg_dist -= neg_dist.max(axis=1, keepdims=True)  # numerical stability
        exp_nd = np.exp(neg_dist)
        return exp_nd / exp_nd.sum(axis=1, keepdims=True)

    def get_prototype_report(self) -> Dict:
        """Report class prototypes for interpretability."""
        report = {}
        desc_names = ["Lyapunov (λ)", "Entropy (H)", "Diffusion (D)",
                      "Energy (E)", "Transition (T)"]
        for cls, proto in self.prototypes_.items():
            report[cls] = {name: float(val) for name, val in zip(desc_names, proto)}
        return report


# =============================================================================
# 3. Energy-Based Classifier
# =============================================================================

class EnergyBasedClassifier(BaseEstimator, ClassifierMixin):
    """
    Energy-based classification (Eq. 9, 18, 26):
        y = argmin_k E_k(z)

    Each disease state corresponds to an energy basin modeled by a
    Gaussian energy function:
        E_k(z) = (z - µ_k)^T Λ_k (z - µ_k) / (2σ_k²)

    Where Λ_k is a learned diagonal weight matrix emphasizing
    the most discriminative descriptors for class k.

    Equivalent to a Gaussian discriminant classifier.
    """

    def __init__(self, sigma: float = 1.0):
        self.sigma = sigma
        self.prototypes_ = {}
        self.weights_ = {}    # Λ_k diagonal weights
        self.classes_ = None
        self.scaler = StandardScaler()

    def fit(self, z: np.ndarray, y: np.ndarray) -> "EnergyBasedClassifier":
        """
        Learn energy basin parameters for each class.
        """
        z_scaled = self.scaler.fit_transform(z)
        self.classes_ = np.unique(y)

        # Compute inter-class discriminability for each descriptor
        overall_mean = z_scaled.mean(axis=0)
        overall_var = z_scaled.var(axis=0) + 1e-8

        for cls in self.classes_:
            mask = (y == cls)
            z_cls = z_scaled[mask]

            # Prototype
            self.prototypes_[cls] = z_cls.mean(axis=0)

            # Weights: how much each descriptor deviates from overall mean
            class_var = z_cls.var(axis=0) + 1e-8
            # Fisher-like ratio: between-class / within-class variance
            between = (self.prototypes_[cls] - overall_mean) ** 2
            weights = between / (class_var + 1e-8)
            weights = np.clip(weights, 0.1, 10.0)  # prevent extreme values
            self.weights_[cls] = weights

        return self

    def energy(self, z: np.ndarray, cls: int) -> np.ndarray:
        """
        Compute energy E_k(z) = (z - µ_k)^T Λ_k (z - µ_k) / (2σ²)
        Returns: (n_samples,) energy values
        """
        diff = z - self.prototypes_[cls]
        weighted = diff * self.weights_[cls]  # element-wise Λ_k weighting
        return np.sum(diff * weighted, axis=1) / (2 * self.sigma ** 2)

    def predict(self, z: np.ndarray) -> np.ndarray:
        z_scaled = self.scaler.transform(z)
        energies = np.column_stack([
            self.energy(z_scaled, cls) for cls in self.classes_
        ])
        return self.classes_[np.argmin(energies, axis=1)]

    def predict_proba(self, z: np.ndarray) -> np.ndarray:
        """Boltzmann distribution: P(k|z) ∝ exp(-E_k(z))"""
        z_scaled = self.scaler.transform(z)
        energies = np.column_stack([
            self.energy(z_scaled, cls) for cls in self.classes_
        ])
        neg_e = -energies
        neg_e -= neg_e.max(axis=1, keepdims=True)
        exp_e = np.exp(neg_e)
        return exp_e / exp_e.sum(axis=1, keepdims=True)


# =============================================================================
# Baseline Classifiers (Phase 10)
# =============================================================================

def get_baseline_classifiers() -> Dict:
    """
    Return baseline classifiers for comparison:
    - Logistic Regression
    - SVM
    - Random Forest
    - KNN
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline

    baselines = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, C=1.0)),
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True)),
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=200, random_state=42)),
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5)),
        ]),
    }
    return baselines


# =============================================================================
# Ensemble / Voting Classifier (combining all 3 strategies)
# =============================================================================

class DynaSysEnsemble(BaseEstimator, ClassifierMixin):
    """
    Ensemble of all three DynaSys classification strategies.
    Uses soft voting (average predicted probabilities).
    """

    def __init__(
        self,
        n_classes: int = 3,
        rbf_gamma: float = 0.5,
        mahal_reg: float = 1e-4,
        energy_sigma: float = 1.0,
    ):
        self.n_classes = n_classes
        self.nonlinear = NonlinearDynamicsClassifier(
            n_classes=n_classes, rbf_gamma=rbf_gamma
        )
        self.prototype = PrototypeClassifier(regularization=mahal_reg)
        self.energy = EnergyBasedClassifier(sigma=energy_sigma)
        self.classes_ = None

    def fit(self, z: np.ndarray, y: np.ndarray) -> "DynaSysEnsemble":
        self.classes_ = np.unique(y)
        self.nonlinear.fit(z, y)
        self.prototype.fit(z, y)
        self.energy.fit(z, y)
        return self

    def predict_proba(self, z: np.ndarray) -> np.ndarray:
        p1 = self.nonlinear.predict_proba(z)
        p2 = self.prototype.predict_proba(z)
        p3 = self.energy.predict_proba(z)
        return (p1 + p2 + p3) / 3.0

    def predict(self, z: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(z)
        idx = np.argmax(proba, axis=1)
        return self.classes_[idx]
