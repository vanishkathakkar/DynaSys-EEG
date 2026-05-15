"""
State Space Reconstruction via Takens' Delay Embedding Theorem (Phase 4).

Takens' theorem states that the dynamics of a system can be faithfully
reconstructed from time-delayed observations of a single variable.

X_t = [x_t, x_{t-τ}, x_{t-2τ}, ..., x_{t-(m-1)τ}]

Where:
  - m = embedding dimension (selected via False Nearest Neighbors)
  - τ = time delay (selected via Mutual Information)
"""

import logging
from typing import Optional, Tuple

import numpy as np
from scipy.signal import correlate
from scipy.stats import entropy as sp_entropy

logger = logging.getLogger(__name__)


def compute_mutual_information(
    x: np.ndarray, max_lag: int = 100, n_bins: int = 64
) -> np.ndarray:
    """
    Compute Average Mutual Information (AMI) for different time lags.
    The optimal delay τ is the first local minimum of AMI.

    I(X; X_{τ}) = sum_{i,j} p(x_i, x_{j+τ}) * log[p(x_i, x_{j+τ}) / (p(x_i) * p(x_{j+τ}))]

    Args:
        x: 1D signal
        max_lag: maximum lag to compute
        n_bins: histogram bins for probability estimation

    Returns:
        ami: (max_lag,) AMI values for lags 1..max_lag
    """
    ami = np.zeros(max_lag)
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-12)

    # Marginal distribution
    hist_x, _ = np.histogram(x_norm, bins=n_bins, density=True)
    hist_x = hist_x / hist_x.sum()
    hist_x = np.clip(hist_x, 1e-12, None)

    for lag in range(1, max_lag + 1):
        x1 = x_norm[:-lag]
        x2 = x_norm[lag:]
        # Joint histogram
        joint, _, _ = np.histogram2d(x1, x2, bins=n_bins)
        joint = joint / joint.sum()
        joint = np.clip(joint, 1e-12, None)

        # Marginals from joint
        marg_x = joint.sum(axis=1)
        marg_y = joint.sum(axis=0)
        marg_x = np.clip(marg_x, 1e-12, None)
        marg_y = np.clip(marg_y, 1e-12, None)

        # AMI = sum p(x,y) log[p(x,y) / (p(x)*p(y))]
        mi = np.sum(joint * np.log(joint / (np.outer(marg_x, marg_y) + 1e-12) + 1e-12))
        ami[lag - 1] = max(0, mi)

    return ami


def select_delay_mutual_info(x: np.ndarray, max_lag: int = 100) -> int:
    """
    Select time delay τ as the first local minimum of AMI.
    Falls back to τ = 10 if no minimum is found.
    """
    ami = compute_mutual_information(x, max_lag=max_lag)
    # Find first local minimum
    for i in range(1, len(ami) - 1):
        if ami[i] < ami[i - 1] and ami[i] < ami[i + 1]:
            return i + 1  # lag is 1-indexed
    logger.debug("No AMI minimum found, using default τ=10")
    return 10


def select_delay_autocorr(x: np.ndarray) -> int:
    """
    Alternative: select τ as the first zero-crossing of autocorrelation.
    """
    acf = np.correlate(x - x.mean(), x - x.mean(), mode='full')
    acf = acf[len(acf) // 2:]
    acf /= acf[0] + 1e-12
    for i in range(1, len(acf)):
        if acf[i] <= 0:
            return i
    return 10


def compute_fnn_ratio(x: np.ndarray, tau: int, dim: int, threshold: float = 15.0) -> float:
    """
    Compute False Nearest Neighbor (FNN) ratio for a given dimension.
    FNN measures whether adding an extra dimension resolves "false" neighbors
    caused by projection.
    """
    n = len(x) - (dim + 1) * tau
    if n <= 2:
        return 0.0

    # Build embedded vectors
    embeds_d = np.array([x[i:i + dim * tau:tau] for i in range(n + tau)])
    embeds_d1 = np.array([x[i:i + (dim + 1) * tau:tau] for i in range(n)])

    n_false = 0
    for i in range(n):
        # Find nearest neighbor in dimension d
        dists_d = np.linalg.norm(embeds_d[:n] - embeds_d[i], axis=1)
        dists_d[i] = np.inf
        nn_idx = np.argmin(dists_d)
        r_d = dists_d[nn_idx]

        if r_d < 1e-10:
            continue

        # Check if neighbor is false in d+1
        r_d1 = np.abs(embeds_d1[i, -1] - embeds_d1[nn_idx, -1])
        if r_d1 / (r_d + 1e-12) > threshold:
            n_false += 1

    return n_false / (n + 1e-12)


def select_embedding_dimension_fnn(
    x: np.ndarray,
    tau: int,
    max_dim: int = 10,
    threshold: float = 0.05,
) -> int:
    """
    Select embedding dimension m as the first dimension where FNN ratio < threshold.
    Falls back to 5 if not found.
    """
    for dim in range(1, max_dim + 1):
        fnn = compute_fnn_ratio(x, tau, dim)
        if fnn < threshold:
            return dim + 1
    logger.debug("FNN did not converge, using default m=5")
    return 5


def delay_embed(
    x: np.ndarray,
    embedding_dim: int,
    time_delay: int,
) -> np.ndarray:
    """
    Perform delay embedding (Takens' theorem).

    X_t = [x_t, x_{t-τ}, x_{t-2τ}, ..., x_{t-(m-1)τ}]

    Args:
        x: 1D signal of length N
        embedding_dim: m (embedding dimension)
        time_delay: τ (time delay in samples)

    Returns:
        X: (n_vectors, embedding_dim) delay-embedded phase space
    """
    n = len(x)
    n_vectors = n - (embedding_dim - 1) * time_delay

    if n_vectors <= 0:
        raise ValueError(
            f"Signal too short for embedding: len={n}, "
            f"m={embedding_dim}, τ={time_delay}. "
            f"Need at least {(embedding_dim - 1) * time_delay + 1} samples."
        )

    X = np.zeros((n_vectors, embedding_dim), dtype=np.float32)
    for i in range(embedding_dim):
        start = i * time_delay
        end = start + n_vectors
        X[:, i] = x[start:end]

    return X


def delay_embed_multichannel(
    data: np.ndarray,
    embedding_dim: int,
    time_delay: int,
) -> np.ndarray:
    """
    Apply delay embedding to each channel independently, then concatenate.

    Args:
        data: (n_channels, n_times)
        embedding_dim: m
        time_delay: τ

    Returns:
        X: (n_vectors, n_channels * embedding_dim) multi-channel phase space
    """
    n_channels = data.shape[0]
    embeddings = []

    for ch in range(n_channels):
        emb = delay_embed(data[ch], embedding_dim, time_delay)
        embeddings.append(emb)

    # Align lengths (take minimum)
    min_len = min(e.shape[0] for e in embeddings)
    embeddings = [e[:min_len] for e in embeddings]

    return np.concatenate(embeddings, axis=1)  # (n_vectors, n_ch * m)


class StateSpaceReconstructor:
    """
    Reconstruct the phase space of EEG signals using Takens' theorem.
    Automatically selects τ and m if not provided.
    """

    def __init__(
        self,
        embedding_dim: Optional[int] = None,
        time_delay: Optional[int] = None,
        method_delay: str = "mutual_info",
        method_dim: str = "fnn",
        max_dim: int = 10,
        fnn_threshold: float = 0.05,
    ):
        self.embedding_dim = embedding_dim
        self.time_delay = time_delay
        self.method_delay = method_delay
        self.method_dim = method_dim
        self.max_dim = max_dim
        self.fnn_threshold = fnn_threshold

    def fit(self, x: np.ndarray) -> "StateSpaceReconstructor":
        """
        Estimate τ and m from a representative signal.
        Args:
            x: 1D EEG signal
        """
        if self.time_delay is None:
            if self.method_delay == "mutual_info":
                self.time_delay = select_delay_mutual_info(x)
            else:
                self.time_delay = select_delay_autocorr(x)
            logger.debug(f"Selected time delay τ = {self.time_delay}")

        if self.embedding_dim is None:
            self.embedding_dim = select_embedding_dimension_fnn(
                x, self.time_delay, self.max_dim, self.fnn_threshold
            )
            logger.debug(f"Selected embedding dimension m = {self.embedding_dim}")

        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Apply delay embedding to EEG segment.
        Args:
            data: (n_channels, n_times) EEG segment

        Returns:
            X: (n_vectors, n_channels * embedding_dim) phase space
        """
        if self.time_delay is None or self.embedding_dim is None:
            # Use first channel to estimate parameters
            self.fit(data[0])

        return delay_embed_multichannel(data, self.embedding_dim, self.time_delay)

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform in one call."""
        self.fit(data[0])
        return self.transform(data)
