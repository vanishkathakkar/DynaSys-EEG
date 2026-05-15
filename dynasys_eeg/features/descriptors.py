"""
System Descriptor Computation (DynaSys-EEG, Phases 6-7).

Computes the 5-dimensional descriptor vector:
    z = [λ, H, D, E, T]

Where:
  λ - Lyapunov Exponent: dynamical stability (Eq. 3, 13)
      λ = lim_{t→∞} (1/t) * log(||δx(t)|| / ||δx(0)||)

  H - Sample Entropy: complexity/unpredictability (Eq. discussed in Sec. XI.E)
      High H → more random; Low H → more regular

  D - Diffusion Coefficient: stochastic variability (Sec. XI.F)
      D = Var[ΔX] / Δt (second moment of state increments)

  E - Energy Landscape: global state organization (Eq. 4, 14)
      E(x) = -log P(x) via KDE

  T - Transition Density: state evolution patterns (Sec. XI.H)
      Discrete state transition frequency matrix

All descriptors operate on delay-embedded phase space trajectories.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import entropy as sp_entropy
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


# =============================================================================
# 1. Lyapunov Exponent (λ) — Dynamical Stability
# =============================================================================

def compute_lyapunov_rosenstein(
    x: np.ndarray,
    embedding_dim: int = 5,
    time_delay: int = 10,
    n_neighbors: int = 1,
    max_iter: int = 100,
    sfreq: float = 500.0,
) -> float:
    """
    Estimate the largest Lyapunov exponent using Rosenstein's algorithm.

    The algorithm tracks how initially close trajectories diverge:
        λ = lim_{t→∞} (1/t) * log(||δx(t)|| / ||δx(0)||)

    Args:
        x: 1D delay-embedded signal or raw 1D signal
        embedding_dim: m for delay embedding
        time_delay: τ for delay embedding
        n_neighbors: number of nearest neighbors to track
        max_iter: maximum iterations to track divergence
        sfreq: sampling frequency (for normalization)

    Returns:
        lambda_max: largest Lyapunov exponent
    """
    # Build phase space
    n = len(x)
    n_vectors = n - (embedding_dim - 1) * time_delay

    if n_vectors < 20:
        return 0.0

    X = np.zeros((n_vectors, embedding_dim))
    for i in range(embedding_dim):
        X[:, i] = x[i * time_delay: i * time_delay + n_vectors]

    # For each trajectory point, find nearest neighbor (excluding temporal neighbors)
    min_temporal_sep = int(0.1 * sfreq)  # 100ms minimum separation
    divergences = []

    n_ref = min(n_vectors // 4, 50)   # limit for speed
    ref_indices = np.linspace(0, n_vectors - max_iter - 1, n_ref, dtype=int)

    for ref_idx in ref_indices:
        distances = np.linalg.norm(X - X[ref_idx], axis=1)

        # Exclude temporal neighbors
        mask = np.abs(np.arange(n_vectors) - ref_idx) < min_temporal_sep
        distances[mask] = np.inf

        nn_idx = np.argmin(distances)
        if distances[nn_idx] == np.inf:
            continue

        # Track divergence over time
        local_div = []
        for step in range(1, min(max_iter, n_vectors - max(ref_idx, nn_idx) - 1)):
            d0 = np.linalg.norm(X[ref_idx] - X[nn_idx]) + 1e-12
            d_t = np.linalg.norm(X[ref_idx + step] - X[nn_idx + step]) + 1e-12
            local_div.append(np.log(d_t / d0))

        if local_div:
            divergences.append(local_div)

    if not divergences:
        return 0.0

    # Average divergence at each time step
    min_len = min(len(d) for d in divergences)
    div_array = np.array([d[:min_len] for d in divergences])
    avg_div = div_array.mean(axis=0)

    # Linear fit: slope = λ
    t = np.arange(min_len) / sfreq
    if len(t) < 2:
        return 0.0

    slope = np.polyfit(t, avg_div, 1)[0]
    return float(slope)


def compute_lyapunov_wolf(
    x: np.ndarray,
    embedding_dim: int = 5,
    time_delay: int = 10,
    n_steps: int = 100,
) -> float:
    """
    Simplified Wolf algorithm for Lyapunov exponent estimation.
    """
    n = len(x)
    n_vectors = n - (embedding_dim - 1) * time_delay
    if n_vectors < 20:
        return 0.0

    X = np.zeros((n_vectors, embedding_dim))
    for i in range(embedding_dim):
        X[:, i] = x[i * time_delay: i * time_delay + n_vectors]

    total_log_div = 0.0
    n_steps_done = 0

    ref_idx = 0
    # Find initial nearest neighbor
    dists = np.linalg.norm(X - X[ref_idx], axis=1)
    dists[ref_idx] = np.inf
    nn_idx = np.argmin(dists)
    d0 = dists[nn_idx] + 1e-12

    for step in range(1, min(n_steps, n_vectors - ref_idx - 1)):
        if ref_idx + step >= n_vectors or nn_idx + step >= n_vectors:
            break
        d_t = np.linalg.norm(X[ref_idx + step] - X[nn_idx + step]) + 1e-12
        total_log_div += np.log(d_t / d0)
        d0 = d_t
        n_steps_done += 1

    if n_steps_done == 0:
        return 0.0

    return total_log_div / n_steps_done


# =============================================================================
# 2. Sample Entropy (H) — Complexity
# =============================================================================

def compute_sample_entropy(
    x: np.ndarray,
    m: int = 2,
    r: Optional[float] = None,
    normalize: bool = True,
) -> float:
    """
    Compute Sample Entropy (SampEn).

    SampEn measures the irregularity of a time series — the negative logarithm
    of the conditional probability that sequences that match for m points also
    match for m+1 points.

    SampEn = -log(A/B)
    where:
        B = number of template matches of length m
        A = number of template matches of length m+1

    Args:
        x: 1D time series
        m: template length (typically 2)
        r: tolerance (typically 0.2 * std); computed automatically if None
        normalize: whether to normalize x first

    Returns:
        sampEn: sample entropy value
    """
    if normalize:
        std = np.std(x)
        if std < 1e-10:
            return 0.0
        x = (x - np.mean(x)) / std

    if r is None:
        r = 0.2 * np.std(x)

    N = len(x)
    if N < 2 * (m + 1):
        return 0.0

    # Build template matrices
    def _template_matches(x, m, r):
        templates = np.array([x[i:i + m] for i in range(N - m)])
        count = 0
        for i in range(N - m - 1):
            dists = np.max(np.abs(templates - templates[i]), axis=1)
            # Exclude self-match
            dists[i] = np.inf
            count += np.sum(dists <= r)
        return count

    B = _template_matches(x, m, r)
    A = _template_matches(x, m + 1, r)

    if B == 0:
        return 0.0
    if A == 0:
        return float(np.log(B))

    return -np.log(A / B)


def compute_approximate_entropy(
    x: np.ndarray,
    m: int = 2,
    r: Optional[float] = None,
) -> float:
    """
    Compute Approximate Entropy (ApEn) as an alternative to SampEn.
    ApEn is slightly less accurate but faster.
    """
    N = len(x)
    if r is None:
        r = 0.2 * np.std(x)
    if N < 2 * (m + 1) or np.std(x) < 1e-10:
        return 0.0

    def _phi(m_val):
        templates = np.array([x[i:i + m_val] for i in range(N - m_val + 1)])
        count = np.zeros(N - m_val + 1)
        for i in range(N - m_val + 1):
            dists = np.max(np.abs(templates - templates[i]), axis=1)
            count[i] = np.sum(dists <= r) / (N - m_val + 1)
        return np.sum(np.log(count + 1e-12)) / (N - m_val + 1)

    return abs(_phi(m) - _phi(m + 1))


def compute_permutation_entropy(
    x: np.ndarray,
    m: int = 3,
    tau: int = 1,
    normalize: bool = True,
) -> float:
    """
    Compute Permutation Entropy (PermEn).
    Quantifies the complexity of a time series based on ordinal patterns.
    """
    N = len(x)
    if N < m + tau:
        return 0.0

    from itertools import permutations
    from math import factorial

    # Build embedded sequence
    n_vecs = N - (m - 1) * tau
    X = np.array([x[i:i + m * tau:tau] for i in range(n_vecs)])

    # Rank patterns
    patterns = np.argsort(X, axis=1)

    # Map each pattern to unique integer
    all_perms = {p: i for i, p in enumerate(permutations(range(m)))}
    hist = np.zeros(factorial(m))

    for row in patterns:
        key = tuple(row)
        if key in all_perms:
            hist[all_perms[key]] += 1

    # Shannon entropy of pattern distribution
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    h = -np.sum(hist * np.log(hist))

    if normalize:
        h /= np.log(factorial(m))
    return float(h)


# =============================================================================
# 3. Diffusion Coefficient (D) — Stochastic Variability
# =============================================================================

def compute_diffusion_coefficient(
    states: np.ndarray,
    dt: float = 1.0,
    method: str = "variance",
) -> float:
    """
    Estimate the diffusion coefficient D = g(x, t) from the SDE:
        dX_t = f(X_t, t)dt + g(X_t, t)dW_t

    Methods:
        'variance': D = Var[ΔX] / (2 * dt)
                     (Einstein's relation for diffusion)
        'quadratic': D = E[(X_{t+1} - X_t)²] / (2 * dt)

    Args:
        states: (n_vectors, state_dim) phase space trajectory
        dt: time step
        method: estimation method

    Returns:
        D: scalar diffusion coefficient
    """
    if states.shape[0] < 2:
        return 0.0

    # State increments ΔX = X_{t+1} - X_t
    dx = np.diff(states, axis=0)  # (n-1, state_dim)

    if method == "variance":
        D = np.var(dx) / (2 * dt)
    elif method == "quadratic":
        D = np.mean(dx ** 2) / (2 * dt)
    else:
        D = np.var(dx) / (2 * dt)

    return float(D)


# =============================================================================
# 4. Energy Landscape (E) — Global State Organization
# =============================================================================

def compute_energy_landscape(
    states: np.ndarray,
    method: str = "kde",
    n_bins: int = 50,
    bandwidth: float = 0.5,
) -> float:
    """
    Compute the energy landscape scalar descriptor.

    E(x) = -log P(x)  (Eq. 4, 14 in paper)

    The energy landscape summarizes how states are distributed:
        - Healthy: deep energy wells (concentrated, stable states)
        - AD: shallow, fragmented energy landscape
        - FTD: altered distribution

    We return: mean energy + variance of energy (two aspects)
    → Then average into a single scalar for the descriptor.

    Args:
        states: (n_vectors, state_dim) trajectory
        method: 'kde' or 'histogram'
        n_bins: bins for histogram method
        bandwidth: KDE bandwidth

    Returns:
        E: scalar energy descriptor (higher = more fragmented/less organized)
    """
    if states.shape[0] < 2:
        return 0.0

    # Project to 1D for tractability (use first PC or first dim)
    x_1d = states[:, 0]  # first dimension of phase space

    if method == "histogram":
        hist, bin_edges = np.histogram(x_1d, bins=n_bins, density=True)
        hist = np.clip(hist, 1e-12, None)
        # Energy E = -log P(x)
        energies = -np.log(hist)
        # Normalize
        p = hist / hist.sum()
        # Mean energy weighted by probability
        E_mean = np.sum(p * energies)
        E_var = np.sum(p * (energies - E_mean) ** 2)
        return float(E_mean + 0.1 * E_var)

    else:  # KDE approximation
        from scipy.stats import gaussian_kde
        try:
            kde = gaussian_kde(x_1d, bw_method=bandwidth)
            # Evaluate on a grid
            x_grid = np.linspace(x_1d.min(), x_1d.max(), n_bins)
            prob = kde(x_grid)
            prob = np.clip(prob, 1e-12, None)
            energies = -np.log(prob)
            # Mean energy
            p = prob / prob.sum()
            E_mean = np.sum(p * energies)
            return float(E_mean)
        except Exception:
            return 0.0


# =============================================================================
# 5. Transition Density (T) — State Evolution
# =============================================================================

def compute_transition_density(
    states: np.ndarray,
    n_discrete_states: int = 10,
    method: str = "kmeans",
) -> float:
    """
    Estimate transition density from phase space trajectory.

    Captures how frequently the system moves between states:
        T = H(transition matrix)  [entropy of transitions]

    Steps:
        1. Discretize phase space into K states (K-means clustering)
        2. Build transition matrix P[i→j]
        3. Return entropy of the transition distribution

    Higher T → more irregular transitions (AD-like)
    Lower T → more regular/predictable transitions (HC-like)

    Args:
        states: (n_vectors, state_dim) trajectory
        n_discrete_states: K for discretization
        method: 'kmeans' or 'quantile'

    Returns:
        T: transition entropy scalar
    """
    if states.shape[0] < n_discrete_states + 1:
        return 0.0

    try:
        # Discretize state space
        if method == "kmeans":
            from sklearn.cluster import MiniBatchKMeans
            n_states = min(n_discrete_states, states.shape[0] // 2)
            kmeans = MiniBatchKMeans(
                n_clusters=n_states, n_init=3, random_state=42, max_iter=50
            )
            labels = kmeans.fit_predict(states)
        else:
            # Quantile-based discretization using first dimension
            x_1d = states[:, 0]
            quantiles = np.percentile(x_1d, np.linspace(0, 100, n_discrete_states + 1))
            labels = np.digitize(x_1d, quantiles[1:-1])
            n_states = n_discrete_states

        # Build transition matrix
        n_states = labels.max() + 1
        trans_matrix = np.zeros((n_states, n_states), dtype=float)
        for t in range(len(labels) - 1):
            trans_matrix[labels[t], labels[t + 1]] += 1

        # Normalize rows
        row_sums = trans_matrix.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        trans_matrix /= row_sums

        # Entropy of transitions
        T = 0.0
        for row in trans_matrix:
            row = row[row > 1e-12]
            if len(row) > 0:
                T += -np.sum(row * np.log(row))

        return float(T / n_states)

    except Exception as e:
        logger.debug(f"Transition density error: {e}")
        return 0.0


# =============================================================================
# Main Descriptor Extractor
# =============================================================================

class DynamicalDescriptorExtractor:
    """
    Compute the complete descriptor vector z = [λ, H, D, E, T]
    for each EEG segment's phase space trajectory.

    This is Phase 6-7 of the DynaSys-EEG pipeline.
    """

    def __init__(
        self,
        embedding_dim: int = 5,
        time_delay: int = 10,
        sfreq: float = 500.0,
        lyapunov_method: str = "rosenstein",
        entropy_method: str = "sample",
        n_bins: int = 50,
        n_discrete_states: int = 10,
    ):
        self.embedding_dim = embedding_dim
        self.time_delay = time_delay
        self.sfreq = sfreq
        self.lyapunov_method = lyapunov_method
        self.entropy_method = entropy_method
        self.n_bins = n_bins
        self.n_discrete_states = n_discrete_states

    def extract(
        self,
        states: np.ndarray,
        raw_signal: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Extract descriptor vector from phase space trajectory.

        Args:
            states: (n_vectors, state_dim) delay-embedded trajectory
            raw_signal: original 1D EEG signal (optional, for entropy)

        Returns:
            z: (5,) descriptor vector [λ, H, D, E, T]
        """
        # Use first dimension for 1D descriptors
        x_1d = states[:, 0] if states.ndim > 1 else states

        # 1. Lyapunov exponent (λ)
        if self.lyapunov_method == "rosenstein":
            lam = compute_lyapunov_rosenstein(
                x_1d,
                embedding_dim=self.embedding_dim,
                time_delay=self.time_delay,
                sfreq=self.sfreq,
            )
        else:
            lam = compute_lyapunov_wolf(
                x_1d,
                embedding_dim=self.embedding_dim,
                time_delay=self.time_delay,
            )

        # 2. Entropy (H) — use fast permutation entropy by default
        src = raw_signal if raw_signal is not None else x_1d
        if self.entropy_method == "sample":
            # For real data with long signals, use permutation entropy (fast)
            if len(src) > 1000:
                H = compute_permutation_entropy(src[:1000], m=4)
            else:
                H = compute_sample_entropy(src, m=2)
        elif self.entropy_method == "permutation":
            H = compute_permutation_entropy(src)
        else:
            H = compute_approximate_entropy(src)

        # 3. Diffusion coefficient (D)
        D = compute_diffusion_coefficient(states, dt=1.0 / self.sfreq)

        # 4. Energy (E)
        E = compute_energy_landscape(states, n_bins=self.n_bins)

        # 5. Transition density (T)
        T = compute_transition_density(states, n_discrete_states=self.n_discrete_states)

        z = np.array([lam, H, D, E, T], dtype=np.float32)
        return z

    def extract_batch(
        self,
        segments: np.ndarray,
        state_space_reconstructor=None,
        max_segments: int = 30,
    ) -> np.ndarray:
        """
        Extract descriptors for a batch of EEG segments.

        Args:
            segments: (n_segments, n_channels, n_times)
            state_space_reconstructor: StateSpaceReconstructor instance
            max_segments: max segments to process per subject (for speed)

        Returns:
            Z: (n_used_segments, 5) descriptor matrix
        """
        from .state_space import StateSpaceReconstructor, delay_embed

        if state_space_reconstructor is None:
            state_space_reconstructor = StateSpaceReconstructor(
                embedding_dim=self.embedding_dim,
                time_delay=self.time_delay,
            )
            # Fit on first segment
            state_space_reconstructor.fit(segments[0, 0])

        n_segments = segments.shape[0]

        # For speed: evenly sample up to max_segments from the recording
        if n_segments > max_segments:
            indices = np.linspace(0, n_segments - 1, max_segments, dtype=int)
        else:
            indices = np.arange(n_segments)

        Z = np.zeros((len(indices), 5), dtype=np.float32)

        for out_i, i in enumerate(indices):
            seg = segments[i]  # (n_channels, n_times)

            # Get states (use first channel for fitting)
            try:
                states = state_space_reconstructor.transform(seg)
            except Exception as e:
                logger.debug(f"Embedding failed for segment {i}: {e}")
                continue

            # Extract descriptors
            z = self.extract(states, raw_signal=seg[0])
            Z[out_i] = z

        return Z
