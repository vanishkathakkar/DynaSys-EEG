"""
Neural Dynamical System Model for DynaSys-EEG (Phase 5 of the paper).

Models the evolution of EEG latent states as:
    X_{t+1} = f_θ(X_t) + ε_t

This is a neural network function approximator trained with MSE reconstruction loss:
    L = ||X_{t+1} - f_θ(X_t)||²

The model is trained on delay-embedded EEG trajectories to capture
the underlying nonlinear dynamical system governing brain activity.

Architecture: Residual MLP with Tanh activations (smooth for dynamical systems)
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


# =============================================================================
# Neural Dynamics Network
# =============================================================================

class ResidualBlock(nn.Module):
    """Residual block with skip connection for stable dynamics learning."""

    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.net(x))


class DynamicsNet(nn.Module):
    """
    Neural approximator for the dynamical system function f_θ.

    Maps from current state X_t → predicted next state X_{t+1}
    where states are delay-embedded EEG windows.

    Architecture:
        Input: X_t ∈ ℝ^(embedding_dim × n_channels) [flattened]
        → Linear encoder → Residual blocks → Linear decoder
        Output: X_{t+1} ∈ ℝ^(embedding_dim × n_channels)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = (128, 256, 128),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.Tanh(),
        )

        # Hidden residual blocks
        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.LayerNorm(hidden_dims[i + 1]))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(dropout))
        self.hidden = nn.Sequential(*layers)

        # Residual block in the bottleneck
        self.residual = ResidualBlock(hidden_dims[-1], dropout=dropout)

        # Decoder
        self.decoder = nn.Linear(hidden_dims[-1], input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim) - flattened delay-embedded state
        Returns:
            x_next: (batch, input_dim) - predicted next state
        """
        h = self.encoder(x)
        h = self.hidden(h)
        h = self.residual(h)
        return self.decoder(h)


# =============================================================================
# Stochastic SDE Model (Extended formulation)
# =============================================================================

class StochasticDynamicsNet(nn.Module):
    """
    Stochastic Dynamical System:
        dX_t = f(X_t, t)dt + g(X_t, t)dW_t

    Implemented as:
        X_{t+1} = f_θ(X_t) + g_φ(X_t) * ε,  ε ~ N(0, I)

    Both f_θ (drift) and g_φ (diffusion) are learned networks.
    The diffusion coefficient g_φ is used as a system descriptor (D).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = (128, 256, 128),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.Tanh(),
            nn.Dropout(dropout),
        )

        # Drift network f_θ
        self.drift_net = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], hidden_dims[-1]),
            nn.Tanh(),
            nn.Linear(hidden_dims[-1], input_dim),
        )

        # Diffusion network g_φ (output > 0, use Softplus)
        self.diffusion_net = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], hidden_dims[-1]),
            nn.Tanh(),
            nn.Linear(hidden_dims[-1], input_dim),
            nn.Softplus(),  # ensures positive diffusion
        )

    def forward(
        self, x: torch.Tensor, add_noise: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, input_dim)
            add_noise: whether to include stochastic term
        Returns:
            x_next: (batch, input_dim) - predicted next state
            diffusion: (batch, input_dim) - diffusion coefficient (used as descriptor D)
        """
        h = self.encoder(x)
        drift = self.drift_net(h)
        diffusion = self.diffusion_net(h)

        x_next = x + drift
        if add_noise and self.training:
            noise = torch.randn_like(x)
            x_next = x_next + diffusion * noise

        return x_next, diffusion

    def get_diffusion(self, x: torch.Tensor) -> torch.Tensor:
        """Get diffusion coefficient (system descriptor D) for a state."""
        h = self.encoder(x)
        return self.diffusion_net(h)


# =============================================================================
# Trainer
# =============================================================================

class DynamicsTrainer:
    """
    Trains the neural dynamical system on delay-embedded EEG sequences.

    Training objective (from paper, Eq. 22):
        L = ||X_{t+1} - f_θ(X_t)||²

    The model learns how brain states evolve over time (system dynamics)
    rather than performing classification directly.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100
        )
        self.train_losses = []
        self.val_losses = []

    def _make_pairs(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create (X_t, X_{t+1}) training pairs from state sequence."""
        return states[:-1], states[1:]

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            if len(batch) == 1:
                x_t = batch[0].to(self.device)
                x_next = batch[0].to(self.device)
            else:
                x_t, x_next = batch[0].to(self.device), batch[1].to(self.device)

            self.optimizer.zero_grad()

            if isinstance(self.model, StochasticDynamicsNet):
                x_pred, _ = self.model(x_t, add_noise=True)
            else:
                x_pred = self.model(x_t)

            loss = F.mse_loss(x_pred, x_next)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def val_epoch(self, dataloader: DataLoader) -> float:
        """Compute validation loss."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            x_t, x_next = batch[0].to(self.device), batch[1].to(self.device)
            if isinstance(self.model, StochasticDynamicsNet):
                x_pred, _ = self.model(x_t, add_noise=False)
            else:
                x_pred = self.model(x_t)
            loss = F.mse_loss(x_pred, x_next)
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def train(
        self,
        states: np.ndarray,          # (n_samples, state_dim)
        n_epochs: int = 100,
        batch_size: int = 64,
        val_split: float = 0.1,
        patience: int = 15,
        verbose: bool = True,
    ) -> None:
        """
        Train the dynamics model on state sequences.
        Args:
            states: (n_samples, state_dim) array of delay-embedded states
        """
        # Build (X_t, X_{t+1}) pairs
        x_t = torch.tensor(states[:-1], dtype=torch.float32)
        x_next = torch.tensor(states[1:], dtype=torch.float32)

        # Train/val split
        n_total = len(x_t)
        n_val = max(1, int(n_total * val_split))
        n_train = n_total - n_val

        perm = torch.randperm(n_total)
        train_idx, val_idx = perm[:n_train], perm[n_train:]

        train_ds = TensorDataset(x_t[train_idx], x_next[train_idx])
        val_ds = TensorDataset(x_t[val_idx], x_next[val_idx])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        best_val = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(1, n_epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.val_epoch(val_loader)
            self.scheduler.step()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                patience_counter = 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1

            if verbose and epoch % 20 == 0:
                logger.info(
                    f"Epoch {epoch:4d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
                )

            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        logger.info(f"Training complete. Best val loss: {best_val:.6f}")

    @torch.no_grad()
    def predict_trajectory(
        self, initial_state: np.ndarray, n_steps: int = 100
    ) -> np.ndarray:
        """
        Simulate a trajectory from an initial state.
        Args:
            initial_state: (state_dim,) initial state
            n_steps: number of prediction steps
        Returns:
            trajectory: (n_steps + 1, state_dim)
        """
        self.model.eval()
        x = torch.tensor(initial_state, dtype=torch.float32).unsqueeze(0).to(self.device)
        traj = [x.cpu().numpy()]

        for _ in range(n_steps):
            if isinstance(self.model, StochasticDynamicsNet):
                x, _ = self.model(x, add_noise=False)
            else:
                x = self.model(x)
            traj.append(x.cpu().numpy())

        return np.concatenate(traj, axis=0)  # (n_steps+1, state_dim)
