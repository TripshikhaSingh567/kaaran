"""
KAARAN Model Architectures and Ensemble System.
- MLP Members: 2 hidden layers of 64 units, tanh, standardized inputs.
- Outputs Gaussian mean and log-sigma for (delta_T, delta_L).
- Gaussian NLL loss, Adam optimizer, bootstrap resampling per member.
- Ensembles: Forecast (input s), Action (input s, a), SCM (input s, a, probe data).
- Baselines: Persistence, Scikit-learn Linear Regression.
"""

import os
import sys
import json
from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class GaussianMLPMember(nn.Module):
    """
    Single ensemble member predicting Gaussian mean and log-sigma of state delta.
    input -> Linear(in_dim, 64) -> Tanh -> Linear(64, 64) -> Tanh -> Linear(64, 4)
    """

    def __init__(self, in_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 4),  # [mu_dT, mu_dL, log_sigma_dT, log_sigma_dL]
        )

        # Normalization buffers
        self.register_buffer("in_mean", torch.zeros(in_dim))
        self.register_buffer("in_std", torch.ones(in_dim))

    def set_normalization(self, mean: np.ndarray, std: np.ndarray):
        std = np.where(std < 1e-6, 1.0, std)
        self.in_mean.copy_(torch.from_numpy(mean.astype(np.float32)))
        self.in_std.copy_(torch.from_numpy(std.astype(np.float32)))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Standardize input
        x_norm = (x - self.in_mean) / self.in_std
        out = self.net(x_norm)
        mu = out[..., :2]
        # Clamp log_sigma between -4.0 and 2.0 for numerical safety
        log_sigma = torch.clamp(out[..., 2:], -4.0, 2.0)
        return mu, log_sigma

    @staticmethod
    def gaussian_nll_loss(y: torch.Tensor, mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
        """
        Gaussian NLL: 0.5 * (log_var + (y - mu)^2 / var) + const
        = log_sigma + 0.5 * (y - mu)^2 / exp(2 * log_sigma)
        """
        var = torch.exp(2.0 * log_sigma)
        nll = log_sigma + 0.5 * ((y - mu) ** 2) / (var + 1e-8)
        return torch.mean(torch.sum(nll, dim=-1))


class EnsembleModel:
    """
    Deep Ensemble of 5 GaussianMLPMember models.
    Total predictive mean: mean of member means.
    Total predictive variance: mean of member variances + variance of member means.
    """

    def __init__(self, in_dim: int, num_members: int = 5, hidden_dim: int = 64):
        self.in_dim = in_dim
        self.num_members = num_members
        self.hidden_dim = hidden_dim
        self.members = [GaussianMLPMember(in_dim, hidden_dim) for _ in range(num_members)]

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 40,
        batch_size: int = 64,
        lr: float = 1e-3,
        seed: int = 42,
    ):
        N = len(X)
        in_mean = np.mean(X, axis=0)
        in_std = np.std(X, axis=0)

        for i, member in enumerate(self.members):
            member.set_normalization(in_mean, in_std)
            member.train()
            optimizer = optim.Adam(member.parameters(), lr=lr, weight_decay=1e-5)

            # Bootstrap resampling
            member_rng = np.random.default_rng(seed + i * 1000)
            boot_idx = member_rng.choice(N, size=N, replace=True)
            X_boot = torch.from_numpy(X[boot_idx].astype(np.float32))
            y_boot = torch.from_numpy(y[boot_idx].astype(np.float32))

            dataset = torch.utils.data.TensorDataset(X_boot, y_boot)
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

            for epoch in range(epochs):
                for b_x, b_y in loader:
                    optimizer.zero_grad()
                    mu, log_sigma = member(b_x)
                    loss = GaussianMLPMember.gaussian_nll_loss(b_y, mu, log_sigma)
                    loss.backward()
                    optimizer.step()

            member.eval()

    def predict_one_step(
        self,
        x: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Compute ensemble predictive distribution:
        - mu_ensemble: mean across members
        - sigma_ensemble: sqrt(mean(sigma_m^2) + var(mu_m))
        - member_means: list of individual member means
        - member_sigmas: list of individual member sigmas
        """
        x_tensor = torch.from_numpy(np.atleast_2d(x).astype(np.float32))
        member_means = []
        member_vars = []
        member_sigmas = []

        with torch.no_grad():
            for member in self.members:
                mu, log_sigma = member(x_tensor)
                m_mu = mu.cpu().numpy()
                m_sigma = np.exp(log_sigma.cpu().numpy())
                member_means.append(m_mu)
                member_vars.append(m_sigma ** 2)
                member_sigmas.append(m_sigma)

        # Shape: (M, B, 2)
        means_arr = np.array(member_means)
        vars_arr = np.array(member_vars)

        # Ensemble total mean and variance
        ens_mean = np.mean(means_arr, axis=0)  # (B, 2)
        # Total variance = mean of variances (aleatoric) + variance of means (epistemic)
        aleatoric_var = np.mean(vars_arr, axis=0)
        epistemic_var = np.var(means_arr, axis=0)
        total_var = aleatoric_var + epistemic_var
        total_std = np.sqrt(total_var)

        is_1d = (x.ndim == 1)
        return {
            "mean": ens_mean[0] if is_1d else ens_mean,
            "std": total_std[0] if is_1d else total_std,
            "aleatoric_std": np.sqrt(aleatoric_var)[0] if is_1d else np.sqrt(aleatoric_var),
            "epistemic_std": np.sqrt(epistemic_var)[0] if is_1d else np.sqrt(epistemic_var),
            "member_means": [m[0] if is_1d else m for m in member_means],
            "member_sigmas": [s[0] if is_1d else s for s in member_sigmas],
        }

    def rollout_monte_carlo(
        self,
        s0: Tuple[float, float],
        actions: List[Tuple[float, float]],
        num_paths: int = 500,
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Monte Carlo multi-step path rollouts (500 paths):
        For each path: sample a member m ~ Uniform(1..M), roll forward step-by-step
        sampling delta ~ N(mu_m, sigma_m).
        """
        if rng is None:
            rng = np.random.default_rng(42)

        H = len(actions)
        paths_T = np.zeros((num_paths, H + 1))
        paths_L = np.zeros((num_paths, H + 1))
        paths_T[:, 0] = s0[0]
        paths_L[:, 0] = s0[1]

        # Sample ensemble member for each path
        chosen_members = rng.integers(0, self.num_members, size=num_paths)

        with torch.no_grad():
            for t in range(H):
                v_t, h_t = actions[t]
                current_T = paths_T[:, t]
                current_L = paths_L[:, t]

                # If model expects (s, a) vs s
                if self.in_dim == 4:
                    X_step = np.column_stack([current_T, current_L, np.full(num_paths, v_t), np.full(num_paths, h_t)])
                else:
                    X_step = np.column_stack([current_T, current_L])

                X_tensor = torch.from_numpy(X_step.astype(np.float32))

                # Step forward per chosen member
                for m_idx in range(self.num_members):
                    mask = (chosen_members == m_idx)
                    if not np.any(mask):
                        continue
                    m_mu, m_log_sig = self.members[m_idx](X_tensor[mask])
                    mu_np = m_mu.cpu().numpy()
                    sig_np = np.exp(m_log_sig.cpu().numpy())

                    eps = rng.normal(0.0, 1.0, size=mu_np.shape)
                    delta = mu_np + sig_np * eps

                    paths_T[mask, t + 1] = np.clip(current_T[mask] + delta[:, 0], 0.0, 120.0)
                    paths_L[mask, t + 1] = np.clip(current_L[mask] + delta[:, 1], 0.0, 100.0)

        # Compute median, 5th percentile, 95th percentile (90% band)
        return {
            "median_T": np.median(paths_T, axis=0),
            "lower_T": np.percentile(paths_T, 5.0, axis=0),
            "upper_T": np.percentile(paths_T, 95.0, axis=0),
            "median_L": np.median(paths_L, axis=0),
            "lower_L": np.percentile(paths_L, 5.0, axis=0),
            "upper_L": np.percentile(paths_L, 95.0, axis=0),
            "all_paths_T": paths_T,
            "all_paths_L": paths_L,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Export full ensemble architecture, normalizers, and weights to a JSON-serializable dictionary."""
        members_data = []
        for member in self.members:
            state = member.state_dict()
            members_data.append({
                "in_mean": member.in_mean.cpu().numpy().tolist(),
                "in_std": member.in_std.cpu().numpy().tolist(),
                "fc1_weight": state["net.0.weight"].cpu().numpy().tolist(),
                "fc1_bias": state["net.0.bias"].cpu().numpy().tolist(),
                "fc2_weight": state["net.2.weight"].cpu().numpy().tolist(),
                "fc2_bias": state["net.2.bias"].cpu().numpy().tolist(),
                "out_weight": state["net.4.weight"].cpu().numpy().tolist(),
                "out_bias": state["net.4.bias"].cpu().numpy().tolist(),
            })

        return {
            "mock": False,
            "in_dim": self.in_dim,
            "num_members": self.num_members,
            "hidden_dim": self.hidden_dim,
            "members": members_data,
        }

    def load_from_dict(self, data: Dict[str, Any]):
        self.in_dim = data["in_dim"]
        self.num_members = data["num_members"]
        self.hidden_dim = data["hidden_dim"]
        self.members = [GaussianMLPMember(self.in_dim, self.hidden_dim) for _ in range(self.num_members)]

        for i, member_data in enumerate(data["members"]):
            member = self.members[i]
            member.set_normalization(
                np.array(member_data["in_mean"]),
                np.array(member_data["in_std"]),
            )
            state_dict = {
                "in_mean": torch.tensor(member_data["in_mean"], dtype=torch.float32),
                "in_std": torch.tensor(member_data["in_std"], dtype=torch.float32),
                "net.0.weight": torch.tensor(member_data["fc1_weight"], dtype=torch.float32),
                "net.0.bias": torch.tensor(member_data["fc1_bias"], dtype=torch.float32),
                "net.2.weight": torch.tensor(member_data["fc2_weight"], dtype=torch.float32),
                "net.2.bias": torch.tensor(member_data["fc2_bias"], dtype=torch.float32),
                "net.4.weight": torch.tensor(member_data["out_weight"], dtype=torch.float32),
                "net.4.bias": torch.tensor(member_data["out_bias"], dtype=torch.float32),
            }
            member.load_state_dict(state_dict)
            member.eval()


class LinearBaselineModel:
    """
    Ordinary least squares linear regression on operator logs with constant empirical noise:
    predicts delta_T and delta_L given [T, L, v, h].
    """

    def __init__(self):
        self.model_T = LinearRegression()
        self.model_L = LinearRegression()
        self.sigma_T = 1.0
        self.sigma_L = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        # y is [delta_T, delta_L]
        self.model_T.fit(X, y[:, 0])
        self.model_L.fit(X, y[:, 1])

        pred_T = self.model_T.predict(X)
        pred_L = self.model_L.predict(X)

        self.sigma_T = float(np.std(y[:, 0] - pred_T))
        self.sigma_L = float(np.std(y[:, 1] - pred_L))

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X_arr = np.atleast_2d(X)
        pred_T = self.model_T.predict(X_arr)
        pred_L = self.model_L.predict(X_arr)
        mean = np.column_stack([pred_T, pred_L])
        std = np.array([self.sigma_T, self.sigma_L])
        return mean, std

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mock": False,
            "coef_T": self.model_T.coef_.tolist(),
            "intercept_T": float(self.model_T.intercept_),
            "sigma_T": float(self.sigma_T),
            "coef_L": self.model_L.coef_.tolist(),
            "intercept_L": float(self.model_L.intercept_),
            "sigma_L": float(self.sigma_L),
        }
