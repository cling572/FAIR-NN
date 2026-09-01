"""Simulation code for DWES Section 5.

This module implements ten methods:
KRR, HDES, FNR, WFNR, RFNR, RWFNR, FAST-FNR, FAST-WFNR,
FAST-RFNR, and FAST-RWFNR.
The FAST variants learn ``Gamma`` in the first-stage quantile model. FAST-RWFNR
also refines ``Gamma`` in the weighted quantile step before the final ES
regressions. The data-generating designs follow
``ESALL/DWES_simulation_revised.tex``:

1. nonlinear factor model,
2. linear factor model,
3. nonlinear non-factor robustness design,
4. linear non-factor robustness design,
5. linear factor model with sparse idiosyncratic heterogeneity,
6. nonlinear factor model with sparse idiosyncratic heterogeneity.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from numpy.linalg import eigh
from scipy.stats import gamma as gamma_dist
from scipy.stats import norm, t
from torch.utils.data import DataLoader, TensorDataset

try:
    from quantes.nonlinear import KRR
except ImportError as exc:  # pragma: no cover - dependency is external.
    KRR = None
    _KRR_IMPORT_ERROR = exc
else:
    _KRR_IMPORT_ERROR = None


P_VALUES = (500, 1000, 1500, 2000)
TAU_VALUES = (0.05, 0.10, 0.20)
ACTIVE_HETERO_COUNT = 8


@dataclass(frozen=True)
class SimulationConfig:
    design: str
    factor_model: bool
    heterogeneity: str = "none"
    innovation: str = "normal"
    nonfactor_distribution: str = "copula-beta"
    nonfactor_case: str = "current"
    n_train: int = 1000
    n_test: int = 1000
    p: int = 500
    r: int = 8
    n_unlabeled: int = 100
    r_bar: int = 16
    tau: float = 0.20
    epochs: int = 100
    repeats: int = 20
    batch_size: int = 64
    learning_rate: float = 1e-3
    throughput_dim: int = 8
    truncation: float = 3.0
    penalty_eps: float = 5e-3
    quantile_loss: str = "pinball"
    quantile_huber_kappa: float = 0.1
    heterogeneity_distribution: str = "normal-sparse"
    heterogeneity_location_strength: float = 0.5
    heterogeneity_scale_strength: float = 0.25
    krr_kernel: str = "auto"
    lambda_q: float | None = None
    lambda_refine: float | None = None
    hdes_lambda_q: float | None = None
    hdes_lambda_e: float | None = None
    hdes_learning_rate: float | None = None
    hdes_epochs: int | None = None
    c_h: float | None = None
    c_h_quantile: float = 0.05
    c_h_scale: float = 0.5
    bootstrap_B: int = 0
    ci_level: float = 0.95
    save_bootstrap_ci: bool = False
    bootstrap_ci_output_dir: str = "bootstrap_ci"
    seed: int = 2026
    device: str = "cpu"


@dataclass
class SimulationData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    q_test: np.ndarray
    es_test: np.ndarray
    x_unlabeled: np.ndarray


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def safe_exp(x: np.ndarray) -> np.ndarray:
    return np.exp(np.clip(x, -30.0, 30.0))


def safe_reciprocal(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    safe_x = np.where(np.abs(x) < eps, np.where(x < 0, -eps, eps), x)
    return 1.0 / safe_x


def nonlinear_location(z: np.ndarray) -> np.ndarray:
    cubic_base = 1 + z[:, 3] + z[:, 4]
    last_base = z[:, 5] + safe_exp(z[:, 6] * z[:, 7])
    return (
        np.cos(2 * np.pi * z[:, 0])
        + (1 + safe_exp(-z[:, 1] - z[:, 2])) ** -1
        + safe_reciprocal(cubic_base) ** 3
        + safe_reciprocal(last_base)
    )


def nonlinear_scale_index(z: np.ndarray) -> np.ndarray:
    return (
        np.sin(0.5 * np.pi * (z[:, 0] + z[:, 1]))
        + np.log1p((z[:, 2] * z[:, 3] * z[:, 4]) ** 2)
        + z[:, 7] * (1 + safe_exp(-z[:, 5] - z[:, 6])) ** -1
    )


def linear_location(z: np.ndarray) -> np.ndarray:
    return 2 * np.sum(z[:, :4], axis=1) + np.sum(z[:, 4:8], axis=1)


def linear_scale_index(z: np.ndarray) -> np.ndarray:
    return 0.5 * np.sum(z[:, :8], axis=1)


def linear_heterogeneous_location(f: np.ndarray, u: np.ndarray, strength: float) -> np.ndarray:
    beta_u = np.array([1.4, -1.2, 1.0, -0.9, 0.8, -0.7, 0.6, -0.5])
    hetero_location = u[:, :ACTIVE_HETERO_COUNT] @ beta_u
    return linear_location(f) + strength * hetero_location


def linear_heterogeneous_scale_index(f: np.ndarray, u: np.ndarray, strength: float) -> np.ndarray:
    u_active = u[:, :ACTIVE_HETERO_COUNT]
    hetero_scale = (
        0.70 * u_active[:, 0]
        - 0.55 * u_active[:, 1]
        + 0.45 * np.abs(u_active[:, 2])
        + 0.35 * u_active[:, 3]
        - 0.30 * u_active[:, 4]
        + 0.25 * u_active[:, 5] ** 2
        + 0.20 * u_active[:, 6]
        - 0.15 * u_active[:, 7]
    )
    return linear_scale_index(f) + strength * hetero_scale


def nonlinear_heterogeneous_location(f: np.ndarray, u: np.ndarray, strength: float) -> np.ndarray:
    u_active = u[:, :ACTIVE_HETERO_COUNT]
    hetero_location = (
        1.20 * np.sin(np.pi * u_active[:, 0])
        + 1.00 * u_active[:, 1] ** 2
        - 0.90 * u_active[:, 2] * u_active[:, 3]
        + 0.80 * (u_active[:, 4] > 0.0).astype(float)
        + 0.70 * np.cos(np.pi * u_active[:, 5])
        + 0.60 * u_active[:, 6] * (1.0 - u_active[:, 6])
        - 0.50 * np.sin(2.0 * np.pi * u_active[:, 7])
    )
    return nonlinear_location(f) + strength * hetero_location


def nonlinear_heterogeneous_scale_index(f: np.ndarray, u: np.ndarray, strength: float) -> np.ndarray:
    u_active = u[:, :ACTIVE_HETERO_COUNT]
    hetero_scale = (
        0.70 * np.sin(np.pi * u_active[:, 0] * u_active[:, 1])
        + 0.55 * u_active[:, 2] ** 2
        - 0.45 * u_active[:, 3]
        + 0.35 * np.abs(u_active[:, 4])
        + 0.30 * u_active[:, 5] * u_active[:, 6]
        - 0.25 * np.cos(np.pi * u_active[:, 7])
    )
    return nonlinear_scale_index(f) + strength * hetero_scale


def positive_scale(raw_scale: np.ndarray) -> np.ndarray:
    """Use a smooth positive shift for sigma in the location-scale model."""
    return np.logaddexp(0.0, raw_scale) + 0.10


def location_and_scale(z: np.ndarray, design: str) -> tuple[np.ndarray, np.ndarray]:
    if design == "nonlinear":
        location = nonlinear_location(z)
        raw_scale = nonlinear_scale_index(z)
    elif design == "linear":
        location = linear_location(z)
        raw_scale = linear_scale_index(z)
    else:
        raise ValueError(f"Unknown design: {design}")
    return location, positive_scale(raw_scale)


def nonfactor_example3a_location_and_scale(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    beta_location = np.zeros(x.shape[1])
    beta_location[:4] = 1.0
    beta_location[4:8] = 2.0

    beta_scale = np.zeros(x.shape[1])
    beta_scale[:4] = 1.0 / 3.0

    location = x @ beta_location
    raw_scale = x @ beta_scale
    return location, positive_scale(raw_scale)


def nonfactor_example3b_location_and_scale(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    location = 2.0 * (x[:, 0] ** 2 + x[:, 1] ** 2)
    scale = safe_exp(np.mean(x[:, :ACTIVE_HETERO_COUNT], axis=1))
    return location, scale


def nonfactor_location_and_scale(x: np.ndarray, design: str, nonfactor_case: str) -> tuple[np.ndarray, np.ndarray]:
    if nonfactor_case == "current":
        return location_and_scale(x, design)
    if nonfactor_case == "3a":
        return nonfactor_example3a_location_and_scale(x)
    if nonfactor_case == "3b":
        return nonfactor_example3b_location_and_scale(x)
    raise ValueError(f"Unknown nonfactor case: {nonfactor_case}")


def heterogeneous_location_and_scale(
    f: np.ndarray,
    u: np.ndarray,
    heterogeneity: str,
    location_strength: float,
    scale_strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    if heterogeneity == "linear":
        location = linear_heterogeneous_location(f, u, location_strength)
        raw_scale = linear_heterogeneous_scale_index(f, u, scale_strength)
    elif heterogeneity == "nonlinear":
        location = nonlinear_heterogeneous_location(f, u, location_strength)
        raw_scale = nonlinear_heterogeneous_scale_index(f, u, scale_strength)
    else:
        raise ValueError(f"Unknown heterogeneity design: {heterogeneity}")
    return location, positive_scale(raw_scale)


def innovation_quantile_es(tau: float, innovation: str) -> tuple[float, float]:
    if innovation == "normal":
        q = norm.ppf(tau)
        es = norm.expect(lambda x: x * (x <= q)) / tau
    elif innovation == "t3":
        df = 3
        q = t.ppf(tau, df)
        es = -((df + q**2) / (df - 1)) * t.pdf(q, df) / tau
    elif innovation == "gamma":
        shape = 3.0
        scale = 1.0
        gamma_q = gamma_dist.ppf(tau, a=shape, scale=scale)
        q = gamma_q - shape * scale
        es = shape * scale * gamma_dist.cdf(gamma_q, a=shape + 1.0, scale=scale) / tau
        es -= shape * scale
    else:
        raise ValueError(f"Unknown innovation: {innovation}")
    return float(q), float(es)


def draw_innovation(size: int, innovation: str) -> np.ndarray:
    if innovation == "normal":
        return np.random.normal(size=size)
    if innovation == "t3":
        return t.rvs(df=3, size=size)
    if innovation == "gamma":
        return np.random.gamma(shape=3.0, scale=1.0, size=size) - 3.0
    raise ValueError(f"Unknown innovation: {innovation}")


def ar_covariance(p: int, rho: float = 0.6) -> np.ndarray:
    idx = np.arange(p)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def draw_normal_ar(n: int, p: int, rho: float = 0.6) -> np.ndarray:
    eps = np.random.normal(size=(n, p))
    x = np.empty_like(eps)
    x[:, 0] = eps[:, 0]
    innovation_scale = np.sqrt(1.0 - rho**2)
    for j in range(1, p):
        x[:, j] = rho * x[:, j - 1] + innovation_scale * eps[:, j]
    return x


def sparse_idiosyncratic_covariance(p: int, strength: float = 0.3, probability: float = 0.05) -> np.ndarray:
    upper = np.triu(np.random.binomial(1, probability, size=(p, p)), k=1)
    sparse_part = strength * (upper + upper.T)
    sigma = np.eye(p) + sparse_part

    min_eig = np.linalg.eigvalsh(sigma)[0]
    if min_eig <= 1e-8:
        sigma = sigma + (abs(min_eig) + 1e-6) * np.eye(p)

    std = np.sqrt(np.diag(sigma))
    sigma = sigma / np.outer(std, std)
    np.fill_diagonal(sigma, 1.0)
    return sigma


def draw_multivariate_t(n: int, covariance: np.ndarray, df: int = 3) -> np.ndarray:
    gaussian = np.random.multivariate_normal(
        mean=np.zeros(covariance.shape[0]),
        cov=covariance,
        size=n,
    )
    scale = np.sqrt(np.random.chisquare(df, size=(n, 1)) / df)
    return gaussian / scale


def draw_idiosyncratic(n: int, p: int, distribution: str) -> np.ndarray:
    return draw_idiosyncratic_with_covariance(n, p, distribution)[0]


def draw_idiosyncratic_with_covariance(
    n: int,
    p: int,
    distribution: str,
    covariance: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    if distribution == "uniform":
        return np.random.uniform(0.0, 1.0, size=(n, p)), covariance
    if distribution == "normal":
        return np.random.normal(size=(n, p)), covariance
    if distribution == "normal-ar":
        return draw_normal_ar(n, p), covariance
    if distribution == "t3":
        df = 3
        gaussian = np.random.normal(size=(n, p))
        scale = np.sqrt(np.random.chisquare(df, size=(n, 1)) / df)
        return gaussian / scale, covariance
    if distribution == "normal-sparse":
        if covariance is None:
            covariance = sparse_idiosyncratic_covariance(p)
        return np.random.multivariate_normal(mean=np.zeros(p), cov=covariance, size=n), covariance
    if distribution == "t3-sparse":
        if covariance is None:
            covariance = sparse_idiosyncratic_covariance(p)
        return draw_multivariate_t(n, covariance, df=3), covariance
    raise ValueError(f"Unknown heterogeneity distribution: {distribution}")


def factor_covariates(
    n: int,
    p: int,
    r: int,
    loading: np.ndarray | None = None,
    idiosyncratic_distribution: str = "uniform",
    idiosyncratic_covariance: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    # Keep the simulation scale consistent with the original PythonProject scripts.
    # This avoids singular nonlinear terms such as (1 + f_4 + f_5)^(-3).
    f = np.random.uniform(0.0, 1.0, size=(n, r))
    if loading is None:
        loading = np.random.uniform(-2.0, 2.0, size=(p, r))
    u, idiosyncratic_covariance = draw_idiosyncratic_with_covariance(
        n,
        p,
        idiosyncratic_distribution,
        idiosyncratic_covariance,
    )
    x = f @ loading.T + u
    return x, f, u, loading, idiosyncratic_covariance


def beta_correlation_matrix(p: int) -> np.ndarray:
    """Create a PSD correlation matrix close to the Beta(5, 2) design."""
    rho = np.random.beta(5, 2, size=(p, p))
    sigma = (rho + rho.T) / 2.0
    np.fill_diagonal(sigma, 1.0)

    min_eig = np.linalg.eigvalsh(sigma)[0]
    if min_eig <= 1e-8:
        sigma = sigma + (abs(min_eig) + 1e-6) * np.eye(p)

    std = np.sqrt(np.diag(sigma))
    sigma = sigma / np.outer(std, std)
    np.fill_diagonal(sigma, 1.0)
    return sigma


def nonfactor_covariates(n: int, p: int, distribution: str, covariance: np.ndarray | None = None) -> np.ndarray:
    if distribution == "normal-iid":
        return np.random.normal(size=(n, p))

    if distribution == "normal-ar":
        if covariance is None:
            covariance = ar_covariance(p, rho=0.8)
        return np.random.multivariate_normal(mean=np.zeros(p), cov=covariance, size=n)

    if distribution != "copula-beta":
        raise ValueError(f"Unknown nonfactor distribution: {distribution}")

    if covariance is None:
        covariance = beta_correlation_matrix(p)
    gaussian = np.random.multivariate_normal(
        mean=np.zeros(p),
        cov=covariance,
        size=n,
    )
    return norm.cdf(gaussian)


def generate_data(config: SimulationConfig) -> SimulationData:
    q_eta, es_eta = innovation_quantile_es(config.tau, config.innovation)

    if config.factor_model:
        idiosyncratic_distribution = config.heterogeneity_distribution
        x_train, f_train, u_train, loading, idiosyncratic_covariance = factor_covariates(
            config.n_train,
            config.p,
            config.r,
            idiosyncratic_distribution=idiosyncratic_distribution,
        )
        x_test, f_test, u_test, _, _ = factor_covariates(
            config.n_test,
            config.p,
            config.r,
            loading=loading,
            idiosyncratic_distribution=idiosyncratic_distribution,
            idiosyncratic_covariance=idiosyncratic_covariance,
        )
        x_unlabeled, _, _, _, _ = factor_covariates(
            config.n_unlabeled,
            config.p,
            config.r,
            loading=loading,
            idiosyncratic_distribution=idiosyncratic_distribution,
            idiosyncratic_covariance=idiosyncratic_covariance,
        )

        if config.heterogeneity == "none":
            train_location, train_scale = location_and_scale(f_train, config.design)
            test_location, test_scale = location_and_scale(f_test, config.design)
        else:
            train_location, train_scale = heterogeneous_location_and_scale(
                f_train,
                u_train,
                config.heterogeneity,
                config.heterogeneity_location_strength,
                config.heterogeneity_scale_strength,
            )
            test_location, test_scale = heterogeneous_location_and_scale(
                f_test,
                u_test,
                config.heterogeneity,
                config.heterogeneity_location_strength,
                config.heterogeneity_scale_strength,
            )
    else:
        covariance = None
        if config.nonfactor_distribution == "normal-ar":
            covariance = ar_covariance(config.p, rho=0.8)
        elif config.nonfactor_distribution == "copula-beta":
            covariance = beta_correlation_matrix(config.p)

        x_train = nonfactor_covariates(
            config.n_train,
            config.p,
            config.nonfactor_distribution,
            covariance,
        )
        x_test = nonfactor_covariates(
            config.n_test,
            config.p,
            config.nonfactor_distribution,
            covariance,
        )
        x_unlabeled = nonfactor_covariates(
            config.n_unlabeled,
            config.p,
            config.nonfactor_distribution,
            covariance,
        )

        train_location, train_scale = nonfactor_location_and_scale(
            x_train,
            config.design,
            config.nonfactor_case,
        )
        test_location, test_scale = nonfactor_location_and_scale(
            x_test,
            config.design,
            config.nonfactor_case,
        )

    y_train = train_location + train_scale * draw_innovation(config.n_train, config.innovation)
    q_test = test_location + test_scale * q_eta
    es_test = test_location + test_scale * es_eta

    return SimulationData(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        q_test=q_test,
        es_test=es_test,
        x_unlabeled=x_unlabeled,
    )


def factor_projection(x_train: np.ndarray, x_test: np.ndarray, x_unlabeled: np.ndarray, r_bar: int) -> tuple[np.ndarray, np.ndarray]:
    p = x_train.shape[1]
    sigma_hat = (x_unlabeled.T @ x_unlabeled) / x_unlabeled.shape[0]
    _, eigvecs = eigh(sigma_hat)
    w = np.sqrt(p) * eigvecs[:, -r_bar:]
    z_train = (w.T @ x_train.T).T / p
    z_test = (w.T @ x_test.T).T / p
    return z_train, z_test


class FeedForwardNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, hidden_layers: int = 3):
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = input_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VarianceNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, hidden_layers: int = 3):
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = input_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        layers.append(nn.Softplus())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LinearLassoModel(nn.Module):
    """Linear model with an unpenalized intercept and L1-penalized slopes."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def l1_penalty(self) -> torch.Tensor:
        return self.linear.weight.abs().sum()


class FASTInputLayer(nn.Module):
    """Build [f_tilde, clip(X Gamma)] with trainable sparse throughput."""

    def __init__(self, p: int, r_bar: int, throughput_dim: int, truncation: float):
        super().__init__()
        self.throughput_dim = throughput_dim
        self.truncation = truncation
        if throughput_dim > 0:
            self.gamma = nn.Parameter(torch.zeros(p, throughput_dim))
            nn.init.normal_(self.gamma, mean=0.0, std=1e-4)
        else:
            self.register_parameter("gamma", None)

    def forward(self, f_tilde: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if self.throughput_dim == 0 or self.gamma is None:
            return f_tilde
        throughput = torch.clamp(x @ self.gamma, -self.truncation, self.truncation)
        return torch.cat([f_tilde, throughput], dim=1)


class FASTNetwork(nn.Module):
    """First-stage network with a trainable FAST sparse-throughput input layer."""

    def __init__(
        self,
        p: int,
        r_bar: int,
        throughput_dim: int,
        truncation: float,
        variance_output: bool = False,
    ):
        super().__init__()
        self.fast_input = FASTInputLayer(p, r_bar, throughput_dim, truncation)
        input_dim = r_bar + throughput_dim
        self.net = VarianceNN(input_dim) if variance_output else FeedForwardNN(input_dim)

    @property
    def gamma(self) -> torch.Tensor | None:
        return self.fast_input.gamma

    def forward(self, f_tilde: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.fast_input(f_tilde, x))


def quantile_loss(pred: torch.Tensor, target: torch.Tensor, tau: float) -> torch.Tensor:
    error = target - pred
    return torch.mean(torch.maximum(tau * error, (tau - 1.0) * error))


def huber_quantile_loss(pred: torch.Tensor, target: torch.Tensor, tau: float, kappa: float) -> torch.Tensor:
    error = target - pred
    abs_error = torch.abs(error)
    huber = torch.where(abs_error <= kappa, 0.5 * error.pow(2) / kappa, abs_error - 0.5 * kappa)
    weight = torch.where(error >= 0, tau, 1.0 - tau)
    return torch.mean(weight * huber)


def weighted_quantile_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    obs_weight: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    error = target - pred
    pinball = torch.maximum(tau * error, (tau - 1.0) * error)
    return torch.mean(obs_weight * pinball)


def weighted_huber_quantile_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    obs_weight: torch.Tensor,
    tau: float,
    kappa: float,
) -> torch.Tensor:
    error = target - pred
    abs_error = torch.abs(error)
    huber = torch.where(abs_error <= kappa, 0.5 * error.pow(2) / kappa, abs_error - 0.5 * kappa)
    quantile_weight = torch.where(error >= 0, tau, 1.0 - tau)
    return torch.mean(obs_weight * quantile_weight * huber)


def first_stage_quantile_loss(pred: torch.Tensor, target: torch.Tensor, config: SimulationConfig) -> torch.Tensor:
    if config.quantile_loss == "pinball":
        return quantile_loss(pred, target, config.tau)
    if config.quantile_loss == "huber":
        return huber_quantile_loss(pred, target, config.tau, config.quantile_huber_kappa)
    raise ValueError(f"Unknown quantile loss: {config.quantile_loss}")


def weighted_first_stage_quantile_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    obs_weight: torch.Tensor,
    config: SimulationConfig,
) -> torch.Tensor:
    if config.quantile_loss == "pinball":
        return weighted_quantile_loss(pred, target, obs_weight, config.tau)
    if config.quantile_loss == "huber":
        return weighted_huber_quantile_loss(
            pred,
            target,
            obs_weight,
            config.tau,
            config.quantile_huber_kappa,
        )
    raise ValueError(f"Unknown quantile loss: {config.quantile_loss}")


def clipped_l1_penalty(gamma: torch.Tensor | None, eps: float) -> torch.Tensor:
    if gamma is None:
        return torch.tensor(0.0)
    return torch.clamp(torch.abs(gamma) / eps, max=1.0).sum()


def fast_lambda(config: SimulationConfig, lambda_value: float | None) -> float:
    if lambda_value is not None:
        return lambda_value
    return 1.3 * np.log(config.p) / config.n_train


def gamma_penalty(model: FASTNetwork, config: SimulationConfig, lambda_value: float | None) -> torch.Tensor:
    penalty = clipped_l1_penalty(model.gamma, config.penalty_eps)
    return fast_lambda(config, lambda_value) * penalty.to(next(model.parameters()).device)


def hdes_lambda(config: SimulationConfig, lambda_value: float | None) -> float:
    if lambda_value is not None:
        return lambda_value
    return float(np.sqrt(np.log(config.p) / config.n_train))


def resolve_c_h(config: SimulationConfig, variance_estimates: torch.Tensor) -> float:
    """Choose c_h, using the paper's low-quantile rule when not specified."""
    if config.c_h is not None:
        return float(config.c_h)

    alpha = min(max(float(config.c_h_quantile), 0.0), 1.0)
    kappa = min(max(float(config.c_h_scale), 0.0), 1.0)
    empirical_quantile = torch.quantile(variance_estimates.detach().float().view(-1), alpha).item()
    return max(kappa * float(empirical_quantile), 1e-8)


def shared_fast_features(
    model: FASTNetwork,
    f_tilde: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """Return the fixed FAST input learned by the first-stage quantile model."""
    model.eval()
    with torch.no_grad():
        return model.fast_input(f_tilde, x).detach()


def es_target(q_pred: np.ndarray, y: np.ndarray, tau: float) -> np.ndarray:
    indicator = (y <= q_pred).astype(float)
    return (y - q_pred) * indicator + tau * q_pred


def standardize_with_train(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (x_train - mean) / std, (x_test - mean) / std


def train_model(
    model: nn.Module,
    loader: DataLoader,
    loss_fn,
    epochs: int,
    learning_rate: float,
) -> None:
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for _ in range(epochs):
        for batch in loader:
            optimizer.zero_grad()
            loss = loss_fn(*batch)
            loss.backward()
            optimizer.step()


def ci_from_deviations(point: np.ndarray, deviations: np.ndarray, ci_level: float) -> tuple[np.ndarray, np.ndarray]:
    alpha = 1.0 - ci_level
    lower_shift = np.quantile(deviations, 1.0 - alpha / 2.0, axis=0)
    upper_shift = np.quantile(deviations, alpha / 2.0, axis=0)
    return point - lower_shift, point - upper_shift


def draw_exp_multipliers(n: int, device: torch.device) -> torch.Tensor:
    return torch.empty((n, 1), dtype=torch.float32, device=device).exponential_(1.0)


def bootstrap_ci_summary(prefix: str, lower: np.ndarray, upper: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    return {
        f"{prefix}_ci_lower_mean": float(np.mean(lower)),
        f"{prefix}_ci_upper_mean": float(np.mean(upper)),
        f"{prefix}_ci_width_mean": float(np.mean(upper - lower)),
        f"{prefix}_ci_coverage": float(np.mean((truth >= lower) & (truth <= upper))),
    }


def fast_rwfnr_pointwise_ci_records(
    config: SimulationConfig,
    repeat: int,
    q_point: np.ndarray,
    q_truth: np.ndarray,
    q_lower: np.ndarray,
    q_upper: np.ndarray,
    es_point: np.ndarray,
    es_truth: np.ndarray,
    es_lower: np.ndarray,
    es_upper: np.ndarray,
) -> list[dict[str, float | int | str]]:
    return [
        {
            "method": "FAST-RWFNR",
            "p": config.p,
            "tau": config.tau,
            "repeat": repeat,
            "test_index": int(idx),
            "q_point": float(q_point[idx]),
            "q_truth": float(q_truth[idx]),
            "q_ci_lower": float(q_lower[idx]),
            "q_ci_upper": float(q_upper[idx]),
            "q_ci_width": float(q_upper[idx] - q_lower[idx]),
            "q_covered": int(q_lower[idx] <= q_truth[idx] <= q_upper[idx]),
            "es_point": float(es_point[idx]),
            "es_truth": float(es_truth[idx]),
            "es_ci_lower": float(es_lower[idx]),
            "es_ci_upper": float(es_upper[idx]),
            "es_ci_width": float(es_upper[idx] - es_lower[idx]),
            "es_covered": int(es_lower[idx] <= es_truth[idx] <= es_upper[idx]),
            "ci_level": float(config.ci_level),
            "bootstrap_B": int(config.bootstrap_B),
        }
        for idx in range(q_point.shape[0])
    ]


def write_bootstrap_ci_records(records: list[dict[str, float | int | str]], output_path: Path) -> None:
    if not records:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def bootstrap_refined_quantile_ci(
    shared_train: torch.Tensor,
    shared_test: torch.Tensor,
    y_tensor: torch.Tensor,
    weights: torch.Tensor,
    q_test_refined: np.ndarray,
    q_test_true: np.ndarray,
    config: SimulationConfig,
    repeat: int | None = None,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    if config.bootstrap_B <= 0:
        return {}, np.array([]), np.array([])
    device = torch.device(config.device)
    n_train = y_tensor.shape[0]
    deviations: list[np.ndarray] = []
    torch.manual_seed(config.seed + 7919 + (0 if repeat is None else repeat))
    for _ in range(config.bootstrap_B):
        xi = draw_exp_multipliers(n_train, device)
        boot_weights = weights * xi
        loader = DataLoader(
            TensorDataset(shared_train, y_tensor, boot_weights),
            batch_size=config.batch_size,
            shuffle=True,
        )
        boot_q = FeedForwardNN(shared_train.shape[1]).to(device)
        train_model(
            boot_q,
            loader,
            lambda z_fast, y, w: weighted_first_stage_quantile_loss(boot_q(z_fast), y, w, config),
            config.epochs,
            config.learning_rate,
        )
        boot_q.eval()
        with torch.no_grad():
            q_star = boot_q(shared_test).cpu().numpy().ravel()
        deviations.append(q_star - q_test_refined)
    dev = np.vstack(deviations)
    q_lower, q_upper = ci_from_deviations(q_test_refined, dev, config.ci_level)
    summary = bootstrap_ci_summary("q", q_lower, q_upper, q_test_true)
    return summary, q_lower, q_upper


def bootstrap_final_es_ci(
    shared_train: torch.Tensor,
    shared_test: torch.Tensor,
    z_refined_tensor: torch.Tensor,
    weights: torch.Tensor,
    es_test_refined: np.ndarray,
    es_test_true: np.ndarray,
    config: SimulationConfig,
    repeat: int | None = None,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    if config.bootstrap_B <= 0:
        return {}, np.array([]), np.array([])
    device = torch.device(config.device)
    n_train = z_refined_tensor.shape[0]
    deviations: list[np.ndarray] = []
    torch.manual_seed(config.seed + 15443 + (0 if repeat is None else repeat))
    for _ in range(config.bootstrap_B):
        xi = draw_exp_multipliers(n_train, device)
        boot_weights = weights * xi
        loader = DataLoader(
            TensorDataset(shared_train, z_refined_tensor, boot_weights),
            batch_size=config.batch_size,
            shuffle=True,
        )
        boot_es = FeedForwardNN(shared_train.shape[1]).to(device)
        train_model(
            boot_es,
            loader,
            lambda z_fast, z, w: torch.mean(w * (config.tau * boot_es(z_fast) - z) ** 2),
            config.epochs,
            config.learning_rate,
        )
        boot_es.eval()
        with torch.no_grad():
            es_star = boot_es(shared_test).cpu().numpy().ravel()
        deviations.append(es_star - es_test_refined)
    dev = np.vstack(deviations)
    es_lower, es_upper = ci_from_deviations(es_test_refined, dev, config.ci_level)
    summary = bootstrap_ci_summary("es", es_lower, es_upper, es_test_true)
    return summary, es_lower, es_upper


def krr_kernel_trials(krr_kernel: str, p: int) -> tuple[tuple[str, dict[str, float]], ...]:
    kernels: dict[str, tuple[str, dict[str, float]]] = {
        "rbf": ("rbf", {"gamma": 1.0 / p}),
        "gaussian": ("gaussian", {"gamma": 1.0 / p}),
        "polynomial": ("polynomial", {"degree": 3, "gamma": 1.0, "coef0": 1.0}),
    }
    if krr_kernel == "auto":
        return (kernels["rbf"], kernels["gaussian"], kernels["polynomial"])
    if krr_kernel in kernels:
        return (kernels[krr_kernel],)
    raise ValueError(f"Unknown KRR kernel: {krr_kernel}")


def run_krr(data: SimulationData, tau: float, krr_kernel: str) -> dict[str, float | str]:
    if KRR is None:
        raise ImportError("quantes is required for KRR") from _KRR_IMPORT_ERROR

    last_error: Exception | None = None
    kernel_trials = krr_kernel_trials(krr_kernel, data.x_train.shape[1])

    for kernel, kernel_params in kernel_trials:
        try:
            model = KRR(data.x_train, data.y_train, kernel=kernel, kernel_params=kernel_params)
            model.qt(tau=tau, alpha=0.5, solver="cvxopt")
            q_pred = model.qt_predict(data.x_test)
            model.ES(tau=tau, alpha=2.0, x=data.x_test)
            es_pred = model.pred_e
            return {
                "method": "KRR",
                "kernel": kernel,
                "mspq": float(np.mean((data.q_test - q_pred) ** 2)),
                "mspe": float(np.mean((data.es_test - es_pred) ** 2)),
            }
        except Exception as exc:  # pragma: no cover - depends on quantes backend.
            last_error = exc

    raise RuntimeError("All KRR kernel attempts failed") from last_error


def run_hdes(data: SimulationData, config: SimulationConfig) -> dict[str, float | str]:
    """L1-penalized high-dimensional ES regression of Zhang et al. (2025)."""
    x_train, x_test = standardize_with_train(data.x_train, data.x_test)
    device = torch.device(config.device)

    x_tensor = torch.tensor(x_train, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(data.y_train, dtype=torch.float32, device=device).view(-1, 1)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32, device=device)

    train_loader = DataLoader(
        TensorDataset(x_tensor, y_tensor),
        batch_size=config.batch_size,
        shuffle=True,
    )
    quantile_model = LinearLassoModel(data.x_train.shape[1]).to(device)
    quantile_lambda = hdes_lambda(config, config.hdes_lambda_q)
    hdes_epochs = config.hdes_epochs or config.epochs
    hdes_learning_rate = config.hdes_learning_rate or config.learning_rate
    train_model(
        quantile_model,
        train_loader,
        lambda x, y: first_stage_quantile_loss(quantile_model(x), y, config)
        + quantile_lambda * quantile_model.l1_penalty(),
        hdes_epochs,
        hdes_learning_rate,
    )

    with torch.no_grad():
        q_train = quantile_model(x_tensor).cpu().numpy().ravel()
        q_test = quantile_model(x_test_tensor).cpu().numpy().ravel()

    z_target = es_target(q_train, data.y_train, config.tau)
    z_tensor = torch.tensor(z_target, dtype=torch.float32, device=device).view(-1, 1)
    es_loader = DataLoader(
        TensorDataset(x_tensor, z_tensor),
        batch_size=config.batch_size,
        shuffle=True,
    )
    es_model = LinearLassoModel(data.x_train.shape[1]).to(device)
    es_lambda = hdes_lambda(config, config.hdes_lambda_e)
    train_model(
        es_model,
        es_loader,
        lambda x, z: 0.5 * torch.mean((z - config.tau * es_model(x)) ** 2)
        + config.tau * es_lambda * es_model.l1_penalty(),
        hdes_epochs,
        hdes_learning_rate,
    )

    with torch.no_grad():
        es_pred = es_model(x_test_tensor).cpu().numpy().ravel()

    return {
        "method": "HDES",
        "kernel": "",
        "mspq": float(np.mean((data.q_test - q_test) ** 2)),
        "mspe": float(np.mean((data.es_test - es_pred) ** 2)),
    }


def run_fnr(data: SimulationData, config: SimulationConfig) -> dict[str, float | str]:
    """Factor-only FNR baseline using f_tilde as the neural-network input."""
    return run_factor_neural_family(data, config, ("FNR",))["FNR"]

    z_train, z_test = factor_projection(data.x_train, data.x_test, data.x_unlabeled, config.r_bar)
    device = torch.device(config.device)

    x_tensor = torch.tensor(z_train, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(data.y_train, dtype=torch.float32, device=device).view(-1, 1)
    x_test_tensor = torch.tensor(z_test, dtype=torch.float32, device=device)

    train_loader = DataLoader(
        TensorDataset(x_tensor, y_tensor),
        batch_size=config.batch_size,
        shuffle=True,
    )
    quantile_model = FeedForwardNN(config.r_bar).to(device)
    train_model(
        quantile_model,
        train_loader,
        lambda x, y: first_stage_quantile_loss(quantile_model(x), y, config),
        config.epochs,
        config.learning_rate,
    )

    with torch.no_grad():
        q_train = quantile_model(x_tensor).cpu().numpy().ravel()
        q_test = quantile_model(x_test_tensor).cpu().numpy().ravel()

    z_target = es_target(q_train, data.y_train, config.tau)
    z_tensor = torch.tensor(z_target, dtype=torch.float32, device=device).view(-1, 1)
    es_loader = DataLoader(
        TensorDataset(x_tensor, z_tensor),
        batch_size=config.batch_size,
        shuffle=True,
    )

    es_model = FeedForwardNN(config.r_bar).to(device)
    mse = nn.MSELoss()
    train_model(
        es_model,
        es_loader,
        lambda x, z: mse(config.tau * es_model(x), z),
        config.epochs,
        config.learning_rate,
    )

    with torch.no_grad():
        es_pred = es_model(x_test_tensor).cpu().numpy().ravel()

    return {
        "method": "FNR",
        "kernel": "",
        "mspq": float(np.mean((data.q_test - q_test) ** 2)),
        "mspe": float(np.mean((data.es_test - es_pred) ** 2)),
    }


def run_fast_fnr(data: SimulationData, config: SimulationConfig) -> dict[str, float | str]:
    """FAST-FNR with Gamma learned in the quantile stage and then fixed."""
    return run_fast_neural_family(data, config, ("FAST-FNR",))[0]["FAST-FNR"]

    z_train, z_test = factor_projection(data.x_train, data.x_test, data.x_unlabeled, config.r_bar)
    x_fast_train, x_fast_test = standardize_with_train(data.x_train, data.x_test)
    device = torch.device(config.device)

    f_tensor = torch.tensor(z_train, dtype=torch.float32, device=device)
    x_tensor = torch.tensor(x_fast_train, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(data.y_train, dtype=torch.float32, device=device).view(-1, 1)
    f_test_tensor = torch.tensor(z_test, dtype=torch.float32, device=device)
    x_test_tensor = torch.tensor(x_fast_test, dtype=torch.float32, device=device)

    train_loader = DataLoader(
        TensorDataset(f_tensor, x_tensor, y_tensor),
        batch_size=config.batch_size,
        shuffle=True,
    )
    quantile_model = FASTNetwork(
        data.x_train.shape[1],
        config.r_bar,
        config.throughput_dim,
        config.truncation,
    ).to(device)
    train_model(
        quantile_model,
        train_loader,
        lambda f, x, y: first_stage_quantile_loss(quantile_model(f, x), y, config)
        + gamma_penalty(quantile_model, config, config.lambda_q),
        config.epochs,
        config.learning_rate,
    )

    with torch.no_grad():
        q_train = quantile_model(f_tensor, x_tensor).cpu().numpy().ravel()
        q_test = quantile_model(f_test_tensor, x_test_tensor).cpu().numpy().ravel()
    shared_train = shared_fast_features(quantile_model, f_tensor, x_tensor)
    shared_test = shared_fast_features(quantile_model, f_test_tensor, x_test_tensor)

    z_target = es_target(q_train, data.y_train, config.tau)
    z_tensor = torch.tensor(z_target, dtype=torch.float32, device=device).view(-1, 1)
    es_loader = DataLoader(
        TensorDataset(shared_train, z_tensor),
        batch_size=config.batch_size,
        shuffle=True,
    )

    es_model = FeedForwardNN(shared_train.shape[1]).to(device)
    mse = nn.MSELoss()
    train_model(
        es_model,
        es_loader,
        lambda z_fast, z: mse(config.tau * es_model(z_fast), z),
        config.epochs,
        config.learning_rate,
    )

    with torch.no_grad():
        es_pred = es_model(shared_test).cpu().numpy().ravel()

    return {
        "method": "FAST-FNR",
        "kernel": "",
        "mspq": float(np.mean((data.q_test - q_test) ** 2)),
        "mspe": float(np.mean((data.es_test - es_pred) ** 2)),
    }


def run_wfnr(data: SimulationData, config: SimulationConfig) -> dict[str, float | str]:
    """Factor-only WFNR baseline using f_tilde and inverse-variance weights."""
    return run_factor_neural_family(data, config, ("WFNR",))["WFNR"]

    z_train, z_test = factor_projection(data.x_train, data.x_test, data.x_unlabeled, config.r_bar)
    device = torch.device(config.device)

    x_tensor = torch.tensor(z_train, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(data.y_train, dtype=torch.float32, device=device).view(-1, 1)
    x_test_tensor = torch.tensor(z_test, dtype=torch.float32, device=device)

    train_loader = DataLoader(
        TensorDataset(x_tensor, y_tensor),
        batch_size=config.batch_size,
        shuffle=True,
    )
    quantile_model = FeedForwardNN(config.r_bar).to(device)
    train_model(
        quantile_model,
        train_loader,
        lambda x, y: first_stage_quantile_loss(quantile_model(x), y, config),
        config.epochs,
        config.learning_rate,
    )

    with torch.no_grad():
        q_train = quantile_model(x_tensor).cpu().numpy().ravel()
        q_test = quantile_model(x_test_tensor).cpu().numpy().ravel()

    z_target = es_target(q_train, data.y_train, config.tau)
    z_tensor = torch.tensor(z_target, dtype=torch.float32, device=device).view(-1, 1)
    es_loader = DataLoader(
        TensorDataset(x_tensor, z_tensor),
        batch_size=config.batch_size,
        shuffle=True,
    )

    pilot_es = FeedForwardNN(config.r_bar).to(device)
    mse = nn.MSELoss()
    train_model(
        pilot_es,
        es_loader,
        lambda x, z: mse(config.tau * pilot_es(x), z),
        config.epochs,
        config.learning_rate,
    )

    with torch.no_grad():
        squared_errors = (config.tau * pilot_es(x_tensor) - z_tensor) ** 2

    variance_loader = DataLoader(
        TensorDataset(x_tensor, squared_errors),
        batch_size=config.batch_size,
        shuffle=True,
    )
    variance_model = VarianceNN(config.r_bar).to(device)
    smooth_l1 = nn.SmoothL1Loss()
    train_model(
        variance_model,
        variance_loader,
        lambda x, err: smooth_l1(variance_model(x), err),
        config.epochs,
        config.learning_rate,
    )

    with torch.no_grad():
        variance_pred = variance_model(x_tensor)
        chosen_c_h = resolve_c_h(config, variance_pred)
        sigma2 = variance_pred.clamp(min=chosen_c_h / 2.0)
        weights = (1.0 / sigma2).view(-1, 1)
        weights = weights / weights.mean()

    weighted_loader = DataLoader(
        TensorDataset(x_tensor, z_tensor, weights),
        batch_size=config.batch_size,
        shuffle=True,
    )
    weighted_es = FeedForwardNN(config.r_bar).to(device)
    train_model(
        weighted_es,
        weighted_loader,
        lambda x, z, w: torch.mean(w * (config.tau * weighted_es(x) - z) ** 2),
        config.epochs,
        config.learning_rate,
    )

    with torch.no_grad():
        es_pred = weighted_es(x_test_tensor).cpu().numpy().ravel()

    return {
        "method": "WFNR",
        "kernel": "",
        "mspq": float(np.mean((data.q_test - q_test) ** 2)),
        "mspe": float(np.mean((data.es_test - es_pred) ** 2)),
    }


def run_fast_wfnr(data: SimulationData, config: SimulationConfig) -> dict[str, float | str]:
    """FAST-WFNR with shared sparse throughput and adaptive weights."""
    return run_fast_neural_family(data, config, ("FAST-WFNR",))[0]["FAST-WFNR"]

    z_train, z_test = factor_projection(data.x_train, data.x_test, data.x_unlabeled, config.r_bar)
    x_fast_train, x_fast_test = standardize_with_train(data.x_train, data.x_test)
    device = torch.device(config.device)

    f_tensor = torch.tensor(z_train, dtype=torch.float32, device=device)
    x_tensor = torch.tensor(x_fast_train, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(data.y_train, dtype=torch.float32, device=device).view(-1, 1)
    f_test_tensor = torch.tensor(z_test, dtype=torch.float32, device=device)
    x_test_tensor = torch.tensor(x_fast_test, dtype=torch.float32, device=device)

    train_loader = DataLoader(
        TensorDataset(f_tensor, x_tensor, y_tensor),
        batch_size=config.batch_size,
        shuffle=True,
    )
    quantile_model = FASTNetwork(
        data.x_train.shape[1],
        config.r_bar,
        config.throughput_dim,
        config.truncation,
    ).to(device)
    train_model(
        quantile_model,
        train_loader,
        lambda f, x, y: first_stage_quantile_loss(quantile_model(f, x), y, config)
        + gamma_penalty(quantile_model, config, config.lambda_q),
        config.epochs,
        config.learning_rate,
    )

    with torch.no_grad():
        q_train = quantile_model(f_tensor, x_tensor).cpu().numpy().ravel()
        q_test = quantile_model(f_test_tensor, x_test_tensor).cpu().numpy().ravel()
    shared_train = shared_fast_features(quantile_model, f_tensor, x_tensor)
    shared_test = shared_fast_features(quantile_model, f_test_tensor, x_test_tensor)

    z_target = es_target(q_train, data.y_train, config.tau)
    z_tensor = torch.tensor(z_target, dtype=torch.float32, device=device).view(-1, 1)
    es_loader = DataLoader(
        TensorDataset(shared_train, z_tensor),
        batch_size=config.batch_size,
        shuffle=True,
    )

    pilot_es = FeedForwardNN(shared_train.shape[1]).to(device)
    mse = nn.MSELoss()
    train_model(
        pilot_es,
        es_loader,
        lambda z_fast, z: mse(config.tau * pilot_es(z_fast), z),
        config.epochs,
        config.learning_rate,
    )

    with torch.no_grad():
        squared_errors = (config.tau * pilot_es(shared_train) - z_tensor) ** 2

    variance_loader = DataLoader(
        TensorDataset(shared_train, squared_errors),
        batch_size=config.batch_size,
        shuffle=True,
    )
    variance_model = VarianceNN(shared_train.shape[1]).to(device)
    smooth_l1 = nn.SmoothL1Loss()
    train_model(
        variance_model,
        variance_loader,
        lambda z_fast, err: smooth_l1(variance_model(z_fast), err),
        config.epochs,
        config.learning_rate,
    )

    with torch.no_grad():
        variance_pred = variance_model(shared_train)
        chosen_c_h = resolve_c_h(config, variance_pred)
        sigma2 = variance_pred.clamp(min=chosen_c_h / 2.0)
        weights = (1.0 / sigma2).view(-1, 1)
        weights = weights / weights.mean()

    weighted_loader = DataLoader(
        TensorDataset(shared_train, z_tensor, weights),
        batch_size=config.batch_size,
        shuffle=True,
    )
    weighted_es = FeedForwardNN(shared_train.shape[1]).to(device)
    train_model(
        weighted_es,
        weighted_loader,
        lambda z_fast, z, w: torch.mean(w * (config.tau * weighted_es(z_fast) - z) ** 2),
        config.epochs,
        config.learning_rate,
    )

    with torch.no_grad():
        es_pred = weighted_es(shared_test).cpu().numpy().ravel()

    return {
        "method": "FAST-WFNR",
        "kernel": "",
        "mspq": float(np.mean((data.q_test - q_test) ** 2)),
        "mspe": float(np.mean((data.es_test - es_pred) ** 2)),
    }


def run_fast_rwfnr(data: SimulationData, config: SimulationConfig) -> dict[str, float | str]:
    """FAST refinement with weighted quantile, FAST-RFNR pilot, and re-estimated weights."""
    return run_fast_neural_family(data, config, ("FAST-RWFNR",))[0]["FAST-RWFNR"]

    z_train, z_test = factor_projection(data.x_train, data.x_test, data.x_unlabeled, config.r_bar)
    x_fast_train, x_fast_test = standardize_with_train(data.x_train, data.x_test)
    device = torch.device(config.device)

    f_tensor = torch.tensor(z_train, dtype=torch.float32, device=device)
    x_tensor = torch.tensor(x_fast_train, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(data.y_train, dtype=torch.float32, device=device).view(-1, 1)
    f_test_tensor = torch.tensor(z_test, dtype=torch.float32, device=device)
    x_test_tensor = torch.tensor(x_fast_test, dtype=torch.float32, device=device)

    train_loader = DataLoader(
        TensorDataset(f_tensor, x_tensor, y_tensor),
        batch_size=config.batch_size,
        shuffle=True,
    )
    quantile_model = FASTNetwork(
        data.x_train.shape[1],
        config.r_bar,
        config.throughput_dim,
        config.truncation,
    ).to(device)
    train_model(
        quantile_model,
        train_loader,
        lambda f, x, y: first_stage_quantile_loss(quantile_model(f, x), y, config)
        + gamma_penalty(quantile_model, config, config.lambda_q),
        config.epochs,
        config.learning_rate,
    )

    with torch.no_grad():
        q_train_initial = quantile_model(f_tensor, x_tensor).cpu().numpy().ravel()
    shared_train = shared_fast_features(quantile_model, f_tensor, x_tensor)
    shared_test = shared_fast_features(quantile_model, f_test_tensor, x_test_tensor)

    z_initial = es_target(q_train_initial, data.y_train, config.tau)
    z_initial_tensor = torch.tensor(z_initial, dtype=torch.float32, device=device).view(-1, 1)
    es_loader = DataLoader(
        TensorDataset(shared_train, z_initial_tensor),
        batch_size=config.batch_size,
        shuffle=True,
    )

    pilot_es = FeedForwardNN(shared_train.shape[1]).to(device)
    mse = nn.MSELoss()
    train_model(
        pilot_es,
        es_loader,
        lambda z_fast, z: mse(config.tau * pilot_es(z_fast), z),
        config.epochs,
        config.learning_rate,
    )

    with torch.no_grad():
        squared_errors = (config.tau * pilot_es(shared_train) - z_initial_tensor) ** 2

    variance_loader = DataLoader(
        TensorDataset(shared_train, squared_errors),
        batch_size=config.batch_size,
        shuffle=True,
    )
    variance_model = VarianceNN(shared_train.shape[1]).to(device)
    smooth_l1 = nn.SmoothL1Loss()
    train_model(
        variance_model,
        variance_loader,
        lambda z_fast, err: smooth_l1(variance_model(z_fast), err),
        config.epochs,
        config.learning_rate,
    )

    with torch.no_grad():
        variance_pred = variance_model(shared_train)
        chosen_c_h = resolve_c_h(config, variance_pred)
        sigma2 = variance_pred.clamp(min=chosen_c_h / 2.0)
        weights = (1.0 / sigma2).view(-1, 1)
        weights = weights / weights.mean()

    refined_quantile_loader = DataLoader(
        TensorDataset(shared_train, y_tensor, weights),
        batch_size=config.batch_size,
        shuffle=True,
    )
    refined_quantile = FeedForwardNN(shared_train.shape[1]).to(device)
    train_model(
        refined_quantile,
        refined_quantile_loader,
        lambda z_fast, y, w: weighted_first_stage_quantile_loss(
            refined_quantile(z_fast),
            y,
            w,
            config,
        ),
        config.epochs,
        config.learning_rate,
    )

    with torch.no_grad():
        q_train_refined = refined_quantile(shared_train).cpu().numpy().ravel()
        q_test_refined = refined_quantile(shared_test).cpu().numpy().ravel()

    z_refined = es_target(q_train_refined, data.y_train, config.tau)
    z_refined_tensor = torch.tensor(z_refined, dtype=torch.float32, device=device).view(-1, 1)
    refined_es_loader = DataLoader(
        TensorDataset(shared_train, z_refined_tensor, weights),
        batch_size=config.batch_size,
        shuffle=True,
    )
    refined_es = FeedForwardNN(shared_train.shape[1]).to(device)
    train_model(
        refined_es,
        refined_es_loader,
        lambda z_fast, z, w: torch.mean(w * (config.tau * refined_es(z_fast) - z) ** 2),
        config.epochs,
        config.learning_rate,
    )

    with torch.no_grad():
        es_pred = refined_es(shared_test).cpu().numpy().ravel()

    return {
        "method": "FAST-RWFNR",
        "kernel": "",
        "mspq": float(np.mean((data.q_test - q_test_refined) ** 2)),
        "mspe": float(np.mean((data.es_test - es_pred) ** 2)),
    }


def run_factor_neural_family(
    data: SimulationData,
    config: SimulationConfig,
    methods: Iterable[str],
) -> dict[str, dict[str, float | str]]:
    """Run factor-only neural methods with shared and refined quantile stages."""
    requested = set(methods)
    results: dict[str, dict[str, float | str]] = {}
    if not requested:
        return results

    z_train, z_test = factor_projection(data.x_train, data.x_test, data.x_unlabeled, config.r_bar)
    device = torch.device(config.device)

    x_tensor = torch.tensor(z_train, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(data.y_train, dtype=torch.float32, device=device).view(-1, 1)
    x_test_tensor = torch.tensor(z_test, dtype=torch.float32, device=device)

    train_loader = DataLoader(
        TensorDataset(x_tensor, y_tensor),
        batch_size=config.batch_size,
        shuffle=True,
    )
    quantile_model = FeedForwardNN(config.r_bar).to(device)
    train_model(
        quantile_model,
        train_loader,
        lambda x, y: first_stage_quantile_loss(quantile_model(x), y, config),
        config.epochs,
        config.learning_rate,
    )

    with torch.no_grad():
        q_train = quantile_model(x_tensor).cpu().numpy().ravel()
        q_test = quantile_model(x_test_tensor).cpu().numpy().ravel()

    z_target = es_target(q_train, data.y_train, config.tau)
    z_tensor = torch.tensor(z_target, dtype=torch.float32, device=device).view(-1, 1)
    es_loader = DataLoader(
        TensorDataset(x_tensor, z_tensor),
        batch_size=config.batch_size,
        shuffle=True,
    )

    pilot_es = FeedForwardNN(config.r_bar).to(device)
    mse = nn.MSELoss()
    train_model(
        pilot_es,
        es_loader,
        lambda x, z: mse(config.tau * pilot_es(x), z),
        config.epochs,
        config.learning_rate,
    )

    with torch.no_grad():
        unweighted_es_pred = pilot_es(x_test_tensor).cpu().numpy().ravel()

    if "FNR" in requested:
        results["FNR"] = {
            "method": "FNR",
            "kernel": "",
            "mspq": float(np.mean((data.q_test - q_test) ** 2)),
            "mspe": float(np.mean((data.es_test - unweighted_es_pred) ** 2)),
        }

    if requested & {"WFNR", "RFNR", "RWFNR"}:
        with torch.no_grad():
            squared_errors = (config.tau * pilot_es(x_tensor) - z_tensor) ** 2

        variance_loader = DataLoader(
            TensorDataset(x_tensor, squared_errors),
            batch_size=config.batch_size,
            shuffle=True,
        )
        variance_model = VarianceNN(config.r_bar).to(device)
        smooth_l1 = nn.SmoothL1Loss()
        train_model(
            variance_model,
            variance_loader,
            lambda x, err: smooth_l1(variance_model(x), err),
            config.epochs,
            config.learning_rate,
        )

        with torch.no_grad():
            variance_pred = variance_model(x_tensor)
            chosen_c_h = resolve_c_h(config, variance_pred)
            sigma2 = variance_pred.clamp(min=chosen_c_h / 2.0)
            weights = (1.0 / sigma2).view(-1, 1)
            weights = weights / weights.mean()

        if "WFNR" in requested:
            weighted_loader = DataLoader(
                TensorDataset(x_tensor, z_tensor, weights),
                batch_size=config.batch_size,
                shuffle=True,
            )
            weighted_es = FeedForwardNN(config.r_bar).to(device)
            train_model(
                weighted_es,
                weighted_loader,
                lambda x, z, w: torch.mean(w * (config.tau * weighted_es(x) - z) ** 2),
                config.epochs,
                config.learning_rate,
            )

            with torch.no_grad():
                weighted_es_pred = weighted_es(x_test_tensor).cpu().numpy().ravel()

            results["WFNR"] = {
                "method": "WFNR",
                "kernel": "",
                "mspq": float(np.mean((data.q_test - q_test) ** 2)),
                "mspe": float(np.mean((data.es_test - weighted_es_pred) ** 2)),
            }

        if requested & {"RFNR", "RWFNR"}:
            refined_quantile_loader = DataLoader(
                TensorDataset(x_tensor, y_tensor, weights),
                batch_size=config.batch_size,
                shuffle=True,
            )
            refined_quantile = FeedForwardNN(config.r_bar).to(device)
            train_model(
                refined_quantile,
                refined_quantile_loader,
                lambda x, y, w: weighted_first_stage_quantile_loss(
                    refined_quantile(x),
                    y,
                    w,
                    config,
                ),
                config.epochs,
                config.learning_rate,
            )

            with torch.no_grad():
                q_train_refined = refined_quantile(x_tensor).cpu().numpy().ravel()
                q_test_refined = refined_quantile(x_test_tensor).cpu().numpy().ravel()

            z_refined = es_target(q_train_refined, data.y_train, config.tau)
            z_refined_tensor = torch.tensor(z_refined, dtype=torch.float32, device=device).view(-1, 1)
            refined_es_loader = DataLoader(
                TensorDataset(x_tensor, z_refined_tensor),
                batch_size=config.batch_size,
                shuffle=True,
            )
            refined_es = FeedForwardNN(config.r_bar).to(device)
            train_model(
                refined_es,
                refined_es_loader,
                lambda x, z: mse(config.tau * refined_es(x), z),
                config.epochs,
                config.learning_rate,
            )

            with torch.no_grad():
                refined_es_pred = refined_es(x_test_tensor).cpu().numpy().ravel()

            if "RFNR" in requested:
                results["RFNR"] = {
                    "method": "RFNR",
                    "kernel": "",
                    "mspq": float(np.mean((data.q_test - q_test_refined) ** 2)),
                    "mspe": float(np.mean((data.es_test - refined_es_pred) ** 2)),
                }

            if "RWFNR" in requested:
                with torch.no_grad():
                    refined_squared_errors = (config.tau * refined_es(x_tensor) - z_refined_tensor) ** 2

                refined_variance_loader = DataLoader(
                    TensorDataset(x_tensor, refined_squared_errors),
                    batch_size=config.batch_size,
                    shuffle=True,
                )
                refined_variance_model = VarianceNN(config.r_bar).to(device)
                train_model(
                    refined_variance_model,
                    refined_variance_loader,
                    lambda x, err: smooth_l1(refined_variance_model(x), err),
                    config.epochs,
                    config.learning_rate,
                )

                with torch.no_grad():
                    refined_variance_pred = refined_variance_model(x_tensor)
                    refined_c_h = resolve_c_h(config, refined_variance_pred)
                    refined_sigma2 = refined_variance_pred.clamp(min=refined_c_h / 2.0)
                    refined_weights = (1.0 / refined_sigma2).view(-1, 1)
                    refined_weights = refined_weights / refined_weights.mean()

                refined_weighted_loader = DataLoader(
                    TensorDataset(x_tensor, z_refined_tensor, refined_weights),
                    batch_size=config.batch_size,
                    shuffle=True,
                )
                refined_weighted_es = FeedForwardNN(config.r_bar).to(device)
                train_model(
                    refined_weighted_es,
                    refined_weighted_loader,
                    lambda x, z, w: torch.mean(w * (config.tau * refined_weighted_es(x) - z) ** 2),
                    config.epochs,
                    config.learning_rate,
                )

                with torch.no_grad():
                    refined_weighted_es_pred = refined_weighted_es(x_test_tensor).cpu().numpy().ravel()

                results["RWFNR"] = {
                    "method": "RWFNR",
                    "kernel": "",
                    "mspq": float(np.mean((data.q_test - q_test_refined) ** 2)),
                    "mspe": float(np.mean((data.es_test - refined_weighted_es_pred) ** 2)),
                }

    return results


def run_fast_neural_family(
    data: SimulationData,
    config: SimulationConfig,
    methods: Iterable[str],
    repeat: int | None = None,
) -> tuple[dict[str, dict[str, float | str]], list[dict[str, float | int | str]]]:
    """Run FAST methods with a shared first-stage FAST quantile and Gamma."""
    requested = set(methods)
    results: dict[str, dict[str, float | str]] = {}
    ci_records: list[dict[str, float | int | str]] = []
    if not requested:
        return results, ci_records

    z_train, z_test = factor_projection(data.x_train, data.x_test, data.x_unlabeled, config.r_bar)
    x_fast_train, x_fast_test = standardize_with_train(data.x_train, data.x_test)
    device = torch.device(config.device)

    f_tensor = torch.tensor(z_train, dtype=torch.float32, device=device)
    x_tensor = torch.tensor(x_fast_train, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(data.y_train, dtype=torch.float32, device=device).view(-1, 1)
    f_test_tensor = torch.tensor(z_test, dtype=torch.float32, device=device)
    x_test_tensor = torch.tensor(x_fast_test, dtype=torch.float32, device=device)

    train_loader = DataLoader(
        TensorDataset(f_tensor, x_tensor, y_tensor),
        batch_size=config.batch_size,
        shuffle=True,
    )
    quantile_model = FASTNetwork(
        data.x_train.shape[1],
        config.r_bar,
        config.throughput_dim,
        config.truncation,
    ).to(device)
    train_model(
        quantile_model,
        train_loader,
        lambda f, x, y: first_stage_quantile_loss(quantile_model(f, x), y, config)
        + gamma_penalty(quantile_model, config, config.lambda_q),
        config.epochs,
        config.learning_rate,
    )

    with torch.no_grad():
        q_train = quantile_model(f_tensor, x_tensor).cpu().numpy().ravel()
        q_test = quantile_model(f_test_tensor, x_test_tensor).cpu().numpy().ravel()
    shared_train = shared_fast_features(quantile_model, f_tensor, x_tensor)
    shared_test = shared_fast_features(quantile_model, f_test_tensor, x_test_tensor)

    z_target = es_target(q_train, data.y_train, config.tau)
    z_tensor = torch.tensor(z_target, dtype=torch.float32, device=device).view(-1, 1)
    es_loader = DataLoader(
        TensorDataset(shared_train, z_tensor),
        batch_size=config.batch_size,
        shuffle=True,
    )

    pilot_es = FeedForwardNN(shared_train.shape[1]).to(device)
    mse = nn.MSELoss()
    train_model(
        pilot_es,
        es_loader,
        lambda z_fast, z: mse(config.tau * pilot_es(z_fast), z),
        config.epochs,
        config.learning_rate,
    )

    with torch.no_grad():
        unweighted_es_pred = pilot_es(shared_test).cpu().numpy().ravel()

    if "FAST-FNR" in requested:
        results["FAST-FNR"] = {
            "method": "FAST-FNR",
            "kernel": "",
            "mspq": float(np.mean((data.q_test - q_test) ** 2)),
            "mspe": float(np.mean((data.es_test - unweighted_es_pred) ** 2)),
        }

    if requested & {"FAST-WFNR", "FAST-RFNR", "FAST-RWFNR"}:
        with torch.no_grad():
            squared_errors = (config.tau * pilot_es(shared_train) - z_tensor) ** 2

        variance_loader = DataLoader(
            TensorDataset(shared_train, squared_errors),
            batch_size=config.batch_size,
            shuffle=True,
        )
        variance_model = VarianceNN(shared_train.shape[1]).to(device)
        smooth_l1 = nn.SmoothL1Loss()
        train_model(
            variance_model,
            variance_loader,
            lambda z_fast, err: smooth_l1(variance_model(z_fast), err),
            config.epochs,
            config.learning_rate,
        )

        with torch.no_grad():
            variance_pred = variance_model(shared_train)
            chosen_c_h = resolve_c_h(config, variance_pred)
            sigma2 = variance_pred.clamp(min=chosen_c_h / 2.0)
            weights = (1.0 / sigma2).view(-1, 1)
            weights = weights / weights.mean()

        if "FAST-WFNR" in requested:
            weighted_loader = DataLoader(
                TensorDataset(shared_train, z_tensor, weights),
                batch_size=config.batch_size,
                shuffle=True,
            )
            weighted_es = FeedForwardNN(shared_train.shape[1]).to(device)
            train_model(
                weighted_es,
                weighted_loader,
                lambda z_fast, z, w: torch.mean(w * (config.tau * weighted_es(z_fast) - z) ** 2),
                config.epochs,
                config.learning_rate,
            )

            with torch.no_grad():
                weighted_es_pred = weighted_es(shared_test).cpu().numpy().ravel()

            results["FAST-WFNR"] = {
                "method": "FAST-WFNR",
                "kernel": "",
                "mspq": float(np.mean((data.q_test - q_test) ** 2)),
                "mspe": float(np.mean((data.es_test - weighted_es_pred) ** 2)),
            }

        if requested & {"FAST-RFNR", "FAST-RWFNR"}:
            refined_quantile_loader = DataLoader(
                TensorDataset(f_tensor, x_tensor, y_tensor, weights),
                batch_size=config.batch_size,
                shuffle=True,
            )
            train_model(
                quantile_model,
                refined_quantile_loader,
                lambda f, x, y, w: weighted_first_stage_quantile_loss(
                    quantile_model(f, x),
                    y,
                    w,
                    config,
                )
                + gamma_penalty(quantile_model, config, config.lambda_refine),
                config.epochs,
                config.learning_rate,
            )

            with torch.no_grad():
                q_train_refined = quantile_model(f_tensor, x_tensor).cpu().numpy().ravel()
                q_test_refined = quantile_model(f_test_tensor, x_test_tensor).cpu().numpy().ravel()
            shared_train = shared_fast_features(quantile_model, f_tensor, x_tensor)
            shared_test = shared_fast_features(quantile_model, f_test_tensor, x_test_tensor)

            z_refined = es_target(q_train_refined, data.y_train, config.tau)
            z_refined_tensor = torch.tensor(z_refined, dtype=torch.float32, device=device).view(-1, 1)
            refined_es_loader = DataLoader(
                TensorDataset(shared_train, z_refined_tensor),
                batch_size=config.batch_size,
                shuffle=True,
            )
            refined_es = FeedForwardNN(shared_train.shape[1]).to(device)
            train_model(
                refined_es,
                refined_es_loader,
                lambda z_fast, z: mse(config.tau * refined_es(z_fast), z),
                config.epochs,
                config.learning_rate,
            )

            with torch.no_grad():
                refined_es_pred = refined_es(shared_test).cpu().numpy().ravel()

            if "FAST-RFNR" in requested:
                results["FAST-RFNR"] = {
                    "method": "FAST-RFNR",
                    "kernel": "",
                    "mspq": float(np.mean((data.q_test - q_test_refined) ** 2)),
                    "mspe": float(np.mean((data.es_test - refined_es_pred) ** 2)),
                }

            if "FAST-RWFNR" in requested:
                with torch.no_grad():
                    refined_squared_errors = (config.tau * refined_es(shared_train) - z_refined_tensor) ** 2

                refined_variance_loader = DataLoader(
                    TensorDataset(shared_train, refined_squared_errors),
                    batch_size=config.batch_size,
                    shuffle=True,
                )
                refined_variance_model = VarianceNN(shared_train.shape[1]).to(device)
                train_model(
                    refined_variance_model,
                    refined_variance_loader,
                    lambda z_fast, err: smooth_l1(refined_variance_model(z_fast), err),
                    config.epochs,
                    config.learning_rate,
                )

                with torch.no_grad():
                    refined_variance_pred = refined_variance_model(shared_train)
                    refined_c_h = resolve_c_h(config, refined_variance_pred)
                    refined_sigma2 = refined_variance_pred.clamp(min=refined_c_h / 2.0)
                    refined_weights = (1.0 / refined_sigma2).view(-1, 1)
                    refined_weights = refined_weights / refined_weights.mean()

                refined_weighted_loader = DataLoader(
                    TensorDataset(shared_train, z_refined_tensor, refined_weights),
                    batch_size=config.batch_size,
                    shuffle=True,
                )
                refined_weighted_es = FeedForwardNN(shared_train.shape[1]).to(device)
                train_model(
                    refined_weighted_es,
                    refined_weighted_loader,
                    lambda z_fast, z, w: torch.mean(w * (config.tau * refined_weighted_es(z_fast) - z) ** 2),
                    config.epochs,
                    config.learning_rate,
                )

                with torch.no_grad():
                    refined_weighted_es_pred = refined_weighted_es(shared_test).cpu().numpy().ravel()

                ci_fields: dict[str, float] = {}
                if config.bootstrap_B > 0:
                    q_ci_fields, q_ci_lower, q_ci_upper = bootstrap_refined_quantile_ci(
                        shared_train,
                        shared_test,
                        y_tensor,
                        weights,
                        q_test_refined,
                        data.q_test,
                        config,
                        repeat,
                    )
                    ci_fields.update(q_ci_fields)
                    es_ci_fields, es_ci_lower, es_ci_upper = bootstrap_final_es_ci(
                        shared_train,
                        shared_test,
                        z_refined_tensor,
                        refined_weights,
                        refined_weighted_es_pred,
                        data.es_test,
                        config,
                        repeat,
                    )
                    ci_fields.update(es_ci_fields)
                    if config.save_bootstrap_ci:
                        ci_records.extend(
                            fast_rwfnr_pointwise_ci_records(
                                config,
                                -1 if repeat is None else repeat,
                                q_test_refined,
                                data.q_test,
                                q_ci_lower,
                                q_ci_upper,
                                refined_weighted_es_pred,
                                data.es_test,
                                es_ci_lower,
                                es_ci_upper,
                            )
                        )

                results["FAST-RWFNR"] = {
                    "method": "FAST-RWFNR",
                    "kernel": "",
                    "mspq": float(np.mean((data.q_test - q_test_refined) ** 2)),
                    "mspe": float(np.mean((data.es_test - refined_weighted_es_pred) ** 2)),
                    "bootstrap_B": int(config.bootstrap_B),
                    "ci_level": float(config.ci_level),
                    **ci_fields,
                }

    return results, ci_records


def summarize(values: list[float]) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values, ddof=0))


def run_one_setting(config: SimulationConfig, methods: Iterable[str]) -> list[dict[str, float | int | str]]:
    records: list[dict[str, float | int | str]] = []
    bootstrap_ci_records: list[dict[str, float | int | str]] = []
    method_names = tuple(methods)
    raw_results: dict[str, dict[str, list[float] | str]] = {
        method: {
            "mspq": [],
            "mspe": [],
            "q_ci_lower": [],
            "q_ci_upper": [],
            "q_ci_width": [],
            "q_ci_coverage": [],
            "es_ci_lower": [],
            "es_ci_upper": [],
            "es_ci_width": [],
            "es_ci_coverage": [],
            "kernel": "",
        }
        for method in method_names
    }

    for repeat in range(config.repeats):
        set_seed(config.seed + repeat)
        data = generate_data(config)

        repeat_results: dict[str, dict[str, float | str]] = {}
        if "KRR" in method_names:
            repeat_results["KRR"] = run_krr(data, config.tau, config.krr_kernel)
        if "HDES" in method_names:
            repeat_results["HDES"] = run_hdes(data, config)

        factor_methods = [
            method
            for method in method_names
            if method in {"FNR", "WFNR", "RFNR", "RWFNR"}
        ]
        repeat_results.update(run_factor_neural_family(data, config, factor_methods))

        fast_methods = [
            method
            for method in method_names
            if method in {"FAST-FNR", "FAST-WFNR", "FAST-RFNR", "FAST-RWFNR"}
        ]
        fast_results, fast_ci_records = run_fast_neural_family(data, config, fast_methods, repeat)
        repeat_results.update(fast_results)
        if config.save_bootstrap_ci:
            bootstrap_ci_records.extend(fast_ci_records)

        for method in method_names:
            if method not in repeat_results:
                raise ValueError(f"Unknown method: {method}")
            result = repeat_results[method]

            raw_results[method]["mspq"].append(float(result["mspq"]))
            raw_results[method]["mspe"].append(float(result["mspe"]))
            if "q_ci_lower_mean" in result:
                raw_results[method]["q_ci_lower"].append(float(result["q_ci_lower_mean"]))
            if "q_ci_upper_mean" in result:
                raw_results[method]["q_ci_upper"].append(float(result["q_ci_upper_mean"]))
            if "q_ci_width_mean" in result:
                raw_results[method]["q_ci_width"].append(float(result["q_ci_width_mean"]))
            if "q_ci_coverage" in result:
                raw_results[method]["q_ci_coverage"].append(float(result["q_ci_coverage"]))
            if "es_ci_lower_mean" in result:
                raw_results[method]["es_ci_lower"].append(float(result["es_ci_lower_mean"]))
            if "es_ci_upper_mean" in result:
                raw_results[method]["es_ci_upper"].append(float(result["es_ci_upper_mean"]))
            if "es_ci_width_mean" in result:
                raw_results[method]["es_ci_width"].append(float(result["es_ci_width_mean"]))
            if "es_ci_coverage" in result:
                raw_results[method]["es_ci_coverage"].append(float(result["es_ci_coverage"]))
            if result.get("kernel"):
                raw_results[method]["kernel"] = str(result["kernel"])

    for method in method_names:
        mspq_mean, mspq_std = summarize(raw_results[method]["mspq"])  # type: ignore[arg-type]
        mspe_mean, mspe_std = summarize(raw_results[method]["mspe"])  # type: ignore[arg-type]
        q_ci_lower_values = raw_results[method]["q_ci_lower"]  # type: ignore[assignment]
        q_ci_upper_values = raw_results[method]["q_ci_upper"]  # type: ignore[assignment]
        q_ci_width_values = raw_results[method]["q_ci_width"]  # type: ignore[assignment]
        q_ci_coverage_values = raw_results[method]["q_ci_coverage"]  # type: ignore[assignment]
        es_ci_lower_values = raw_results[method]["es_ci_lower"]  # type: ignore[assignment]
        es_ci_upper_values = raw_results[method]["es_ci_upper"]  # type: ignore[assignment]
        es_ci_width_values = raw_results[method]["es_ci_width"]  # type: ignore[assignment]
        es_ci_coverage_values = raw_results[method]["es_ci_coverage"]  # type: ignore[assignment]
        q_ci_lower_mean = float(np.mean(q_ci_lower_values)) if q_ci_lower_values else float("nan")
        q_ci_upper_mean = float(np.mean(q_ci_upper_values)) if q_ci_upper_values else float("nan")
        q_ci_width_mean = float(np.mean(q_ci_width_values)) if q_ci_width_values else float("nan")
        q_ci_coverage_mean = float(np.mean(q_ci_coverage_values)) if q_ci_coverage_values else float("nan")
        es_ci_lower_mean = float(np.mean(es_ci_lower_values)) if es_ci_lower_values else float("nan")
        es_ci_upper_mean = float(np.mean(es_ci_upper_values)) if es_ci_upper_values else float("nan")
        es_ci_width_mean = float(np.mean(es_ci_width_values)) if es_ci_width_values else float("nan")
        es_ci_coverage_mean = float(np.mean(es_ci_coverage_values)) if es_ci_coverage_values else float("nan")
        suffix = "_factor" if config.factor_model else "_nonfactor"
        if config.heterogeneity != "none":
            suffix = f"_factor_sparse_{config.heterogeneity}_heterogeneity"
        records.append(
            {
                "example": config.design + suffix,
                "method": method,
                "kernel": raw_results[method]["kernel"],
                "heterogeneity": config.heterogeneity,
                "heterogeneity_distribution": config.heterogeneity_distribution,
                "heterogeneity_location_strength": config.heterogeneity_location_strength,
                "heterogeneity_scale_strength": config.heterogeneity_scale_strength,
                "nonfactor_distribution": config.nonfactor_distribution,
                "nonfactor_case": config.nonfactor_case,
                "innovation": config.innovation,
                "krr_kernel": config.krr_kernel,
                "p": config.p,
                "tau": config.tau,
                "repeats": config.repeats,
                "throughput_dim": config.throughput_dim,
                "truncation": config.truncation,
                "quantile_loss": config.quantile_loss,
                "quantile_huber_kappa": config.quantile_huber_kappa,
                "hdes_lambda_q": config.hdes_lambda_q,
                "hdes_lambda_e": config.hdes_lambda_e,
                "hdes_learning_rate": config.hdes_learning_rate,
                "hdes_epochs": config.hdes_epochs,
                "lambda_refine": config.lambda_refine,
                "c_h": config.c_h,
                "c_h_quantile": config.c_h_quantile,
                "c_h_scale": config.c_h_scale,
                "bootstrap_B": config.bootstrap_B,
                "ci_level": config.ci_level,
                "save_bootstrap_ci": config.save_bootstrap_ci,
                "q_ci_lower": q_ci_lower_mean,
                "q_ci_upper": q_ci_upper_mean,
                "q_ci_width_mean": q_ci_width_mean,
                "q_ci_coverage_mean": q_ci_coverage_mean,
                "es_ci_lower": es_ci_lower_mean,
                "es_ci_upper": es_ci_upper_mean,
                "es_ci_width_mean": es_ci_width_mean,
                "es_ci_coverage_mean": es_ci_coverage_mean,
                "mspq_mean": mspq_mean,
                "mspq_std": mspq_std,
                "mspe_mean": mspe_mean,
                "mspe_std": mspe_std,
            }
        )
    if config.save_bootstrap_ci and bootstrap_ci_records:
        write_bootstrap_ci_records(
            bootstrap_ci_records,
            Path(config.bootstrap_ci_output_dir) / f"bootstrap_ci_p{config.p}_tau{config.tau:g}.csv",
        )
    return records


def write_csv(records: list[dict[str, float | int | str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def run_grid(
    base_config: SimulationConfig,
    output_path: str | Path,
    p_values: Iterable[int] = P_VALUES,
    tau_values: Iterable[float] = TAU_VALUES,
    methods: Iterable[str] = (
        "KRR",
        "HDES",
        "FNR",
        "WFNR",
        "RFNR",
        "RWFNR",
        "FAST-FNR",
        "FAST-WFNR",
        "FAST-RFNR",
        "FAST-RWFNR",
    ),
) -> list[dict[str, float | int | str]]:
    output_path = Path(output_path)
    ci_output_dir = Path(base_config.bootstrap_ci_output_dir)
    if not ci_output_dir.is_absolute():
        ci_output_dir = output_path.parent / ci_output_dir
    base_config = SimulationConfig(
        **{**base_config.__dict__, "bootstrap_ci_output_dir": str(ci_output_dir)}
    )

    all_records: list[dict[str, float | int | str]] = []
    for p in p_values:
        for tau in tau_values:
            config = SimulationConfig(**{**base_config.__dict__, "p": p, "tau": tau})
            print(f"Running {config.design}, factor={config.factor_model}, p={p}, tau={tau}")
            all_records.extend(run_one_setting(config, methods))

    write_csv(all_records, output_path)
    return all_records


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--p-values", nargs="+", type=int, default=list(P_VALUES))
    parser.add_argument("--taus", nargs="+", type=float, default=list(TAU_VALUES))
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--innovation", choices=("t3", "normal", "gamma"), default="normal")
    parser.add_argument(
        "--nonfactor-distribution",
        choices=("normal-iid", "normal-ar", "copula-beta"),
        default="copula-beta",
        help="Covariate distribution for non-factor designs.",
    )
    parser.add_argument(
        "--nonfactor-case",
        choices=("current", "3a", "3b"),
        default="current",
        help="Response design for non-factor data. Use 3a/3b to reproduce Example 3.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--throughput-dim", type=int, default=8)
    parser.add_argument("--truncation", type=float, default=3.0)
    parser.add_argument("--penalty-eps", type=float, default=5e-3)
    parser.add_argument("--quantile-loss", choices=("pinball", "huber"), default="pinball")
    parser.add_argument("--quantile-huber-kappa", type=float, default=0.1)
    parser.add_argument(
        "--heterogeneity-distribution",
        choices=("uniform", "normal", "normal-ar", "t3", "normal-sparse", "t3-sparse"),
        default="normal-sparse",
        help="Idiosyncratic component distribution for sparse heterogeneity designs.",
    )
    parser.add_argument(
        "--heterogeneity-location-strength",
        type=float,
        default=0.5,
        help="Multiplier for sparse heterogeneity in the location function.",
    )
    parser.add_argument(
        "--heterogeneity-scale-strength",
        type=float,
        default=0.25,
        help="Multiplier for sparse heterogeneity in the scale index.",
    )
    parser.add_argument(
        "--krr-kernel",
        choices=("auto", "rbf", "gaussian", "polynomial"),
        default="auto",
        help="KRR kernel. auto tries rbf, gaussian, then polynomial.",
    )
    parser.add_argument("--lambda-q", type=float, default=None)
    parser.add_argument("--lambda-refine", type=float, default=None)
    parser.add_argument("--hdes-lambda-q", type=float, default=None)
    parser.add_argument("--hdes-lambda-e", type=float, default=None)
    parser.add_argument("--hdes-learning-rate", type=float, default=None)
    parser.add_argument("--hdes-epochs", type=int, default=None)
    parser.add_argument("--c-h", type=float, default=None)
    parser.add_argument("--c-h-quantile", type=float, default=0.05)
    parser.add_argument("--c-h-scale", type=float, default=0.5)
    parser.add_argument("--bootstrap-B", type=int, default=0)
    parser.add_argument("--ci-level", type=float, default=0.95)
    parser.add_argument("--save-bootstrap-ci", action="store_true")
    parser.add_argument("--bootstrap-ci-output-dir", default="bootstrap_ci")
    parser.add_argument("--output", default=None)


def config_from_args(
    args: argparse.Namespace,
    design: str,
    factor_model: bool,
    heterogeneity: str = "none",
) -> SimulationConfig:
    return SimulationConfig(
        design=design,
        factor_model=factor_model,
        heterogeneity=heterogeneity,
        innovation=args.innovation,
        nonfactor_distribution=args.nonfactor_distribution,
        nonfactor_case=args.nonfactor_case,
        n_train=args.n_train,
        n_test=args.n_test,
        epochs=args.epochs,
        repeats=args.repeats,
        throughput_dim=args.throughput_dim,
        truncation=args.truncation,
        penalty_eps=args.penalty_eps,
        quantile_loss=args.quantile_loss,
        quantile_huber_kappa=args.quantile_huber_kappa,
        heterogeneity_distribution=args.heterogeneity_distribution,
        heterogeneity_location_strength=args.heterogeneity_location_strength,
        heterogeneity_scale_strength=args.heterogeneity_scale_strength,
        krr_kernel=args.krr_kernel,
        lambda_q=args.lambda_q,
        lambda_refine=args.lambda_refine,
        hdes_lambda_q=args.hdes_lambda_q,
        hdes_lambda_e=args.hdes_lambda_e,
        hdes_learning_rate=args.hdes_learning_rate,
        hdes_epochs=args.hdes_epochs,
        c_h=args.c_h,
        c_h_quantile=args.c_h_quantile,
        c_h_scale=args.c_h_scale,
        bootstrap_B=args.bootstrap_B,
        ci_level=args.ci_level,
        save_bootstrap_ci=args.save_bootstrap_ci,
        bootstrap_ci_output_dir=args.bootstrap_ci_output_dir,
        device=args.device,
    )