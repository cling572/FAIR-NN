"""Fixed-split FAIR-NN experiment for Tecator upper-tail ES."""

from __future__ import annotations

import itertools
import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn, optim
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:  # pragma: no cover - dependency is external.
    torch = None
    nn = None
    optim = None
    DataLoader = None
    TensorDataset = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None

try:
    from sklearn.decomposition import PCA
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.kernel_ridge import KernelRidge
except ImportError as exc:  # pragma: no cover - dependency is external.
    PCA = None
    GradientBoostingRegressor = None
    KernelRidge = None
    _SKLEARN_IMPORT_ERROR = exc
else:
    _SKLEARN_IMPORT_ERROR = None


METHOD_ORDER = ("RLR", "KRR", "FA-NN", "FAIR-NN")
_TorchModule = nn.Module if nn is not None else object


@dataclass
class ExperimentConfig:
    """Configuration for the Tecator upper-tail conditional ES experiment."""

    data_path: str = "data/tecator_fair_nn_213.csv"
    output_dir: str = "results/fair_nn_upper_tail"
    methods: list[str] = field(default_factory=lambda: list(METHOD_ORDER))
    taus: list[float] = field(default_factory=lambda: [0.70, 0.80, 0.90])
    train_partitions: list[str] = field(default_factory=lambda: ["C", "M"])
    test_partitions: list[str] = field(default_factory=lambda: ["T"])
    train_size: int = 170
    test_size: int = 43
    seed: int = 20260819
    factor_dims: list[int] = field(default_factory=lambda: [4])
    throughput_dims: list[int] = field(default_factory=lambda: [4])
    lambda_gamma_candidates: list[float | None] = field(default_factory=lambda: [None])
    epochs: int = 250
    rlr_epochs: int = 300
    hidden_dim: int = 32
    hidden_layers: int = 2
    batch_size: int = 32
    learning_rate: float = 1e-3
    truncation: float = 3.0
    gamma_penalty_eps: float = 0.005
    lasso_lambda: float | None = None
    krr_alpha: float = 1.0
    krr_kernel: str = "auto"
    krr_kernel_candidates: list[str] = field(default_factory=list)
    krr_gamma: float | str | None = "auto"
    krr_degree: int = 3
    krr_coef0: float = 1.0
    c_h: float | None = None
    c_h_quantile: float = 0.05
    c_h_scale: float = 0.50
    response_standardize: bool = True
    device: str = "auto"


@dataclass
class TecatorData:
    """Validated raw Tecator analysis data."""

    sample_ids: np.ndarray
    source_partitions: np.ndarray
    x_raw: np.ndarray
    y_raw: np.ndarray
    feature_names: list[str]
    source_sha256: str


@dataclass
class FoldData:
    """Training-fold preprocessing and its matched evaluation fold."""

    train_ids: np.ndarray
    eval_ids: np.ndarray
    x_train: np.ndarray
    x_eval: np.ndarray
    y_train: np.ndarray
    y_eval: np.ndarray
    y_train_raw: np.ndarray
    y_eval_raw: np.ndarray
    response_mean: float
    response_scale: float
    feature_names: list[str]


@dataclass
class MethodResult:
    """Standardized predictions returned by one fitted method."""

    method: str
    q_train: np.ndarray
    es_train: np.ndarray
    q_eval: np.ndarray
    es_eval: np.ndarray
    z_train: np.ndarray
    gamma: np.ndarray | None = None
    variance_train: np.ndarray | None = None
    variance_eval: np.ndarray | None = None
    weight_train: np.ndarray | None = None
    weight_eval: np.ndarray | None = None


class FeedForwardNN(_TorchModule):
    """Compact MLP used for quantile, ES, and variance regressions."""

    def __init__(self, input_dim: int, config: ExperimentConfig, positive_output: bool = False):
        require_torch()
        super().__init__()
        layers: list[Any] = []
        current_dim = input_dim
        for _ in range(config.hidden_layers):
            layers.extend([nn.Linear(current_dim, config.hidden_dim), nn.ReLU()])
            current_dim = config.hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        if positive_output:
            layers.append(nn.Softplus())
        self.net = nn.Sequential(*layers)

    def forward(self, x: Any) -> Any:
        return self.net(x)


class LinearLassoNN(_TorchModule):
    """Linear quantile/ES baseline with an L1 penalty."""

    def __init__(self, input_dim: int):
        require_torch()
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: Any) -> Any:
        return self.linear(x)

    def l1_penalty(self) -> Any:
        return self.linear.weight.abs().sum()


class SparseThroughput(_TorchModule):
    """Trainable sparse projection of all standardized spectral channels."""

    def __init__(self, p: int, throughput_dim: int, truncation: float):
        require_torch()
        super().__init__()
        self.throughput_dim = throughput_dim
        self.truncation = truncation
        if throughput_dim > 0:
            self.gamma = nn.Parameter(torch.zeros(p, throughput_dim))
            nn.init.normal_(self.gamma, mean=0.0, std=1e-4)
        else:
            self.register_parameter("gamma", None)

    def forward(self, factors: Any, x_full: Any) -> Any:
        if self.gamma is None:
            return factors
        throughput = torch.clamp(x_full @ self.gamma, -self.truncation, self.truncation)
        return torch.cat([factors, throughput], dim=1)


class FAIRQuantileNN(_TorchModule):
    """Factor-augmented quantile network with sparse-throughput input."""

    def __init__(
        self,
        p: int,
        factor_dim: int,
        throughput_dim: int,
        config: ExperimentConfig,
    ):
        require_torch()
        super().__init__()
        self.throughput = SparseThroughput(p, throughput_dim, config.truncation)
        self.net = FeedForwardNN(factor_dim + throughput_dim, config)

    @property
    def gamma(self) -> Any:
        return self.throughput.gamma

    def features(self, factors: Any, x_full: Any) -> Any:
        return self.throughput(factors, x_full)

    def forward(self, factors: Any, x_full: Any) -> Any:
        return self.net(self.features(factors, x_full))


def require_torch() -> None:
    if torch is None:
        raise ImportError(
            "PyTorch is required. Install the packages in requirements.txt before training."
        ) from _TORCH_IMPORT_ERROR


def require_sklearn() -> None:
    if PCA is None or GradientBoostingRegressor is None or KernelRidge is None:
        raise ImportError(
            "scikit-learn is required. Install the packages in requirements.txt before training."
        ) from _SKLEARN_IMPORT_ERROR


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def resolve_device(config: ExperimentConfig) -> Any:
    require_torch()
    if config.device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(config.device)


def normalize_methods(methods: Iterable[str]) -> list[str]:
    aliases = {
        "RLR": "RLR",
        "KRR": "KRR",
        "FA-NN": "FA-NN",
        "FAN": "FA-NN",
        "FAIR-NN": "FAIR-NN",
        "FAIRNN": "FAIR-NN",
    }
    normalized = []
    for method in methods:
        key = str(method).upper()
        if key not in aliases:
            raise ValueError(f"Unknown method: {method}. Allowed: {sorted(METHOD_ORDER)}")
        normalized.append(aliases[key])
    return list(dict.fromkeys(normalized))


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_tecator_data(path: Path) -> TecatorData:
    if not path.exists():
        raise FileNotFoundError(f"Cannot find Tecator analysis data: {path}")
    frame = pd.read_csv(path)
    feature_names = [name for name in frame.columns if name.startswith("absorbance_")]
    required = {"sample_id", "source_partition", "fat_pct"}
    if required.difference(frame.columns):
        raise ValueError(f"Missing required columns: {sorted(required.difference(frame.columns))}")
    if len(feature_names) != 100:
        raise ValueError(f"Expected 100 absorbance columns, found {len(feature_names)}")
    if frame.shape[0] != 213:
        raise ValueError(f"Expected 213 Tecator analysis rows, found {frame.shape[0]}")
    sample_ids = frame["sample_id"].to_numpy(dtype=int)
    if {103, 105}.intersection(sample_ids):
        raise ValueError("Tecator analysis data must exclude samples 103 and 105")
    if sample_ids.min() != 1 or sample_ids.max() != 215:
        raise ValueError("Tecator analysis data must use the retained 1--215 interpolation cohort")
    x_raw = frame[feature_names].to_numpy(dtype=float)
    y_raw = frame["fat_pct"].to_numpy(dtype=float)
    if not np.isfinite(x_raw).all() or not np.isfinite(y_raw).all():
        raise ValueError("Tecator analysis data contain missing or non-finite values")
    return TecatorData(
        sample_ids=sample_ids,
        source_partitions=frame["source_partition"].to_numpy(dtype=str),
        x_raw=x_raw,
        y_raw=y_raw,
        feature_names=feature_names,
        source_sha256=sha256_file(path),
    )


def fixed_split(data: TecatorData, config: ExperimentConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return the configured official C/M training and T testing partition."""

    train_partitions = {str(value) for value in config.train_partitions}
    test_partitions = {str(value) for value in config.test_partitions}
    if not train_partitions or not test_partitions:
        raise ValueError("train_partitions and test_partitions must both be nonempty")
    if train_partitions.intersection(test_partitions):
        raise ValueError("train_partitions and test_partitions must not overlap")

    train_idx = np.flatnonzero(np.isin(data.source_partitions, list(train_partitions)))
    test_idx = np.flatnonzero(np.isin(data.source_partitions, list(test_partitions)))
    if len(train_idx) != config.train_size:
        raise ValueError(
            f"Configured train_size={config.train_size}, but selected {len(train_idx)} rows "
            f"from partitions {sorted(train_partitions)}"
        )
    if len(test_idx) != config.test_size:
        raise ValueError(
            f"Configured test_size={config.test_size}, but selected {len(test_idx)} rows "
            f"from partitions {sorted(test_partitions)}"
        )
    if len(train_idx) + len(test_idx) != len(data.sample_ids):
        raise ValueError(
            "The fixed train/test partitions must exhaust the Tecator analysis cohort"
        )
    return np.sort(train_idx), np.sort(test_idx)


def prepare_fold(
    data: TecatorData,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    config: ExperimentConfig,
) -> FoldData:
    x_train_raw = data.x_raw[train_idx]
    x_eval_raw = data.x_raw[eval_idx]
    x_mean = x_train_raw.mean(axis=0)
    x_scale = x_train_raw.std(axis=0, ddof=0)
    x_scale[x_scale <= 1e-12] = 1.0
    x_train = (x_train_raw - x_mean) / x_scale
    x_eval = (x_eval_raw - x_mean) / x_scale

    y_train_raw = data.y_raw[train_idx]
    y_eval_raw = data.y_raw[eval_idx]
    response_mean = 0.0
    response_scale = 1.0
    if config.response_standardize:
        response_mean = float(y_train_raw.mean())
        response_scale = float(y_train_raw.std(ddof=0))
        if response_scale <= 1e-12:
            response_scale = 1.0
    return FoldData(
        train_ids=data.sample_ids[train_idx],
        eval_ids=data.sample_ids[eval_idx],
        x_train=x_train,
        x_eval=x_eval,
        y_train=(y_train_raw - response_mean) / response_scale,
        y_eval=(y_eval_raw - response_mean) / response_scale,
        y_train_raw=y_train_raw,
        y_eval_raw=y_eval_raw,
        response_mean=response_mean,
        response_scale=response_scale,
        feature_names=data.feature_names,
    )


def to_tensor(array: np.ndarray, device: Any) -> Any:
    return torch.tensor(array, dtype=torch.float32, device=device)


def train_model(model: Any, loader: Any, loss_fn: Any, epochs: int, learning_rate: float) -> None:
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for _ in range(epochs):
        for batch in loader:
            optimizer.zero_grad()
            loss = loss_fn(*batch)
            loss.backward()
            optimizer.step()


def quantile_loss(prediction: Any, target: Any, tau: float) -> Any:
    residual = target - prediction
    return torch.mean(torch.maximum(tau * residual, (tau - 1.0) * residual))


def weighted_quantile_loss(prediction: Any, target: Any, weights: Any, tau: float) -> Any:
    residual = target - prediction
    pinball = torch.maximum(tau * residual, (tau - 1.0) * residual)
    return torch.mean(weights * pinball)


def upper_tail_pseudo_outcome(q: np.ndarray, y: np.ndarray, tau: float) -> np.ndarray:
    return (1.0 - tau) * q + (y - q) * (y > q)


def upper_tail_calibration_score(q: np.ndarray, es: np.ndarray, y: np.ndarray, tau: float) -> np.ndarray:
    return es - q - (y - q) * (y > q) / (1.0 - tau)


def pca_features(
    fold: FoldData,
    factor_dim: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    require_sklearn()
    dimension = min(int(factor_dim), fold.x_train.shape[0], fold.x_train.shape[1])
    if dimension < 1:
        raise ValueError("factor_dim must be at least one")
    pca = PCA(n_components=dimension, random_state=seed)
    # Apple Accelerate may emit spurious floating-point warnings for finite
    # matrix products. Validate the result explicitly instead of treating them
    # as evidence of an invalid factor representation.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        factors_train = pca.fit_transform(fold.x_train)
        factors_eval = pca.transform(fold.x_eval)
    # With D = sqrt(p) V in the manuscript, p^{-1} D^T X equals the
    # conventional PCA score V^T X divided by sqrt(p).
    projection_scale = math.sqrt(fold.x_train.shape[1])
    factors_train = factors_train / projection_scale
    factors_eval = factors_eval / projection_scale
    if not np.isfinite(factors_train).all() or not np.isfinite(factors_eval).all():
        raise FloatingPointError("PCA produced non-finite factor scores")
    return factors_train, factors_eval


def resolve_gamma_lambda(value: float | None, p: int, n_train: int) -> float:
    if value is not None:
        return float(value)
    return 1.3 * math.log(max(p, 2)) / max(n_train, 1)


def gamma_penalty(model: FAIRQuantileNN, eps: float, level: float) -> Any:
    if model.gamma is None:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    return level * torch.clamp(model.gamma.abs() / eps, max=1.0).sum()


def resolve_c_h(config: ExperimentConfig, variance: np.ndarray, response_scale: float) -> float:
    if config.c_h is not None:
        return max(float(config.c_h) / response_scale**2, 1e-8)
    finite = variance[np.isfinite(variance)]
    return max(config.c_h_scale * float(np.quantile(finite, config.c_h_quantile)), 1e-8)


def adaptive_weights(
    config: ExperimentConfig,
    variance: np.ndarray,
    response_scale: float,
    variance_floor: float | None = None,
) -> np.ndarray:
    lower = (
        resolve_c_h(config, variance, response_scale)
        if variance_floor is None
        else variance_floor
    )
    return 1.0 / np.maximum(variance, lower / 2.0)


def fit_es_network(
    features_train: np.ndarray,
    features_eval: np.ndarray,
    z_train: np.ndarray,
    tau: float,
    config: ExperimentConfig,
    epochs: int,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    device = resolve_device(config)
    x_train_t = to_tensor(features_train, device)
    z_train_t = to_tensor(z_train.reshape(-1, 1), device)
    tensors = [x_train_t, z_train_t]
    if weights is not None:
        tensors.append(to_tensor(weights.reshape(-1, 1), device))
    loader = DataLoader(TensorDataset(*tensors), batch_size=config.batch_size, shuffle=True)
    model = FeedForwardNN(features_train.shape[1], config).to(device)
    tail_probability = 1.0 - tau
    if weights is None:
        train_model(
            model,
            loader,
            lambda x, z: torch.mean((tail_probability * model(x) - z) ** 2),
            epochs,
            config.learning_rate,
        )
    else:
        train_model(
            model,
            loader,
            lambda x, z, w: torch.mean(w * (tail_probability * model(x) - z) ** 2),
            epochs,
            config.learning_rate,
        )
    model.eval()
    with torch.no_grad():
        es_train = model(x_train_t).cpu().numpy().ravel()
        es_eval = model(to_tensor(features_eval, device)).cpu().numpy().ravel()
    return es_train, es_eval


def fit_variance_network(
    features_train: np.ndarray,
    features_eval: np.ndarray,
    squared_residual: np.ndarray,
    config: ExperimentConfig,
    epochs: int,
) -> tuple[np.ndarray, np.ndarray]:
    device = resolve_device(config)
    x_train_t = to_tensor(features_train, device)
    residual_t = to_tensor(squared_residual.reshape(-1, 1), device)
    loader = DataLoader(TensorDataset(x_train_t, residual_t), batch_size=config.batch_size, shuffle=True)
    model = FeedForwardNN(features_train.shape[1], config, positive_output=True).to(device)
    smooth_l1 = nn.SmoothL1Loss()
    train_model(
        model,
        loader,
        lambda x, residual: smooth_l1(model(x), residual),
        epochs,
        config.learning_rate,
    )
    model.eval()
    with torch.no_grad():
        variance_train = model(x_train_t).cpu().numpy().ravel()
        variance_eval = model(to_tensor(features_eval, device)).cpu().numpy().ravel()
    return variance_train, variance_eval


def fit_fa_nn(
    fold: FoldData,
    tau: float,
    factor_dim: int,
    throughput_dim: int,
    lambda_gamma: float | None,
    config: ExperimentConfig,
    epochs: int,
    seed: int,
) -> MethodResult:
    require_torch()
    set_seed(seed)
    factors_train, factors_eval = pca_features(fold, factor_dim, seed)
    device = resolve_device(config)
    f_train_t = to_tensor(factors_train, device)
    f_eval_t = to_tensor(factors_eval, device)
    x_train_t = to_tensor(fold.x_train, device)
    x_eval_t = to_tensor(fold.x_eval, device)
    y_train_t = to_tensor(fold.y_train.reshape(-1, 1), device)
    penalty = resolve_gamma_lambda(lambda_gamma, fold.x_train.shape[1], len(fold.y_train))
    penalty /= fold.response_scale

    q_model = FAIRQuantileNN(
        fold.x_train.shape[1],
        factors_train.shape[1],
        throughput_dim,
        config,
    ).to(device)
    loader = DataLoader(
        TensorDataset(f_train_t, x_train_t, y_train_t),
        batch_size=config.batch_size,
        shuffle=True,
    )
    train_model(
        q_model,
        loader,
        lambda f, x, y: quantile_loss(q_model(f, x), y, tau)
        + gamma_penalty(q_model, config.gamma_penalty_eps, penalty),
        epochs,
        config.learning_rate,
    )
    q_model.eval()
    with torch.no_grad():
        q_train = q_model(f_train_t, x_train_t).cpu().numpy().ravel()
        q_eval = q_model(f_eval_t, x_eval_t).cpu().numpy().ravel()
        shared_train = q_model.features(f_train_t, x_train_t).cpu().numpy()
        shared_eval = q_model.features(f_eval_t, x_eval_t).cpu().numpy()
    z_train = upper_tail_pseudo_outcome(q_train, fold.y_train, tau)
    es_train, es_eval = fit_es_network(shared_train, shared_eval, z_train, tau, config, epochs)
    return MethodResult("FA-NN", q_train, es_train, q_eval, es_eval, z_train)


def fit_fair_nn(
    fold: FoldData,
    tau: float,
    factor_dim: int,
    throughput_dim: int,
    lambda_gamma: float | None,
    config: ExperimentConfig,
    epochs: int,
    seed: int,
) -> MethodResult:
    """Fit pilot q/ES/weights, then one refined weighted FAIR-NN iteration."""

    require_torch()
    set_seed(seed)
    factors_train, factors_eval = pca_features(fold, factor_dim, seed)
    device = resolve_device(config)
    f_train_t = to_tensor(factors_train, device)
    f_eval_t = to_tensor(factors_eval, device)
    x_train_t = to_tensor(fold.x_train, device)
    x_eval_t = to_tensor(fold.x_eval, device)
    y_train_t = to_tensor(fold.y_train.reshape(-1, 1), device)
    penalty = resolve_gamma_lambda(lambda_gamma, fold.x_train.shape[1], len(fold.y_train))
    penalty /= fold.response_scale
    tail_probability = 1.0 - tau

    pilot_q = FAIRQuantileNN(
        fold.x_train.shape[1],
        factors_train.shape[1],
        throughput_dim,
        config,
    ).to(device)
    pilot_loader = DataLoader(
        TensorDataset(f_train_t, x_train_t, y_train_t),
        batch_size=config.batch_size,
        shuffle=True,
    )
    train_model(
        pilot_q,
        pilot_loader,
        lambda f, x, y: quantile_loss(pilot_q(f, x), y, tau)
        + gamma_penalty(pilot_q, config.gamma_penalty_eps, penalty),
        epochs,
        config.learning_rate,
    )
    pilot_q.eval()
    with torch.no_grad():
        q_pilot_train = pilot_q(f_train_t, x_train_t).cpu().numpy().ravel()
        pilot_features_train = pilot_q.features(f_train_t, x_train_t).cpu().numpy()
        pilot_features_eval = pilot_q.features(f_eval_t, x_eval_t).cpu().numpy()
    z_pilot = upper_tail_pseudo_outcome(q_pilot_train, fold.y_train, tau)
    pilot_es_train, _ = fit_es_network(
        pilot_features_train,
        pilot_features_eval,
        z_pilot,
        tau,
        config,
        epochs,
    )
    pilot_variance_train, _ = fit_variance_network(
        pilot_features_train,
        pilot_features_eval,
        (tail_probability * pilot_es_train - z_pilot) ** 2,
        config,
        epochs,
    )
    pilot_floor = resolve_c_h(config, pilot_variance_train, fold.response_scale)
    pilot_weights = adaptive_weights(
        config,
        pilot_variance_train,
        fold.response_scale,
        variance_floor=pilot_floor,
    )
    # This is the weighted ES fit in the initial FAIR-NN ES step. The
    # subsequent quantile update consumes the same pilot weights; the final
    # weighted ES fit is recomputed after the refined representation is learned.
    fit_es_network(
        pilot_features_train,
        pilot_features_eval,
        z_pilot,
        tau,
        config,
        epochs,
        weights=pilot_weights,
    )

    refined_q = FAIRQuantileNN(
        fold.x_train.shape[1],
        factors_train.shape[1],
        throughput_dim,
        config,
    ).to(device)
    refined_q.load_state_dict(pilot_q.state_dict())
    weights_t = to_tensor(pilot_weights.reshape(-1, 1), device)
    refined_loader = DataLoader(
        TensorDataset(f_train_t, x_train_t, y_train_t, weights_t),
        batch_size=config.batch_size,
        shuffle=True,
    )
    train_model(
        refined_q,
        refined_loader,
        lambda f, x, y, w: weighted_quantile_loss(refined_q(f, x), y, w, tau)
        + gamma_penalty(refined_q, config.gamma_penalty_eps, penalty),
        epochs,
        config.learning_rate,
    )
    refined_q.eval()
    with torch.no_grad():
        q_train = refined_q(f_train_t, x_train_t).cpu().numpy().ravel()
        q_eval = refined_q(f_eval_t, x_eval_t).cpu().numpy().ravel()
        shared_train = refined_q.features(f_train_t, x_train_t).cpu().numpy()
        shared_eval = refined_q.features(f_eval_t, x_eval_t).cpu().numpy()
        gamma = refined_q.gamma.cpu().numpy().copy() if refined_q.gamma is not None else None
    z_train = upper_tail_pseudo_outcome(q_train, fold.y_train, tau)
    refined_es_train, _ = fit_es_network(shared_train, shared_eval, z_train, tau, config, epochs)
    refined_variance_train, refined_variance_eval = fit_variance_network(
        shared_train,
        shared_eval,
        (tail_probability * refined_es_train - z_train) ** 2,
        config,
        epochs,
    )
    refined_floor = resolve_c_h(config, refined_variance_train, fold.response_scale)
    refined_weights = adaptive_weights(
        config,
        refined_variance_train,
        fold.response_scale,
        variance_floor=refined_floor,
    )
    refined_weights_eval = adaptive_weights(
        config,
        refined_variance_eval,
        fold.response_scale,
        variance_floor=refined_floor,
    )
    es_train, es_eval = fit_es_network(
        shared_train,
        shared_eval,
        z_train,
        tau,
        config,
        epochs,
        weights=refined_weights,
    )
    return MethodResult(
        "FAIR-NN",
        q_train,
        es_train,
        q_eval,
        es_eval,
        z_train,
        gamma=gamma,
        variance_train=refined_variance_train,
        variance_eval=refined_variance_eval,
        weight_train=refined_weights,
        weight_eval=refined_weights_eval,
    )


def fit_rlr(
    fold: FoldData,
    tau: float,
    config: ExperimentConfig,
    seed: int,
) -> MethodResult:
    require_torch()
    set_seed(seed)
    device = resolve_device(config)
    x_train_t = to_tensor(fold.x_train, device)
    x_eval_t = to_tensor(fold.x_eval, device)
    y_train_t = to_tensor(fold.y_train.reshape(-1, 1), device)
    loader = DataLoader(TensorDataset(x_train_t, y_train_t), batch_size=config.batch_size, shuffle=True)
    base_penalty = config.lasso_lambda
    if base_penalty is None:
        base_penalty = math.sqrt(math.log(max(fold.x_train.shape[1], 2)) / len(fold.y_train))
    quantile_penalty = base_penalty / fold.response_scale
    es_penalty = base_penalty / fold.response_scale**2

    q_model = LinearLassoNN(fold.x_train.shape[1]).to(device)
    train_model(
        q_model,
        loader,
        lambda x, y: quantile_loss(q_model(x), y, tau) + quantile_penalty * q_model.l1_penalty(),
        config.rlr_epochs,
        config.learning_rate,
    )
    q_model.eval()
    with torch.no_grad():
        q_train = q_model(x_train_t).cpu().numpy().ravel()
        q_eval = q_model(x_eval_t).cpu().numpy().ravel()
    z_train = upper_tail_pseudo_outcome(q_train, fold.y_train, tau)
    z_train_t = to_tensor(z_train.reshape(-1, 1), device)
    es_loader = DataLoader(TensorDataset(x_train_t, z_train_t), batch_size=config.batch_size, shuffle=True)
    es_model = LinearLassoNN(fold.x_train.shape[1]).to(device)
    tail_probability = 1.0 - tau
    train_model(
        es_model,
        es_loader,
        lambda x, z: torch.mean((tail_probability * es_model(x) - z) ** 2)
        + tail_probability * es_penalty * es_model.l1_penalty(),
        config.rlr_epochs,
        config.learning_rate,
    )
    es_model.eval()
    with torch.no_grad():
        es_train = es_model(x_train_t).cpu().numpy().ravel()
        es_eval = es_model(x_eval_t).cpu().numpy().ravel()
    return MethodResult("RLR", q_train, es_train, q_eval, es_eval, z_train)


def fit_krr(
    fold: FoldData,
    tau: float,
    config: ExperimentConfig,
    kernel_name: str,
    seed: int,
) -> MethodResult:
    """Fit the common quantile stage and a configured KRR ES stage."""

    require_sklearn()
    gamma = config.krr_gamma
    if gamma is None or str(gamma).lower() == "auto":
        gamma = 1.0 / fold.x_train.shape[1]
    kernel, kernel_params = sklearn_krr_kernel(config, kernel_name, float(gamma))
    q_model = GradientBoostingRegressor(
        loss="quantile",
        alpha=tau,
        n_estimators=300,
        max_depth=2,
        learning_rate=0.03,
        random_state=seed,
    )
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        q_model.fit(fold.x_train, fold.y_train)
        q_train = q_model.predict(fold.x_train)
        q_eval = q_model.predict(fold.x_eval)
        z_train = upper_tail_pseudo_outcome(q_train, fold.y_train, tau)
        es_model = KernelRidge(alpha=config.krr_alpha, kernel=kernel, **kernel_params)
        es_model.fit(fold.x_train, z_train)
        tail_probability = 1.0 - tau
        es_train = es_model.predict(fold.x_train) / tail_probability
        es_eval = es_model.predict(fold.x_eval) / tail_probability
    if not all(np.isfinite(values).all() for values in (q_train, q_eval, es_train, es_eval)):
        raise FloatingPointError("KRR produced non-finite quantile or ES predictions")
    return MethodResult(f"KRR-{kernel_name}", q_train, es_train, q_eval, es_eval, z_train)


def krr_kernel_candidates(config: ExperimentConfig) -> list[str]:
    """Resolve the same KRR kernel choices accepted by the NHANES runner."""

    candidates = config.krr_kernel_candidates or [config.krr_kernel]
    if isinstance(candidates, str):
        candidates = [candidates]
    resolved: list[str] = []
    for candidate in candidates:
        kernel = str(candidate).strip().lower()
        if kernel == "auto":
            resolved.extend(["rbf", "gaussian", "polynomial"])
        else:
            resolved.append(kernel)
    resolved = list(dict.fromkeys(resolved))
    valid = {"rbf", "gaussian", "polynomial", "poly", "linear"}
    invalid = sorted(set(resolved).difference(valid))
    if invalid:
        raise ValueError(
            "Unsupported KRR kernel(s): {}. Supported: auto, rbf, gaussian, "
            "polynomial, linear.".format(", ".join(invalid))
        )
    return resolved


def sklearn_krr_kernel(
    config: ExperimentConfig,
    kernel_name: str,
    gamma: float,
) -> tuple[str, dict[str, float | int]]:
    """Map configured KRR names to scikit-learn KernelRidge arguments."""

    kernel = str(kernel_name).strip().lower()
    if kernel in {"rbf", "gaussian"}:
        return "rbf", {"gamma": gamma}
    if kernel in {"polynomial", "poly"}:
        return "poly", {
            "degree": int(config.krr_degree),
            "gamma": 1.0,
            "coef0": float(config.krr_coef0),
        }
    if kernel == "linear":
        return "linear", {}
    raise ValueError(f"Unsupported KRR kernel: {kernel_name}")


def ordered_result_methods(methods: Iterable[str]) -> list[str]:
    """Return a stable display order, including configured KRR variants."""

    available = set(methods)
    order = ["RLR"]
    krr_rank = {
        "KRR-rbf": 0,
        "KRR-gaussian": 1,
        "KRR-polynomial": 2,
        "KRR-poly": 2,
        "KRR-linear": 3,
    }
    order.extend(
        sorted(
            (method for method in available if method.startswith("KRR-")),
            key=lambda method: (krr_rank.get(method, 99), method),
        )
    )
    order.extend(["FA-NN", "FAIR-NN"])
    return [method for method in order if method in available]


def fit_factor_method(
    method: str,
    fold: FoldData,
    tau: float,
    candidate: dict[str, Any],
    config: ExperimentConfig,
    epochs: int,
    seed: int,
) -> MethodResult:
    if method == "FA-NN":
        return fit_fa_nn(
            fold,
            tau,
            int(candidate["factor_dim"]),
            int(candidate["throughput_dim"]),
            candidate["lambda_gamma"],
            config,
            epochs,
            seed,
        )
    if method == "FAIR-NN":
        return fit_fair_nn(
            fold,
            tau,
            int(candidate["factor_dim"]),
            int(candidate["throughput_dim"]),
            candidate["lambda_gamma"],
            config,
            epochs,
            seed,
        )
    raise ValueError(f"{method} is not a factor-network method")


def factor_candidates(config: ExperimentConfig) -> list[dict[str, Any]]:
    candidates = [
        {
            "factor_dim": int(factor_dim),
            "throughput_dim": int(throughput_dim),
            "lambda_gamma": None if value is None else float(value),
        }
        for factor_dim, throughput_dim, value in itertools.product(
            config.factor_dims,
            config.throughput_dims,
            config.lambda_gamma_candidates,
        )
    ]
    if not candidates:
        raise ValueError("At least one factor/throughput configuration is required")
    if any(item["factor_dim"] < 1 or item["throughput_dim"] < 0 for item in candidates):
        raise ValueError("factor_dims must be positive and throughput_dims must be nonnegative")
    return candidates


def restore(values: np.ndarray, fold: FoldData) -> np.ndarray:
    return values * fold.response_scale + fold.response_mean


def prediction_frame(
    result: MethodResult,
    fold: FoldData,
    tau: float,
) -> pd.DataFrame:
    q_hat = restore(result.q_eval, fold)
    es_hat = restore(result.es_eval, fold)
    pseudo_outcome = upper_tail_pseudo_outcome(q_hat, fold.y_eval_raw, tau)
    score = upper_tail_calibration_score(q_hat, es_hat, fold.y_eval_raw, tau)
    frame = pd.DataFrame(
        {
            "split": "test",
            "sample_id": fold.eval_ids,
            "method": result.method,
            "tau": tau,
            "fat_pct": fold.y_eval_raw,
            "q_hat": q_hat,
            "es_hat": es_hat,
            "pseudo_outcome": pseudo_outcome,
            "calibration_score": score,
        }
    )
    if result.variance_eval is not None:
        frame["conditional_variance_hat"] = result.variance_eval * fold.response_scale**2
    if result.weight_eval is not None:
        frame["inverse_variance_weight_hat"] = result.weight_eval / fold.response_scale**2
    return frame


def metrics_from_predictions(predictions: pd.DataFrame) -> dict[str, float | int]:
    tau = float(predictions["tau"].iloc[0])
    y = predictions["fat_pct"].to_numpy(dtype=float)
    q = predictions["q_hat"].to_numpy(dtype=float)
    es = predictions["es_hat"].to_numpy(dtype=float)
    pseudo_outcome = predictions["pseudo_outcome"].to_numpy(dtype=float)
    exceedance = y > q
    score = upper_tail_calibration_score(q, es, y, tau)
    metrics: dict[str, float | int] = {
        "qce": float(abs(exceedance.mean() - (1.0 - tau))),
        "ce": float(abs(score.mean())),
        "pseudo_outcome_loss": float(np.mean((pseudo_outcome - (1.0 - tau) * es) ** 2)),
        "exceedance_rate": float(exceedance.mean()),
        "n_exceedances": int(exceedance.sum()),
        "n": int(len(predictions)),
        "q_hat_variance": float(np.var(q, ddof=1)),
        "es_hat_variance": float(np.var(es, ddof=1)),
        "pseudo_outcome_variance": float(np.var(pseudo_outcome, ddof=1)),
        "calibration_score_variance": float(np.var(score, ddof=1)),
    }
    if "conditional_variance_hat" in predictions:
        conditional_variance = predictions["conditional_variance_hat"].to_numpy(dtype=float)
        finite = conditional_variance[np.isfinite(conditional_variance)]
        metrics["conditional_variance_hat_mean"] = (
            float(finite.mean()) if finite.size else float("nan")
        )
        metrics["conditional_variance_hat_variance"] = (
            float(np.var(finite, ddof=1)) if finite.size > 1 else float("nan")
        )
    return metrics


def throughput_importance_frame(
    gamma: np.ndarray,
    feature_names: list[str],
    tau: float,
) -> pd.DataFrame:
    """Return final FAIR-NN throughput coefficients and row-wise importance."""

    if gamma.shape[0] != len(feature_names):
        raise ValueError("The throughput matrix does not match the feature names")
    frame = pd.DataFrame(
        {
            "tau": tau,
            "channel": feature_names,
            "importance": np.linalg.norm(gamma, axis=1),
        }
    )
    for index in range(gamma.shape[1]):
        frame[f"throughput_{index + 1}"] = gamma[:, index]
    frame["importance_rank"] = frame["importance"].rank(
        method="first",
        ascending=False,
    ).astype(int)
    return frame.sort_values("importance_rank").reset_index(drop=True)


def write_metrics_latex(metrics: pd.DataFrame, path: Path, config: ExperimentConfig) -> None:
    taus = sorted(metrics["tau"].unique())
    best = {
        (tau, metric): float(
            metrics.loc[np.isclose(metrics["tau"], tau), metric].min()
        )
        for tau in taus
        for metric in ("qce", "ce", "pseudo_outcome_loss")
    }
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\small",
        f"\\caption{{Held-out Tecator upper-tail conditional fat ES results. Models are fitted on C and M ($n={config.train_size}$) and evaluated on T ($n={config.test_size}$). Smaller values are better; boldface denotes the lowest value at each tail level and criterion.}}",
        "\\label{tab:tecator-upper-tail-es}",
        "\\setlength{\\tabcolsep}{3.5pt}",
        "\\renewcommand{\\arraystretch}{1.08}",
        "\\begin{tabular}{l" + "c" * (3 * len(taus)) + "}",
        "\\hline",
        "Method & "
        + " & ".join(
            f"\\multicolumn{{3}}{{c}}{{${tau:.2f}$}}" for tau in taus
        )
        + " \\\\",
        "\\cline{2-" + str(1 + 3 * len(taus)) + "}",
        " & " + " & ".join(["QCE & CE & PO loss"] * len(taus)) + " \\\\",
        "\\hline",
    ]
    for method in ordered_result_methods(metrics["method"].unique()):
        cells = []
        for tau in taus:
            row = metrics.loc[
                (metrics["method"].eq(method)) & np.isclose(metrics["tau"], tau)
            ]
            if row.empty:
                cells.extend(["--", "--", "--"])
                continue
            value = row.iloc[0]
            for metric in ("qce", "ce", "pseudo_outcome_loss"):
                display = f"{value[metric]:.3f}"
                if np.isclose(value[metric], best[(tau, metric)]):
                    display = f"\\textbf{{{display}}}"
                cells.append(display)
        lines.append(method + " & " + " & ".join(cells) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def validate_config(config: ExperimentConfig) -> None:
    methods = normalize_methods(config.methods)
    if not methods:
        raise ValueError("methods cannot be empty")
    if not config.taus or any(not 0.5 < float(tau) < 1.0 for tau in config.taus):
        raise ValueError("taus must contain upper-tail levels strictly between 0.5 and 1")
    if len(set(float(tau) for tau in config.taus)) != len(config.taus):
        raise ValueError("taus must not contain duplicates")
    if len(factor_candidates(config)) != 1:
        raise ValueError(
            "The fixed train/test design has no monitoring set for tuning; "
            "factor_dims, throughput_dims, and lambda_gamma_candidates must define one setting."
        )
    if "KRR" in methods:
        krr_kernel_candidates(config)


def run_audit(data: TecatorData, config: ExperimentConfig, output_dir: Path) -> None:
    validate_config(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_idx, test_idx = fixed_split(data, config)
    manifest = {
        "data_path": config.data_path,
        "data_sha256": data.source_sha256,
        "n": int(len(data.y_raw)),
        "p": int(data.x_raw.shape[1]),
        "target": "fat_pct",
        "excluded_sample_ids": [103, 105],
        "response_range": {
            "min": float(data.y_raw.min()),
            "max": float(data.y_raw.max()),
            "mean": float(data.y_raw.mean()),
        },
        "source_partition_counts": {
            str(name): int(count)
            for name, count in pd.Series(data.source_partitions).value_counts().items()
        },
        "fixed_split": {
            "train_partitions": config.train_partitions,
            "test_partitions": config.test_partitions,
            "train_size": int(len(train_idx)),
            "test_size": int(len(test_idx)),
        },
        "config": asdict(config),
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def run_experiment(data: TecatorData, config: ExperimentConfig, output_dir: Path) -> None:
    """Fit all requested methods and write only report-ready result artifacts."""

    validate_config(config)
    require_torch()
    require_sklearn()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_audit(data, config, output_dir)
    methods = normalize_methods(config.methods)
    train_idx, test_idx = fixed_split(data, config)
    train_test_data = prepare_fold(data, train_idx, test_idx, config)
    candidate = factor_candidates(config)[0]
    metric_rows: list[dict[str, Any]] = []
    all_predictions: list[pd.DataFrame] = []
    throughput_frames: list[pd.DataFrame] = []
    for tau_position, tau in enumerate(sorted(float(value) for value in config.taus)):
        fit_seed = config.seed + 100_000 * tau_position
        results: dict[str, MethodResult] = {}
        if "RLR" in methods:
            results["RLR"] = fit_rlr(train_test_data, tau, config, fit_seed + 10)
        if "KRR" in methods:
            for kernel_name in krr_kernel_candidates(config):
                result = fit_krr(
                    train_test_data,
                    tau,
                    config,
                    kernel_name,
                    fit_seed + 20,
                )
                results[result.method] = result
        for offset, method in enumerate(("FA-NN", "FAIR-NN")):
            if method in methods:
                results[method] = fit_factor_method(
                    method,
                    train_test_data,
                    tau,
                    candidate,
                    config,
                    config.epochs,
                    fit_seed + 100 + offset,
                )
        for result in results.values():
            frame = prediction_frame(result, train_test_data, tau)
            all_predictions.append(frame)
            metric_rows.append(
                {
                    "method": result.method,
                    "tau": tau,
                    **metrics_from_predictions(frame),
                }
            )
            if result.method == "FAIR-NN" and result.gamma is not None:
                throughput_frames.append(
                    throughput_importance_frame(
                        result.gamma,
                        train_test_data.feature_names,
                        tau,
                    )
                )

    metrics = pd.DataFrame(metric_rows)
    method_rank = {
        method: rank for rank, method in enumerate(ordered_result_methods(metrics["method"]))
    }
    metrics = metrics.sort_values(
        ["tau", "method"],
        key=lambda series: series.map(method_rank) if series.name == "method" else series,
    ).reset_index(drop=True)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    write_metrics_latex(metrics, output_dir / "metrics.tex", config)
    pd.concat(all_predictions, ignore_index=True).to_csv(
        output_dir / "test_predictions.csv",
        index=False,
    )
    if throughput_frames:
        pd.concat(throughput_frames, ignore_index=True).to_csv(
            output_dir / "fair_nn_throughput_importance.csv",
            index=False,
        )

    manifest_path = output_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["fixed_factor_throughput_configuration"] = candidate
    manifest["result_files"] = [
        "metrics.csv",
        "metrics.tex",
        "test_predictions.csv",
        "run_manifest.json",
    ]
    if throughput_frames:
        manifest["result_files"].append("fair_nn_throughput_importance.csv")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved Tecator results to {output_dir.resolve()}")
