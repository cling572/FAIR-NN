"""Audited NHANES cotinine rerun for upper-tail FAIR-NN analysis.

The module follows the current FAIR-NN paper's representation:

* The 24 audited quantitative variables form the continuous factor block.
  They are standardized with training-sample moments and only this block is
  passed to PCA.
* The 449 binary/dummy variables bypass PCA and remain in the full input for
  FAIR-NN's sparse throughput.
* The training target is upper-tail conditional quantile/ES at ``tau``.
  Cotinine is standardized using training-sample moments for numerical
  optimization and restored to its original scale for all reported results.

Commands are invoked through ``run_from_config.py``:

    python3 run_from_config.py audit conf/config.yaml
    python3 run_from_config.py train conf/config.yaml

``audit`` has only NumPy/Pandas dependencies. ``train`` additionally requires
PyTorch and scikit-learn.
"""

from __future__ import annotations

import itertools
import json
import math
import random
from dataclasses import asdict, dataclass, field, replace
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


DEFAULT_METHODS = ["RLR", "KRR", "FA-NN", "FAIR-NN"]
ALL_METHODS = set(DEFAULT_METHODS)
_TorchModule = nn.Module if nn is not None else object


@dataclass
class NHANESConfig:
    """Configuration for the audited baseline NHANES application."""

    data_path: str = "data/design_matrix_new.csv"
    audit_path: str = "data/data_audit.csv"
    output_dir: str = "results/baseline_upper_tail"
    target_col: str = "cotinine"
    response_standardize: bool = True
    excluded_race_indicator: str = "raceNA"
    group_columns: list[str] = field(default_factory=lambda: ["raceA", "raceB", "raceM"])
    group_names: list[str] = field(default_factory=lambda: ["Asian", "Black", "Hispanic"])
    white_group_name: str = "White"
    analysis_group: str = "all"
    methods: list[str] = field(default_factory=lambda: DEFAULT_METHODS.copy())
    tau: float = 0.90
    tau_values: list[float] = field(default_factory=lambda: [0.70, 0.80, 0.90])
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    factor_dim_candidates: list[int] = field(default_factory=lambda: [4, 8])
    throughput_dim: int | None = None
    throughput_dim_candidates: list[int] = field(default_factory=lambda: [4, 8])
    lambda_gamma_candidates: list[float | None] = field(default_factory=lambda: [None])
    tuning_epochs: int = 100
    truncation: float = 3.0
    gamma_penalty_eps: float = 5e-3
    selected_top_k: int = 30
    hidden_dim: int = 64
    hidden_layers: int = 2
    epochs: int = 350
    hdes_epochs: int = 500
    batch_size: int = 64
    learning_rate: float = 1e-3
    lasso_lambda: float | None = None
    krr_alpha: float = 1.0
    krr_kernel: str = "rbf"
    krr_kernel_candidates: list[str] = field(default_factory=list)
    krr_gamma: float | str | None = "auto"
    krr_degree: int = 3
    krr_coef0: float = 1.0
    c_h: float | None = None
    c_h_quantile: float = 0.10
    c_h_scale: float = 0.50
    plot: bool = True
    seed: int = 2026
    device: str = "auto"


@dataclass
class PreparedData:
    """Data after audit-enforced filtering, splitting, and block preprocessing."""

    source_index: np.ndarray
    groups: np.ndarray
    analysis_group: str
    y_raw: np.ndarray
    y: np.ndarray
    response_train_mean: float
    response_train_scale: float
    train_idx: np.ndarray
    validation_idx: np.ndarray
    test_idx: np.ndarray
    x_continuous: np.ndarray
    x_discrete: np.ndarray
    x_full: np.ndarray
    continuous_names: list[str]
    discrete_names: list[str]
    full_feature_names: list[str]
    removed_constant_features: list[str]
    continuous_mean: np.ndarray
    continuous_scale: np.ndarray


@dataclass
class MethodResult:
    """Fitted model outputs for either validation or test evaluation."""

    method: str
    q_train: np.ndarray
    es_train: np.ndarray
    q_eval: np.ndarray
    es_eval: np.ndarray
    z_train: np.ndarray
    factor_dim: int | None = None
    throughput_dim: int | None = None
    lambda_gamma: float | None = None
    gamma: np.ndarray | None = None
    train_weights: np.ndarray | None = None


class FeedForwardNN(_TorchModule):
    """Compact MLP used for quantile, ES, and variance regressions."""

    def __init__(self, input_dim: int, hidden_dim: int, hidden_layers: int, positive_output: bool = False):
        require_torch()
        super().__init__()
        layers: list[Any] = []
        current_dim = input_dim
        for _ in range(hidden_layers):
            layers.extend([nn.Linear(current_dim, hidden_dim), nn.ReLU()])
            current_dim = hidden_dim
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
    """Learned sparse map from the full covariate vector to N throughput units."""

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
        if self.gamma is None or self.throughput_dim == 0:
            return factors
        throughput = torch.clamp(x_full @ self.gamma, -self.truncation, self.truncation)
        return torch.cat([factors, throughput], dim=1)


class FAIRQuantileNN(_TorchModule):
    """Factor-augmented quantile NN with a trainable sparse throughput."""

    def __init__(
        self,
        p: int,
        factor_dim: int,
        throughput_dim: int,
        truncation: float,
        config: NHANESConfig,
    ):
        require_torch()
        super().__init__()
        self.throughput = SparseThroughput(p, throughput_dim, truncation)
        self.net = FeedForwardNN(
            factor_dim + throughput_dim,
            config.hidden_dim,
            config.hidden_layers,
        )

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
            "PyTorch is required for RLR, FA-NN, and FAIR-NN. "
            "Install torch in the execution environment before running train."
        ) from _TORCH_IMPORT_ERROR


def require_sklearn() -> None:
    if PCA is None or GradientBoostingRegressor is None or KernelRidge is None:
        raise ImportError(
            "scikit-learn is required for PCA and KRR. "
            "Install scikit-learn in the execution environment before running train."
        ) from _SKLEARN_IMPORT_ERROR


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def resolve_device(config: NHANESConfig) -> Any:
    require_torch()
    if config.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(config.device)


def normalize_methods(methods: Iterable[str]) -> list[str]:
    aliases = {
        "RLR": "RLR",
        "HDES": "RLR",
        "KRR": "KRR",
        "FA-NN": "FA-NN",
        "FAN": "FA-NN",
        "FAIR-NN": "FAIR-NN",
        "FAIRNN": "FAIR-NN",
        "FAST-FNR": "FA-NN",
        "FAST-RWFNR": "FAIR-NN",
    }
    normalized = []
    for method in methods:
        key = str(method).upper()
        if key not in aliases:
            raise ValueError(f"Unknown method: {method}. Allowed methods: {sorted(ALL_METHODS)}")
        normalized.append(aliases[key])
    return list(dict.fromkeys(normalized))


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


def restore_response_scale(values: np.ndarray, data: PreparedData) -> np.ndarray:
    """Map model-scale quantile or ES predictions back to cotinine units."""

    return np.asarray(values, dtype=float) * data.response_train_scale + data.response_train_mean


def audit_boolean(series: pd.Series) -> pd.Series:
    """Read a bool column robustly whether CSV parsing yields bool or strings."""
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1"})


def group_labels(df: pd.DataFrame, config: NHANESConfig) -> np.ndarray:
    if len(config.group_columns) != len(config.group_names):
        raise ValueError("group_columns and group_names must have the same length")
    labels = np.full(len(df), "", dtype=object)
    for column, name in zip(config.group_columns, config.group_names):
        if column not in df.columns:
            raise KeyError(f"Missing group indicator: {column}")
        labels[df[column].to_numpy(dtype=int) == 1] = name
    race_sum = df[config.group_columns].sum(axis=1).to_numpy()
    labels[race_sum == 0] = config.white_group_name
    if (labels == "").any():
        raise ValueError("Could not assign every retained row to an analytic group")
    return labels


def resolve_analysis_group(config: NHANESConfig) -> str:
    """Normalize and validate the requested pooled or single-group analysis."""

    requested = str(config.analysis_group).strip()
    choices = ["all", *config.group_names, config.white_group_name]
    lookup = {choice.casefold(): choice for choice in choices}
    if requested.casefold() not in lookup:
        raise ValueError(
            "analysis_group must be one of: {}".format(", ".join(choices))
        )
    return lookup[requested.casefold()]


def analysis_output_dir(config: NHANESConfig) -> Path:
    """Return a group-specific output root without changing pooled outputs."""

    output_dir = Path(config.output_dir)
    analysis_group = resolve_analysis_group(config)
    if analysis_group == "all":
        return output_dir
    group_slug = analysis_group.casefold()
    if group_slug in output_dir.parts:
        return output_dir
    return output_dir / group_slug


def stratified_split(groups: np.ndarray, config: NHANESConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fractions = config.train_fraction + config.validation_fraction + config.test_fraction
    if not np.isclose(fractions, 1.0):
        raise ValueError("train_fraction + validation_fraction + test_fraction must equal 1")
    if min(config.train_fraction, config.validation_fraction, config.test_fraction) <= 0.0:
        raise ValueError("All split fractions must be positive")

    rng = np.random.default_rng(config.seed)
    train, validation, test = [], [], []
    for group in sorted(np.unique(groups)):
        indices = np.flatnonzero(groups == group)
        if len(indices) < 10:
            raise ValueError(f"Group {group} is too small for a three-way split")
        shuffled = rng.permutation(indices)
        n_test = max(1, int(round(len(indices) * config.test_fraction)))
        n_validation = max(1, int(round(len(indices) * config.validation_fraction)))
        n_train = len(indices) - n_test - n_validation
        if n_train < 1:
            raise ValueError(f"Group {group} leaves no training observations")
        train.extend(shuffled[:n_train])
        validation.extend(shuffled[n_train : n_train + n_validation])
        test.extend(shuffled[n_train + n_validation :])

    return (
        np.sort(np.asarray(train, dtype=int)),
        np.sort(np.asarray(validation, dtype=int)),
        np.sort(np.asarray(test, dtype=int)),
    )


def load_prepared_data(config: NHANESConfig) -> PreparedData:
    data_path = Path(config.data_path)
    audit_path = Path(config.audit_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Cannot find data_path: {data_path}")
    if not audit_path.exists():
        raise FileNotFoundError(f"Cannot find audit_path: {audit_path}")

    df = pd.read_csv(data_path)
    audit = pd.read_csv(audit_path)
    required_audit = {
        "source_variable",
        "baseline_included",
        "baseline_block",
        "source_role",
    }
    missing = required_audit.difference(audit.columns)
    if missing:
        raise ValueError(f"Audit manifest is missing required columns: {sorted(missing)}")
    if config.target_col not in df.columns:
        raise KeyError(f"Missing target column: {config.target_col}")
    if config.excluded_race_indicator not in df.columns:
        raise KeyError(f"Missing exclusion indicator: {config.excluded_race_indicator}")

    included_mask = audit_boolean(audit["baseline_included"])
    continuous_names = audit.loc[
        included_mask & audit["baseline_block"].eq("continuous_factor_candidate"),
        "source_variable",
    ].tolist()
    discrete_names = audit.loc[
        included_mask & audit["baseline_block"].eq("sparse_throughput"),
        "source_variable",
    ].tolist()
    full_feature_names = continuous_names + discrete_names
    if not continuous_names or not discrete_names:
        raise ValueError("The audit must contain both continuous and discrete baseline blocks")
    if len(full_feature_names) != len(set(full_feature_names)):
        raise ValueError("Audit baseline feature names are not unique")
    if set(full_feature_names).difference(df.columns):
        raise KeyError("Some baseline audit predictors are missing from the data matrix")

    retained = df.loc[df[config.excluded_race_indicator].eq(0)].copy()
    if retained[full_feature_names + [config.target_col]].isna().any().any():
        raise ValueError("The audited baseline cohort contains missing model inputs")
    if config.excluded_race_indicator in full_feature_names:
        raise ValueError("The excluded race indicator must not enter the baseline input")

    analysis_group = resolve_analysis_group(config)
    groups = group_labels(retained, config)
    removed_constant_features: list[str] = []
    if analysis_group != "all":
        selected = groups == analysis_group
        if not selected.any():
            raise ValueError(f"No retained observations found for analysis_group={analysis_group}")
        retained = retained.loc[selected].copy()
        groups = groups[selected]

        removed_constant_features = [
            feature
            for feature in full_feature_names
            if retained[feature].nunique(dropna=False) <= 1
        ]
        continuous_names = [
            feature for feature in continuous_names if feature not in removed_constant_features
        ]
        discrete_names = [
            feature for feature in discrete_names if feature not in removed_constant_features
        ]
        full_feature_names = continuous_names + discrete_names
        if not continuous_names or not discrete_names:
            raise ValueError(
                "Single-group filtering removed an entire baseline feature block"
            )

    train_idx, validation_idx, test_idx = stratified_split(groups, config)

    y_raw = retained[config.target_col].to_numpy(dtype=float)
    response_train_mean = 0.0
    response_train_scale = 1.0
    if config.response_standardize:
        response_train_mean = float(y_raw[train_idx].mean())
        response_train_scale = float(y_raw[train_idx].std(ddof=0))
        if response_train_scale <= 1e-12:
            response_train_scale = 1.0
    y = (y_raw - response_train_mean) / response_train_scale
    x_cont_raw = retained[continuous_names].to_numpy(dtype=float)
    x_discrete = retained[discrete_names].to_numpy(dtype=float)
    mean = x_cont_raw[train_idx].mean(axis=0)
    scale = x_cont_raw[train_idx].std(axis=0, ddof=0)
    scale[scale <= 1e-12] = 1.0
    x_cont = (x_cont_raw - mean) / scale
    x_full = np.column_stack([x_cont, x_discrete])

    # #region debug-point E:response-profile
    exec("try:\n import urllib.request\n _p='.dbg/fairnn-underperformance.env'; _u,_s='http://127.0.0.1:7777/event','fairnn-underperformance'\n try:\n  with open(_p) as _f: _c=_f.read(); _u=next((l.split('=',1)[1] for l in _c.split('\\n') if l.startswith('DEBUG_SERVER_URL=')),_u); _s=next((l.split('=',1)[1] for l in _c.split('\\n') if l.startswith('DEBUG_SESSION_ID=')),_s)\n except: pass\n urllib.request.urlopen(urllib.request.Request(_u,data=json.dumps({'sessionId':_s,'runId':'pre-fix','hypothesisId':'E','location':'simulation.py:load_prepared_data','msg':'[DEBUG] Audited target profile','data':{'analysis_group':analysis_group,'response_standardize':bool(config.response_standardize),'response_train_mean':response_train_mean,'response_train_scale':response_train_scale,'n':int(len(y_raw)),'p':int(x_full.shape[1]),'y_q50':float(np.quantile(y_raw,0.50)),'y_q90':float(np.quantile(y_raw,0.90)),'y_q99':float(np.quantile(y_raw,0.99)),'y_max':float(np.max(y_raw))}}).encode(),headers={'Content-Type':'application/json'}),timeout=1).read()\nexcept: pass")
    # #endregion

    return PreparedData(
        source_index=retained.index.to_numpy(dtype=int),
        groups=groups,
        analysis_group=analysis_group,
        y_raw=y_raw,
        y=y,
        response_train_mean=response_train_mean,
        response_train_scale=response_train_scale,
        train_idx=train_idx,
        validation_idx=validation_idx,
        test_idx=test_idx,
        x_continuous=x_cont,
        x_discrete=x_discrete,
        x_full=x_full,
        continuous_names=continuous_names,
        discrete_names=discrete_names,
        full_feature_names=full_feature_names,
        removed_constant_features=removed_constant_features,
        continuous_mean=mean,
        continuous_scale=scale,
    )


def split_name(data: PreparedData) -> np.ndarray:
    names = np.full(len(data.y), "", dtype=object)
    names[data.train_idx] = "train"
    names[data.validation_idx] = "validation"
    names[data.test_idx] = "test"
    return names


def write_audit_outputs(data: PreparedData, config: NHANESConfig) -> Path:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_frame = pd.DataFrame(
        {
            "source_row": data.source_index,
            "group": data.groups,
            "split": split_name(data),
            "cotinine": data.y_raw,
        }
    )
    split_frame.to_csv(output_dir / "split_assignments.csv", index=False)

    group_counts = split_frame.pivot_table(
        index="group",
        columns="split",
        values="source_row",
        aggfunc="count",
        fill_value=0,
    )
    summary = {
        "data_path": config.data_path,
        "audit_path": config.audit_path,
        "target": config.target_col,
        "analysis_group": data.analysis_group,
        "response_standardize": config.response_standardize,
        "response_train_mean": data.response_train_mean,
        "response_train_scale": data.response_train_scale,
        "tau": config.tau,
        "n": int(len(data.y)),
        "p": int(data.x_full.shape[1]),
        "continuous_factor_dimension": int(data.x_continuous.shape[1]),
        "discrete_throughput_dimension": int(data.x_discrete.shape[1]),
        "n_train": int(len(data.train_idx)),
        "n_validation": int(len(data.validation_idx)),
        "n_test": int(len(data.test_idx)),
        "group_split_counts": group_counts.astype(int).to_dict(),
        "continuous_features": data.continuous_names,
        "discrete_features": data.discrete_names,
        "removed_constant_features": data.removed_constant_features,
        "continuous_train_mean": data.continuous_mean.tolist(),
        "continuous_train_scale": data.continuous_scale.tolist(),
        "config": asdict(config),
    }
    (output_dir / "data_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output_dir


def run_audit(config: NHANESConfig) -> None:
    data = load_prepared_data(config)
    output_config = replace(config, output_dir=str(analysis_output_dir(config)))
    output_dir = write_audit_outputs(data, output_config)
    print(
        f"Audit split saved to {output_dir.resolve()} "
        f"(n={len(data.y)}, p={data.x_full.shape[1]}, "
        f"train/validation/test={len(data.train_idx)}/{len(data.validation_idx)}/{len(data.test_idx)})"
    )


def train_model(model: Any, loader: Any, loss_fn: Any, epochs: int, learning_rate: float) -> None:
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for _ in range(epochs):
        for batch in loader:
            optimizer.zero_grad()
            loss = loss_fn(*batch)
            loss.backward()
            optimizer.step()


def to_tensor(array: np.ndarray, device: Any) -> Any:
    return torch.tensor(array, dtype=torch.float32, device=device)


def pca_features(
    data: PreparedData,
    factor_dim: int,
    eval_idx: np.ndarray,
    config: NHANESConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    require_sklearn()
    train_x = data.x_continuous[data.train_idx]
    dim = min(int(factor_dim), train_x.shape[0], train_x.shape[1])
    if dim < 1:
        raise ValueError("factor_dim must be at least one")
    pca = PCA(n_components=dim, random_state=config.seed)
    f_train = pca.fit_transform(train_x)
    f_eval = pca.transform(data.x_continuous[eval_idx])
    info = {
        "factor_dim": dim,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "explained_variance_total": float(pca.explained_variance_ratio_.sum()),
    }
    return f_train, f_eval, info


def resolve_gamma_lambda(
    p: int,
    n_train: int,
    value: float | None,
) -> float:
    """Return the sparse penalty on the original response scale."""

    if value is not None:
        return float(value)
    return 1.3 * math.log(max(p, 2)) / max(n_train, 1)


def as_int_candidates(value: Any, name: str) -> list[int]:
    """Accept either a scalar candidate or a list of candidates from YAML."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        raw_values = value
    else:
        raw_values = [value]
    candidates = [int(candidate) for candidate in raw_values]
    if not candidates:
        raise ValueError(f"{name} must contain at least one candidate")
    if min(candidates) < 0:
        raise ValueError(f"{name} must be nonnegative")
    return candidates


def factor_dim_candidates(config: NHANESConfig) -> list[int]:
    candidates = as_int_candidates(config.factor_dim_candidates, "factor_dim_candidates")
    if min(candidates) < 1:
        raise ValueError("factor_dim_candidates must be positive")
    return candidates


def throughput_dim_candidates(config: NHANESConfig) -> list[int]:
    if config.throughput_dim is not None:
        return as_int_candidates(config.throughput_dim, "throughput_dim")
    return as_int_candidates(config.throughput_dim_candidates, "throughput_dim_candidates")


def krr_kernel_candidates(config: NHANESConfig) -> list[str]:
    candidates = config.krr_kernel_candidates or [config.krr_kernel]
    if isinstance(candidates, str):
        candidates = [candidates]
    normalized = []
    for candidate in candidates:
        kernel = str(candidate).strip().lower()
        if kernel == "auto":
            normalized.extend(["rbf", "gaussian", "polynomial"])
        else:
            normalized.append(kernel)
    return list(dict.fromkeys(normalized))


def gamma_penalty(model: FAIRQuantileNN, eps: float, penalty_level: float) -> Any:
    if model.gamma is None:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    clipped_l1 = torch.clamp(model.gamma.abs() / eps, max=1.0).sum()
    return penalty_level * clipped_l1


def resolve_c_h(
    config: NHANESConfig,
    variance: np.ndarray,
    response_train_scale: float,
) -> float:
    if config.c_h is not None:
        return max(float(config.c_h) / response_train_scale**2, 1e-8)
    q = float(np.quantile(variance[np.isfinite(variance)], config.c_h_quantile))
    return max(config.c_h_scale * q, 1e-8)


def adaptive_weights(
    config: NHANESConfig,
    variance: np.ndarray,
    response_train_scale: float,
) -> np.ndarray:
    lower = resolve_c_h(config, variance, response_train_scale)
    weights = 1.0 / np.maximum(variance, lower / 2.0)
    return weights / np.mean(weights)


def fit_es_network(
    features_train: np.ndarray,
    features_eval: np.ndarray,
    z_train: np.ndarray,
    config: NHANESConfig,
    epochs: int,
    weights: np.ndarray | None = None,
) -> tuple[Any, np.ndarray, np.ndarray]:
    device = resolve_device(config)
    x_train_t = to_tensor(features_train, device)
    z_train_t = to_tensor(z_train.reshape(-1, 1), device)
    tensors = [x_train_t, z_train_t]
    if weights is not None:
        tensors.append(to_tensor(weights.reshape(-1, 1), device))
    loader = DataLoader(TensorDataset(*tensors), batch_size=config.batch_size, shuffle=True)
    model = FeedForwardNN(features_train.shape[1], config.hidden_dim, config.hidden_layers).to(device)
    tail_probability = 1.0 - config.tau

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
    return model, es_train, es_eval


def fit_variance_network(
    features_train: np.ndarray,
    squared_residual: np.ndarray,
    config: NHANESConfig,
    epochs: int,
) -> np.ndarray:
    device = resolve_device(config)
    x_train_t = to_tensor(features_train, device)
    residual_t = to_tensor(squared_residual.reshape(-1, 1), device)
    loader = DataLoader(TensorDataset(x_train_t, residual_t), batch_size=config.batch_size, shuffle=True)
    model = FeedForwardNN(
        features_train.shape[1],
        config.hidden_dim,
        config.hidden_layers,
        positive_output=True,
    ).to(device)
    smooth_l1 = nn.SmoothL1Loss()
    train_model(model, loader, lambda x, residual: smooth_l1(model(x), residual), epochs, config.learning_rate)
    model.eval()
    with torch.no_grad():
        return model(x_train_t).cpu().numpy().ravel()


def fit_fa_nn(
    data: PreparedData,
    factor_dim: int,
    throughput_dim: int,
    lambda_gamma: float | None,
    eval_idx: np.ndarray,
    config: NHANESConfig,
    epochs: int,
    seed_offset: int = 0,
) -> MethodResult:
    """Fit FA-NN (FAST-FNR): factor plus sparse-throughput, without refinement."""
    require_torch()
    set_seed(config.seed + seed_offset)
    f_train, f_eval, _ = pca_features(data, factor_dim, eval_idx, config)
    y_train = data.y[data.train_idx]
    device = resolve_device(config)
    f_train_t = to_tensor(f_train, device)
    f_eval_t = to_tensor(f_eval, device)
    x_train_t = to_tensor(data.x_full[data.train_idx], device)
    x_eval_t = to_tensor(data.x_full[eval_idx], device)
    y_train_t = to_tensor(y_train.reshape(-1, 1), device)
    lambda_gamma_raw = resolve_gamma_lambda(
        data.x_full.shape[1],
        len(y_train),
        lambda_gamma,
    )
    penalty_level = lambda_gamma_raw / data.response_train_scale
    q_model = FAIRQuantileNN(
        data.x_full.shape[1],
        f_train.shape[1],
        throughput_dim,
        config.truncation,
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
        lambda f, x, y: quantile_loss(q_model(f, x), y, config.tau)
        + gamma_penalty(q_model, config.gamma_penalty_eps, penalty_level),
        epochs,
        config.learning_rate,
    )
    q_model.eval()
    with torch.no_grad():
        q_train = q_model(f_train_t, x_train_t).cpu().numpy().ravel()
        q_eval = q_model(f_eval_t, x_eval_t).cpu().numpy().ravel()
        shared_train = q_model.features(f_train_t, x_train_t).cpu().numpy()
        shared_eval = q_model.features(f_eval_t, x_eval_t).cpu().numpy()
        gamma = q_model.gamma.cpu().numpy().copy() if q_model.gamma is not None else None
    z_train = upper_tail_pseudo_outcome(q_train, y_train, config.tau)
    _, es_train, es_eval = fit_es_network(shared_train, shared_eval, z_train, config, epochs)
    return MethodResult(
        method="FA-NN",
        q_train=q_train,
        es_train=es_train,
        q_eval=q_eval,
        es_eval=es_eval,
        z_train=z_train,
        factor_dim=f_train.shape[1],
        throughput_dim=throughput_dim,
        lambda_gamma=lambda_gamma_raw,
        gamma=gamma,
    )


def fit_fair_nn(
    data: PreparedData,
    factor_dim: int,
    throughput_dim: int,
    lambda_gamma: float | None,
    eval_idx: np.ndarray,
    config: NHANESConfig,
    epochs: int,
    seed_offset: int = 0,
) -> MethodResult:
    """Fit FAIR-NN, aligned with the reference FAST-RWFNR implementation.

    The sequence is pilot FAST quantile -> pilot ES -> pilot variance weights
    -> weighted FAST quantile refinement -> refined ES -> refined variance
    weights -> final weighted ES.  The final ES therefore uses weights estimated
    on the same refined quantile/Gamma representation that feeds the last stage.
    """
    require_torch()
    set_seed(config.seed + seed_offset)
    f_train, f_eval, _ = pca_features(data, factor_dim, eval_idx, config)
    x_train = data.x_full[data.train_idx]
    x_eval = data.x_full[eval_idx]
    y_train = data.y[data.train_idx]
    device = resolve_device(config)
    f_train_t = to_tensor(f_train, device)
    f_eval_t = to_tensor(f_eval, device)
    x_train_t = to_tensor(x_train, device)
    x_eval_t = to_tensor(x_eval, device)
    y_train_t = to_tensor(y_train.reshape(-1, 1), device)
    lambda_gamma_raw = resolve_gamma_lambda(
        x_train.shape[1],
        len(y_train),
        lambda_gamma,
    )
    penalty_level = lambda_gamma_raw / data.response_train_scale

    pilot_q = FAIRQuantileNN(
        x_train.shape[1],
        f_train.shape[1],
        throughput_dim,
        config.truncation,
        config,
    ).to(device)
    loader = DataLoader(TensorDataset(f_train_t, x_train_t, y_train_t), batch_size=config.batch_size, shuffle=True)
    train_model(
        pilot_q,
        loader,
        lambda f, x, y: quantile_loss(pilot_q(f, x), y, config.tau)
        + gamma_penalty(pilot_q, config.gamma_penalty_eps, penalty_level),
        epochs,
        config.learning_rate,
    )
    pilot_q.eval()
    with torch.no_grad():
        q_train = pilot_q(f_train_t, x_train_t).cpu().numpy().ravel()
        pilot_shared_train = pilot_q.features(f_train_t, x_train_t).cpu().numpy()
        pilot_shared_eval = pilot_q.features(f_eval_t, x_eval_t).cpu().numpy()
    z_train = upper_tail_pseudo_outcome(q_train, y_train, config.tau)
    _, pilot_es_train, _ = fit_es_network(pilot_shared_train, pilot_shared_eval, z_train, config, epochs)

    tail_probability = 1.0 - config.tau
    variance = fit_variance_network(
        pilot_shared_train,
        (tail_probability * pilot_es_train - z_train) ** 2,
        config,
        epochs,
    )
    weights = adaptive_weights(config, variance, data.response_train_scale)

    # #region debug-point A:pilot-weights
    exec("try:\n import urllib.request\n _p='.dbg/fairnn-underperformance.env'; _u,_s='http://127.0.0.1:7777/event','fairnn-underperformance'\n try:\n  with open(_p) as _f: _c=_f.read(); _u=next((l.split('=',1)[1] for l in _c.split('\\n') if l.startswith('DEBUG_SERVER_URL=')),_u); _s=next((l.split('=',1)[1] for l in _c.split('\\n') if l.startswith('DEBUG_SESSION_ID=')),_s)\n except: pass\n urllib.request.urlopen(urllib.request.Request(_u,data=json.dumps({'sessionId':_s,'runId':'pre-fix','hypothesisId':'A','location':'simulation.py:fit_fair_nn:pseudo-weights','msg':'[DEBUG] Pilot variance and adaptive-weight profile','data':{'variance_q01':float(np.quantile(variance,0.01)),'variance_q50':float(np.quantile(variance,0.50)),'variance_q99':float(np.quantile(variance,0.99)),'variance_floor':float(resolve_c_h(config,variance,data.response_train_scale)/2.0),'weight_q50':float(np.quantile(weights,0.50)),'weight_q95':float(np.quantile(weights,0.95)),'weight_q99':float(np.quantile(weights,0.99)),'weight_max':float(np.max(weights)),'weight_max_to_median':float(np.max(weights)/max(np.median(weights),1e-12))}}).encode(),headers={'Content-Type':'application/json'}),timeout=1).read()\nexcept: pass")
    # #endregion

    refined_q = FAIRQuantileNN(
        x_train.shape[1],
        f_train.shape[1],
        throughput_dim,
        config.truncation,
        config,
    ).to(device)
    refined_q.load_state_dict(pilot_q.state_dict())
    weights_t = to_tensor(weights.reshape(-1, 1), device)
    refined_loader = DataLoader(
        TensorDataset(f_train_t, x_train_t, y_train_t, weights_t),
        batch_size=config.batch_size,
        shuffle=True,
    )
    train_model(
        refined_q,
        refined_loader,
        lambda f, x, y, w: weighted_quantile_loss(refined_q(f, x), y, w, config.tau)
        + gamma_penalty(refined_q, config.gamma_penalty_eps, penalty_level),
        epochs,
        config.learning_rate,
    )
    refined_q.eval()
    with torch.no_grad():
        q_refined_train = refined_q(f_train_t, x_train_t).cpu().numpy().ravel()
        q_refined_eval = refined_q(f_eval_t, x_eval_t).cpu().numpy().ravel()
        shared_train = refined_q.features(f_train_t, x_train_t).cpu().numpy()
        shared_eval = refined_q.features(f_eval_t, x_eval_t).cpu().numpy()
        gamma = refined_q.gamma.cpu().numpy().copy() if refined_q.gamma is not None else None
    z_refined = upper_tail_pseudo_outcome(q_refined_train, y_train, config.tau)

    # #region debug-point B:refined-quantile
    exec("try:\n import urllib.request\n _p='.dbg/fairnn-underperformance.env'; _u,_s='http://127.0.0.1:7777/event','fairnn-underperformance'\n try:\n  with open(_p) as _f: _c=_f.read(); _u=next((l.split('=',1)[1] for l in _c.split('\\n') if l.startswith('DEBUG_SERVER_URL=')),_u); _s=next((l.split('=',1)[1] for l in _c.split('\\n') if l.startswith('DEBUG_SESSION_ID=')),_s)\n except: pass\n urllib.request.urlopen(urllib.request.Request(_u,data=json.dumps({'sessionId':_s,'runId':'pre-fix','hypothesisId':'B','location':'simulation.py:fit_fair_nn:refined-quantile','msg':'[DEBUG] Quantile refinement profile','data':{'tau':float(config.tau),'pilot_exceedance':float(np.mean(y_train>q_train)),'refined_exceedance':float(np.mean(y_train>q_refined_train)),'mean_abs_quantile_change':float(np.mean(np.abs(q_refined_train-q_train))),'quantile_change_q95':float(np.quantile(np.abs(q_refined_train-q_train),0.95)),'z_refined_q99':float(np.quantile(z_refined,0.99)),'z_refined_max':float(np.max(z_refined))}}).encode(),headers={'Content-Type':'application/json'}),timeout=1).read()\nexcept: pass")
    # #endregion

    # #region debug-point C:refined-throughput
    exec("try:\n import urllib.request\n _p='.dbg/fairnn-underperformance.env'; _u,_s='http://127.0.0.1:7777/event','fairnn-underperformance'\n try:\n  with open(_p) as _f: _c=_f.read(); _u=next((l.split('=',1)[1] for l in _c.split('\\n') if l.startswith('DEBUG_SERVER_URL=')),_u); _s=next((l.split('=',1)[1] for l in _c.split('\\n') if l.startswith('DEBUG_SESSION_ID=')),_s)\n except: pass\n urllib.request.urlopen(urllib.request.Request(_u,data=json.dumps({'sessionId':_s,'runId':'pre-fix','hypothesisId':'C','location':'simulation.py:fit_fair_nn:refined-throughput','msg':'[DEBUG] Refined sparse-throughput profile','data':{'gamma_l2_norm':float(np.linalg.norm(gamma)) if gamma is not None else 0.0,'gamma_abs_sum':float(np.abs(gamma).sum()) if gamma is not None else 0.0,'gamma_active_eps':int((np.abs(gamma)>config.gamma_penalty_eps).sum()) if gamma is not None else 0,'throughput_abs_q99':float(np.quantile(np.abs(shared_train[:,f_train.shape[1]:]),0.99)) if throughput_dim>0 else 0.0,'throughput_clipping_share':float(np.mean(np.abs(shared_train[:,f_train.shape[1]:])>=config.truncation-1e-6)) if throughput_dim>0 else 0.0}}).encode(),headers={'Content-Type':'application/json'}),timeout=1).read()\nexcept: pass")
    # #endregion

    _, refined_es_train, _ = fit_es_network(
        shared_train,
        shared_eval,
        z_refined,
        config,
        epochs,
    )
    refined_variance = fit_variance_network(
        shared_train,
        (tail_probability * refined_es_train - z_refined) ** 2,
        config,
        epochs,
    )
    refined_weights = adaptive_weights(
        config,
        refined_variance,
        data.response_train_scale,
    )

    # #region debug-point A:refined-weights
    exec("try:\n import urllib.request\n _p='.dbg/fairnn-underperformance.env'; _u,_s='http://127.0.0.1:7777/event','fairnn-underperformance'\n try:\n  with open(_p) as _f: _c=_f.read(); _u=next((l.split('=',1)[1] for l in _c.split('\\n') if l.startswith('DEBUG_SERVER_URL=')),_u); _s=next((l.split('=',1)[1] for l in _c.split('\\n') if l.startswith('DEBUG_SESSION_ID=')),_s)\n except: pass\n urllib.request.urlopen(urllib.request.Request(_u,data=json.dumps({'sessionId':_s,'runId':'pre-fix','hypothesisId':'A','location':'simulation.py:fit_fair_nn:refined-weights','msg':'[DEBUG] Refined adaptive-weight profile','data':{'variance_q01':float(np.quantile(refined_variance,0.01)),'variance_q50':float(np.quantile(refined_variance,0.50)),'variance_q99':float(np.quantile(refined_variance,0.99)),'variance_floor':float(resolve_c_h(config,refined_variance,data.response_train_scale)/2.0),'weight_q50':float(np.quantile(refined_weights,0.50)),'weight_q95':float(np.quantile(refined_weights,0.95)),'weight_q99':float(np.quantile(refined_weights,0.99)),'weight_max':float(np.max(refined_weights)),'weight_max_to_median':float(np.max(refined_weights)/max(np.median(refined_weights),1e-12))}}).encode(),headers={'Content-Type':'application/json'}),timeout=1).read()\nexcept: pass")
    # #endregion

    _, es_train, es_eval = fit_es_network(
        shared_train,
        shared_eval,
        z_refined,
        config,
        epochs,
        weights=refined_weights,
    )
    return MethodResult(
        method="FAIR-NN",
        q_train=q_refined_train,
        es_train=es_train,
        q_eval=q_refined_eval,
        es_eval=es_eval,
        z_train=z_refined,
        factor_dim=f_train.shape[1],
        throughput_dim=throughput_dim,
        lambda_gamma=lambda_gamma_raw,
        gamma=gamma,
        train_weights=refined_weights,
    )


def fit_rlr(
    data: PreparedData,
    eval_idx: np.ndarray,
    config: NHANESConfig,
    seed_offset: int = 0,
) -> MethodResult:
    """Fit the regularized linear quantile/ES comparator (RLR)."""
    require_torch()
    set_seed(config.seed + seed_offset)
    x_train = data.x_full[data.train_idx]
    x_eval = data.x_full[eval_idx]
    y_train = data.y[data.train_idx]
    device = resolve_device(config)
    x_train_t = to_tensor(x_train, device)
    y_train_t = to_tensor(y_train.reshape(-1, 1), device)
    loader = DataLoader(TensorDataset(x_train_t, y_train_t), batch_size=config.batch_size, shuffle=True)
    base_penalty = config.lasso_lambda
    if base_penalty is None:
        base_penalty = math.sqrt(math.log(max(x_train.shape[1], 2)) / len(y_train))
    quantile_penalty = base_penalty / data.response_train_scale
    es_penalty = base_penalty / data.response_train_scale**2
    q_model = LinearLassoNN(x_train.shape[1]).to(device)
    train_model(
        q_model,
        loader,
        lambda x, y: quantile_loss(q_model(x), y, config.tau)
        + quantile_penalty * q_model.l1_penalty(),
        config.hdes_epochs,
        config.learning_rate,
    )
    q_model.eval()
    with torch.no_grad():
        q_train = q_model(x_train_t).cpu().numpy().ravel()
        q_eval = q_model(to_tensor(x_eval, device)).cpu().numpy().ravel()
    z_train = upper_tail_pseudo_outcome(q_train, y_train, config.tau)
    z_train_t = to_tensor(z_train.reshape(-1, 1), device)
    es_loader = DataLoader(TensorDataset(x_train_t, z_train_t), batch_size=config.batch_size, shuffle=True)
    es_model = LinearLassoNN(x_train.shape[1]).to(device)
    tail_probability = 1.0 - config.tau
    train_model(
        es_model,
        es_loader,
        lambda x, z: torch.mean((tail_probability * es_model(x) - z) ** 2)
        + tail_probability * es_penalty * es_model.l1_penalty(),
        config.hdes_epochs,
        config.learning_rate,
    )
    es_model.eval()
    with torch.no_grad():
        es_train = es_model(x_train_t).cpu().numpy().ravel()
        es_eval = es_model(to_tensor(x_eval, device)).cpu().numpy().ravel()
    return MethodResult("RLR", q_train, es_train, q_eval, es_eval, z_train)


def fit_krr(data: PreparedData, eval_idx: np.ndarray, config: NHANESConfig) -> MethodResult:
    """Fit a nonlinear KRR ES comparator on the audited full input."""
    require_sklearn()
    x_train = data.x_full[data.train_idx]
    x_eval = data.x_full[eval_idx]
    y_train = data.y[data.train_idx]
    gamma = config.krr_gamma
    if gamma is None or str(gamma).lower() == "auto":
        gamma = 1.0 / x_train.shape[1]
    kernel, kernel_params = sklearn_krr_kernel(config, float(gamma))
    q_model = GradientBoostingRegressor(
        loss="quantile",
        alpha=config.tau,
        n_estimators=300,
        max_depth=2,
        learning_rate=0.03,
        random_state=config.seed,
    )
    q_model.fit(x_train, y_train)
    q_train = q_model.predict(x_train)
    q_eval = q_model.predict(x_eval)
    z_train = upper_tail_pseudo_outcome(q_train, y_train, config.tau)
    es_model = KernelRidge(alpha=config.krr_alpha, kernel=kernel, **kernel_params)
    es_model.fit(x_train, z_train)
    tail_probability = 1.0 - config.tau
    es_train = es_model.predict(x_train) / tail_probability
    es_eval = es_model.predict(x_eval) / tail_probability
    display_kernel = str(config.krr_kernel).lower()
    return MethodResult(f"KRR-{display_kernel}", q_train, es_train, q_eval, es_eval, z_train)


def sklearn_krr_kernel(config: NHANESConfig, gamma: float) -> tuple[Any, dict[str, Any]]:
    """Map reference KRR kernel names to sklearn KernelRidge settings."""
    kernel = str(config.krr_kernel).strip().lower()
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
    raise ValueError(
        "Unknown KRR kernel: "
        f"{config.krr_kernel}. Supported: auto, rbf, gaussian, polynomial, linear."
    )


def pseudo_loss(result: MethodResult, y_eval: np.ndarray, tau: float) -> float:
    z_eval = upper_tail_pseudo_outcome(result.q_eval, y_eval, tau)
    return float(np.mean((z_eval - (1.0 - tau) * result.es_eval) ** 2))


def tune_factor_models(data: PreparedData, config: NHANESConfig, methods: list[str]) -> tuple[dict[str, Any], pd.DataFrame]:
    """Select factor/throughput settings solely with the validation split."""
    records: list[dict[str, Any]] = []
    selected: dict[str, Any] = {}
    y_validation = data.y[data.validation_idx]

    if "FA-NN" in methods:
        candidate_iter = itertools.product(
            factor_dim_candidates(config),
            throughput_dim_candidates(config),
            config.lambda_gamma_candidates,
        )
        for offset, (factor_dim, throughput_dim, lambda_gamma) in enumerate(candidate_iter):
            result = fit_fa_nn(
                data,
                int(factor_dim),
                int(throughput_dim),
                None if lambda_gamma is None else float(lambda_gamma),
                data.validation_idx,
                config,
                config.tuning_epochs,
                seed_offset=1000 + offset,
            )
            records.append(
                {
                    "method": "FA-NN",
                    "factor_dim": result.factor_dim,
                    "throughput_dim": result.throughput_dim,
                    "lambda_gamma": result.lambda_gamma,
                    "validation_pseudo_loss": pseudo_loss(result, y_validation, config.tau),
                }
            )
        fa_records = [record for record in records if record["method"] == "FA-NN"]
        selected["FA-NN"] = min(fa_records, key=lambda record: record["validation_pseudo_loss"])

    if "FAIR-NN" in methods:
        candidate_iter = itertools.product(
            factor_dim_candidates(config),
            throughput_dim_candidates(config),
            config.lambda_gamma_candidates,
        )
        for offset, (factor_dim, throughput_dim, lambda_gamma) in enumerate(candidate_iter):
            result = fit_fair_nn(
                data,
                int(factor_dim),
                int(throughput_dim),
                None if lambda_gamma is None else float(lambda_gamma),
                data.validation_idx,
                config,
                config.tuning_epochs,
                seed_offset=1000 + offset,
            )
            records.append(
                {
                    "method": "FAIR-NN",
                    "factor_dim": result.factor_dim,
                    "throughput_dim": result.throughput_dim,
                    "lambda_gamma": result.lambda_gamma,
                    "validation_pseudo_loss": pseudo_loss(result, y_validation, config.tau),
                }
            )
        fair_records = [record for record in records if record["method"] == "FAIR-NN"]
        selected["FAIR-NN"] = min(fair_records, key=lambda record: record["validation_pseudo_loss"])

    return selected, pd.DataFrame(records)


def selected_throughput(gamma: np.ndarray, feature_names: list[str], top_k: int) -> pd.DataFrame:
    importance = np.linalg.norm(gamma, axis=1)
    order = np.argsort(-importance)[: min(top_k, len(feature_names))]
    return pd.DataFrame(
        {
            "rank": np.arange(1, len(order) + 1),
            "variable": [feature_names[index] for index in order],
            "gamma_l2_norm": importance[order],
        }
    )


def add_summary_columns(row: dict[str, Any], prefix: str, values: np.ndarray | None) -> None:
    if values is None:
        return
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return
    row[f"{prefix}_min"] = float(np.min(finite))
    row[f"{prefix}_q01"] = float(np.quantile(finite, 0.01))
    row[f"{prefix}_q05"] = float(np.quantile(finite, 0.05))
    row[f"{prefix}_median"] = float(np.median(finite))
    row[f"{prefix}_q95"] = float(np.quantile(finite, 0.95))
    row[f"{prefix}_q99"] = float(np.quantile(finite, 0.99))
    row[f"{prefix}_max"] = float(np.max(finite))


def result_diagnostics(
    result: MethodResult,
    data: PreparedData,
    config: NHANESConfig,
) -> dict[str, Any]:
    """Summarize fitted outputs in the original cotinine unit."""

    row: dict[str, Any] = {
        "method": result.method,
        "tau": config.tau,
        "factor_dim": result.factor_dim,
        "throughput_dim": result.throughput_dim,
        "lambda_gamma": result.lambda_gamma,
    }
    add_summary_columns(row, "q_train", restore_response_scale(result.q_train, data))
    add_summary_columns(row, "es_train", restore_response_scale(result.es_train, data))
    if result.gamma is not None:
        gamma_abs = np.abs(result.gamma)
        row["gamma_abs_sum"] = float(gamma_abs.sum())
        row["gamma_l2_norm"] = float(np.linalg.norm(result.gamma))
        row["gamma_effective_nonzero_eps"] = int((gamma_abs > config.gamma_penalty_eps).sum())
        row["gamma_effective_nonzero_1e_4"] = int((gamma_abs > 1e-4).sum())
    add_summary_columns(row, "adaptive_weight", result.train_weights)
    return row


def evaluate_result(
    result: MethodResult,
    data: PreparedData,
    eval_idx: np.ndarray,
    split: str,
    config: NHANESConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y_eval = data.y_raw[eval_idx]
    q_eval = restore_response_scale(result.q_eval, data)
    es_eval = restore_response_scale(result.es_eval, data)
    groups = data.groups[eval_idx]
    z_eval = upper_tail_pseudo_outcome(q_eval, y_eval, config.tau)
    score = upper_tail_calibration_score(q_eval, es_eval, y_eval, config.tau)
    prediction_frame = pd.DataFrame(
        {
            "source_row": data.source_index[eval_idx],
            "split": split,
            "group": groups,
            "method": result.method,
            "tau": config.tau,
            "cotinine": y_eval,
            "q_hat": q_eval,
            "es_hat": es_eval,
            "pseudo_outcome": z_eval,
            "calibration_score": score,
        }
    )
    metric_rows = []
    for group in ["Overall", *sorted(np.unique(groups))]:
        mask = np.ones(len(groups), dtype=bool) if group == "Overall" else groups == group
        metric_rows.append(
            {
                "split": split,
                "method": result.method,
                "group": group,
                "tau": config.tau,
                "n": int(mask.sum()),
                "upper_tail_calibration_error": float(abs(score[mask].mean())),
                "upper_tail_calibration_mean": float(score[mask].mean()),
                "upper_tail_calibration_sd": float(score[mask].std(ddof=0)),
                "pseudo_outcome_loss": float(
                    np.mean((z_eval[mask] - (1.0 - config.tau) * es_eval[mask]) ** 2)
                ),
                "quantile_exceedance_rate": float(np.mean(y_eval[mask] > q_eval[mask])),
            }
        )
    return pd.DataFrame(metric_rows), prediction_frame


def group_es_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    return (
        predictions.groupby(["method", "group", "tau"], as_index=False)
        .agg(
            n=("es_hat", "size"),
            cotinine_mean=("cotinine", "mean"),
            q_hat_mean=("q_hat", "mean"),
            es_hat_mean=("es_hat", "mean"),
            es_hat_median=("es_hat", "median"),
            es_hat_q25=("es_hat", lambda x: x.quantile(0.25)),
            es_hat_q75=("es_hat", lambda x: x.quantile(0.75)),
        )
        .sort_values(["method", "group"])
    )


def maybe_plot_group_es(predictions: pd.DataFrame, output_dir: Path, config: NHANESConfig) -> None:
    if not config.plot:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fair = predictions.loc[predictions["method"].eq("FAIR-NN")].copy()
    if fair.empty:
        return
    groups = sorted(fair["group"].unique())
    values = [fair.loc[fair["group"].eq(group), "es_hat"].to_numpy() for group in groups]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.boxplot(values, labels=groups, showfliers=False)
    ax.set_xlabel("Racial/ethnic group")
    ax.set_ylabel("Predicted upper-tail conditional ES of serum cotinine")
    ax.set_title(f"FAIR-NN test-set upper-tail ES, $\\tau={config.tau:.2f}$")
    fig.tight_layout()
    fig.savefig(output_dir / "fair_nn_group_es_boxplot.png", dpi=180)
    plt.close(fig)


def run_training(config: NHANESConfig) -> None:
    if not 0.5 < config.tau < 1.0:
        raise ValueError("This NHANES application uses upper-tail levels with 0.5 < tau < 1.")
    require_torch()
    require_sklearn()
    set_seed(config.seed)
    methods = normalize_methods(config.methods)
    data = load_prepared_data(config)
    output_config = replace(config, output_dir=str(analysis_output_dir(config)))
    output_dir = write_audit_outputs(data, output_config)

    selected, tuning = tune_factor_models(data, config, methods)
    tuning.to_csv(output_dir / "tuning_results.csv", index=False)
    (output_dir / "selected_hyperparameters.json").write_text(
        json.dumps(selected, indent=2),
        encoding="utf-8",
    )

    results: dict[str, MethodResult] = {}
    if "RLR" in methods:
        results["RLR"] = fit_rlr(data, data.test_idx, config, seed_offset=3000)
    if "KRR" in methods:
        for kernel in krr_kernel_candidates(config):
            kernel_config = replace(config, krr_kernel=kernel)
            result = fit_krr(data, data.test_idx, kernel_config)
            results[result.method] = result
    if "FA-NN" in methods:
        choice = selected["FA-NN"]
        results["FA-NN"] = fit_fa_nn(
            data,
            int(choice["factor_dim"]),
            int(choice["throughput_dim"]),
            float(choice["lambda_gamma"]),
            data.test_idx,
            config,
            config.epochs,
            seed_offset=4000,
        )
    if "FAIR-NN" in methods:
        choice = selected["FAIR-NN"]
        results["FAIR-NN"] = fit_fair_nn(
            data,
            int(choice["factor_dim"]),
            int(choice["throughput_dim"]),
            float(choice["lambda_gamma"]),
            data.test_idx,
            config,
            config.epochs,
            seed_offset=4000,
        )

    metrics, prediction_frames = [], []
    for result in results.values():
        method_metrics, method_predictions = evaluate_result(result, data, data.test_idx, "test", config)
        metrics.append(method_metrics)
        prediction_frames.append(method_predictions)
    metrics_frame = pd.concat(metrics, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics_frame.to_csv(output_dir / "metrics_test.csv", index=False)
    predictions.to_csv(output_dir / "predictions_test.csv", index=False)
    group_es_summary(predictions).to_csv(output_dir / "group_es_summary.csv", index=False)
    pd.DataFrame([result_diagnostics(result, data, config) for result in results.values()]).to_csv(
        output_dir / "model_diagnostics.csv",
        index=False,
    )

    # #region debug-point D:test-comparison
    exec("try:\n import urllib.request\n _p='.dbg/fairnn-underperformance.env'; _u,_s='http://127.0.0.1:7777/event','fairnn-underperformance'\n try:\n  with open(_p) as _f: _c=_f.read(); _u=next((l.split('=',1)[1] for l in _c.split('\\n') if l.startswith('DEBUG_SERVER_URL=')),_u); _s=next((l.split('=',1)[1] for l in _c.split('\\n') if l.startswith('DEBUG_SESSION_ID=')),_s)\n except: pass\n _m=metrics_frame.loc[(metrics_frame['group']=='Overall') & metrics_frame['method'].isin(['FA-NN','FAIR-NN']),['method','upper_tail_calibration_error','pseudo_outcome_loss','quantile_exceedance_rate']]; urllib.request.urlopen(urllib.request.Request(_u,data=json.dumps({'sessionId':_s,'runId':'pre-fix','hypothesisId':'D','location':'simulation.py:run_training:test-metrics','msg':'[DEBUG] FA-NN and FAIR-NN overall test comparison','data':json.loads(_m.to_json(orient='records'))}).encode(),headers={'Content-Type':'application/json'}),timeout=1).read()\nexcept: pass")
    # #endregion

    for method in ("FA-NN", "FAIR-NN"):
        result = results.get(method)
        if result is not None and result.gamma is not None:
            filename = method.lower().replace("-", "_") + "_selected_throughput.csv"
            selected_throughput(result.gamma, data.full_feature_names, config.selected_top_k).to_csv(
                output_dir / filename,
                index=False,
            )
    maybe_plot_group_es(predictions, output_dir, config)

    run_info = {
        "n": int(len(data.y)),
        "p": int(data.x_full.shape[1]),
        "n_train": int(len(data.train_idx)),
        "n_validation": int(len(data.validation_idx)),
        "n_test": int(len(data.test_idx)),
        "methods": methods,
        "tau": config.tau,
        "response_standardize": config.response_standardize,
        "response_train_mean": data.response_train_mean,
        "response_train_scale": data.response_train_scale,
        "config": asdict(config),
    }
    (output_dir / "run_info.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    print(f"Saved NHANES rerun outputs to {output_dir.resolve()}")


def run_training_grid(config: NHANESConfig) -> None:
    """Run the same audited split separately for every configured tail level."""
    tau_values = [float(value) for value in (config.tau_values or [config.tau])]
    if len(set(tau_values)) != len(tau_values):
        raise ValueError("tau_values must not contain duplicates")
    root_output = analysis_output_dir(config)
    for tau in tau_values:
        if not 0.5 < tau < 1.0:
            raise ValueError(f"Invalid upper-tail level: {tau}")
        suffix = f"tau_{int(round(100 * tau)):03d}"
        run_config = replace(
            config,
            tau=tau,
            tau_values=tau_values,
            output_dir=str(root_output / suffix),
        )
        run_training(run_config)
