"""Run DWES simulations from a YAML configuration file.

Methods are named as in the paper (RLR, KRR, FA-NN, FAIR-NN). These paper
names are translated to the internal implementation names used by
``simulation_methods.py`` before running, and the output ``method`` column is
translated back to paper names. The numerical core in ``simulation_methods.py``
is untouched.

Run:
    python run_from_config.py train conf/config.yaml
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import fields
from pathlib import Path
from typing import Any

from simulation_methods import P_VALUES, TAU_VALUES, SimulationConfig, run_grid


# ---------------------------------------------------------------------------
# Paper <-> internal method-name translation
# ---------------------------------------------------------------------------
# The paper reports four methods. Their internal implementation names in
# simulation_methods.py are historical and differ from the paper:
#   RLR     -> HDES        (regularized two-step linear ES regression)
#   KRR     -> KRR         (kernel ridge regression, unchanged)
#   FA-NN   -> FAST-FNR    (factor-augmented NN, non-iterative, unweighted)
#   FAIR-NN -> FAST-RWFNR  (factor-augmented iterative reweighting NN)
PAPER_TO_INTERNAL = {
    "RLR": "HDES",
    "KRR": "KRR",
    "FA-NN": "FAST-FNR",
    "FAIR-NN": "FAST-RWFNR",
}
INTERNAL_TO_PAPER = {internal: paper for paper, internal in PAPER_TO_INTERNAL.items()}

# The paper reports these four methods by default, in this order.
DEFAULT_METHODS = ("RLR", "KRR", "FA-NN", "FAIR-NN")

# Internal-only variants remain runnable for ablation/debugging if a user asks
# for them explicitly, but they are not part of the paper comparison.
KNOWN_INTERNAL = {
    "KRR", "HDES", "FNR", "WFNR", "RFNR", "RWFNR",
    "FAST-FNR", "FAST-WFNR", "FAST-RFNR", "FAST-RWFNR",
}

EXTRA_CONFIG_KEYS = {"p_values", "taus", "methods", "output"}


def to_internal_method(name: str) -> str:
    """Translate a requested method name to its internal implementation name."""
    if name in PAPER_TO_INTERNAL:
        return PAPER_TO_INTERNAL[name]
    if name in KNOWN_INTERNAL:
        return name
    allowed = ", ".join(sorted(PAPER_TO_INTERNAL) + sorted(KNOWN_INTERNAL))
    raise ValueError(f"Unknown method {name!r}. Allowed values: {allowed}.")


def to_paper_method(internal_name: str) -> str:
    """Translate an internal implementation name back to the paper name."""
    return INTERNAL_TO_PAPER.get(internal_name, internal_name)


def parse_scalar(value: str) -> Any:
    value = value.strip()
    lowered = value.lower()
    if lowered in {"null", "~"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if value.startswith("[") and value.endswith("]"):
        return [parse_scalar(item) for item in value[1:-1].split(",") if item.strip()]
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value.strip("\"'")


def parse_simple_yaml(path: Path) -> dict[str, Any]:
    """Parse the flat YAML format used by conf/config.yaml without extra dependencies."""
    config: dict[str, Any] = {}
    current_list_key: str | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue

        stripped = line.strip()
        if stripped.startswith("- "):
            if current_list_key is None:
                raise ValueError(f"List item without a key in {path}: {raw_line}")
            config[current_list_key].append(parse_scalar(stripped[2:]))
            continue

        if ":" not in stripped:
            raise ValueError(f"Invalid config line in {path}: {raw_line}")

        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not value:
            config[key] = []
            current_list_key = key
        else:
            config[key] = parse_scalar(value)
            current_list_key = None

    return config


def load_yaml_config(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError:
        return parse_simple_yaml(path)

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Top-level YAML content must be a mapping: {path}")
    return data


def build_config(
    data: dict[str, Any],
) -> tuple[SimulationConfig, list[int], list[float], list[str], Path]:
    config_fields = {field.name for field in fields(SimulationConfig)}
    allowed_keys = config_fields | EXTRA_CONFIG_KEYS
    unknown_keys = sorted(set(data) - allowed_keys)
    if unknown_keys:
        raise ValueError(f"Unknown config keys: {', '.join(unknown_keys)}")

    missing_required = [key for key in ("design", "factor_model") if key not in data]
    if missing_required:
        raise ValueError(f"Missing required config keys: {', '.join(missing_required)}")

    config_kwargs = {key: data[key] for key in config_fields if key in data}
    sim_config = SimulationConfig(**config_kwargs)

    p_values = [int(value) for value in data.get("p_values", P_VALUES)]
    taus = [float(value) for value in data.get("taus", TAU_VALUES)]
    requested_methods = [str(value) for value in data.get("methods", DEFAULT_METHODS)]
    internal_methods = [to_internal_method(name) for name in requested_methods]
    output = Path(str(data.get("output", "results_from_config.csv")))
    if not output.is_absolute():
        output = Path(__file__).resolve().parent / output

    return sim_config, p_values, taus, internal_methods, output


def train(config_path: Path) -> None:
    data = load_yaml_config(config_path)
    sim_config, p_values, taus, internal_methods, output = build_config(data)
    records = run_grid(
        sim_config, output, p_values=p_values, tau_values=taus, methods=internal_methods
    )
    # Translate the internal method labels in the written CSV back to paper names.
    _relabel_method_column(output)
    return records


def _relabel_method_column(output_path: Path) -> None:
    """Rewrite the ``method`` column of the results CSV using paper names."""
    import csv

    if not output_path.exists():
        return
    with output_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not fieldnames or "method" not in fieldnames:
        return
    for row in rows:
        row["method"] = to_paper_method(row["method"])
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train",), help="Action to run.")
    parser.add_argument("config", type=Path, help="Path to a YAML configuration file.")
    args = parser.parse_args()

    train(args.config)


if __name__ == "__main__":
    main()
