#!/usr/bin/env python3
"""Run the Tecator upper-tail conditional ES experiment from a YAML config."""

from __future__ import annotations

import argparse
import ast
from dataclasses import fields
from pathlib import Path
from typing import Any

from tecator_es import ExperimentConfig, load_tecator_data, run_audit, run_experiment


ROOT = Path(__file__).resolve().parent


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
    """Parse the flat YAML format used by the local experiment config."""

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
        key, value = key.strip(), value.strip()
        if value:
            config[key] = parse_scalar(value)
            current_list_key = None
        else:
            config[key] = []
            current_list_key = key
    return config


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError:
        return parse_simple_yaml(path)
    with path.open("r", encoding="utf-8") as handle:
        result = yaml.safe_load(handle)
    if result is None:
        return {}
    if not isinstance(result, dict):
        raise ValueError(f"Top-level YAML content must be a mapping: {path}")
    return result


def build_config(config_path: Path) -> ExperimentConfig:
    values = load_yaml(config_path)
    allowed = {item.name for item in fields(ExperimentConfig)}
    unknown = sorted(set(values).difference(allowed))
    if unknown:
        raise ValueError(f"Unknown config keys: {', '.join(unknown)}")
    config = ExperimentConfig(**{key: values[key] for key in allowed if key in values})
    for attribute in ("data_path", "output_dir"):
        path = Path(getattr(config, attribute))
        if not path.is_absolute():
            setattr(config, attribute, str((ROOT / path).resolve()))
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "train"), help="Action to run.")
    parser.add_argument(
        "--config",
        default="conf/experiment.yaml",
        type=Path,
        help="YAML configuration path, relative to this experiment directory by default.",
    )
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = build_config(config_path)
    data = load_tecator_data(Path(config.data_path))
    output_dir = Path(config.output_dir)
    if args.command == "audit":
        run_audit(data, config, output_dir)
        print(f"Saved Tecator audit manifest to {output_dir.resolve()}")
    else:
        run_experiment(data, config, output_dir)


if __name__ == "__main__":
    main()
