"""Run the audited NHANES cotinine application from a YAML configuration file."""

from __future__ import annotations

import argparse
import ast
from dataclasses import fields
from pathlib import Path
from typing import Any

from simulation import NHANESConfig, run_audit, run_training_grid


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
    """Parse the flat YAML format used by the local application configs."""
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


def load_yaml_config(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError:
        return parse_simple_yaml(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Top-level YAML content must be a mapping: {path}")
    return data


def build_config(data: dict[str, Any], config_path: Path) -> NHANESConfig:
    allowed = {field.name for field in fields(NHANESConfig)}
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"Unknown config keys: {', '.join(unknown)}")

    config = NHANESConfig(**{key: data[key] for key in allowed if key in data})
    code_dir = config_path.resolve().parent.parent
    for field_name in ("data_path", "audit_path", "output_dir"):
        value = Path(getattr(config, field_name))
        if not value.is_absolute():
            setattr(config, field_name, str((code_dir / value).resolve()))
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "train"), help="Action to run.")
    parser.add_argument("config", type=Path, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    config = build_config(load_yaml_config(args.config), args.config)
    if args.command == "audit":
        run_audit(config)
    else:
        run_training_grid(config)


if __name__ == "__main__":
    main()
