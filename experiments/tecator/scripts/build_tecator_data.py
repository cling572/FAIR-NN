#!/usr/bin/env python3
"""Parse the official Tecator text file into reproducible FAIR-NN datasets."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "raw" / "tecator_original.txt"
DATA_DIR = ROOT / "data"

N_ABSORBANCE = 100
N_PCS = 22
N_OUTPUTS = 3
VALUES_PER_SAMPLE = N_ABSORBANCE + N_PCS + N_OUTPUTS
N_SAMPLES = 240


def source_partition(sample_id: int) -> str:
    if sample_id <= 129:
        return "C"
    if sample_id <= 172:
        return "M"
    if sample_id <= 215:
        return "T"
    if sample_id <= 223:
        return "E1"
    return "E2"


def parse_rows() -> list[dict[str, float | int | str]]:
    lines = RAW_PATH.read_text(encoding="utf-8").splitlines()
    header_locations = [index for index, line in enumerate(lines) if line == "real_in=122"]
    if not header_locations:
        raise ValueError("Could not find the Tecator machine-readable header.")

    start = header_locations[-1] + 5
    values = [float(token) for line in lines[start:] for token in line.split()]
    expected_values = N_SAMPLES * VALUES_PER_SAMPLE
    if len(values) != expected_values:
        raise ValueError(
            f"Expected {expected_values} numeric values but found {len(values)}."
        )

    rows: list[dict[str, float | int | str]] = []
    for offset in range(N_SAMPLES):
        sample_id = offset + 1
        block_start = offset * VALUES_PER_SAMPLE
        block = values[block_start : block_start + VALUES_PER_SAMPLE]
        row: dict[str, float | int | str] = {
            "sample_id": sample_id,
            "source_partition": source_partition(sample_id),
        }
        row.update(
            {
                f"absorbance_{index:03d}": block[index - 1]
                for index in range(1, N_ABSORBANCE + 1)
            }
        )
        row.update(
            {
                f"pc_{index:02d}": block[N_ABSORBANCE + index - 1]
                for index in range(1, N_PCS + 1)
            }
        )
        row["moisture_pct"] = block[-3]
        row["fat_pct"] = block[-2]
        row["protein_pct"] = block[-1]
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, float | int | str]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({column: row[column] for column in columns} for row in rows)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rows = parse_rows()
    absorbance_columns = [f"absorbance_{index:03d}" for index in range(1, 101)]
    pc_columns = [f"pc_{index:02d}" for index in range(1, 23)]
    all_columns = [
        "sample_id",
        "source_partition",
        *absorbance_columns,
        *pc_columns,
        "moisture_pct",
        "fat_pct",
        "protein_pct",
    ]
    write_csv(DATA_DIR / "tecator_240_official.csv", rows, all_columns)

    # Mai and Zou (2015) use samples 1--215 and remove samples 103 and 105.
    analysis_rows = [
        row
        for row in rows
        if int(row["sample_id"]) <= 215 and int(row["sample_id"]) not in {103, 105}
    ]
    analysis_columns = ["sample_id", "source_partition", *absorbance_columns, "fat_pct"]
    write_csv(DATA_DIR / "tecator_fair_nn_213.csv", analysis_rows, analysis_columns)

    raw_sha256 = hashlib.sha256(RAW_PATH.read_bytes()).hexdigest()
    print(f"raw_sha256={raw_sha256}")
    print(f"official_rows={len(rows)}")
    print(f"fair_nn_rows={len(analysis_rows)}")
    print(
        "fair_nn_partitions="
        + ",".join(
            f"{partition}:{sum(row['source_partition'] == partition for row in analysis_rows)}"
            for partition in ("C", "M", "T")
        )
    )


if __name__ == "__main__":
    main()
