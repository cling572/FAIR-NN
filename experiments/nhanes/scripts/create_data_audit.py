#!/usr/bin/env python3
"""Create an audit manifest for the NHANES cotinine FAIR-NN application.

The audit records the source variables and locks the first-stage analysis
cohort: Asian, Black, Hispanic, and White respondents.  The source field
``raceM`` is the confirmed Hispanic indicator.  ``raceNA`` is excluded from
this four-group analysis; its resulting zero-variance indicator is removed.
``WTSVOCPR`` is retained as a survey-design field but excluded from the
baseline predictor matrix.
"""

from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
SOURCE_FILE = DATA_DIR / "design_matrix_new.csv"
CSV_OUTPUT = DATA_DIR / "data_audit.csv"
MD_OUTPUT = DATA_DIR / "data_audit.md"

OUTCOME = "cotinine"
SURVEY_WEIGHT = "WTSVOCPR"
RACE_COLUMNS = ["raceA", "raceB", "raceM", "raceNA"]
ANALYTIC_GROUPS = {
    "Asian": "raceA",
    "Black": "raceB",
    "Hispanic": "raceM",
    "White": None,  # Baseline category: all four source race dummies equal 0.
}


def is_binary(series: pd.Series) -> bool:
    values = set(series.dropna().unique())
    return values.issubset({0, 1})


def format_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.10g}"


def main() -> None:
    df = pd.read_csv(SOURCE_FILE)
    if OUTCOME not in df.columns:
        raise ValueError(f"Missing response column: {OUTCOME}")
    if any(column not in df.columns for column in RACE_COLUMNS):
        raise ValueError("Missing one or more race indicators")

    source_n, source_columns = df.shape
    source_predictors = source_columns - 1
    race_sum = df[RACE_COLUMNS].sum(axis=1)
    if not (race_sum <= 1).all():
        raise ValueError("Race indicators are not mutually exclusive")

    group_counts = {
        "Asian": int(df["raceA"].eq(1).sum()),
        "Black": int(df["raceB"].eq(1).sum()),
        "Hispanic": int(df["raceM"].eq(1).sum()),
        "White": int(race_sum.eq(0).sum()),
        "Excluded raceNA": int(df["raceNA"].eq(1).sum()),
    }
    analytic_mask = df["raceNA"].eq(0)
    analytic = df.loc[analytic_mask].copy()

    rows = []
    for column in df.columns:
        source = df[column]
        selected = analytic[column]
        binary = is_binary(source)
        integer_valued = bool(
            np.all(
                np.isclose(
                    source.to_numpy(dtype=float),
                    np.round(source.to_numpy(dtype=float)),
                )
            )
        )

        included = True
        block = ""
        role = "predictor"
        rationale = ""
        if column == OUTCOME:
            included = False
            role = "outcome"
            block = "response"
            rationale = "Raw serum cotinine response; not part of the predictor matrix."
        elif column == SURVEY_WEIGHT:
            included = False
            role = "survey_design"
            block = "excluded"
            rationale = (
                "NHANES survey weight; excluded from the baseline predictor matrix "
                "and not substituted for FAIR-NN adaptive residual-variance weights."
            )
        elif column == "raceNA":
            included = False
            role = "race_indicator"
            block = "excluded"
            rationale = (
                "Corresponds to respondents outside the confirmed four-group analysis; "
                "all values are zero after the raceNA exclusion."
            )
        elif binary:
            role = "discrete_predictor"
            block = "sparse_throughput"
            rationale = (
                "Binary dummy retained in the full high-dimensional input; "
                "not used for PCA/factor extraction."
            )
        else:
            role = "quantitative_predictor"
            block = "continuous_factor_candidate"
            rationale = (
                "Non-binary quantitative variable; standardize using training-sample "
                "statistics before continuous-block factor extraction."
            )

        rows.append(
            {
                "source_variable": column,
                "source_role": role,
                "baseline_included": included,
                "baseline_block": block,
                "storage_dtype": str(source.dtype),
                "source_n_unique": int(source.nunique(dropna=False)),
                "source_missing": int(source.isna().sum()),
                "source_min": format_number(float(source.min())),
                "source_max": format_number(float(source.max())),
                "analytic_n_unique": int(selected.nunique(dropna=False)),
                "analytic_missing": int(selected.isna().sum()),
                "analytic_min": format_number(float(selected.min())),
                "analytic_max": format_number(float(selected.max())),
                "binary_0_1": binary,
                "integer_valued": integer_valued,
                "rationale": rationale,
            }
        )

    audit = pd.DataFrame(rows)
    audit.to_csv(CSV_OUTPUT, index=False)

    baseline = audit.loc[audit["baseline_included"]]
    continuous = baseline.loc[baseline["baseline_block"].eq("continuous_factor_candidate")]
    discrete = baseline.loc[baseline["baseline_block"].eq("sparse_throughput")]
    final_p = len(baseline)

    if source_n != sum(group_counts.values()):
        raise ValueError("Race-group counts do not reconcile to source sample size")
    if len(analytic) != sum(group_counts[group] for group in ANALYTIC_GROUPS):
        raise ValueError("Analytic group counts do not reconcile to selected sample")
    if int(analytic.isna().sum().sum()) != 0:
        raise ValueError("Selected analytic data contain missing values")
    if final_p != len(continuous) + len(discrete):
        raise ValueError("Baseline predictor blocks do not reconcile")

    continuous_names = ", ".join(continuous["source_variable"].tolist())
    md = f"""# NHANES Cotinine Data Audit

## Source

- File: `{SOURCE_FILE.name}`
- Observations: `{source_n}`
- Source columns: `{source_columns}`
- Response: `{OUTCOME}`
- Source predictor count: `{source_predictors}`
- Missing cells in source matrix: `{int(df.isna().sum().sum())}`

## Confirmed race mapping and baseline cohort

| Analytic group | Source coding | Count | Baseline treatment |
|---|---:|---:|---|
| Asian | `raceA = 1` | {group_counts["Asian"]} | Retain |
| Black | `raceB = 1` | {group_counts["Black"]} | Retain |
| Hispanic | `raceM = 1` | {group_counts["Hispanic"]} | Retain |
| White | `raceA = raceB = raceM = raceNA = 0` | {group_counts["White"]} | Retain |
| Outside four-group analysis | `raceNA = 1` | {group_counts["Excluded raceNA"]} | Exclude |

The four retained groups contain **n = {len(analytic)}** respondents. The source
race indicators are mutually exclusive. `raceM` is the confirmed source field
for the Hispanic group.

## Baseline filtering rules

1. Use `cotinine` as the untransformed upper-tail ES response.
2. Restrict to Asian, Black, Hispanic, and White respondents by excluding
   `raceNA = 1` ({group_counts["Excluded raceNA"]} observations).
3. No additional listwise deletion is needed: the selected matrix has no missing
   values.
4. Exclude `WTSVOCPR` from the baseline predictor matrix. It is an NHANES survey
   design weight and must not be confused with FAIR-NN's learned adaptive
   residual-variance weights.
5. Drop `raceNA` from the predictor matrix because it is identically zero after
   Step 2.
6. Retain the other binary indicators, including `raceA`, `raceB`, and `raceM`,
   in the sparse-throughput input. White is their baseline category.

## Final baseline dimension

| Component | Count | Treatment |
|---|---:|---|
| Response | 1 | `cotinine` |
| Quantitative block $X^{{(c)}}$ | {len(continuous)} | Standardize using training-sample moments; estimate factors only from this block |
| Binary/discrete block $X^{{(d)}}$ | {len(discrete)} | Do not use PCA; retain for sparse throughput |
| Final predictor dimension $p$ | **{final_p}** | Full input for sparse throughput |
| Final analytic sample $n$ | **{len(analytic)}** | Four confirmed racial/ethnic groups |

The quantitative factor candidate variables are:

`{continuous_names}`

## Reproducibility notes

- `data_audit.csv` is the variable-level manifest for every source column,
  including its role, inclusion status, range, missingness, and baseline block.
- This audit locks the baseline *data* specification only. Train/validation/test
  splitting, factor dimension selection, neural-network tuning, and outcome
  evaluation are subsequent steps.
- A sensitivity analysis may include `WTSVOCPR` as an input covariate, but it is
  not part of the baseline specification summarized here.
"""
    MD_OUTPUT.write_text(md)
    print(f"Wrote {CSV_OUTPUT}")
    print(f"Wrote {MD_OUTPUT}")
    print(
        f"Baseline audit: n={len(analytic)}, p={final_p}, "
        f"continuous={len(continuous)}, discrete={len(discrete)}"
    )


if __name__ == "__main__":
    main()
