# NHANES Serum-Cotinine Upper-Tail Conditional ES Experiment

This experiment applies FAIR-NN and three benchmarks (RLR, KRR, FA-NN) to
NHANES data, estimating the conditional upper-tail quantile and expected
shortfall (ES) of serum-cotinine exposure, and reporting group-specific tail
risk across racial/ethnic groups.

The NHANES-derived design matrix and audit files are **not** distributed with
this repository. Regenerate them locally from the public NHANES source using
the steps below.

## 1. Data: obtain and build

**Source.** The data come from the U.S. CDC National Health and Nutrition
Examination Survey (NHANES), <https://wwwn.cdc.gov/nchs/nhanes/>. Serum cotinine
is the laboratory analyte; the predictors are demographic, questionnaire, and
examination variables, with race/ethnicity indicators (`raceA`, `raceB`,
`raceM`, `raceNA`). Download the relevant NHANES cycles and assemble the
analysis design matrix as `data/design_matrix_new.csv`, with the response column
`cotinine` and the predictor columns used by the audit.

> NHANES files are public but large and versioned by cycle. We therefore ship
> only the assembly/audit scripts and configuration, not the derived matrix.

**Build the audit manifest.** Once `data/design_matrix_new.csv` exists:

```bash
python scripts/create_data_audit.py
```

This locks the four-group analysis cohort (Asian, Black, Hispanic, White),
excludes `raceNA`, and writes `data/data_audit.csv` and `data/data_audit.md`.

## 2. Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Run

```bash
# Preflight audit: build the stratified split manifest and validate the cohort.
python run_from_config.py audit conf/config.yaml

# Full experiment: predictions, group metrics, tuning records, throughput.
python run_from_config.py train conf/config.yaml
```

Configuration lives in `conf/config.yaml`: methods `[RLR, KRR, FA-NN, FAIR-NN]`,
tail levels `[0.70, 0.80, 0.90]`, stratified train/validation/test split, and
group columns `[raceA, raceB, raceM]` with White as the baseline. Set
`analysis_group` to `all` for the pooled analysis or to a single group to fit it
separately.

## 4. Figures

```bash
python generate_cotinine_quantile_figure.py
```

## Files

- `run_from_config.py` — entry point (`audit` / `train`).
- `simulation.py` — models, split logic, group metrics, and throughput ranking.
- `generate_cotinine_quantile_figure.py` — cotinine quantile-curve figure.
- `scripts/create_data_audit.py` — build the audit manifest and lock the cohort.
- `conf/config.yaml` — experiment configuration.
- `data/` — created locally (design matrix and audit; not committed).

## Note on data paths

`conf/config.yaml` reads `data_path` and `audit_path`. In this repository layout
both the design matrix and the audit files live under `data/`, so set:

```yaml
data_path: data/design_matrix_new.csv
audit_path: data/data_audit.csv
```
