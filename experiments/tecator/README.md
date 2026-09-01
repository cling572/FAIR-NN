# Tecator Upper-Tail Conditional ES Experiment

This experiment applies FAIR-NN and three benchmarks (RLR, KRR, FA-NN) to the
Tecator near-infrared spectroscopy data, estimating the conditional upper-tail
quantile and expected shortfall (ES) of the fat content.

The raw data and all derived CSVs are **not** distributed with this repository.
Use the download-and-build step below to regenerate them locally.

## 1. Data: download and build

**Source.** The Tecator dataset is publicly available from StatLib at
<https://lib.stat.cmu.edu/datasets/tecator>. It contains 240 meat samples, each
with 100 near-infrared absorbance channels, 22 supplied principal components,
and chemically measured moisture, fat, and protein percentages.

**Steps.**

1. Download the official text file and save it as `raw/tecator_original.txt`.
2. Build the reproducible analysis CSVs:

   ```bash
   python scripts/build_tecator_data.py
   ```

   This writes `data/tecator_240_official.csv` (all 240 samples) and
   `data/tecator_fair_nn_213.csv` (the analysis cohort). The build script prints
   the SHA-256 of the raw file so you can verify the download.

**Analysis cohort.** Following the real-data setup of Mai and Zou (2015), the
analysis file keeps samples 1--215 and removes the outliers #103 and #105,
yielding 213 observations. Partitions C (127) and M (43) form the training set
(n = 170); T (43) is the held-out test set. The extrapolation sets E1 and E2 are
excluded. The 22 supplied principal components are deliberately dropped: any
factor representation is learned only on the training samples to prevent
test-set leakage.

## 2. Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Run

```bash
# Preflight audit of the split and cohort:
python run_experiment.py audit

# Full experiment (writes metrics, predictions, and manifest):
python run_experiment.py train
```

Configuration lives in `conf/experiment.yaml`: methods `[RLR, KRR, FA-NN,
FAIR-NN]`, tail levels `[0.70, 0.80, 0.90]`, fixed `train_size: 170` and
`test_size: 43`. The factor dimension, throughput dimension, and penalties are
prespecified singletons so that nothing is tuned on the held-out test set.

## 4. Figures

```bash
python generate_figures.py
```

This regenerates the method-comparison, tail-prediction, throughput-heatmap,
and covariate-correlation figures from the saved result CSVs.

## Files

- `run_experiment.py` — entry point (`audit` / `train`).
- `tecator_es.py` — models, split logic, metrics, and table generation.
- `generate_figures.py` — publication figures from the result CSVs.
- `scripts/build_tecator_data.py` — parse the raw file into analysis CSVs.
- `conf/experiment.yaml` — experiment configuration.
- `data/`, `raw/` — created locally by the build step (not committed).

## Reference

Mai, Q. and Zou, H. (2015). The fused Kolmogorov filter: A nonparametric
model-free screening method.
