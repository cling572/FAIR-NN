# FAIR-NN: A Robust Framework for Joint Learning of High-dimensional Conditional Quantiles and Expected Shortfall

This repository provides the reproducible research code for **FAIR-NN**, a
factor-augmented iterative reweighting neural network for high-dimensional
conditional quantile and expected shortfall (ES) estimation.

FAIR-NN combines factor-based dimension reduction with a trainable sparse
idiosyncratic throughput, a factor-augmented sparse quantile network, an ES
pseudo-outcome regression, adaptive inverse-variance weighting, and an iterative
refinement step. The repository reproduces the paper's simulation studies and
its two real-data applications.

> Paper: *(title / authors / arXiv or DOI link to be added)*

## Repository layout

```text
DWES/
├── README.md                 # this file
├── LICENSE                   # MIT
├── requirements.txt          # shared dependency stack
├── .gitignore
├── simulations/              # Section 5 simulation studies
│   ├── run_from_config.py    # single entry point: reads a YAML config
│   ├── simulation_methods.py # methods + data-generating processes (numerical core)
│   └── conf/                 # config.yaml + example1a..example3b.yaml presets
└── experiments/
    ├── tecator/              # Tecator near-infrared upper-tail ES application
    └── nhanes/               # NHANES serum-cotinine upper-tail ES application
```

## Methods

The four methods reported in the paper are named consistently across the code
and configuration:

| Name in config/output | Description |
|---|---|
| `RLR` | Regularized two-step linear ES regression (Zhang et al., 2025) |
| `KRR` | Two-step kernel ridge regression, Gaussian RBF kernel (Yu et al., 2024) |
| `FA-NN` | Factor-augmented neural network (non-iterative, unweighted) |
| `FAIR-NN` | Factor-augmented iterative reweighting neural network (proposed) |

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.9+ is recommended. `torch` provides the neural-network backend (CPU is
sufficient; set `device: cuda` in a config to use a GPU). The `quantes` package
is required only for the `KRR` baseline.

## Reproducing the simulations

The simulations use a single config-driven entry point. Choose an example
preset and run it:

```bash
cd simulations
python run_from_config.py train conf/example1a.yaml
```

Presets map one-to-one to the paper's examples:

| Preset | Setting |
|---|---|
| `conf/example1a.yaml` | Example 1, Case 1a — factor only, linear |
| `conf/example1b.yaml` | Example 1, Case 1b — factor only, additive |
| `conf/example2a.yaml` | Example 2, Case 2a — factor + sparse heterogeneity, linear |
| `conf/example2b.yaml` | Example 2, Case 2b — factor + sparse heterogeneity, nonlinear |
| `conf/example3a.yaml` | Example 3, Case 3a — no factor, linear |
| `conf/example3b.yaml` | Example 3, Case 3b — no factor, nonlinear |

Each preset fixes the paper grid (`p ∈ {500,1000,1500,2000}`,
`τ ∈ {0.05,0.10,0.20}`, `n_train=2000`, `n_test=1000`, depth 3, width 64) and
writes a results CSV whose `method` column uses the paper names above. To build
a custom setting, copy `conf/config.yaml` and edit the data-generating switches
documented at the top of that file.

## Reproducing the applications

Both applications ship scripts and configuration only; the raw and derived data
are not distributed. Each subdirectory has a README with download-and-build
instructions.

- **Tecator** (`experiments/tecator/`): public StatLib data; a build script
  regenerates the analysis CSVs. See its README.
- **NHANES** (`experiments/nhanes/`): public CDC NHANES data; an audit script
  locks the analysis cohort. See its README.

## Data availability

No raw or derived datasets are committed to this repository. Simulation data are
generated on the fly by `simulation_methods.py`; application data are
regenerated locally from their public sources using the provided scripts.

## License

Released under the MIT License. See [LICENSE](LICENSE).

## Citation

If you use this code, please cite the paper. A `CITATION.cff` entry will be
added once the bibliographic details are final.
