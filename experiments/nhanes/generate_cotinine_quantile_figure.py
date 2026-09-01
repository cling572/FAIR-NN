"""Generate descriptive NHANES serum-cotinine figures.

The figures use the audited four-group analytic cohort and are written directly
to the supplementary-material figures directory. The first figure compares
racial/ethnic empirical quantile curves. The second figure compares kernel
density estimates of log-transformed cotinine across the four groups.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_PATH = Path(__file__).resolve().parent / "design_matrix_new.csv"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "ESFinal" / "figures"
QUANTILE_OUTPUT_STEM = OUTPUT_DIR / "nhanes_cotinine_quantile_curves"
DISTRIBUTION_OUTPUT_STEM = OUTPUT_DIR / "nhanes_cotinine_distribution"

GROUPS = {
    "Asian": {"column": "raceA", "color": "#009E73"},
    "Black": {"column": "raceB", "color": "#D55E00"},
    "Hispanic": {"column": "raceM", "color": "#CC79A7"},
    "White": {"column": None, "color": "#0072B2"},
}


def set_journal_style():
    """Apply a restrained, print-friendly style shared by both figures."""

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.linewidth": 0.8,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "savefig.dpi": 300,
        }
    )


def group_masks(data):
    """Return mutually exclusive masks for the audited four-group cohort."""

    eligible = data["raceNA"].eq(0)
    masks = {
        name: eligible & data[spec["column"]].eq(1)
        for name, spec in GROUPS.items()
        if spec["column"] is not None
    }
    masks["White"] = (
        eligible
        & data["raceA"].eq(0)
        & data["raceB"].eq(0)
        & data["raceM"].eq(0)
    )

    membership = sum(mask.astype(int) for mask in masks.values())
    if not membership.loc[eligible].eq(1).all():
        raise ValueError("The retained observations do not form four exclusive groups.")
    return masks


def finish_axes(ax, grid_axis="y"):
    """Remove nonessential spines and add subtle major grid lines."""

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(
        axis=grid_axis,
        color="#D9D9D9",
        linewidth=0.55,
        linestyle="-",
        zorder=0,
    )


def save_figure(fig, output_stem):
    """Write vector and raster versions of one figure."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def smooth_monotone_quantiles(log_quantiles, smoothing_penalty=1500.0):
    """Smooth log-quantile curves with a second-difference penalty.

    The display smoother is fitted on an equally spaced probability grid.
    A cumulative maximum restores the required nondecreasing quantile shape
    after smoothing.
    """

    n_levels = log_quantiles.shape[0]
    main_diagonal = np.full(n_levels, 6.0)
    main_diagonal[:2] = [1.0, 5.0]
    main_diagonal[-2:] = [5.0, 1.0]
    first_off_diagonal = np.full(n_levels - 1, -4.0)
    first_off_diagonal[[0, -1]] = -2.0
    second_off_diagonal = np.ones(n_levels - 2)

    system_matrix = np.eye(n_levels)
    indices = np.arange(n_levels)
    system_matrix[indices, indices] += smoothing_penalty * main_diagonal
    system_matrix[indices[:-1], indices[1:]] += (
        smoothing_penalty * first_off_diagonal
    )
    system_matrix[indices[1:], indices[:-1]] += (
        smoothing_penalty * first_off_diagonal
    )
    system_matrix[indices[:-2], indices[2:]] += (
        smoothing_penalty * second_off_diagonal
    )
    system_matrix[indices[2:], indices[:-2]] += (
        smoothing_penalty * second_off_diagonal
    )
    smoothed = np.linalg.solve(system_matrix, log_quantiles)
    return np.maximum.accumulate(smoothed, axis=0)


def plot_group_quantiles(data, masks):
    """Plot penalized-smoothed empirical quantile curves."""

    plotting_levels = np.linspace(0.001, 0.999, 401)
    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    raw_log_quantiles = []
    for name in GROUPS:
        values = data.loc[masks[name], "cotinine"].to_numpy()
        raw_log_quantiles.append(np.log10(np.quantile(values, plotting_levels)))
    smooth_log_quantiles = smooth_monotone_quantiles(
        np.column_stack(raw_log_quantiles)
    )

    for index, (name, spec) in enumerate(GROUPS.items()):
        ax.plot(
            plotting_levels,
            10**smooth_log_quantiles[:, index],
            color=spec["color"],
            linewidth=2.0,
            label=name,
        )

    for tau in (0.70, 0.80, 0.90):
        ax.axvline(
            tau,
            color="#9E9E9E",
            linestyle=(0, (3, 2.5)),
            linewidth=0.8,
            zorder=0,
        )

    ax.set_yscale("log")
    ax.set_xlim(0, 1)
    ax.set_xticks(np.arange(0, 1.01, 0.20))
    ax.set_xticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1.0"])
    ax.set_xlabel("Quantile level")
    ax.set_ylabel("Serum cotinine (ng/mL)")
    finish_axes(ax)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.21),
        ncol=2,
        frameon=False,
        handlelength=2.5,
        columnspacing=2.0,
    )
    fig.subplots_adjust(top=0.79, bottom=0.18, left=0.17, right=0.98)
    save_figure(fig, QUANTILE_OUTPUT_STEM)


def silverman_bandwidth(values):
    """Return a robust Silverman bandwidth for univariate Gaussian KDE."""

    standard_deviation = np.std(values, ddof=1)
    interquartile_range = np.subtract(*np.percentile(values, [75, 25]))
    scale = min(standard_deviation, interquartile_range / 1.34)
    return 0.9 * scale * len(values) ** (-0.2)


def gaussian_kde(values, evaluation_grid, bandwidth):
    """Evaluate a Gaussian kernel density estimate in chunks."""

    standardized_distance = (evaluation_grid[:, None] - values[None, :]) / bandwidth
    kernel_values = np.exp(-0.5 * standardized_distance**2)
    return kernel_values.mean(axis=1) / (bandwidth * np.sqrt(2 * np.pi))


def plot_group_log_kdes(data, masks):
    """Plot group-specific KDEs of log10 serum cotinine."""

    log_values = {
        name: np.log10(data.loc[masks[name], "cotinine"].to_numpy())
        for name in GROUPS
    }
    lower_bound = min(values.min() for values in log_values.values()) - 0.10
    upper_bound = max(values.max() for values in log_values.values()) + 0.10
    evaluation_grid = np.linspace(lower_bound, upper_bound, 600)
    fig, ax = plt.subplots(figsize=(5.5, 3.55))

    for name, spec in GROUPS.items():
        values = log_values[name]
        density = gaussian_kde(values, evaluation_grid, silverman_bandwidth(values))
        ax.plot(
            evaluation_grid,
            density,
            color=spec["color"],
            linewidth=2.0,
            label=name,
        )

    ax.set_xlim(-2.1, 3.2)
    ax.set_xticks([-2, -1, 0, 1, 2, 3])
    ax.set_xlabel(r"$\log_{10}$ serum cotinine (ng/mL)")
    ax.set_ylabel("Kernel density")
    finish_axes(ax)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.22),
        ncol=2,
        frameon=False,
        handlelength=2.5,
        columnspacing=2.0,
    )
    fig.subplots_adjust(top=0.78, bottom=0.20, left=0.17, right=0.98)
    save_figure(fig, DISTRIBUTION_OUTPUT_STEM)


def main():
    data = pd.read_csv(
        DATA_PATH,
        usecols=["cotinine", "raceA", "raceB", "raceM", "raceNA"],
    )
    masks = group_masks(data)
    set_journal_style()
    plot_group_quantiles(data, masks)
    plot_group_log_kdes(data, masks)


if __name__ == "__main__":
    main()
