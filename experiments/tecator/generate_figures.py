#!/usr/bin/env python3
"""Create publication-ready Tecator FAIR-NN figures from saved result files."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent

METHOD_COLORS = {
    "RLR": "#4E5D6C",
    "KRR-gaussian": "#2E86AB",
    "KRR-rbf": "#2E86AB",
    "KRR-polynomial": "#5C7CFA",
    "KRR-linear": "#8C6D62",
    "FA-NN": "#159F8C",
    "FAIR-NN": "#D84A6B",
}


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "normal",
            "axes.linewidth": 0.7,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def channel_number(value: str) -> int:
    match = re.search(r"(\d+)$", str(value))
    if match is None:
        raise ValueError(f"Could not parse a spectral channel number from {value!r}")
    return int(match.group(1))


def save_figure(figure: plt.Figure, output_stem: Path) -> None:
    figure.savefig(output_stem.with_suffix(".pdf"))
    figure.savefig(output_stem.with_suffix(".png"), dpi=300)
    plt.close(figure)


def plot_covariate_correlation(data_path: Path, output_dir: Path) -> None:
    """Correlation heatmap and KDE of pairwise correlations of the spectral covariates."""
    frame = pd.read_csv(data_path)
    channels = sorted(
        [column for column in frame.columns if column.startswith("absorbance_")],
        key=channel_number,
    )
    matrix = frame[channels].to_numpy(dtype=float)
    correlation = np.corrcoef(matrix, rowvar=False)
    n_channels = correlation.shape[0]

    upper = correlation[np.triu_indices(n_channels, k=1)]

    # The spectral channels are so collinear that every pairwise correlation
    # falls in a narrow band near one. Focus both panels on that observed range
    # so the near-diagonal decay structure and the density peak are visible.
    low = float(np.floor(upper.min() * 100.0) / 100.0)

    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), constrained_layout=True)

    heat_axis = axes[0]
    image = heat_axis.imshow(
        correlation,
        origin="lower",
        cmap="magma",
        vmin=low,
        vmax=1.0,
        aspect="equal",
    )
    ticks = [0, 19, 39, 59, 79, 99]
    heat_axis.set_xticks(ticks)
    heat_axis.set_xticklabels([str(t + 1) for t in ticks])
    heat_axis.set_yticks(ticks)
    heat_axis.set_yticklabels([str(t + 1) for t in ticks])
    heat_axis.set_xlabel("Spectral channel")
    heat_axis.set_ylabel("Spectral channel")
    heat_axis.set_title("Pairwise correlation matrix", pad=6)
    colorbar = figure.colorbar(image, ax=heat_axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Correlation")

    grid = np.linspace(low, 1.0, 512)
    kde = _gaussian_kde(upper, grid)
    density_axis = axes[1]
    density_axis.plot(grid, kde, color="#2E86AB", linewidth=1.8)
    density_axis.fill_between(grid, kde, color="#2E86AB", alpha=0.18)
    mean_value = float(np.mean(upper))
    median_value = float(np.median(upper))
    density_axis.axvline(
        mean_value, color="#D84A6B", linestyle="--", linewidth=1.3,
        label=f"Mean = {mean_value:.3f}",
    )
    density_axis.axvline(
        median_value, color="#159F8C", linestyle=":", linewidth=1.3,
        label=f"Median = {median_value:.3f}",
    )
    density_axis.set_xlim(low, 1.0)
    density_axis.set_ylim(bottom=0.0)
    density_axis.set_xlabel("Pairwise correlation")
    density_axis.set_ylabel("Density")
    density_axis.set_title(
        f"Density of {upper.size} pairwise correlations", pad=6
    )
    density_axis.grid(axis="y", color="#DDE2E7", linewidth=0.55)
    density_axis.set_axisbelow(True)
    density_axis.spines[["top", "right"]].set_visible(False)
    density_axis.legend(frameon=False, loc="upper left")

    save_figure(figure, output_dir / "tecator_covariate_correlation")


def _gaussian_kde(samples: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Evaluate a Gaussian KDE with Silverman bandwidth on ``grid``."""
    samples = np.asarray(samples, dtype=float)
    size = samples.size
    std = float(np.std(samples, ddof=1))
    if std <= 0.0:
        std = 1.0
    bandwidth = 1.06 * std * size ** (-1.0 / 5.0)
    if bandwidth <= 0.0:
        bandwidth = 1e-3
    difference = (grid[:, None] - samples[None, :]) / bandwidth
    kernels = np.exp(-0.5 * difference ** 2) / np.sqrt(2.0 * np.pi)
    return kernels.sum(axis=1) / (size * bandwidth)


def stable_method_order(methods: list[str]) -> list[str]:
    base = ["RLR"]
    base.extend(sorted([name for name in methods if name.startswith("KRR-")]))
    base.extend(["FA-NN", "FAIR-NN"])
    return [name for name in base if name in methods]


def grouped_bars(
    axis: plt.Axes,
    metrics: pd.DataFrame,
    column: str,
    title: str,
    log_scale: bool = False,
) -> None:
    taus = sorted(metrics["tau"].unique())
    methods = stable_method_order(metrics["method"].unique().tolist())
    positions = np.arange(len(taus))
    width = 0.78 / len(methods)
    for index, method in enumerate(methods):
        values = [
            metrics.loc[
                (metrics["method"].eq(method)) & np.isclose(metrics["tau"], tau),
                column,
            ].iloc[0]
            for tau in taus
        ]
        offset = (index - (len(methods) - 1) / 2.0) * width
        axis.bar(
            positions + offset,
            values,
            width=width,
            color=METHOD_COLORS.get(method, "#6C757D"),
            edgecolor="white",
            linewidth=0.35,
            label=method,
        )
    axis.set_title(title, loc="left", pad=4)
    axis.set_xticks(positions, [f"{tau:.2f}" for tau in taus])
    axis.set_xlabel(r"Tail level $\tau$")
    axis.grid(axis="y", color="#DDE2E7", linewidth=0.55)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    if log_scale:
        axis.set_yscale("log")


def plot_metric_variance_overview(metrics: pd.DataFrame, output_dir: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(12.3, 7.0), constrained_layout=True)
    panels = [
        ("pseudo_outcome_loss", "(a) Pseudo-outcome loss", True),
        ("q_hat_variance", r"(b) $\mathrm{Var}_T(\widehat q_\tau)$", False),
        ("es_hat_variance", r"(c) $\mathrm{Var}_T(\widehat e_\tau)$", False),
        ("pseudo_outcome_variance", r"(d) $\mathrm{Var}_T(\widehat V_\tau)$", False),
        (
            "calibration_score_variance",
            r"(e) $\mathrm{Var}_T(\mathrm{calibration\ score})$",
            True,
        ),
    ]
    for axis, (column, title, log_scale) in zip(axes.flat, panels):
        grouped_bars(axis, metrics, column, title, log_scale)

    fair = metrics.loc[metrics["method"].eq("FAIR-NN")].sort_values("tau")
    axis = axes.flat[-1]
    taus = fair["tau"].to_numpy()
    means = fair["conditional_variance_hat_mean"].to_numpy()
    variances = fair["conditional_variance_hat_variance"].to_numpy()
    axis.errorbar(
        taus,
        means,
        yerr=np.sqrt(np.maximum(variances, 0.0)),
        color=METHOD_COLORS["FAIR-NN"],
        marker="o",
        markersize=5,
        linewidth=1.4,
        capsize=3,
    )
    axis.set_title(r"(f) FAIR-NN estimated conditional variance $\widehat h_\tau$", loc="left", pad=4)
    axis.set_xlabel(r"Tail level $\tau$")
    axis.set_ylabel(r"Test-set mean $\pm$ SD")
    axis.set_xticks(taus, [f"{tau:.2f}" for tau in taus])
    axis.grid(axis="y", color="#DDE2E7", linewidth=0.55)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(labels),
        frameon=False,
        bbox_to_anchor=(0.5, 1.04),
    )
    save_figure(figure, output_dir / "test_metric_variance_overview")


def plot_throughput_heatmap(throughput: pd.DataFrame, output_dir: Path) -> None:
    taus = sorted(throughput["tau"].unique())
    units = sorted(
        [column for column in throughput.columns if column.startswith("throughput_")],
        key=lambda value: int(value.rsplit("_", 1)[1]),
    )
    figure, axes = plt.subplots(
        len(taus),
        2,
        figsize=(10.5, 9.5),
        constrained_layout=True,
        width_ratios=[1.0, 1.2],
    )
    if len(taus) == 1:
        axes = np.asarray([axes])

    image = None
    for row, tau in enumerate(taus):
        subset = throughput.loc[np.isclose(throughput["tau"], tau)].copy()
        subset["channel_number"] = subset["channel"].map(channel_number)
        subset = subset.sort_values("channel_number")
        coefficients = np.abs(subset[units].to_numpy(dtype=float))
        maximum = float(coefficients.max())
        normalized = coefficients / maximum if maximum > 0.0 else coefficients

        heat_axis = axes[row, 0]
        image = heat_axis.imshow(
            normalized,
            aspect="auto",
            origin="lower",
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        heat_axis.set_title(
            rf"$\tau={tau:.2f}$: normalized $|\widehat\Gamma|$ (max={maximum:.3g})",
            loc="left",
            pad=4,
        )
        heat_axis.set_xlabel("Throughput unit")
        heat_axis.set_ylabel("Spectral channel")
        heat_axis.set_xticks(np.arange(len(units)), [item.rsplit("_", 1)[1] for item in units])
        heat_axis.set_yticks([0, 24, 49, 74, 99], ["1", "25", "50", "75", "100"])

        importance_axis = axes[row, 1]
        channels = subset["channel_number"].to_numpy()
        importance = subset["importance"].to_numpy(dtype=float)
        importance_axis.plot(channels, importance, color=METHOD_COLORS["FAIR-NN"], linewidth=1.25)
        top = subset.nsmallest(5, "importance_rank")
        importance_axis.scatter(
            top["channel_number"],
            top["importance"],
            s=14,
            color="#F59F00",
            zorder=3,
        )
        importance_axis.set_title(rf"$\tau={tau:.2f}$: $\|\widehat\Gamma_{{j\cdot}}\|_2$", loc="left", pad=4)
        importance_axis.set_xlabel("Spectral channel")
        importance_axis.set_ylabel("Importance")
        importance_axis.set_xlim(1, 100)
        importance_axis.grid(axis="y", color="#DDE2E7", linewidth=0.55)
        importance_axis.set_axisbelow(True)
        importance_axis.spines[["top", "right"]].set_visible(False)

    if image is not None:
        colorbar = figure.colorbar(image, ax=axes[:, 0], shrink=0.86, pad=0.02)
        colorbar.set_label(r"Normalized $|\widehat\Gamma_{jk}|$")
    save_figure(figure, output_dir / "fair_nn_throughput_heatmap")


def plot_tail_predictions(predictions: pd.DataFrame, output_dir: Path) -> None:
    fair = predictions.loc[predictions["method"].eq("FAIR-NN")].copy()
    taus = sorted(fair["tau"].unique())
    figure, axes = plt.subplots(1, len(taus), figsize=(12.3, 3.7), constrained_layout=True)
    if len(taus) == 1:
        axes = [axes]

    for index, tau in enumerate(taus):
        axis = axes[index]
        subset = fair.loc[np.isclose(fair["tau"], tau)].sort_values("q_hat").reset_index(drop=True)
        rank = np.arange(1, len(subset) + 1)
        exceedance = subset["fat_pct"].to_numpy() > subset["q_hat"].to_numpy()
        axis.scatter(
            rank[~exceedance],
            subset.loc[~exceedance, "fat_pct"],
            color="#4E5D6C",
            s=18,
            alpha=0.82,
            zorder=3,
            label="Observed fat" if index == 0 else None,
        )
        axis.scatter(
            rank[exceedance],
            subset.loc[exceedance, "fat_pct"],
            color="#F59F00",
            s=25,
            edgecolor="white",
            linewidth=0.35,
            zorder=4,
            label=r"$Y>\widehat q_\tau$" if index == 0 else None,
        )
        axis.plot(
            rank,
            subset["q_hat"],
            color="#2E86AB",
            linewidth=1.55,
            zorder=2,
            label=r"$\widehat q_\tau$" if index == 0 else None,
        )
        axis.plot(
            rank,
            subset["es_hat"],
            color=METHOD_COLORS["FAIR-NN"],
            linewidth=1.55,
            zorder=2,
            label=r"$\widehat e_\tau^+$" if index == 0 else None,
        )
        axis.set_title(
            rf"$\tau={tau:.2f}$; {int(exceedance.sum())} exceedances",
            loc="left",
            pad=4,
        )
        axis.set_xlabel(r"Test observation rank by $\widehat q_\tau$")
        axis.set_xticks([1, 22, 43])
        axis.grid(axis="y", color="#DDE2E7", linewidth=0.55)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        if index == 0:
            axis.set_ylabel("Fat content (%)")

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.07),
    )
    save_figure(figure, output_dir / "fair_nn_tail_predictions")


def plot_method_comparison(
    predictions: pd.DataFrame,
    output_dir: Path,
    value_column: str,
    output_name: str,
    title: str,
    y_label: str,
) -> None:
    """Plot all methods on the same FAIR-NN-ordered held-out observations."""

    styles = {
        "RLR": {"color": "#B6A4D1", "linestyle": "--", "linewidth": 1.10, "alpha": 0.78, "zorder": 1},
        "KRR-gaussian": {"color": "#D9B980", "linestyle": "--", "linewidth": 1.10, "alpha": 0.78, "zorder": 1},
        "FA-NN": {"color": "#159F8C", "linestyle": "-", "linewidth": 1.75, "alpha": 0.96, "zorder": 3},
        "FAIR-NN": {
            "color": "#D84A6B",
            "linestyle": "-",
            "linewidth": 2.65,
            "alpha": 1.00,
            "zorder": 5,
        },
    }
    labels = {"RLR": "RLR", "KRR-gaussian": "KRR", "FA-NN": "FA-NN", "FAIR-NN": "FAIR-NN"}
    methods = [method for method in ("RLR", "KRR-gaussian", "FA-NN", "FAIR-NN") if method in set(predictions["method"])]
    legend_order = ["FAIR-NN", "FA-NN", "KRR", "RLR", "Observed fat"]
    taus = sorted(predictions["tau"].unique())
    figure, axes = plt.subplots(1, len(taus), figsize=(12.3, 3.7), constrained_layout=True)
    if len(taus) == 1:
        axes = [axes]

    values = pd.concat([predictions["fat_pct"], predictions[value_column]], ignore_index=True)
    y_min = max(0.0, float(values.min()) - 1.5)
    y_max = float(values.max()) + 1.8
    for index, tau in enumerate(taus):
        axis = axes[index]
        fair = predictions.loc[
            predictions["method"].eq("FAIR-NN") & np.isclose(predictions["tau"], tau)
        ].sort_values("q_hat")
        sample_ids = fair["sample_id"].to_numpy()
        ranks = np.arange(1, len(sample_ids) + 1)
        axis.scatter(
            ranks,
            fair["fat_pct"],
            color="#4E5D6C",
            s=20,
            alpha=0.82,
            zorder=5,
            label="Observed fat" if index == 0 else None,
        )
        for method in methods:
            fitted = predictions.loc[
                predictions["method"].eq(method) & np.isclose(predictions["tau"], tau)
            ].set_index("sample_id").loc[sample_ids, value_column]
            axis.plot(
                ranks,
                fitted,
                **styles[method],
                label=labels[method] if index == 0 else None,
            )
        axis.set_title(rf"$\tau={tau:.2f}$", loc="left", pad=4)
        axis.set_xlabel(r"Test observation rank by FAIR-NN $\widehat q_\tau$")
        axis.set_xticks([1, 22, 43])
        axis.set_ylim(y_min, y_max)
        axis.grid(axis="y", color="#DDE2E7", linewidth=0.55)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        if index == 0:
            axis.set_ylabel(y_label)

    legend_handles, plotted_labels = axes[0].get_legend_handles_labels()
    handle_map = dict(zip(plotted_labels, legend_handles))
    handles = [handle_map[label] for label in legend_order if label in handle_map]
    legend_labels = [labels.get(label, label) for label in legend_order if label in handle_map]
    figure.legend(
        handles,
        legend_labels,
        loc="upper center",
        ncol=len(legend_labels),
        frameon=False,
        bbox_to_anchor=(0.5, 1.07),
    )
    figure.suptitle(title, x=0.015, y=1.01, ha="left", fontsize=10)
    save_figure(figure, output_dir / output_name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "results" / "fair_nn_upper_tail",
        help="Directory containing metrics.csv, test_predictions.csv, and throughput results.",
    )
    parser.add_argument(
        "--metrics-file",
        default="metrics.csv",
        help="Metrics CSV filename relative to --results-dir.",
    )
    parser.add_argument(
        "--predictions-file",
        default="test_predictions.csv",
        help="Prediction CSV filename relative to --results-dir.",
    )
    parser.add_argument(
        "--throughput-file",
        default="fair_nn_throughput_importance.csv",
        help="Throughput CSV filename relative to --results-dir.",
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        default=ROOT / "data" / "tecator_fair_nn_213.csv",
        help="Analysis CSV with the 100 absorbance covariates for the correlation figure.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for figure outputs; defaults to --results-dir/figures.",
    )
    args = parser.parse_args()
    results_dir = args.results_dir.resolve()
    figure_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else results_dir / "figures"
    )
    figure_dir.mkdir(parents=True, exist_ok=True)

    configure_matplotlib()
    metrics = pd.read_csv(results_dir / args.metrics_file)
    predictions = pd.read_csv(results_dir / args.predictions_file)
    throughput = pd.read_csv(results_dir / args.throughput_file)

    required_metrics = {
        "pseudo_outcome_loss",
        "q_hat_variance",
        "es_hat_variance",
        "pseudo_outcome_variance",
        "calibration_score_variance",
    }
    if not required_metrics.issubset(metrics.columns):
        raise ValueError("metrics.csv does not contain the required diagnostic columns")
    if not predictions["method"].eq("FAIR-NN").any():
        raise ValueError("test_predictions.csv does not contain FAIR-NN predictions")

    plot_metric_variance_overview(metrics, figure_dir)
    plot_covariate_correlation(args.data_file.resolve(), figure_dir)
    plot_throughput_heatmap(throughput, figure_dir)
    plot_tail_predictions(predictions, figure_dir)
    plot_method_comparison(
        predictions,
        figure_dir,
        "q_hat",
        "method_quantile_predictions",
        "Conditional quantile predictions on held-out T samples",
        "Conditional quantile / fat content (%)",
    )
    plot_method_comparison(
        predictions,
        figure_dir,
        "es_hat",
        "method_es_predictions",
        "Upper-tail conditional ES predictions on held-out T samples",
        "Conditional ES / fat content (%)",
    )
    print(f"Saved figures to {figure_dir}")


if __name__ == "__main__":
    main()
