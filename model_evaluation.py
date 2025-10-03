"""Model evaluation utilities for sales forecasting predictions.

This module provides functions to compute quantitative error metrics for a
collection of forecasting models, generate comparison visualizations, and
assemble a narrative performance report that summarizes the relative accuracy
of the models.  It is designed to work with the forecasting helpers defined in
``forecasting_models.py`` but can be used with any dictionary of model
predictions indexed by ``datetime``.
"""

from __future__ import annotations

import math
import pathlib
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DEFAULT_OUTPUT_DIR = pathlib.Path("outputs")
DEFAULT_REPORT_PATH = pathlib.Path("performance_report.md")


def _prepare_actual_series(actual: pd.Series) -> pd.Series:
    """Validate and standardize the ground-truth target series.

    Parameters
    ----------
    actual:
        Series or single-column DataFrame containing the observed demand values.

    Returns
    -------
    pandas.Series
        Cleaned series indexed by ``DatetimeIndex`` sorted in ascending order.
    """

    if isinstance(actual, pd.DataFrame):
        if actual.shape[1] != 1:
            raise ValueError("actual must be a Series or single-column DataFrame")
        actual = actual.iloc[:, 0]

    if not isinstance(actual.index, pd.DatetimeIndex):
        raise TypeError("actual series must have a DatetimeIndex for plotting")

    return actual.sort_index()


def _align_series(actual: pd.Series, predicted: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Return aligned actual and predicted series with overlapping timestamps.

    Parameters
    ----------
    actual:
        Prepared ground-truth series indexed by datetime.
    predicted:
        Forecast series indexed by datetime.

    Returns
    -------
    tuple[pandas.Series, pandas.Series]
        Synchronized actual and predicted series restricted to shared timestamps.
    """

    if not isinstance(predicted.index, pd.DatetimeIndex):
        raise TypeError("predicted series must have a DatetimeIndex for plotting")

    aligned_actual, aligned_pred = actual.align(predicted.sort_index(), join="inner")
    aligned_actual = aligned_actual.astype(float)
    aligned_pred = aligned_pred.astype(float)

    if aligned_actual.empty:
        raise ValueError("No overlapping timestamps between actuals and predictions")

    return aligned_actual, aligned_pred


def calculate_metrics(actual: pd.Series, predictions: Dict[str, pd.Series]) -> pd.DataFrame:
    """Compute regression accuracy metrics for each model.

    Parameters
    ----------
    actual:
        Series containing the ground-truth values indexed by ``DatetimeIndex``.
    predictions:
        Mapping of model names to their predicted values indexed by
        ``DatetimeIndex``.

    Returns
    -------
    pandas.DataFrame
        Table with the metrics for each model sorted by RMSE.
    """

    prepared_actual = _prepare_actual_series(actual)

    records = []
    for name, pred in predictions.items():
        aligned_actual, aligned_pred = _align_series(prepared_actual, pred)

        mae = mean_absolute_error(aligned_actual, aligned_pred)
        rmse = mean_squared_error(aligned_actual, aligned_pred, squared=False)

        non_zero_mask = aligned_actual != 0
        if non_zero_mask.any():
            mape = np.mean(
                np.abs(
                    (aligned_actual[non_zero_mask] - aligned_pred[non_zero_mask])
                    / aligned_actual[non_zero_mask]
                )
            )
            mape *= 100.0
        else:
            mape = np.nan

        r2 = r2_score(aligned_actual, aligned_pred)

        records.append(
            {
                "Model": name,
                "MAE": mae,
                "RMSE": rmse,
                "MAPE": mape,
                "R2": r2,
            }
        )

    metrics_df = pd.DataFrame(records).sort_values("RMSE").reset_index(drop=True)
    metrics_df["Rank"] = metrics_df.index + 1
    return metrics_df


def _ensure_output_dir(directory: pathlib.Path) -> pathlib.Path:
    """Create the output directory if it does not yet exist.

    Parameters
    ----------
    directory:
        Target folder for saving plots and tables.

    Returns
    -------
    pathlib.Path
        Resolved path pointing to the ensured directory.
    """

    directory.mkdir(parents=True, exist_ok=True)
    return directory


def plot_actual_vs_predictions(
    actual: pd.Series,
    predictions: Dict[str, pd.Series],
    *,
    best_model: str,
    output_dir: pathlib.Path = DEFAULT_OUTPUT_DIR,
) -> pathlib.Path:
    """Create a single line plot comparing actual vs all model predictions.

    Parameters
    ----------
    actual:
        Ground-truth demand series for the evaluation window.
    predictions:
        Mapping of model names to forecast series.
    best_model:
        Name of the top-ranked model to highlight in the visualization.
    output_dir:
        Directory where the generated plot will be saved.

    Returns
    -------
    pathlib.Path
        Path to the saved line chart image.
    """

    output_dir = _ensure_output_dir(output_dir)
    prepared_actual = _prepare_actual_series(actual)

    plt.figure(figsize=(14, 7))
    sns.lineplot(x=prepared_actual.index, y=prepared_actual.values, label="Actual", linewidth=2.5)

    for name, pred in predictions.items():
        aligned_actual, aligned_pred = _align_series(prepared_actual, pred)
        line_width = 2.5 if name == best_model else 1.5
        alpha = 1.0 if name == best_model else 0.8
        sns.lineplot(x=aligned_pred.index, y=aligned_pred.values, label=name, linewidth=line_width, alpha=alpha)

    plt.title("Actual vs. Model Predictions")
    plt.xlabel("Date")
    plt.ylabel("Quantity Sold")
    plt.legend()
    plt.tight_layout()

    output_path = output_dir / "actual_vs_predictions.png"
    plt.savefig(output_path)
    plt.close()
    return output_path


def plot_error_histograms(
    actual: pd.Series,
    predictions: Dict[str, pd.Series],
    *,
    output_dir: pathlib.Path = DEFAULT_OUTPUT_DIR,
) -> pathlib.Path:
    """Create side-by-side histograms showing error distributions for each model.

    Parameters
    ----------
    actual:
        Ground-truth demand series.
    predictions:
        Mapping of model names to predicted values.
    output_dir:
        Directory where the histogram grid will be saved.

    Returns
    -------
    pathlib.Path
        Path to the histogram figure.
    """

    prepared_actual = _prepare_actual_series(actual)
    num_models = len(predictions)
    output_dir = _ensure_output_dir(output_dir)

    fig, axes = plt.subplots(1, num_models, figsize=(4 * num_models, 4), sharey=True)
    if num_models == 1:
        axes = [axes]

    for ax, (name, pred) in zip(axes, predictions.items()):
        aligned_actual, aligned_pred = _align_series(prepared_actual, pred)
        errors = aligned_actual - aligned_pred
        sns.histplot(errors, bins=20, kde=True, ax=ax, color="#457b9d")
        ax.set_title(f"Errors: {name}")
        ax.set_xlabel("Actual - Predicted")
        ax.axvline(0, color="black", linestyle="--", linewidth=1)

    plt.tight_layout()
    output_path = output_dir / "error_histograms.png"
    plt.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_rmse_comparison(metrics_df: pd.DataFrame, *, output_dir: pathlib.Path = DEFAULT_OUTPUT_DIR) -> pathlib.Path:
    """Generate a bar chart comparing RMSE across models.

    Parameters
    ----------
    metrics_df:
        DataFrame produced by :func:`calculate_metrics` with RMSE values per model.
    output_dir:
        Directory where the bar chart will be stored.

    Returns
    -------
    pathlib.Path
        Path to the RMSE comparison figure.
    """

    output_dir = _ensure_output_dir(output_dir)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=metrics_df, x="Model", y="RMSE", palette="viridis")
    plt.title("RMSE Comparison by Model")
    plt.xlabel("Model")
    plt.ylabel("RMSE")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    output_path = output_dir / "rmse_comparison.png"
    plt.savefig(output_path)
    plt.close()
    return output_path


def render_metrics_table(metrics_df: pd.DataFrame, *, output_dir: pathlib.Path = DEFAULT_OUTPUT_DIR) -> pathlib.Path:
    """Render the metrics DataFrame as an image table.

    Parameters
    ----------
    metrics_df:
        Table of evaluation metrics and model ranks.
    output_dir:
        Directory where the rendered table image should be saved.

    Returns
    -------
    pathlib.Path
        Path to the saved metrics table image.
    """

    output_dir = _ensure_output_dir(output_dir)
    display_df = metrics_df.copy()
    display_df[["MAE", "RMSE", "MAPE", "R2"]] = display_df[["MAE", "RMSE", "MAPE", "R2"]].applymap(
        lambda x: np.nan if pd.isna(x) else round(float(x), 4)
    )

    fig, ax = plt.subplots(figsize=(max(8, len(display_df) * 2.2), 1.2 + 0.4 * len(display_df)))
    ax.axis("off")
    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)

    output_path = output_dir / "metrics_table.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def identify_best_model(metrics_df: pd.DataFrame) -> str:
    """Return the name of the best performing model (lowest RMSE).

    Parameters
    ----------
    metrics_df:
        DataFrame of evaluation metrics sorted or sortable by RMSE.

    Returns
    -------
    str
        Name of the model with the minimum RMSE value.
    """

    if metrics_df.empty:
        raise ValueError("metrics_df must contain at least one model")
    return metrics_df.sort_values("RMSE").iloc[0]["Model"]


def _generate_sales_insights(actual: pd.Series) -> Dict[str, str]:
    """Create simple descriptive insights about the sales time series.

    Parameters
    ----------
    actual:
        Ground-truth series for the evaluation horizon.

    Returns
    -------
    dict[str, str]
        Mapping of insight headings to explanatory text.
    """

    prepared_actual = _prepare_actual_series(actual)
    total_quantity = float(prepared_actual.sum())
    average_daily = float(prepared_actual.mean())
    peak_date = prepared_actual.idxmax()
    peak_quantity = float(prepared_actual.max())
    recent_trend = prepared_actual.iloc[-1] - prepared_actual.iloc[0]

    if recent_trend > 0:
        trend_text = (
            "Demand trended upward by "
            f"{recent_trend:,.2f} units between the start and end of the evaluation period."
        )
    elif recent_trend < 0:
        trend_text = (
            "Demand declined by "
            f"{abs(recent_trend):,.2f} units between the start and end of the evaluation period."
        )
    else:
        trend_text = "Demand remained stable across the evaluation window."

    insights = {
        "Total quantity": f"Total quantity sold in the evaluated window: {total_quantity:,.0f} units.",
        "Average daily": f"Average daily quantity: {average_daily:,.2f} units.",
        "Peak day": f"Peak demand on {peak_date.date()}: {peak_quantity:,.0f} units.",
        "Trend": trend_text,
    }
    return insights


def generate_performance_report(
    metrics_df: pd.DataFrame,
    best_model: str,
    actual: pd.Series,
    *,
    report_path: pathlib.Path = DEFAULT_REPORT_PATH,
) -> pathlib.Path:
    """Create a markdown report summarizing model performance and insights.

    Parameters
    ----------
    metrics_df:
        Table of evaluation metrics including model ranks.
    best_model:
        Name of the model recommended for deployment.
    actual:
        Ground-truth series used for generating descriptive insights.
    report_path:
        Destination path for the markdown report.

    Returns
    -------
    pathlib.Path
        Path to the written markdown report.
    """

    report_path = report_path.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_sorted = metrics_df.sort_values("Rank")
    insights = _generate_sales_insights(actual)

    lines = [
        "# Sales Forecasting Model Performance Report",
        "",
        "## Model Rankings",
    ]

    for _, row in metrics_sorted.iterrows():
        if math.isnan(row["MAPE"]):
            metrics_summary = (
                f"MAE: {row['MAE']:.2f}, RMSE: {row['RMSE']:.2f}, MAPE: N/A, R²: {row['R2']:.3f}"
            )
        else:
            metrics_summary = (
                f"MAE: {row['MAE']:.2f}, RMSE: {row['RMSE']:.2f}, MAPE: {row['MAPE']:.2f}%, R²: {row['R2']:.3f}"
            )

        lines.append(f"{int(row['Rank'])}. **{row['Model']}** – {metrics_summary}")

    lines.extend(
        [
            "",
            f"**Top Recommendation:** The **{best_model}** model achieved the lowest RMSE and is the primary recommendation for deployment.",
            "",
            "## Metric Explanations",
            "- **MAE (Mean Absolute Error):** Average magnitude of the errors without considering their direction.",
            "- **RMSE (Root Mean Squared Error):** Quadratic mean of residuals that penalizes larger errors more heavily.",
            "- **MAPE (Mean Absolute Percentage Error):** Average absolute percentage difference between predicted and actual values.",
            "- **R² Score:** Proportion of variance in the target explained by the model (closer to 1 indicates better fit).",
            "",
            "## Model Recommendations",
            f"- **Primary Model:** Use **{best_model}** for operational forecasts due to its leading RMSE and balanced error profile.",
            "- **Monitoring:** Track MAE and RMSE each week to ensure model stability. Retrain models if RMSE increases by more than 10%.",
            "- **Ensemble Consideration:** Consider blending the top two models if their error profiles are complementary to reduce variance further.",
            "",
            "## Sales Data Insights",
        ]
    )

    for insight in insights.values():
        lines.append(f"- {insight}")

    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def evaluate_models(
    actual: pd.Series,
    predictions: Dict[str, pd.Series],
    *,
    output_dir: pathlib.Path = DEFAULT_OUTPUT_DIR,
    report_path: pathlib.Path = DEFAULT_REPORT_PATH,
) -> pd.DataFrame:
    """Run the full evaluation workflow and return the metrics DataFrame.

    Parameters
    ----------
    actual:
        Ground-truth demand values for the forecast horizon.
    predictions:
        Mapping of model names to their predicted series.
    output_dir:
        Directory where diagnostic plots and tables will be saved.
    report_path:
        Location to write the markdown performance report.

    Returns
    -------
    pandas.DataFrame
        Evaluation metrics with model rankings.
    """

    metrics_df = calculate_metrics(actual, predictions)
    best_model = identify_best_model(metrics_df)

    plot_actual_vs_predictions(actual, predictions, best_model=best_model, output_dir=output_dir)
    plot_error_histograms(actual, predictions, output_dir=output_dir)
    plot_rmse_comparison(metrics_df, output_dir=output_dir)
    render_metrics_table(metrics_df, output_dir=output_dir)
    generate_performance_report(metrics_df, best_model, actual, report_path=report_path)

    return metrics_df


__all__ = [
    "calculate_metrics",
    "plot_actual_vs_predictions",
    "plot_error_histograms",
    "plot_rmse_comparison",
    "render_metrics_table",
    "identify_best_model",
    "generate_performance_report",
    "evaluate_models",
]
