"""
STEP 4 - make_plots.py
======================
Creates every figure for the report as a PNG in ./figures. All numbers come
from data/cycles.csv, results/*.csv and the saved models - nothing is typed in
by hand.

Figures:
    1_capacity_fade.png          SOH vs cycle for all cells + 75% EOL line
    2_correlation_heatmap.png    how strongly each feature relates to SOH/RUL
    3_actual_vs_pred_soh.png     actual vs predicted SOH (model x test cell)
    4_actual_vs_pred_rul.png     actual vs predicted RUL (model x test cell)
    5_feature_importance.png     Random Forest and XGBoost importances
    6_model_comparison.png       average MAE / RMSE of the four models
    7_rul_error_by_stage.png     RUL error in early / middle / late life
Also writes results/rul_error_by_stage.csv (the numbers behind figure 7).
"""
import joblib
import matplotlib

# "Agg" draws straight to image files and needs no screen/GUI. It must be set
# BEFORE pyplot is imported. This makes the script work on any OS, including
# servers and terminals without a display.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (CELLS, CYCLES_CSV, EOL_SOH, FEATURES, FIGURES_DIR,
                    METRICS_CSV, MODEL_NAMES, PREDICTIONS_CSV, RESULTS_DIR,
                    model_file, read_csv, summary_table, write_csv)

# ------------------------------------------------------------- style
# A colour-blind-checked categorical palette, used in a FIXED order so a given
# cell / model always has the same colour in every figure.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
CELL_COLOR = dict(zip(CELLS, PALETTE))
MODEL_COLOR = dict(zip(MODEL_NAMES, PALETTE))
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DIVERGING = "RdBu_r"  # blue = negative, white = 0, red = positive

plt.rcParams.update({
    "figure.dpi": 150, "savefig.bbox": "tight", "font.size": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "lines.linewidth": 2,
    "legend.frameon": False,
})


def save(fig, name):
    path = FIGURES_DIR / name
    fig.savefig(path)
    plt.close(fig)
    print("saved", path.relative_to(FIGURES_DIR.parent))


# ------------------------------------------------------------- figures
def capacity_fade(df):
    """SOH of each cell over its life, with the 75% end-of-life threshold."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for cell in CELLS:
        d = df[df.battery == cell]
        ax.plot(d.cycle, d.soh, color=CELL_COLOR[cell], label=cell)
        # mark where this cell's EOL was detected
        eol = d.eol_cycle.iloc[0]
        ax.plot(eol, d.loc[d.cycle == eol, "soh"].iloc[0], "o", ms=8,
                color=CELL_COLOR[cell], mec="white", mew=2)
    ax.axhline(EOL_SOH, color=INK, ls="--", lw=1.2)
    ax.text(0, EOL_SOH - 0.8, f"EOL threshold ({EOL_SOH:.0f}% SOH)",
            ha="left", va="top", color=INK)
    ax.set(xlabel="Discharge cycle", ylabel="State of Health (%)",
           title="Capacity fade of the four NASA cells (dots = detected EOL)")
    ax.legend(ncol=4, loc="upper right", bbox_to_anchor=(1, 0.93))
    save(fig, "1_capacity_fade.png")


def correlation_heatmap(df):
    """Pearson correlation between every feature and both targets."""
    cols = FEATURES + ["soh", "rul"]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(corr, cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
    ax.set_yticks(range(len(cols)), cols)
    ax.grid(False)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.6 else INK)
    fig.colorbar(im, ax=ax, fraction=0.046, label="Pearson correlation")
    ax.set_title("Correlation between features, SOH and RUL")
    save(fig, "2_correlation_heatmap.png")


def actual_vs_predicted(preds, target, fname, unit):
    """Grid: one row per model, one column per test cell."""
    p = preds[preds.target == target]
    fig, axes = plt.subplots(len(MODEL_NAMES), len(CELLS), figsize=(14, 11),
                             sharex="col", sharey=True)
    for r, model in enumerate(MODEL_NAMES):
        for c, cell in enumerate(CELLS):
            ax = axes[r, c]
            d = p[(p.model == model) & (p.battery == cell)].sort_values("cycle")
            ax.plot(d.cycle, d.actual, color=INK, lw=1.5, label="Actual")
            ax.plot(d.cycle, d.predicted, color=MODEL_COLOR[model], label="Predicted")
            mae = np.mean(np.abs(d.actual - d.predicted))
            ax.text(0.97, 0.94, f"MAE {mae:.2f}", transform=ax.transAxes,
                    ha="right", va="top", color=MUTED, fontsize=9)
            if r == 0:
                ax.set_title(f"Test cell {cell}")
            if c == 0:
                ax.set_ylabel(f"{model}\n{target.upper()} ({unit})")
            if r == len(MODEL_NAMES) - 1:
                ax.set_xlabel("Cycle")
    handles = [plt.Line2D([], [], color=INK, label="Actual")] + [
        plt.Line2D([], [], color=MODEL_COLOR[m], label=f"Predicted ({m})")
        for m in MODEL_NAMES]
    fig.legend(handles=handles, loc="upper center", ncol=5,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"Actual vs predicted {target.upper()} - leave-one-battery-out",
                 y=1.05, fontweight="bold")
    fig.tight_layout()
    save(fig, fname)


def feature_importance():
    """Importances stored inside the final (all-cell) RF and XGBoost models."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for r, target in enumerate(["soh", "rul"]):
        for c, model_name in enumerate(["Random Forest", "XGBoost"]):
            model = joblib.load(model_file(target, model_name))
            imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
            ax = axes[r, c]
            ax.barh(imp.index, imp.values, color=MODEL_COLOR[model_name], height=0.6)
            for y, v in enumerate(imp.values):
                ax.text(v, y, f" {v:.2f}", va="center", fontsize=8, color=MUTED)
            ax.set_title(f"{model_name} - {target.upper()}")
            ax.set_xlim(0, imp.max() * 1.18)
            ax.grid(axis="y", visible=False)
    fig.suptitle("Feature importance (final models trained on all four cells)",
                 fontweight="bold")
    fig.tight_layout()
    save(fig, "5_feature_importance.png")


def model_comparison(metrics):
    """Average MAE and RMSE per model, separately for SOH and RUL."""
    table = summary_table(metrics)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, target, unit in zip(axes, ["soh", "rul"], ["% SOH", "cycles"]):
        t = table[table.target == target].set_index("model").loc[MODEL_NAMES]
        x = np.arange(len(MODEL_NAMES))
        w = 0.38
        for k, (metric, shade) in enumerate([("MAE", 1.0), ("RMSE", 0.55)]):
            bars = ax.bar(x + (k - 0.5) * w, t[metric], w * 0.92,
                          color=[MODEL_COLOR[m] for m in MODEL_NAMES], alpha=shade,
                          label=metric)
            ax.bar_label(bars, fmt="%.2f", fontsize=8, color=MUTED, padding=2)
        ax.set_xticks(x, MODEL_NAMES)
        ax.set(ylabel=f"Error ({unit})",
               title=f"{target.upper()}: mean over 4 test cells")
        ax.grid(axis="x", visible=False)
    fig.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, color=MUTED, alpha=1.0, label="MAE (solid bar)"),
        plt.Rectangle((0, 0), 1, 1, color=MUTED, alpha=0.55, label="RMSE (light bar)")],
        loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Model comparison (lower is better)", fontweight="bold")
    fig.tight_layout()
    save(fig, "6_model_comparison.png")


def rul_error_by_stage(preds, df):
    """Is RUL harder to predict early in life or close to the end?

    Stage is set by life fraction = cycle / eol_cycle of that cell:
        early  < 1/3,  middle 1/3 - 2/3,  late >= 2/3 (includes cycles past EOL)
    """
    p = preds[preds.target == "rul"].merge(
        df[["battery", "cycle", "eol_cycle"]], on=["battery", "cycle"])
    frac = p.cycle / p.eol_cycle
    p["stage"] = pd.cut(frac, [0, 1 / 3, 2 / 3, np.inf],
                        labels=["Early", "Middle", "Late"], right=False)
    p["abs_err"] = (p.actual - p.predicted).abs()
    table = (p.groupby(["stage", "model"], observed=True)["abs_err"].mean()
             .unstack("model")[MODEL_NAMES])
    write_csv(table.round(3), RESULTS_DIR / "rul_error_by_stage.csv")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(table.index))
    w = 0.2
    for k, model in enumerate(MODEL_NAMES):
        bars = ax.bar(x + (k - 1.5) * w, table[model], w * 0.9,
                      color=MODEL_COLOR[model], label=model)
        ax.bar_label(bars, fmt="%.1f", fontsize=8, color=MUTED, padding=2)
    ax.set_xticks(x, [f"{s} life" for s in table.index])
    ax.set(ylabel="RUL MAE (cycles)",
           title="RUL error by degradation stage (all test cells pooled)")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    save(fig, "7_rul_error_by_stage.png")
    return table


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = read_csv(CYCLES_CSV)
    metrics = read_csv(METRICS_CSV)
    preds = read_csv(PREDICTIONS_CSV)

    capacity_fade(df)
    correlation_heatmap(df)
    actual_vs_predicted(preds, "soh", "3_actual_vs_pred_soh.png", "%")
    actual_vs_predicted(preds, "rul", "4_actual_vs_pred_rul.png", "cycles")
    feature_importance()
    model_comparison(metrics)
    stage = rul_error_by_stage(preds, df)
    print("\nRUL MAE (cycles) by degradation stage:")
    print(stage.round(2).to_string())


if __name__ == "__main__":
    main()
