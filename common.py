"""
common.py - settings and helpers shared by every script.

Keeping the feature list in one place guarantees that training, the LSTM,
the plots and the Streamlit app all use exactly the same inputs.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

CELLS = ["B0005", "B0006", "B0007", "B0018"]

# ---------------------------------------------------------------- paths
# All paths are built with pathlib, starting from the folder this file lives
# in. pathlib uses the right separator on each OS ("\\" on Windows, "/" on
# macOS/Linux), and anchoring at __file__ means the scripts work no matter
# which folder you run them from.
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"
FIGURES_DIR = BASE_DIR / "figures"
CYCLES_CSV = DATA_DIR / "cycles.csv"
METRICS_CSV = RESULTS_DIR / "metrics.csv"
PREDICTIONS_CSV = RESULTS_DIR / "predictions.csv"

EOL_SOH = 75.0  # % - same threshold used in prepare_data.py

# Input features. NOTE: "capacity" is deliberately NOT here - SOH is computed
# directly from capacity, so using it would leak the answer into the input.
FEATURES = [
    "cycle", "discharge_time", "avg_voltage", "min_voltage",
    "time_to_3v5", "max_temp", "avg_temp", "time_to_max_temp",
]
TARGETS = ["soh", "rul"]
MODEL_NAMES = ["Linear Regression", "Random Forest", "XGBoost", "LSTM"]


def model_file(target, model_name, ext="joblib"):
    """models/soh_Random_Forest.joblib, models/rul_LSTM.keras, ..."""
    return MODELS_DIR / f"{target}_{model_name.replace(' ', '_')}.{ext}"


# ---------------------------------------------------------------- CSV I/O
# Every CSV read/write in the project goes through these two functions.
# - encoding="utf-8": Windows would otherwise use its local code page.
# - lineterminator="\n": the same bytes on every OS. (Given a file path,
#   pandas opens the file itself with newline="", which is what the Python
#   csv docs require, so no blank lines appear between rows on Windows.)
def read_csv(path, **kwargs):
    return pd.read_csv(path, encoding="utf-8", **kwargs)


def write_csv(df, path, **kwargs):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, encoding="utf-8", lineterminator="\n", **kwargs)


def score(y_true, y_pred):
    """The three metrics asked for: MAE, RMSE and R2."""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
    }


def save_results(metrics_rows, prediction_frames, model_names):
    """Write metrics/predictions, replacing old rows for `model_names` only.

    This lets train_models.py and train_lstm.py both write into the same two
    files, in any order, and be re-run safely without creating duplicates.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    new_metrics = pd.DataFrame(metrics_rows)
    new_preds = pd.concat(prediction_frames, ignore_index=True)

    for path, new in [(METRICS_CSV, new_metrics), (PREDICTIONS_CSV, new_preds)]:
        if path.exists():
            # float_precision="round_trip" reads the other models' rows back
            # EXACTLY as they were written, so re-saving them does not change
            # their 16th digit. (Only used here, for stored results - model
            # inputs keep pandas' default parser so the results stay as before.)
            old = read_csv(path, float_precision="round_trip")
            old = old[~old["model"].isin(model_names)]  # drop stale rows
            new = pd.concat([old, new], ignore_index=True)
        write_csv(new, path, index=False)


def summary_table(metrics):
    """Average MAE/RMSE/R2 over the four test cells, per target and model."""
    order = {m: i for i, m in enumerate(MODEL_NAMES)}
    table = (metrics.groupby(["target", "model"])[["MAE", "RMSE", "R2"]]
             .mean().reset_index())
    table["_o"] = table["model"].map(order)
    table["_t"] = table["target"].map({t: i for i, t in enumerate(TARGETS)})
    return (table.sort_values(["_t", "_o"]).drop(columns=["_o", "_t"])
            .reset_index(drop=True))
