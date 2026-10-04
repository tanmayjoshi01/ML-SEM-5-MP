"""
STEP 3 - train_lstm.py
======================
A small LSTM (Long Short-Term Memory) neural network that looks at the LAST 10
CYCLES of a battery, instead of a single cycle, to predict SOH and RUL.

Why an LSTM?  Degradation is a trend over time. An LSTM reads a sequence step
by step and keeps an internal memory, so it can use how the features have
been CHANGING, not just their current value.

Architecture (kept small because we only have ~630 cycles in total):
    input (10 cycles x 8 features) -> LSTM(32 units) -> Dense(1 output)

Evaluation is the same leave-one-battery-out scheme as train_models.py.
Important: the StandardScalers are fitted on the 3 TRAINING cells only, so no
information about the test cell leaks into training.

Results are appended to results/metrics.csv and results/predictions.csv
(as model "LSTM"), and the final models are saved in ./models.
"""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # hide TensorFlow info logs

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from common import (CELLS, CYCLES_CSV, FEATURES, MODELS_DIR, TARGETS,
                    METRICS_CSV, model_file, read_csv, save_results, score,
                    summary_table)

WINDOW = 10    # number of past cycles the LSTM sees
EPOCHS = 150
BATCH = 32
SEED = 42


def make_windows(cell_df, x_scaler):
    """Build sliding windows for ONE cell.

    For cycle k the window holds cycles k-9 ... k. The first 9 cycles do not
    have 9 earlier cycles, so we pad by repeating the first cycle. This way
    every cycle gets a prediction and the LSTM is scored on exactly the same
    cycles as the other models.
    """
    x = x_scaler.transform(cell_df[FEATURES])
    padded = np.vstack([np.repeat(x[:1], WINDOW - 1, axis=0), x])
    return np.stack([padded[i:i + WINDOW] for i in range(len(x))])


def build_model():
    """One LSTM layer + one Dense output layer, as specified."""
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(WINDOW, len(FEATURES))),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return model


def fit_lstm(train_df, target):
    """Fit scalers + LSTM on the given training cells. Returns all three."""
    x_scaler = StandardScaler().fit(train_df[FEATURES])
    # The target is scaled too (RUL goes up to ~125); the network learns faster
    # on values around 0 and we convert back with inverse_transform.
    y_scaler = StandardScaler().fit(train_df[[target]])

    xs, ys = [], []
    for _, cell_df in train_df.groupby("battery"):  # windows never cross cells
        xs.append(make_windows(cell_df, x_scaler))
        ys.append(y_scaler.transform(cell_df[[target]]))
    x, y = np.concatenate(xs), np.concatenate(ys)

    tf.keras.utils.set_random_seed(SEED)  # reproducible weights + shuffling
    model = build_model()
    model.fit(x, y, epochs=EPOCHS, batch_size=BATCH, verbose=0, shuffle=True)
    return model, x_scaler, y_scaler


def predict(model, x_scaler, y_scaler, cell_df):
    x = make_windows(cell_df, x_scaler)
    # Calling the model directly (instead of model.predict) avoids a
    # TensorFlow "retracing" warning when we predict many times in a loop.
    y_scaled = model(x.astype("float32"), training=False).numpy()
    return y_scaler.inverse_transform(y_scaled).ravel()


def main():
    df = read_csv(CYCLES_CSV).sort_values(["battery", "cycle"])
    metrics_rows, prediction_frames = [], []

    for target in TARGETS:
        for test_cell in CELLS:
            train = df[df["battery"] != test_cell]
            test = df[df["battery"] == test_cell]
            model, xs, ys = fit_lstm(train, target)
            pred = predict(model, xs, ys, test)

            metrics_rows.append({"target": target, "test_cell": test_cell,
                                 "model": "LSTM", **score(test[target], pred)})
            prediction_frames.append(pd.DataFrame({
                "battery": test_cell, "cycle": test["cycle"].values,
                "target": target, "model": "LSTM",
                "actual": test[target].values, "predicted": pred,
            }))
            print(f"  {target.upper()} test={test_cell}  "
                  f"MAE={metrics_rows[-1]['MAE']:.3f}")

    save_results(metrics_rows, prediction_frames, ["LSTM"])

    # Final LSTM per target, trained on all four cells, for the app
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for target in TARGETS:
        model, xs, ys = fit_lstm(df, target)
        model.save(str(model_file(target, "LSTM", ext="keras")))  # Keras wants a str
        joblib.dump({"x_scaler": xs, "y_scaler": ys, "window": WINDOW},
                    model_file(target, "LSTM_scalers"))

    metrics = read_csv(METRICS_CSV)
    pd.set_option("display.width", 200)
    print("\nLSTM per test cell:")
    print(metrics[metrics["model"] == "LSTM"].round(3).to_string(index=False))
    print("\nAll models, averaged over the four test cells:")
    print(summary_table(metrics).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
