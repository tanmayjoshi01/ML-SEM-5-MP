"""
STEP 2 - train_models.py
========================
Trains three classical ML models, each one separately for SOH and for RUL:
    1. Linear Regression (inputs standardised with StandardScaler)
    2. Random Forest
    3. XGBoost

Evaluation: LEAVE-ONE-BATTERY-OUT cross-validation.
    Train on 3 cells, test on the 4th, then rotate so every cell is the test
    cell once. This checks whether a model works on a battery it has NEVER
    seen - a random row split would leak neighbouring cycles of the same cell
    into training and give over-optimistic scores.

Outputs:
    results/metrics.csv      MAE, RMSE, R2 for every (target, test cell, model)
    results/predictions.csv  actual vs predicted value for every cycle
    models/*.joblib          final models re-trained on all four cells
"""
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from common import (CELLS, CYCLES_CSV, FEATURES, MODELS_DIR, TARGETS,
                    METRICS_CSV, model_file, read_csv, save_results, score,
                    summary_table)

SEED = 42  # fixed random seed so results are reproducible


def make_models():
    """Return fresh, untrained copies of the three models."""
    return {
        # Scaling matters for Linear Regression because features have very
        # different units (seconds vs volts); the pipeline fits the scaler on
        # the training data only, then applies it to the test data.
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        # Random Forest: average of 300 decision trees, each on a bootstrap sample
        "Random Forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, random_state=SEED, n_jobs=-1),
        # XGBoost: trees added one at a time, each correcting the previous errors
        "XGBoost": XGBRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=SEED),
    }


def main():
    df = read_csv(CYCLES_CSV)
    metrics_rows, prediction_frames = [], []

    for target in TARGETS:
        for test_cell in CELLS:
            train = df[df["battery"] != test_cell]
            test = df[df["battery"] == test_cell]

            for name, model in make_models().items():
                model.fit(train[FEATURES], train[target])
                pred = model.predict(test[FEATURES])

                metrics_rows.append({"target": target, "test_cell": test_cell,
                                     "model": name, **score(test[target], pred)})
                prediction_frames.append(pd.DataFrame({
                    "battery": test_cell, "cycle": test["cycle"].values,
                    "target": target, "model": name,
                    "actual": test[target].values, "predicted": pred,
                }))

    save_results(metrics_rows, prediction_frames, list(make_models()))

    # Final models: after evaluation, retrain on ALL four cells so the app
    # uses as much data as possible. These are not used for the metrics above.
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for target in TARGETS:
        for name, model in make_models().items():
            model.fit(df[FEATURES], df[target])
            joblib.dump(model, model_file(target, name))

    # Print results
    metrics = read_csv(METRICS_CSV)
    metrics = metrics[metrics["model"].isin(make_models())]
    pd.set_option("display.width", 200)
    print("Per test cell (leave-one-battery-out):")
    print(metrics.round(3).to_string(index=False))
    print("\nAveraged over the four test cells:")
    print(summary_table(metrics).round(3).to_string(index=False))
    print(f"\nSaved metrics, predictions and {len(TARGETS) * 3} models.")


if __name__ == "__main__":
    main()
