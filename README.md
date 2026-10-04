# EV Battery State of Health (SOH) and Remaining Useful Life (RUL) Estimation Using Machine Learning

This project predicts how healthy a lithium-ion cell is (SOH, in %) and how many
discharge cycles it has left before end of life (RUL, in cycles). It uses the NASA
PCoE Li-ion Battery Aging dataset, cells **B0005, B0006, B0007 and B0018**.

Four models are compared: Linear Regression, Random Forest, XGBoost and a small LSTM.
Every model is tested on a battery it has never seen (leave-one-battery-out).

## Requirements

- **Python 3.12 or 3.13, 64-bit.** numpy, scipy and xgboost need at least 3.12, and TensorFlow 2.21 supports at most 3.13.
- **Windows 10/11, macOS on Apple Silicon, or Linux.** Every pinned package has a prebuilt wheel for these, so nothing needs compiling.
- Intel Macs are not supported, because TensorFlow 2.21 has no macOS x86_64 wheel.
- `git` is needed only if the data files have to be downloaded.

## Quick start: one command

The run-all scripts create the virtual environment, install the requirements,
download the data if it is missing, run the four scripts in order and start the app.

```bash
bash run_all.sh        # macOS / Linux
```

```bat
run_all.bat            :: Windows (or double-click it in Explorer)
```

## Step by step: macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python prepare_data.py    # Step 1: data/cycles.csv
python train_models.py    # Step 2: LR, RF, XGBoost -> results/, models/
python train_lstm.py      # Step 3: LSTM -> appended to results/, models/  (~3 min on CPU)
python make_plots.py      # Step 4: figures/*.png
streamlit run app.py      # Step 5: dashboard at http://localhost:8501
```

Stop the app with **Ctrl+C** in the terminal where it is running.

## Step by step: Windows

Run these in Command Prompt from the project folder.

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python prepare_data.py
python train_models.py
python train_lstm.py
python make_plots.py
streamlit run app.py
```

- If `python` is not 3.12 or 3.13, create the environment with `py -3.12 -m venv .venv` instead.
- In PowerShell, activate with `.venv\Scripts\Activate.ps1`. If scripts are blocked, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.
- **Stop the app with Ctrl+C** in the terminal where it is running. `pkill` does not exist on Windows.

## Data

The four `.mat` files must be in `./data/`. The run-all scripts download them automatically.
To do it by hand:

```bash
git clone --depth 1 https://github.com/anirudhkhatry/SOH-prediction-using-NASA-Dataset
```

Then copy `B0005.mat`, `B0006.mat`, `B0007.mat` and `B0018.mat` from that folder into `data/`.

## Cross-platform notes

- All paths are built with `pathlib`, starting from the project folder, so the scripts work from any working directory.
- Every CSV is read and written as UTF-8 with `\n` line endings, so the file format is the same on every OS.
- Console output is plain ASCII, so the Windows console never raises `UnicodeEncodeError`.
- `make_plots.py` uses matplotlib's `Agg` backend, so it needs no display.
- The first time you run `streamlit run app.py`, Streamlit may ask for an e-mail address. Press Enter to skip it.
  The run-all scripts avoid this prompt by starting Streamlit with `--server.headless true`.
- On Linux without an NVIDIA GPU, `train_lstm.py` may print `failed call to cuInit`. This is harmless.
  TensorFlow is only checking for a GPU, and training runs on the CPU.

## Project structure

| File / folder | What it does |
|---|---|
| `common.py` | Shared settings: feature list, pathlib paths, UTF-8 CSV helpers, metric and result-saving helpers |
| `run_all.sh`, `run_all.bat` | One-command setup and full pipeline for macOS/Linux and Windows |
| `prepare_data.py` | Reads the `.mat` files, keeps discharge cycles, builds features and SOH/RUL labels |
| `train_models.py` | Leave-one-battery-out evaluation of LR, RF and XGBoost; saves final models |
| `train_lstm.py` | Same evaluation for a one-layer LSTM on 10-cycle windows |
| `make_plots.py` | All report figures |
| `app.py` | Streamlit dashboard; loads saved models and CSVs, never retrains |
| `data/cycles.csv` | One row per discharge cycle (636 rows) |
| `results/metrics.csv` | MAE, RMSE, R² per target, test cell and model |
| `results/predictions.csv` | Actual vs predicted value for every cycle, target and model |
| `results/rul_error_by_stage.csv` | RUL MAE in early, middle and late life |
| `models/` | Final models trained on all four cells (`.joblib`, `.keras`) |
| `figures/` | PNG figures for the report |

## Method

### Features (one row per discharge cycle)

All features are computed over the samples where the load is on (current below −1 A).

| Feature | Meaning |
|---|---|
| `cycle` | Discharge cycle number |
| `discharge_time` | Seconds the 2 A load lasted |
| `avg_voltage`, `min_voltage` | Mean and lowest terminal voltage |
| `time_to_3v5` | Seconds until the voltage first falls to 3.5 V |
| `max_temp`, `avg_temp` | Peak and mean cell temperature |
| `time_to_max_temp` | Seconds until the peak temperature |

Capacity is **never** used as an input, because SOH is calculated from it.

### Labels

- **SOH** = capacity / 2.0 Ah × 100.
- **End of life (EOL)** = first cycle where the 5-cycle centred moving average of SOH falls below 75%.
  The common 70% rule is not used because B0007 never drops below 70% in this dataset.
- **RUL** = EOL cycle − current cycle, clipped at 0.
- Small upward capacity jumps are real recovery after rest periods. They are kept, not removed as outliers.
  The smoothing stops them from shifting the EOL point.

### Evaluation

Leave-one-battery-out: train on three cells, test on the fourth, and rotate through all four.
This is stricter than a random split, because neighbouring cycles of the same cell are very
similar and would leak into training. Reported metrics are MAE, RMSE and R², per cell and averaged.

- **Linear Regression** uses a `StandardScaler` inside a pipeline, so scaling is learned from training cells only.
- **LSTM** reads the last 10 cycles. Its scalers are fitted on the three training cells only.
  The first 9 cycles of a cell are padded by repeating cycle 1, so the LSTM is scored on exactly the same cycles as the other models.
- Final models for the app are re-trained on all four cells after evaluation.

## Results

All numbers below were produced by the scripts in this repository and copied from `results/`.

### Average over the four test cells

| Target | Model | MAE | RMSE | R² |
|---|---|---|---|---|
| SOH (%) | Linear Regression | 0.587 | 0.619 | 0.993 |
| SOH (%) | Random Forest | 1.348 | 1.570 | 0.968 |
| SOH (%) | XGBoost | 1.526 | 1.783 | 0.965 |
| SOH (%) | LSTM | 0.950 | 1.155 | 0.981 |
| RUL (cycles) | Linear Regression | 18.766 | 21.757 | 0.269 |
| RUL (cycles) | Random Forest | 9.663 | 14.460 | 0.733 |
| RUL (cycles) | XGBoost | 9.560 | 13.761 | 0.769 |
| RUL (cycles) | LSTM | 17.809 | 21.259 | 0.479 |

### MAE per test cell

SOH (% points):

| Test cell | Linear Regression | Random Forest | XGBoost | LSTM |
|---|---|---|---|---|
| B0005 | 0.626 | 0.784 | 1.013 | 0.471 |
| B0006 | 0.405 | 2.725 | 2.767 | 0.962 |
| B0007 | 1.132 | 1.576 | 1.454 | 1.212 |
| B0018 | 0.186 | 0.309 | 0.871 | 1.154 |

RUL (cycles):

| Test cell | Linear Regression | Random Forest | XGBoost | LSTM |
|---|---|---|---|---|
| B0005 | 8.605 | 5.314 | 3.774 | 6.419 |
| B0006 | 31.190 | 10.730 | 9.211 | 15.436 |
| B0007 | 23.860 | 17.732 | 19.309 | 35.292 |
| B0018 | 11.408 | 4.877 | 5.946 | 14.090 |

### Cell summary and RUL error by life stage

| Battery | Discharge cycles | First SOH (%) | Min SOH (%) | EOL cycle |
|---|---|---|---|---|
| B0005 | 168 | 92.82 | 64.37 | 98 |
| B0006 | 168 | 101.77 | 57.69 | 79 |
| B0007 | 168 | 94.55 | 70.02 | 126 |
| B0018 | 132 | 92.75 | 67.05 | 74 |

Stage = cycle / EOL cycle: early below 1/3, middle 1/3 to 2/3, late from 2/3 onwards (including cycles after EOL).

| Stage | Linear Regression | Random Forest | XGBoost | LSTM |
|---|---|---|---|---|
| Early | 22.70 | 27.53 | 26.21 | 35.54 |
| Middle | 12.92 | 14.58 | 13.62 | 22.18 |
| Late | 20.10 | 2.83 | 3.28 | 11.10 |

## Interpreting the results

- **SOH is easy, RUL is hard.** SOH reflects the cell's current state, which the discharge features measure directly.
  RUL is a forecast of the future, and each cell ages at a different rate.
- **Why Linear Regression wins on SOH.** The NASA tests discharge at a constant 2 A, so capacity ≈ 2 A × discharge time.
  On this data the correlation between capacity and `2 × discharge_time / 3600` is 0.9974.
  `discharge_time` is therefore close to a direct measurement of SOH, and the relationship is linear.
  This is physically legitimate (it is how capacity is measured), but it explains the very high SOH scores and should be mentioned in the viva.
  `time_to_max_temp` behaves the same way, because the temperature peaks at the end of discharge.
- **Why B0007 RUL is worst.** B0007 ages slowest and reaches EOL at cycle 126.
  The training cells reach EOL between cycles 74 and 98, so the tree models cannot predict an RUL larger than they have seen.
- **Tree models are best near end of life.** Random Forest and XGBoost have the lowest RUL error in late life, which is when an RUL warning matters most.
  Linear Regression cannot model the flat RUL of 0 after EOL. Its predictions drift away from zero there, which hurts it in the late stage.
- **The LSTM is limited by data.** There are only three training cells per fold, giving 468 to 504 training windows. A deeper network would overfit.
- **B0006 starts above 100% SOH.** Its first measured capacities exceed the 2.0 Ah rating. This is in the raw data and was kept.

## Dashboard

`streamlit run app.py` opens a dashboard with a battery selector, model selector and cycle slider.

- **Top cards:** predicted SOH and RUL, peak temperature, and Healthy/Degraded status (Degraded once measured SOH is below 75%).
- **Battery health:** capacity fade curve with the 75% threshold and the selected cycle marked.
- **Predictions:** actual vs predicted SOH and RUL for the chosen model.
- **Model comparison:** metrics table and bar chart for all four models.
- **Insights:** Random Forest and XGBoost feature importance and the correlation heatmap.
- **Try it:** enter feature values and get predicted SOH and RUL.

The cards and Predictions tab show leave-one-battery-out predictions, so the model never saw that battery.
The Try-it tab uses the final models trained on all four cells.
For the LSTM, it uses the 9 cycles before the selected cycle as history, followed by your values.
