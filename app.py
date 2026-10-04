"""
STEP 5 - app.py  (Streamlit dashboard)
======================================
Run with:   streamlit run app.py

The app NEVER trains anything. It only reads:
    data/cycles.csv           measured features + true SOH/RUL
    results/metrics.csv       leave-one-battery-out scores
    results/predictions.csv   leave-one-battery-out predictions
    models/*                  final models (trained on all four cells)

Predictions shown in the cards and in the "Predictions" tab come from
predictions.csv, i.e. from a model that had NOT seen that battery during
training - the honest numbers. The "Try it" tab uses the final saved models.
"""
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common import (CELLS, CYCLES_CSV, EOL_SOH, FEATURES, METRICS_CSV,
                    MODEL_NAMES, PREDICTIONS_CSV, model_file, read_csv,
                    summary_table)

st.set_page_config(page_title="EV Battery SOH & RUL", page_icon="🔋", layout="wide")

# Same fixed colours as the report figures (one colour per model / per cell)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MODEL_COLOR = dict(zip(MODEL_NAMES, PALETTE))
CELL_COLOR = dict(zip(CELLS, PALETTE))
ACTUAL_COLOR = "#52514e"

FEATURE_LABELS = {
    "cycle": "Cycle number",
    "discharge_time": "Discharge time (s)",
    "avg_voltage": "Average voltage (V)",
    "min_voltage": "Minimum voltage (V)",
    "time_to_3v5": "Time to reach 3.5 V (s)",
    "max_temp": "Max temperature (°C)",
    "avg_temp": "Average temperature (°C)",
    "time_to_max_temp": "Time to max temperature (s)",
}


# ------------------------------------------------------------ data loading
# st.cache_* means files are read once, not on every click.
@st.cache_data
def load_tables():
    return (read_csv(CYCLES_CSV), read_csv(METRICS_CSV),
            read_csv(PREDICTIONS_CSV))


@st.cache_resource
def load_model(target, model_name):
    """Load one saved model. The LSTM also needs its two scalers."""
    if model_name == "LSTM":
        import tensorflow as tf  # imported only when needed (slow import)
        net = tf.keras.models.load_model(str(model_file(target, "LSTM", ext="keras")))
        return net, joblib.load(model_file(target, "LSTM_scalers"))
    return joblib.load(model_file(target, model_name)), None


def predict_one(target, model_name, row, history):
    """Predict SOH or RUL for one feature row.

    For the LSTM we need a 10-cycle window: the 9 cycles before it (taken from
    the selected battery, padded with its first cycle if needed) + this row.
    """
    model, extra = load_model(target, model_name)
    x = pd.DataFrame([row])[FEATURES]
    if model_name != "LSTM":
        return float(model.predict(x)[0])
    win, xs, ys = extra["window"], extra["x_scaler"], extra["y_scaler"]
    past = history[FEATURES].tail(win - 1)
    seq = pd.concat([past, x], ignore_index=True)
    seq = pd.concat([seq.iloc[[0]]] * (win - len(seq)) + [seq], ignore_index=True)
    y = model.predict(xs.transform(seq)[np.newaxis], verbose=0)
    return float(ys.inverse_transform(y)[0, 0])


missing = [p.name for p in (CYCLES_CSV, METRICS_CSV, PREDICTIONS_CSV) if not p.exists()]
if missing:
    st.error(f"Missing files: {missing}. Run prepare_data.py, train_models.py "
             "and train_lstm.py first (see README).")
    st.stop()

cycles, metrics, preds = load_tables()

# ------------------------------------------------------------ sidebar
st.sidebar.title("🔋 Controls")
battery = st.sidebar.selectbox("Battery", CELLS)
model_name = st.sidebar.selectbox("Model", MODEL_NAMES, index=2)
cell = cycles[cycles.battery == battery].sort_values("cycle")
cycle = st.sidebar.slider("Cycle", int(cell.cycle.min()), int(cell.cycle.max()),
                          int(cell.cycle.min()))
st.sidebar.caption(
    "Predictions are leave-one-battery-out: the model never saw the selected "
    "battery during training.")

row = cell[cell.cycle == cycle].iloc[0]
p_cell = preds[(preds.battery == battery) & (preds.model == model_name)]


def predicted(target):
    return float(p_cell[(p_cell.target == target) & (p_cell.cycle == cycle)].predicted.iloc[0])


# ------------------------------------------------------------ header + cards
st.title("EV Battery State of Health & Remaining Useful Life")
st.caption(f"NASA PCoE Li-ion aging data · battery **{battery}** · cycle "
           f"**{cycle}** · model **{model_name}**")

c1, c2, c3, c4 = st.columns(4)
soh_pred, rul_pred = predicted("soh"), predicted("rul")
c1.metric("State of Health (predicted)", f"{soh_pred:.1f} %",
          f"{soh_pred - row.soh:+.1f} vs actual {row.soh:.1f} %", delta_color="off")
c2.metric("RUL (predicted)", f"{max(rul_pred, 0):.0f} cycles",
          f"{rul_pred - row.rul:+.0f} vs actual {row.rul:.0f}", delta_color="off")
c3.metric("Peak temperature", f"{row.max_temp:.1f} °C")
degraded = row.soh < EOL_SOH
c4.metric("Status (measured SOH)", "⚠️ Degraded" if degraded else "✅ Healthy",
          f"threshold {EOL_SOH:.0f} % SOH", delta_color="off")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Battery health", "Predictions", "Model comparison", "Insights", "Try it"])

# ------------------------------------------------------------ tab 1
with tab1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cell.cycle, y=cell.soh, mode="lines",
                             name=f"{battery} SOH",
                             line=dict(color=CELL_COLOR[battery], width=2)))
    fig.add_hline(y=EOL_SOH, line_dash="dash", line_color="#0b0b0b",
                  annotation_text=f"EOL threshold {EOL_SOH:.0f}%",
                  annotation_position="bottom right")
    fig.add_trace(go.Scatter(x=[cycle], y=[row.soh], mode="markers",
                             name=f"Cycle {cycle}",
                             marker=dict(size=13, color="#e34948",
                                         line=dict(color="white", width=2))))
    fig.update_layout(title=f"Capacity fade - {battery} (EOL at cycle "
                            f"{int(cell.eol_cycle.iloc[0])})",
                      xaxis_title="Discharge cycle", yaxis_title="SOH (%)",
                      hovermode="x unified", height=450)
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Capacity at cycle {cycle}: {row.capacity:.3f} Ah of 2.0 Ah rated. "
               "Small upward bumps are real capacity recovery after rest periods.")

# ------------------------------------------------------------ tab 2
with tab2:
    cols = st.columns(2)
    for col, target, unit in zip(cols, ["soh", "rul"], ["%", "cycles"]):
        d = p_cell[p_cell.target == target].sort_values("cycle")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=d.cycle, y=d.actual, name="Actual",
                                 line=dict(color=ACTUAL_COLOR, width=2)))
        fig.add_trace(go.Scatter(x=d.cycle, y=d.predicted, name="Predicted",
                                 line=dict(color=MODEL_COLOR[model_name], width=2)))
        fig.add_vline(x=cycle, line_dash="dot", line_color="#999")
        m = metrics[(metrics.model == model_name) & (metrics.target == target)
                    & (metrics.test_cell == battery)].iloc[0]
        fig.update_layout(
            title=f"{target.upper()} - MAE {m.MAE:.2f}, RMSE {m.RMSE:.2f}, R² {m.R2:.3f}",
            xaxis_title="Cycle", yaxis_title=f"{target.upper()} ({unit})",
            hovermode="x unified", height=420,
            legend=dict(orientation="h", y=-0.2))
        col.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------ tab 3
with tab3:
    summary = summary_table(metrics)
    st.subheader("Average over the four leave-one-battery-out folds")
    st.dataframe(summary.style.format({"MAE": "{:.3f}", "RMSE": "{:.3f}", "R2": "{:.3f}"}),
                 hide_index=True, width="stretch")
    metric = st.radio("Metric", ["MAE", "RMSE", "R2"], horizontal=True)
    cols = st.columns(2)
    for col, target in zip(cols, ["soh", "rul"]):
        t = summary[summary.target == target]
        fig = px.bar(t, x="model", y=metric, color="model",
                     color_discrete_map=MODEL_COLOR, text_auto=".3f",
                     category_orders={"model": MODEL_NAMES},
                     title=f"{target.upper()} - average {metric}")
        fig.update_layout(showlegend=False, height=380, xaxis_title=None)
        col.plotly_chart(fig, width="stretch")
    with st.expander("Per-cell results"):
        st.dataframe(metrics.round(3), hide_index=True, width="stretch")

# ------------------------------------------------------------ tab 4
with tab4:
    target = st.radio("Target", ["soh", "rul"], horizontal=True,
                      format_func=str.upper, key="imp_target")
    cols = st.columns(2)
    for col, name in zip(cols, ["Random Forest", "XGBoost"]):
        model, _ = load_model(target, name)
        imp = (pd.DataFrame({"feature": FEATURES,
                             "importance": model.feature_importances_})
               .sort_values("importance"))
        fig = px.bar(imp, x="importance", y="feature", orientation="h",
                     text_auto=".2f", title=f"{name} feature importance ({target.upper()})")
        fig.update_traces(marker_color=MODEL_COLOR[name])
        fig.update_layout(height=400, yaxis_title=None)
        col.plotly_chart(fig, width="stretch")

    corr = cycles[FEATURES + ["soh", "rul"]].corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1, aspect="auto",
                    title="Correlation heatmap (all four cells)")
    fig.update_layout(height=600)
    st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------ tab 5
with tab5:
    st.write("Enter feature values for one discharge cycle. The boxes start "
             f"with the measured values of **{battery}, cycle {cycle}**.")
    with st.form("try_it"):
        cols = st.columns(4)
        user = {}
        for i, f in enumerate(FEATURES):
            is_int = f == "cycle"
            user[f] = cols[i % 4].number_input(
                FEATURE_LABELS[f], value=int(row[f]) if is_int else float(row[f]),
                step=1 if is_int else 0.01, format="%d" if is_int else "%.3f",
                # key includes battery + cycle so the boxes refill when the
                # sidebar selection changes (otherwise old values would stick)
                key=f"in_{f}_{battery}_{cycle}")
        submitted = st.form_submit_button("Predict SOH and RUL")
    if submitted:
        history = cell[cell.cycle < cycle]
        soh = predict_one("soh", model_name, user, history)
        rul = predict_one("rul", model_name, user, history)
        a, b, c = st.columns(3)
        a.metric(f"Predicted SOH ({model_name})", f"{soh:.1f} %")
        b.metric(f"Predicted RUL ({model_name})", f"{max(rul, 0):.0f} cycles")
        c.metric("Status", "⚠️ Degraded" if soh < EOL_SOH else "✅ Healthy")
    st.caption("Uses the final models trained on all four cells. For the LSTM, "
               "the 9 cycles before the selected cycle of the selected battery "
               "are used as history, followed by your values.")
