"""
STEP 1 - prepare_data.py
========================
Turns the raw NASA PCoE battery files (B0005, B0006, B0007, B0018) into a
clean table with ONE ROW PER DISCHARGE CYCLE.

Output: data/cycles.csv
    battery, cycle, capacity, 7 engineered features, soh, eol_cycle, rul

Why only discharge cycles?
    Capacity (how many Ah the cell can deliver) is only measured while the
    cell is being discharged at a constant 2 A load, so each discharge cycle
    gives us one labelled data point.
"""
import numpy as np
import pandas as pd
from scipy.io import loadmat

# DATA_DIR / CYCLES_CSV are pathlib paths anchored at the project folder, so
# this script works from any working directory and on Windows/macOS/Linux.
from common import CYCLES_CSV, DATA_DIR, write_csv

# ---------------------------------------------------------------- settings
CELLS = ["B0005", "B0006", "B0007", "B0018"]
RATED_CAPACITY = 2.0   # Ah - nominal capacity of a fresh cell (dataset README)
EOL_SOH = 75.0         # % SOH treated as end of life.
                       # 70% is the usual EV rule, but B0007 never drops below
                       # 70% in this dataset, so it would have no EOL point.
LOAD_CURRENT = -1.0    # A - samples below this are "load on" (discharging)
VOLTAGE_KNEE = 3.5     # V - we measure how long the cell takes to reach this
SMOOTH_WINDOW = 5      # cycles used to smooth SOH before finding EOL


def extract_cell(cell):
    """Read one .mat file and return a DataFrame of per-cycle features."""
    # simplify_cells=True turns MATLAB structs into plain Python dicts/lists
    mat = loadmat(DATA_DIR / f"{cell}.mat", simplify_cells=True)

    rows = []
    cycle_no = 0  # we number discharge cycles 1, 2, 3, ... for each cell
    for c in mat[cell]["cycle"]:
        # The file mixes charge, discharge and impedance tests; keep discharge
        if c["type"] != "discharge":
            continue

        d = c["data"]
        voltage = np.asarray(d["Voltage_measured"], dtype=float)
        current = np.asarray(d["Current_measured"], dtype=float)
        temp = np.asarray(d["Temperature_measured"], dtype=float)
        time = np.asarray(d["Time"], dtype=float)
        capacity = float(np.atleast_1d(d["Capacity"])[0])

        # Keep only the samples where the 2 A load is actually applied.
        # Before/after the load the current is ~0 and the voltage "relaxes"
        # back up, which would distort the averages.
        load = current < LOAD_CURRENT
        if load.sum() < 10 or not np.isfinite(capacity):
            continue  # skip broken/empty records (none expected, but be safe)

        v, t, tm = voltage[load], temp[load], time[load]
        tm = tm - tm[0]  # time measured from the moment the load starts

        # time_to_3v5: first sample where voltage has fallen to 3.5 V.
        # A worn cell's voltage drops faster, so this time gets shorter.
        below = np.where(v <= VOLTAGE_KNEE)[0]
        time_to_3v5 = tm[below[0]] if len(below) else tm[-1]

        cycle_no += 1
        rows.append({
            "battery": cell,
            "cycle": cycle_no,
            "capacity": capacity,             # used ONLY to build the SOH label
            "discharge_time": tm[-1],         # s, how long the load lasted
            "avg_voltage": v.mean(),          # V
            "min_voltage": v.min(),           # V, cut-off voltage reached
            "time_to_3v5": time_to_3v5,       # s
            "max_temp": t.max(),              # deg C, peak during discharge
            "avg_temp": t.mean(),             # deg C
            "time_to_max_temp": tm[t.argmax()],  # s, when the peak happened
        })

    df = pd.DataFrame(rows)

    # ---- Label 1: State of Health = current capacity / rated capacity
    df["soh"] = df["capacity"] / RATED_CAPACITY * 100

    # ---- Label 2: Remaining Useful Life
    # Capacity sometimes jumps up a little after a rest period (a real
    # electrochemical "recovery" effect, so we keep those points). To stop such
    # a bump from moving the EOL point, we look at a 5-cycle centred moving
    # average of SOH and take the first cycle where it is below 75%.
    smooth = df["soh"].rolling(SMOOTH_WINDOW, center=True, min_periods=1).mean()
    hit = np.where(smooth < EOL_SOH)[0]
    if len(hit) == 0:
        raise ValueError(f"{cell} never reaches {EOL_SOH}% SOH")
    eol_cycle = int(df["cycle"].iloc[hit[0]])

    df["eol_cycle"] = eol_cycle
    # RUL = cycles left until EOL; after EOL it stays at 0 (already worn out)
    df["rul"] = (eol_cycle - df["cycle"]).clip(lower=0)
    return df


if __name__ == "__main__":
    all_cells = pd.concat([extract_cell(c) for c in CELLS], ignore_index=True)
    out_path = CYCLES_CSV
    write_csv(all_cells, out_path, index=False)

    # Print a short summary so we can sanity-check the data
    summary = all_cells.groupby("battery").agg(
        discharge_cycles=("cycle", "max"),
        first_soh=("soh", "first"),
        last_soh=("soh", "last"),
        min_soh=("soh", "min"),
        eol_cycle=("eol_cycle", "first"),
    ).round(2)
    print(f"Saved {len(all_cells)} rows to {out_path}\n")
    print(summary)
    missing = all_cells.isna().sum()
    print("\nMissing values:", "none" if missing.sum() == 0 else missing[missing > 0])
