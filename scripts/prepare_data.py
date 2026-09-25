"""
Rebuild DATA/model_ready/*.npy from the committed DATA/pre_processed_data
parquet files. This is a script version of notebooks/Pre-processing.ipynb,
picking up from the point where op3/s1/s5/s10/s16/s19 have already been
dropped (that step produced the committed pre_processed_data files and is
not repeated here).

Steps (must match the notebook exactly, see FIX_REPORT.md):
    1. cap RUL at 125 in the RUL_*.parquet files (test labels)
    2. 10-cycle rolling mean on s3, per engine
    3. KMeans (6 regimes, random_state=42, n_init=10) on FD002/FD004,
       fitted on train only; FD001/FD003 get a single condition (0)
    4. per-condition MinMax scaling, fitted on train only
    5. compute train RUL (max_cycle - cycle, capped at 125)
    6. 50-cycle sliding windows (stride 1) for train; last-50,
       edge-padded window per engine for test

Usage:
    python scripts/prepare_data.py

Writes DATA/model_ready/{X,y,cond}_{train,test}.npy (gitignored).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler

ROOT = Path(__file__).parent.parent
PRE_PROCESSED_DIR = ROOT / "DATA" / "pre_processed_data"
MODEL_READY_DIR = ROOT / "DATA" / "model_ready"

DATASETS = ["FD001", "FD002", "FD003", "FD004"]
MAX_RUL = 125
NOISY_SENSOR = "s3"
ROLLING_WINDOW = 10
ENGINE_ID_COL = "unit"
WINDOW_SIZE = 50
KMEANS_DATASETS = ["FD002", "FD004"]
SEED = 42


def require_pre_processed_data():
    missing = [
        f"{kind}_{ds}.parquet"
        for ds in DATASETS
        for kind in ("train", "test", "RUL")
        if not (PRE_PROCESSED_DIR / f"{kind}_{ds}.parquet").exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} in {PRE_PROCESSED_DIR}. "
            "These are committed to the repo — check your checkout."
        )


def load_data():
    train, test, rul = {}, {}, {}
    for ds in DATASETS:
        train[ds] = pd.read_parquet(PRE_PROCESSED_DIR / f"train_{ds}.parquet")
        test[ds] = pd.read_parquet(PRE_PROCESSED_DIR / f"test_{ds}.parquet")
        rul[ds] = pd.read_parquet(PRE_PROCESSED_DIR / f"RUL_{ds}.parquet")
    return train, test, rul


def cap_test_rul(rul):
    for ds in DATASETS:
        if "RUL" in rul[ds].columns:
            rul[ds]["RUL"] = rul[ds]["RUL"].clip(upper=MAX_RUL)


def smooth_noisy_sensor(train, test):
    for ds in DATASETS:
        for frame in (train[ds], test[ds]):
            if NOISY_SENSOR in frame.columns and ENGINE_ID_COL in frame.columns:
                frame[NOISY_SENSOR] = (
                    frame.groupby(ENGINE_ID_COL)[NOISY_SENSOR]
                    .transform(lambda x: x.rolling(window=ROLLING_WINDOW, min_periods=1).mean())
                )


def assign_conditions(train, test):
    for ds in ["FD001", "FD003"]:
        train[ds]["condition"] = 0
        test[ds]["condition"] = 0

    for ds in KMEANS_DATASETS:
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train[ds][["op1", "op2"]])
        test_scaled = scaler.transform(test[ds][["op1", "op2"]])

        kmeans = KMeans(n_clusters=6, random_state=SEED, n_init=10)
        train[ds]["condition"] = kmeans.fit_predict(train_scaled)
        test[ds]["condition"] = kmeans.predict(test_scaled)


def scale_sensors(train, test):
    sensor_cols = [col for col in train["FD001"].columns if col.startswith("s")]

    for ds in DATASETS:
        train[ds][sensor_cols] = train[ds][sensor_cols].astype(float)
        test[ds][sensor_cols] = test[ds][sensor_cols].astype(float)

    for ds in DATASETS:
        scaler_dict = {}
        for condition in train[ds]["condition"].unique():
            mask = train[ds]["condition"] == condition
            scaler = MinMaxScaler()
            train[ds].loc[mask, sensor_cols] = scaler.fit_transform(train[ds].loc[mask, sensor_cols])
            scaler_dict[condition] = scaler

        for condition in test[ds]["condition"].unique():
            mask = test[ds]["condition"] == condition
            if condition in scaler_dict:
                test[ds].loc[mask, sensor_cols] = scaler_dict[condition].transform(test[ds].loc[mask, sensor_cols])

    return sensor_cols


def add_train_rul(train):
    for ds in DATASETS:
        max_cycle = train[ds].groupby("unit")["cycle"].max().reset_index()
        max_cycle.columns = ["unit", "max_cycle"]
        train[ds] = train[ds].merge(max_cycle, on="unit")
        train[ds]["RUL"] = (train[ds]["max_cycle"] - train[ds]["cycle"]).clip(upper=MAX_RUL)
        train[ds].drop(columns=["max_cycle"], inplace=True)
    return train


def create_windows(df, sensor_cols, window_size=WINDOW_SIZE):
    """Returns X, y, conditions, plus the engine unit id and life-cycle number
    of each window's last row — needed downstream for engine-level splits
    (v2 calibration) and life-position filtering (v2 anomaly detector).
    Not present in the original notebook, which never needed to split by
    engine after windowing.
    """
    X, y, conditions, units, cycles = [], [], [], [], []
    for unit in df["unit"].unique():
        engine = df[df["unit"] == unit].reset_index(drop=True)
        if len(engine) < window_size:
            continue
        for i in range(len(engine) - window_size + 1):
            last_row = engine.iloc[i + window_size - 1]
            window = engine.iloc[i:i + window_size][sensor_cols].values
            X.append(window)
            y.append(last_row["RUL"])
            conditions.append(last_row["condition"])
            units.append(unit)
            cycles.append(last_row["cycle"])
    return np.array(X), np.array(y), np.array(conditions), np.array(units), np.array(cycles)


def create_windows_test(df, sensor_cols, window_size=WINDOW_SIZE):
    X, conditions, units = [], [], []
    for unit in df["unit"].unique():
        engine = df[df["unit"] == unit].reset_index(drop=True)
        if len(engine) < window_size:
            window = engine[sensor_cols].values
            window = np.pad(window, ((window_size - len(window), 0), (0, 0)), mode="edge")
        else:
            window = engine.iloc[-window_size:][sensor_cols].values
        condition = engine.iloc[-1]["condition"]
        X.append(window)
        conditions.append(condition)
        units.append(unit)
    return np.array(X), np.array(conditions), np.array(units)


def main():
    require_pre_processed_data()

    print("Loading pre-processed data...")
    train, test, rul = load_data()

    print(f"Capping RUL at {MAX_RUL} (test labels)...")
    cap_test_rul(rul)

    print(f"Applying {ROLLING_WINDOW}-cycle rolling mean to {NOISY_SENSOR}...")
    smooth_noisy_sensor(train, test)

    print("Assigning operating conditions (KMeans on FD002/FD004)...")
    assign_conditions(train, test)
    for ds in DATASETS:
        print(f"  {ds} — conditions: {sorted(train[ds]['condition'].unique())}")

    print("Scaling sensors per condition (MinMax, fit on train)...")
    sensor_cols = scale_sensors(train, test)

    print("Computing train RUL labels...")
    train = add_train_rul(train)

    print("Building sliding windows...")
    X_list, y_list, cond_list, unit_list, cycle_list, subset_list = [], [], [], [], [], []
    for ds in DATASETS:
        X, y, cond, units, cycles = create_windows(train[ds], sensor_cols)
        X_list.append(X)
        y_list.append(y)
        cond_list.append(cond)
        unit_list.append(units)
        cycle_list.append(cycles)
        subset_list.append(np.full(len(X), ds))
        print(f"  {ds} train — X: {X.shape}")

    X_train = np.concatenate(X_list, axis=0)
    y_train = np.concatenate(y_list, axis=0)
    cond_train = np.concatenate(cond_list, axis=0)
    unit_train = np.concatenate(unit_list, axis=0)
    cycle_train = np.concatenate(cycle_list, axis=0)
    subset_train = np.concatenate(subset_list, axis=0)

    X_test_list, y_test_list, cond_test_list, unit_test_list, subset_test_list = [], [], [], [], []
    for ds in DATASETS:
        X, cond, units = create_windows_test(test[ds], sensor_cols)
        y = rul[ds]["RUL"].values
        X_test_list.append(X)
        y_test_list.append(y)
        cond_test_list.append(cond)
        unit_test_list.append(units)
        subset_test_list.append(np.full(len(X), ds))
        print(f"  {ds} test  — X: {X.shape}")

    X_test = np.concatenate(X_test_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)
    cond_test = np.concatenate(cond_test_list, axis=0)
    unit_test = np.concatenate(unit_test_list, axis=0)
    subset_test = np.concatenate(subset_test_list, axis=0)

    print(f"\nFinal X_train: {X_train.shape}")
    print(f"Final y_train: {y_train.shape}")
    print(f"Final X_test:  {X_test.shape}")
    print(f"Final y_test:  {y_test.shape}")

    MODEL_READY_DIR.mkdir(parents=True, exist_ok=True)
    np.save(MODEL_READY_DIR / "X_train.npy", X_train)
    np.save(MODEL_READY_DIR / "y_train.npy", y_train)
    np.save(MODEL_READY_DIR / "cond_train.npy", cond_train)
    np.save(MODEL_READY_DIR / "X_test.npy", X_test)
    np.save(MODEL_READY_DIR / "y_test.npy", y_test)
    np.save(MODEL_READY_DIR / "cond_test.npy", cond_test)

    # Engine-identity metadata (not part of the original notebook output) —
    # needed for the v2 engine-level calibration split and life-position
    # filtering. Doesn't change any of the six arrays above.
    np.save(MODEL_READY_DIR / "unit_train.npy", unit_train)
    np.save(MODEL_READY_DIR / "cycle_train.npy", cycle_train)
    np.save(MODEL_READY_DIR / "subset_train.npy", subset_train)
    np.save(MODEL_READY_DIR / "unit_test.npy", unit_test)
    np.save(MODEL_READY_DIR / "subset_test.npy", subset_test)

    print(f"\nSaved arrays to {MODEL_READY_DIR}")


if __name__ == "__main__":
    main()
