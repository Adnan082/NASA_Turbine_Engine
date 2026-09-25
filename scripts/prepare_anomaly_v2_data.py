"""
Build the v2 anomaly-detector data: early-life-only windows (first 30% of
each engine's life) from the fit-train engines, for training, and a
held-out early-life set from the calibration engines, for choosing a
threshold at a target false-alarm rate (see FIX_REPORT.md, Phase 4).

Reuses the exact same engine split as scripts/prepare_calibration_split.py
(same seed -> identical fit/calibration engines), so this never touches
the 707 test engines and stays consistent with the v2 RUL calibration.

Regime labels are made unique across sub-datasets (e.g. "FD002_r3")
instead of the v1 scheme, where FD001/FD003 (hardcoded to condition 0)
and each FD002/FD004 KMeans cluster all landed on the same handful of
condition numbers — mixing four physically different regimes under
"condition 0" (see FIX_REPORT.md, known problem #4).

Usage:
    python scripts/prepare_anomaly_v2_data.py

Writes DATA/model_ready/{X,regime}_early_{fit,cal}.npy (gitignored).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent))
from prepare_calibration_split import SEED, split_engines  # noqa: E402
from prepare_data import DATASETS, MODEL_READY_DIR, build_processed_frames, create_windows  # noqa: E402

EARLY_LIFE_FRACTION = 0.30


def build_early_life_windows(train, sensor_cols, engine_sets):
    X_list, regime_list = [], []
    for ds in DATASETS:
        subset_df = train[ds][train[ds]["unit"].isin(engine_sets[ds])]
        max_cycle = subset_df.groupby("unit")["cycle"].max()

        X, _y, cond, units, cycles = create_windows(subset_df, sensor_cols)
        if len(X) == 0:
            continue
        life_fraction = cycles / max_cycle.loc[units].values
        early_mask = life_fraction <= EARLY_LIFE_FRACTION

        X_list.append(X[early_mask])
        regime_list.append(np.array([f"{ds}_r{int(c)}" for c in cond[early_mask]]))

    return np.concatenate(X_list), np.concatenate(regime_list)


def main():
    train, _test, _rul, sensor_cols = build_processed_frames(verbose=False)
    fit_engines, _val_engines, cal_engines = split_engines(train, SEED)

    print(f"Building early-life (first {EARLY_LIFE_FRACTION:.0%} of life) windows for fit-train engines...")
    X_early_fit, regime_early_fit = build_early_life_windows(train, sensor_cols, fit_engines)
    print(f"  X_early_fit: {X_early_fit.shape}")

    print("Building early-life windows for calibration engines (threshold tuning)...")
    X_early_cal, regime_early_cal = build_early_life_windows(train, sensor_cols, cal_engines)
    print(f"  X_early_cal: {X_early_cal.shape}")

    MODEL_READY_DIR.mkdir(parents=True, exist_ok=True)
    np.save(MODEL_READY_DIR / "X_early_fit.npy", X_early_fit)
    np.save(MODEL_READY_DIR / "regime_early_fit.npy", regime_early_fit)
    np.save(MODEL_READY_DIR / "X_early_cal.npy", X_early_cal)
    np.save(MODEL_READY_DIR / "regime_early_cal.npy", regime_early_cal)

    print(f"\nSaved to {MODEL_READY_DIR}")


if __name__ == "__main__":
    main()
