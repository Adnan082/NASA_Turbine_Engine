"""
Build the v2 engine-level fit/val/calibration split for the corrected
conformal calibration (see FIX_REPORT.md, Phase 3).

v1's calibration set was the last 20% of *test* rows (agents/mapie.py) —
which, because of file concatenation order, is FD004 only, and worse, is
part of the data used for final evaluation. This script instead splits the
709 *training* engines, seeded and stratified by sub-dataset:

    80% fit engines          -> further split 90/10 into fit-train / fit-val
                                 (fit-val is for early stopping only)
    20% calibration engines  -> one prediction per engine, at a uniformly
                                 random cut point in its life, built with the
                                 same truncated-history / edge-padding
                                 protocol used for the real test engines
                                 (rather than every sliding window)

The 707 test engines are never referenced here — they're untouched until
scripts/evaluate_v2.py.

Usage:
    python scripts/prepare_calibration_split.py

Writes DATA/model_ready/{X,y,cond}_{fit,val}.npy,
DATA/model_ready/{X,y,subset,unit,cutoff}_cal.npy and
DATA/model_ready/v2_split_manifest.json (gitignored).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent))
from prepare_data import (  # noqa: E402
    DATASETS, MAX_RUL, MODEL_READY_DIR, WINDOW_SIZE,
    build_processed_frames, create_windows,
)

SEED = 42
CAL_FRACTION = 0.20
VAL_FRACTION_OF_FIT = 0.10


def split_engines(train, seed):
    """Engine-level 80/20 fit/calibration split, stratified by sub-dataset;
    the 80% fit engines are further split 90/10 into fit-train/fit-val."""
    rng = np.random.RandomState(seed)
    fit_engines, val_engines, cal_engines = {}, {}, {}

    for ds in DATASETS:
        units = np.sort(train[ds]["unit"].unique())
        shuffled = rng.permutation(units)

        n_cal = max(1, round(len(shuffled) * CAL_FRACTION))
        cal, fit_pool = shuffled[:n_cal], shuffled[n_cal:]

        n_val = max(1, round(len(fit_pool) * VAL_FRACTION_OF_FIT))
        val, fit = fit_pool[:n_val], fit_pool[n_val:]

        fit_engines[ds] = set(fit.tolist())
        val_engines[ds] = set(val.tolist())
        cal_engines[ds] = set(cal.tolist())

    return fit_engines, val_engines, cal_engines


def build_windows_for_engines(train, sensor_cols, engine_sets):
    """Sliding windows (stride 1), same as v1, restricted to a set of engines."""
    X_list, y_list, cond_list = [], [], []
    for ds in DATASETS:
        subset_df = train[ds][train[ds]["unit"].isin(engine_sets[ds])]
        X, y, cond, _, _ = create_windows(subset_df, sensor_cols)
        X_list.append(X)
        y_list.append(y)
        cond_list.append(cond)
    return np.concatenate(X_list), np.concatenate(y_list), np.concatenate(cond_list)


def build_random_cutoff_calibration_set(train, sensor_cols, cal_engines, seed):
    """One prediction target per calibration engine, at a uniformly random
    cycle in its life — truncated history, last-50-cycle window, edge-padded
    if shorter — matching exactly how real test engines are windowed."""
    rng = np.random.RandomState(seed)
    X, y, subset_labels, units, cutoffs = [], [], [], [], []

    for ds in DATASETS:
        for unit in sorted(cal_engines[ds]):
            engine = train[ds][train[ds]["unit"] == unit].reset_index(drop=True)
            max_cycle = int(engine["cycle"].max())
            cutoff = int(rng.randint(1, max_cycle + 1))

            history = engine[engine["cycle"] <= cutoff]
            if len(history) < WINDOW_SIZE:
                window = history[sensor_cols].values
                window = np.pad(window, ((WINDOW_SIZE - len(window), 0), (0, 0)), mode="edge")
            else:
                window = history.iloc[-WINDOW_SIZE:][sensor_cols].values

            true_rul = min(max_cycle - cutoff, MAX_RUL)

            X.append(window)
            y.append(true_rul)
            subset_labels.append(ds)
            units.append(unit)
            cutoffs.append(cutoff)

    return (
        np.array(X), np.array(y, dtype="float32"), np.array(subset_labels),
        np.array(units), np.array(cutoffs),
    )


def main():
    train, _test, _rul, sensor_cols = build_processed_frames(verbose=False)

    fit_engines, val_engines, cal_engines = split_engines(train, SEED)

    print(f"Engine split (seed={SEED}):")
    manifest = {
        "seed": SEED, "cal_fraction": CAL_FRACTION, "val_fraction_of_fit": VAL_FRACTION_OF_FIT,
        "subsets": {},
    }
    for ds in DATASETS:
        n_fit, n_val, n_cal = len(fit_engines[ds]), len(val_engines[ds]), len(cal_engines[ds])
        print(f"  {ds} — fit-train {n_fit}  fit-val {n_val}  calibration {n_cal}")
        manifest["subsets"][ds] = {
            "fit_train_engines": sorted(int(u) for u in fit_engines[ds]),
            "fit_val_engines": sorted(int(u) for u in val_engines[ds]),
            "calibration_engines": sorted(int(u) for u in cal_engines[ds]),
        }

    print("Building fit-train windows (sliding, stride 1)...")
    X_fit, y_fit, cond_fit = build_windows_for_engines(train, sensor_cols, fit_engines)
    print(f"  X_fit: {X_fit.shape}")

    print("Building fit-val windows...")
    X_val, y_val, cond_val = build_windows_for_engines(train, sensor_cols, val_engines)
    print(f"  X_val: {X_val.shape}")

    print("Building calibration set (one random-cutoff window per engine)...")
    X_cal, y_cal, subset_cal, unit_cal, cutoff_cal = build_random_cutoff_calibration_set(
        train, sensor_cols, cal_engines, SEED
    )
    print(f"  X_cal: {X_cal.shape}")

    MODEL_READY_DIR.mkdir(parents=True, exist_ok=True)
    np.save(MODEL_READY_DIR / "X_fit.npy", X_fit)
    np.save(MODEL_READY_DIR / "y_fit.npy", y_fit)
    np.save(MODEL_READY_DIR / "cond_fit.npy", cond_fit)
    np.save(MODEL_READY_DIR / "X_val.npy", X_val)
    np.save(MODEL_READY_DIR / "y_val.npy", y_val)
    np.save(MODEL_READY_DIR / "cond_val.npy", cond_val)
    np.save(MODEL_READY_DIR / "X_cal.npy", X_cal)
    np.save(MODEL_READY_DIR / "y_cal.npy", y_cal)
    np.save(MODEL_READY_DIR / "subset_cal.npy", subset_cal)
    np.save(MODEL_READY_DIR / "unit_cal.npy", unit_cal)
    np.save(MODEL_READY_DIR / "cutoff_cal.npy", cutoff_cal)

    with open(MODEL_READY_DIR / "v2_split_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nSaved v2 split arrays and manifest to {MODEL_READY_DIR}")


if __name__ == "__main__":
    main()
