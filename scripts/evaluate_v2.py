"""
Evaluate the v2 RUL model with a correctly constructed conformal
calibration (see FIX_REPORT.md, Phase 3). Uses the engine-level
fit/val/calibration split built by scripts/prepare_calibration_split.py
and touches the 707 test engines exactly once, here, at the end.

Finite-sample split-conformal quantile: for a calibration set of size n
and miscoverage alpha, take the ceil((n+1)(1-alpha))-th smallest absolute
residual. If that index exceeds n, the calibration set is too small to
guarantee the target coverage at this alpha — the maximum observed
residual is used instead, and this is flagged in the output rather than
silently understating the interval.

Also reports a Mondrian (group-conditional) variant: calibration
residuals are grouped by the model's own predicted-RUL band, and each
test engine is assigned the quantile for the band its own prediction
falls into. This can give narrower intervals near end-of-life; both
variants are reported side by side and neither overwrites the other.

Usage:
    python scripts/evaluate_v2.py

Requires models/v2/agent2_rul_predictor.pt (scripts/train_rul.py) and
DATA/model_ready/{X,y,subset}_cal.npy (scripts/prepare_calibration_split.py).
Writes reports/v2/metrics.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT / "agents"))
sys.path.append(str(ROOT / "scripts"))

from agent2_rul import RULAgent  # noqa: E402
from evaluate_rul import nasa_score, SUBSETS  # noqa: E402

MODEL_DIR = ROOT / "models"
MODEL_READY_DIR = ROOT / "DATA" / "model_ready"
REPORTS_DIR = ROOT / "reports" / "v2"

ALPHA = 0.10
RUL_BANDS = [("0-30", 0, 30), ("31-60", 31, 60), ("61-100", 61, 100), (">100", 101, np.inf)]


def require_inputs():
    v2_model = MODEL_DIR / "v2" / "agent2_rul_predictor.pt"
    required = [
        v2_model, MODEL_READY_DIR / "X_cal.npy", MODEL_READY_DIR / "y_cal.npy",
        MODEL_READY_DIR / "subset_cal.npy", MODEL_READY_DIR / "X_test.npy",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing}. Run `python scripts/prepare_calibration_split.py` "
            "and `python scripts/train_rul.py ... --output models/v2/agent2_rul_predictor.pt` first."
        )
    return v2_model


def finite_sample_quantile(residuals, alpha):
    """ceil((n+1)(1-alpha))-th smallest residual (1-indexed). Returns
    (quantile, saturated) — saturated=True means n was too small to hit
    the target coverage exactly and the max residual was used instead."""
    n = len(residuals)
    sorted_res = np.sort(residuals)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return float(sorted_res[-1]), True
    return float(sorted_res[k - 1]), False


def band_mask(y, lo, hi):
    return (y >= lo) & (y <= hi)


def evaluate_split_conformal(cal_preds, y_cal, test_preds, y_test, alpha):
    residuals = np.abs(cal_preds - y_cal)
    quantile, saturated = finite_sample_quantile(residuals, alpha)

    lower = np.clip(test_preds - quantile, 0, 125)
    upper = np.clip(test_preds + quantile, 0, 125)
    covered = (y_test >= lower) & (y_test <= upper)
    width = upper - lower

    return {
        "quantile": quantile, "quantile_saturated": saturated,
        "coverage": float(covered.mean()), "mean_width": float(width.mean()),
        "covered": covered, "width": width,
    }


def evaluate_mondrian_conformal(cal_preds, y_cal, test_preds, y_test, alpha):
    """Group-conditional conformal: bucket calibration residuals by the
    model's own predicted-RUL band, then apply the matching band's
    quantile to each test engine based on its own predicted RUL."""
    band_quantiles = {}
    for name, lo, hi in RUL_BANDS:
        mask = band_mask(cal_preds, lo, hi)
        if mask.sum() == 0:
            band_quantiles[name] = None
            continue
        residuals = np.abs(cal_preds[mask] - y_cal[mask])
        quantile, saturated = finite_sample_quantile(residuals, alpha)
        band_quantiles[name] = {"quantile": quantile, "saturated": saturated, "n_cal": int(mask.sum())}

    fallback_quantile, _ = finite_sample_quantile(np.abs(cal_preds - y_cal), alpha)

    lower = np.zeros_like(test_preds)
    upper = np.zeros_like(test_preds)
    for name, lo, hi in RUL_BANDS:
        mask = band_mask(test_preds, lo, hi)
        q_info = band_quantiles.get(name)
        q = q_info["quantile"] if q_info else fallback_quantile
        lower[mask] = np.clip(test_preds[mask] - q, 0, 125)
        upper[mask] = np.clip(test_preds[mask] + q, 0, 125)

    covered = (y_test >= lower) & (y_test <= upper)
    width = upper - lower

    return {
        "band_quantiles": band_quantiles,
        "coverage": float(covered.mean()), "mean_width": float(width.mean()),
        "covered": covered, "width": width,
    }


def summarize_by_band(y_true, covered, width):
    out = {}
    for name, lo, hi in RUL_BANDS:
        mask = band_mask(y_true, lo, hi)
        if mask.sum() == 0:
            out[name] = None
            continue
        out[name] = {
            "n": int(mask.sum()),
            "coverage": float(covered[mask].mean()),
            "mean_width": float(width[mask].mean()),
        }
    return out


def main():
    v2_model_path = require_inputs()

    X_test = np.load(MODEL_READY_DIR / "X_test.npy").astype("float32")
    y_test = np.load(MODEL_READY_DIR / "y_test.npy").astype("float32")
    subset_test = np.load(MODEL_READY_DIR / "subset_test.npy")

    X_cal = np.load(MODEL_READY_DIR / "X_cal.npy").astype("float32")
    y_cal = np.load(MODEL_READY_DIR / "y_cal.npy").astype("float32")

    agent = RULAgent(model_path=v2_model_path)

    test_results = agent.predict(X_test)
    test_preds = np.array([r["predicted_RUL"] for r in test_results], dtype="float32")

    cal_results = agent.predict(X_cal)
    cal_preds = np.array([r["predicted_RUL"] for r in cal_results], dtype="float32")

    mae = float(np.mean(np.abs(test_preds - y_test)))
    rmse = float(np.sqrt(np.mean((test_preds - y_test) ** 2)))

    per_subset = {}
    for name in SUBSETS:
        mask = subset_test == name
        yt, yp = y_test[mask], test_preds[mask]
        per_subset[name] = {
            "n": int(mask.sum()),
            "mae": float(np.mean(np.abs(yp - yt))),
            "rmse": float(np.sqrt(np.mean((yp - yt) ** 2))),
            "nasa_score": nasa_score(yt, yp),
        }

    split_conformal = evaluate_split_conformal(cal_preds, y_cal, test_preds, y_test, ALPHA)
    mondrian = evaluate_mondrian_conformal(cal_preds, y_cal, test_preds, y_test, ALPHA)

    split_by_subset, mondrian_by_subset = {}, {}
    for name in SUBSETS:
        mask = subset_test == name
        split_by_subset[name] = {
            "n": int(mask.sum()),
            "coverage": float(split_conformal["covered"][mask].mean()),
            "mean_width": float(split_conformal["width"][mask].mean()),
        }
        mondrian_by_subset[name] = {
            "n": int(mask.sum()),
            "coverage": float(mondrian["covered"][mask].mean()),
            "mean_width": float(mondrian["width"][mask].mean()),
        }

    split_by_band = summarize_by_band(y_test, split_conformal["covered"], split_conformal["width"])
    mondrian_by_band = summarize_by_band(y_test, mondrian["covered"], mondrian["width"])

    metrics = {
        "n_test_engines": len(y_test),
        "n_calibration_engines": len(y_cal),
        "pooled": {"mae": mae, "rmse": rmse},
        "per_subset": per_subset,
        "conformal_split": {
            "alpha": ALPHA,
            "quantile": split_conformal["quantile"],
            "quantile_saturated": split_conformal["quantile_saturated"],
            "coverage": split_conformal["coverage"],
            "mean_width": split_conformal["mean_width"],
            "by_subset": split_by_subset,
            "by_rul_band": split_by_band,
        },
        "conformal_mondrian_by_predicted_rul_band": {
            "alpha": ALPHA,
            "band_quantiles": mondrian["band_quantiles"],
            "coverage": mondrian["coverage"],
            "mean_width": mondrian["mean_width"],
            "by_subset": mondrian_by_subset,
            "by_rul_band": mondrian_by_band,
        },
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"v2 pooled — MAE {mae:.2f}  RMSE {rmse:.2f}  (n={len(y_test)})")
    for name in SUBSETS:
        m = per_subset[name]
        print(f"  {name:6} — RMSE {m['rmse']:.2f}  MAE {m['mae']:.2f}  NASA {m['nasa_score']:.0f}  (n={m['n']})")

    sat_note = ", SATURATED (n_cal too small for exact target — see report)" if split_conformal["quantile_saturated"] else ""
    print(f"\nSplit conformal — quantile=±{split_conformal['quantile']:.1f} (n_cal={len(y_cal)}{sat_note})")
    print(f"  Coverage: {split_conformal['coverage']*100:.1f}%   Mean width: {split_conformal['mean_width']:.1f}")
    for name in SUBSETS:
        s = split_by_subset[name]
        print(f"    {name:6} — coverage {s['coverage']*100:.1f}%  width {s['mean_width']:.1f}  (n={s['n']})")
    for name, _, _ in RUL_BANDS:
        b = split_by_band[name]
        if b:
            print(f"    RUL {name:6} — coverage {b['coverage']*100:.1f}%  width {b['mean_width']:.1f}  (n={b['n']})")

    print(f"\nMondrian (by predicted-RUL band) — Coverage: {mondrian['coverage']*100:.1f}%   "
          f"Mean width: {mondrian['mean_width']:.1f}")
    for name in SUBSETS:
        s = mondrian_by_subset[name]
        print(f"    {name:6} — coverage {s['coverage']*100:.1f}%  width {s['mean_width']:.1f}  (n={s['n']})")
    for name, _, _ in RUL_BANDS:
        b = mondrian_by_band[name]
        if b:
            print(f"    RUL {name:6} — coverage {b['coverage']*100:.1f}%  width {b['mean_width']:.1f}  (n={b['n']})")

    print(f"\nSaved metrics to {REPORTS_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
