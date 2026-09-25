# Fix report — reproducibility, honesty and testing pass

Branch: `fix/reproducible-v2`. Not merged — for review.

Environment this was done in: Windows, Python 3.14.3, CPU only
(`torch.cuda.is_available()` → `False`). No GPU was available at any point,
so every training run below is CPU-only and timed accordingly.

## Summary

All "must do" and "should do" phases are complete, plus the optional Phase 4.
v1's model files, checkpoint and README numbers are untouched — every v2
artefact lives under `models/v2/` and `reports/v2/`. Nothing was tuned,
selected, or calibrated on the 707 test engines; they're touched exactly
once per evaluation script, at the very end.

## Phase 0 — setup and go/no-go check

Commit `518ecd4`.

`scripts/evaluate_rul.py` didn't exist yet, so it was written first (this is
also most of Phase 1, but the check had to happen before anything else could
proceed). Result on a clean run against the already-committed v1 checkpoint:

| Metric | CV-quoted | Reproduced here | Match |
|---|---|---|---|
| Pooled MAE | 12.20 | 12.20 | exact |
| Pooled RMSE | 17.58 | 17.58 | exact |
| Conformal coverage (non-cal engines) | 92.9% | 92.9% | exact |
| FD001–FD004 RMSE | 16.98 / 18.05 / 14.52 / 18.43 | same | exact |
| FD001–FD004 NASA score | 810 / 2127 / 603 / 2837 | 810 / 2126 / 602 / 2841 | within a few points |

The NASA-score gap (≤4 points on scores in the hundreds/thousands) is
consistent with CPU-vs-GPU floating-point non-determinism in the LSTM
forward pass — the model was originally trained and evaluated on a Colab
GPU, this run is CPU-only. MAE/RMSE/coverage — the numbers actually quoted
on the CV — reproduce exactly, so this passed the go/no-go gate and work
continued.

## Phase 1 — reproducible v1 pipeline (commit `4a86e6d`, hygiene in `fec23c9`)

- `scripts/prepare_data.py` is a script version of `notebooks/Pre-processing.ipynb`,
  starting from the committed `DATA/pre_processed_data/*.parquet` (columns
  already dropped there). Rebuilds `DATA/model_ready/*.npy` and was checked
  with `np.allclose` against the arrays the original notebook run had
  produced — bit-for-bit identical, including after a later refactor.
- Also saves per-window engine/subset/cycle metadata (`unit_train.npy`,
  `subset_train.npy`, `cycle_train.npy`, `unit_test.npy`, `subset_test.npy`)
  that the original notebook never needed — required for the v2 engine-level
  split in Phase 3.
- `scripts/train_rul.py` / `scripts/train_anomaly.py` reproduce the notebook
  training code with fixed seeds, a `--device` flag, and default output
  under `models/v2/` so v1 checkpoints can never be overwritten by running
  them.
- `main.py` / `agents/mapie.py` now raise a clear `FileNotFoundError` naming
  `scripts/prepare_data.py` if `DATA/model_ready` is missing.
- Hygiene: deleted `agents/Agent_4_descion.py` (an unused duplicate that
  still imported pre-rename module names and would crash if run — nothing
  imported it), added an MIT `LICENSE`, added `pyarrow`/`pytest`/`optuna` to
  `requirements.txt` (all were being used but undeclared), added
  `.env.example` and documented that `docker-compose.yml` requires `.env`.
- Added a `Makefile` (`prepare` / `evaluate` / `test`) and a README
  "Reproduce" section.

## Phase 2 — tests and CI (commit `d5b5919`)

`tests/test_data.py` adds, on a clean clone with no GPU and no API key:
shape/engine-count/NaN/RUL-range checks, the s6 scaled-range check (down to
≈ −0.6 in FD004 — verified, not guessed), a check that `prepare_data.py`
reproduces the committed arrays exactly, a v1 MAE/RMSE regression test
(±0.01), and a ≥90% conformal-coverage test. `.github/workflows/ci.yml` runs
Python 3.11 + CPU-only torch + `prepare_data.py` + pytest on push/PR, so the
data-backed tests actually run in CI instead of skipping.

## Phase 3 — v2 conformal calibration (commits `dcf61b4`, `20958d4`)

v1's calibration set was "the last 20% of `X_test`" (`agents/mapie.py`) —
file order makes this FD004-only, and it's part of the data used for final
evaluation. `scripts/prepare_calibration_split.py` splits the 709 *training*
engines instead, seeded (42) and stratified by sub-dataset: 80% fit
(further split 90/10 into fit-train/fit-val for early stopping), 20%
calibration. Calibration engines get one prediction each, at a uniformly
random cut point in their life, built the same way real test engines are
windowed (truncated history, edge-padded).

Engine counts (fit-train / fit-val / calibration): FD001 72/8/20, FD002
187/21/52, FD003 72/8/20, FD004 179/20/50 — sums to the known 100/260/100/249
training totals.

The CNN-BiLSTM was retrained on the 510 fit-train engines (89,451 windows)
with the **same v1 hyperparameters** (no re-tuning), using the 57 fit-val
engines (9,537 windows) for early stopping (patience 10). One CPU epoch
took ≈ 2 minutes; training stopped at **epoch 13/100** — well inside the
2.5-hour budget. `scripts/evaluate_v2.py` then computed the conformal
quantile from the 142 calibration engines (finite-sample quantile,
`ceil((n+1)(1−α))`-th order statistic) and evaluated once on all 707 test
engines.

Also computed, since time allowed: a Mondrian (group-conditional, by
predicted-RUL band) conformal variant. It narrows the average interval but
drops marginal coverage below the 90% target and makes the weakest band
worse — reported as a documented trade-off, not adopted as the default. See
the "before/after" table below for numbers.

## Phase 4 — v2 anomaly detector (commit `08ab58a`)

Attempted and completed, since Phases 1–3 were done and the CPU freed up
quickly (v2 RUL training took under 45 minutes against the 2.5-hour
budget).

`scripts/prepare_anomaly_v2_data.py` builds early-life-only windows (first
30% of each engine's life) from the same fit-train/calibration engine split
as Phase 3, with regime labels made unique across sub-datasets (`FD002_r3`,
not just `3`) — v1 pooled FD001, FD003 and each sub-dataset's KMeans cluster
0 under a single "condition 0". This required a small generalisation to
`agents/agent1_anomaly.py`: `AnomalyAgent.predict()` no longer force-casts
the condition to `float` before the threshold lookup, so it accepts either
v1's numeric keys or v2's regime strings. Verified both paths still resolve
correctly.

The autoencoder trained on 9,383 early-life windows (100 epochs, ≈ 1 minute
on CPU — far fewer windows than v1's 125,618). `scripts/evaluate_anomaly_v2.py`
calibrated a threshold per regime for a 5% false-alarm rate on 3,152
held-out early-life windows from the calibration engines (achieved 5.2%),
then evaluated once on the 707 test engines.

**Bug caught and fixed during this phase, before evaluating anything:** the
RUL-band boundaries in `evaluate_anomaly_v2.py` originally overlapped at
RUL = 100 (`61–100` and `≥100` both included it), double-counting 5 engines.
Fixed to a non-overlapping `>100` band matching the convention already used
in `evaluate_v2.py`, then re-run. The numbers in this report and in
`reports/v2/anomaly_metrics.json` are post-fix.

## Before / after: every metric, v1 vs v2

All numbers below are read from `reports/v1/metrics.json`,
`reports/v2/metrics.json` and `reports/v2/anomaly_metrics.json` — nothing
here is hand-typed past what those files already say.

**RUL prediction (point accuracy):**

| Sub-dataset | v1 RMSE | v2 RMSE | v1 MAE | v2 MAE | v1 NASA | v2 NASA |
|---|---|---|---|---|---|---|
| FD001 | 16.98 | 13.25 | 12.21 | 9.58 | 810 | 323 |
| FD002 | 18.05 | 13.52 | 12.86 | 9.47 | 2126 | 1037 |
| FD003 | 14.52 | 14.97 | 9.93 | 10.24 | 602 | 952 |
| FD004 | 18.43 | 14.60 | 12.42 | 10.29 | 2841 | 1586 |
| Pooled | 17.58 | 14.08 | 12.20 | 9.88 | — | — |

v2 is better on FD001/FD002/FD004 and pooled; **worse on FD003** on every
one of RMSE/MAE/NASA score, notably the NASA score (602 → 952), which the
asymmetric scoring function penalises heavily for a handful of late
predictions. Reported as-is — the brief is explicit that a worse result
still gets reported.

**Conformal calibration:**

| | v1 | v2 (split-conformal) | v2 (Mondrian, by predicted-RUL band) |
|---|---|---|---|
| Calibration set | last 20% of test (142 engines, FD004 only) | 142 held-out training engines, all 4 sub-datasets | same 142, grouped by predicted-RUL band |
| Interval | ±33.3 | ±27.8 | ~±36.2 avg (band-dependent) |
| Evaluated on | 565 of 707 test engines | all 707 | all 707 |
| Marginal coverage (target 90%) | 92.9% | 92.8% | 89.1% |
| RUL 0–30 coverage / width | not measured | 100.0% / 45.4 | 96.2% / 22.5 |
| RUL 31–60 coverage / width | not measured | 93.9% / 55.1 | 93.9% / 47.2 |
| RUL 61–100 coverage / width | not measured | 82.5% / 49.6 | 72.9% / 49.8 |
| RUL >100 coverage / width | not measured | 94.9% / 36.9 | 93.8% / 30.5 |

v2's split-conformal interval is ~17% narrower than v1's at essentially the
same marginal coverage, and — unlike v1 — can honestly report coverage on
the full test set rather than a 565-engine subset, because calibration
never touches test data. But band-level coverage reveals a real weak spot
neither version's marginal number would show: the 61–100 RUL band sits at
82.5%, under the 90% target. Mondrian narrows the average interval
further but pushes marginal coverage under target and makes that same weak
band worse (72.9%) with only 29 calibration engines backing its quantile
there — reported as a trade-off, not adopted as the default.

**Anomaly detection:**

| RUL band | v1 flag rate | v2 flag rate |
|---|---|---|
| ≤ 30 (near failure) | 39.6% | 56.6% |
| 31–60 | not measured | 24.6% |
| 61–100 | not measured | 10.2% |
| >100 (healthy) | 19.5% | 5.4% |

v2 catches more near-failure engines while raising far fewer false alarms
on healthy ones. Achieved false-alarm rate on calibration windows: 5.2%
against a 5% target. RUL ≤ 30 precision 0.60, recall 0.57 (150/707 flagged
overall vs 190/707 for v1).

## What wasn't done, and why

- **Optuna re-search for v2**: deliberately skipped, per the brief
  ("don't search hyperparameters again") — v2 reuses v1's exact
  hyperparameters.
- **Deeper investigation of the FD003 regression**: v2 is worse on FD003 by
  every RUL metric. I didn't dig into *why* (e.g. which engines drive the
  NASA-score jump) — flagging this as worth a look before trusting FD003
  numbers for anything beyond what's reported here.
- **Mondrian conformal** was computed but not adopted as the default output,
  for the coverage/band reasons above — it's fully computed and saved in
  `reports/v2/metrics.json` under `conformal_mondrian_by_predicted_rul_band`
  if you want to reconsider.
- Docker's `.env` requirement was **documented and given an example file**,
  not made optional — changing `docker-compose.yml`'s `env_file` behaviour
  felt like a larger, riskier change than the brief's "document it" option
  for a one-line fix.

Nothing else from the brief's Phase 0–5 scope was skipped.

## Exact commands to reproduce every number above

```bash
python -m venv venv && source venv/Scripts/activate   # or venv/bin/activate on Linux/Mac
pip install -r requirements.txt

python scripts/prepare_data.py                 # rebuilds DATA/model_ready/*.npy
python scripts/evaluate_rul.py                 # -> reports/v1/metrics.json (Phase 0/1 numbers)
python -m pytest tests/ -q                      # Phase 2

python scripts/prepare_calibration_split.py    # Phase 3 engine split
python scripts/train_rul.py \
  --train-x DATA/model_ready/X_fit.npy --train-y DATA/model_ready/y_fit.npy \
  --val-x DATA/model_ready/X_val.npy --val-y DATA/model_ready/y_val.npy \
  --epochs 100 --patience 10 --max-minutes 150 --device cpu \
  --output models/v2/agent2_rul_predictor.pt
python scripts/evaluate_v2.py                  # -> reports/v2/metrics.json

python scripts/prepare_anomaly_v2_data.py      # Phase 4 early-life windows
python scripts/train_anomaly.py \
  --train-x DATA/model_ready/X_early_fit.npy --train-cond DATA/model_ready/regime_early_fit.npy \
  --epochs 100 --max-minutes 20 --device cpu \
  --output models/v2/agent1_autoencoder.pt
python scripts/evaluate_anomaly_v2.py          # -> reports/v2/anomaly_metrics.json
```

## Before merging, please double-check

1. **The FD003 regression in v2** (RMSE 14.52 → 14.97, NASA score 602 → 952).
   It's real and reproducible, not a bug I could find, but I'd want a second
   look before calling v2 a strict upgrade for FD003 specifically.
2. **Which conformal variant to feature**, if either, beyond split-conformal
   — Mondrian's numbers are real but I made a judgement call not to lead
   with them given the sub-90%-target marginal coverage.
3. **`.env` handling for Docker** — I documented it rather than changing
   `docker-compose.yml`'s behaviour; flag if you'd rather it be optional.
4. Branch is pushed but **not merged**, per the brief.
