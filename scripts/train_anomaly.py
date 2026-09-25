"""
Train the LSTM autoencoder anomaly detector (Agent 1).

With no arguments this reproduces notebooks/agent_1_anomaly_detection.ipynb:
an Optuna search (15 trials, 10 epochs, 10% subsample) followed by a
100-epoch retrain on the full training set (all windows, not just healthy
ones — the notebook never filtered by life stage, see FIX_REPORT.md), then
a per-condition threshold at the 75th percentile of training reconstruction
error. Pass --no-search to train fixed hyperparameters directly, and
--train-x/--train-cond to point at a filtered subset (used by the v2
early-life-only detector).

This never writes to models/agent1_autoencoder.pt — the default output is
models/v2/, so the v1 checkpoint stays untouched.

Usage:
    python scripts/train_anomaly.py --device cpu
    python scripts/train_anomaly.py --no-search --train-x DATA/model_ready/X_early_life.npy \\
        --train-cond DATA/model_ready/cond_early_life.npy --threshold-percentile 95
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT / "agents"))

from agent1_anomaly import LSTMAutoencoder  # noqa: E402

MODEL_READY_DIR = ROOT / "DATA" / "model_ready"
DEFAULT_OUTPUT = ROOT / "models" / "v2" / "agent1_autoencoder.pt"

# Optuna's best trial from notebooks/agent_1_anomaly_detection.ipynb (v1).
V1_PARAMS = dict(hidden_size=64, num_layers=1, dropout=0.39975334326526507, lr=0.009621810900958797)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    for (x_batch,) in loader:
        x_batch = x_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x_batch), x_batch)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def reconstruction_errors(model, X, device, batch_size=256):
    model.eval()
    criterion = nn.MSELoss(reduction="none")
    loader = DataLoader(TensorDataset(torch.tensor(X)), batch_size=batch_size, shuffle=False)
    errors = []
    for (x_batch,) in loader:
        x_batch = x_batch.to(device)
        out = model(x_batch)
        error = criterion(out, x_batch).mean(dim=[1, 2]).cpu().numpy()
        errors.extend(error)
    return np.array(errors)


def run_search(X_train, device, seed, trials, search_epochs, search_frac):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    rng = np.random.RandomState(seed)
    search_size = int(len(X_train) * search_frac)
    idx = rng.choice(len(X_train), size=search_size, replace=False)
    X_search = X_train[idx]

    search_loader = DataLoader(TensorDataset(torch.tensor(X_search)), batch_size=256, shuffle=True)
    input_size = X_train.shape[2]

    def objective(trial):
        hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128, 256])
        num_layers = trial.suggest_categorical("num_layers", [1, 2])
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        dropout = trial.suggest_float("dropout", 0.1, 0.4)

        model = LSTMAutoencoder(input_size=input_size, hidden_size=hidden_size,
                                 num_layers=num_layers, dropout=dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        loss = None
        for _ in range(search_epochs):
            loss = train_one_epoch(model, search_loader, optimizer, criterion, device)
        return loss

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=trials)
    print(f"Best params: {study.best_params}  (loss {study.best_value:.6f})")
    return study.best_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--train-x", default=str(MODEL_READY_DIR / "X_train.npy"))
    parser.add_argument("--train-cond", default=str(MODEL_READY_DIR / "cond_train.npy"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-minutes", type=float, default=None)
    parser.add_argument("--threshold-percentile", type=float, default=75.0,
                         help="per-condition reconstruction-error percentile used as the anomaly threshold")
    parser.add_argument("--search", action="store_true")
    parser.add_argument("--search-trials", type=int, default=15)
    parser.add_argument("--search-epochs", type=int, default=10)
    parser.add_argument("--search-frac", type=float, default=0.10)
    parser.add_argument("--hidden-size", type=int, default=V1_PARAMS["hidden_size"])
    parser.add_argument("--num-layers", type=int, default=V1_PARAMS["num_layers"])
    parser.add_argument("--dropout", type=float, default=V1_PARAMS["dropout"])
    parser.add_argument("--lr", type=float, default=V1_PARAMS["lr"])
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(args.seed)
    print(f"Device: {device}")

    X_train = np.load(args.train_x).astype(np.float32)
    cond_train = np.load(args.train_cond)
    input_size = X_train.shape[2]

    if args.search:
        print(f"Running Optuna search ({args.search_trials} trials, "
              f"{args.search_epochs} epochs, {args.search_frac:.0%} subsample)...")
        best_params = run_search(X_train, device, args.seed,
                                  args.search_trials, args.search_epochs, args.search_frac)
    else:
        best_params = dict(hidden_size=args.hidden_size, num_layers=args.num_layers, dropout=args.dropout)
        print(f"Using fixed hyperparameters: {best_params}  lr={args.lr}")

    model = LSTMAutoencoder(
        input_size=input_size, hidden_size=best_params["hidden_size"],
        num_layers=best_params["num_layers"], dropout=best_params["dropout"],
    ).to(device)

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train)), batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    start_time = time.time()
    actual_epochs = 0
    for epoch in range(args.epochs):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        actual_epochs = epoch + 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{args.epochs} — loss {loss:.6f}")

        elapsed_minutes = (time.time() - start_time) / 60
        if args.max_minutes and elapsed_minutes > args.max_minutes:
            print(f"Hit --max-minutes={args.max_minutes} after {actual_epochs} epochs — stopping.")
            break

    print(f"Computing per-condition thresholds (p{args.threshold_percentile})...")
    train_errors = reconstruction_errors(model, X_train, device)
    thresholds = {}
    for condition in np.unique(cond_train):
        mask = cond_train == condition
        thresholds[condition] = float(np.percentile(train_errors[mask], args.threshold_percentile))
        print(f"  Condition {condition}: threshold {thresholds[condition]:.6f}  (n={mask.sum()})")

    best_params["lr"] = args.lr
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "best_params": best_params,
        "thresholds": thresholds,
        "input_size": input_size,
        "actual_epochs": actual_epochs,
        "seed": args.seed,
        "threshold_percentile": args.threshold_percentile,
    }, output_path)

    print(f"\nSaved model to {output_path} ({actual_epochs} epochs trained)")


if __name__ == "__main__":
    main()
