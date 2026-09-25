"""
Train the CNN-BiLSTM RUL model (Agent 2).

With no arguments this reproduces notebooks/Agent_2_RUL_prediction.ipynb:
an Optuna search (15 trials, 10 epochs, 10% subsample) followed by a
100-epoch retrain on the full training set, seeded for reproducibility.
Pass --no-search to skip the search and train fixed hyperparameters
directly (used by scripts/evaluate_v2.py, which reuses the v1
hyperparameters and adds a validation split for early stopping).

This never writes to models/agent2_rul_predictor.pt — the default output
is models/v2/, so the v1 checkpoint stays untouched. Use --output to
point elsewhere explicitly.

Usage:
    python scripts/train_rul.py --device cpu
    python scripts/train_rul.py --no-search --train-x DATA/model_ready/X_fit.npy \\
        --train-y DATA/model_ready/y_fit.npy --val-x DATA/model_ready/X_val.npy \\
        --val-y DATA/model_ready/y_val.npy --patience 5 --max-minutes 120
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

from agent2_rul import CNNBiLSTM  # noqa: E402 — reuse the exact inference architecture

MODEL_READY_DIR = ROOT / "DATA" / "model_ready"
DEFAULT_OUTPUT = ROOT / "models" / "v2" / "agent2_rul_predictor.pt"

# Optuna's best trial from notebooks/Agent_2_RUL_prediction.ipynb (v1).
V1_PARAMS = dict(
    num_filters=64, kernel_size=3, hidden_size=128, num_layers=2,
    dropout=0.15199452886924075, lr=0.000830315325478894,
)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    """Returns (avg MSE loss, MAE, RMSE) over a loader."""
    model.eval()
    criterion = nn.MSELoss()
    total_loss, preds, targets = 0.0, [], []
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        out = model(X_batch)
        total_loss += criterion(out, y_batch).item()
        preds.append(out.cpu().numpy())
        targets.append(y_batch.cpu().numpy())
    preds, targets = np.concatenate(preds), np.concatenate(targets)
    mae = float(np.mean(np.abs(preds - targets)))
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    return total_loss / len(loader), mae, rmse


def run_search(X_train, y_train, device, seed, trials, search_epochs, search_frac):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    rng = np.random.RandomState(seed)
    search_size = int(len(X_train) * search_frac)
    idx = rng.choice(len(X_train), size=search_size, replace=False)
    X_search, y_search = X_train[idx], y_train[idx]

    search_loader = DataLoader(
        TensorDataset(torch.tensor(X_search), torch.tensor(y_search)),
        batch_size=256, shuffle=True,
    )
    input_size = X_train.shape[2]

    def objective(trial):
        params = dict(
            num_filters=trial.suggest_categorical("num_filters", [32, 64, 128]),
            kernel_size=trial.suggest_categorical("kernel_size", [3, 5, 7]),
            hidden_size=trial.suggest_categorical("hidden_size", [32, 64, 128]),
            num_layers=trial.suggest_categorical("num_layers", [1, 2]),
        )
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        dropout = trial.suggest_float("dropout", 0.1, 0.4)

        model = CNNBiLSTM(input_size=input_size, dropout=dropout, **params).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        loss = None
        for _ in range(search_epochs):
            loss = train_one_epoch(model, search_loader, optimizer, criterion, device)
        return loss

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=trials)
    print(f"Best params: {study.best_params}  (loss {study.best_value:.4f})")
    return study.best_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--train-x", default=str(MODEL_READY_DIR / "X_train.npy"))
    parser.add_argument("--train-y", default=str(MODEL_READY_DIR / "y_train.npy"))
    parser.add_argument("--val-x", default=None)
    parser.add_argument("--val-y", default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=None,
                         help="early-stopping patience on val loss; requires --val-x/--val-y")
    parser.add_argument("--max-minutes", type=float, default=None,
                         help="wall-clock training cap; stops early and records actual epochs")
    parser.add_argument("--search", action="store_true",
                         help="run the v1-style Optuna search instead of fixed hyperparameters")
    parser.add_argument("--search-trials", type=int, default=15)
    parser.add_argument("--search-epochs", type=int, default=10)
    parser.add_argument("--search-frac", type=float, default=0.10)
    parser.add_argument("--num-filters", type=int, default=V1_PARAMS["num_filters"])
    parser.add_argument("--kernel-size", type=int, default=V1_PARAMS["kernel_size"])
    parser.add_argument("--hidden-size", type=int, default=V1_PARAMS["hidden_size"])
    parser.add_argument("--num-layers", type=int, default=V1_PARAMS["num_layers"])
    parser.add_argument("--dropout", type=float, default=V1_PARAMS["dropout"])
    parser.add_argument("--lr", type=float, default=V1_PARAMS["lr"])
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(args.seed)
    print(f"Device: {device}")

    X_train = np.load(args.train_x).astype(np.float32)
    y_train = np.load(args.train_y).astype(np.float32)
    input_size = X_train.shape[2]

    if args.search:
        print(f"Running Optuna search ({args.search_trials} trials, "
              f"{args.search_epochs} epochs, {args.search_frac:.0%} subsample)...")
        best_params = run_search(X_train, y_train, device, args.seed,
                                  args.search_trials, args.search_epochs, args.search_frac)
    else:
        best_params = dict(
            num_filters=args.num_filters, kernel_size=args.kernel_size,
            hidden_size=args.hidden_size, num_layers=args.num_layers,
            dropout=args.dropout, lr=args.lr,
        )
        print(f"Using fixed hyperparameters: {best_params}")

    model = CNNBiLSTM(
        input_size=input_size,
        num_filters=best_params["num_filters"], kernel_size=best_params["kernel_size"],
        hidden_size=best_params["hidden_size"], num_layers=best_params["num_layers"],
        dropout=best_params["dropout"],
    ).to(device)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=args.batch_size, shuffle=True,
    )

    val_loader = None
    if args.val_x and args.val_y:
        X_val = np.load(args.val_x).astype(np.float32)
        y_val = np.load(args.val_y).astype(np.float32)
        val_loader = DataLoader(
            TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
            batch_size=args.batch_size, shuffle=False,
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"])
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    start_time = time.time()
    actual_epochs = 0

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        actual_epochs = epoch + 1

        if val_loader is not None:
            val_loss, val_mae, val_rmse = evaluate(model, val_loader, device)
            scheduler.step(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
        else:
            scheduler.step(train_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            msg = f"Epoch {epoch + 1}/{args.epochs} — train loss {train_loss:.4f}"
            if val_loader is not None:
                msg += f"  val loss {val_loss:.4f}  val MAE {val_mae:.2f}  val RMSE {val_rmse:.2f}"
            print(msg)

        elapsed_minutes = (time.time() - start_time) / 60
        if args.max_minutes and elapsed_minutes > args.max_minutes:
            print(f"Hit --max-minutes={args.max_minutes} after {actual_epochs} epochs — stopping.")
            break

        if args.patience and val_loader is not None and epochs_without_improvement >= args.patience:
            print(f"Early stopping: no val improvement in {args.patience} epochs "
                  f"(stopped at epoch {actual_epochs}).")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        _, mae, rmse = evaluate(model, val_loader, device)
    else:
        # No validation split (v1-parity full-data run) — report training-set
        # fit only; it is not a generalisation estimate. Real v2 numbers come
        # from scripts/evaluate_v2.py against the untouched test set.
        _, mae, rmse = evaluate(model, train_loader, device)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "best_params": best_params,
        "input_size": input_size,
        "mae": mae,
        "rmse": rmse,
        "actual_epochs": actual_epochs,
        "seed": args.seed,
        "used_validation_split": val_loader is not None,
    }, output_path)

    print(f"\nSaved model to {output_path} ({actual_epochs} epochs trained)")
    print(f"{'Val' if val_loader is not None else 'Train'} MAE: {mae:.2f}  RMSE: {rmse:.2f}")


if __name__ == "__main__":
    main()
