import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

# Sensors kept after preprocessing (dropped: s1, s5, s10, s16, s19 + op1, op2 used for condition only)
FEATURE_NAMES = [
    "T24", "T30", "T50", "P15", "P30",
    "Nf",  "Nc",  "Ps30", "phi", "NRf",
    "NRc", "BPR", "htBleed", "Nf_dmd", "W31", "W32"
]


class CNNBiLSTM(nn.Module):
    def __init__(self, input_size, num_filters, kernel_size, hidden_size, num_layers, dropout):
        super(CNNBiLSTM, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, num_filters, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(num_filters),
            nn.Conv1d(num_filters, num_filters, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(num_filters)
        )

        self.bilstm = nn.LSTM(
            input_size=num_filters,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        x, _ = self.bilstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x.squeeze(1)


class RULAgent:
    def __init__(self, model_path, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        params = checkpoint["best_params"]
        self.model = CNNBiLSTM(
            input_size=checkpoint["input_size"],
            num_filters=params["num_filters"],
            kernel_size=params["kernel_size"],
            hidden_size=params["hidden_size"],
            num_layers=params["num_layers"],
            dropout=params["dropout"]
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.mae  = checkpoint["mae"]
        self.rmse = checkpoint["rmse"]

    def predict(self, X, batch_size=256):
        preds   = []
        dataset = TensorDataset(torch.tensor(X.astype(np.float32)))
        loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        with torch.no_grad():
            for batch in loader:
                x   = batch[0].to(self.device)
                out = self.model(x)
                preds.extend(out.cpu().numpy())

        preds = np.clip(np.array(preds), 0, 125)

        results = []
        for i, pred in enumerate(preds):
            results.append({
                "engine_index"  : i,
                "predicted_RUL" : round(float(pred), 1),
                "confidence"    : round(float(self.mae), 1)
            })

        return results

    def explain(self, X, background_size=100):
        """
        Compute per-sensor SHAP importances using GradientExplainer.

        Returns
        -------
        mean_shap : np.ndarray, shape (n_engines, 16)
            Mean absolute SHAP value per sensor per engine.
        feature_names : list[str]
            Sensor names matching the 16 columns.
        global_importance : np.ndarray, shape (16,)
            Fleet-level mean absolute SHAP per sensor (for bar charts).
        """
        import shap

        # background: random subset used as reference distribution
        idx = np.random.choice(len(X), min(background_size, len(X)), replace=False)
        background = torch.tensor(X[idx], dtype=torch.float32).to(self.device)

        # SHAP needs 2D output — wrap model to unsqueeze
        class ModelWrapper(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
            def forward(self, x):
                return self.m(x).unsqueeze(1)

        wrapper     = ModelWrapper(self.model)
        explainer   = shap.GradientExplainer(wrapper, background)
        X_tensor    = torch.tensor(X, dtype=torch.float32).to(self.device)
        shap_values = explainer.shap_values(X_tensor)  # list of 1 array: (n, 50, 16)
        shap_values = shap_values[0] if isinstance(shap_values, list) else shap_values

        # average absolute SHAP over the 50 time steps → (n, 16)
        mean_shap         = np.abs(shap_values).mean(axis=1)   # (n, 16)
        global_importance = mean_shap.mean(axis=0).flatten()   # (16,)

        return mean_shap, FEATURE_NAMES, global_importance


if __name__ == "__main__":
    MODEL_PATH      = Path(__file__).parent.parent / "models" / "agent2_rul_predictor.pt"
    MODEL_READY_DIR = Path(__file__).parent.parent / "DATA" / "model_ready"

    X_test = np.load(MODEL_READY_DIR / "X_test.npy").astype(np.float32)
    y_test = np.load(MODEL_READY_DIR / "y_test.npy").astype(np.float32)

    agent   = RULAgent(model_path=MODEL_PATH)
    results = agent.predict(X_test)

    preds = np.array([r["predicted_RUL"] for r in results])
    mae   = np.mean(np.abs(preds - y_test))
    rmse  = np.sqrt(np.mean((preds - y_test) ** 2))

    print(f"Total engines:  {len(results)}")
    print(f"MAE:            {mae:.2f} cycles")
    print(f"RMSE:           {rmse:.2f} cycles")
    print(f"\nSample result:  {results[0]}")

    # SHAP explanations
    print("\nComputing SHAP values (this may take ~30 seconds)...")
    mean_shap, feature_names, global_importance = agent.explain(X_test)

    print("\nGlobal sensor importance (mean |SHAP|):")
    ranked = sorted(zip(feature_names, global_importance.flatten().tolist()), key=lambda x: x[1], reverse=True)
    for name, score in ranked:
        bar = "█" * int(score * 500)
        print(f"  {name:10} {score:.5f}  {bar}")
