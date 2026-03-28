import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(LSTMAutoencoder, self).__init__()

        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        self.output_layer = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        _, (hidden, _) = self.encoder(x)
        repeated = hidden[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        decoded, _ = self.decoder(repeated)
        out = self.output_layer(decoded)
        return out


class AnomalyAgent:
    def __init__(self, model_path, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        params = checkpoint["best_params"]
        self.model = LSTMAutoencoder(
            input_size=checkpoint["input_size"],
            hidden_size=params["hidden_size"],
            num_layers=params["num_layers"],
            dropout=params["dropout"]
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.thresholds = checkpoint["thresholds"]

    def get_reconstruction_errors(self, X, batch_size=256):
        errors  = []
        criterion = nn.MSELoss(reduction="none")
        dataset = TensorDataset(torch.tensor(X.astype(np.float32)))
        loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        with torch.no_grad():
            for batch in loader:
                x = batch[0].to(self.device)
                reconstructed = self.model(x)
                error = criterion(reconstructed, x).mean(dim=[1, 2]).cpu().numpy()
                errors.extend(error)

        return np.array(errors)

    def get_status(self, error, threshold):
        if error < threshold:
            return "NORMAL"
        elif error < threshold * 1.5:
            return "SUSPICIOUS"
        else:
            return "CRITICAL"

    def predict(self, X, conditions):
        errors  = self.get_reconstruction_errors(X)
        results = []

        for i in range(len(X)):
            condition = float(conditions[i])
            threshold = self.thresholds.get(condition, list(self.thresholds.values())[0])
            status    = self.get_status(errors[i], threshold)

            results.append({
                "engine_index"         : i,
                "reconstruction_error" : float(errors[i]),
                "threshold"            : float(threshold),
                "status"               : status
            })

        return results


if __name__ == "__main__":
    MODEL_PATH      = Path(__file__).parent.parent / "models" / "agent1_autoencoder.pt"
    MODEL_READY_DIR = Path(__file__).parent.parent / "DATA" / "model_ready"

    X_test    = np.load(MODEL_READY_DIR / "X_test.npy").astype(np.float32)
    cond_test = np.load(MODEL_READY_DIR / "cond_test.npy")

    agent   = AnomalyAgent(model_path=MODEL_PATH)
    results = agent.predict(X_test, cond_test)

    normal     = sum(1 for r in results if r["status"] == "NORMAL")
    suspicious = sum(1 for r in results if r["status"] == "SUSPICIOUS")
    critical   = sum(1 for r in results if r["status"] == "CRITICAL")

    print(f"Total engines:  {len(results)}")
    print(f"Normal:         {normal}")
    print(f"Suspicious:     {suspicious}")
    print(f"Critical:       {critical}")
    print(f"\nSample result:  {results[0]}")
