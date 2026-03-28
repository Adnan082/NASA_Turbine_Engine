import torch
from pathlib import Path

model_path = Path(__file__).parent / "agent2_rul_predictor.pt"
checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

print("Best params:", checkpoint["best_params"])
print("Input size:",  checkpoint["input_size"])
print("MAE:",         checkpoint["mae"])
print("RMSE:",        checkpoint["rmse"])