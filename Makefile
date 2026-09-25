.PHONY: prepare evaluate test

# Rebuild DATA/model_ready/*.npy from the committed parquet files.
prepare:
	python scripts/prepare_data.py

# Reproduce the v1 headline numbers (MAE, RMSE, NASA score, conformal coverage).
evaluate:
	python scripts/evaluate_rul.py

test:
	python -m pytest tests/ -q
