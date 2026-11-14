# ML pipeline

Turns the processed table into a trained model that the API serves. This package is empty for now, ready to be built.

## What it reads

- **The processed table.** The data pipeline pushes it to `data.remote_uri` in `config.yaml` (under `gs://climate-impact-pipeline-data/processed/`). The bucket is public, so downloading it needs no login. `shared.storage.download()` copies it to `data.local_path`, and `shared.tables.read_table()` reads CSV or Parquet.
- **The shared part of `config.yaml`.** `data.target` and the `features` lists name the columns to train on. The data pipeline refuses to push a table that lacks them (`shared.tables.missing_columns()`), so they're the contract between the two pipelines. Load the config with `shared.config.load_config()`.
- **The ML part of `config.yaml`.** It holds `seed`, `task`, `split`, `model`, `evaluation`, `tracking`, and `output`. `shared/config.py` validates these sections. This is a general template taken from guidance online, so keep, change, or replace them as the pipeline needs.

## What the API needs from it

[`api/main.py`](../api/main.py) loads two files from the folder `MODEL_URI` points to:

- **`model.joblib`:** an object saved with `joblib.dump`, usually a scikit-learn `Pipeline` that includes its own preprocessing. The API calls `.predict()` on a one-row DataFrame whose columns are the feature names, and `.predict_proba()` too for classification.
- **`metadata.json`:** the API builds its `/predict` request schema from this file. It needs:
  - `model_version`, `created_at`, `task` (`classification` or `regression`), `target`, `model_type`, `primary_metric`
  - `metrics`, e.g. `{"val": {"f1": 0.8}, "test": {"f1": 0.78}}`
  - `features`: a list of `{"name", "kind": "numeric", "min", "max", "example"}` or `{"name", "kind": "categorical", "categories", "example"}`
  - `example`: one valid request, keyed by feature name
  - `classes` (classification only), and `versions` with the `scikit-learn` version, which the API checks against its own

Publish each run to `gs://climate-impact-pipeline-data/models/<run-id>/` and copy it to `models/latest/`, which the deployed API loads at startup. Uploads need a Google login with write access to the bucket. See "Set up your .env" in the [main README](../README.md#set-up-your-env).

## Boundaries

- **Imports:** this package may import from `shared` only, never from `data_pipeline`. `tests/test_layout.py` checks that. Put tests in `tests/ml_pipeline/`.
- **Tracking:** runs log to the W&B team `dhs27-duke-university` (`tracking.entity`).
- **Tests:** when you add the model training pipeline, we need to add smoke tests for the Docker container and for the API it serves. That can be done in a later step.
