"""API checks that need no trained model.

Tests that train a model and call /health, /metadata, and /predict come back
with the ML pipeline (ml_pipeline/README.md).
"""

import json
from pathlib import Path

import joblib
import pytest

import api.main
from api.main import create_app, load_bundle


def test_missing_model_says_how_to_fix_it(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="set MODEL_URI"):
        create_app(str(tmp_path / "model.joblib"))


def test_cloud_model_downloads_both_files_to_the_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Stand-ins for the two files the ML pipeline publishes.
    published = tmp_path / "published"
    published.mkdir()
    joblib.dump({"stand-in": "model"}, published / "model.joblib")
    metadata = {"model_version": "test-run", "features": [{"name": "age"}, {"name": "plan"}]}
    (published / "metadata.json").write_text(json.dumps(metadata))
    requested: list[str] = []

    def fake_download(uri: str, dest: Path) -> Path:
        requested.append(uri)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((published / uri.rsplit("/", 1)[1]).read_bytes())
        return dest

    monkeypatch.setattr(api.main, "download", fake_download)
    monkeypatch.setenv("MODEL_CACHE_DIR", str(tmp_path / "cache"))

    bundle = load_bundle("gs://bucket/models/latest/model.joblib")

    assert requested == [
        "gs://bucket/models/latest/model.joblib",
        "gs://bucket/models/latest/metadata.json",
    ]
    assert (bundle.version, bundle.feature_names) == ("test-run", ["age", "plan"])
