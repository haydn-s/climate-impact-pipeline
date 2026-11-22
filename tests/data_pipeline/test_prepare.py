"""Stage 2 (data_pipeline/prepare.py): cleaning, joining, and feature steps run end to end."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_pipeline import clean, features, prepare
from data_pipeline.ingest import ingest
from tests.conftest import pipeline_config


def write_csv(path: Path, data: pd.DataFrame) -> str:
    data.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def two_sources(tmp_path: Path) -> list[dict]:
    """A customers table plus a region lookup table."""
    rng = np.random.default_rng(0)
    customers = pd.DataFrame(
        {
            "customer_id": range(60),
            "region": rng.choice(["north", "south", "east", "west"], 60),
            "monthly_spend": rng.uniform(5, 120, 60).round(2),
        }
    )
    regions = pd.DataFrame(
        {"Region": ["north", "south", "east", "west"], "Avg Income": [50, 40, 60, 55]}
    )
    return [
        {"name": "customers", "type": "file", "uri": write_csv(tmp_path / "c.csv", customers)},
        {
            "name": "regions",
            "type": "file",
            "uri": write_csv(tmp_path / "r.csv", regions),
            "join_on": ["region"],
        },
    ]


def test_cleaners_and_feature_steps_run_in_order(
    tmp_path: Path, two_sources: list[dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    def upper_regions(data: pd.DataFrame) -> pd.DataFrame:
        return data.assign(region=data["region"].str.upper())

    def spend_share(table: pd.DataFrame, sources: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        return table.assign(spend_share=table["monthly_spend"] / table["avg_income"])

    # Both sources get uppercased, so the join still matches.
    monkeypatch.setitem(clean.CLEANERS, "customers", upper_regions)
    monkeypatch.setitem(clean.CLEANERS, "regions", upper_regions)
    monkeypatch.setitem(features.FEATURE_STEPS, "spend_share", spend_share)
    cfg = pipeline_config(tmp_path, two_sources, "prepare.feature_steps=[spend_share]")
    ingest(cfg)

    table = prepare.prepare(cfg)

    assert set(table["region"]) <= {"NORTH", "SOUTH", "EAST", "WEST"}
    assert table["avg_income"].notna().all()
    assert table["spend_share"].notna().all()
    assert pd.read_parquet(cfg.data.local_path).shape == table.shape


def test_sources_without_join_on_reach_feature_steps_only(
    tmp_path: Path, two_sources: list[dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    del two_sources[1]["join_on"]
    seen: dict[str, pd.DataFrame] = {}

    def look(table: pd.DataFrame, sources: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        seen.update(sources)
        return table

    monkeypatch.setitem(features.FEATURE_STEPS, "look", look)
    cfg = pipeline_config(tmp_path, two_sources, "prepare.feature_steps=[look]")
    ingest(cfg)

    table = prepare.prepare(cfg)

    assert "avg_income" not in table.columns
    assert list(seen["regions"].columns) == ["region", "avg_income"]


def test_unknown_feature_step_fails_before_any_work(
    tmp_path: Path, two_sources: list[dict]
) -> None:
    cfg = pipeline_config(tmp_path, two_sources, "prepare.feature_steps=[nope]")
    with pytest.raises(ValueError, match="aren't registered"):
        prepare.prepare(cfg)


def test_missing_snapshot_says_to_ingest(tmp_path: Path, two_sources: list[dict]) -> None:
    with pytest.raises(FileNotFoundError, match="make ingest"):
        prepare.prepare(pipeline_config(tmp_path, two_sources))


def test_missing_snapshot_is_downloaded_from_the_remote_prefix(
    tmp_path: Path, two_sources: list[dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    staged = tmp_path / "bucket"
    staged.mkdir()
    ingest(pipeline_config(staged, two_sources))  # snapshots in staged/raw

    def fake_download(uri: str, dest: Path) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((staged / "raw" / uri.rsplit("/", 1)[1]).read_bytes())
        return dest

    monkeypatch.setattr(clean, "download", fake_download)
    cfg = pipeline_config(tmp_path, two_sources, "ingest.remote_prefix=gs://bucket/raw")

    assert len(prepare.prepare(cfg)) == 60


def test_command_line_writes_the_processed_table(tmp_path: Path, two_sources: list[dict]) -> None:
    cfg = pipeline_config(tmp_path, two_sources)
    ingest(cfg)
    config = tmp_path / "config.yaml"
    config.write_text(cfg.model_dump_json())  # JSON is valid YAML

    table = prepare.main(["--config", str(config)])

    assert len(table) == 60
    assert cfg.data.local_path.exists()


def test_cleaner_must_return_a_frame(
    tmp_path: Path, two_sources: list[dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    def forgets_to_return(data: pd.DataFrame) -> None:
        data["region"] = data["region"].str.upper()

    monkeypatch.setitem(clean.CLEANERS, "customers", forgets_to_return)
    cfg = pipeline_config(tmp_path, two_sources)
    ingest(cfg)

    with pytest.raises(TypeError, match="the cleaner for customers returned NoneType"):
        prepare.prepare(cfg)


def test_empty_snapshot_names_the_source(tmp_path: Path, two_sources: list[dict]) -> None:
    cfg = pipeline_config(tmp_path, two_sources)
    cfg.ingest.raw_dir.mkdir(parents=True)
    (cfg.ingest.raw_dir / "customers.csv").write_text("")

    with pytest.raises(ValueError, match=r"snapshot of source 'customers'.*is empty"):
        prepare.prepare(cfg)
