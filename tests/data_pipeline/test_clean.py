"""Stage 2a (data_pipeline/clean.py): the standard clean and the checks that keep joins honest."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from data_pipeline import clean
from shared.config import SourceConfig


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("AverageTemperature", "average_temperature"),
        ("State Name", "state_name"),
        ("CV (%)", "cv"),
        ("PM2.5", "pm2_5"),
        ("already_snake", "already_snake"),
    ],
)
def test_snake_case(raw: str, expected: str) -> None:
    assert clean.snake_case(raw) == expected


def test_standard_clean() -> None:
    raw = pd.DataFrame(
        {
            "State Name": ["  North Carolina ", "Ohio", "", "Ohio", None],
            "Value": [1.0, 2.0, np.nan, 2.0, np.nan],
            "Empty": [None] * 5,
            "value": [5, 6, 7, 6, 8],  # becomes a second "value"
        }
    )

    cleaned = clean.standard_clean(raw)

    assert list(cleaned.columns) == ["state_name", "value", "value_2"]  # empty column dropped
    assert cleaned["state_name"].tolist()[:2] == ["North Carolina", "Ohio"]
    assert cleaned["state_name"].isna().sum() == 2  # the blank and the None
    assert len(cleaned) == 4  # one duplicate row dropped


def frame(**columns: list) -> pd.DataFrame:
    return pd.DataFrame(columns)


def source(name: str, join_on: list[str] | None = None, how: str = "left") -> SourceConfig:
    return SourceConfig(name=name, type="file", uri=f"{name}.csv", join_on=join_on, how=how)


def test_left_join_suffixes_overlapping_columns_and_reports_unmatched_rows(
    caplog: pytest.LogCaptureFixture,
) -> None:
    left = frame(state=["NC", "OH", "TX"], value=[1, 2, 3])
    right = frame(state=["NC", "OH"], value=[10, 20], temp=[15.0, 10.0])

    with caplog.at_level(logging.WARNING):
        joined = clean.join_one(left, right, source("climate", ["state"]))

    assert list(joined.columns) == ["state", "value", "value_climate", "temp"]
    assert joined["temp"].tolist()[:2] == [15.0, 10.0]
    assert np.isnan(joined["temp"].iloc[2])
    assert "1 of 3 rows found no match" in caplog.text


def test_inner_join_drops_unmatched_rows() -> None:
    left = frame(state=["NC", "OH", "TX"])
    right = frame(state=["NC"], temp=[15.0])
    joined = clean.join_one(left, right, source("climate", ["state"], how="inner"))
    assert joined["state"].tolist() == ["NC"]


@pytest.mark.parametrize(
    ("right", "message"),
    [
        (frame(state=["NC", "NC"], temp=[1.0, 2.0]), "repeat a"),
        (frame(region=["NC"], temp=[1.0]), "has no column"),
        (frame(state=[1], temp=[1.0]), "int64 in the source"),
    ],
    ids=["repeated-keys", "missing-key", "key-types-differ"],
)
def test_joins_that_would_go_wrong_are_rejected(right: pd.DataFrame, message: str) -> None:
    left = frame(state=["NC", "OH"])
    with pytest.raises(ValueError, match=message):
        clean.join_one(left, right, source("climate", ["state"]))


def test_blank_join_keys_never_match() -> None:
    left = frame(state=["NC", None])
    right = frame(state=["NC", None], temp=[15.0, 99.0])

    joined = clean.join_one(left, right, source("climate", ["state"]))

    assert joined["temp"].tolist()[0] == 15.0
    assert np.isnan(joined["temp"].iloc[1])  # not the 99.0 from the other blank-key row
