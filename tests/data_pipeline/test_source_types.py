"""The dataset-specific source types in data_pipeline/ingest.py, with the network faked."""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from data_pipeline import ingest
from shared.config import load_config
from tests.conftest import REPO, REPO_CONFIG, pipeline_config

TOKEN = "not-a-real-token-5678"
CDO_URL = "https://cdo.example.org/api/v2/data"
CMR_URL = "https://cmr.example.org/search/granules.json"
FILE_URL = "https://archive.example.org/protected/greenland_mass_200204_202609.txt"


class FakeResponse:
    def __init__(self, payload: Any = None, *, text: str = "", content_type: str = "") -> None:
        self.payload, self.text = payload, text
        self.content = text.encode()
        self.headers = {"Content-Type": content_type or "application/json"}

    def json(self) -> Any:
        return self.payload


@pytest.fixture
def http(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Replace http_get: `respond(url, params)` picks each response, and every call is recorded."""
    fake = SimpleNamespace(respond=None, calls=[])

    def http_get(url: str, *, params=None, headers=None, secrets=()) -> FakeResponse:
        fake.calls.append({"url": url, "params": params, "headers": headers})
        return fake.respond(url, params or {})

    monkeypatch.setattr(ingest, "http_get", http_get)
    monkeypatch.setattr(ingest, "NOAA_CDO_PAUSE_SECONDS", 0)
    monkeypatch.setenv("TEST_TOKEN", TOKEN)
    return fake


def cdo_source(**params: Any) -> dict[str, Any]:
    return {
        "name": "climate",
        "type": "noaa_cdo",
        "url": CDO_URL,
        "headers": {"token": "${TEST_TOKEN}"},
        "params": {"datasetid": "GSOY", "datatypeid": ["TAVG", "PRCP"], **params},
    }


def test_noaa_cdo_pages_through_every_result(
    tmp_path: Path, http: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ingest, "NOAA_CDO_PAGE_SIZE", 2)
    rows = [
        {"date": f"20{y}-01-01T00:00:00", "datatype": "TAVG", "value": y} for y in range(15, 20)
    ]

    def respond(url: str, params: dict[str, Any]) -> FakeResponse:
        start = params["offset"] - 1
        page = rows[start : start + params["limit"]]
        return FakeResponse({"metadata": {"resultset": {"count": len(rows)}}, "results": page})

    http.respond = respond
    ingest.ingest(pipeline_config(tmp_path, [cdo_source()]))

    assert [call["params"]["offset"] for call in http.calls] == [1, 3, 5]
    assert all(call["headers"] == {"token": TOKEN} for call in http.calls)
    assert http.calls[0]["params"]["datatypeid"] == ["TAVG", "PRCP"]
    snapshot = pd.read_csv(tmp_path / "raw" / "climate.csv")
    assert snapshot["value"].tolist() == [15, 16, 17, 18, 19]


def test_noaa_cdo_with_no_matches_is_an_error(tmp_path: Path, http: SimpleNamespace) -> None:
    http.respond = lambda url, params: FakeResponse({})  # CDO's answer when nothing matches

    with pytest.raises(ValueError, match="returned no data"):
        ingest.ingest(pipeline_config(tmp_path, [cdo_source()]))


def test_noaa_cdo_needs_a_url(tmp_path: Path, http: SimpleNamespace) -> None:
    source = {**cdo_source(), "url": None}

    with pytest.raises(ValueError, match="needs a url"):
        ingest.ingest(pipeline_config(tmp_path, [source]))


GRANULE = {
    "title": "greenland_mass_200204_202609",
    "links": [
        {"rel": "http://esipfed.org/ns/fedsearch/1.1/data#", "href": FILE_URL},
        {"rel": "http://esipfed.org/ns/fedsearch/1.1/s3#", "href": "s3://bucket/file.txt"},
        {
            "rel": "http://esipfed.org/ns/fedsearch/1.1/data#",
            "href": "https://search.example.org/granules",
            "inherited": True,
        },
    ],
}
GRACE_TEXT = """HDR Greenland Mass Trend (2004/01-2023/12): -265.32 +/-23.52 Gt/yr
HDR
HDR   Column 1 = Time (decimal year)
HDR Header_End---------------------------------------
  2002.29        0.00      164.18
  2002.35      -28.64      101.51

  2002.62     -150.27       88.02
"""


def grace_source(**options: Any) -> dict[str, Any]:
    return {
        "name": "ice_mass",
        "type": "earthdata_timeseries",
        "url": CMR_URL,
        "params": {"short_name": "GREENLAND_MASS", "sort_key": "-start_date", "page_size": 1},
        "headers": {"Authorization": "Bearer ${TEST_TOKEN}"},
        "options": {"columns": ["decimal_year", "mass_gt", "uncertainty_gt"], **options},
    }


def grace_responses(file: FakeResponse) -> Any:
    def respond(url: str, params: dict[str, Any]) -> FakeResponse:
        return FakeResponse({"feed": {"entry": [GRANULE]}}) if url == CMR_URL else file

    return respond


def test_earthdata_downloads_the_newest_file_with_the_token(
    tmp_path: Path, http: SimpleNamespace
) -> None:
    http.respond = grace_responses(FakeResponse(text=GRACE_TEXT, content_type="text/plain"))

    ingest.ingest(pipeline_config(tmp_path, [grace_source()]))

    catalog, download = http.calls
    assert catalog["params"]["short_name"] == "GREENLAND_MASS"
    assert not catalog["headers"]  # the public catalog never gets the token
    assert download["url"] == FILE_URL  # the granule's own link, not an inherited one
    assert download["headers"] == {"Authorization": f"Bearer {TOKEN}"}
    snapshot = pd.read_csv(tmp_path / "raw" / "ice_mass.csv")
    assert snapshot.columns.tolist() == ["decimal_year", "mass_gt", "uncertainty_gt", "granule"]
    assert snapshot["mass_gt"].tolist() == [0.0, -28.64, -150.27]
    assert set(snapshot["granule"]) == {"greenland_mass_200204_202609"}


def test_earthdata_sign_in_page_means_a_bad_token(tmp_path: Path, http: SimpleNamespace) -> None:
    http.respond = grace_responses(
        FakeResponse(text="<html>Sign in</html>", content_type="text/html")
    )

    with pytest.raises(PermissionError, match="EARTHDATA_TOKEN"):
        ingest.ingest(pipeline_config(tmp_path, [grace_source()]))


def test_earthdata_column_count_must_match(tmp_path: Path, http: SimpleNamespace) -> None:
    http.respond = grace_responses(FakeResponse(text=GRACE_TEXT, content_type="text/plain"))
    source = grace_source(columns=["decimal_year", "mass_gt"])

    with pytest.raises(ValueError, match="has 3 columns"):
        ingest.ingest(pipeline_config(tmp_path, [source]))


def test_earthdata_with_no_files_is_an_error(tmp_path: Path, http: SimpleNamespace) -> None:
    http.respond = lambda url, params: FakeResponse({"feed": {"entry": []}})

    with pytest.raises(ValueError, match="found no files"):
        ingest.ingest(pipeline_config(tmp_path, [grace_source()]))


def test_config_sources_use_registered_types_and_documented_keys() -> None:
    cfg = load_config(REPO_CONFIG)
    documented = set(
        re.findall(r"^#?\s*([A-Z][A-Z0-9_]*)=", (REPO / ".env.example").read_text(), re.M)
    )

    for source in cfg.ingest.sources:
        assert source.type in ingest.FETCHERS, source.name
        undocumented = set(ingest.env_names(source)) - documented
        assert not undocumented, (
            f"{source.name} uses {undocumented}, which .env.example doesn't list"
        )
