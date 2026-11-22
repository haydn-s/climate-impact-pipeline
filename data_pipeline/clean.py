"""Data pipeline, stage 2a: clean each raw snapshot, then join the sources into one table.

`data_pipeline/prepare.py` runs this, then the feature steps in `features.py`.

1. Clean. Every source gets `standard_clean`: snake_case column names, trimmed
   text, blank strings as missing, and no empty rows, empty columns, or
   duplicate rows. Then the source's own cleaner runs, if one is registered
   below with `@cleaner("<source name>")`. Cleaners see the snake_case names.
2. Join. The first source in `ingest.sources` is the base table. Each later
   source with `join_on` is merged onto it by those columns (a left join, unless
   it sets `how: inner`). A source without `join_on` stays out of the join, but
   feature steps can still use it, e.g. to compute a distance to the nearest fault.

Snapshots are CSV, so codes with leading zeros (state FIPS "01", ZIP codes)
come back as numbers. Restore them in the source's cleaner, e.g.
`frame["fips"] = frame["fips"].astype(str).str.zfill(2)`.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Mapping

import pandas as pd

from data_pipeline.ingest import snapshot_file
from shared.config import PipelineConfig, SourceConfig
from shared.storage import download, join_uri

log = logging.getLogger(__name__)

Cleaner = Callable[[pd.DataFrame], pd.DataFrame]
CLEANERS: dict[str, Cleaner] = {}


def cleaner(source_name: str) -> Callable[[Cleaner], Cleaner]:
    """Register `fn(frame) -> frame` as the cleaner for one source. It runs after standard_clean."""

    def register(fn: Cleaner) -> Cleaner:
        CLEANERS[source_name] = fn
        return fn

    return register


def checked_frame(result: object, what: str) -> pd.DataFrame:
    """Return `result` if a cleaner or feature step gave back a DataFrame, else say what went wrong."""
    if not isinstance(result, pd.DataFrame):
        raise TypeError(
            f"{what} returned {type(result).__name__}, not a DataFrame. Is `return` missing?"
        )
    return result


# Cleaning


def snake_case(name: object) -> str:
    """`AverageTemperature` -> `average_temperature`, `CV (%)` -> `cv`, `PM2.5` -> `pm2_5`."""
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(name).strip())
    return re.sub(r"[^0-9a-zA-Z]+", "_", text).strip("_").lower()


def standard_clean(frame: pd.DataFrame, *, source: str = "data") -> pd.DataFrame:
    """The cleaning every source gets before its own cleaner."""
    frame = frame.copy()
    frame.columns = _unique_names(
        [snake_case(c) or f"column_{i}" for i, c in enumerate(frame.columns)], source
    )

    for column in frame.columns:
        values = frame[column]
        if pd.api.types.is_string_dtype(values.dtype):
            trimmed = values.map(lambda v: v.strip() if isinstance(v, str) else v)
            frame[column] = trimmed.mask(trimmed == "")

    empty_columns = [c for c in frame.columns if frame[c].isna().all()]
    if empty_columns:
        log.info("%s: dropping columns with no values: %s", source, empty_columns)
    frame = frame.drop(columns=empty_columns).dropna(how="all")

    deduplicated = frame.drop_duplicates().reset_index(drop=True)
    if len(deduplicated) < len(frame):
        log.info("%s: dropped %d duplicate rows", source, len(frame) - len(deduplicated))
    return deduplicated


def _unique_names(names: list[str], source: str) -> list[str]:
    seen: dict[str, int] = {}
    unique = []
    for name in names:
        count = seen.get(name, 0)
        seen[name] = count + 1
        unique.append(name if count == 0 else f"{name}_{count + 1}")
    renamed = sorted({name for name, count in seen.items() if count > 1})
    if renamed:
        log.warning(
            "%s: several columns became %s in snake_case, so they got _2, _3, ...", source, renamed
        )
    return unique


def load_snapshot(cfg: PipelineConfig, source: SourceConfig) -> pd.DataFrame:
    """Read a source's raw snapshot, downloading it from ingest.remote_prefix if it isn't local."""
    path = cfg.ingest.raw_dir / snapshot_file(source.name)
    if not path.exists():
        if not cfg.ingest.remote_prefix:
            raise FileNotFoundError(
                f"no snapshot of source '{source.name}' at {path}. Run: make ingest"
            )
        download(join_uri(cfg.ingest.remote_prefix, snapshot_file(source.name)), path)
    try:
        # low_memory=False reads each column's type from the whole file, not chunk by chunk.
        return pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError as err:
        raise ValueError(f"the snapshot of source '{source.name}' at {path} is empty") from err


def clean_sources(cfg: PipelineConfig) -> dict[str, pd.DataFrame]:
    """Load and clean every source's snapshot, keyed by source name."""
    frames = {}
    for source in cfg.ingest.sources:
        frame = standard_clean(load_snapshot(cfg, source), source=source.name)
        custom = CLEANERS.get(source.name)
        if custom is None:
            log.info(
                "%s has no cleaner registered, so it gets only the standard clean", source.name
            )
        else:
            frame = checked_frame(custom(frame), f"the cleaner for {source.name}")
        log.info("%s: %d rows x %d columns after cleaning", source.name, len(frame), frame.shape[1])
        frames[source.name] = frame
    return frames


# Joining


def join_sources(cfg: PipelineConfig, frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge each source with `join_on` onto the first source, in config order."""
    base, *others = cfg.ingest.sources
    table = frames[base.name]
    for source in others:
        if source.join_on is not None:
            table = join_one(table, frames[source.name], source)
    return table


def join_one(left: pd.DataFrame, right: pd.DataFrame, source: SourceConfig) -> pd.DataFrame:
    """Merge one source onto the table so far, checking the things that silently break joins."""
    keys = list(source.join_on or [])
    for side, frame in (("the table so far", left), (f"source '{source.name}'", right)):
        missing = [key for key in keys if key not in frame.columns]
        if missing:
            raise ValueError(f"can't join '{source.name}': {side} has no column {missing}")
    for key in keys:
        if pd.api.types.is_numeric_dtype(left[key]) != pd.api.types.is_numeric_dtype(right[key]):
            raise ValueError(
                f"can't join '{source.name}' on '{key}': it's {left[key].dtype} in the table so "
                f"far but {right[key].dtype} in the source. Convert one of them in a cleaner"
            )
    # pandas matches missing keys to each other, so a blank key would pick up an
    # unrelated row. Those rows can't really match anything, so leave them out.
    blank_keys = right[keys].isna().any(axis=1)
    if blank_keys.any():
        log.warning(
            "%s: leaving out %d rows with a missing %s", source.name, int(blank_keys.sum()), keys
        )
        right = right[~blank_keys]
    repeated = right.duplicated(keys)
    if repeated.any():
        raise ValueError(
            f"source '{source.name}' has {int(repeated.sum())} rows that repeat a {keys} "
            "combination, so joining it would duplicate rows. Reduce it to one row per key "
            "(e.g. aggregate) in its cleaner"
        )
    overlap = sorted((set(left.columns) & set(right.columns)) - set(keys))
    if overlap:
        log.warning(
            "%s also has columns %s, so its copies get the suffix _%s",
            source.name,
            overlap,
            source.name,
        )

    merged = left.merge(
        right,
        on=keys,
        how=source.how,
        suffixes=("", f"_{source.name}"),
        indicator="_matched",
    )
    unmatched = int((merged["_matched"] == "left_only").sum())
    if unmatched:
        log.warning(
            "%d of %d rows found no match in %s on %s. Check that the keys are formatted the same way",
            unmatched,
            len(merged),
            source.name,
            keys,
        )
    if len(merged) < len(left):
        log.info("Inner join with %s dropped %d rows", source.name, len(left) - len(merged))
    return merged.drop(columns="_matched")


# Dataset-specific cleaners go below.
#
# The weather dataset (config.yaml): NOAA's daily readings at five North
# Carolina airports, joined with each station's name and location.


@cleaner("weather")
def clean_weather(frame: pd.DataFrame) -> pd.DataFrame:
    """NOAA's one row per reading, as one row per station and day with its high and low.

    Readings come as (date, datatype TMAX or TMIN, station, attributes, value).
    `attributes` holds NOAA's flags, "measurement,quality,source,time". A
    quality flag means the reading failed one of NOAA's checks, so it's dropped.
    """
    flags = frame["attributes"].fillna("").astype(str).str.split(",")
    failed = flags.str[1].fillna("").str.strip() != ""
    if failed.any():
        log.info("weather: dropping %d readings that failed NOAA's quality checks", failed.sum())
    readings = frame.loc[~failed]
    daily = (
        readings.pivot_table(
            index=["station", "date"], columns="datatype", values="value", aggfunc="first"
        )
        .rename(columns={"TMAX": "high_f", "TMIN": "low_f"})
        .reset_index()
        .rename(columns={"station": "station_id"})
    )
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    return daily[["station_id", "date", "high_f", "low_f"]]


@cleaner("stations")
def clean_stations(frame: pd.DataFrame) -> pd.DataFrame:
    """Each station's id, a short name for the form, and where it is.

    "RALEIGH DURHAM INTERNATIONAL AIRPORT, NC US" becomes "Raleigh Durham".
    """
    names = (
        frame["name"]
        .str.replace(r",\s*[A-Z]{2}\s+US$", "", regex=True)
        .str.replace(r"\s+(?:INTERNATIONAL\s+|REGIONAL\s+)?AIRPORT$", "", regex=True)
        .str.title()
    )
    return pd.DataFrame(
        {
            "station_id": frame["id"],
            "station": names,
            "latitude": frame["latitude"],
            "longitude": frame["longitude"],
            "elevation_m": frame["elevation"],
        }
    )
