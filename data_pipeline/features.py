"""Data pipeline, stage 2b: add engineered features and the label to the joined table.

`data_pipeline/prepare.py` runs the steps named in `prepare.feature_steps`, in
order, after cleaning and joining (`clean.py`). Register each one below with
`@feature_step("<name>")`.

Where features belong: the API gets one row of inputs from the front-end, so a
column computed here has to be something the front-end can supply or look up
too. Transforms of a row's own inputs (ratios, logs, date parts) are better
left to the ML pipeline's saved model, which the API applies for you, along
with model-time transforms (imputing, scaling, encoding).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd

from data_pipeline.clean import checked_frame
from shared.config import PipelineConfig

log = logging.getLogger(__name__)

FeatureStep = Callable[..., pd.DataFrame]  # (table, sources, **options) -> table
FEATURE_STEPS: dict[str, FeatureStep] = {}


def feature_step(name: str) -> Callable[[FeatureStep], FeatureStep]:
    """Register `fn(table, sources, **options) -> table` as a feature step.

    `table` is the joined table. `sources` maps each source name to its cleaned
    frame, including sources left out of the join. `options` come from
    `prepare.step_options.<name>` in the config, so a step's settings live there.
    """

    def register(fn: FeatureStep) -> FeatureStep:
        FEATURE_STEPS[name] = fn
        return fn

    return register


def add_features(
    cfg: PipelineConfig, table: pd.DataFrame, frames: Mapping[str, pd.DataFrame]
) -> pd.DataFrame:
    """Run the feature steps named in `prepare.feature_steps`, in order."""
    for name in cfg.prepare.feature_steps:
        before = set(table.columns)
        options = cfg.prepare.step_options.get(name, {})
        table = checked_frame(FEATURE_STEPS[name](table, frames, **options), f"feature step {name}")
        log.info(
            "Feature step %s added %s", name, sorted(set(table.columns) - before) or "no columns"
        )
    return table


# Dataset-specific feature steps go below.
#
# The weather dataset (config.yaml): one row per North Carolina airport per
# day, with today's and yesterday's temperatures as features, and the label
# `tomorrow_category`, whether tomorrow's high is expected, an extreme high,
# or an extreme low for that station and time of year.

EXPECTED, EXTREME_HIGH, EXTREME_LOW = "expected", "extreme high", "extreme low"
WEATHER_COLUMNS = [
    "date",
    "station_id",
    "station",
    "latitude",
    "longitude",
    "elevation_m",
    "month",
    "today_high_f",
    "today_low_f",
    "yesterday_high_f",
    "tomorrow_high_f",
    "today_category",
    "tomorrow_category",
]


@feature_step("daily_lags")
def daily_lags(table: pd.DataFrame, sources: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Add yesterday's and tomorrow's highs, matched by calendar date rather than row order.

    So a day a station didn't report leaves a gap (missing) instead of pairing
    the wrong days. Renames high_f and low_f to today_high_f and today_low_f.
    """
    highs = table[["station_id", "date", "high_f"]]
    one_day = pd.Timedelta(days=1)
    yesterday = highs.assign(date=highs["date"] + one_day).rename(
        columns={"high_f": "yesterday_high_f"}
    )
    tomorrow = highs.assign(date=highs["date"] - one_day).rename(
        columns={"high_f": "tomorrow_high_f"}
    )
    return (
        table.merge(yesterday, on=["station_id", "date"], how="left")
        .merge(tomorrow, on=["station_id", "date"], how="left")
        .rename(columns={"high_f": "today_high_f", "low_f": "today_low_f"})
    )


@feature_step("tomorrow_label")
def tomorrow_label(
    table: pd.DataFrame,
    sources: Mapping[str, pd.DataFrame],
    *,
    baseline_years: tuple[int, int] = (1991, 2020),
    window_days: int = 7,
    percentiles: tuple[float, float] = (10, 90),
) -> pd.DataFrame:
    """Label today's and tomorrow's highs against the station's normal range for the date.

    The normal range for a station and calendar day runs from the low to the
    high `percentiles` of its daily highs in `baseline_years`, using the days
    within `window_days` of that date in each year. Above it is "extreme
    high", below it "extreme low", and in between "expected". A fixed baseline
    keeps later (test) years from shaping their own labels.

    Adds `month`, `today_category` (a persistence baseline, not a feature), and
    the target `tomorrow_category`. Drops rows missing a feature or the label.
    """
    first, last = baseline_years
    low_pct, high_pct = percentiles
    today = _calendar_day(table["date"])
    in_baseline = table["date"].dt.year.between(first, last) & table["today_high_f"].notna()
    ranges = _normal_ranges(
        table.loc[in_baseline, "station_id"],
        today[in_baseline],
        table.loc[in_baseline, "today_high_f"],
        window_days=window_days,
        percentiles=(low_pct, high_pct),
    )
    tomorrow = _calendar_day(table["date"] + pd.Timedelta(days=1))
    table = table.assign(
        month=table["date"].dt.month_name(),
        today_category=_categorize(table["today_high_f"], table["station_id"], today, ranges),
        tomorrow_category=_categorize(
            table["tomorrow_high_f"], table["station_id"], tomorrow, ranges
        ),
    )

    required = ["today_high_f", "today_low_f", "yesterday_high_f", "tomorrow_category"]
    complete = table[required].notna().all(axis=1)
    if not complete.all():
        log.info("weather: dropping %d rows missing a feature or the label", (~complete).sum())
    table = table.loc[complete].reset_index(drop=True)
    shares = table["tomorrow_category"].value_counts(normalize=True).round(3).to_dict()
    log.info("weather: tomorrow_category shares %s", shares)
    return table[[c for c in WEATHER_COLUMNS if c in table.columns]]


def _calendar_day(dates: pd.Series) -> pd.Series:
    """Day of the year, 1 to 365, counting Feb 29 as Feb 28 so leap years line up."""
    leap = dates.dt.is_leap_year
    after_feb_28 = leap & ((dates.dt.month > 2) | ((dates.dt.month == 2) & (dates.dt.day == 29)))
    return dates.dt.day_of_year - after_feb_28.astype(int)


def _normal_ranges(
    stations: pd.Series,
    days: pd.Series,
    highs: pd.Series,
    *,
    window_days: int,
    percentiles: tuple[float, float],
) -> pd.DataFrame:
    """The (low, high) percentiles of each station's highs around each calendar day."""
    rows = []
    baseline = pd.DataFrame({"station_id": stations, "day": days, "high": highs})
    for station_id, group in baseline.groupby("station_id"):
        group_days, group_highs = group["day"].to_numpy(), group["high"].to_numpy()
        for day in range(1, 366):
            gap = np.abs(group_days - day)
            gap = np.minimum(gap, 365 - gap)  # Dec 31 is next to Jan 1
            window = group_highs[gap <= window_days]
            low, high = np.percentile(window, percentiles) if window.size else (np.nan, np.nan)
            rows.append((station_id, day, low, high))
    if not rows:
        raise ValueError("no rows in the baseline years to measure the normal range from")
    return pd.DataFrame(rows, columns=["station_id", "day", "low", "high"]).set_index(
        ["station_id", "day"]
    )


def _categorize(
    highs: pd.Series, stations: pd.Series, days: pd.Series, ranges: pd.DataFrame
) -> pd.Series:
    """Each high as expected / extreme high / extreme low, or missing if either side is."""
    bounds = ranges.reindex(pd.MultiIndex.from_arrays([stations, days]))
    low, high = bounds["low"].to_numpy(), bounds["high"].to_numpy()
    values = highs.to_numpy(dtype=float)
    labels = np.where(values > high, EXTREME_HIGH, np.where(values < low, EXTREME_LOW, EXPECTED))
    known = ~(np.isnan(values) | np.isnan(low))
    return pd.Series(labels, index=highs.index, dtype=object).where(known)
