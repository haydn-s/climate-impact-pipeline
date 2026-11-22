"""Data pipeline, stage 2 of 3: clean the raw snapshots, join them, and add engineered features.

    python -m data_pipeline.prepare --config config.yaml

This command runs two steps, each in its own module:

- data_pipeline/clean.py     clean each source's raw snapshot, then join the sources
- data_pipeline/features.py  run the feature steps named in `prepare.feature_steps`

The result is written to `data.local_path`, ready for stage 3 (data_pipeline/push.py).
"""

from __future__ import annotations

import logging

import pandas as pd

from data_pipeline.clean import clean_sources, join_sources
from data_pipeline.features import FEATURE_STEPS, add_features
from shared.cli import config_parser, load
from shared.config import PipelineConfig
from shared.tables import missing_columns, write_table

log = logging.getLogger(__name__)


def prepare(cfg: PipelineConfig) -> pd.DataFrame:
    """Clean, join, and add features, then write the processed table to data.local_path."""
    if not cfg.ingest.sources:
        raise ValueError("no sources to prepare. Add them under ingest.sources in the config")
    unknown = [name for name in cfg.prepare.feature_steps if name not in FEATURE_STEPS]
    if unknown:
        raise ValueError(
            f"feature steps {unknown} aren't registered in data_pipeline/features.py. "
            f"Registered: {sorted(FEATURE_STEPS) or 'none yet'}"
        )

    frames = clean_sources(cfg)
    table = add_features(cfg, join_sources(cfg, frames), frames).reset_index(drop=True)

    missing = missing_columns(table, cfg)
    if missing:
        log.warning("The processed data lacks columns the config expects: %s", missing)
    write_table(table, cfg.data.local_path)
    log.info("Wrote %d rows x %d columns to %s", len(table), table.shape[1], cfg.data.local_path)
    return table


def main(argv: list[str] | None = None) -> pd.DataFrame:
    parser = config_parser("Clean, join, and add features to the raw data.")
    return prepare(load(parser.parse_args(argv)))


if __name__ == "__main__":
    main()
