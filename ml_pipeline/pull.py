"""ML pipeline, stage 1 of 3: fetch the data from the config and save as X and y tables.

[insert documentation]
"""


from shared.config import load_config
from shared.storage import download
from shared.tables import missing_columns, read_table

cfg = load_config("config.yaml")
if cfg.data.remote_uri is None:
    raise ValueError("Set data.remote_uri in config.yaml before training")

download(cfg.data.remote_uri, cfg.data.local_path)
table = read_table(cfg.data.local_path)

missing = missing_columns(table, cfg)
if missing:
    raise ValueError(f"Processed table is missing required columns: {missing}")

X = table[cfg.features.all]
y = table[cfg.data.target]