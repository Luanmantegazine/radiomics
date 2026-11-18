"""CLI: train prognostic models on one site's data (local models).

    python -m client.models --features var/sites/client-01/features.csv \
        --manifest site/manifest.csv --data-origin real \
        --config experiments/configs/models/local_v1.yaml \
        --feature-names hippo_l__original_shape_VoxelVolume,... \
        --output-dir var/sites/client-01/models

Features come from ``--feature-names`` or from a local selection result
(``--selection-csv selected_features.csv --selector <name>``). Nothing is sent.
"""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from client.feature_table import FeatureTableDataSource
from client.models.local import load_local_config, train_locally
from shared.logging import configure_logging
from shared.protocol import DataOrigin


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m client.models")
    parser.add_argument("--features", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-origin", required=True, choices=[o.value for o in DataOrigin])
    parser.add_argument("--config", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--feature-names")
    group.add_argument("--selection-csv")
    parser.add_argument("--selector")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    if args.selection_csv:
        if not args.selector:
            parser.error("--selector is required with --selection-csv")
        table = pd.read_csv(args.selection_csv)
        rows = table[table["selector"] == args.selector].sort_values("rank")
        names = tuple(rows["feature"])
    else:
        names = tuple(n.strip() for n in args.feature_names.split(",") if n.strip())
    if not names:
        print("no features to train on", file=sys.stderr)
        return 2
    dataset = FeatureTableDataSource(
        args.features, args.manifest, DataOrigin(args.data_origin)
    ).load()
    report = train_locally(dataset, names, load_local_config(args.config), args.output_dir)
    json.dump({k: report[k] for k in ("feature_names", "n_train", "n_events_train", "models")},
              sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0 if all(v == "trained" for v in report["models"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
