"""CLI: run a local selection study on an extracted feature table.

    python -m client.selection --features var/sites/client-01/features.csv \
        --manifest site/manifest.csv --data-origin real \
        --config experiments/configs/selection/local_v1.yaml \
        --reliability var/sites/client-01/reliability/reliability_summary.csv \
        --output-dir var/sites/client-01/selection
"""

from __future__ import annotations

import argparse
import json
import sys

from client.feature_table import FeatureTableDataSource
from client.reliability.study import load_reliability
from client.selection.config import load_selection_config
from client.selection.selectors import SelectionError
from client.selection.study import run_selection_study, write_study
from shared.logging import configure_logging
from shared.protocol import DataOrigin


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m client.selection")
    parser.add_argument("--features", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-origin", required=True, choices=[o.value for o in DataOrigin])
    parser.add_argument("--config", required=True)
    parser.add_argument("--reliability", help="reliability_summary.csv from a reliability study")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    config = load_selection_config(args.config)
    reliability = None
    if args.reliability:
        reliability = {
            column: load_reliability(args.reliability, column)
            for column in config.needs_reliability
        }
    dataset = FeatureTableDataSource(
        args.features, args.manifest, DataOrigin(args.data_origin)
    ).load()
    try:
        result = run_selection_study(dataset, config, reliability)
    except SelectionError as exc:
        print(f"selection not run: {exc}", file=sys.stderr)
        return 2
    write_study(result, config, args.output_dir)
    json.dump(
        {
            "study_id": config.study_id,
            "data_origin": args.data_origin,
            "n_participants": result.n_participants,
            "n_events": result.n_events,
            "selected": {name: len(s.selected) for name, s in result.selections.items()},
            "nogueira_stability": {
                name: None if b.stability != b.stability else round(b.stability, 4)
                for name, b in result.bootstrap.items()
            },
            "output_dir": args.output_dir,
        },
        sys.stdout,
        indent=2,
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
