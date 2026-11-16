"""CLI: run a reliability study locally.

    python -m client.reliability --manifest site/manifest.csv \
        --extraction-config experiments/configs/extraction/t1_original_v1.yaml \
        --study-config experiments/configs/reliability/perturbation_v1.yaml \
        --output-dir var/sites/client-01/reliability
"""

from __future__ import annotations

import argparse
import json
import sys

from client.imaging.extraction import load_extraction_config
from client.reliability.icc import ReliabilityUnavailable
from client.reliability.study import load_study_config, run_reliability_study
from shared.logging import configure_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m client.reliability")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--extraction-config", required=True)
    parser.add_argument("--study-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    study = load_study_config(args.study_config)
    try:
        summary = run_reliability_study(
            args.manifest, load_extraction_config(args.extraction_config), study, args.output_dir
        )
    except ReliabilityUnavailable as exc:
        print(f"reliability not estimated: {exc}", file=sys.stderr)
        return 2
    values = summary["min_icc"]
    json.dump(
        {
            "study_id": study.study_id,
            "icc_form": study.icc_form.value,
            "n_features": int(len(summary)),
            "n_undefined": int(values.isna().sum()),
            "min_icc_quartiles": [round(float(q), 4) for q in values.quantile([0.25, 0.5, 0.75])],
            "output_dir": args.output_dir,
        },
        sys.stdout,
        indent=2,
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
