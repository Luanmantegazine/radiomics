"""CLI of the local radiomic pipeline.

    python -m client.imaging check   --manifest site/manifest.csv --config <extraction.yaml>
    python -m client.imaging extract --manifest site/manifest.csv --config <extraction.yaml> \
                                     --output-dir var/sites/client-01

Runs entirely inside the site; nothing is transmitted.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict

from client.imaging.extraction import load_extraction_config
from client.imaging.manifest import ManifestError, load_manifest
from client.imaging.pipeline import run_extraction
from client.imaging.validation import validate_pair
from shared.logging import configure_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m client.imaging")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "extract"):
        sub = commands.add_parser(name)
        sub.add_argument("--manifest", required=True)
        sub.add_argument("--config", required=True)
        sub.add_argument("--log-level", default="INFO")
        if name == "extract":
            sub.add_argument("--output-dir", required=True)
            sub.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    try:
        manifest = load_manifest(args.manifest)
    except ManifestError as exc:
        print(f"manifest error: {exc}", file=sys.stderr)
        return 2
    config = load_extraction_config(args.config)

    if args.command == "check":
        issues: Counter[str] = Counter()
        n_ok = 0
        for case in manifest.cases:
            report = validate_pair(
                case.image_path, case.mask_path, case.mask_label, config.validation
            )
            n_ok += report.ok
            issues.update(f"{i.severity.value}:{i.code.value}" for i in report.issues)
        summary = {
            "cases": len(manifest.cases),
            "cases_ok": n_ok,
            "issues": dict(sorted(issues.items())),
            "followup": asdict(manifest.followup_summary()),
        }
        json.dump(summary, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0 if n_ok == len(manifest.cases) else 1

    result = run_extraction(manifest, config, args.output_dir, overwrite=args.overwrite)
    summary = {
        "output_dir": str(result.output_dir),
        "config_id": config.config_id,
        "config_hash": config.content_hash,
        "cases": result.n_cases,
        "cases_extracted": result.n_extracted,
        "participants": result.n_participants,
        "participants_complete": result.n_complete_participants,
        "n_features": len(result.feature_names),
        "features_sha256": result.features_sha256,
        "duration_s": round(result.duration_s, 3),
    }
    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0 if result.n_extracted == result.n_cases else 1


if __name__ == "__main__":
    raise SystemExit(main())
