"""Local data source backed by an extracted feature table (Stage 2 output).

Bridges the radiomic pipeline to the federation: the table is loaded into a
:class:`client.data.LocalDataset`, which still leaves the site only through
authorized task payloads.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from client.data import LocalDataset
from client.imaging.manifest import load_manifest
from client.imaging.pipeline import PROVENANCE_FILE, sha256_file
from shared.logging import get_logger
from shared.protocol import DataOrigin


class FeatureTableError(RuntimeError):
    pass


class FeatureTableDataSource:
    """Reads ``features.csv`` and joins the follow-up from the site manifest.

    Participants without usable follow-up are **excluded** (and counted in the
    local log): a missing follow-up is unknown, never "did not convert".
    When a ``provenance.json`` sits next to the table, the table's hash is
    verified against it so that a modified table is not used silently.
    """

    def __init__(
        self,
        features_path: str | Path,
        manifest_path: str | Path,
        origin: DataOrigin,
        reliability_path: str | Path | None = None,
    ) -> None:
        self._features_path = Path(features_path)
        self._manifest_path = Path(manifest_path)
        self._reliability_path = Path(reliability_path) if reliability_path else None
        self.origin = origin

    def load(self) -> LocalDataset:
        provenance_path = self._features_path.parent / PROVENANCE_FILE
        if provenance_path.exists():
            expected = json.loads(provenance_path.read_text(encoding="utf-8"))["features_sha256"]
            if sha256_file(self._features_path) != expected:
                raise FeatureTableError("features table does not match its provenance record")
        table = pd.read_csv(self._features_path, dtype={"participant_id": str})
        feature_names = tuple(c for c in table.columns if c != "participant_id")
        if not feature_names or table["participant_id"].duplicated().any():
            raise FeatureTableError("features table is empty or has duplicate participants")
        followup = load_manifest(self._manifest_path).followup
        keep = table["participant_id"].isin(followup.keys())
        get_logger("client.feature_table").info(
            "feature_table_loaded",
            participants=int(len(table)),
            excluded_without_followup=int((~keep).sum()),
            n_features=len(feature_names),
        )
        table = table[keep].sort_values("participant_id")
        ids = tuple(table["participant_id"])
        return LocalDataset(
            participant_ids=ids,
            feature_names=feature_names,
            features=table[list(feature_names)].to_numpy(dtype=float),
            time_months=np.array([followup[p].time_months for p in ids], dtype=float),
            event=np.array([followup[p].event for p in ids], dtype=bool),
            origin=self.origin,
            reliability=self._load_reliability(),
        )

    def _load_reliability(self):
        if self._reliability_path is None:
            return None
        from client.reliability.study import load_reliability_estimates

        return load_reliability_estimates(self._reliability_path)
