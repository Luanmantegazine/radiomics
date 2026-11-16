"""Contracts of the client's scientific modules (Stage 2 implemented; 3-5 provisional).

Each provisional interface is
reviewed and may change when its stage starts. All of them operate on local
data and return local objects; turning their outputs into wire payloads is the
job of a :class:`client.tasks.LocalTask`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np

from client.data import LocalDataset


class ImagePairValidator(Protocol):
    """Stage 2 - implemented by :func:`client.imaging.validation.validate_pair`."""

    def __call__(self, image_path: Path, mask_path: Path, label: int, settings: Any) -> Any: ...


class FeatureExtractor(Protocol):
    """Stage 2 - radiomic extraction with an explicit, versioned configuration.

    Implemented by :class:`client.imaging.extraction.PyRadiomicsExtractor`.
    """

    config_id: str

    def extract(self, image: Path, mask: Path, label: int = 1) -> Any:
        """Return an object with ``features`` (name -> value) and ``diagnostics``."""
        ...

    def provenance(self) -> dict[str, Any]:
        """Library version, parameters, preprocessing, discretisation, ROI."""
        ...


class ReliabilityEstimator(Protocol):
    """Stage 3 - e.g. ICC. Requires repeated or alternative observations;
    implementations must refuse to estimate reliability without them."""

    def estimate(self, repeated_features: np.ndarray) -> np.ndarray:
        """``repeated_features`` has shape (n_subjects, n_repeats, n_features)."""
        ...


class RelevanceScorer(Protocol):
    """Stage 3 - prognostic relevance per feature, censoring-aware."""

    def score(self, dataset: LocalDataset) -> np.ndarray: ...


class RedundancyEstimator(Protocol):
    """Stage 3 - feature-by-feature redundancy (shape (p, p))."""

    def estimate(self, dataset: LocalDataset) -> np.ndarray: ...


class LocalTrainer(Protocol):
    """Stage 5 - local step of a survival model."""

    def fit(self, dataset: LocalDataset, global_state: dict[str, Any]) -> dict[str, Any]: ...
