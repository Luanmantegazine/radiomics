"""Clinical data layer for the OASIS-3 Alzheimer radiomics study.

This package links OASIS-3 clinical assessments to MRI sessions so that
radiomic features can later be analysed against cognitive status and
progression. It performs **no** machine learning and does not alter the frozen
radiomics protocol in :mod:`oasis_radiomics.protocol`.

Layers
------
``models``          typed records for MRI sessions, clinical visits and matches
``readers``         normalisation of the raw OASIS CSVs (never modified in place)
``matching``        D1+B4 merge, and nearest-in-time MRI <-> clinical linkage
``classification``  codebook-driven derivation of cognitive status / AD etiology
``trajectories``    per-subject diagnosis history, conversions, leakage guards
``validation``      auditable issue collection and the JSON linkage report

Scientific guard rails
----------------------
* The numeric semantics of the NACC/UDS D1 variables are **not** documented in
  this repository. Nothing here invents them: until a codebook is frozen in
  ``clinical_classification.yaml`` the classifier reports
  ``unresolved_codebook`` and every raw variable is passed through untouched.
* A diagnosis recorded *after* an MRI never relabels that MRI. See
  :mod:`oasis_radiomics.clinical.trajectories`.
"""

from __future__ import annotations

#: Bumped whenever the linkage rules change in a way that alters outputs.
CLINICAL_LINKAGE_VERSION = "oasis3-clinical-linkage-v1.0"

#: Default half-width, in days, of the MRI <-> clinical matching window.
DEFAULT_CLINICAL_WINDOW_DAYS = 180

#: Matching strategy identifier recorded in every run's metadata.
MATCHING_STRATEGY = "nearest-absolute-gap;ties-prefer-earlier"

__all__ = [
    "CLINICAL_LINKAGE_VERSION",
    "DEFAULT_CLINICAL_WINDOW_DAYS",
    "MATCHING_STRATEGY",
]
