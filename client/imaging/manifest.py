"""Site manifest: which image, mask and follow-up belong to which participant.

One CSV row per (participant, ROI):

    participant_id,roi,image_path,mask_path[,mask_label][,time_months,event]

* ``image_path`` / ``mask_path``: NIfTI files; relative paths are resolved
  against the manifest's directory.
* ``mask_label``: integer label of the ROI inside the mask volume (default 1),
  so one multi-label segmentation can serve several ROIs.
* ``time_months`` / ``event``: follow-up time and conversion indicator
  (1 = converted, 0 = right-censored at ``time_months``). Both blank means the
  participant has no usable longitudinal follow-up.

Hypotheses and limitations
--------------------------
* One image per participant (the baseline scan). Longitudinal imaging would
  need a session key and is out of scope.
* A blank follow-up is "unknown", never "did not convert": such participants
  are reported and later excluded from prognostic datasets.
* The manifest is trusted for the participant/image/mask association; it
  cannot be verified from the image files themselves.
"""

from __future__ import annotations

import csv
import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path

REQUIRED_COLUMNS = ("participant_id", "roi", "image_path", "mask_path")
OPTIONAL_COLUMNS = ("mask_label", "time_months", "event")
ROI_PATTERN = re.compile(r"^[a-z0-9_]{1,48}$")
PARTICIPANT_PATTERN = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")
_TRUE, _FALSE = {"1", "true", "yes"}, {"0", "false", "no"}


class ManifestError(ValueError):
    """The manifest is malformed; nothing is processed."""


@dataclass(frozen=True, slots=True)
class Case:
    """One (participant, ROI) extraction unit."""

    participant_id: str
    roi: str
    image_path: Path
    mask_path: Path
    mask_label: int


@dataclass(frozen=True, slots=True)
class FollowUp:
    time_months: float
    event: bool  # False = right-censored at time_months


@dataclass(frozen=True, slots=True)
class FollowUpSummary:
    n_participants: int
    n_with_followup: int
    n_without_followup: int
    n_events: int
    n_censored: int


@dataclass(frozen=True, slots=True)
class Manifest:
    path: Path
    sha256: str
    cases: tuple[Case, ...]
    followup: dict[str, FollowUp]
    """Participants with usable follow-up only."""

    @property
    def participants(self) -> tuple[str, ...]:
        return tuple(sorted({case.participant_id for case in self.cases}))

    @property
    def rois(self) -> tuple[str, ...]:
        return tuple(sorted({case.roi for case in self.cases}))

    def followup_summary(self) -> FollowUpSummary:
        n = len(self.participants)
        events = sum(f.event for f in self.followup.values())
        return FollowUpSummary(
            n_participants=n,
            n_with_followup=len(self.followup),
            n_without_followup=n - len(self.followup),
            n_events=events,
            n_censored=len(self.followup) - events,
        )


def _parse_followup(time_raw: str, event_raw: str, line: int) -> FollowUp | None:
    time_raw, event_raw = time_raw.strip(), event_raw.strip().lower()
    if not time_raw and not event_raw:
        return None
    if not time_raw or not event_raw:
        raise ManifestError(f"line {line}: time_months and event must be both set or both blank")
    try:
        time_months = float(time_raw)
    except ValueError:
        raise ManifestError(f"line {line}: time_months is not a number") from None
    if not math.isfinite(time_months) or time_months <= 0:
        raise ManifestError(f"line {line}: time_months must be a positive finite number")
    if event_raw not in _TRUE | _FALSE:
        raise ManifestError(f"line {line}: event must be 0 or 1")
    return FollowUp(time_months=time_months, event=event_raw in _TRUE)


def load_manifest(path: str | Path) -> Manifest:
    """Parse and validate a site manifest (structure only; files are not opened)."""
    path = Path(path).resolve()
    raw = path.read_bytes()
    reader = csv.DictReader(raw.decode("utf-8-sig").splitlines())
    columns = set(reader.fieldnames or ())
    missing = [c for c in REQUIRED_COLUMNS if c not in columns]
    if missing:
        raise ManifestError(f"missing column(s): {', '.join(missing)}")
    unknown = columns - set(REQUIRED_COLUMNS) - set(OPTIONAL_COLUMNS)
    if unknown:
        raise ManifestError(f"unknown column(s): {', '.join(sorted(unknown))}")
    if ("time_months" in columns) != ("event" in columns):
        raise ManifestError("time_months and event columns must be given together")

    cases: list[Case] = []
    seen: set[tuple[str, str]] = set()
    images: dict[str, Path] = {}
    followups: dict[str, FollowUp | None] = {}
    for line, row in enumerate(reader, start=2):
        participant, roi = (row["participant_id"] or "").strip(), (row["roi"] or "").strip()
        if not PARTICIPANT_PATTERN.match(participant):
            raise ManifestError(f"line {line}: invalid participant_id")
        if not ROI_PATTERN.match(roi):
            raise ManifestError(f"line {line}: roi must match {ROI_PATTERN.pattern}")
        if (participant, roi) in seen:
            raise ManifestError(f"line {line}: duplicate (participant_id, roi)")
        seen.add((participant, roi))
        if not (row["image_path"] or "").strip() or not (row["mask_path"] or "").strip():
            raise ManifestError(f"line {line}: image_path and mask_path are required")
        image = (path.parent / row["image_path"].strip()).resolve()
        if images.setdefault(participant, image) != image:
            raise ManifestError(f"line {line}: participant has more than one image")
        label_raw = (row.get("mask_label") or "1").strip()
        if not label_raw.isdigit() or int(label_raw) < 1:
            raise ManifestError(f"line {line}: mask_label must be a positive integer")
        followup = _parse_followup(row.get("time_months") or "", row.get("event") or "", line)
        if followups.setdefault(participant, followup) != followup:
            raise ManifestError(f"line {line}: inconsistent follow-up for the participant")
        cases.append(
            Case(
                participant_id=participant,
                roi=roi,
                image_path=image,
                mask_path=(path.parent / row["mask_path"].strip()).resolve(),
                mask_label=int(label_raw),
            )
        )
    if not cases:
        raise ManifestError("manifest has no rows")
    return Manifest(
        path=path,
        sha256=hashlib.sha256(raw).hexdigest(),
        cases=tuple(sorted(cases, key=lambda c: (c.participant_id, c.roi))),
        followup={p: f for p, f in followups.items() if f is not None},
    )
