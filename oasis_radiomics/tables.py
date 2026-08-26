"""CSV writing helpers with a stable, human-readable column order.

The pipeline builds plain lists of dictionaries; pandas is used only at the
edge, to serialise them. Keeping the core pandas-free makes the transformations
trivial to unit test.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

logger = logging.getLogger(__name__)


def ordered_columns(rows: Sequence[Mapping[str, Any]], leading: Sequence[str]) -> list[str]:
    """Column order: ``leading`` first (when present), then first-seen order."""
    seen: list[str] = []
    for row in rows:
        for column in row:
            if column not in seen:
                seen.append(column)

    head = [column for column in leading if column in seen]
    tail = [column for column in seen if column not in head]
    return head + tail


def write_csv(
    rows: Sequence[Mapping[str, Any]], path: Path, leading: Sequence[str] = ()
) -> Path:
    """Write ``rows`` to ``path`` as CSV, creating parent directories.

    An empty ``rows`` still produces a file, so downstream steps and reviewers
    can tell "ran and found nothing" from "never ran".
    """
    import pandas as pd

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        logger.warning("No rows to write; creating an empty file: %s", path)
        path.write_text("", encoding="utf-8")
        return path

    frame = pd.DataFrame(list(rows))
    frame = frame.reindex(columns=ordered_columns(rows, leading))
    frame.to_csv(path, index=False)
    logger.info("Wrote %s (%d rows x %d columns)", path, len(frame), len(frame.columns))
    return path


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    """Read a CSV written by :func:`write_csv` back into a list of dicts."""
    import pandas as pd

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Table not found: {path}")
    if path.stat().st_size == 0:
        logger.warning("Table is empty: %s", path)
        return []

    frame = pd.read_csv(path)
    return frame.to_dict(orient="records")
