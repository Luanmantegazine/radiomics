"""Download of OASIS-3 FreeSurfer sessions via the official NITRC-IR scripts.

This module is a thin wrapper around ``NrgXnat/oasis-scripts``: OASIS-3 is a
credentialed dataset and the official downloader is the sanctioned way of
retrieving it. **Nothing here runs unless it is called explicitly** - the
pipeline's ``extract``/``run`` commands work exclusively on data that is already
on disk, so no bulk download can be triggered by accident.

Credentials are never read, stored or logged here: the NITRC-IR password is
prompted for by the official shell script itself.
"""

from __future__ import annotations

import csv
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)

OASIS_SCRIPTS_REPO = "https://github.com/NrgXnat/oasis-scripts.git"
DOWNLOADER_RELATIVE_PATH = Path("download_freesurfer") / "download_oasis_freesurfer.sh"

#: Example FreeSurfer ids taken from the official repository's sample file.
EXAMPLE_FREESURFER_IDS = ("OAS30001_Freesurfer53_d0129", "OAS30001_Freesurfer53_d0757")


class DownloadError(RuntimeError):
    """Raised when the official downloader cannot be prepared or executed."""


def read_freesurfer_ids(csv_path: Path, max_cases: int | None = None) -> list[str]:
    """Read FreeSurfer ids from a CSV with an optional ``freesurfer_id`` header.

    A bare one-id-per-line file is accepted as well.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is empty or holds no usable id.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found.\n"
            "Create it with a 'freesurfer_id' header, for example:\n"
            "freesurfer_id\n"
            "OAS30001_Freesurfer53_d0129\n"
            "OAS30001_Freesurfer53_d0757\n"
        )

    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))

    if not rows:
        raise ValueError(f"The FreeSurfer id CSV is empty: {csv_path}")

    start = 1 if rows[0] and rows[0][0].strip().lower() == "freesurfer_id" else 0

    ids: list[str] = []
    for row in rows[start:]:
        if not row:
            continue
        freesurfer_id = row[0].strip()
        if freesurfer_id:
            ids.append(freesurfer_id)
        if max_cases is not None and len(ids) >= max_cases:
            break

    if not ids:
        raise ValueError(f"No FreeSurfer ids were found in {csv_path}")

    logger.info("Read %d FreeSurfer id(s) from %s", len(ids), csv_path)
    return ids


def write_freesurfer_ids(ids: Sequence[str], path: Path) -> Path:
    """Write ``ids`` as a one-column CSV that the official downloader accepts."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["freesurfer_id"])
        for freesurfer_id in ids:
            writer.writerow([freesurfer_id])
    logger.debug("Wrote %d FreeSurfer id(s) to %s", len(ids), path)
    return path


def create_example_ids_csv(path: Path) -> Path:
    """Write a two-session example id file (the sessions used by the pilot)."""
    written = write_freesurfer_ids(EXAMPLE_FREESURFER_IDS, path)
    logger.info("Example FreeSurfer id CSV created: %s", written)
    return written


def _require_command(name: str) -> None:
    """Fail loudly when an external command the downloader needs is missing."""
    if shutil.which(name) is None:
        raise DownloadError(f"Required command not found on PATH: {name}")


def ensure_oasis_scripts(repo_dir: Path) -> Path:
    """Return the path to the official downloader, cloning the repo if needed."""
    repo_dir = Path(repo_dir)
    downloader = repo_dir / DOWNLOADER_RELATIVE_PATH
    if downloader.exists():
        logger.debug("Using existing OASIS downloader at %s", downloader)
        return downloader

    _require_command("git")
    logger.info("Cloning the official OASIS downloader into %s", repo_dir)
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", OASIS_SCRIPTS_REPO, str(repo_dir)], check=True
        )
    except subprocess.CalledProcessError as exc:
        raise DownloadError(f"Could not clone {OASIS_SCRIPTS_REPO}: {exc}") from exc

    if not downloader.exists():
        raise DownloadError(
            f"The official downloader was not found after cloning: {downloader}"
        )
    return downloader


def download_freesurfer_sessions(
    ids_csv: Path,
    download_dir: Path,
    nitrc_user: str,
    repo_dir: Path,
) -> Path:
    """Run the official downloader for the ids listed in ``ids_csv``.

    The NITRC-IR password is prompted for by the official script and is never
    handled by this code.

    Returns
    -------
    Path
        The directory the sessions were downloaded into.
    """
    for command in ("bash", "curl", "unzip"):
        _require_command(command)

    downloader = ensure_oasis_scripts(Path(repo_dir))
    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting the OASIS-3 FreeSurfer download into %s", download_dir)
    logger.info("The official downloader will prompt for your NITRC-IR password.")

    try:
        subprocess.run(
            [
                "bash",
                str(downloader),
                str(Path(ids_csv).resolve()),
                str(download_dir.resolve()),
                nitrc_user,
            ],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise DownloadError(f"The official OASIS downloader failed: {exc}") from exc

    return download_dir
