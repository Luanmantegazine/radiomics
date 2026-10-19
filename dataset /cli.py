#!/usr/bin/env python3
"""Repository-level entry point: ``python cli.py <command> ...``.

Delegates to :mod:`oasis_radiomics.cli`, which is also importable as a module
(``python -m oasis_radiomics.cli``).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from oasis_radiomics.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
