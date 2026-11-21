"""Write the OpenAPI schema (used to generate the dashboard's TypeScript types).

    python -m backend.app.openapi frontend/openapi.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from backend.app.main import create_app
from backend.app.settings import Settings


def main(argv: list[str] | None = None) -> int:
    target = Path((argv or sys.argv[1:] or ["openapi.json"])[0])
    app = create_app(Settings(database_url="sqlite://", mlflow_tracking_uri="none",
                              launcher="none"), launcher=None)
    target.write_text(json.dumps(app.openapi(), indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
