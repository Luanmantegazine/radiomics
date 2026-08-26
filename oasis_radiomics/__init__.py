"""OASIS-3 longitudinal hippocampal radiomics pipeline.

The package is deliberately **not** named ``radiomics``: that name belongs to
PyRadiomics and a local package with the same name would shadow it whenever the
repository root ends up on ``sys.path``.

Public entry points
-------------------
``oasis_radiomics.cli``       command line interface (``extract``/``longitudinal``/``run``/``qc``)
``oasis_radiomics.pipeline``  programmatic orchestration of the same steps
"""

from __future__ import annotations

__version__ = "0.2.0"

__all__ = ["__version__"]
