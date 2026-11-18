"""Dependency rules between components, enforced on the import graph."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
COMPONENTS = {
    "shared", "client", "coordinator", "backend", "evaluation", "experiments", "harmonization",
}

#: component -> components it is allowed to import
ALLOWED = {
    "shared": set(),
    "client": {"shared", "evaluation", "harmonization"},
    "coordinator": {"shared", "evaluation", "harmonization"},
    "evaluation": {"shared"},
    "harmonization": {"shared"},
    "backend": {"shared", "coordinator"},
    "experiments": {"shared", "client", "coordinator", "backend", "evaluation", "harmonization"},
}
#: scientific / image libraries that must never be imported by the backend
SCIENTIFIC = {"radiomics", "SimpleITK", "sksurv", "sklearn", "torch", "scipy", "nibabel", "numpy"}


def _imports(package: str) -> set[tuple[str, str]]:
    found = set()
    for path in (ROOT / package).rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found.update((str(path), alias.name.split(".")[0]) for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                found.add((str(path), node.module.split(".")[0]))
    return found


@pytest.mark.parametrize("package", sorted(ALLOWED))
def test_component_dependencies(package):
    violations = [
        (path, target)
        for path, target in _imports(package)
        if target in COMPONENTS and target != package and target not in ALLOWED[package]
    ]
    assert not violations


def test_backend_contains_no_scientific_code():
    assert not [(p, t) for p, t in _imports("backend") if t in SCIENTIFIC]


def test_persistence_stays_out_of_clients_and_coordinator():
    for package in ("client", "coordinator", "shared"):
        assert not [
            (p, t) for p, t in _imports(package) if t in {"sqlalchemy", "mlflow", "fastapi"}
        ]


def test_imaging_stack_is_confined_to_the_client():
    """Images are read only inside a site; no other component can open them."""
    imaging = {"radiomics", "SimpleITK", "nibabel", "pandas"}
    for package in ("shared", "coordinator", "backend", "evaluation", "harmonization"):
        assert not [(p, t) for p, t in _imports(package) if t in imaging]


def test_imaging_pipeline_has_no_network_or_transport_dependency():
    transport = {"coordinator", "backend", "httpx", "requests", "urllib", "socket", "flwr"}
    assert not [
        (p, t) for p, t in _imports("client") if "imaging" in p and t in transport
    ]


def test_coordinator_aggregation_needs_only_numpy():
    """The coordinator runs where the backend runs; it must not need the science stack."""
    heavy = {"scipy", "sksurv", "sklearn", "radiomics", "SimpleITK", "pandas", "torch"}
    for package in ("coordinator", "evaluation", "harmonization"):
        assert not [(p, t) for p, t in _imports(package) if t in heavy]
