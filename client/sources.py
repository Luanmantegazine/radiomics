"""Build a site's local data source from its (local) specification."""

from __future__ import annotations

from client.data import LocalDataSource
from client.synthetic import SyntheticDataSource
from shared.config import SimulatedClientSpec, SyntheticDataSourceSpec
from shared.seeding import derive_seed


def build_data_source(spec: SimulatedClientSpec, seed: int) -> LocalDataSource:
    source = spec.data_source
    if isinstance(source, SyntheticDataSourceSpec):
        return SyntheticDataSource(
            source, client_id=spec.client_id, seed=derive_seed(seed, "client-data", spec.client_id)
        )
    # Imported lazily: the imaging stack is an optional dependency.
    from client.feature_table import FeatureTableDataSource

    return FeatureTableDataSource(
        source.features_path, source.manifest_path, source.data_origin, source.reliability_path
    )
