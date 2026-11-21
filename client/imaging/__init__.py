"""Local radiomic pipeline of a federated client (Stage 2).

manifest -> validation -> PyRadiomics extraction -> local feature table.

Everything in this package reads or writes individual-level data and runs
strictly inside one site. Its outputs are *local files*; the only bridge to the
federation is :class:`client.feature_table.FeatureTableDataSource`, whose data
still leaves the site exclusively through authorized task payloads.
"""
