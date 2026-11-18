"""Prognostic models (Stage 5): local fitting, local prediction, and the local
contributions to federated fitting.

Every model uses follow-up time, the event indicator and right censoring.
A participant without an observed conversion and with incomplete follow-up is
never labelled a non-converter.
"""
