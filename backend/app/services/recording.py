"""Persists coordinator progress: implements ``coordinator.ports.RunRecorder``.

Only aggregate results, metadata and costs are written. Per-client payloads
never reach this module (the coordinator hands over ``RoundOutcome`` objects,
which do not contain them).
"""

from __future__ import annotations

from sqlalchemy.orm import Session, sessionmaker

from backend.app.models import (
    ClientRecord,
    CostRecord,
    Experiment,
    FederatedRound,
    RoundParticipation,
    utcnow,
)
from backend.app.tracking import ExperimentTracker
from coordinator.outcomes import ExperimentOutcome, RoundOutcome
from shared.protocol import ClientRegistration, ReasonCode


class DatabaseRunRecorder:
    def __init__(
        self, experiment_id: str, sessions: sessionmaker[Session], tracker: ExperimentTracker
    ) -> None:
        self._experiment_id = experiment_id
        self._sessions = sessions
        self._tracker = tracker
        self._run_id: str | None = None

    def experiment_started(self, experiment_id: str) -> None:
        with self._sessions() as session:
            experiment = session.get_one(Experiment, experiment_id)
            self._run_id = self._tracker.start_run(
                experiment_name=experiment.name,
                run_name=experiment.id,
                params={
                    "seed": experiment.seed,
                    "config_hash": experiment.config_hash,
                    "protocol_version": experiment.protocol_version,
                    "n_rounds": len(experiment.config["rounds"]),
                    "min_cohort_size": experiment.config["privacy"]["min_cohort_size"],
                },
                tags={"data_origin": experiment.data_origin, "experiment_id": experiment.id},
            )
            self._tracker.log_dict(self._run_id, experiment.config, "experiment_config.json")
            experiment.status = "running"
            experiment.started_at = utcnow()
            experiment.mlflow_run_id = self._run_id
            session.commit()

    def client_registered(
        self, registration: ClientRegistration, accepted: bool, reason: ReasonCode | None
    ) -> None:
        with self._sessions() as session:
            session.add(
                ClientRecord(
                    experiment_id=self._experiment_id,
                    client_id=registration.client_id,
                    status="registered" if accepted else "rejected",
                    reason_code=reason.value if reason else None,
                    data_origin=registration.data_origin.value,
                    software=dict(registration.software),
                )
            )
            session.commit()

    def round_finished(self, outcome: RoundOutcome) -> None:
        def cost(scope, metric, value, unit, client_id=None) -> CostRecord:
            return CostRecord(
                experiment_id=self._experiment_id, round_id=outcome.round_id, scope=scope,
                client_id=client_id, metric=metric, value=float(value), unit=unit,
            )

        bytes_up = sum(c.bytes_sent for c in outcome.clients)
        bytes_down = sum(c.bytes_received for c in outcome.clients)
        with self._sessions() as session:
            session.add(
                FederatedRound(
                    id=outcome.round_id,
                    experiment_id=self._experiment_id,
                    round_number=outcome.round_number,
                    phase=outcome.phase.value,
                    task=outcome.task.value,
                    aggregator=outcome.aggregator,
                    status="completed" if outcome.completed else "failed",
                    failure_reason=outcome.failure_reason,
                    aggregate=outcome.aggregate.model_dump(mode="json")
                    if outcome.aggregate
                    else None,
                )
            )
            session.add_all(
                [
                    cost("coordinator", "communication_time", outcome.communication_s, "s"),
                    cost("coordinator", "aggregation_time", outcome.aggregation_s, "s"),
                    cost("coordinator", "round_time", outcome.duration_s, "s"),
                    cost("coordinator", "bytes_from_clients", bytes_up, "bytes"),
                    cost("coordinator", "bytes_to_clients", bytes_down, "bytes"),
                ]
            )
            for client in outcome.clients:
                session.add(
                    RoundParticipation(
                        round_id=outcome.round_id,
                        client_id=client.client_id,
                        status=client.status.value,
                        reason_code=client.reason_code.value if client.reason_code else None,
                        n_samples=client.n_samples,
                    )
                )
                cid = client.client_id
                session.add_all(
                    [
                        cost("client", "bytes_sent", client.bytes_sent, "bytes", cid),
                        cost("client", "bytes_received", client.bytes_received, "bytes", cid),
                        cost("client", "round_trip_time", client.round_trip_s, "s", cid),
                        *(
                            cost("client", f"{stage}_time", seconds, "s", cid)
                            for stage, seconds in client.costs.durations_s.items()
                        ),
                    ]
                )
                if client.costs.peak_memory_bytes is not None:
                    session.add(
                        cost("client", "peak_memory", client.costs.peak_memory_bytes, "bytes", cid)
                    )
            session.commit()
        self._tracker.log_metrics(
            self._run_id,
            {
                "round_time_s": outcome.duration_s,
                "communication_time_s": outcome.communication_s,
                "aggregation_time_s": outcome.aggregation_s,
                "bytes_from_clients": bytes_up,
                "bytes_to_clients": bytes_down,
                "n_completed_clients": sum(c.status.value == "completed" for c in outcome.clients),
            },
            step=outcome.round_number,
        )
        if outcome.aggregate is not None:
            self._tracker.log_dict(
                self._run_id,
                outcome.aggregate.model_dump(mode="json"),
                f"aggregates/round_{outcome.round_number:03d}.json",
            )

    def experiment_finished(self, outcome: ExperimentOutcome) -> None:
        with self._sessions() as session:
            experiment = session.get_one(Experiment, self._experiment_id)
            experiment.status = "completed" if outcome.completed else "failed"
            if not outcome.completed:
                experiment.error = next(
                    (r.failure_reason for r in outcome.rounds if r.failure_reason),
                    "not enough registered clients",
                )
            experiment.finished_at = utcnow()
            session.add_all(
                [
                    CostRecord(experiment_id=self._experiment_id, scope="experiment",
                               metric="total_time", value=outcome.duration_s, unit="s"),
                    CostRecord(experiment_id=self._experiment_id, scope="experiment",
                               metric="total_bytes", value=outcome.total_bytes, unit="bytes"),
                    CostRecord(experiment_id=self._experiment_id, scope="experiment",
                               metric="n_rounds", value=len(outcome.rounds), unit="count"),
                ]
            )
            session.commit()
        self._tracker.log_metrics(
            self._run_id,
            {"total_time_s": outcome.duration_s, "total_bytes": outcome.total_bytes,
             "n_rounds": len(outcome.rounds)},
            step=len(outcome.rounds),
        )
        self._tracker.end_run(self._run_id, outcome.completed)
