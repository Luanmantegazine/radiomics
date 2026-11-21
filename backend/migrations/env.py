"""Alembic environment: runs on a connection supplied by the application."""

from __future__ import annotations

from alembic import context

from backend.app.models import Base

target_metadata = Base.metadata
connection = context.config.attributes.get("connection")
if connection is None:  # pragma: no cover - always supplied by backend.app.migrations
    raise RuntimeError("run migrations through backend.app.migrations.upgrade()")

context.configure(
    connection=connection,
    target_metadata=target_metadata,
    render_as_batch=connection.dialect.name == "sqlite",
    compare_type=True,
)
with context.begin_transaction():
    context.run_migrations()
