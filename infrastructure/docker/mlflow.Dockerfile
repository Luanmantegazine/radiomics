# MLflow tracking server with the PostgreSQL driver.
# Versions match uv.lock (mlflow-skinny / psycopg used by the backend). MLflow
# opens postgresql:// URLs with psycopg 3, so psycopg2 is not enough.
FROM python:3.12-slim
RUN pip install --no-cache-dir "mlflow==3.18.0" "psycopg[binary]==3.3.9"
ENV MLFLOW_DISABLE_TELEMETRY=true DO_NOT_TRACK=true
EXPOSE 5000
