# MLflow tracking server with the PostgreSQL driver.
FROM python:3.12-slim
RUN pip install --no-cache-dir "mlflow>=3.0,<4" "psycopg2-binary>=2.9,<3"
EXPOSE 5000
