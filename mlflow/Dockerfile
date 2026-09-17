FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# psycopg2-binary: MLflow's Postgres backend store driver.
RUN pip install --no-cache-dir "mlflow>=2.17,<3.0" "psycopg2-binary>=2.9"

EXPOSE 5000
