#!/bin/bash
# Runs once, on first Postgres init. MLflow keeps its tracking tables in a
# separate database so a tracking-store migration can never touch Kairos data.
set -e
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-SQL
    SELECT 'CREATE DATABASE mlflow'
    WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'mlflow')\gexec
SQL
