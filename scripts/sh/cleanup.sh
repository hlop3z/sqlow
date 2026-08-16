#!/usr/bin/env bash

NAME=src/

# Python (uv)
echo "Starting Cleanup..."

# uv add --dev ssort isort black ruff mypy pylint
uv run ssort $NAME &&
    uv run isort $NAME &&
    uv run black $NAME &&
    uv run ruff format $NAME &&
    uv run mypy $NAME &&
    uv run ruff check $NAME --fix &&
    uv run pylint $NAME

echo "All Done!"
