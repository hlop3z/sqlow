"""Field types must resolve in modules using postponed annotations.

Under ``from __future__ import annotations`` every annotation is a string, so
field classification has to resolve them rather than read ``field.type``
directly. This module exists solely to exercise that path -- the rest of the
suite uses eager annotations.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, date, datetime, time

import pytest

from sqlow import SQL, Model

TEST_DB = "test_future_annotations.sqlite3"


@dataclass
class Record(Model):
    name: str = ""
    active: bool = False
    meta: dict | None = None
    tags: list | None = None
    starts_at: datetime | None = None
    on_date: date | None = None
    at_time: time | None = None


def _remove_db():
    """Remove the database and its WAL sidecar files."""
    for suffix in ("", "-wal", "-shm"):
        path = TEST_DB + suffix
        if os.path.exists(path):
            os.remove(path)


@pytest.fixture(autouse=True)
def cleanup():
    _remove_db()
    yield
    _remove_db()


@pytest.fixture
def table():
    db = SQL(TEST_DB)
    yield db(Record)
    db.close()


def test_json_fields_roundtrip(table):
    result = table.create(meta={"k": "v"}, tags=["a", "b"])

    assert result[0].meta == {"k": "v"}
    assert result[0].tags == ["a", "b"]


def test_datetime_fields_roundtrip(table):
    dt = datetime(2024, 6, 15, 10, 30, tzinfo=UTC)
    result = table.create(starts_at=dt, on_date=date(2024, 6, 15), at_time=time(10, 30))

    assert result[0].starts_at == dt
    assert result[0].on_date == date(2024, 6, 15)
    assert result[0].at_time == time(10, 30)


def test_bool_field_roundtrip(table):
    assert table.create(active=True)[0].active is True
    assert table.create(active=False)[0].active is False


def test_column_types_are_not_all_text(table):
    """Unresolved annotations would silently fall back to TEXT for everything."""
    types = {f.name: f.sql_type for f in table._fields}

    assert types["active"] == "INTEGER"
    assert types["name"] == "TEXT"


def test_from_dict_parses_iso_strings():
    record = Record.from_dict({"name": "x", "starts_at": "2024-06-15T10:30:00+00:00"})

    assert record.starts_at == datetime(2024, 6, 15, 10, 30, tzinfo=UTC)
