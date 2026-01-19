"""
SQL - Dataclass-native SQLite. Zero boilerplate CRUD.

Usage:
    from dataclasses import dataclass
    from sqlow import SQL, Model

    db = SQL("app.db")

    @dataclass
    class Component(Model):
        name: str = ""
        project_id: int = 0

    @dataclass
    class Project(Model):
        title: str = ""

    components = db(Component)
    projects = db(Project)

    components.create(name="button")            # -> [Component(...)]
    components.read(id="abc-123")                # -> [Component(...)] or []
    components.update(id="abc-123", name="new")  # -> [Component(...)]
    components.delete(id="abc-123")              # -> [Component(...)]
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
from typing import Any, TypeVar, get_origin

T = TypeVar("T")

# Type mapping: Python -> SQLite
TYPE_MAP = {
    int: "INTEGER",
    str: "TEXT",
    float: "REAL",
    bool: "INTEGER",
    dict: "TEXT",  # JSON
    list: "TEXT",  # JSON
}

# Auto-managed fields
AUTO_FIELDS = {"id", "created_at", "updated_at", "deleted_at"}


def _now() -> str:
    """Current UTC timestamp as ISO string."""
    return datetime.now(timezone.utc).isoformat()


def _get_sqlite_type(py_type: Any) -> str:
    """Map Python type to SQLite type."""
    origin = get_origin(py_type)
    if origin is not None:
        py_type = next((a for a in py_type.__args__ if a is not type(None)), str)
    return TYPE_MAP.get(py_type, "TEXT")


def _is_json_type(py_type: Any) -> bool:
    """Check if type should be JSON serialized."""
    origin = get_origin(py_type)
    if origin is not None:
        py_type = next((a for a in py_type.__args__ if a is not type(None)), str)
    return py_type in (dict, list)


def _is_bool_type(py_type: Any) -> bool:
    """Check if type is bool."""
    origin = get_origin(py_type)
    if origin is not None:
        py_type = next((a for a in py_type.__args__ if a is not type(None)), str)
    return py_type is bool


@dataclass
class _FieldInfo:
    """Field metadata for SQL generation."""

    name: str
    py_type: Any
    sql_type: str
    is_json: bool
    is_bool: bool


@dataclass
class Count:
    """Pagination info returned by count()."""

    total: int
    pages: int
    per_page: int


@dataclass
class Model:
    """
    Base model with auto-managed fields.

    Inherit from this to get:
    - id: UUID auto-generated on insert
    - created_at: timestamp auto-set on insert
    - updated_at: timestamp auto-set on update
    - deleted_at: timestamp for soft deletes
    """

    id: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    deleted_at: str | None = None


def _get_fields(cls: type) -> list[_FieldInfo]:
    """Extract field info from dataclass."""
    result = []
    for f in fields(cls):
        result.append(
            _FieldInfo(
                name=f.name,
                py_type=f.type,
                sql_type=_get_sqlite_type(f.type),
                is_json=_is_json_type(f.type),
                is_bool=_is_bool_type(f.type),
            )
        )
    return result


def _has_soft_delete(cls: type) -> bool:
    """Check if class has deleted_at field (inherits from Model)."""
    return any(f.name == "deleted_at" for f in fields(cls))


class Table[T]:
    """CRUD operations for a dataclass table. Always returns list."""

    def __init__(self, db: SQL, cls: type[T]):
        if not is_dataclass(cls):
            raise TypeError(f"{cls.__name__} must be a dataclass")

        self._db = db
        self._cls = cls
        self._table = cls.__name__.lower()
        self._fields = _get_fields(cls)
        self._field_map = {f.name: f for f in self._fields}
        self._soft_delete = _has_soft_delete(cls)
        self._create_table()

    def _sql(self, sql: str, params: tuple[Any, ...] = ()) -> tuple[list[sqlite3.Row], int]:
        """Execute SQL, return (rows, lastrowid). Always closes connection."""
        conn = sqlite3.connect(self._db.path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(sql, params)
            rows = cursor.fetchall()
            conn.commit()
            return rows, cursor.lastrowid or 0
        finally:
            conn.close()

    def _create_table(self) -> None:
        """Create table if not exists."""
        cols = []
        for f in self._fields:
            if f.name == "id":
                cols.append("id TEXT PRIMARY KEY")
            else:
                cols.append(f"{f.name} {f.sql_type}")
        sql = f"CREATE TABLE IF NOT EXISTS {self._table} ({', '.join(cols)})"
        self._sql(sql)

    def _to_row(self, **kwargs: Any) -> dict[str, Any]:
        """Convert Python values to SQLite values."""
        row: dict[str, Any] = {}
        for key, value in kwargs.items():
            if key not in self._field_map:
                raise KeyError(f"Unknown field: {key}")
            field = self._field_map[key]
            if value is None:
                row[key] = None
            elif field.is_json:
                row[key] = json.dumps(value)
            else:
                row[key] = value
        return row

    def _from_row(self, row: sqlite3.Row) -> T:
        """Convert SQLite row to dataclass instance."""
        data: dict[str, Any] = {}
        for f in self._fields:
            value = row[f.name]
            if value is None:
                data[f.name] = None
            elif f.is_json:
                data[f.name] = json.loads(value)
            elif f.is_bool:
                data[f.name] = bool(value)
            else:
                data[f.name] = value
        return self._cls(**data)

    def create(self, *items: dict[str, Any] | T, **kwargs: Any) -> list[T]:
        """
        Insert records. Returns list of created items with IDs.

        Usage:
            table.create(name="button")
            table.create({"name": "a"}, {"name": "b"})
            table.create(Component(name="x"))
        """
        records: list[dict[str, Any]] = []
        if kwargs:
            records.append(kwargs)
        for item in items:
            if is_dataclass(item) and not isinstance(item, type):
                records.append(
                    {f.name: getattr(item, f.name) for f in self._fields if f.name not in AUTO_FIELDS}
                )
            elif isinstance(item, dict):
                records.append(item)
            else:
                raise TypeError(f"Expected dict or {self._cls.__name__}, got {type(item)}")

        results: list[T] = []
        for record in records:
            # Strip auto fields from input
            row = self._to_row(**{k: v for k, v in record.items() if k not in AUTO_FIELDS})
            # Set auto fields
            row["id"] = str(uuid.uuid4())
            if "created_at" in self._field_map:
                row["created_at"] = _now()
            if "updated_at" in self._field_map:
                row["updated_at"] = _now()

            cols = ", ".join(row.keys())
            placeholders = ", ".join("?" for _ in row)
            sql = f"INSERT INTO {self._table} ({cols}) VALUES ({placeholders})"
            self._sql(sql, tuple(row.values()))

            # Fetch inserted row
            rows, _ = self._sql(f"SELECT * FROM {self._table} WHERE id = ?", (row["id"],))
            if rows:
                results.append(self._from_row(rows[0]))

        return results

    def read(
        self,
        include_deleted: bool = False,
        page: int | None = None,
        per_page: int = 10,
        **kwargs: Any,
    ) -> list[T]:
        """
        Select records. Returns list (empty if none found).
        Excludes soft-deleted records by default.

        Usage:
            table.read()                      # all (non-deleted)
            table.read(id="abc")              # by id
            table.read(include_deleted=True)  # include soft-deleted
            table.read(page=1)                # first page (10 items)
            table.read(page=2, per_page=20)   # second page, 20 per page
        """
        conditions = []
        params: list[Any] = []

        # Filter by kwargs
        if kwargs:
            row = self._to_row(**kwargs)
            for k, v in row.items():
                conditions.append(f"{k} = ?")
                params.append(v)

        # Exclude soft-deleted unless requested
        if self._soft_delete and not include_deleted:
            conditions.append("deleted_at IS NULL")

        if conditions:
            sql = f"SELECT * FROM {self._table} WHERE {' AND '.join(conditions)}"
        else:
            sql = f"SELECT * FROM {self._table}"

        # Pagination (1-indexed pages)
        if page is not None:
            offset = (max(1, page) - 1) * per_page
            sql += f" LIMIT {int(per_page)} OFFSET {int(offset)}"

        rows, _ = self._sql(sql, tuple(params))
        return [self._from_row(r) for r in rows]

    def update(self, *items: dict[str, Any] | T, **kwargs: Any) -> list[T]:
        """
        Update records by id. Returns list of updated items.
        Auto-updates updated_at timestamp.

        Usage:
            table.update(id="abc", name="new")
            table.update({"id": "abc", "name": "a"}, {"id": "def", "name": "b"})
        """
        records: list[dict[str, Any]] = []
        if kwargs:
            records.append(kwargs)
        for item in items:
            if is_dataclass(item) and not isinstance(item, type):
                records.append({f.name: getattr(item, f.name) for f in self._fields})
            elif isinstance(item, dict):
                records.append(item)
            else:
                raise TypeError(f"Expected dict or {self._cls.__name__}, got {type(item)}")

        results: list[T] = []
        for record in records:
            if "id" not in record or record["id"] is None:
                raise ValueError("id required for update")

            item_id = record["id"]
            # Exclude auto fields except updated_at
            update_data = {k: v for k, v in record.items() if k not in {"id", "created_at", "deleted_at"}}
            if not update_data and "updated_at" not in self._field_map:
                continue

            row = self._to_row(**{k: v for k, v in update_data.items() if k != "updated_at"})
            # Auto-update timestamp
            if "updated_at" in self._field_map:
                row["updated_at"] = _now()

            if not row:
                continue

            set_clause = ", ".join(f"{k} = ?" for k in row.keys())
            sql = f"UPDATE {self._table} SET {set_clause} WHERE id = ?"
            self._sql(sql, (*row.values(), item_id))

            # Fetch updated row
            rows, _ = self._sql(f"SELECT * FROM {self._table} WHERE id = ?", (item_id,))
            if rows:
                results.append(self._from_row(rows[0]))

        return results

    def delete(self, *items: dict[str, Any] | T, hard: bool = False, **kwargs: Any) -> list[T]:
        """
        Delete records. Returns list of deleted items.
        Uses soft delete by default (sets deleted_at).

        Usage:
            table.delete(id="abc")             # soft delete by id
            table.delete(name="button")        # soft delete by field
            table.delete()                     # soft delete all
            table.delete(id="abc", hard=True)  # permanent delete
            table.delete({"id": "a"}, {"id": "b"})  # batch delete
        """
        # Collect records from *items
        records: list[dict[str, Any]] = []
        if kwargs:
            records.append(kwargs)
        for item in items:
            if is_dataclass(item) and not isinstance(item, type):
                # For delete, only use id from dataclass instances
                item_id = getattr(item, "id", None)
                if item_id is not None:
                    records.append({"id": item_id})
                else:
                    raise ValueError("id required for delete")
            elif isinstance(item, dict):
                records.append(item)
            else:
                raise TypeError(f"Expected dict or {self._cls.__name__}, got {type(item)}")

        # If batch mode (records provided), delete each by its filter
        if records:
            results: list[T] = []
            for record in records:
                # Get items to return
                found = self.read(include_deleted=hard, **record)
                if not found:
                    continue

                row = self._to_row(**record)
                if self._soft_delete and not hard:
                    now = _now()
                    conditions = " AND ".join(f"{k} = ?" for k in row.keys())
                    sql = f"UPDATE {self._table} SET deleted_at = ? WHERE {conditions} AND deleted_at IS NULL"
                    self._sql(sql, (now, *row.values()))
                else:
                    conditions = " AND ".join(f"{k} = ?" for k in row.keys())
                    sql = f"DELETE FROM {self._table} WHERE {conditions}"
                    self._sql(sql, tuple(row.values()))

                results.extend(found)
            return results

        # No filters: delete all
        all_items = self.read(include_deleted=hard)
        if not all_items:
            return []

        if self._soft_delete and not hard:
            sql = f"UPDATE {self._table} SET deleted_at = ? WHERE deleted_at IS NULL"
            self._sql(sql, (_now(),))
        else:
            self._sql(f"DELETE FROM {self._table}")

        return all_items

    def count(
        self, include_deleted: bool = False, per_page: int = 10, **kwargs: Any
    ) -> Count:
        """
        Count records and return pagination info.

        Usage:
            info = table.count()           # Count(total=42, pages=5, per_page=10)
            info = table.count(per_page=20)  # Count(total=42, pages=3, per_page=20)
            info.total   # 42
            info.pages   # 5
        """
        conditions = []
        params: list[Any] = []

        if kwargs:
            row = self._to_row(**kwargs)
            for k, v in row.items():
                conditions.append(f"{k} = ?")
                params.append(v)

        if self._soft_delete and not include_deleted:
            conditions.append("deleted_at IS NULL")

        if conditions:
            sql = f"SELECT COUNT(*) FROM {self._table} WHERE {' AND '.join(conditions)}"
        else:
            sql = f"SELECT COUNT(*) FROM {self._table}"

        rows, _ = self._sql(sql, tuple(params))
        total = rows[0][0] if rows else 0
        pages = (total + per_page - 1) // per_page if total > 0 else 0

        return Count(total=total, pages=pages, per_page=per_page)

    def drop(self) -> None:
        """Drop the table."""
        self._sql(f"DROP TABLE IF EXISTS {self._table}")


class SQL:
    """
    SQLite database instance. Create tables by calling with a dataclass.

    Example:
        db = SQL("app.db")

        @dataclass
        class Component(Model):
            name: str = ""

        components = db(Component)
        components.create(name="button")
    """

    def __init__(self, path: str):
        self.path = path

    def __call__(self, cls: type[T]) -> Table[T]:
        """Create a table for the given dataclass."""
        return Table(self, cls)
