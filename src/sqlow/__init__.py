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
from datetime import date, datetime, time, timezone
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
    datetime: "TEXT",  # ISO format
    date: "TEXT",  # ISO format
    time: "TEXT",  # ISO format
}

# Auto-managed fields
AUTO_FIELDS = {"id", "created_at", "updated_at", "deleted_at"}


def _now() -> str:
    """Return current UTC timestamp as ISO string."""
    return datetime.now(timezone.utc).isoformat()


def _get_sqlite_type(py_type: Any) -> str:
    """Map Python type to SQLite type.

    Args:
        py_type: Python type annotation.

    Returns:
        SQLite type string (TEXT, INTEGER, REAL).
    """
    origin = get_origin(py_type)
    if origin is not None:
        py_type = next((a for a in py_type.__args__ if a is not type(None)), str)
    return TYPE_MAP.get(py_type, "TEXT")


def _is_json_type(py_type: Any) -> bool:
    """Check if type should be JSON serialized.

    Args:
        py_type: Python type annotation.

    Returns:
        True if type is dict or list.
    """
    origin = get_origin(py_type)
    if origin is not None:
        py_type = next((a for a in py_type.__args__ if a is not type(None)), str)
    return py_type in (dict, list)


def _is_bool_type(py_type: Any) -> bool:
    """Check if type is bool.

    Args:
        py_type: Python type annotation.

    Returns:
        True if type is bool.
    """
    origin = get_origin(py_type)
    if origin is not None:
        py_type = next((a for a in py_type.__args__ if a is not type(None)), str)
    return py_type is bool


def _is_datetime_type(py_type: Any) -> type[datetime] | type[date] | type[time] | None:
    """Check if type is datetime, date, or time.

    Args:
        py_type: Python type annotation.

    Returns:
        The datetime/date/time type, or None if not a datetime type.
    """
    origin = get_origin(py_type)
    if origin is not None:
        py_type = next((a for a in py_type.__args__ if a is not type(None)), str)
    if py_type is datetime:
        return datetime
    if py_type is date:
        return date
    if py_type is time:
        return time
    return None


@dataclass
class _FieldInfo:
    """Field metadata for SQL generation.

    Attributes:
        name: Field name.
        py_type: Python type annotation.
        sql_type: SQLite type string.
        is_json: True if field should be JSON serialized.
        is_bool: True if field is boolean.
        datetime_type: datetime/date/time type, or None.
    """

    name: str
    py_type: Any
    sql_type: str
    is_json: bool
    is_bool: bool
    datetime_type: type[datetime] | type[date] | type[time] | None


@dataclass
class Count:
    """Pagination info returned by count().

    Attributes:
        total: Total number of records.
        pages: Total number of pages.
        per_page: Records per page.
    """

    total: int
    pages: int
    per_page: int


@dataclass
class Model:
    """Base model with auto-managed fields.

    Inherit from this to get automatic field management:
        - id: UUID auto-generated on create
        - created_at: ISO timestamp set on create
        - updated_at: ISO timestamp set on update
        - deleted_at: ISO timestamp set on soft delete

    Attributes:
        id: UUID primary key, auto-generated.
        created_at: Creation timestamp in ISO format.
        updated_at: Last update timestamp in ISO format.
        deleted_at: Soft delete timestamp, None if not deleted.
    """

    id: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    deleted_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert instance to dict with datetime types as ISO strings.

        Returns:
            Dict with all fields. Datetime/date/time values are ISO strings.

        Example:
            >>> user.to_dict()
            {"id": "abc", "name": "Alice", "created_at": "2024-01-01T00:00:00+00:00"}
        """
        result: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, (datetime, date, time)):
                result[f.name] = value.isoformat()
            else:
                result[f.name] = value
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Model":
        """Create instance from dict, parsing ISO strings to datetime types.

        Args:
            data: Dict with field values. ISO strings are parsed to datetime.

        Returns:
            New instance of the model class.

        Example:
            >>> User.from_dict({"name": "Alice", "created_at": "2024-01-01T00:00:00+00:00"})
            User(name="Alice", created_at=datetime(...))
        """
        parsed: dict[str, Any] = {}
        for f in fields(cls):
            if f.name not in data:
                continue
            value = data[f.name]
            if value is None:
                parsed[f.name] = None
            else:
                dt_type = _is_datetime_type(f.type)
                if dt_type is datetime and isinstance(value, str):
                    dt = datetime.fromisoformat(value)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    parsed[f.name] = dt
                elif dt_type is date and isinstance(value, str):
                    parsed[f.name] = date.fromisoformat(value)
                elif dt_type is time and isinstance(value, str):
                    parsed[f.name] = time.fromisoformat(value)
                else:
                    parsed[f.name] = value
        return cls(**parsed)


def _get_fields(cls: type) -> list[_FieldInfo]:
    """Extract field metadata from dataclass.

    Args:
        cls: Dataclass type.

    Returns:
        List of field info objects.
    """
    result = []
    for f in fields(cls):
        result.append(
            _FieldInfo(
                name=f.name,
                py_type=f.type,
                sql_type=_get_sqlite_type(f.type),
                is_json=_is_json_type(f.type),
                is_bool=_is_bool_type(f.type),
                datetime_type=_is_datetime_type(f.type),
            )
        )
    return result


def _has_soft_delete(cls: type) -> bool:
    """Check if class supports soft delete.

    Args:
        cls: Dataclass type.

    Returns:
        True if class has deleted_at field.
    """
    return any(f.name == "deleted_at" for f in fields(cls))


class Table[T]:
    """CRUD operations for a dataclass table.

    All operations return list[T] for consistency.

    Args:
        db: SQL database instance.
        cls: Dataclass type for the table.

    Raises:
        TypeError: If cls is not a dataclass.
    """

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

    def _sql(
        self, sql: str, params: tuple[Any, ...] = ()
    ) -> tuple[list[sqlite3.Row], int]:
        """Execute SQL and return results.

        Args:
            sql: SQL query string.
            params: Query parameters.

        Returns:
            Tuple of (rows, lastrowid).
        """
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
        """Create table if it does not exist."""
        cols = []
        for f in self._fields:
            if f.name == "id":
                cols.append("id TEXT PRIMARY KEY")
            else:
                cols.append(f"{f.name} {f.sql_type}")
        sql = f"CREATE TABLE IF NOT EXISTS {self._table} ({', '.join(cols)})"
        self._sql(sql)

    def _to_row(self, **kwargs: Any) -> dict[str, Any]:
        """Convert Python values to SQLite values.

        Args:
            **kwargs: Field name/value pairs.

        Returns:
            Dict with values converted for SQLite storage.

        Raises:
            KeyError: If field name is unknown.
        """
        row: dict[str, Any] = {}
        for key, value in kwargs.items():
            if key not in self._field_map:
                raise KeyError(f"Unknown field: {key}")
            field = self._field_map[key]
            if value is None:
                row[key] = None
            elif field.is_json:
                row[key] = json.dumps(value)
            elif field.datetime_type is datetime and isinstance(value, datetime):
                # Always store datetime in UTC
                if value.tzinfo is None:
                    value = value.replace(tzinfo=timezone.utc)
                else:
                    value = value.astimezone(timezone.utc)
                row[key] = value.isoformat()
            elif field.datetime_type is not None and isinstance(value, (date, time)):
                row[key] = value.isoformat()
            else:
                row[key] = value
        return row

    def _from_row(self, row: sqlite3.Row) -> T:
        """Convert SQLite row to dataclass instance.

        Args:
            row: SQLite row object.

        Returns:
            Dataclass instance with values from row.
        """
        data: dict[str, Any] = {}
        for f in self._fields:
            value = row[f.name]
            if value is None:
                data[f.name] = None
            elif f.is_json:
                data[f.name] = json.loads(value)
            elif f.is_bool:
                data[f.name] = bool(value)
            elif f.datetime_type is datetime and isinstance(value, str):
                dt = datetime.fromisoformat(value)
                # Ensure UTC timezone
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                data[f.name] = dt
            elif f.datetime_type is date and isinstance(value, str):
                data[f.name] = date.fromisoformat(value)
            elif f.datetime_type is time and isinstance(value, str):
                data[f.name] = time.fromisoformat(value)
            else:
                data[f.name] = value
        return self._cls(**data)

    def create(self, *items: dict[str, Any] | T, **kwargs: Any) -> list[T]:
        """Insert records into the table.

        Args:
            *items: Dicts or dataclass instances to insert.
            **kwargs: Field values for single record insert.

        Returns:
            List of created items with auto-generated IDs.

        Raises:
            TypeError: If item is not a dict or dataclass instance.

        Example:
            >>> table.create(name="button")
            >>> table.create({"name": "a"}, {"name": "b"})
            >>> table.create(Component(name="x"))
        """
        records: list[dict[str, Any]] = []
        if kwargs:
            records.append(kwargs)
        for item in items:
            if is_dataclass(item) and not isinstance(item, type):
                records.append(
                    {
                        f.name: getattr(item, f.name)
                        for f in self._fields
                        if f.name not in AUTO_FIELDS
                    }
                )
            elif isinstance(item, dict):
                records.append(item)
            else:
                raise TypeError(
                    f"Expected dict or {self._cls.__name__}, got {type(item)}"
                )

        results: list[T] = []
        for record in records:
            # Strip auto fields from input
            row = self._to_row(
                **{k: v for k, v in record.items() if k not in AUTO_FIELDS}
            )
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
            rows, _ = self._sql(
                f"SELECT * FROM {self._table} WHERE id = ?", (row["id"],)
            )
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
        """Select records from the table.

        Excludes soft-deleted records by default.

        Args:
            include_deleted: If True, include soft-deleted records.
            page: Page number (1-indexed) for pagination.
            per_page: Records per page. Defaults to 10.
            **kwargs: Field filters (e.g., name="Alice").

        Returns:
            List of matching records, empty list if none found.

        Example:
            >>> table.read()                      # all non-deleted
            >>> table.read(id="abc")              # by id
            >>> table.read(page=1, per_page=20)   # paginated
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
        """Update records by id.

        Auto-updates the updated_at timestamp.

        Args:
            *items: Dicts or dataclass instances with id and fields to update.
            **kwargs: Field values for single record update (must include id).

        Returns:
            List of updated records.

        Raises:
            ValueError: If id is not provided.
            TypeError: If item is not a dict or dataclass instance.

        Example:
            >>> table.update(id="abc", name="new")
            >>> table.update({"id": "a", "name": "x"}, {"id": "b", "name": "y"})
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
                raise TypeError(
                    f"Expected dict or {self._cls.__name__}, got {type(item)}"
                )

        results: list[T] = []
        for record in records:
            if "id" not in record or record["id"] is None:
                raise ValueError("id required for update")

            item_id = record["id"]
            # Exclude auto fields except updated_at
            update_data = {
                k: v
                for k, v in record.items()
                if k not in {"id", "created_at", "deleted_at"}
            }
            if not update_data and "updated_at" not in self._field_map:
                continue

            row = self._to_row(
                **{k: v for k, v in update_data.items() if k != "updated_at"}
            )
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

    def delete(
        self, *items: dict[str, Any] | T, hard: bool = False, **kwargs: Any
    ) -> list[T]:
        """Delete records from the table.

        Uses soft delete by default (sets deleted_at timestamp).

        Args:
            *items: Dicts or dataclass instances to delete.
            hard: If True, permanently delete instead of soft delete.
            **kwargs: Field filters for deletion.

        Returns:
            List of deleted records.

        Raises:
            ValueError: If dataclass instance has no id.
            TypeError: If item is not a dict or dataclass instance.

        Example:
            >>> table.delete(id="abc")                    # soft delete
            >>> table.delete(id="abc", hard=True)         # permanent
            >>> table.delete({"id": "a"}, {"id": "b"})    # batch
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
                raise TypeError(
                    f"Expected dict or {self._cls.__name__}, got {type(item)}"
                )

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
        """Count records and return pagination info.

        Args:
            include_deleted: If True, include soft-deleted records.
            per_page: Records per page for pagination calculation.
            **kwargs: Field filters.

        Returns:
            Count object with total, pages, and per_page.

        Example:
            >>> info = table.count(per_page=20)
            >>> info.total   # 42
            >>> info.pages   # 3
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
        """Drop the table from the database."""
        self._sql(f"DROP TABLE IF EXISTS {self._table}")


class SQL:
    """SQLite database instance.

    Create tables by calling the instance with a dataclass.

    Args:
        path: Path to SQLite database file. Use ":memory:" for in-memory DB.

    Example:
        >>> db = SQL("app.db")
        >>> @dataclass
        ... class User(Model):
        ...     name: str = ""
        >>> users = db(User)
        >>> users.create(name="Alice")
    """

    def __init__(self, path: str):
        self.path = path

    def __call__(self, cls: type[T]) -> Table[T]:
        """Create a table for the given dataclass.

        Args:
            cls: Dataclass type to create table for.

        Returns:
            Table instance for CRUD operations.
        """
        return Table(self, cls)
