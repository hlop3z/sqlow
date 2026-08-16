"""
SQLow - Dataclass-native SQLite. Zero boilerplate CRUD.

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

    components.create(name="button")             # -> [Component(...)]
    components.read(id="abc-123")                # -> [Component(...)] or []
    components.update(id="abc-123", name="new")  # -> [Component(...)]
    components.delete(id="abc-123")              # -> [Component(...)]
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
import weakref
from collections.abc import Callable, Generator, MutableMapping
from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass
from datetime import UTC, date, datetime, time
from itertools import batched
from time import time_ns
from typing import Any, Final, Self, get_origin, get_type_hints

__all__ = ["SQL", "Count", "Model", "Table"]

# Type mapping: Python -> SQLite
TYPE_MAP: Final[dict[type, str]] = {
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

# Auto-managed fields, owned by the library and never written from user input
AUTO_FIELDS: Final = frozenset({"id", "created_at", "updated_at", "deleted_at"})

# Rows per statement in batched IN-clause queries. Old SQLite builds cap bound
# parameters at 999 (SQLITE_MAX_VARIABLE_NUMBER), so stay under that.
_BATCH_SIZE: Final = 500


def _uuid7() -> uuid.UUID:
    """Generate a UUID version 7 (RFC 9562 section 5.7).

    Layout, most significant bit first: a 48-bit big-endian Unix timestamp in
    milliseconds, the 4-bit version, 12 random bits, the 2-bit variant, and 62
    more random bits.

    Returns:
        A time-ordered UUID.
    """
    rand = int.from_bytes(os.urandom(10), "big")
    value = (time_ns() // 1_000_000 & 0xFFFF_FFFF_FFFF) << 80  # unix_ts_ms
    value |= 0x7 << 76  # ver
    value |= (rand >> 62 & 0xFFF) << 64  # rand_a
    value |= 0b10 << 62  # var
    value |= rand & (1 << 62) - 1  # rand_b
    return uuid.UUID(int=value)


# uuid.uuid7 is stdlib from Python 3.14; fall back on 3.12/3.13
_new_id: Callable[[], uuid.UUID] = getattr(uuid, "uuid7", _uuid7)


def _now() -> str:
    """Return current UTC timestamp as ISO string."""
    return datetime.now(UTC).isoformat()


def _quote(name: str) -> str:
    """Quote an SQL identifier.

    Lets fields use reserved words (``order``, ``when``, ``group``) as names.

    Args:
        name: Table or column name.

    Returns:
        The name wrapped in double quotes.
    """
    escaped = name.replace('"', '""')
    return f'"{escaped}"'


def _unwrap(py_type: Any) -> Any:
    """Strip Optional/Union wrappers down to the underlying type.

    Args:
        py_type: Python type annotation.

    Returns:
        The first non-None member of a union, or py_type unchanged.
    """
    if get_origin(py_type) is None:
        return py_type
    args = getattr(py_type, "__args__", ())
    return next((a for a in args if a is not type(None)), str)


@dataclass(frozen=True, slots=True)
class _FieldInfo:
    """Field metadata for SQL generation.

    Attributes:
        name: Field name.
        py_type: Python type annotation, with Optional unwrapped.
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
    datetime_type: type[datetime | date | time] | None


def _field_info(name: str, py_type: Any) -> _FieldInfo:
    """Classify one annotation into field metadata.

    Args:
        name: Field name.
        py_type: Python type annotation.

    Returns:
        Field info describing how the field maps to SQLite.
    """
    py_type = _unwrap(py_type)
    return _FieldInfo(
        name=name,
        py_type=py_type,
        sql_type=TYPE_MAP.get(py_type, "TEXT"),
        is_json=py_type in (dict, list),
        is_bool=py_type is bool,
        datetime_type=py_type if py_type in (datetime, date, time) else None,
    )


# _get_fields cache; model classes live for the process, so entries never rot
_FIELDS_CACHE: Final[dict[type, tuple[_FieldInfo, ...]]] = {}


def _get_fields(cls: type) -> tuple[_FieldInfo, ...]:
    """Extract field metadata from a dataclass.

    Annotations are resolved through ``get_type_hints`` so modules using
    ``from __future__ import annotations`` (where ``field.type`` is a string)
    classify identically to modules that do not. Resolution is expensive, so
    results are cached per class.

    Args:
        cls: Dataclass type.

    Returns:
        Tuple of field info objects.
    """
    cached = _FIELDS_CACHE.get(cls)
    if cached is None:
        try:
            hints = get_type_hints(cls)
        except (NameError, TypeError):  # unresolvable forward reference
            hints = {}
        cached = _FIELDS_CACHE[cls] = tuple(
            _field_info(f.name, hints.get(f.name, f.type)) for f in fields(cls)
        )
    return cached


def _encode(info: _FieldInfo, value: Any) -> Any:
    """Convert a Python value to its SQLite representation.

    Args:
        info: Metadata for the target field.
        value: Python value.

    Returns:
        Value suitable for binding as a SQLite parameter.
    """
    if value is None:
        return None
    if info.is_json:
        # Compact separators: no cosmetic whitespace in stored JSON
        return json.dumps(value, separators=(",", ":"))
    if info.datetime_type is datetime and isinstance(value, datetime):
        # Always store datetime in UTC
        value = (
            value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
        )
        return value.isoformat()
    if info.datetime_type is not None and isinstance(value, (date, time)):
        return value.isoformat()
    return value


def _decode(info: _FieldInfo, value: Any) -> Any:
    """Convert a stored value back to its Python type.

    Args:
        info: Metadata for the source field.
        value: Value from SQLite or a plain dict.

    Returns:
        Python value of the field's declared type.
    """
    if value is None:
        return None
    if info.is_json:
        return json.loads(value) if isinstance(value, str) else value
    if info.is_bool:
        return bool(value)
    if info.datetime_type is not None and isinstance(value, str):
        if info.datetime_type is datetime:
            parsed = datetime.fromisoformat(value)
            # Naive timestamps (legacy rows) are treated as UTC
            return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
        return info.datetime_type.fromisoformat(value)
    return value


@dataclass(slots=True)
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
        - id: UUIDv7 auto-generated on create
        - created_at: ISO timestamp set on create
        - updated_at: ISO timestamp set on update
        - deleted_at: ISO timestamp set on soft delete

    Attributes:
        id: UUIDv7 primary key, auto-generated and time-ordered.
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
                value = value.isoformat()
            result[f.name] = value
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create instance from dict, parsing ISO strings to datetime types.

        Args:
            data: Dict with field values. ISO strings are parsed to datetime.

        Returns:
            New instance of the model class.

        Example:
            >>> User.from_dict({"name": "Alice", "created_at": "2024-01-01T00:00:00+00:00"})
            User(name="Alice", created_at=datetime(...))
        """
        parsed = {
            info.name: _decode(info, data[info.name])
            for info in _get_fields(cls)
            if info.name in data
        }
        return cls(**parsed)


class Table[T]:
    """CRUD operations for a dataclass table.

    All operations return list[T] for consistency.

    Args:
        db: SQL database instance.
        cls: Dataclass type for the table.

    Raises:
        TypeError: If cls is not a dataclass.
    """

    def _create_table(self) -> None:
        """Create table if it does not exist.

        New tables are STRICT, so SQLite rejects values that do not match the
        declared column type rather than storing them as-is. Tables created by
        earlier versions keep their original, non-STRICT schema.
        """
        cols = [
            (
                "id TEXT PRIMARY KEY"
                if f.name == "id"
                else f"{_quote(f.name)} {f.sql_type}"
            )
            for f in self._fields
        ]
        self._db.execute(
            f"CREATE TABLE IF NOT EXISTS {self._quoted} ({', '.join(cols)}) STRICT"
        )

    def __init__(self, db: SQL, cls: type[T]):
        if not is_dataclass(cls):
            raise TypeError(f"{cls.__name__} must be a dataclass")

        self._db = db
        self._cls = cls
        self._table = cls.__name__.lower()
        self._quoted = _quote(self._table)
        self._fields = _get_fields(cls)
        self._field_map = {f.name: f for f in self._fields}
        self._soft_delete = "deleted_at" in self._field_map
        self._create_table()

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
            row[key] = _encode(self._field_map[key], value)
        return row

    def _from_row(self, row: sqlite3.Row) -> T:
        """Convert SQLite row to dataclass instance.

        Args:
            row: SQLite row object.

        Returns:
            Dataclass instance with values from row.
        """
        return self._cls(**{f.name: _decode(f, row[f.name]) for f in self._fields})

    def _records(
        self,
        items: tuple[dict[str, Any] | T, ...],
        kwargs: dict[str, Any],
        id_only: bool = False,
    ) -> list[dict[str, Any]]:
        """Normalize *items and **kwargs into a list of field dicts.

        Args:
            items: Dicts or dataclass instances.
            kwargs: Field values for a single record.
            id_only: If True, read only `id` off dataclass instances and
                require it to be set.

        Returns:
            List of field dicts, kwargs first.

        Raises:
            TypeError: If an item is not a dict or dataclass instance.
            ValueError: If id_only is set and an instance has no id.
        """
        records: list[dict[str, Any]] = [kwargs] if kwargs else []
        for item in items:
            if isinstance(item, dict):
                records.append(item)
            elif is_dataclass(item) and not isinstance(item, type):
                if id_only:
                    item_id = getattr(item, "id", None)
                    if item_id is None:
                        raise ValueError("id required for delete")
                    records.append({"id": item_id})
                else:
                    records.append(
                        {f.name: getattr(item, f.name) for f in self._fields}
                    )
            else:
                raise TypeError(
                    f"Expected dict or {self._cls.__name__}, got {type(item)}"
                )
        return records

    def _where(
        self, filters: dict[str, Any], include_deleted: bool
    ) -> tuple[str, tuple[Any, ...]]:
        """Build a WHERE clause from field filters.

        Args:
            filters: Field name/value pairs to match.
            include_deleted: If False, also exclude soft-deleted rows.

        Returns:
            Tuple of (clause including leading " WHERE", parameters).
        """
        row = self._to_row(**filters)
        conditions = [f"{_quote(k)} = ?" for k in row]
        if self._soft_delete and not include_deleted:
            conditions.append("deleted_at IS NULL")
        if not conditions:
            return "", ()
        return f" WHERE {' AND '.join(conditions)}", tuple(row.values())

    def _fetch_by_ids(self, ids: list[str]) -> list[T]:
        """Read back rows by id with batched ``IN`` queries.

        One SELECT per _BATCH_SIZE ids instead of one per id.

        Args:
            ids: Primary key values, in the order results should come back.

        Returns:
            Matching rows in ids order; ids no longer present are skipped.
        """
        by_id: dict[str, T] = {}
        for chunk in batched(ids, _BATCH_SIZE):
            placeholders = ", ".join("?" * len(chunk))
            rows = self._db.execute(
                f"SELECT * FROM {self._quoted} WHERE id IN ({placeholders})",
                chunk,
            )
            for row in rows:
                by_id[row["id"]] = self._from_row(row)
        return [by_id[i] for i in ids if i in by_id]

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
        stamp = _now()
        rows: list[dict[str, Any]] = []
        for record in self._records(items, kwargs):
            # Auto fields are owned by the library, never taken from input
            row = self._to_row(
                **{k: v for k, v in record.items() if k not in AUTO_FIELDS}
            )
            row["id"] = str(_new_id())
            for name in ("created_at", "updated_at"):
                if name in self._field_map:
                    row[name] = stamp
            rows.append(row)
        if not rows:
            return []

        # One executemany per distinct column set, then one batched read-back,
        # instead of an INSERT + SELECT pair per record.
        groups: dict[tuple[str, ...], list[tuple[Any, ...]]] = {}
        for row in rows:
            groups.setdefault(tuple(row), []).append(tuple(row.values()))

        with self._db._transaction():
            for columns, params in groups.items():
                cols = ", ".join(_quote(c) for c in columns)
                placeholders = ", ".join("?" * len(columns))
                self._db.executemany(
                    f"INSERT INTO {self._quoted} ({cols}) VALUES ({placeholders})",
                    params,
                )
            return self._fetch_by_ids([row["id"] for row in rows])

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
        clause, params = self._where(kwargs, include_deleted)
        sql = f"SELECT * FROM {self._quoted}{clause}"

        # Pagination (1-indexed pages). LIMIT/OFFSET without ORDER BY returns
        # rows in undefined order, so pin it to id: UUIDv7 sorts by insert time.
        if page is not None:
            per_page = max(1, per_page)
            sql += " ORDER BY id LIMIT ? OFFSET ?"
            params = (*params, per_page, (max(1, page) - 1) * per_page)

        return [self._from_row(r) for r in self._db.execute(sql, params)]

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
        stamp = _now()
        updates: list[tuple[dict[str, Any], Any]] = []
        for record in self._records(items, kwargs):
            item_id = record.get("id")
            if item_id is None:
                raise ValueError("id required for update")

            row = self._to_row(
                **{k: v for k, v in record.items() if k not in AUTO_FIELDS}
            )
            # Auto-update timestamp
            if "updated_at" in self._field_map:
                row["updated_at"] = stamp
            if row:
                updates.append((row, item_id))
        if not updates:
            return []

        # One executemany per distinct column set, then one batched read-back,
        # instead of an UPDATE + SELECT pair per record.
        groups: dict[tuple[str, ...], list[tuple[Any, ...]]] = {}
        for row, item_id in updates:
            groups.setdefault(tuple(row), []).append((*row.values(), item_id))

        with self._db._transaction():
            for columns, params in groups.items():
                set_clause = ", ".join(f"{_quote(c)} = ?" for c in columns)
                self._db.executemany(
                    f"UPDATE {self._quoted} SET {set_clause} WHERE id = ?",
                    params,
                )
            return self._fetch_by_ids([item_id for _, item_id in updates])

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
        soft = self._soft_delete and not hard
        # Dataclass instances are matched on id alone
        records = self._records(items, kwargs, id_only=True)

        # No filters given: delete every visible row
        if not records:
            records = [{}]

        results: list[T] = []
        ids: dict[str, None] = {}  # dedups overlapping filters, keeps order
        with self._db._transaction():
            for record in records:
                clause, params = self._where(record, include_deleted=hard)
                for row in self._db.execute(
                    f"SELECT * FROM {self._quoted}{clause}", params
                ):
                    ids[row["id"]] = None
                    results.append(self._from_row(row))

            # One batched write on the matched ids instead of one per filter
            stamp = _now()
            for chunk in batched(ids, _BATCH_SIZE):
                placeholders = ", ".join("?" * len(chunk))
                if soft:
                    self._db.execute(
                        f"UPDATE {self._quoted} SET deleted_at = ? "
                        f"WHERE id IN ({placeholders})",
                        (stamp, *chunk),
                    )
                else:
                    self._db.execute(
                        f"DELETE FROM {self._quoted} WHERE id IN ({placeholders})",
                        chunk,
                    )

        return results

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
        clause, params = self._where(kwargs, include_deleted)
        rows = self._db.execute(f"SELECT COUNT(*) FROM {self._quoted}{clause}", params)
        total = rows[0][0] if rows else 0
        per_page = max(1, per_page)
        return Count(
            total=total,
            pages=(total + per_page - 1) // per_page,
            per_page=per_page,
        )

    def drop(self) -> None:
        """Drop the table from the database."""
        self._db.execute(f"DROP TABLE IF EXISTS {self._quoted}")
        # Evict so the next db(cls) recreates the table
        self._db._tables.pop(self._cls, None)


class SQL:
    """SQLite database instance.

    Create tables by calling the instance with a dataclass. The connection is
    opened on first use and reused, so ":memory:" databases persist for the
    lifetime of the instance. Access is serialized with a lock, making a single
    instance safe to share across threads.

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
        # Reentrant: _transaction holds the lock across nested execute calls
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        self._depth = 0
        # Weak values: a Table holds its SQL, so a strong cache would create a
        # reference cycle and delay closing the connection to the cyclic GC.
        self._tables: MutableMapping[type, Any] = weakref.WeakValueDictionary()

    def _connect(self) -> sqlite3.Connection:
        """Open the connection on first use and apply pragmas.

        Returns:
            The live connection.
        """
        if self._conn is None:
            conn = sqlite3.connect(self.path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            # WAL lets readers run concurrently with a writer; NORMAL trades an
            # fsync per commit for durability only against OS/power loss, not
            # against process crashes. Both are no-ops for ":memory:".
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            self._conn = conn
        return self._conn

    @contextmanager
    def _transaction(self) -> Generator[None]:
        """Group statements into one commit.

        Nests safely: only the outermost block commits or rolls back, so a
        batch write costs one fsync instead of one per statement.

        Yields:
            None.
        """
        with self._lock:
            conn = self._connect()
            self._depth += 1
            try:
                yield
            except BaseException:
                self._depth -= 1
                if self._depth == 0:
                    conn.rollback()
                raise
            self._depth -= 1
            if self._depth == 0:
                conn.commit()

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        """Execute a statement.

        Commits immediately unless a transaction is open, in which case the
        commit is deferred to the end of that transaction.

        Args:
            sql: SQL query string.
            params: Query parameters.

        Returns:
            All rows produced by the statement.
        """
        with self._lock:
            conn = self._connect()
            rows = conn.execute(sql, params).fetchall()
            if self._depth == 0:
                conn.commit()
            return rows

    def executemany(self, sql: str, params: list[tuple[Any, ...]]) -> None:
        """Execute one statement against every parameter tuple.

        A single prepared statement run in C, so a batch write costs one
        statement instead of one Python-level execute per row. Commits like
        execute(): immediately, unless a transaction is open.

        Args:
            sql: SQL statement string.
            params: One parameter tuple per row.
        """
        with self._lock:
            conn = self._connect()
            conn.executemany(sql, params)
            if self._depth == 0:
                conn.commit()

    def close(self) -> None:
        """Close the connection. An in-memory database is discarded."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
            self._tables.clear()

    def __call__[T](self, cls: type[T]) -> Table[T]:
        """Create a table for the given dataclass.

        Tables are cached per class, so repeated calls reuse the same instance
        instead of re-issuing CREATE TABLE.

        Args:
            cls: Dataclass type to create table for.

        Returns:
            Table instance for CRUD operations.
        """
        table = self._tables.get(cls)
        if table is None:
            table = self._tables[cls] = Table(self, cls)
        return table

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        # Release the file handle as soon as the instance is collected, so the
        # database file can be removed without an explicit close().
        conn = getattr(self, "_conn", None)
        if conn is not None:
            conn.close()
