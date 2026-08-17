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

    db.dump("seed.json", Component, Project)     # fixtures out
    db.load("seed.json", Component, Project)     # and back in
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
import weakref
from collections.abc import Callable, Generator, Iterable, Mapping, MutableMapping
from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass
from datetime import UTC, date, datetime, time
from itertools import batched
from time import time_ns
from types import NoneType, UnionType
from typing import (
    Any,
    Final,
    Literal,
    Self,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

__all__ = ["SQL", "Count", "Fixtures", "Model", "Synchronous", "Table"]

type Fixtures = str | os.PathLike[str] | Iterable[Mapping[str, Any]]
"""What ``load()`` accepts: a path to a JSON fixture file, or the records
themselves as an iterable of mappings (what ``dump()`` returns)."""

type Synchronous = Literal["OFF", "NORMAL", "FULL", "EXTRA"]
"""Durability levels for ``SQL(synchronous=...)``, in increasing order of cost.

- ``"OFF"``: no syncing. Fastest, and a crash can corrupt the database.
- ``"NORMAL"``: commits survive a process crash, but the most recent ones can
  roll back on an OS crash or power loss. The default.
- ``"FULL"``: one fsync per commit. Nothing acknowledged is lost.
- ``"EXTRA"``: ``"FULL"`` plus a directory sync on journal changes.
"""

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

# Accepted PRAGMA synchronous levels, derived from the annotation so the type
# and the runtime check cannot drift apart. A pragma value cannot be bound as a
# parameter, so the level is checked against this set before interpolation.
SYNC_LEVELS: Final[frozenset[str]] = frozenset(
    get_args(Synchronous.__value__)  # pylint: disable=no-member
)

# Rows per statement in batched IN-clause queries. Old SQLite builds cap bound
# parameters at 999 (SQLITE_MAX_VARIABLE_NUMBER), so stay under that.
_BATCH_SIZE: Final = 500


# Monotonic counter state for _uuid7. A plain timestamp is only millisecond
# resolution, so without a counter a burst of inserts inside one millisecond
# comes out in random order -- which silently breaks the insert ordering that
# read(after=...) walks and ORDER BY id depend on.
_UUID7_LOCK: Final = threading.Lock()
_UUID7_COUNTER_MAX: Final = (1 << 42) - 1
_uuid7_last_ms = -1
_uuid7_counter = 0


def _uuid7() -> uuid.UUID:
    """Generate a UUID version 7 (RFC 9562 sections 5.7 and 6.2).

    Layout, most significant bit first: a 48-bit big-endian Unix timestamp in
    milliseconds, the 4-bit version, a 42-bit counter occupying rand_a and the
    top of rand_b, the 2-bit variant, and 32 random bits.

    Values are strictly increasing, including within a single millisecond and
    across a clock that steps backwards. The counter is reseeded randomly each
    millisecond with its high bit clear, which leaves room for 2**41 ids in one
    millisecond and keeps successive ids unguessable.

    Returns:
        A time-ordered UUID, strictly greater than every id returned before it.
    """
    # Process-wide state is the point: monotonicity has to hold across every
    # caller, so the counter cannot live on an instance
    global _uuid7_last_ms, _uuid7_counter  # pylint: disable=global-statement

    with _UUID7_LOCK:
        now_ms = time_ns() // 1_000_000
        if now_ms > _uuid7_last_ms:
            _uuid7_last_ms = now_ms
            _uuid7_counter = int.from_bytes(os.urandom(6), "big") >> 7
        else:
            # Same millisecond, or the clock moved backwards: carry on from the
            # last id rather than emitting one that sorts before it
            _uuid7_counter += 1
            if _uuid7_counter > _UUID7_COUNTER_MAX:
                _uuid7_last_ms += 1
                _uuid7_counter = int.from_bytes(os.urandom(6), "big") >> 7
        ts_ms, counter = _uuid7_last_ms, _uuid7_counter

    value = (ts_ms & 0xFFFF_FFFF_FFFF) << 80  # unix_ts_ms
    value |= 0x7 << 76  # ver
    value |= (counter >> 30 & 0xFFF) << 64  # rand_a: counter, high 12 bits
    value |= 0b10 << 62  # var
    value |= (counter & 0x3FFF_FFFF) << 32  # rand_b: counter, low 30 bits
    value |= int.from_bytes(os.urandom(4), "big")  # rand_b: 32 random bits
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


def _table_name(cls: type) -> str:
    """Return the table a dataclass maps to.

    Args:
        cls: Dataclass type.

    Returns:
        The class name, lowercased.
    """
    return cls.__name__.lower()


def _sync_level(value: str) -> Synchronous:
    """Validate and canonicalize a PRAGMA synchronous level.

    Args:
        value: Level name, in any case.

    Returns:
        The level, uppercased.

    Raises:
        ValueError: If the level is not one of SYNC_LEVELS.
    """
    level = value.upper()
    if level not in SYNC_LEVELS:
        raise ValueError(
            f"synchronous must be one of {', '.join(sorted(SYNC_LEVELS))}, "
            f"got {value!r}"
        )
    return cast(Synchronous, level)


def _unwrap(py_type: Any) -> Any:
    """Strip unions and generic parameters down to the type that maps to SQL.

    Parameters carry no storage meaning here -- ``list[str]`` and ``list`` are
    both a JSON column -- so a subscripted annotation collapses to its origin.
    Unions are resolved to their first non-None member and unwrapped in turn,
    which is what makes ``dict[str, Any] | None`` land on ``dict``.

    Args:
        py_type: Python type annotation.

    Returns:
        A bare type suitable for looking up in TYPE_MAP: the origin of a
        generic, the first non-None member of a union, or py_type unchanged.
    """
    origin = get_origin(py_type)
    if origin is None:
        return py_type
    if origin is Union or origin is UnionType:
        return _unwrap(next((a for a in get_args(py_type) if a is not NoneType), str))
    # list[str] -> list, dict[str, Any] -> dict, Annotated[dict, ...] -> dict
    return origin


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


def _jsonable(value: Any) -> Any:
    """Convert a decoded field value into a JSON-serializable one.

    Args:
        value: A Python value read off a model instance or a decoded row.

    Returns:
        The value, with datetime/date/time rendered as ISO strings. Everything
        else is already JSON-native: dict and list fields are decoded objects
        rather than the TEXT they are stored as.
    """
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    return value


def _read_fixtures(source: Fixtures) -> list[dict[str, Any]]:
    """Collect fixture records from a JSON file or an in-memory iterable.

    Args:
        source: Path to a JSON file holding a list of records, or the records
            themselves.

    Returns:
        One dict per record, in file order.

    Raises:
        TypeError: If the file does not hold a JSON list, or a record is not
            a mapping.
    """
    if isinstance(source, (str, os.PathLike)):
        with open(source, encoding="utf-8") as handle:
            data: Any = json.load(handle)
        if not isinstance(data, list):
            raise TypeError(f"{os.fspath(source)} must hold a JSON list of records")
        items: Iterable[Any] = data
    else:
        items = source

    records: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, Mapping):
            raise TypeError(f"Expected a fixture record, got {type(item)}")
        records.append(dict(item))
    return records


def _write_fixtures(
    path: str | os.PathLike[str], records: list[dict[str, Any]], indent: int | None
) -> None:
    """Write fixture records to a JSON file, replacing what is there.

    Args:
        path: Destination file.
        records: Records to serialize.
        indent: Indentation passed to ``json.dump``; None writes one line.
    """
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=indent, ensure_ascii=False)
        handle.write("\n")


def _record_fields(record: dict[str, Any], table: str) -> dict[str, Any]:
    """Unwrap one fixture record down to its field values.

    Both shapes are accepted: the wrapped ``{"model": ..., "fields": {...}}``
    that ``dump()`` writes, and a bare dict of field values.

    Args:
        record: One fixture record.
        table: Table the record is being loaded into.

    Returns:
        The record's field values.

    Raises:
        ValueError: If a wrapped record names a different model.
    """
    inner = record.get("fields")
    model = record.get("model")
    if isinstance(inner, dict) and isinstance(model, str):
        if model != table:
            raise ValueError(f"Record belongs to model {model!r}, not {table!r}")
        return dict(inner)
    return record


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
        return {f.name: _jsonable(getattr(self, f.name)) for f in fields(self)}

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
        """Create table if it does not exist, then add any new columns.

        New tables are STRICT, so SQLite rejects values that do not match the
        declared column type rather than storing them as-is. Tables created by
        earlier versions keep their original, non-STRICT schema.

        Fields added to the dataclass after the table was created become new
        nullable columns; existing rows read back None for them. Renames,
        removals, and type changes are not handled.
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
        existing = {
            row["name"]
            for row in self._db.execute(f"PRAGMA table_info({self._quoted})")
        }
        for f in self._fields:
            if f.name not in existing:
                self._db.execute(
                    f"ALTER TABLE {self._quoted} "
                    f"ADD COLUMN {_quote(f.name)} {f.sql_type}"
                )
        # No index on deleted_at: it is low-cardinality, and without ANALYZE
        # statistics SQLite prefers it over the primary key for
        # "deleted_at IS NULL AND id > ?", which forces a temp B-tree sort and
        # makes keyset reads ~80x slower. A full scan is the better plan here.

    def __init__(self, db: SQL, cls: type[T]):
        if not is_dataclass(cls):
            raise TypeError(f"{cls.__name__} must be a dataclass")

        self._db = db
        self._cls = cls
        self._table = _table_name(cls)
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

    def _defaults(self, given: dict[str, Any]) -> dict[str, Any]:
        """Fill in dataclass defaults for fields the caller left out.

        Every insert is a whole record, so it goes through the dataclass
        constructor: ``default``, ``default_factory``, and ``__post_init__``
        then behave exactly as they do anywhere else, and a stored row can
        never contradict its own annotations with an unasked-for NULL.

        Args:
            given: Field values supplied by the caller, auto fields removed.

        Returns:
            Values for every non-auto field on the model.

        Raises:
            KeyError: If a supplied name is not a field.
            TypeError: If a field without a default was not supplied.
        """
        # Checked before constructing, so an unknown name still raises KeyError
        # rather than the dataclass's TypeError
        for name in given:
            if name not in self._field_map:
                raise KeyError(f"Unknown field: {name}")
        instance = self._cls(**given)
        return {
            f.name: getattr(instance, f.name)
            for f in self._fields
            if f.name not in AUTO_FIELDS
        }

    def create(self, *items: dict[str, Any] | T, **kwargs: Any) -> list[T]:
        """Insert records into the table.

        Omitted fields fall back to the dataclass default rather than NULL, so
        a created row round-trips as a valid instance of the model.

        Args:
            *items: Dicts or dataclass instances to insert.
            **kwargs: Field values for single record insert.

        Returns:
            List of created items with auto-generated IDs.

        Raises:
            KeyError: If a field name is unknown.
            TypeError: If item is not a dict or dataclass instance, or if the
                model has a field with no default and it was not supplied.

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
                **self._defaults(
                    {k: v for k, v in record.items() if k not in AUTO_FIELDS}
                )
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
        order_by: str | None = None,
        after: str | None = None,
        **kwargs: Any,
    ) -> list[T]:
        """Select records from the table.

        Excludes soft-deleted records by default.

        Two pagination styles: page= counts rows from the start (simple, but
        cost grows with the page number), after= seeks past the given id in
        id order (constant cost at any depth — use it to walk big tables).

        Args:
            include_deleted: If True, include soft-deleted records.
            page: Page number (1-indexed) for pagination.
            per_page: Records per page. Defaults to 10.
            order_by: Field to sort by; prefix with "-" for descending.
            after: Keyset cursor: return up to per_page rows with id greater
                than this, in id order. Pass the last id of the previous
                batch; an empty result means the walk is done.
            **kwargs: Field filters (e.g., name="Alice").

        Returns:
            List of matching records, empty list if none found.

        Raises:
            KeyError: If order_by names an unknown field.
            ValueError: If after is combined with page, or with an order_by
                other than "id".

        Example:
            >>> table.read()                      # all non-deleted
            >>> table.read(id="abc")              # by id
            >>> table.read(page=1, per_page=20)   # paginated
            >>> table.read(order_by="-created_at")  # newest first
            >>> table.read(after=last.id, per_page=500)  # keyset walk
        """
        clause, params = self._where(kwargs, include_deleted)

        # Keyset pagination is pinned to id order; UUIDv7 makes that insert
        # order, so a walk sees each pre-existing row exactly once.
        if after is not None:
            if page is not None:
                raise ValueError("after and page are mutually exclusive")
            if order_by not in (None, "id"):
                raise ValueError("after requires id order")
            clause += (" AND " if clause else " WHERE ") + "id > ?"
            params = (*params, after)
            return [
                self._from_row(r)
                for r in self._db.execute(
                    f"SELECT * FROM {self._quoted}{clause} ORDER BY id LIMIT ?",
                    (*params, max(1, per_page)),
                )
            ]

        sql = f"SELECT * FROM {self._quoted}{clause}"

        if order_by is not None:
            desc = order_by.startswith("-")
            field = order_by[1:] if desc else order_by
            if field not in self._field_map:
                raise KeyError(f"Unknown field: {field}")
            sql += f" ORDER BY {_quote(field)}{' DESC' if desc else ''}"

        # Pagination (1-indexed pages). LIMIT/OFFSET without ORDER BY returns
        # rows in undefined order, so pin it to id: UUIDv7 sorts by insert time.
        if page is not None:
            if order_by is None:
                sql += " ORDER BY id"
            per_page = max(1, per_page)
            sql += " LIMIT ? OFFSET ?"
            params = (*params, per_page, (max(1, page) - 1) * per_page)

        return [self._from_row(r) for r in self._db.execute(sql, params)]

    def query(self, sql: str, params: tuple[Any, ...] = ()) -> list[T]:
        """Run raw SQL and decode the rows into model instances.

        The supported escape hatch for everything ``read()`` deliberately does
        not do -- ranges, ``LIKE``, ``IN``, aggregates, joins, ``GROUP BY``.
        Nothing is added to the statement, so soft-deleted rows are included
        unless the query excludes them.

        The projection must cover every field on the model; ``SELECT *`` is the
        straightforward way to guarantee that.

        Args:
            sql: A statement returning rows of this table.
            params: Values bound to the statement's placeholders. Always
                parameterize user input rather than formatting it into sql.

        Returns:
            One model instance per row, in the order the query returned them.

        Raises:
            IndexError: If the projection omits a field of the model.

        Example:
            >>> table.query(
            ...     "SELECT * FROM component WHERE name LIKE ? "
            ...     "AND deleted_at IS NULL ORDER BY created_at DESC",
            ...     ("btn-%",),
            ... )
        """
        return [self._from_row(row) for row in self._db.execute(sql, params)]

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

    # `all` shadows the builtin, but it's the clearest name for the flag
    def delete(  # pylint: disable=redefined-builtin,too-many-locals
        self,
        *items: dict[str, Any] | T,
        hard: bool = False,
        all: bool = False,
        **kwargs: Any,
    ) -> list[T]:
        """Delete records from the table.

        Uses soft delete by default (sets deleted_at timestamp). Refuses to
        run with no filters unless all=True is passed explicitly.

        Args:
            *items: Dicts or dataclass instances to delete.
            hard: If True, permanently delete instead of soft delete.
            all: If True, allow deleting every row when no filters are given.
            **kwargs: Field filters for deletion.

        Returns:
            List of deleted records.

        Raises:
            ValueError: If dataclass instance has no id, or if called with no
                filters and all=False.
            TypeError: If item is not a dict or dataclass instance.

        Example:
            >>> table.delete(id="abc")                    # soft delete
            >>> table.delete(id="abc", hard=True)         # permanent
            >>> table.delete({"id": "a"}, {"id": "b"})    # batch
            >>> table.delete(all=True)                    # every row
        """
        soft = self._soft_delete and not hard
        # Dataclass instances are matched on id alone
        records = self._records(items, kwargs, id_only=True)

        # No filters given: delete every visible row, but only on explicit
        # all=True — a bare delete() is more likely a bug than an intent
        if not records:
            if not all:
                raise ValueError(
                    "delete() without filters would delete every row; "
                    "pass all=True to confirm"
                )
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

    def dump(
        self,
        path: str | os.PathLike[str] | None = None,
        *,
        include_deleted: bool = True,
        indent: int | None = 2,
    ) -> list[dict[str, Any]]:
        """Serialize the table to fixture records.

        Rows come back exactly as stored -- ids, timestamps, and soft-delete
        tombstones included -- so a dump reloads as the same data rather than
        as new records. Values are JSON-native: ``dict`` and ``list`` fields
        are objects, not the TEXT they are stored as, and datetimes are ISO
        strings, so the file stays readable and hand-editable.

        The whole table is materialized in memory. For tables too large for
        that, walk them yourself with ``read(after=...)``.

        Args:
            path: Where to write the JSON file. Omit to only return records.
            include_deleted: If False, leave soft-deleted rows out.
            indent: Indentation for the file; None writes it on one line.

        Returns:
            One ``{"model": ..., "fields": {...}}`` record per row, in id
            order.

        Example:
            >>> users.dump("fixtures/users.json")
            >>> records = users.dump(include_deleted=False)
        """
        clause, params = self._where({}, include_deleted)
        records = [
            {
                "model": self._table,
                "fields": {
                    f.name: _jsonable(_decode(f, row[f.name])) for f in self._fields
                },
            }
            for row in self._db.execute(
                f"SELECT * FROM {self._quoted}{clause} ORDER BY id", params
            )
        ]
        if path is not None:
            _write_fixtures(path, records, indent)
        return records

    def load(self, source: Fixtures, *, replace: bool = True) -> list[T]:
        """Insert fixture records into the table.

        The inverse of ``dump()``: auto fields are taken from the fixture
        instead of being generated, so ids, timestamps, and tombstones survive
        a dump/load round trip and references between records keep pointing at
        the right rows. A record that omits them still gets a fresh id and
        timestamps, which is what makes a hand-written fixture work.

        Records that omit a regular field get the dataclass default, exactly
        like ``create()``. The whole load is one transaction.

        Args:
            source: Path to a JSON fixture file, or the records themselves.
                Both record shapes are accepted: the wrapped
                ``{"model": ..., "fields": {...}}`` that ``dump()`` writes, and
                a bare dict of field values.
            replace: If True, a record whose id already exists overwrites that
                row. If False, it raises ``sqlite3.IntegrityError`` instead.

        Returns:
            The loaded records, read back from the table.

        Raises:
            KeyError: If a record names a field the model does not have.
            TypeError: If a record is not a mapping, or omits a field that has
                no default.
            ValueError: If a wrapped record names a different model.

        Example:
            >>> users.load("fixtures/users.json")
            >>> users.load([{"name": "Alice"}, {"name": "Bob"}])
        """
        stamp = _now()
        rows: list[dict[str, Any]] = []
        for record in _read_fixtures(source):
            given = _record_fields(record, self._table)
            # Checked up front so an unknown name raises here rather than as a
            # confusing TypeError out of the dataclass constructor
            for name in given:
                if name not in self._field_map:
                    raise KeyError(f"Unknown field: {name}")
            values = {
                name: _decode(self._field_map[name], value)
                for name, value in given.items()
            }

            data = self._defaults(
                {k: v for k, v in values.items() if k not in AUTO_FIELDS}
            )
            data["id"] = values.get("id") or str(_new_id())
            for name in ("created_at", "updated_at"):
                if name in self._field_map:
                    data[name] = values.get(name) or stamp
            if self._soft_delete:
                data["deleted_at"] = values.get("deleted_at")
            rows.append(self._to_row(**data))
        if not rows:
            return []

        # Every row carries the full column set, so one executemany covers the
        # batch and REPLACE cannot drop a column back to its default
        columns = tuple(rows[0])
        cols = ", ".join(_quote(c) for c in columns)
        placeholders = ", ".join("?" * len(columns))
        verb = "INSERT OR REPLACE" if replace else "INSERT"
        with self._db._transaction():
            self._db.executemany(
                f"{verb} INTO {self._quoted} ({cols}) VALUES ({placeholders})",
                [tuple(row.values()) for row in rows],
            )
            return self._fetch_by_ids([row["id"] for row in rows])

    def drop(self) -> None:
        """Drop the table from the database."""
        self._db.execute(f"DROP TABLE IF EXISTS {self._quoted}")
        # Evict so the next db(cls) recreates the table, and so a db-wide
        # dump() does not resurrect it as an empty one
        self._db._tables.pop(self._cls, None)
        self._db._models.pop(self._table, None)


class SQL:
    """SQLite database instance.

    Create tables by calling the instance with a dataclass. The connection is
    opened on first use and reused, so ":memory:" databases persist for the
    lifetime of the instance. Access is serialized with a lock, making a single
    instance safe to share across threads.

    Args:
        path: Path to SQLite database file. Use ":memory:" for in-memory DB.
        synchronous: Durability level, keyword-only. Defaults to "NORMAL",
            which skips an fsync per commit; pass "FULL" to survive an OS
            crash or power loss. Lowercase is accepted. See Synchronous.

    Raises:
        ValueError: If synchronous is not a recognized level.

    Example:
        >>> db = SQL("app.db")
        >>> @dataclass
        ... class User(Model):
        ...     name: str = ""
        >>> users = db(User)
        >>> users.create(name="Alice")
        >>> durable = SQL("outbox.db", synchronous="FULL")
    """

    def __init__(self, path: str, *, synchronous: Synchronous = "NORMAL"):
        self.path = path
        self.synchronous = _sync_level(synchronous)
        # Reentrant: _transaction holds the lock across nested execute calls
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        self._depth = 0
        # Weak values: a Table holds its SQL, so a strong cache would create a
        # reference cycle and delay closing the connection to the cyclic GC.
        self._tables: MutableMapping[type, Any] = weakref.WeakValueDictionary()
        # The dataclasses seen so far, by table name. Held strongly, unlike the
        # tables themselves: dump() and load() have to know what a database
        # contains even when the caller kept no Table around.
        self._models: dict[str, type[Any]] = {}

    def __del__(self) -> None:
        # Release the file handle as soon as the instance is collected, so the
        # database file can be removed without an explicit close().
        conn = getattr(self, "_conn", None)
        if conn is not None:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        """Open the connection on first use and apply pragmas.

        Returns:
            The live connection.
        """
        if self._conn is None:
            conn = sqlite3.connect(self.path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            # WAL lets readers run concurrently with a writer. At the default
            # synchronous=NORMAL a commit is durable against a process crash,
            # but the most recent commits can roll back on an OS crash or power
            # loss; synchronous="FULL" fsyncs each commit to close that window.
            # Neither pragma applies to ":memory:".
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute(f"PRAGMA synchronous = {self.synchronous}")
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

    # path comes first to match Table.dump() and load(); the models follow it
    def dump(  # pylint: disable=keyword-arg-before-vararg
        self,
        path: str | os.PathLike[str] | None = None,
        *models: type[Any],
        include_deleted: bool = True,
        indent: int | None = 2,
    ) -> list[dict[str, Any]]:
        """Serialize whole tables to one fixture file.

        Args:
            path: Where to write the JSON file. Pass None to only return the
                records; it is the first argument, so selecting models without
                writing a file reads ``db.dump(None, User)``.
            *models: Dataclasses to dump, in order. With none given, every
                table this instance has opened is dumped, in the order they
                were first used.
            include_deleted: If False, leave soft-deleted rows out.
            indent: Indentation for the file; None writes it on one line.

        Returns:
            The records of every dumped table, tagged with their model name.

        Raises:
            TypeError: If path is not a path, which usually means a model was
                passed in its place.

        Example:
            >>> db.dump("fixtures/seed.json", User, Post)
            >>> db.dump("fixtures/all.json")   # every table db() has opened
            >>> records = db.dump(None, User)  # no file
        """
        if path is not None and not isinstance(path, (str, os.PathLike)):
            raise TypeError(
                f"dump() path must be a path or None, got {type(path)}; "
                "models go after it"
            )
        chosen = list(models) or list(self._models.values())
        tables = [self(model) for model in chosen]
        records = [
            record
            for table in tables
            for record in table.dump(include_deleted=include_deleted)
        ]
        if path is not None:
            _write_fixtures(path, records, indent)
        return records

    def load(
        self, source: Fixtures, *models: type[Any], replace: bool = True
    ) -> dict[str, list[Any]]:
        """Load a fixture file across tables.

        Records are routed to a table by their ``"model"`` name and loaded in
        one transaction, so a fixture that fails partway leaves the database
        untouched. See ``Table.load`` for what happens to each record.

        Args:
            source: Path to a JSON fixture file, or the records themselves.
            *models: Dataclasses the fixture refers to. Models this instance
                has already opened a table for are found without being listed.
            replace: If True, a record whose id already exists overwrites that
                row. If False, it raises ``sqlite3.IntegrityError`` instead.

        Returns:
            The loaded records, read back from their tables, keyed by model
            name.

        Raises:
            KeyError: If a record names a model with no matching table.
            TypeError: If a record has no ``"model"`` name.

        Example:
            >>> db.load("fixtures/seed.json", User, Post)
            >>> db.load(db.dump())   # round trip
        """
        records = _read_fixtures(source)
        for model in models:
            self(model)

        groups: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            model_name = record.get("model")
            if not isinstance(model_name, str):
                raise TypeError(
                    'Fixture record has no "model" name; load a file of bare '
                    "field dicts into a single table with Table.load() instead"
                )
            groups.setdefault(model_name, []).append(record)

        # Resolved before the transaction opens: an unknown model should fail
        # without writing anything, and opening a table is DDL that a rollback
        # would undo behind the cache's back
        tables = []
        for name in groups:
            cls = self._models.get(name)
            if cls is None:
                raise KeyError(f"Unknown model: {name}. Pass its dataclass to load().")
            tables.append((name, self(cls)))

        loaded: dict[str, list[Any]] = {}
        with self._transaction():
            for name, table in tables:
                loaded[name] = table.load(groups[name], replace=replace)
        return loaded

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
            self._models[_table_name(cls)] = cls
        return table

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
