# Changelog

## 0.4.0 (unreleased)

### Fixed

- **Parameterized generics no longer break inserts.** `_unwrap` handled bare
  types and `Optional`, but mangled subscripted ones: `list[str]` collapsed to
  `str`, and `dict[str, Any] | None` stayed `dict[str, Any]`, which is not the
  `dict` that marks a field as JSON. Either way the value was handed to SQLite
  unserialized and the insert raised `sqlite3.ProgrammingError` — and because
  `create()` builds the whole record, one such field broke *every* insert into
  that table, not just the ones that set it. Type parameters now collapse to
  their origin and unions are unwrapped recursively, so `dict[str, Any]`,
  `list[str]`, `list[dict[str, str]] | None`, `Optional[dict[str, int]]`, and
  `Annotated[dict[str, Any], ...]` are all JSON columns, while `int | None`
  stays INTEGER. The workaround of annotating these fields as bare `dict` or
  `list` is no longer needed.

  Column types are unchanged, so no migration is required. The one exception is
  a field annotated as a generic collection that was being used to store plain
  text — it is a JSON column now, and old rows holding non-JSON text will fail
  to decode on read.

- **Ids are now strictly monotonic.** UUIDv7 generation on Python 3.12 and 3.13
  used a millisecond timestamp plus random bits, so ids created inside the same
  millisecond came out in random order. A keyset walk over such a burst could
  return rows out of insert order or skip them entirely — data loss for anyone
  using `read(after=...)` to drain a queue. Generation now carries a 42-bit
  counter ([RFC 9562 §6.2](https://www.rfc-editor.org/rfc/rfc9562#name-monotonicity-and-counters)),
  which also keeps ids increasing across a clock that steps backwards.
  Python 3.14+ used the monotonic `uuid.uuid7()` from the standard library and
  was never affected.
- The `synchronous=NORMAL` durability comment had the trade backwards. WAL plus
  `NORMAL` is durable against a *process* crash; it is an *OS crash or power
  loss* that can roll back recent commits.

### Added

- `SQL(path, synchronous=...)` selects the durability level: `"OFF"`,
  `"NORMAL"` (default), `"FULL"`, or `"EXTRA"`. Keyword-only, validated at
  construction. The `Synchronous` type is exported for annotating settings that
  flow into it.
- `Table.query(sql, params)` runs raw SQL and decodes the rows into model
  instances — the supported path for ranges, `LIKE`, `IN`, aggregates, and
  joins, none of which `read()` covers.
- **Fixtures: `dump()` and `load()`**, on both `SQL` and `Table`, for seed
  data, test fixtures, and backups the way Django's `dumpdata`/`loaddata` work.
  `db.dump("seed.json", User, Post)` writes JSON records shaped
  `{"model": ..., "fields": {...}}`, with `dict`/`list` fields as objects and
  datetimes as ISO strings, so the file is readable and hand-editable;
  `db.load("seed.json", User, Post)` reads them back.

  Auto fields are carried over rather than regenerated, so ids, timestamps, and
  soft-delete tombstones survive a round trip and references between records
  stay valid. A record without an id still gets a fresh one and dataclass
  defaults for omitted fields, which is what makes a hand-written fixture work.
  A load is one transaction across every table in the file, and a record whose
  id already exists overwrites that row unless `replace=False` is passed.
  Omitting the path returns the records instead of writing a file.
- README: a "Fitting Into a Larger System" section on where SQLow belongs as a
  component (outbox and spool buffers, worker checkpoints, per-tenant files,
  edge store-and-forward) and where it does not.

### Changed

- **`create()` now applies dataclass defaults for omitted fields.** Previously
  only instances carried their defaults into storage; dicts and kwargs wrote
  `NULL` for anything omitted, so a field declared `status: str = "pending"`
  could read back as `None` and contradict its own annotation. All three input
  forms now agree, and `default_factory` and `__post_init__` are honored.

  Migration: a model with a field that has **no default** now raises
  `TypeError` on `create()` if that field is not supplied, where it previously
  stored `NULL`. Supply the field, or give it a default.

  `update()` is unchanged and still patches only the fields you name.

### Notes

- `isort` is now configured with `profile = "black"`, so `task fmt` no longer
  produces import formatting that `black` and `ruff` reject.
