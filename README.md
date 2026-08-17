# SQLow

[![PyPI](https://img.shields.io/pypi/v/sqlow)](https://pypi.org/project/sqlow/)
[![Tests](https://github.com/hlop3z/sqlow/actions/workflows/test.yml/badge.svg)](https://github.com/hlop3z/sqlow/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/hlop3z/sqlow/graph/badge.svg)](https://codecov.io/gh/hlop3z/sqlow)
[![License](https://img.shields.io/pypi/l/sqlow)](https://github.com/hlop3z/sqlow/blob/main/LICENSE)

Dataclass-native SQLite. **JSON-file experience with database-grade durability**.

```python
from dataclasses import dataclass
from sqlow import SQL, Model

db = SQL("app.db")

@dataclass
class Task(Model):
    title: str = ""
    done: bool = False

tasks = db(Task)
tasks.create(title="Build something")
```

## Install

```sh
pip install sqlow
```

Requires Python 3.12+. No dependencies.

## Why SQLow?

SQLow replaces the JSON or pickle file, not your database layer. When a script
or app has record-shaped data, you define a dataclass and get durable, typed
storage — without writing any persistence code.

- **Zero boilerplate** - Define a dataclass, get a database
- **100% typed** - Full type hints, mypy strict compatible
- **100% tested** - Complete test coverage
- **Standard library only** - No dependencies beyond Python
- **JSON-friendly** - Returns dataclass instances (easy `to_dict()` and `from_dict()` for JSON)

## vs a JSON File

If you are replacing `json.dump`/`json.load`, the honest answer is that it
depends on your access pattern. Ratios below are from records of five small
fields; treat them as orders of magnitude, not exact figures.

**Incremental writes — SQLow wins by orders of magnitude**

| Operation             |     1K |    10K |       100K |
| --------------------- | -----: | -----: | ---------: |
| Append one record     |    94x |   900x |     7,546x |
| Update one field      |   104x | 1,016x |    11,570x |
| Cold read one by `id` |     2x |    22x |       273x |

Changing one field in a JSON file means re-serializing and rewriting *every*
record: 640 ms at 100K records versus 0.06 ms for an `UPDATE`. The gap grows
linearly with your data, so it is not something you outgrow gracefully. Reading
one record cold is the same story — JSON parses the whole file to find it.

**In-memory work — the JSON file wins**

| Operation                  |   1K |  10K |               100K |
| -------------------------- | ---: | ---: | -----------------: |
| Cold load of all records   |   3x |   3x | 3x (150 vs 434 ms) |
| Warm lookup by `id`        | 150x | 145x |               141x |
| Filter by field, in memory |   2x |   2x |  4x (2.9 vs 12 ms) |

`json.load` is a tight C loop, and once it has run a `dict[id]` lookup costs
essentially nothing, while SQLow still pays a query round trip plus dataclass
construction on every call. Building 100K instances is what makes a full
`read()` slower than the equivalent parse.

So: **load once and query in RAM, and a plain dict is faster.** **Read and
write as you go, and SQLow wins by 100x to 10,000x.**

**What the timings leave out**

- **Crash safety.** A JSON rewrite has a window where a crash leaves a
  truncated file and no data at all. Doing it safely (temp file, `fsync`,
  atomic replace) costs *more* than the naive version measured above — 8.1 ms
  vs 6.5 ms at 1K, 656 ms vs 641 ms at 100K — and still gives you no rollback
  on partial failure. SQLow gets transactions and WAL for free.
- **Concurrency.** Two writers racing on one JSON file corrupt it, and there is
  no locking to opt into. SQLow serializes threads on an instance, and SQLite
  locks the file across processes, so a concurrent write is slow rather than
  destructive. Throughput is still one writer at a time.
- **Memory.** JSON needs the whole dataset resident. `read(after=...)` walks a
  table of any size in constant memory at ~2 ms per 1000 rows.
- **Disk.** Comparable: at 100K records, 23.3 MB of JSON vs 20.9 MB closed.
  A live database with an un-checkpointed WAL can be larger (41.9 MB here)
  until it checkpoints or closes.

See [Scale](#scale) for absolute numbers at 1M rows.

## When to Use Something Else

SQLow is deliberately small: everything it has hardens the small case, and
everything it lacks is what the large cases need. Reach for an ORM
([SQLAlchemy](https://www.sqlalchemy.org/), [SQLModel](https://sqlmodel.tiangolo.com/),
[peewee](https://docs.peewee-orm.com/)) or raw `sqlite3` when you need:

- **Relations** - There are no joins and no enforced foreign keys
- **Rich queries** - `read()` filters by equality only. Ranges, `LIKE`, and
  aggregates mean writing the SQL yourself via [`query()`](#raw-sql)
- **Schema migrations** - New dataclass fields become new columns automatically,
  but renames, removals, and type changes are on you
- **Multi-process or server workloads** - One connection per instance,
  serialized by a lock: a single writer at a time, by design
- **Async services** - `sqlite3` calls are blocking, so an `await`-free query
  under the lock stalls the event loop; wrap calls in `asyncio.to_thread`

See [Fitting Into a Larger System](#fitting-into-a-larger-system) for where it
does belong once your architecture has more than one moving part.

## API

### Define Tables

Inherit from `Model` to get auto-managed fields:

```python
from dataclasses import dataclass
from sqlow import SQL, Model

db = SQL("app.db")

@dataclass
class User(Model):
    # Model provides: id, created_at, updated_at, deleted_at
    name: str = ""
    email: str = ""
    active: bool = True
    meta: dict | None = None  # JSON field
    tags: list | None = None  # JSON field

users = db(User)
```

### CRUD Operations

All operations return `list[T]` for consistency:

```python
# Create
users.create(name="Alice", email="alice@example.com")
users.create({"name": "Bob"}, {"name": "Charlie"})  # batch
users.create(User(name="Dana"))                     # from an instance

# Omitted fields take the dataclass default, so these three agree:
users.create(name="Bob")            # active -> True
users.create({"name": "Bob"})       # active -> True
users.create(User(name="Bob"))      # active -> True

# Read
users.read()                        # all
users.read(id="abc-123")            # by id
users.read(name="Alice")            # by field
users.read(page=1, per_page=10)     # paginated
users.read(order_by="name")         # sorted
users.read(order_by="-created_at")  # sorted descending

# Filters match by equality only; multiple filters AND together.
# For anything richer, use query() -- see Raw SQL below.

# Update
users.update(id="abc-123", name="Alicia")
users.update({"id": "a", "name": "A"}, {"id": "b", "name": "B"})  # batch

# Delete (soft by default)
users.delete(id="abc-123")            # soft delete
users.delete(id="abc-123", hard=True) # permanent
users.delete({"id": "a"}, {"id": "b"})  # batch delete
users.delete(all=True)                # every row; refuses without all=True
```

### Create Fills In Defaults, Update Patches

`create()` writes a whole record. Fields you omit take their dataclass default,
including `default_factory`, and `__post_init__` runs — so a stored row always
reads back as a valid instance of your model, never one with an unasked-for
`None` in a field typed `str`:

```python
@dataclass
class Task(Model):
    title: str = ""
    status: str = "pending"
    tags: list = field(default_factory=list)

tasks.create(title="Ship it")  # status="pending", tags=[]
```

A field with no default at all is required, and omitting it raises `TypeError`
rather than storing `NULL`.

`update()` is the opposite: a patch. Only the fields you name are written, and
everything else keeps its stored value:

```python
tasks.update(id=task.id, status="done")   # title and tags untouched
```

One sharp edge in that asymmetry: passing an **instance** to `update()` writes
every field off that instance, so it is a full replace, not a patch.

```python
tasks.update(Task(id=task.id, status="done"))  # title reset to "", tags to []!
tasks.update(id=task.id, status="done")        # what you usually want
```

### Model Fields

When you inherit from `Model`, these fields are auto-managed:

| Field        | Type          | Behavior                                |
| ------------ | ------------- | --------------------------------------- |
| `id`         | `str`         | UUIDv7, auto-generated on create        |
| `created_at` | `str`         | ISO timestamp, set on create            |
| `updated_at` | `str`         | ISO timestamp, set on create and update |
| `deleted_at` | `str \| None` | ISO timestamp, set on soft delete       |

### Pagination

```python
# Read paginated results (1-indexed)
page1 = users.read(page=1, per_page=20)
page2 = users.read(page=2, per_page=20)

# Get count info
info = users.count(per_page=20)
info.total    # 42
info.pages    # 3
info.per_page # 20
```

`page=` counts rows from the start, so the cost grows with the page number.
To walk a large table, use the keyset cursor instead — `after=` seeks past an
id directly, so every batch costs the same regardless of depth:

```python
batch = users.read(page=1, per_page=500)
while batch:
    process(batch)
    batch = users.read(after=batch[-1].id, per_page=500)
```

Rows come back in `id` order (UUIDv7, so insert order), and filters compose
as usual.

### Scale

Measured on 1M rows (single field-value filter, 178 MB database):

| Operation                        | Cost                                  |
| -------------------------------- | ------------------------------------- |
| `read(after=..., per_page=1000)` | ~2 ms per batch, flat at any depth    |
| full keyset walk of 1M rows      | ~2.2 s                                |
| `read(page=900, per_page=1000)`  | ~750 ms — grows with the page number  |
| `read(name="x")`                 | ~110 ms — full scan, no field indexes |
| `count()`                        | ~100 ms                               |
| `read()` (materialize 1M rows)   | ~2.7 s                                |

Sequential walking stays cheap into the millions. Filtering does not: only
`id` is indexed, so every `read(field=value)` scans the table regardless of
how many rows match. That is the real ceiling — up to ~100K rows keeps every
operation in the single-digit-to-low-tens of milliseconds, which is the scale
SQLow is built for. Beyond that, add your own indexes with `db.execute()`:

```python
db.execute("CREATE INDEX IF NOT EXISTS idx_user_email ON user (email)")
```

### Raw SQL

`read()` filters by equality only. Everything else — ranges, `LIKE`, `IN`,
aggregates, joins, `GROUP BY` — goes through `query()`, which runs your
statement and decodes the rows back into model instances:

```python
recent = users.query(
    'SELECT * FROM "user" WHERE created_at > ? AND deleted_at IS NULL '
    "ORDER BY created_at DESC LIMIT 20",
    (cutoff,),
)
# -> list[User], with JSON, bool, and datetime fields decoded as usual
```

Two things to keep in mind. Nothing is added to your statement, so soft-deleted
rows are included unless you exclude them yourself. And the projection has to
cover every field on the model — `SELECT *` is the easy way to be sure.

Always pass values as `params` rather than formatting them into the string:

```python
users.query('SELECT * FROM "user" WHERE email LIKE ?', (f"%@{domain}",))
```

For statements that do not return model rows — DDL, `PRAGMA`, aggregates,
writes — use `db.execute()`, which returns raw `sqlite3.Row` objects.

### Soft Delete

Records are soft-deleted by default (sets `deleted_at`):

```python
users.delete(id="abc-123")              # soft delete
users.read()                            # excludes deleted
users.read(include_deleted=True)        # includes deleted
users.delete(id="abc-123", hard=True)   # permanent delete
```

Deleting with no filters raises `ValueError` unless you opt in explicitly:

```python
users.delete(all=True)             # soft delete every row
users.delete(all=True, hard=True)  # empty the table permanently
```

Soft-deleted rows still occupy the table, and reads filter them out with a
scan. `deleted_at` is deliberately *not* indexed: it has two distinct values,
and without `ANALYZE` statistics SQLite will prefer that index over the primary
key for keyset reads, which throws away the id ordering and makes them roughly
80x slower. Purge with `delete(all=True, hard=True)` or a filtered hard delete
if deleted rows dominate the table.

### Multiple Tables

One database, multiple tables:

```python
db = SQL("app.db")

@dataclass
class User(Model):
    name: str = ""

@dataclass
class Post(Model):
    title: str = ""
    user_id: str = ""

users = db(User)
posts = db(Post)
```

### Schema Changes

Adding a field to a dataclass adds the column to an existing table
automatically — no migration step:

```python
@dataclass
class User(Model):
    name: str = ""
    nickname: str = ""  # new field: column is added on next db(User)
```

Rows written before the change read back `None` for the new column. Renames,
removals, and type changes are not handled; for those, migrate by hand with
`db.execute()` or recreate the table.

### Type Support

| Python Type | SQLite Type | Notes           |
| ----------- | ----------- | --------------- |
| `str`       | TEXT        |                 |
| `int`       | INTEGER     |                 |
| `float`     | REAL        |                 |
| `bool`      | INTEGER     | Stored as 0/1   |
| `dict`      | TEXT        | JSON serialized |
| `list`      | TEXT        | JSON serialized |
| `datetime`  | TEXT        | ISO format, UTC |
| `date`      | TEXT        | ISO format      |
| `time`      | TEXT        | ISO format      |

Type parameters and `| None` do not change how a field is stored, so write the
annotation you actually mean:

```python
@dataclass
class Node(Model):
    properties: dict[str, Any] = field(default_factory=dict)  # JSON column
    labels: list[str] = field(default_factory=list)           # JSON column
    edges: list[dict[str, str]] | None = None                 # JSON column
    weight: int | None = None                                 # INTEGER column
```

`Optional[...]` and `Annotated[...]` resolve the same way. Anything else —
`bytes`, `set`, `tuple`, a nested dataclass — has no mapping: the column is
created as TEXT, and the insert fails when SQLite is handed a value it cannot
bind. Serialize those yourself, or hold them in a `dict` or `list` field.

### Datetime Support

Native support for `datetime`, `date`, and `time` types. Datetimes are always stored in UTC:

```python
from datetime import datetime, date, time

@dataclass
class Event(Model):
    title: str = ""
    starts_at: datetime | None = None
    event_date: date | None = None
    event_time: time | None = None

events = db(Event)
events.create(title="Meeting", starts_at=datetime.now())  # Stored as UTC
```

### JSON Serialization

Use `to_dict()` and `from_dict()` for JSON-safe roundtrips:

```python
import json

# Serialize
users = db(User)
data = users.read()
json.dumps([u.to_dict() for u in data])  # datetime -> ISO string

# Deserialize
user = User.from_dict({"name": "Alice", "starts_at": "2024-06-15T10:30:00+00:00"})
```

### Fixtures

`dump()` and `load()` move whole tables in and out of JSON files, the way
Django's `dumpdata` and `loaddata` do — for seed data, test fixtures, backups,
or moving a database between machines:

```python
db.dump("fixtures/seed.json", User, Post)   # selected tables
db.dump("fixtures/all.json")                # every table db() has opened

db.load("fixtures/seed.json", User, Post)   # -> {"user": [...], "post": [...]}
```

Records are written as `{"model": ..., "fields": {...}}`, with values in their
JSON form — `dict` and `list` fields as objects, datetimes as ISO strings — so
the file stays readable and hand-editable:

```json
[
  {
    "model": "user",
    "fields": {
      "id": "01a00c5b-e413-77b6-8051-481b36527d64",
      "created_at": "2024-06-15T10:30:00+00:00",
      "updated_at": "2024-06-15T10:30:00+00:00",
      "deleted_at": null,
      "name": "Alice",
      "tags": ["admin"]
    }
  }
]
```

**A dump reloads as the same data, not as copies.** Ids, timestamps, and
soft-delete tombstones are carried over rather than regenerated, so
`user_id`-style references keep pointing at the right rows and a round trip is
lossless:

```python
snapshot = db.dump()
fresh = SQL("copy.db")
fresh.load(snapshot, User, Post)
assert fresh.dump(None, User, Post) == snapshot
```

Loading is a single transaction across every table in the file, so a fixture
that fails partway through leaves the database untouched. Records with an id
that already exists overwrite that row; pass `replace=False` to get an
`sqlite3.IntegrityError` instead.

`load()` routes records by their `"model"` name, so it needs the dataclass —
pass it, or open the table with `db(Model)` first. Single tables work the same
way, and there a record can be a bare dict of fields:

```python
users = db(User)
users.dump("fixtures/users.json")
users.load("fixtures/users.json")

# Hand-written fixtures: omitted fields take their dataclass default, and a
# record with no id gets a fresh one, exactly like create()
users.load([{"name": "Alice"}, {"name": "Bob", "active": False}])
```

Both `dump()` methods return the records as well as writing them, and omitting
the path skips the file entirely (`db.dump(None, User)` selects models without
writing one). Soft-deleted rows are included by default; pass
`include_deleted=False` to leave tombstones out. The whole table is
materialized in memory — for tables too large for that, walk them yourself with
`read(after=...)`.

### Connections

`SQL` opens one connection on first use and reuses it, so `":memory:"` databases
persist for the lifetime of the instance. Access is serialized with a lock, so a
single instance can be shared across threads.

The connection is released when the instance is garbage collected. Close it
explicitly when you need the file handle freed at a known point:

```python
db = SQL("app.db")
db.close()

# Or as a context manager
with SQL("app.db") as db:
    users = db(User)
    users.create(name="Alice")
```

File databases run in [WAL mode](https://sqlite.org/wal.html), so readers never
block the writer. WAL keeps `app.db-wal` and `app.db-shm` beside the database;
both are removed on a clean close. Delete them along with the database if you
remove it by hand.

### Durability

The default is `synchronous=NORMAL`: a committed write survives a crash of your
process, but the most recent commits can roll back on an OS crash or power loss.
That is the right trade for caches, scratch state, and anything reconstructible.

When losing the last few commits is unacceptable — a queue, an outbox, an audit
log — pay the fsync:

```python
db = SQL("outbox.db", synchronous="FULL")
```

| Level    | Cost            | Survives process crash | Survives power loss |
| -------- | --------------- | ---------------------- | ------------------- |
| `OFF`    | none            | no                     | no                  |
| `NORMAL` | no fsync/commit | yes                    | no                  |
| `FULL`   | fsync/commit    | yes                    | yes                 |
| `EXTRA`  | fsync + dir     | yes                    | yes                 |

`synchronous` is keyword-only and validated at construction, so a typo fails
where you wrote it rather than at the first query. It is a no-op for
`":memory:"`. The `Synchronous` type is exported if you want to annotate a
setting that flows into it:

```python
from sqlow import SQL, Synchronous

def open_db(path: str, *, durability: Synchronous = "NORMAL") -> SQL:
    return SQL(path, synchronous=durability)
```

### IDs

Primary keys are [UUID version 7](https://www.rfc-editor.org/rfc/rfc9562#name-uuid-version-7)
strings: a millisecond timestamp, then a counter, then random bits. Unlike
random UUIDv4 keys, they are generated in ascending order, which keeps index
inserts sequential and makes `ORDER BY id` chronological.

```python
users.create(name="Alice")   # id="01a00c5b-e413-77b6-8051-481b36527d64"

# Newest last, no extra column needed
for u in users.read(order_by="id"):
    ...
```

Ids are **strictly** increasing, not merely time-ordered: the counter
([RFC 9562 §6.2](https://www.rfc-editor.org/rfc/rfc9562#name-monotonicity-and-counters))
orders ids created inside the same millisecond, and a clock that steps backwards
still cannot produce an id that sorts earlier than one already handed out. That
is what makes `read(after=...)` safe — a millisecond-resolution timestamp alone
would let a burst of inserts come back out of order, and a keyset walk would
skip rows.

`uuid.uuid7()` is used on Python 3.14+; older versions use an equivalent
built-in implementation, so there are still no dependencies.

### Batching

Each call to `create`, `update`, or `delete` runs as a single transaction —
one commit no matter how many records it touches. Pass records together rather
than looping, and the whole batch either lands or rolls back:

```python
# One transaction, one commit
users.create(*[{"name": f"user{i}"} for i in range(1000)])

# A loop is 1000 transactions - much slower
for i in range(1000):
    users.create(name=f"user{i}")
```

### Type Enforcement

New tables are created as [STRICT](https://sqlite.org/stricttables.html), so
values that do not match the declared type are rejected instead of silently
stored. Lossless conversions still apply.

```python
users.create(name="Alice", age="42")     # ok, converted to 42
users.create(name="Alice", age="forty")  # sqlite3.IntegrityError
```

Tables created before this version keep their original non-STRICT schema; no
migration is performed.

## Fitting Into a Larger System

Standalone, SQLow replaces a JSON file. Inside a larger architecture its role is
narrower and more useful: **durable state owned by one process, sitting next to
that process.** Three properties decide every judgment below.

1. **One writer, one connection, lock-serialized.** The file belongs to a single
   process. It is not a shared tier.
2. **UUIDv7 ids plus `read(after=...)`.** An insert-ordered, resumable, constant
   cost cursor over a table of any size — a durable queue primitive.
3. **Stdlib only.** Safe to embed in a library you publish or a container you
   ship, without imposing a dependency tree on anyone.

### Good fits

**Outbox / spool buffer.** A service records events locally; a drainer ships them
upstream and marks them done. Single-writer is not a constraint here — one
process owns the spool by definition.

```python
@dataclass
class Outgoing(Model):
    payload: dict | None = None
    status: str = "pending"
    attempts: int = 0

outbox = SQL("outbox.db", synchronous="FULL")(Outgoing)

outbox.create(payload={"event": "signup"})  # status defaults to "pending"

def drain(batch_size: int = 100):
    cursor = None
    while True:
        batch = (
            outbox.read(status="pending", after=cursor, per_page=batch_size)
            if cursor
            else outbox.read(status="pending", page=1, per_page=batch_size)
        )
        if not batch:
            return
        for row in batch:
            try:
                ship(row.payload)
                outbox.update(id=row.id, status="sent")
            except TransportError:
                outbox.update(id=row.id, attempts=row.attempts + 1)
        cursor = batch[-1].id
```

The cursor is what makes this terminate: a row that keeps failing stays `pending`,
so re-reading from the start would retry it forever instead of moving on.

**Worker checkpoints.** Long ETL, scrape, or training jobs that must resume after
a crash. `read(after=last_id)` picks up exactly where the walk stopped, in
constant memory, without holding a cursor open across the failure.

**Per-tenant or per-unit files.** This turns the single-writer limit into a
scaling axis: contention is per file, so N tenants means N independent writers.
Same pattern for per-project, per-repo, or per-session state.

```python
def db_for(tenant_id: str) -> SQL:
    return SQL(f"data/{tenant_id}.db")
```

**Edge collector with store-and-forward.** Buffer telemetry locally, batch
upstream, survive network partitions. WAL means the shipper reading never blocks
the sampler writing.

**Control-plane metadata inside a tool.** Job registries, run manifests, session
history, plugin indexes — small record-shaped data that must outlive a restart.
Especially good in a published library, where a dependency would be unwelcome.

**Local read cache or projection** of a remote API or upstream database.
`updated_at` is a staleness check and `deleted_at` a tombstone, both free.

**Integration-test double.** `SQL(":memory:")` behind the same repository
interface as your production store: real transactions, real type rejection via
STRICT, no fixtures to tear down.

### Poor fits

| Situation                                     | Why it breaks                                                     |
| --------------------------------------------- | ----------------------------------------------------------------- |
| Shared system of record behind app replicas   | No cross-process pooling; SQLite locking is unsafe on NFS/EFS      |
| Async web service                             | Blocking calls under a lock serialize the event loop              |
| Hot filtered reads past ~100K rows            | Only `id` is indexed, so `read(field=...)` full-scans             |
| Relational core domain                        | No joins, no enforced foreign keys, equality-only filters         |
| Ephemeral container filesystem, no volume      | The durability story disappears with the container                |

### Keeping the exit open

Put it behind a repository interface from the start. `Model` subclasses are
already DTO-shaped through `to_dict()` / `from_dict()`, so the seam is nearly
free and swapping the backend later stays a local change:

```python
class TaskStore(Protocol):
    def add(self, **fields) -> Task: ...
    def pending(self, after: str | None = None) -> list[Task]: ...

class SqlowTaskStore:
    def __init__(self, path: str):
        self._tasks = SQL(path)(Task)

    def add(self, **fields) -> Task:
        return self._tasks.create(**fields)[0]

    def pending(self, after: str | None = None) -> list[Task]:
        return self._tasks.read(status="pending", after=after, per_page=500)
```

## Use Cases

All the places you would otherwise reach for a JSON file:

### CLI Tools & Scripts

```python
@dataclass
class Job(Model):
    command: str = ""
    status: str = "pending"
    output: str = ""

jobs = SQL("jobs.db")(Job)
jobs.create(command="python train.py")
jobs.update(id=job_id, status="completed", output=result)
```

### Local-First Desktop Apps

SQLite ships with the app. No server needed.

```python
@dataclass
class Note(Model):
    title: str = ""
    content: str = ""
    folder_id: str = ""

notes = SQL("~/.myapp/notes.db")(Note)
```

### Internal Tools

Admin panels, data entry, batch processing.

```python
@dataclass
class Customer(Model):
    company: str = ""
    contact: str = ""
    notes: str = ""
    tags: list | None = None

customers = SQL("crm.db")(Customer)
customers.read(page=1, per_page=50)
```

### Per-Tenant Databases

Each customer gets their own SQLite file.

```python
def get_db(tenant_id: str):
    return SQL(f"data/{tenant_id}.db")

db = get_db("acme-corp")
projects = db(Project)
```

### Embedded & Edge

IoT devices, Raspberry Pi, edge computing.

```python
@dataclass
class SensorReading(Model):
    device_id: str = ""
    temperature: float = 0.0
    humidity: float = 0.0

readings = SQL("/var/lib/sensors/data.db")(SensorReading)
readings.create(device_id="sensor-1", temperature=22.5, humidity=45.0)
```

### Test Fixtures

Easy setup and teardown for tests.

```python
@pytest.fixture
def db():
    db = SQL(":memory:")
    users = db(User)
    users.load("tests/fixtures/users.json")  # or users.create(...) inline
    yield users
    # SQLite in-memory DB auto-cleans
```

### Configuration Storage

Replace JSON config files with queryable storage.

```python
@dataclass
class Setting(Model):
    key: str = ""
    value: str = ""
    scope: str = "global"

settings = SQL("config.db")(Setting)
settings.create(key="theme", value="dark", scope="user:123")
settings.read(scope="user:123")
```

### Caching Layer

Local cache for remote API data.

```python
@dataclass
class CachedResponse(Model):
    url: str = ""
    data: dict | None = None
    expires_at: str = ""

cache = SQL("cache.db")(CachedResponse)

def fetch(url: str):
    cached = cache.read(url=url)
    if cached and cached[0].expires_at > now():
        return cached[0].data
    # fetch from remote, cache result
```

## License

MIT
