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

## When to Use Something Else

SQLow is deliberately small: everything it has hardens the small case, and
everything it lacks is what the large cases need. Reach for an ORM
([SQLAlchemy](https://www.sqlalchemy.org/), [SQLModel](https://sqlmodel.tiangolo.com/),
[peewee](https://docs.peewee-orm.com/)) or raw `sqlite3` when you need:

- **Relations** - There are no joins and no enforced foreign keys
- **Rich queries** - Filters are equality-only: no ranges, `LIKE`, aggregates, or custom ordering
- **Schema migrations** - Adding a field to a dataclass does not alter an existing table
- **Multi-process or server workloads** - One connection per instance, serialized by a lock

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

# Read
users.read()                    # all
users.read(id="abc-123")        # by id
users.read(name="Alice")        # by field
users.read(page=1, per_page=10) # paginated

# Filters match by equality only; multiple filters AND together.
# For anything richer, drop down to raw SQL via db.execute().

# Update
users.update(id="abc-123", name="Alicia")
users.update({"id": "a", "name": "A"}, {"id": "b", "name": "B"})  # batch

# Delete (soft by default)
users.delete(id="abc-123")            # soft delete
users.delete(id="abc-123", hard=True) # permanent
users.delete({"id": "a"}, {"id": "b"})  # batch delete
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

### Soft Delete

Records are soft-deleted by default (sets `deleted_at`):

```python
users.delete(id="abc-123")              # soft delete
users.read()                            # excludes deleted
users.read(include_deleted=True)        # includes deleted
users.delete(id="abc-123", hard=True)   # permanent delete
```

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

File databases run in [WAL mode](https://sqlite.org/wal.html) with
`synchronous=NORMAL`, so readers never block the writer. WAL keeps `app.db-wal`
and `app.db-shm` beside the database; both are removed on a clean close. Delete
them along with the database if you remove it by hand.

### IDs

Primary keys are [UUID version 7](https://www.rfc-editor.org/rfc/rfc9562#name-uuid-version-7)
strings: a millisecond timestamp followed by random bits. Unlike random UUIDv4
keys, they are generated in ascending order, which keeps index inserts
sequential and makes `ORDER BY id` chronological.

```python
users.create(name="Alice")   # id="01a00c5b-e413-77b6-8051-481b36527d64"

# Newest last, no extra column needed
for u in sorted(users.read(), key=lambda u: u.id):
    ...
```

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
    users.create({"name": "Alice"}, {"name": "Bob"})
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
