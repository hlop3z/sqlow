# SQLow

[![PyPI](https://img.shields.io/pypi/v/sqlow)](https://pypi.org/project/sqlow/)
[![Python](https://img.shields.io/pypi/pyversions/sqlow)](https://pypi.org/project/sqlow/)
[![Tests](https://github.com/hlop3z/sqlow/actions/workflows/test.yml/badge.svg)](https://github.com/hlop3z/sqlow/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/hlop3z/sqlow/graph/badge.svg)](https://codecov.io/gh/hlop3z/sqlow)
[![License](https://img.shields.io/pypi/l/sqlow)](https://github.com/hlop3z/sqlow/blob/main/LICENSE)

Dataclass-native SQLite. Zero boilerplate CRUD.

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

## Why SQLow?

- **Zero boilerplate** - Define a dataclass, get a database
- **100% typed** - Full type hints, mypy strict compatible
- **100% tested** - Complete test coverage
- **Standard library only** - No dependencies beyond Python
- **JSON-friendly** - Returns dataclass instances (easy `asdict()` for JSON)

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

| Field        | Type          | Behavior                       |
| ------------ | ------------- | ------------------------------ |
| `id`         | `str`         | UUID, auto-generated on create |
| `created_at` | `str`         | ISO timestamp, set on create   |
| `updated_at` | `str`         | ISO timestamp, set on update   |
| `deleted_at` | `str \| None` | ISO timestamp, set on delete   |

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

## Use Cases

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

### Prototyping & MVPs

Get a working backend in minutes. Migrate to Postgres later.

```python
# Flask + SQL
@app.post("/users")
def create_user(name: str):
    return asdict(users.create(name=name)[0])

@app.get("/users")
def list_users(page: int = 1):
    return [asdict(u) for u in users.read(page=page)]
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

### Audit Logs & Event Sourcing

Track changes with timestamps built-in.

```python
@dataclass
class AuditLog(Model):
    user_id: str = ""
    action: str = ""
    resource: str = ""
    details: dict | None = None

logs = SQL("audit.db")(AuditLog)
logs.create(user_id=user.id, action="delete", resource="project:123")
# created_at automatically set
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
