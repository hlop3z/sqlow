"""Tests for sqlow - dataclass-native SQLite CRUD."""

import json
import os
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time
from typing import Annotated, Any, Optional

import pytest

from sqlow import SQL, Count, Model

# Test fixtures
TEST_DB = "test_sqlow.sqlite3"


@dataclass
class Item(Model):
    name: str = ""
    count: int = 0
    price: float = 0.0
    active: bool = False
    meta: dict | None = None
    tags: list | None = None


@dataclass
class Project(Model):
    title: str = ""


@dataclass
class SimpleItem:
    """Dataclass without Model - no auto fields."""

    id: str | None = None
    name: str = ""


@dataclass
class Event(Model):
    """Model with datetime fields for testing."""

    title: str = ""
    starts_at: datetime | None = None
    event_date: date | None = None
    event_time: time | None = None


def _remove_db():
    """Remove the database and its WAL sidecar files."""
    for suffix in ("", "-wal", "-shm"):
        path = TEST_DB + suffix
        if os.path.exists(path):
            os.remove(path)


@pytest.fixture(autouse=True)
def cleanup():
    """Clean up test database before and after each test."""
    _remove_db()
    yield
    _remove_db()


class TestModel:
    """Test Model base class."""

    def test_model_has_auto_fields(self):
        db = SQL(TEST_DB)
        items = db(Item)
        result = items.create(name="test")

        item = result[0]
        assert item.id is not None
        assert item.created_at is not None
        assert item.updated_at is not None
        assert item.deleted_at is None

    def test_created_at_set_on_insert(self):
        db = SQL(TEST_DB)
        items = db(Item)
        result = items.create(name="test")

        assert result[0].created_at is not None
        assert "T" in result[0].created_at  # ISO format

    def test_updated_at_changes_on_update(self):
        db = SQL(TEST_DB)
        items = db(Item)
        added = items.create(name="test")
        original_updated = added[0].updated_at

        # Small delay to ensure timestamp differs
        import time

        time.sleep(0.01)

        updated = items.update(id=added[0].id, name="changed")
        assert updated[0].updated_at != original_updated


class TestSoftDelete:
    """Test soft delete functionality."""

    def test_delete_soft_deletes_by_default(self):
        db = SQL(TEST_DB)
        items = db(Item)
        added = items.create(name="test")

        deleted = items.delete(id=added[0].id)
        assert len(deleted) == 1

        # Should not appear in normal get
        assert items.read(id=added[0].id) == []

    def test_read_excludes_deleted_by_default(self):
        db = SQL(TEST_DB)
        items = db(Item)
        items.create({"name": "keep"}, {"name": "delete"})

        items.delete(name="delete")

        result = items.read()
        assert len(result) == 1
        assert result[0].name == "keep"

    def test_read_include_deleted(self):
        db = SQL(TEST_DB)
        items = db(Item)
        added = items.create(name="test")
        items.delete(id=added[0].id)

        # With include_deleted=True
        result = items.read(include_deleted=True)
        assert len(result) == 1
        assert result[0].deleted_at is not None

    def test_delete_hard_delete(self):
        db = SQL(TEST_DB)
        items = db(Item)
        added = items.create(name="test")

        items.delete(id=added[0].id, hard=True)

        # Should not exist even with include_deleted
        assert items.read(include_deleted=True) == []

    def test_delete_all_soft_delete(self):
        db = SQL(TEST_DB)
        items = db(Item)
        items.create({"name": "a"}, {"name": "b"}, {"name": "c"})

        deleted = items.delete(all=True)
        assert len(deleted) == 3

        # All soft deleted
        assert items.read() == []
        assert len(items.read(include_deleted=True)) == 3


class TestSQL:
    """Test SQL database instance."""

    def test_single_db_multiple_tables(self):
        db = SQL(TEST_DB)
        items = db(Item)
        projects = db(Project)

        items.create(name="button")
        projects.create(title="My Project")

        assert len(items.read()) == 1
        assert len(projects.read()) == 1
        assert items.read()[0].name == "button"
        assert projects.read()[0].title == "My Project"


class TestCreate:
    """Test add() - insert records."""

    def test_create_single_with_kwargs(self):
        db = SQL(TEST_DB)
        items = db(Item)
        result = items.create(name="button", count=5)

        assert len(result) == 1
        assert result[0].id is not None
        assert isinstance(result[0].id, str)
        assert result[0].name == "button"
        assert result[0].count == 5

    def test_create_nothing_returns_empty(self):
        db = SQL(TEST_DB)
        items = db(Item)

        assert items.create() == []

    def test_create_single_with_dict(self):
        db = SQL(TEST_DB)
        items = db(Item)
        result = items.create({"name": "alert", "count": 10})

        assert len(result) == 1
        assert result[0].name == "alert"

    def test_create_single_with_dataclass(self):
        db = SQL(TEST_DB)
        items = db(Item)
        result = items.create(Item(name="modal", count=3))

        assert len(result) == 1
        assert result[0].name == "modal"

    def test_create_multiple(self):
        db = SQL(TEST_DB)
        items = db(Item)
        result = items.create({"name": "a"}, {"name": "b"}, {"name": "c"})

        assert len(result) == 3
        assert result[0].id is not None
        assert result[1].id is not None
        assert result[2].id is not None
        # All IDs should be unique
        ids = [r.id for r in result]
        assert len(set(ids)) == 3

    def test_create_with_json_fields(self):
        db = SQL(TEST_DB)
        items = db(Item)
        result = items.create(
            name="widget",
            meta={"author": "John", "version": 2},
            tags=["ui", "core"],
        )

        assert result[0].meta == {"author": "John", "version": 2}
        assert result[0].tags == ["ui", "core"]

    def test_create_with_bool(self):
        db = SQL(TEST_DB)
        items = db(Item)
        result = items.create(name="toggle", active=True)

        assert result[0].active is True


class TestRead:
    """Test get() - select records."""

    def test_read_all_empty(self):
        db = SQL(TEST_DB)
        items = db(Item)
        result = items.read()

        assert result == []

    def test_read_all(self):
        db = SQL(TEST_DB)
        items = db(Item)
        items.create({"name": "a"}, {"name": "b"})
        result = items.read()

        assert len(result) == 2

    def test_read_by_id(self):
        db = SQL(TEST_DB)
        items = db(Item)
        added = items.create({"name": "a"}, {"name": "b"})
        result = items.read(id=added[1].id)

        assert len(result) == 1
        assert result[0].name == "b"

    def test_read_by_field(self):
        db = SQL(TEST_DB)
        items = db(Item)
        items.create({"name": "button", "count": 5})
        result = items.read(name="button")

        assert len(result) == 1
        assert result[0].count == 5

    def test_read_not_found(self):
        db = SQL(TEST_DB)
        items = db(Item)
        result = items.read(id="nonexistent-id")

        assert result == []

    def test_read_preserves_types(self):
        db = SQL(TEST_DB)
        items = db(Item)
        added = items.create(
            name="test",
            count=42,
            price=3.14,
            active=True,
            meta={"key": "value"},
            tags=[1, 2, 3],
        )
        result = items.read(id=added[0].id)

        item = result[0]
        assert isinstance(item, Item)
        assert isinstance(item.id, str)
        assert isinstance(item.count, int)
        assert isinstance(item.price, float)
        assert item.active is True
        assert isinstance(item.meta, dict)
        assert isinstance(item.tags, list)


class TestUpdate:
    """Test set() - update records."""

    def test_update_single(self):
        db = SQL(TEST_DB)
        items = db(Item)
        added = items.create(name="old")
        result = items.update(id=added[0].id, name="new")

        assert len(result) == 1
        assert result[0].name == "new"

    def test_update_multiple_fields(self):
        db = SQL(TEST_DB)
        items = db(Item)
        added = items.create(name="item", count=0, active=False)
        result = items.update(id=added[0].id, count=10, active=True)

        assert result[0].count == 10
        assert result[0].active is True

    def test_update_batch(self):
        db = SQL(TEST_DB)
        items = db(Item)
        added = items.create({"name": "a"}, {"name": "b"})
        result = items.update(
            {"id": added[0].id, "name": "x"}, {"id": added[1].id, "name": "y"}
        )

        assert len(result) == 2
        assert result[0].name == "x"
        assert result[1].name == "y"

    def test_update_requires_id(self):
        db = SQL(TEST_DB)
        items = db(Item)
        items.create(name="test")

        with pytest.raises(ValueError, match="id required"):
            items.update(name="new")

    def test_update_json_field(self):
        db = SQL(TEST_DB)
        items = db(Item)
        added = items.create(name="item", meta={"a": 1})
        result = items.update(id=added[0].id, meta={"b": 2})

        assert result[0].meta == {"b": 2}


class TestDelete:
    """Test rm() - delete records."""

    def test_delete_by_id(self):
        db = SQL(TEST_DB)
        items = db(Item)
        added = items.create({"name": "a"}, {"name": "b"})
        deleted = items.delete(id=added[0].id)

        assert len(deleted) == 1
        assert deleted[0].name == "a"
        assert len(items.read()) == 1

    def test_delete_by_field(self):
        db = SQL(TEST_DB)
        items = db(Item)
        items.create({"name": "keep"}, {"name": "delete"})
        deleted = items.delete(name="delete")

        assert len(deleted) == 1
        assert items.read()[0].name == "keep"

    def test_delete_all(self):
        db = SQL(TEST_DB)
        items = db(Item)
        items.create({"name": "a"}, {"name": "b"}, {"name": "c"})
        deleted = items.delete(all=True)

        assert len(deleted) == 3
        assert items.read() == []

    def test_delete_without_filters_raises(self):
        db = SQL(TEST_DB)
        items = db(Item)
        items.create(name="keep")

        with pytest.raises(ValueError, match="all=True"):
            items.delete()

        # Nothing was deleted
        assert len(items.read()) == 1

    def test_delete_not_found(self):
        db = SQL(TEST_DB)
        items = db(Item)
        deleted = items.delete(id="nonexistent-id")

        assert deleted == []

    def test_delete_batch(self):
        db = SQL(TEST_DB)
        items = db(Item)
        added = items.create({"name": "a"}, {"name": "b"}, {"name": "c"})

        deleted = items.delete({"id": added[0].id}, {"id": added[1].id})
        assert len(deleted) == 2
        assert len(items.read()) == 1
        assert items.read()[0].name == "c"

    def test_delete_batch_with_dataclass(self):
        db = SQL(TEST_DB)
        items = db(Item)
        added = items.create({"name": "a"}, {"name": "b"})

        deleted = items.delete(Item(id=added[0].id), Item(id=added[1].id))
        assert len(deleted) == 2
        assert items.read() == []

    def test_delete_batch_hard(self):
        db = SQL(TEST_DB)
        items = db(Item)
        added = items.create({"name": "a"}, {"name": "b"})

        deleted = items.delete({"id": added[0].id}, {"id": added[1].id}, hard=True)
        assert len(deleted) == 2
        assert items.read(include_deleted=True) == []

    def test_delete_invalid_type_raises(self):
        db = SQL(TEST_DB)
        items = db(Item)

        with pytest.raises(TypeError, match="Expected dict"):
            items.delete("invalid")  # type: ignore

    def test_delete_dataclass_without_id_raises(self):
        db = SQL(TEST_DB)
        items = db(Item)

        with pytest.raises(ValueError, match="id required"):
            items.delete(Item(name="test"))

    def test_delete_all_empty_table(self):
        db = SQL(TEST_DB)
        items = db(Item)

        # Delete all on empty table returns empty list
        deleted = items.delete(all=True)
        assert deleted == []


class TestPagination:
    """Test pagination functionality."""

    def test_read_with_page(self):
        db = SQL(TEST_DB)
        items = db(Item)
        items.create(
            {"name": "a"}, {"name": "b"}, {"name": "c"}, {"name": "d"}, {"name": "e"}
        )

        result = items.read(page=1, per_page=3)
        assert len(result) == 3

    def test_read_pagination(self):
        db = SQL(TEST_DB)
        items = db(Item)
        items.create(
            {"name": "a"}, {"name": "b"}, {"name": "c"}, {"name": "d"}, {"name": "e"}
        )

        page1 = items.read(page=1, per_page=2)
        page2 = items.read(page=2, per_page=2)
        page3 = items.read(page=3, per_page=2)

        assert len(page1) == 2
        assert len(page2) == 2
        assert len(page3) == 1  # Only 1 left

    def test_read_default_per_page(self):
        db = SQL(TEST_DB)
        items = db(Item)
        # Add 15 items
        for i in range(15):
            items.create(name=f"item-{i}")

        # Default per_page is 10
        page1 = items.read(page=1)
        assert len(page1) == 10

    def test_count_returns_object(self):
        db = SQL(TEST_DB)
        items = db(Item)
        items.create({"name": "a"}, {"name": "b"}, {"name": "c"})

        info = items.count()
        assert isinstance(info, Count)
        assert info.total == 3
        assert info.pages == 1
        assert info.per_page == 10

    def test_count_calculates_pages(self):
        db = SQL(TEST_DB)
        items = db(Item)
        for i in range(25):
            items.create(name=f"item-{i}")

        info = items.count(per_page=10)
        assert info.total == 25
        assert info.pages == 3

    def test_count_with_filter(self):
        db = SQL(TEST_DB)
        items = db(Item)
        items.create(
            {"name": "a", "count": 1},
            {"name": "b", "count": 2},
            {"name": "c", "count": 1},
        )

        assert items.count(count=1).total == 2
        assert items.count(count=2).total == 1

    def test_count_excludes_deleted(self):
        db = SQL(TEST_DB)
        items = db(Item)
        items.create({"name": "a"}, {"name": "b"}, {"name": "c"})
        items.delete(name="c")

        assert items.count().total == 2
        assert items.count(include_deleted=True).total == 3


class TestDrop:
    """Test drop() - delete table."""

    def test_drop(self):
        db = SQL(TEST_DB)
        items = db(Item)
        items.create(name="test")
        items.drop()

        # Table should be recreated on next call
        items2 = db(Item)
        assert items2.read() == []


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_not_dataclass_raises(self):
        class NotADataclass:
            pass

        db = SQL(TEST_DB)
        with pytest.raises(TypeError, match="must be a dataclass"):
            db(NotADataclass)

    def test_unknown_field_raises(self):
        db = SQL(TEST_DB)
        items = db(Item)

        with pytest.raises(KeyError, match="Unknown field"):
            items.create(nonexistent="value")

    def test_unknown_filter_field_raises(self):
        db = SQL(TEST_DB)
        items = db(Item)

        with pytest.raises(KeyError, match="Unknown field"):
            items.read(nonexistent="value")

    def test_unknown_update_field_raises(self):
        db = SQL(TEST_DB)
        items = db(Item)
        added = items.create(name="a")

        with pytest.raises(KeyError, match="Unknown field"):
            items.update(id=added[0].id, nonexistent="value")

    def test_unknown_delete_filter_raises(self):
        db = SQL(TEST_DB)
        items = db(Item)

        with pytest.raises(KeyError, match="Unknown field"):
            items.delete(nonexistent="value")

    def test_null_values(self):
        db = SQL(TEST_DB)
        items = db(Item)
        result = items.create(name="minimal")

        assert result[0].meta is None
        assert result[0].tags is None

    def test_returns_dataclass_instances(self):
        db = SQL(TEST_DB)
        items = db(Item)
        result = items.create(name="test")

        assert isinstance(result[0], Item)

    def test_consistency_always_returns_list(self):
        db = SQL(TEST_DB)
        items = db(Item)

        # All operations return lists
        added = items.create(name="a")
        assert isinstance(added, list)
        assert isinstance(items.read(), list)
        assert isinstance(items.read(id=added[0].id), list)
        assert isinstance(items.update(id=added[0].id, name="b"), list)
        assert isinstance(items.delete(id=added[0].id), list)

    def test_create_invalid_type_raises(self):
        db = SQL(TEST_DB)
        items = db(Item)

        with pytest.raises(TypeError, match="Expected dict"):
            items.create("invalid")  # type: ignore

    def test_update_invalid_type_raises(self):
        db = SQL(TEST_DB)
        items = db(Item)

        with pytest.raises(TypeError, match="Expected dict"):
            items.update("invalid")  # type: ignore

    def test_update_with_dataclass_instance(self):
        db = SQL(TEST_DB)
        items = db(Item)
        added = items.create(name="original")

        # Update using dataclass instance
        updated_item = Item(id=added[0].id, name="updated")
        result = items.update(updated_item)

        assert len(result) == 1
        assert result[0].name == "updated"

    def test_delete_hard_delete_all(self):
        db = SQL(TEST_DB)
        items = db(Item)
        items.create({"name": "a"}, {"name": "b"})

        # Hard delete all
        deleted = items.delete(hard=True, all=True)
        assert len(deleted) == 2

        # Nothing left, even with include_deleted
        assert items.read(include_deleted=True) == []

    def test_update_only_id_no_update(self):
        """Test set with only id and no other fields on non-Model dataclass."""
        db = SQL(TEST_DB)
        items = db(SimpleItem)
        added = items.create(name="test")

        # Set with only id - should skip update
        result = items.update(id=added[0].id)
        assert result == []

    def test_dataclass_without_model(self):
        """Test dataclass without Model base - no soft delete."""
        db = SQL(TEST_DB)
        items = db(SimpleItem)

        added = items.create(name="test")
        assert added[0].id is not None

        # rm does hard delete (no deleted_at field)
        deleted = items.delete(id=added[0].id)
        assert len(deleted) == 1
        assert items.read() == []

    def test_update_with_unknown_auto_field_skips(self):
        """Test set with only unknown auto field on non-Model class."""
        db = SQL(TEST_DB)
        items = db(SimpleItem)
        added = items.create(name="test")

        # Pass updated_at which SimpleItem doesn't have - should skip
        result = items.update({"id": added[0].id, "updated_at": "ignored"})
        assert result == []


class TestDatetime:
    """Test datetime, date, time support."""

    def test_create_with_datetime(self):
        db = SQL(TEST_DB)
        events = db(Event)
        dt = datetime(2024, 6, 15, 10, 30, 0, tzinfo=UTC)
        result = events.create(title="Meeting", starts_at=dt)

        assert result[0].starts_at == dt
        assert isinstance(result[0].starts_at, datetime)

    def test_datetime_always_utc(self):
        """Naive datetime is treated as UTC."""
        db = SQL(TEST_DB)
        events = db(Event)
        # The missing tzinfo is the point of this test
        naive_dt = datetime(2024, 6, 15, 10, 30, 0)  # noqa: DTZ001
        result = events.create(title="Meeting", starts_at=naive_dt)

        # Should be stored and returned as UTC
        assert result[0].starts_at.tzinfo == UTC
        assert result[0].starts_at == datetime(2024, 6, 15, 10, 30, 0, tzinfo=UTC)

    def test_create_with_date(self):
        db = SQL(TEST_DB)
        events = db(Event)
        d = date(2024, 6, 15)
        result = events.create(title="Holiday", event_date=d)

        assert result[0].event_date == d
        assert isinstance(result[0].event_date, date)

    def test_create_with_time(self):
        db = SQL(TEST_DB)
        events = db(Event)
        t = time(10, 30, 0)
        result = events.create(title="Daily standup", event_time=t)

        assert result[0].event_time == t
        assert isinstance(result[0].event_time, time)

    def test_read_preserves_datetime_types(self):
        db = SQL(TEST_DB)
        events = db(Event)
        dt = datetime(2024, 6, 15, 10, 30, 0, tzinfo=UTC)
        d = date(2024, 6, 15)
        t = time(10, 30, 0)

        added = events.create(
            title="Full event", starts_at=dt, event_date=d, event_time=t
        )
        result = events.read(id=added[0].id)

        assert result[0].starts_at == dt
        assert result[0].event_date == d
        assert result[0].event_time == t

    def test_update_datetime(self):
        db = SQL(TEST_DB)
        events = db(Event)
        dt1 = datetime(2024, 6, 15, 10, 30, 0, tzinfo=UTC)
        dt2 = datetime(2024, 7, 20, 14, 0, 0, tzinfo=UTC)

        added = events.create(title="Meeting", starts_at=dt1)
        updated = events.update(id=added[0].id, starts_at=dt2)

        assert updated[0].starts_at == dt2

    def test_filter_by_datetime(self):
        db = SQL(TEST_DB)
        events = db(Event)
        dt = datetime(2024, 6, 15, 10, 30, 0, tzinfo=UTC)
        events.create(title="Meeting", starts_at=dt)

        result = events.read(starts_at=dt)
        assert len(result) == 1
        assert result[0].title == "Meeting"

    def test_null_datetime(self):
        db = SQL(TEST_DB)
        events = db(Event)
        result = events.create(title="No date")

        assert result[0].starts_at is None
        assert result[0].event_date is None
        assert result[0].event_time is None

    def test_datetime_without_tz_in_db(self):
        """Datetime stored without timezone (legacy data) is treated as UTC."""
        import sqlite3

        db = SQL(TEST_DB)
        events = db(Event)

        # Insert directly with naive datetime string (no timezone)
        conn = sqlite3.connect(TEST_DB)
        conn.execute(
            "INSERT INTO event (id, title, starts_at, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (
                "test-id",
                "Legacy",
                "2024-06-15T10:30:00",
                "2024-01-01T00:00:00+00:00",
                "2024-01-01T00:00:00+00:00",
            ),
        )
        conn.commit()
        conn.close()

        # Read back - should be UTC
        result = events.read(id="test-id")
        assert result[0].starts_at.tzinfo == UTC


class TestToDict:
    """Test to_dict() method."""

    def test_to_dict_basic(self):
        db = SQL(TEST_DB)
        items = db(Item)
        result = items.create(name="test", count=5)

        d = result[0].to_dict()
        assert isinstance(d, dict)
        assert d["name"] == "test"
        assert d["count"] == 5
        assert d["id"] is not None

    def test_to_dict_with_datetime(self):
        db = SQL(TEST_DB)
        events = db(Event)
        dt = datetime(2024, 6, 15, 10, 30, 0, tzinfo=UTC)
        result = events.create(title="Meeting", starts_at=dt)

        d = result[0].to_dict()
        assert d["starts_at"] == "2024-06-15T10:30:00+00:00"
        assert isinstance(d["starts_at"], str)

    def test_to_dict_with_date(self):
        db = SQL(TEST_DB)
        events = db(Event)
        d = date(2024, 6, 15)
        result = events.create(title="Holiday", event_date=d)

        data = result[0].to_dict()
        assert data["event_date"] == "2024-06-15"

    def test_to_dict_with_time(self):
        db = SQL(TEST_DB)
        events = db(Event)
        t = time(10, 30, 0)
        result = events.create(title="Standup", event_time=t)

        data = result[0].to_dict()
        assert data["event_time"] == "10:30:00"

    def test_to_dict_json_serializable(self):
        db = SQL(TEST_DB)
        events = db(Event)
        dt = datetime(2024, 6, 15, 10, 30, 0, tzinfo=UTC)
        d = date(2024, 6, 15)
        t = time(10, 30, 0)

        result = events.create(title="Event", starts_at=dt, event_date=d, event_time=t)

        # Should not raise - all values are JSON serializable
        json_str = json.dumps(result[0].to_dict())
        assert "2024-06-15" in json_str

    def test_to_dict_list(self):
        db = SQL(TEST_DB)
        events = db(Event)
        events.create({"title": "A"}, {"title": "B"})

        result = events.read()
        data = [e.to_dict() for e in result]

        assert len(data) == 2
        json_str = json.dumps(data)
        assert "A" in json_str
        assert "B" in json_str


class TestFromDict:
    """Test from_dict() class method."""

    def test_from_dict_basic(self):
        item = Item.from_dict({"name": "test", "count": 5})

        assert item.name == "test"
        assert item.count == 5

    def test_from_dict_with_datetime(self):
        event = Event.from_dict(
            {"title": "Meeting", "starts_at": "2024-06-15T10:30:00+00:00"}
        )

        assert event.title == "Meeting"
        assert isinstance(event.starts_at, datetime)
        assert event.starts_at == datetime(2024, 6, 15, 10, 30, 0, tzinfo=UTC)

    def test_from_dict_datetime_naive_becomes_utc(self):
        event = Event.from_dict(
            {"title": "Meeting", "starts_at": "2024-06-15T10:30:00"}  # no timezone
        )

        assert event.starts_at.tzinfo == UTC

    def test_from_dict_with_date(self):
        event = Event.from_dict({"title": "Holiday", "event_date": "2024-06-15"})

        assert isinstance(event.event_date, date)
        assert event.event_date == date(2024, 6, 15)

    def test_from_dict_with_time(self):
        event = Event.from_dict({"title": "Standup", "event_time": "10:30:00"})

        assert isinstance(event.event_time, time)
        assert event.event_time == time(10, 30, 0)

    def test_from_dict_null_values(self):
        event = Event.from_dict({"title": "No date", "starts_at": None})

        assert event.starts_at is None

    def test_from_dict_partial(self):
        """from_dict with only some fields."""
        event = Event.from_dict({"title": "Partial"})

        assert event.title == "Partial"
        assert event.starts_at is None

    def test_from_dict_roundtrip(self):
        """to_dict -> from_dict should preserve data."""
        db = SQL(TEST_DB)
        events = db(Event)
        dt = datetime(2024, 6, 15, 10, 30, 0, tzinfo=UTC)
        d = date(2024, 6, 15)
        t = time(10, 30, 0)

        original = events.create(
            title="Event", starts_at=dt, event_date=d, event_time=t
        )[0]

        # Roundtrip
        data = original.to_dict()
        restored = Event.from_dict(data)

        assert restored.title == original.title
        assert restored.starts_at == original.starts_at
        assert restored.event_date == original.event_date
        assert restored.event_time == original.event_time


@dataclass
class Reserved(Model):
    """Model whose fields collide with SQL reserved words."""

    order: int = 0
    group: str = ""
    where: str = ""


class TestReservedWords:
    """Field names that are SQL keywords must still work."""

    def test_crud_with_reserved_field_names(self):
        db = SQL(TEST_DB)
        table = db(Reserved)

        created = table.create(order=1, group="a", where="x")
        assert created[0].order == 1

        assert len(table.read(group="a")) == 1
        assert table.count(group="a").total == 1

        updated = table.update(id=created[0].id, order=2)
        assert updated[0].order == 2

        assert len(table.delete(id=created[0].id)) == 1
        assert table.read() == []


class TestInMemory:
    """An in-memory database must persist across statements."""

    def test_memory_db_persists_between_calls(self):
        db = SQL(":memory:")
        items = db(Item)

        items.create(name="a")
        items.create(name="b")

        assert items.count().total == 2
        assert {i.name for i in items.read()} == {"a", "b"}

    def test_memory_dbs_are_isolated(self):
        first = SQL(":memory:")(Item)
        second = SQL(":memory:")(Item)

        first.create(name="only-in-first")

        assert first.count().total == 1
        assert second.count().total == 0

    def test_context_manager_closes(self):
        with SQL(TEST_DB) as db:
            items = db(Item)
            items.create(name="a")
            assert items.count().total == 1

        assert db._conn is None


class TestUuid7:
    """IDs are UUID version 7: time-ordered and k-sortable."""

    def test_id_is_uuid7(self):
        db = SQL(TEST_DB)
        items = db(Item)
        created = items.create(name="a")

        parsed = uuid.UUID(created[0].id)
        assert parsed.version == 7

    def test_ids_sort_chronologically(self):
        import time as clock  # datetime.time shadows the module here

        db = SQL(TEST_DB)
        items = db(Item)

        ids = []
        for i in range(5):
            ids.append(items.create(name=f"n{i}")[0].id)
            clock.sleep(0.002)  # cross a millisecond boundary

        assert ids == sorted(ids)

    def test_ids_unique(self):
        db = SQL(TEST_DB)
        items = db(Item)
        created = items.create(*[{"name": f"n{i}"} for i in range(500)])

        assert len({c.id for c in created}) == 500

    def test_ids_monotonic_within_one_millisecond(self):
        """The counter, not the clock, is what orders a burst of ids.

        A bare millisecond timestamp plus random bits sorts randomly inside one
        millisecond, which would let a keyset walk skip rows.
        """
        db = SQL(TEST_DB)
        items = db(Item)
        created = items.create(*[{"name": f"n{i}"} for i in range(1000)])

        ids = [c.id for c in created]
        assert ids == sorted(ids)
        assert len(set(ids)) == 1000
        # The point of the test: this really was one clock tick
        assert len({i[:8] for i in ids}) <= 2

    def test_fallback_generator_is_monotonic(self):
        """Exercised directly, since 3.14+ uses stdlib uuid7 instead."""
        from sqlow import _uuid7

        ids = [str(_uuid7()) for _ in range(1000)]
        assert ids == sorted(ids)
        assert len(set(ids)) == 1000

    def test_fallback_survives_clock_going_backwards(self, monkeypatch):
        """An NTP step backwards must not produce an id that sorts earlier."""
        from time import time_ns

        import sqlow
        from sqlow import _uuid7

        before = str(_uuid7())
        # Pin the clock an hour into the past
        past = time_ns() - 3_600 * 10**9
        monkeypatch.setattr(sqlow, "time_ns", lambda: past)
        after = str(_uuid7())

        assert after > before

    def test_counter_overflow_borrows_the_next_millisecond(self, monkeypatch):
        """Exhausting the 42-bit counter must not repeat or reorder an id.

        2**41 ids in one millisecond is unreachable in a test, so the counter is
        placed at its maximum directly.
        """
        from time import time_ns

        import sqlow
        from sqlow import _UUID7_COUNTER_MAX, _uuid7

        now_ms = time_ns() // 1_000_000
        # Pin the clock so the "same millisecond" branch is the one taken
        monkeypatch.setattr(sqlow, "time_ns", lambda: now_ms * 1_000_000)
        monkeypatch.setattr(sqlow, "_uuid7_last_ms", now_ms)
        monkeypatch.setattr(sqlow, "_uuid7_counter", _UUID7_COUNTER_MAX)

        overflowed = _uuid7()

        # The timestamp advanced by exactly one millisecond, and the counter was
        # reseeded below the halfway mark so there is headroom again
        assert overflowed.int >> 80 == now_ms + 1
        assert sqlow._uuid7_counter <= _UUID7_COUNTER_MAX >> 1
        assert overflowed.version == 7

    def test_ids_unique_across_threads(self):
        import threading

        from sqlow import _uuid7

        per_thread: list[list[str]] = []
        lock = threading.Lock()

        def generate():
            mine = [str(_uuid7()) for _ in range(500)]
            with lock:
                per_thread.append(mine)

        threads = [threading.Thread(target=generate) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Each thread sees its own ids in order, and no id is ever reused
        for mine in per_thread:
            assert mine == sorted(mine)
        everything = [i for mine in per_thread for i in mine]
        assert len(set(everything)) == 8 * 500

    def test_keyset_walk_sees_a_burst_exactly_once(self):
        """The guarantee read(after=...) rests on, end to end."""
        db = SQL(TEST_DB)
        items = db(Item)
        expected = [c.count for c in items.create(*({"count": n} for n in range(500)))]

        seen: list[int] = []
        batch = items.read(page=1, per_page=50)
        while batch:
            seen.extend(i.count for i in batch)
            batch = items.read(after=batch[-1].id, per_page=50)

        # Same rows, same order, nothing skipped and nothing repeated
        assert seen == expected
        assert len(seen) == 500

    def test_fallback_generator_is_uuid7(self):
        """The pre-3.14 fallback must produce a valid, current UUIDv7.

        On 3.14+ ids come from stdlib ``uuid.uuid7``, so the fallback is
        exercised directly to keep it tested on every supported version.
        """
        from time import time_ns

        from sqlow import _uuid7

        generated = _uuid7()

        assert generated.version == 7
        assert generated.variant == uuid.RFC_4122
        # The 48-bit prefix is the unix timestamp in milliseconds
        assert abs((generated.int >> 80) - time_ns() // 1_000_000) < 5_000


class TestUnresolvableAnnotations:
    """Annotations that cannot resolve fall back to the raw string -> TEXT."""

    def test_unresolvable_forward_reference_falls_back(self):
        @dataclass
        class Ghost(Model):
            blob: "Undefined" = None  # type: ignore[name-defined] # noqa: F821

        db = SQL(TEST_DB)
        table = db(Ghost)

        assert table._field_map["blob"].sql_type == "TEXT"
        assert table.create(blob="raw")[0].blob == "raw"


class TestStrict:
    """New tables are STRICT, so declared types are enforced."""

    def test_wrong_type_rejected(self):
        db = SQL(TEST_DB)
        items = db(Item)

        with pytest.raises(sqlite3.IntegrityError):
            items.create(name="x", count="not-a-number")

    def test_lossless_conversion_still_allowed(self):
        db = SQL(TEST_DB)
        items = db(Item)

        assert items.create(name="x", count="123")[0].count == 123


class TestTransactions:
    """Each operation commits once, and rolls back as a unit."""

    def test_failed_batch_rolls_back(self):
        db = SQL(TEST_DB)
        items = db(Item)

        with pytest.raises(sqlite3.IntegrityError):
            items.create({"name": "ok"}, {"name": "bad", "count": "nope"})

        # The first record must not survive the failed batch
        assert items.read() == []

    def test_batch_create_is_one_commit(self):
        db = SQL(TEST_DB)
        items = db(Item)
        created = items.create(*[{"name": f"n{i}"} for i in range(50)])

        assert len(created) == 50
        assert items.count().total == 50

    def test_executemany_commits_outside_transaction(self):
        """Direct executemany() with no open transaction must commit itself."""
        db = SQL(TEST_DB)
        db(Item)  # create the table
        db.executemany(
            'INSERT INTO "item" (id, name) VALUES (?, ?)',
            [("a", "x"), ("b", "y")],
        )
        db.close()

        # A fresh connection only sees the rows if the commit happened
        assert {i.id for i in SQL(TEST_DB)(Item).read()} == {"a", "b"}


class TestTableCache:
    """db(cls) reuses the Table instead of re-running CREATE TABLE."""

    def test_same_instance_returned(self):
        db = SQL(TEST_DB)

        assert db(Item) is db(Item)

    def test_drop_evicts_cache(self):
        db = SQL(TEST_DB)
        items = db(Item)
        items.create(name="a")
        items.drop()

        # A fresh Table must recreate the dropped table
        assert db(Item).read() == []


class TestPragmas:
    def test_wal_enabled_for_file_db(self):
        db = SQL(TEST_DB)
        db(Item).create(name="a")

        assert db.execute("PRAGMA journal_mode")[0][0].lower() == "wal"

    def test_synchronous_defaults_to_normal(self):
        db = SQL(TEST_DB)
        db(Item)

        assert db.execute("PRAGMA synchronous")[0][0] == 1  # NORMAL

    def test_synchronous_full_is_applied(self):
        db = SQL(TEST_DB, synchronous="FULL")
        db(Item).create(name="a")

        assert db.execute("PRAGMA synchronous")[0][0] == 2  # FULL

    def test_synchronous_is_case_insensitive(self):
        db = SQL(TEST_DB, synchronous="full")
        db(Item)

        assert db.synchronous == "FULL"
        assert db.execute("PRAGMA synchronous")[0][0] == 2

    def test_synchronous_invalid_raises(self):
        with pytest.raises(ValueError, match="synchronous must be one of"):
            SQL(TEST_DB, synchronous="SOMETIMES")

    def test_synchronous_rejects_injection(self):
        """The level is interpolated, so anything unrecognized must be refused."""
        with pytest.raises(ValueError, match="synchronous must be one of"):
            SQL(TEST_DB, synchronous="NORMAL; DROP TABLE item")

    def test_synchronous_is_keyword_only(self):
        """Positional slot 2 stays free for future options."""
        with pytest.raises(TypeError, match="positional argument"):
            SQL(TEST_DB, "FULL")  # type: ignore[misc]

    def test_sync_levels_match_the_annotation(self):
        """The runtime allowlist is derived from Synchronous, never duplicated."""
        from typing import get_args

        from sqlow import SYNC_LEVELS, Synchronous

        assert SYNC_LEVELS == frozenset(get_args(Synchronous.__value__))
        assert SYNC_LEVELS == {"OFF", "NORMAL", "FULL", "EXTRA"}


class TestOrderBy:
    """read(order_by=...) sorts by a declared field."""

    def test_order_by_ascending(self):
        db = SQL(TEST_DB)
        items = db(Item)
        items.create({"name": "b"}, {"name": "c"}, {"name": "a"})

        assert [i.name for i in items.read(order_by="name")] == ["a", "b", "c"]

    def test_order_by_descending(self):
        db = SQL(TEST_DB)
        items = db(Item)
        items.create({"name": "b"}, {"name": "c"}, {"name": "a"})

        assert [i.name for i in items.read(order_by="-name")] == ["c", "b", "a"]

    def test_order_by_with_filters(self):
        db = SQL(TEST_DB)
        items = db(Item)
        items.create(
            {"name": "x", "count": 2},
            {"name": "x", "count": 1},
            {"name": "y", "count": 3},
        )

        result = items.read(order_by="count", name="x")
        assert [i.count for i in result] == [1, 2]

    def test_order_by_with_pagination(self):
        db = SQL(TEST_DB)
        items = db(Item)
        items.create(*({"count": n} for n in (5, 3, 1, 4, 2)))

        page1 = items.read(order_by="-count", page=1, per_page=2)
        page2 = items.read(order_by="-count", page=2, per_page=2)
        assert [i.count for i in page1] == [5, 4]
        assert [i.count for i in page2] == [3, 2]

    def test_order_by_unknown_field_raises(self):
        db = SQL(TEST_DB)
        items = db(Item)

        with pytest.raises(KeyError, match="Unknown field"):
            items.read(order_by="nope")


class TestKeysetPagination:
    """read(after=...) walks the table in id order at constant cost."""

    def test_after_walks_all_rows_once(self):
        db = SQL(TEST_DB)
        items = db(Item)
        items.create(*({"count": n} for n in range(25)))

        seen: list[int] = []
        cursor = None
        while True:
            batch = (
                items.read(after=cursor, per_page=10)
                if cursor
                else items.read(page=1, per_page=10)
            )
            if not batch:
                break
            seen.extend(i.count for i in batch)
            cursor = batch[-1].id

        assert sorted(seen) == list(range(25))
        assert len(seen) == 25

    def test_after_respects_filters(self):
        db = SQL(TEST_DB)
        items = db(Item)
        items.create(
            {"name": "x", "count": 1},
            {"name": "y", "count": 2},
            {"name": "x", "count": 3},
        )

        first = items.read(name="x", page=1, per_page=1)
        rest = items.read(name="x", after=first[0].id, per_page=10)
        assert [i.count for i in rest] == [3]

    def test_after_excludes_soft_deleted(self):
        db = SQL(TEST_DB)
        items = db(Item)
        added = items.create({"name": "a"}, {"name": "b"}, {"name": "c"})
        items.delete(id=added[1].id)

        batch = items.read(after=added[0].id, per_page=10)
        assert [i.name for i in batch] == ["c"]

    def test_after_with_page_raises(self):
        db = SQL(TEST_DB)
        items = db(Item)

        with pytest.raises(ValueError, match="mutually exclusive"):
            items.read(after="abc", page=1)

    def test_after_with_order_by_raises(self):
        db = SQL(TEST_DB)
        items = db(Item)

        with pytest.raises(ValueError, match="id order"):
            items.read(after="abc", order_by="name")

    def test_after_allows_id_order(self):
        db = SQL(TEST_DB)
        items = db(Item)
        items.create({"name": "a"}, {"name": "b"})
        first = items.read(page=1, per_page=1)

        batch = items.read(after=first[0].id, order_by="id")
        assert [i.name for i in batch] == ["b"]


class TestNoDeletedAtIndex:
    """No index on deleted_at: it makes the planner pessimize keyset reads."""

    def test_keyset_query_uses_primary_key(self):
        db = SQL(TEST_DB)
        items = db(Item)
        items.create({"name": "a"}, {"name": "b"})

        plan = " ".join(
            row["detail"]
            for row in db.execute(
                "EXPLAIN QUERY PLAN SELECT * FROM item "
                "WHERE deleted_at IS NULL AND id > ? ORDER BY id LIMIT 10",
                ("x",),
            )
        )
        # A temp B-tree here means the id ordering was thrown away
        assert "TEMP B-TREE" not in plan

    def test_no_deleted_at_index_exists(self):
        db = SQL(TEST_DB)
        db(Item)

        rows = db.execute("PRAGMA index_list('item')")
        assert all("deleted_at" not in row["name"] for row in rows)


@dataclass
class Defaulted(Model):
    """Every field carries a default, including a factory."""

    status: str = "pending"
    attempts: int = 0
    active: bool = True
    tags: list = field(default_factory=lambda: ["new"])


@dataclass(kw_only=True)
class Required(Model):
    """A field with no default at all.

    kw_only, because a bare required field cannot follow Model's defaulted ones.
    """

    owner: str
    name: str = ""


@dataclass
class Computed(Model):
    """__post_init__ derives a field from another."""

    name: str = ""
    slug: str = ""

    def __post_init__(self):
        if not self.slug:
            self.slug = self.name.lower().replace(" ", "-")


class TestDefaults:
    """Omitted fields fall back to the dataclass default, never to NULL."""

    def test_defaults_applied_from_kwargs(self):
        db = SQL(TEST_DB)
        table = db(Defaulted)
        created = table.create(attempts=3)

        assert created[0].status == "pending"
        assert created[0].attempts == 3
        assert created[0].active is True

    def test_defaults_applied_from_dict(self):
        db = SQL(TEST_DB)
        table = db(Defaulted)
        created = table.create({"attempts": 1}, {"status": "sent"})

        assert created[0].status == "pending"
        assert created[1].attempts == 0
        assert created[1].status == "sent"

    def test_default_factory_applied(self):
        db = SQL(TEST_DB)
        table = db(Defaulted)

        assert table.create(status="x")[0].tags == ["new"]

    def test_default_factory_not_shared_between_rows(self):
        db = SQL(TEST_DB)
        table = db(Defaulted)
        created = table.create({"status": "a"}, {"status": "b"})

        created[0].tags.append("mutated")
        assert created[1].tags == ["new"]

    def test_defaults_persisted_not_just_returned(self):
        """The default must be in the column, not applied on read-back."""
        db = SQL(TEST_DB)
        db(Defaulted).create(attempts=1)

        row = db.execute('SELECT status, active FROM "defaulted"')[0]
        assert row["status"] == "pending"
        assert row["active"] == 1

    def test_dict_and_instance_agree(self):
        db = SQL(TEST_DB)
        table = db(Defaulted)

        from_dict = table.create({"status": "x"})[0]
        from_instance = table.create(Defaulted(status="x"))[0]

        # ids and timestamps differ by construction; the declared fields must not
        declared = ("status", "attempts", "active", "tags")
        assert [getattr(from_dict, n) for n in declared] == [
            getattr(from_instance, n) for n in declared
        ]

    def test_post_init_runs(self):
        db = SQL(TEST_DB)
        table = db(Computed)

        assert table.create(name="Hello World")[0].slug == "hello-world"

    def test_missing_required_field_raises(self):
        db = SQL(TEST_DB)
        table = db(Required)

        with pytest.raises(TypeError, match="owner"):
            table.create(name="x")

    def test_required_field_supplied_is_fine(self):
        db = SQL(TEST_DB)
        table = db(Required)

        assert table.create(name="x", owner="alice")[0].owner == "alice"

    def test_unknown_field_still_raises_key_error(self):
        """The KeyError must survive the trip through the constructor."""
        db = SQL(TEST_DB)
        table = db(Defaulted)

        with pytest.raises(KeyError, match="Unknown field"):
            table.create(nonexistent="value")

    def test_update_does_not_apply_defaults(self):
        """update() is a patch: untouched columns keep their stored values."""
        db = SQL(TEST_DB)
        table = db(Defaulted)
        created = table.create(status="sent", attempts=5)

        updated = table.update(id=created[0].id, attempts=6)
        assert updated[0].status == "sent"  # not reset to "pending"
        assert updated[0].attempts == 6


class TestQuery:
    """query() maps raw SQL back onto the model."""

    def test_query_with_like(self):
        db = SQL(TEST_DB)
        items = db(Item)
        items.create({"name": "btn-a"}, {"name": "btn-b"}, {"name": "card"})

        found = items.query(
            'SELECT * FROM "item" WHERE name LIKE ? ORDER BY name', ("btn-%",)
        )
        assert [i.name for i in found] == ["btn-a", "btn-b"]

    def test_query_returns_model_instances(self):
        db = SQL(TEST_DB)
        items = db(Item)
        items.create(name="x", count=3, meta={"a": 1}, active=True)

        found = items.query('SELECT * FROM "item"')
        assert isinstance(found[0], Item)
        # Decoding still applies: JSON and bool are not raw column values
        assert found[0].meta == {"a": 1}
        assert found[0].active is True

    def test_query_supports_ranges_and_aggregates(self):
        db = SQL(TEST_DB)
        items = db(Item)
        items.create(*({"count": n} for n in (1, 5, 10)))

        found = items.query('SELECT * FROM "item" WHERE count BETWEEN ? AND ?', (2, 10))
        assert sorted(i.count for i in found) == [5, 10]

    def test_query_includes_soft_deleted(self):
        """Nothing is added to the statement, so tombstones are visible."""
        db = SQL(TEST_DB)
        items = db(Item)
        added = items.create({"name": "a"}, {"name": "b"})
        items.delete(id=added[0].id)

        assert len(items.query('SELECT * FROM "item"')) == 2
        assert len(items.query('SELECT * FROM "item" WHERE deleted_at IS NULL')) == 1

    def test_query_empty_result(self):
        db = SQL(TEST_DB)
        items = db(Item)

        assert items.query('SELECT * FROM "item" WHERE name = ?', ("nope",)) == []

    def test_query_partial_projection_raises(self):
        db = SQL(TEST_DB)
        items = db(Item)
        items.create(name="x")

        with pytest.raises(IndexError):
            items.query('SELECT name FROM "item"')


class TestSchemaEvolution:
    """Fields added to a dataclass become new columns on existing tables."""

    def test_new_field_adds_column(self):
        @dataclass
        class Grow(Model):
            name: str = ""

        db = SQL(TEST_DB)
        db(Grow).create(name="old")
        db.close()

        @dataclass
        class GrowV2(Model):
            name: str = ""
            extra: str = ""

        # Same table name as Grow, one extra field
        GrowV2.__name__ = "Grow"
        db = SQL(TEST_DB)
        table = db(GrowV2)

        created = table.create(name="new", extra="x")
        assert created[0].extra == "x"

        # Old row survives and reads back None for the new column
        rows = {r.name: r for r in table.read()}
        assert rows["old"].extra is None
        assert rows["new"].extra == "x"
        db.close()

    def test_existing_columns_untouched(self):
        db = SQL(TEST_DB)
        items = db(Item)
        items.create(name="keep", count=7)
        db.close()

        # Re-opening with the same schema adds nothing and loses nothing
        db = SQL(TEST_DB)
        result = db(Item).read()
        assert len(result) == 1
        assert result[0].count == 7
        db.close()


@dataclass
class Node(Model):
    """Parameterized annotations, the way real models are written."""

    name: str = ""
    properties: dict[str, Any] = field(default_factory=dict)
    labels: list[str] = field(default_factory=list)
    edges: list[dict[str, str]] | None = None
    scores: Optional[dict[str, int]] = None  # noqa: UP045 - the legacy spelling
    tagged: Annotated[dict[str, Any], "note"] = field(default_factory=dict)
    weight: int | None = None


class TestParameterizedTypes:
    """Generic annotations map to storage the same way their bare form does."""

    def test_generic_dict_and_list_are_json_columns(self):
        db = SQL(":memory:")
        nodes = db(Node)

        created = nodes.create(
            name="a",
            properties={"w": 1.5, "nested": {"x": [1, 2]}},
            labels=["Person", "Admin"],
        )[0]

        assert created.properties == {"w": 1.5, "nested": {"x": [1, 2]}}
        assert created.labels == ["Person", "Admin"]
        # Stored as JSON text, not a Python repr
        row = db.execute('SELECT properties, labels FROM "node"')[0]
        assert json.loads(row[0]) == created.properties
        assert json.loads(row[1]) == created.labels

    def test_optional_and_nested_generics_are_json_columns(self):
        db = SQL(":memory:")
        nodes = db(Node)

        created = nodes.create(
            name="a",
            edges=[{"to": "n1"}],
            scores={"s": 3},
            tagged={"t": True},
        )[0]

        assert nodes.read(id=created.id)[0] == created
        assert created.edges == [{"to": "n1"}]
        assert created.scores == {"s": 3}
        assert created.tagged == {"t": True}

    def test_generic_field_does_not_break_unrelated_inserts(self):
        """create() builds the whole record, so one bad field broke them all."""
        db = SQL(":memory:")
        nodes = db(Node)

        created = nodes.create(name="a")[0]

        assert created.properties == {}
        assert created.labels == []
        assert created.edges is None

    def test_generic_fields_survive_update_and_reread(self):
        db = SQL(":memory:")
        nodes = db(Node)
        created = nodes.create(name="a", properties={"w": 1})[0]

        nodes.update(id=created.id, properties={"w": 2}, labels=["x"])

        reread = nodes.read(id=created.id)[0]
        assert reread.properties == {"w": 2}
        assert reread.labels == ["x"]

    def test_generic_fields_round_trip_through_fixtures(self):
        db = SQL(":memory:")
        nodes = db(Node)
        nodes.create(name="a", properties={"w": 1}, labels=["x"])

        records = db.dump(None, Node)
        # JSON-native in the fixture, not a string holding JSON
        assert records[0]["fields"]["properties"] == {"w": 1}

        target = SQL(":memory:")(Node)
        assert target.load(records) == nodes.read()

    def test_scalar_unions_still_map_to_their_own_type(self):
        db = SQL(":memory:")
        nodes = db(Node)
        nodes.create(name="a", weight=7)

        columns = {row[1]: row[2] for row in db.execute('PRAGMA table_info("node")')}
        assert columns["weight"] == "INTEGER"  # int | None, not TEXT
        assert columns["properties"] == "TEXT"
        assert nodes.read()[0].weight == 7


class TestTableFixtures:
    """dump() and load() on a single table."""

    def test_dump_returns_wrapped_records(self):
        db = SQL(":memory:")
        items = db(Item)
        created = items.create({"name": "a"}, {"name": "b"})

        records = items.dump()
        assert [r["model"] for r in records] == ["item", "item"]
        assert [r["fields"]["name"] for r in records] == ["a", "b"]
        assert records[0]["fields"]["id"] == created[0].id

    def test_dump_values_are_json_native(self):
        db = SQL(":memory:")
        events = db(Event)
        items = db(Item)
        events.create(
            title="x",
            starts_at=datetime(2024, 6, 15, 10, 30, tzinfo=UTC),
            event_date=date(2024, 6, 15),
            event_time=time(10, 30),
        )
        items.create(name="y", meta={"a": 1}, tags=["t"], active=True)

        event_fields = events.dump()[0]["fields"]
        item_fields = items.dump()[0]["fields"]
        # Datetimes are ISO strings; JSON fields are objects, not TEXT
        assert event_fields["starts_at"] == "2024-06-15T10:30:00+00:00"
        assert event_fields["event_date"] == "2024-06-15"
        assert event_fields["event_time"] == "10:30:00"
        assert item_fields["meta"] == {"a": 1}
        assert item_fields["tags"] == ["t"]
        assert item_fields["active"] is True
        # And the whole thing survives json.dumps unchanged
        assert json.loads(json.dumps(items.dump())) == items.dump()

    def test_dump_writes_a_json_file(self, tmp_path):
        db = SQL(":memory:")
        items = db(Item)
        items.create(name="a")

        path = tmp_path / "items.json"
        records = items.dump(path)

        assert json.loads(path.read_text(encoding="utf-8")) == records

    def test_dump_indent_none_writes_one_line(self, tmp_path):
        db = SQL(":memory:")
        items = db(Item)
        items.create(name="a")

        path = tmp_path / "items.json"
        items.dump(path, indent=None)

        assert len(path.read_text(encoding="utf-8").splitlines()) == 1

    def test_dump_includes_soft_deleted_by_default(self):
        db = SQL(":memory:")
        items = db(Item)
        created = items.create({"name": "a"}, {"name": "b"})
        items.delete(id=created[0].id)

        assert len(items.dump()) == 2
        assert items.dump()[0]["fields"]["deleted_at"] is not None
        assert len(items.dump(include_deleted=False)) == 1

    def test_round_trip_preserves_ids_and_timestamps(self):
        source = SQL(":memory:")
        items = source(Item)
        created = items.create(
            {"name": "a", "count": 1, "meta": {"k": "v"}},
            {"name": "b", "tags": ["x"]},
        )
        items.delete(id=created[1].id)

        target = SQL(":memory:")
        loaded = target(Item).load(items.dump())

        assert loaded == items.read(include_deleted=True)
        assert target(Item).dump() == items.dump()

    def test_load_from_a_file(self, tmp_path):
        source = SQL(":memory:")
        source(Item).create(name="a")
        path = tmp_path / "items.json"
        source(Item).dump(path)

        target = SQL(":memory:")
        loaded = target(Item).load(path)

        assert [i.name for i in loaded] == ["a"]

    def test_load_bare_dicts_generates_auto_fields(self):
        db = SQL(":memory:")
        items = db(Item)

        loaded = items.load([{"name": "a"}, {"name": "b"}])

        assert [i.name for i in loaded] == ["a", "b"]
        assert loaded[0].id != loaded[1].id
        assert loaded[0].created_at is not None
        assert loaded[0].deleted_at is None

    def test_load_applies_defaults_for_omitted_fields(self):
        db = SQL(":memory:")
        table = db(Defaulted)

        loaded = table.load([{"attempts": 2}])

        assert loaded[0].attempts == 2
        assert loaded[0].status == "pending"
        assert loaded[0].tags == ["new"]

    def test_load_missing_required_field_raises(self):
        db = SQL(":memory:")
        table = db(Required)

        with pytest.raises(TypeError):
            table.load([{"name": "x"}])

    def test_load_replaces_a_row_with_the_same_id(self):
        db = SQL(":memory:")
        items = db(Item)
        created = items.create(name="old", count=1)[0]

        records = items.dump()
        records[0]["fields"]["name"] = "new"
        loaded = items.load(records)

        assert loaded[0].id == created.id
        assert loaded[0].name == "new"
        # Replaced, not duplicated, and untouched fields survive
        assert items.count().total == 1
        assert loaded[0].count == 1

    def test_load_without_replace_raises_on_conflict(self):
        db = SQL(":memory:")
        items = db(Item)
        items.create(name="a")

        with pytest.raises(sqlite3.IntegrityError):
            items.load(items.dump(), replace=False)

    def test_load_unknown_field_raises(self):
        db = SQL(":memory:")
        items = db(Item)

        with pytest.raises(KeyError):
            items.load([{"nope": 1}])

    def test_load_wrong_model_raises(self):
        db = SQL(":memory:")
        db(Project).create(title="t")

        with pytest.raises(ValueError):
            db(Item).load(db(Project).dump())

    def test_load_non_mapping_record_raises(self):
        db = SQL(":memory:")

        with pytest.raises(TypeError):
            db(Item).load(["not-a-record"])

    def test_load_file_without_a_list_raises(self, tmp_path):
        path = tmp_path / "items.json"
        path.write_text('{"model": "item"}', encoding="utf-8")

        db = SQL(":memory:")
        with pytest.raises(TypeError):
            db(Item).load(path)

    def test_load_nothing_is_a_no_op(self):
        db = SQL(":memory:")
        items = db(Item)

        assert items.load([]) == []
        assert items.count().total == 0

    def test_model_without_auto_fields(self):
        """A plain dataclass has no timestamps or tombstone to carry over."""
        db = SQL(":memory:")
        table = db(SimpleItem)
        table.create(name="a")

        records = table.dump()
        assert set(records[0]["fields"]) == {"id", "name"}

        target = SQL(":memory:")(SimpleItem)
        assert target.load(records) == table.read()


class TestDatabaseFixtures:
    """dump() and load() across every table in a database."""

    def test_dump_selected_models(self):
        db = SQL(":memory:")
        db(Item).create(name="a")
        db(Project).create(title="p")

        records = db.dump(None, Item, Project)

        assert [r["model"] for r in records] == ["item", "project"]

    def test_dump_every_open_table(self):
        """Tables are remembered by model even if no Table is held onto."""
        db = SQL(":memory:")
        db(Item).create(name="a")
        db(Project).create(title="p")

        assert {r["model"] for r in db.dump()} == {"item", "project"}

    def test_dump_with_a_model_as_path_raises(self):
        """db.dump(Item) is a mistake for db.dump(None, Item), not a filename."""
        db = SQL(":memory:")

        with pytest.raises(TypeError):
            db.dump(Item)

    def test_dropped_table_is_not_dumped(self):
        db = SQL(":memory:")
        items = db(Item)
        db(Project).create(title="p")
        items.create(name="a")
        items.drop()

        assert [r["model"] for r in db.dump()] == ["project"]

    def test_dump_writes_one_file_for_all_tables(self, tmp_path):
        db = SQL(":memory:")
        db(Item).create(name="a")
        db(Project).create(title="p")

        path = tmp_path / "seed.json"
        records = db.dump(path, Item, Project)

        assert json.loads(path.read_text(encoding="utf-8")) == records

    def test_round_trip_across_tables(self, tmp_path):
        source = SQL(":memory:")
        source(Item).create({"name": "a"}, {"name": "b"})
        source(Project).create(title="p")
        path = tmp_path / "seed.json"
        source.dump(path, Item, Project)

        target = SQL(":memory:")
        loaded = target.load(path, Item, Project)

        assert {name: len(rows) for name, rows in loaded.items()} == {
            "item": 2,
            "project": 1,
        }
        assert target(Item).read() == source(Item).read()
        assert target(Project).read() == source(Project).read()

    def test_load_finds_already_open_tables(self):
        source = SQL(":memory:")
        source(Item).create(name="a")

        target = SQL(":memory:")
        items = target(Item)  # opened, so load() does not need the dataclass
        assert len(target.load(source.dump())["item"]) == 1
        assert items.count().total == 1

    def test_load_unknown_model_raises(self):
        db = SQL(":memory:")

        with pytest.raises(KeyError):
            db.load([{"model": "ghost", "fields": {"name": "a"}}])

    def test_load_record_without_a_model_name_raises(self):
        """Bare field dicts belong to Table.load(), which knows the table."""
        db = SQL(":memory:")

        with pytest.raises(TypeError):
            db.load([{"name": "a"}], Item)

    def test_load_is_one_transaction(self):
        """A record that fails leaves nothing behind, in any table."""
        db = SQL(":memory:")
        items = db(Item)
        projects = db(Project)

        with pytest.raises(KeyError):
            db.load(
                [
                    {"model": "item", "fields": {"name": "a"}},
                    {"model": "project", "fields": {"nope": 1}},
                ]
            )

        assert items.count().total == 0
        assert projects.count().total == 0

    def test_load_without_replace_raises_on_conflict(self):
        db = SQL(":memory:")
        items = db(Item)
        items.create(name="a")

        with pytest.raises(sqlite3.IntegrityError):
            db.load(db.dump(), replace=False)
