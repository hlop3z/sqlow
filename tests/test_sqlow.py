"""Tests for sqlow - dataclass-native SQLite CRUD."""

import os
import pytest
from dataclasses import dataclass

from sqlow import SQL, Model, Count


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


@pytest.fixture(autouse=True)
def cleanup():
    """Clean up test database before and after each test."""
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
    yield
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


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

        deleted = items.delete()
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
        result = items.update({"id": added[0].id, "name": "x"}, {"id": added[1].id, "name": "y"})

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
        deleted = items.delete()

        assert len(deleted) == 3
        assert items.read() == []

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
        deleted = items.delete()
        assert deleted == []


class TestPagination:
    """Test pagination functionality."""

    def test_read_with_page(self):
        db = SQL(TEST_DB)
        items = db(Item)
        items.create({"name": "a"}, {"name": "b"}, {"name": "c"}, {"name": "d"}, {"name": "e"})

        result = items.read(page=1, per_page=3)
        assert len(result) == 3

    def test_read_pagination(self):
        db = SQL(TEST_DB)
        items = db(Item)
        items.create({"name": "a"}, {"name": "b"}, {"name": "c"}, {"name": "d"}, {"name": "e"})

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
        items.create({"name": "a", "count": 1}, {"name": "b", "count": 2}, {"name": "c", "count": 1})

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
        deleted = items.delete(hard=True)
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
