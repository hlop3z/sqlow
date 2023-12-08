from sqlow import create_table
import pytest


# Define a table <Fields>
fields = {
    "project_id": int,
    "docs": str,
    "meta": dict,
    "info": list,
}


# Create an instance of the table
Components = lambda: create_table("db.sqlite3", "Components", **fields)


fake_data = dict(
    name="button",
    project_id=1,
    docs="Component documentation",
    meta={"author": "John Doe"},
    info=[1, 2, 3],
)


@pytest.fixture(scope="session", autouse=True)
def after_tests():
    # Code to run after all tests are done
    print("Cleaning up...")

    # Perform your cleanup actions here
    table = Components()
    table.drop()


def test_insert_and_get():
    # Create an instance of the table
    table = Components()

    # Insert data into the table
    table.set(**fake_data)

    # Retrieve a single record by name
    item = table.get_by(name="button")
    assert item["name"] == "button"
    assert item["project_id"] == 1
    assert item["docs"] == "Component documentation"
    # Add more assertions for other fields


def test_set_and_get():
    # Create an instance of the table
    table = Components()

    # Retrieve a single record by name
    item = table.get_by(name="button")
    # Insert data into the table
    item["meta"] = {"new-meta": "planeta"}
    table.set(**item)

    # item = table.get_by(name="button")

    assert item["meta"] == {"new-meta": "planeta"}
    # Add more assertions for other fields


def test_all():
    # Create an instance of the table
    table = Components()

    # Retrieve all records from the table
    all_items = table.all()
    assert len(all_items) == 1  # Ensure the previous test inserted a record


def test_delete():
    # Create an instance of the table
    table = Components()

    # Insert second-item into the table
    data = {**fake_data, **{"name": "alert"}}
    table.set(**data)

    # Retrieve all records from the table
    all_items = table.all()
    assert len(all_items) == 2

    # Delete One
    table.delete(name="button")

    # Retrieve all records from the table
    all_items = table.all()
    assert len(all_items) == 1


def test_delete_all():
    # Create an instance of the table
    table = Components()

    data = {**fake_data, **{"name": "button"}}
    # Insert second-item into the table
    table.set(**data)

    # Retrieve all records from the table
    all_items = table.all()
    assert len(all_items) == 2

    # Delete All
    table.delete_all()

    # Retrieve all records from the table
    all_items = table.all()
    assert len(all_items) == 0
