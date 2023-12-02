from sqlow import sqlow
import pytest


# Initialize SQLow with the SQLite database
sqlite = sqlow("db.sqlite3")


# Define a table using the SQLow decorator
@sqlite
class Components:
    project_id: int
    docs: str
    meta: dict
    info: list


fake_data = dict(
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
    table.set(name="button", **fake_data)

    # Retrieve a single record by name
    item = table.get(name="button")
    assert item["name"] == "button"
    assert item["project_id"] == 1
    assert item["docs"] == "Component documentation"
    # Add more assertions for other fields

def test_rename():
    # Create an instance of the table
    table = Components()

    # Insert data into the table
    table.rename("button", "alert")

    # Retrieve a single record by name
    button = table.get(name="button")
    item = table.get(name="alert")
    assert button == None
    assert item["name"] == "alert"
    assert item["project_id"] == 1
    assert item["docs"] == "Component documentation"
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
    table.set(name="table", **fake_data)

    # Retrieve all records from the table
    all_items = table.all()
    assert len(all_items) == 2

    # Delete One
    table.delete("button")

    # Retrieve all records from the table
    all_items = table.all()
    assert len(all_items) == 2


def test_delete_all():
    # Create an instance of the table
    table = Components()

    # Insert second-item into the table
    table.set(name="button", **fake_data)

    # Retrieve all records from the table
    all_items = table.all()
    assert len(all_items) == 3

    # Delete All
    table.delete_all()

    # Retrieve all records from the table
    all_items = table.all()
    assert len(all_items) == 0
