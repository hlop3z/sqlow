# **SQLow**: **`DataClass`** SQLite Database Management for **File-like** Operations

SQLow is a lightweight Python library that simplifies SQLite database operations, specifically tailored for **file-like data management**. If you work with frontend components written in TypeScript or JavaScript, SQLow offers an intuitive way to manage data as if they were files, all while benefiting from the power and efficiency of an SQLite database.

## Key Features

- **Simplified Database Operations:** SQLow streamlines database interactions using **data classes** and **decorators**, abstracting away the complexity of SQL queries.
- **Efficient Data Serialization:** It efficiently handles data serialization for various data types, ensuring seamless integration with your codebase.
- **Automatic Table Management:** SQLow automatically creates and manages database tables, sparing you from manual table setup.
- **Customizable Table Configuration:** Tables can be configured with decorators, allowing you to define unique constraints and relationships.

## Installation

You can install **`SQLow`** using the following command:

```sh
pip install sqlow
```

## Method Descriptions

- **`set(kwargs)`**: **Inserts** or **Updates** a row in the database. If a row with the given name exists, it updates the row.

- **`get(name: str)`**: **Retrieves a single row** from the database by its name.

- **`all()`**: **Retrieves all rows** from the database and returns them as a list of dictionaries.

- **`delete(name: str)`**: **Deletes a single row** from the database by its name.

- **`delete_all()`**: **Deletes all rows** from the database.

- **`drop()`**: **Drops the entire table** from the database.

## Note

In SQLow, all tables include `id` and `name` columns. This design choice aligns with the file-like nature of the data and simplifies operations.

## Usage Example

Here's a practical example that demonstrates how to use SQLow to manage file-like data in an SQLite database.

```python
from sqlow import sqlow

# Initialize SQLow with the SQLite database
sqlite = sqlow("db.sqlite3")

# Define a table using the SQLow decorator
@sqlite
class Components:
    project_id: int
    docs: str
    meta: dict
    info: list

# Create an instance of the table
table = Components()

# Insert data into the table
table.set(
    name="button",
    project_id=1,
    docs="Component documentation",
    meta={"author": "John Doe"},
    info=[1, 2, 3]
)

# Retrieve a single record by name
item = table.get_by(name="button")
print("Retrieved Item:", item)

# Retrieve all records from the table
all_items = table.all()
print("All Items:", all_items)

# Retrieve a single record by name
item_to_update = table.get_by(name="button")

# Update an existing record by name
item_to_update["meta"] = {"new-meta": "planeta"}
table.set(**item_to_update)

# Delete a record by name
table.delete(name="button")

# Delete all records from the table
table.delete_all()

# Drop the entire table
table.drop()
```

## Usage of **`create_table`**

```python
from sqlow import create_table

fields = {
    "project_id": int,
    "docs": str,
    "meta": dict,
    "info": list,
}

table = create_table("db.sqlite3", "Components", **fields)
```
