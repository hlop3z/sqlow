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
    info=[1, 2, 3],
)

# Retrieve a single record by name
item = table.get(name="button")
print("Retrieved Item:", item)

# Retrieve all records from the table
all_items = table.all()
print("All Items:", all_items)

# Update an existing record by name
table.set(name="button", docs="Updated documentation")

# Delete a record by name
table.delete(name="button")

# Delete all records from the table
table.delete_all()

# Drop the entire table
table.drop()
