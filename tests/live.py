from sqlow import sqlow
import datetime

# Initialize SQLow with the SQLite database
sqlite = sqlow("db.sqlite3")


# Define a table using the SQLow decorator
@sqlite
class Components:
    project_id: int
    is_good: bool
    docs: str
    meta: dict
    info: list
    date: datetime.datetime


fake_data = dict(
    name="button",
    is_good=True,
    project_id=1,
    docs="Component documentation",
    meta={"author": "John Doe"},
    info=[1, 2, 3],
    date=datetime.datetime.now(datetime.UTC),
)


def after_tests():
    # Code to run after all tests are done
    print("Cleaning up...")

    # Perform your cleanup actions here
    table = Components()
    table.drop()


def create_test():
    table = Components()

    # table.insert(id=4, **fake_data)
    # print(table.get(4))

    # table.dump("demo-dump.json")

    print(table.all())
    table.load("demo-dump.json")
    print(table.all())


create_test()
# after_tests()
