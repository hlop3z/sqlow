from dataclasses import dataclass, fields
from decimal import Decimal
import functools
import sqlite3
import types
import json
import typing
import re


class PreDefinedClass:
    """PreDefinedClass"""

    name: str = None


class SQLowDatabase:
    """SQLow Database"""

    def __init__(
        self, table_class: dataclass = typing.Any, db_name: str = "db.sqlite3"
    ):
        """
        Initialize the SQLite Manager.

        Args:
            table_class (dataclass, optional): The dataclass representing the table structure.
            db_name (str, optional): The name of the SQLite database file.
        """
        self.db_name = db_name
        self.table_name = table_class.__objconfig__.table_name
        self._dataclass_type = table_class
        self._initialize_table()

    def _connect(self):
        """
        Connect to the SQLite database and create a cursor.
        """
        self.connection = sqlite3.connect(self.db_name)
        self.connection.row_factory = sqlite3.Row
        self.cursor = self.connection.cursor()

    def _close(self):
        """
        Commit changes and close the database connection.
        """
        self.connection.commit()
        self.connection.close()

    def execute(self, query, params=None):
        self._connect()
        response = self.cursor.execute(query, params or ())
        self._close()
        return response

    def fetch_one(self, query, params=None):
        self._connect()
        self.cursor.execute(query, params or ())
        response = self.cursor.fetchone()
        self._close()
        return response

    def fetch_all(self, query, params=None):
        self._connect()
        self.cursor.execute(query, params or ())
        response = self.cursor.fetchall()
        self._close()
        return response

    def insert(self, **kwargs):
        self._connect()
        response = self.cursor.execute(*kwargs_insert(self, **kwargs))
        self._close()
        return response

    def update(self, query, params=None):
        self._connect()
        response = self.cursor.execute(query, params or ())
        self._close()
        return response

    def set(self, **kwargs):
        """
        {Add | Update} <Row> in the Database.
        """
        name = kwargs.get("name")
        row = None
        if name:
            row = self.get(name=name)
        if row:
            del kwargs["name"]
            self.update(*kwargs_update(self, name, **kwargs))
        else:
            self.insert(**kwargs)
        return None

    def get(self, name: str):
        """
        {Get} <Row> in the Database.
        """
        self._connect()
        self.cursor.execute(*kwargs_select(self, name=name))
        row = self.cursor.fetchone()
        self._close()
        return self._load_value(row)

    def all(self):
        """
        {Get-All} <Rows> in the Database.
        """
        self._connect()
        self.cursor.execute(*kwargs_select(self))
        rows = self.cursor.fetchall()
        self._close()
        return [self._load_value(row) for row in rows]

    def delete(self, name: str):
        """
        {Delete} <Row> in the Database.
        """
        self._connect()
        self.cursor.execute(*kwargs_delete(self, name=name))
        self._close()
        return None

    def delete_all(self):
        """
        {Delete-All} <Rows> in the Database.
        """
        self._connect()
        self.cursor.execute(f"DELETE FROM {self.table_name}")
        self._close()
        return None

    def drop(self):
        self.execute(f"DROP TABLE IF EXISTS {self.table_name}")
        return None

    def _initialize_table(self):
        """
        Initialize the 'Component' table in the database if it doesn't exist.
        """
        self._connect()

        # Create the 'Component' table if it doesn't exist
        self.cursor.execute(self._create_table_from_dataclass)
        self._close()
        return None

    def _load_value(self, loaded_object):
        dataclass_type = self._dataclass_type
        obj_data = dict(loaded_object) if loaded_object else None
        processed_object = {}
        if not obj_data:
            return None
        for field in fields(dataclass_type):
            field_name = field.name
            field_value = obj_data.get(field_name)

            if field.type == float:
                processed_object[field_name] = Decimal(str(field_value))
            elif field.type == dict or field.type == list:
                processed_object[field_name] = (
                    json.loads(field_value) if field_value else None
                )
            else:
                processed_object[field_name] = field_value

        return processed_object

    def _process_values(self, **kwargs):
        processed_values = {}

        for key, value in kwargs.items():
            field = next(f for f in fields(self._dataclass_type) if f.name == key)
            field_type = field.type

            if field_type == float:
                # Convert float to Decimal
                processed_values[key] = Decimal(str(value))
            elif field_type == dict or field_type == list:
                # Convert dict or list to JSON string
                processed_values[key] = json.dumps(value)
            else:
                processed_values[key] = value
            if key == "name":
                processed_values[key] = slugify(value)
        return processed_values

    @property
    def _create_table_from_dataclass(self):
        dataclass_type = self._dataclass_type
        dataclass_config = dataclass_type.__objconfig__.__dict__
        table_name = dataclass_config.get("table_name")
        table_unique = dataclass_config.get("unique", [])
        table_unique_together = dataclass_config.get("unique_together", [])
        table_columns = []
        table_unique.append("name")
        columns = ["id INTEGER PRIMARY KEY"]
        for field in fields(dataclass_type):
            field_type = field.type
            field_config: str = ""
            if field_type == str:
                field_config = f"{field.name} TEXT"
            elif field_type == int:
                field_config = f"{field.name} INTEGER"
            elif field_type == float:
                field_config = f"{field.name} DECIMAL"
            elif field_type == bool:
                field_config = f"{field.name} BOOLEAN"
            elif field_type == dict or field_type == list:
                field_config = f"{field.name} JSON"
            # Other Configs
            if field.name in table_unique:
                field_config = f"{field_config} UNIQUE"
            # Append
            columns.append(field_config)
            table_columns.append(field.name)

        # unique_together
        for items in table_unique_together:
            columns.append(f"UNIQUE({ ', '.join(items) })")

        # Build
        columns_str = ", ".join(columns)
        return f"CREATE TABLE IF NOT EXISTS {table_name} ({columns_str})"


def slugify(text):
    """
    Convert a string to a slug format.

    Args:
        text (str): The input text.

    Returns:
        str: The slugified text.
    """
    text = re.sub(r"[^\w\s-]", "", text.lower())
    text = re.sub(r"[-\s]+", "-", text)
    text = re.sub(r"^-|-$", "", text)  # Remove leading or trailing "-"
    text = re.sub(r"--+", "-", text)  # Replace double "--" with single "-"
    return text


def kwargs_insert(self, **kwargs):
    """
    Generate the INSERT query and parameters for inserting data into the table.

    Args:
        self: The SQLowDatabase instance.
        **kwargs: Key-value pairs representing the data to be inserted.

    Returns:
        list: A list containing the query and parameters.
    """
    processed_values = self._process_values(**kwargs)
    keys = ", ".join(processed_values.keys())
    values = ", ".join("?" for _ in processed_values.values())
    query = f"INSERT INTO {self.table_name} ({keys}) VALUES ({values})"
    return [query, tuple(processed_values.values())]


def kwargs_update(self, name: str, **kwargs):
    """
    Generate the UPDATE query and parameters for updating data in the table.

    Args:
        self: The SQLowDatabase instance.
        name (str): The name of the row to be updated.
        **kwargs: Key-value pairs representing the data to be updated.

    Returns:
        list: A list containing the query and parameters.
    """
    processed_values = self._process_values(**kwargs)
    update_columns = ", ".join(f"{column} = ?" for column in processed_values.keys())
    query = f"UPDATE {self.table_name} SET {update_columns} WHERE name = ?"
    return [query, tuple(processed_values.values()) + (name,)]


def kwargs_delete(self, **kwargs):
    """
    Generate the DELETE query and parameters for deleting data from the table.

    Args:
        self: The SQLowDatabase instance.
        **kwargs: Key-value pairs representing the data to be used as conditions for deletion.

    Returns:
        list: A list containing the query and parameters.
    """
    processed_values = self._process_values(**kwargs)
    conditions = " AND ".join(f"{key} = ?" for key in processed_values.keys())
    query = f"DELETE FROM {self.table_name} WHERE {conditions}"
    return [query, tuple(processed_values.values())]


def kwargs_select(self, **kwargs):
    """
    Generate the SELECT query and parameters for retrieving data from the table.

    Args:
        self: The SQLowDatabase instance.
        **kwargs: Key-value pairs representing the conditions for selection.

    Returns:
        list: A list containing the query and parameters.
    """
    select_columns_str = "*"

    if kwargs:
        processed_values = self._process_values(**kwargs)
        conditions = " AND ".join(f"{key} = ?" for key in processed_values.keys())
        query = f"SELECT {select_columns_str} FROM {self.table_name} WHERE {conditions}"
        params = tuple(processed_values.values())
    else:
        query = f"SELECT {select_columns_str} FROM {self.table_name}"
        params = ()

    return [query, params]


def class_schema_kwargs(cls, **kwargs):
    """
    Generate a dictionary of class schema arguments.

    Args:
        cls: The dataclass.
        **kwargs: Additional keyword arguments.

    Returns:
        dict: A dictionary of class schema arguments.
    """
    data = {key: None for key in cls.__annotations__.keys()}
    data.update(kwargs)
    return data


def sqlow_base_init(database: str):
    """
    Initialize the SQLow database decorator.

    Args:
        database (str): The name of the database.

    Returns:
        function: The SQLow database decorator.
    """

    def sqlow_database(_class=None, **params):
        """Decorator with (Optional-Arguments)."""

        # Optional Arguments
        if _class is None:
            return functools.partial(sqlow_database, **params)

        # The Wrapper
        @functools.wraps(_class)
        def the_wrapper(*args, **kwargs):
            cls = merge_data_classes(database, _class, [PreDefinedClass], **params)
            data = class_schema_kwargs(cls, **kwargs)
            return cls(*args, **data).db

        # Return @Decorator
        return the_wrapper

    return sqlow_database


def merge_data_classes(database, new_class, class_list, **config):
    """
    Merge multiple data classes into a single class.

    Args:
        database (str): The name of the database.
        new_class: The new data class.
        class_list: A list of existing data classes.
        **config: Additional configuration.

    Returns:
        type: The merged class.
    """
    class_list.append(dataclass(new_class))
    merged_annotations = {}
    merged_attrs = {}

    for cls in class_list:
        merged_annotations.update(getattr(cls, "__annotations__", {}))

    merged_attrs["__annotations__"] = merged_annotations

    _class = type(
        new_class.__name__, tuple(class_list[::-1]), merged_attrs
    )  # Reverse the order of class_list

    # Database Setup
    decorator_config(_class, config)
    _class.db = SQLowDatabase(db_name=database, table_class=_class)

    return _class


def decorator_config(_class, config: dict):
    """
    Configure the decorator for the data class.

    Args:
        _class: The data class.
        config (dict): Configuration settings.

    Returns:
        None
    """
    table_name = config.get("table_name", _class.__name__)
    _class = dataclass(_class)
    config["table_name"] = table_name
    config = types.SimpleNamespace(**config)
    _class.__objconfig__ = config
    return None


def sqlow(database: str):
    """
    Initialize the SQLow decorator.

    Args:
        database (str): The name of the database.

    Returns:
        function: The SQLow decorator.
    """
    # ... (Function implementation)
    return sqlow_base_init(database)
