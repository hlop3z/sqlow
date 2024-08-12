Handling Complex Data Types

```py
import json
from datetime import datetime

# Prepare the data
data = {
    "name": "John Doe",
    "age": 30,
    "city": "New York",
    "is_student": False,
    "courses": ["Math", "Science", "English"],
    "registration_date": datetime.now()
}

# Custom serialization function
def serialize_datetime(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError("Type not serializable")

# Write the data to a JSON file
with open('data.json', 'w') as json_file:
    json.dump(data, json_file, indent=4, default=serialize_datetime)

```
