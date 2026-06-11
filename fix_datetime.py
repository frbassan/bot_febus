import re

with open("combined_app.py", "r") as f:
    text = f.read()

# Remove inline imports
text = re.sub(r'^[ \t]*import datetime\n', '', text, flags=re.MULTILINE)
# Replace datetime.datetime with datetime
text = text.replace('datetime.datetime', 'datetime')

with open("combined_app.py", "w") as f:
    f.write(text)
