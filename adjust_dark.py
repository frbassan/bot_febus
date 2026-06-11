import re

with open("combined_app.py", "r") as f:
    text = f.read()

text = text.replace(
    'template="plotly_white",',
    'template="plotly_dark",'
)

text = text.replace(
    'plot_bgcolor="#f8f9fa", paper_bgcolor="white",',
    'plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",'
)

text = text.replace(
    "gridcolor='white', gridwidth=1.5, zeroline=True, zerolinecolor='lightgrey'",
    "gridcolor='rgba(255, 255, 255, 0.3)', gridwidth=1, zeroline=True, zerolinecolor='rgba(255, 255, 255, 0.6)'"
)

with open("combined_app.py", "w") as f:
    f.write(text)

print("Dark mode adjusted!")
