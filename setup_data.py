import os
import pandas as pd

# If 'data' exists as a file instead of a folder, remove it
if os.path.exists("data") and not os.path.isdir("data"):
    os.remove("data")

# Ensure the data/processed directory exists safely
os.makedirs(r"C:\Users\Hp\runway\data\processed", exist_ok=True)

print("Directory check passed successfully!")