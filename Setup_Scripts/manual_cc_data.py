import requests
import time
import csv
import os
from datetime import datetime

url = "https://lab.bpm.in.tum.de/sugarpi/"
filename = datetime.now().strftime("%Y%m%d_%H%M%S") + "_sugarpi_data.csv"

# Define the expected keys
fields = ["timestamp", "EnvH", "EnvT", "InH", "InT", "IrA", "IrO"]

# Initialize CSV
if not os.path.exists(filename):
    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(fields)

# Main loop
while True:
    try:
        response = requests.get(url)
        data = response.json()
        row = [f"{data.get(k):.2f}" if isinstance(data.get(k), (float, int)) else data.get(k) for k in fields]

        with open(filename, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

        print(f"[{data.get('timestamp')}] Row written.")
    except Exception as e:
        print(f"Error: {e}")

    time.sleep(1)
