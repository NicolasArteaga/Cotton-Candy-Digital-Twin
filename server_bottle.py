from bottle import Bottle, run, response, request
import board
import busio
import adafruit_hdc302x
import adafruit_tca9548a
import adafruit_mlx90614
import statistics
import time
from datetime import datetime
import json
import os

# Setup I2C bus
i2c = busio.I2C(board.SCL, board.SDA)
multiplexer = adafruit_tca9548a.TCA9548A(i2c)

# Access channels
channel_0 = multiplexer[0]  # Env Hum Sensor
channel_1 = multiplexer[1]  # Inside Hum Sensor
channel_2 = multiplexer[2]  # IR Sensor 1
channel_3 = multiplexer[3]  # IR Sensor 2

# Initialize sensors
try: 
    hdc0 = adafruit_hdc302x.HDC302x(channel_0)
    hdc1 = adafruit_hdc302x.HDC302x(channel_1)
    mlx1 = adafruit_mlx90614.MLX90614(channel_3)
    mlx = adafruit_mlx90614.MLX90614(channel_2)
except Exception as e:
    print(f"Error initializing sensors, probably one cable disconnected itself: {e}")

def read_all_sensors():
    entry = {}

    try:
        entry['EnvH'] = round(hdc0.relative_humidity, 2)
        entry['EnvT'] = round(hdc0.temperature, 2)
    except Exception as e:
        print(f"Error reading hdc0: {e}")
        entry['EnvH'] = None
        entry['EnvT'] = None

    try:
        entry['InH'] = round(hdc1.relative_humidity, 2)
        entry['InT'] = round(hdc1.temperature, 2)
    except Exception as e:
        print(f"Error reading hdc1: {e}")
        entry['InH'] = None
        entry['InT'] = None

    # Commented out IR sensor readings (channel 2 and 3)
    try:
       entry['IrA1'] = round(mlx.ambient_temperature, 2)
       entry['IrO1'] = round(mlx.object_temperature, 2)
    except Exception as e:
       print(f"Error reading mlx: {e}")
       entry['IrA1'] = None
       entry['IrO1'] = None

    try:
       entry['IrA2'] = round(mlx1.ambient_temperature, 2)
       entry['IrO2'] = round(mlx1.object_temperature, 2)
    except Exception as e:
       print(f"Error reading mlx1: {e}")
       entry['IrA2'] = None
       entry['IrO2'] = None

    entry['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f") 

    return entry

# File to store data
log_file = "sensor_log.json"
if not os.path.exists(log_file):
    with open(log_file, 'w') as f:
        json.dump([], f)

# Helper to append JSON data
def append_log(entry):
    with open(log_file, 'r+') as f:
        data = json.load(f)
        data.append(entry)
        f.seek(0)
        json.dump(data, f, indent=2)
        f.truncate()

# Read sensors
def read_all_sensors():
    entry = {}

    now = datetime.now()
    ms_two = f"{int(now.microsecond / 1000):03d}"[:2]
    entry['timestamp'] = now.strftime("%Y-%m-%d %H:%M:%S{ms_two}")  # Timestamp with milliseconds

    try:
        entry['EnvH'] = round(hdc0.relative_humidity, 2)
        entry['EnvT'] = round(hdc0.temperature, 2)
    except Exception as e:
        print(f"Error reading hdc0: {e}")
        entry['EnvH'] = None
        entry['EnvT'] = None

    try:
        entry['InH'] = round(hdc1.relative_humidity, 2)
        entry['InT'] = round(hdc1.temperature, 2)
    except Exception as e:
        print(f"Error reading hdc1: {e}")
        entry['InH'] = None
        entry['InT'] = None

    try:
        entry['IrA1'] = round(mlx.ambient_temperature, 2)
        entry['IrO1'] = round(mlx.object_temperature, 2)
    except Exception as e:
        print(f"Error reading mlx: {e}")
        entry['IrA1'] = None
        entry['IrO1'] = None

    try:
        entry['IrA2'] = round(mlx1.ambient_temperature, 2)
        entry['IrO2'] = round(mlx1.object_temperature, 2)
    except Exception as e:
        print(f"Error reading mlx1: {e}")
        entry['IrA2'] = None
        entry['IrO2'] = None

    return entry

# Create Bottle app
app = Bottle()

@app.route('/')
def read_and_store():
    reading = read_all_sensors()
    append_log(reading)
    return reading

@app.route('/history')
def get_history():
    with open(log_file, 'r') as f:
        data = json.load(f)
    return data

if __name__ == '__main__':
    run(app, host='0.0.0.0', port=7201, debug=True)
