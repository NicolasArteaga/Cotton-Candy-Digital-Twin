from bottle import Bottle, run, response
import board
import busio
import adafruit_hdc302x
import adafruit_tca9548a
import adafruit_mlx90614
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

# Initialize sensors
try: 
    hdc0 = adafruit_hdc302x.HDC302x(channel_0)
    hdc1 = adafruit_hdc302x.HDC302x(channel_1)
    mlx = adafruit_mlx90614.MLX90614(channel_2)
except Exception as e:
    print(f"Error initializing sensors, probably one cable disconnected itself: {e}")

# Read sensors
def read_all_sensors():
    entry = {}

    now = datetime.now()
    ms_two_digits = f"{int(now.microsecond / 1000):03d}"[:2]
    entry['timestamp'] = now.strftime(f"%Y-%m-%d %H:%M:%S.{ms_two_digits}")  # Timestamp with milliseconds

    try:
        entry['EnvH'] = round(float(hdc0.relative_humidity), 2)
        entry['EnvT'] = round(float(hdc0.temperature), 2)
    except Exception as e:
        print(f"Error reading hdc0: {e}")
        entry['EnvH'] = None
        entry['EnvT'] = None

    try:
        entry['InH'] = round(float(hdc1.relative_humidity), 2)
        entry['InT'] = round(float(hdc1.temperature), 2)
    except Exception as e:
        print(f"Error reading hdc1: {e}")
        entry['InH'] = None
        entry['InT'] = None

    # IrA is ambient temperature, IrO is object temperature
    try:
        entry['IrA'] = round(float(mlx.ambient_temperature), 2)
        entry['IrO'] = round(float(mlx.object_temperature), 2)
    except Exception as e:
        print(f"Error reading mlx: {e}")
        entry['IrA'] = None
        entry['IrO'] = None

    return entry

# Create Bottle app
app = Bottle()

@app.route('/')
def read():
    data = read_all_sensors()
    # Debug: print the actual types
    for key, value in data.items():
        if key != 'timestamp':
            print(f"{key}: {value} (type: {type(value)})")
    response.content_type = 'application/json'
    return json.dumps(data)

if __name__ == '__main__':
    run(app, host='0.0.0.0', port=7201, debug=True)
