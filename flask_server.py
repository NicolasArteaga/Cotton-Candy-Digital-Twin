import time
import board
import busio
import adafruit_hdc302x
import adafruit_tca9548a
import adafruit_mlx90614
import statistics
import time
import csv
import os
from flask import Flask, request
import requests

# Setup I2C bus
i2c = busio.I2C(board.SCL, board.SDA)

# Initialize PCA9548A multiplexer
multiplexer = adafruit_tca9548a.TCA9548A(i2c)

# Access a channel
channel_0 = multiplexer[0]  # For first sensor
channel_1 = multiplexer[1]  # For second sensor
channel_2 = multiplexer[2]  # For the IR
channel_3 = multiplexer[3]  # For the IR 2


# Initialize sensors
#mlx = adafruit_mlx90614.MLX90614(i2c)
hdc0 = adafruit_hdc302x.HDC302x(channel_0)
hdc1 = adafruit_hdc302x.HDC302x(channel_1)
mlx = adafruit_mlx90614.MLX90614(channel_2)
mlx1 = adafruit_mlx90614.MLX90614(channel_3)

try:
    var = hdc0.relative_humidity
except Exception as e:
    print("Exception")


app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    humidity = hdc0.relative_humidity
    print("Current Humidity:", humidity)
    # Directly return the humidity as a string response
    return str(humidity)


@app.route('/sugarpi/', methods=['POST'])
def receive_data():
    try:
        humidity = hdc0.relative_humidity
        print("Current Humidity:", humidity)
        # You can also read other sensor data here if needed
        return jsonify({
            "status": "success",
            "humidity": humidity
        }), 200
    except Exception as e:
        print("Error reading sensor data:", e)
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7201)

