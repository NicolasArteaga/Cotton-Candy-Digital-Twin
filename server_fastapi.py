from fastapi import FastAPI
import board
import busio
import adafruit_hdc302x
import adafruit_tca9548a
import adafruit_mlx90614
from datetime import date

# Setup I2C bus
i2c = busio.I2C(board.SCL, board.SDA)

# Initialize PCA9548A multiplexer
multiplexer = adafruit_tca9548a.TCA9548A(i2c)

# Access sensor through multiplexer
channel_0 = multiplexer[0]
hdc0 = adafruit_hdc302x.HDC302x(channel_0)

# Initialize FastAPI
app = FastAPI()

@app.get("/")
def get_humidity():
    humidity = hdc0.relative_humidity
    print("Current Humidity:", humidity)
    print("Current Time:", date.today())
    return {"humidity": humidity}
