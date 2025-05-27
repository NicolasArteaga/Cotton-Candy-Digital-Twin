import time
import board
import busio
import adafruit_hdc302x
import adafruit_tca9548a
import adafruit_mlx90614

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

# Read loop
while True:
    print(f"1. HDC302x0 Humidity: {hdc0.relative_humidity:.2f} %")
    print(f"1. HDC302x0 Temp: {hdc0.temperature:.2f} °C")
    print(f"2. MLX90614 Ambient Temp: {mlx.ambient_temperature:.2f} °C")
    print(f"2. MLX90614 Object Temp: {mlx.object_temperature:.2f} °C")
    print(f"3. MLX90614 Ambient Temp: {mlx1.ambient_temperature:.2f} °C")
    print(f"3. MLX90614 Object Temp: {mlx1.object_temperature:.2f} °C")
    print(f"4. HDC302x1 Humidity: {hdc1.relative_humidity:.2f} %")
    print(f"4. HDC302x1 Temp: {hdc1.temperature:.2f} °C")
    print("---------------------")
    time.sleep(1)
