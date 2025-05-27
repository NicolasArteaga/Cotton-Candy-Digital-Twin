import time
import board
import busio
import adafruit_hdc302x
import adafruit_tca9548a
import adafruit_mlx90614
import statistics

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

def print_median_ir_temps():
    ambient_temps = [mlx.ambient_temperature, mlx1.ambient_temperature]
    object_temps = [mlx.object_temperature, mlx1.object_temperature]
    median_ambient = statistics.median(ambient_temps)
    median_object = statistics.median(object_temps)
    print(f"Median IR Ambient Temp: {median_ambient:.2f} °C")
    print(f"Median IR Object Temp: {median_object:.2f} °C")


# Read loop
while True:
    try:
        print(f"1. Env Humidity: {hdc0.relative_humidity:.2f} %")
        print(f"1. Env Temp: {hdc0.temperature:.2f} °C")
    except Exception as e:
        print(f"Error reading Environment Humidity: {e}")

    try:
        print(f"2. IR Ambient Temp: {mlx.ambient_temperature:.2f} °C")
        print(f"2. IR Object Temp: {mlx.object_temperature:.2f} °C")
    except Exception as e:
        print(f"Error reading Infrared 1 (channel 2): {e}")

    try:
        print(f"3. IR Ambient Temp: {mlx1.ambient_temperature:.2f} °C")
        print(f"3. IR Object Temp: {mlx1.object_temperature:.2f} °C")
    except Exception as e:
        print(f"Error reading Infrared 2 (channel 3): {e}")

    try:
        print(f"4. HDC302x1 Humidity: {hdc1.relative_humidity:.2f} %")
        print(f"4. HDC302x1 Temp: {hdc1.temperature:.2f} °C")
    except Exception as e:
        print(f"Error reading HDC302x1: {e}")

    print("---------------------")
    try:
        print_median_ir_temps()
    except Exception as e:
        print(f"Error getting the median: {e}")
    print("---------------------")
    time.sleep(1)
