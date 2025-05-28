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
#    print(f"Median IR Ambient Temp: {median_ambient:.2f} °C")
#    print(f"Median IR Object Temp: {median_object:.2f} °C")


# Create the CSV file if it doesn't exist, and add the header
if not os.path.exists("seconds_log.csv"):
    with open("seconds_log.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["ElapsedSeconds"])

start_time = time.time()

# Read loop
while True:
    try:
        hdc0_humidity = hdc0.relative_humidity
        hdc0_temp = hdc0.temperature
    except Exception as e:
        hdc0_humidity = None
        hdc0_temp = None
        print(f"Error reading Environment Humidity: {e}")

    try:
        mlx_ambient = mlx.ambient_temperature
        mlx_object = mlx.object_temperature
    except Exception as e:
        mlx_ambient = None
        mlx_object = None
        print(f"Error reading Infrared 1 (channel 2): {e}")

    try:
        mlx1_ambient = mlx1.ambient_temperature
        mlx1_object = mlx1.object_temperature
    except Exception as e:
        mlx1_ambient = None
        mlx1_object = None
        print(f"Error reading Infrared 2 (channel 3): {e}")

    try:
        hdc1_humidity = hdc1.relative_humidity
        hdc1_temp = hdc1.temperature
    except Exception as e:
        hdc1_humidity = None
        hdc1_temp = None
        print(f"Error reading HDC302x1: {e}")
    
    seconds = int(time.time() - start_time)

    # Format the line exactly as the print output
    log_line = (
        f"{seconds}s, "
        f"EnvH:{hdc0_humidity:.2f}%, "
        f"InH:{hdc1_humidity:.2f}%, "
        f"EnvT:{hdc0_temp:.2f}°C, "
        f"InT:{hdc1_temp:.2f}°C, "
        f"IrA1:{mlx_ambient:.2f}°C, "
        f"IrA2:{mlx1_ambient:.2f}°C, "
        f"IrO1:{mlx_object:.2f}°C, "
        f"IrO2:{mlx1_object:.2f}°C"
    )
    print(log_line)  # print to terminal

    # Append to CSV file
    with open("seconds_log.csv", mode='a', newline='') as file:
        writer = csv.writer(file)
        # Split by ',' to get separate columns in the CSV
        writer.writerow(log_line.split(','))
    #try:
    #    print_median_ir_temps()
    #except Exception as e:
    #    print(f"Error getting the median: {e}")
#    print("---------------------")
    time.sleep(1)
