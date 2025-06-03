import board
import busio
import adafruit_hdc302x
import adafruit_tca9548a
import adafruit_mlx90614

def setup_sensors():
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
    hdc0 = adafruit_hdc302x.HDC302x(channel_0)
    hdc1 = adafruit_hdc302x.HDC302x(channel_1)
    mlx = adafruit_mlx90614.MLX90614(channel_2)
    mlx1 = adafruit_mlx90614.MLX90614(channel_3)

    return hdc0, hdc1, mlx, mlx1
