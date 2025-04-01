import time
import board
import busio
from adafruit_tca9548a import TCA9548A
from adafruit_hdc3021 import HDC3021
from smbus2 import SMBus
import paho.mqtt.client as mqtt

# MLX90614 driver
class MLX90614:
    def __init__(self, bus):
        self.bus = bus
        self.address = 0x5A

    def read_word(self, reg):
        data = self.bus.read_word_data(self.address, reg)
        return ((data << 8) & 0xFF00) + (data >> 8)

    def read_temp(self, reg):
        raw = self.read_word(reg)
        return raw * 0.02 - 273.15

    def read_object_temp(self):
        return self.read_temp(0x07)

    def read_ambient_temp(self):
        return self.read_temp(0x06)

# Setup main I2C bus and TCA9548A
i2c = busio.I2C(board.SCL, board.SDA)
tca = TCA9548A(i2c)

# Attach HDC3021 sensors to channel 0 and 1
hdc1 = HDC3021(tca[0])
hdc2 = HDC3021(tca[1])

# Attach MLX90614 sensors to channel 2 and 3
bus = SMBus(1)
mlx1 = MLX90614(tca[2])
mlx2 = MLX90614(tca[3])

# MQTT setup
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC_PREFIX = "sensors"

client = mqtt.Client()
client.connect(MQTT_BROKER, MQTT_PORT, 60)

# Loop to read and publish values
while True:
    t1, h1 = hdc1.temperature, hdc1.relative_humidity
    t2, h2 = hdc2.temperature, hdc2.relative_humidity
    a1, o1 = mlx1.read_ambient_temp(), mlx1.read_object_temp()
    a2, o2 = mlx2.read_ambient_temp(), mlx2.read_object_temp()

    print("=== Sensor Readings ===")
    print(f"HDC1 Temp: {t1:.2f} °C, Humidity: {h1:.2f}%")
    print(f"HDC2 Temp: {t2:.2f} °C, Humidity: {h2:.2f}%")
    print(f"IR1 Ambient: {a1:.2f} °C, Object: {o1:.2f} °C")
    print(f"IR2 Ambient: {a2:.2f} °C, Object: {o2:.2f} °C")

    # Publish to MQTT
    client.publish(f"{MQTT_TOPIC_PREFIX}/hdc1/temperature", f"{t1:.2f}")
    client.publish(f"{MQTT_TOPIC_PREFIX}/hdc1/humidity", f"{h1:.2f}")
    client.publish(f"{MQTT_TOPIC_PREFIX}/hdc2/temperature", f"{t2:.2f}")
    client.publish(f"{MQTT_TOPIC_PREFIX}/hdc2/humidity", f"{h2:.2f}")
    client.publish(f"{MQTT_TOPIC_PREFIX}/ir1/ambient", f"{a1:.2f}")
    client.publish(f"{MQTT_TOPIC_PREFIX}/ir1/object", f"{o1:.2f}")
    client.publish(f"{MQTT_TOPIC_PREFIX}/ir2/ambient", f"{a2:.2f}")
    client.publish(f"{MQTT_TOPIC_PREFIX}/ir2/object", f"{o2:.2f}")

    time.sleep(2)