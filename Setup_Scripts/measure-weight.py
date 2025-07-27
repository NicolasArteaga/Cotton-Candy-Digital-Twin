import serial
from datetime import datetime
from time import sleep, time

def parse_weight(raw):
    return float(raw.replace("g", "").replace("+", "").strip())

def timestamp():
    now = datetime.now()
    return f"{now.strftime('%Y-%m-%d %H:%M:%S')}.{now.microsecond // 10000:02d}"

ser = serial.Serial('/dev/ttyUSB0', baudrate=9600, bytesize=8, parity='N', stopbits=1, timeout=None)

try:
    print("Measuring weight of Cotton Candy")
    cup = ser.readline().decode('utf-8', errors='ignore').strip()
    print(f"{timestamp()}: Baseline weight: {cup}")
    cup_value = parse_weight(cup)

    print("Waiting 5 seconds so you can place the cotton candy...")
    sleep(5)

    print("Reading new weight for 3 seconds...")
    ser.reset_input_buffer()
    start_time = time()
    max_value = cup_value
    max_raw = cup

    while time() - start_time < 3:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        try:
            value = parse_weight(line)
            if value > max_value:
                max_value = value
                max_raw = line
        except ValueError:
            continue  # skip invalid lines

    delta = max_value - cup_value
    print(f"{timestamp()}: Max weight: {max_raw}")
    print(f"{timestamp()}: {max_value} - {cup_value} = {delta:.2f} g")

except KeyboardInterrupt:
    ser.close()
    print("Serial port closed.")
