from bottle import Bottle, run
from datetime import datetime
from time import sleep, time
import serial

ser = serial.Serial('/dev/ttyUSB0', baudrate=9600, bytesize=8, parity='N', stopbits=1, timeout=None)

def parse_weight(raw):
    return float(raw.replace("g", "").replace("+", "").strip())

def timestamp():
    now = datetime.now()
    return f"{now.strftime('%Y-%m-%d %H:%M:%S')}.{now.microsecond // 10000:02d}"

def measure_weight():
    cup = ser.readline().decode('utf-8', errors='ignore').strip()
    cup_value = parse_weight(cup)

    #Wait 10 seconds so that the robot arm can place the cotton candy on the scale
    sleep(10)
    
    ser.reset_input_buffer()
    start_time = time()
    max_value = cup_value

    #For 2 seconds store the biggest value that the scale shows and return it
    while time() - start_time < 2:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        try:
            cc_value = parse_weight(line)
            if cc_value > max_value:
                max_value = cc_value
        except ValueError:
            continue  # skip invalid lines

    delta = max_value - cup_value
    return delta

app = Bottle()

@app.route('/')
def index():
    delta = measure_weight()
    return f"{delta:.2f}"


if __name__ == '__main__':
    run(app, host='0.0.0.0', port=7202, debug=True)