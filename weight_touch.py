from bottle import Bottle, run
import serial

ser = serial.Serial('/dev/ttyUSB0', baudrate=9600, bytesize=8, parity='N', stopbits=1, timeout=None)

app = Bottle()

@app.route('/')
def index():
    return "Welcome to the Weigh Touch Server!"

def weigh_touch():
    count = 0
    try:
        while True:
            recv = ser.readline().decode('utf-8', errors='ignore').strip()
            ser.reset_input_buffer()
            print(f"{count}: {recv}")
            count += 1
            return count
    except KeyboardInterrupt:
        ser.close()
        print("Serial port closed.")

run(app, host='0.0.0.0', port=7202, debug=True)