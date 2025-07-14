from bottle import Bottle, run
from time import time
import serial

ser = serial.Serial('/dev/ttyUSB0', baudrate=9600, bytesize=8, parity='N', stopbits=1, timeout=None)  # Set timeout to 1 second

app = Bottle()

@app.route('/')
def index():
    try:
        ser.reset_input_buffer()
        start_time = time()
        max_val = None
        # For 2 seconds, store the biggest value that the scale shows and return it
        while time() - start_time < 0.5:
            raw = ser.readline().decode('utf-8', errors='ignore').strip()
            if raw:
                try:
                    val = float(raw.replace("g", "").replace("+", "").strip())
                    if (max_val is None) or (val > max_val):
                        max_val = val
                except ValueError:
                    continue
        if max_val is not None:
            return f"{max_val:.2f}"
        else:
            return "0.00"  # No valid weight detected
    except Exception as e:
        return f"Measurement failed. Please check the scale. Error: {e}"

if __name__ == '__main__':
    run(app, host='0.0.0.0', port=7202, debug=True)