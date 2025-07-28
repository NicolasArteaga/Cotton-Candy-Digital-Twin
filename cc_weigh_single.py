from bottle import Bottle, run, response
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
        # For 0.5 seconds, store the biggest value that the scale shows and return it
        while time() - start_time < 0.5:
            raw = ser.readline().decode('utf-8', errors='ignore').strip()
            if raw:
                try:
                    val = float(raw.replace("g", "").replace("+", "").replace("-", "").strip())
                    # Treat any originally negative values as 0
                    if "-" in raw:
                        val = 0.0
                    # Handle negative values - take the maximum (least negative or most positive)
                    if (max_val is None) or (val > max_val):
                        max_val = val
                except ValueError:
                    continue
        if max_val is not None:
            response.content_type = 'application/json'
            # Return the maximum value found, even if it's negative
            return {'weight' : f"{max_val:.2f}"}
        else:
            response.content_type = 'application/json'
            return {'weight': "0.00"}  # No valid weight detected
    except Exception as e:
        return f"Measurement failed. Please check the scale. Error: {e}"

if __name__ == '__main__':
    run(app, host='0.0.0.0', port=7202, debug=True)