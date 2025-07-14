from bottle import Bottle, run
from time import time
import serial
import json

#Do you want to have the two decimal places in each output?

app = Bottle()

ser = serial.Serial('/dev/ttyUSB0', baudrate=9600, bytesize=8, parity='N', stopbits=1, timeout=1)

@app.route('/')
def index():
    try:
        ser.reset_input_buffer()
        start_time = time()
        max_val = None
        weights = []
        while time() - start_time < 3:
            raw = ser.readline().decode('utf-8', errors='ignore').strip()
            if raw:
                try:
                    val = float(raw.replace("g", "").replace("+", "").strip())
                    weights.append(val)
                    if (max_val is None) or (val > max_val):
                        max_val = val
                except ValueError:
                    continue
        result = {
            'weight_max': (round(max_val, 2) if max_val is not None else None),
            'weights': f"{[round(w, 2) for w in weights]}"
        }
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": f"Measurement failed. Please check the scale. Error: {e}"})

if __name__ == '__main__':
    run(app, host='0.0.0.0', port=7203, debug=True)