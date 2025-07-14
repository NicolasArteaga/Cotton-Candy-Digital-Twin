from bottle import Bottle, run
from datetime import datetime
from time import sleep, time
import serial

ser = serial.Serial('/dev/ttyUSB0', baudrate=9600, bytesize=8, parity='N', stopbits=1, timeout=None)    

app = Bottle()

@app.route('/')
def index():
    try:
        ser.reset_input_buffer()
        start_time = time()
        #For 2 seconds store the biggest value that the scale shows and return it
        while time() - start_time < 1:
            raw = ser.readline().decode('utf-8', errors='ignore').strip()
            ser.reset_input_buffer()
            return float(raw.replace("g", "").replace("+", "").strip())
    except Exception:
        return "Measurement failed. Please check the scale."        
    

if __name__ == '__main__':
    run(app, host='0.0.0.0', port=7202, debug=True)