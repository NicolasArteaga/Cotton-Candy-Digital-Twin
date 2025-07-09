from bottle import Bottle, run
import serial

app = Bottle()

# Returns True if the weight is greater than or equal to 1.0 gram
# Signaling that the scale has been touched
def weigh_touch():
    try:
        ser = serial.Serial('/dev/ttyUSB0', baudrate=9600, bytesize=8, parity='N', stopbits=1, timeout=2)
        recv = ser.readline().decode('utf-8', errors='ignore').strip()
        ser.reset_input_buffer()
        try:
            weight = float(recv)
            return weight >= 1.0
        except ValueError:
            return False
    except Exception:
        return False


@app.route('/')
def index():
    return str(weigh_touch())

if __name__ == '__main__':
    run(app, host='0.0.0.0', port=7204, debug=True)