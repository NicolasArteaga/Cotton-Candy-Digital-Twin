import serial

ser = serial.Serial('/dev/ttyUSB0', baudrate=9600, bytesize=8, parity='N', stopbits=1, timeout=None)

count = 0
try:
    while True:
        recv = ser.readline().decode('utf-8', errors='ignore').strip()
        ser.reset_input_buffer()
        print(f"{count}: {recv}")
        count += 1
except KeyboardInterrupt:
    ser.close()
    print("Serial port closed.")

