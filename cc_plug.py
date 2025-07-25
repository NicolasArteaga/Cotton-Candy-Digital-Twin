from bottle import Bottle, run, response
import paho.mqtt.client as mqtt
import json
import time
import threading
from datetime import datetime

# MQTT Configuration
MQTT_BROKER = "lab.bpm.in.tum.de"
MQTT_PORT = 1883
SOCKET_ID = "socket-3"
COMMAND_TOPIC = f"/lab-power/{SOCKET_ID}/cmnd/STATUS"
STATUS_TOPIC = f"/lab-power/{SOCKET_ID}/STATUS10"

# Global variable to store the latest power data
latest_power_data = {
    'timestamp': None,
    'power_status': None,
    'connected': False
}

# MQTT Client setup
def on_connect(client, userdata, flags, rc):
    print(f"Connected to MQTT broker with result code {rc}")
    # Subscribe to the status topic
    client.subscribe(STATUS_TOPIC)
    latest_power_data['connected'] = True

def on_disconnect(client, userdata, rc):
    print(f"Disconnected from MQTT broker with result code {rc}")
    latest_power_data['connected'] = False

def on_message(client, userdata, msg):
    try:
        # Parse the received message
        message = msg.payload.decode('utf-8')
        print(f"Received message: {message} on topic: {msg.topic}")
        
        # Update timestamp
        now = datetime.now()
        ms_two_digits = f"{int(now.microsecond / 1000):03d}"[:2]
        latest_power_data['timestamp'] = now.strftime(f"%Y-%m-%d %H:%M:%S.{ms_two_digits}")
        
        # Parse and store the power status data
        try:
            # Try to parse as JSON first
            power_data = json.loads(message)
            latest_power_data['power_status'] = power_data
        except json.JSONDecodeError:
            # If not JSON, store as raw message
            latest_power_data['power_status'] = message
            
    except Exception as e:
        print(f"Error processing MQTT message: {e}")

# Initialize MQTT client
mqtt_client = mqtt.Client()
mqtt_client.on_connect = on_connect
mqtt_client.on_disconnect = on_disconnect
mqtt_client.on_message = on_message

def start_mqtt():
    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        mqtt_client.loop_forever()
    except Exception as e:
        print(f"Error connecting to MQTT broker: {e}")
        latest_power_data['connected'] = False

def request_power_status():
    """Request power status from the socket"""
    try:
        if latest_power_data['connected']:
            mqtt_client.publish(COMMAND_TOPIC, "10")
            print(f"Published STATUS request to {COMMAND_TOPIC}")
            return True
        else:
            print("MQTT client not connected")
            return False
    except Exception as e:
        print(f"Error publishing MQTT message: {e}")
        return False

# Create Bottle app
app = Bottle()

@app.route('/')
def get_power_status():
    try:
        # Request fresh status data
        request_success = request_power_status()
        
        if not request_success:
            response.content_type = 'application/json'
            return {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-4],
                'power_status': None,
                'connected': False,
                'error': 'Failed to request power status'
            }
        
        # Wait a short time for response (max 2 seconds)
        start_time = time.time()
        while time.time() - start_time < 2.0:
            if latest_power_data['timestamp'] is not None:
                # Check if the timestamp is recent (within last 5 seconds)
                try:
                    data_time = datetime.strptime(latest_power_data['timestamp'], "%Y-%m-%d %H:%M:%S.%f")
                    if (datetime.now() - data_time).total_seconds() < 5:
                        break
                except:
                    pass
            time.sleep(0.1)
        
        response.content_type = 'application/json'
        return {
            'timestamp': latest_power_data['timestamp'] or datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-4],
            'power_status': latest_power_data['power_status'],
            'connected': latest_power_data['connected'],
            'socket_id': SOCKET_ID
        }
        
    except Exception as e:
        response.content_type = 'application/json'
        return {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-4],
            'power_status': None,
            'connected': False,
            'error': f'Service error: {e}'
        }

if __name__ == '__main__':
    # Start MQTT client in a separate thread
    mqtt_thread = threading.Thread(target=start_mqtt, daemon=True)
    mqtt_thread.start()
    
    # Give MQTT client time to connect
    time.sleep(2)
    
    print(f"Starting Bottle server on port 7205")
    print(f"MQTT broker: {MQTT_BROKER}:{MQTT_PORT}")
    print(f"Command topic: {COMMAND_TOPIC}")
    print(f"Status topic: {STATUS_TOPIC}")
    
    run(app, host='0.0.0.0', port=7205, debug=True)
