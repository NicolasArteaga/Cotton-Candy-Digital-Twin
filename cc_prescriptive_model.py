#!/usr/bin/env python3
# Cotton Candy Prescriptive Model Service
# POST to "/" with environmental sensor data to get cooking parameters
# Accepts JSON: {"EnvH": 56.11, "EnvT": 21.83, "InH": 60.33, "InT": 21.88, "IrA": 22.41, "IrO": 21.73, "timestamp": "2025-09-16 23:01:18.35"}
# Returns JSON: {"start_temp": 22.41, "cook_temp": 85.50, "cool_temp": 18.75, "cook_time": 83}

from bottle import Bottle, request, response, HTTPError
import json
import random
from datetime import datetime

app = Bottle()

# --- Settings ---
HOST = "0.0.0.0"
PORT = 7207
CORS_ALLOW = "*"

# --- Helpers ---
def _set_common_headers():
    if response.content_type is None:
        response.content_type = "application/json; charset=utf-8"
    response.set_header("Access-Control-Allow-Origin", CORS_ALLOW)
    response.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
    response.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

@app.hook('after_request')
def _after(): 
    _set_common_headers()

@app.route("/<:re:.*>", method=["OPTIONS"])
def _options(): 
    return {}

# --- Mock prescriptive model logic ---
def calculate_cooking_parameters(env_data):
    """
    Mock function to calculate cooking parameters based on environmental data.
    TODO: Replace with actual machine learning model logic.
    
    Args:
        env_data (dict): Environmental sensor data
        
    Returns:
        dict: Cooking parameters (start_temp, cook_temp, cool_temp, cook_time)
    """
    # For now, return mock values with some variation based on input
    env_h = env_data.get('EnvH', 50.0)
    env_t = env_data.get('EnvT', 20.0)
    in_h = env_data.get('InH', 55.0)
    in_t = env_data.get('InT', 21.0)
    ir_a = env_data.get('IrA', 22.0)
    ir_o = env_data.get('IrO', 21.0)
    
    # Mock calculation with some logic based on environmental conditions
    # Higher humidity -> longer cook time
    # Higher temperature -> adjust cooking temps
    
    base_cook_time = 75
    humidity_factor = (env_h + in_h) / 100.0  # Average humidity as factor
    cook_time = int(base_cook_time + (humidity_factor * 20))  # 75-95 seconds range
    
    # Temperature adjustments
    temp_avg = (env_t + in_t + ir_a + ir_o) / 4.0
    start_temp = round(temp_avg, 2)
    cook_temp = round(temp_avg + 60.0 + random.uniform(-5.0, 5.0), 2)  # ~80-90°C
    cool_temp = round(temp_avg - 3.0 + random.uniform(-2.0, 2.0), 2)   # ~18-22°C
    
    return {
        "start_temp": start_temp,
        "cook_temp": cook_temp,
        "cool_temp": cool_temp,
        "cook_time": cook_time
    }

# --- Core: POST to "/" ---
@app.post("/")
def predict_parameters():
    """
    Accepts environmental sensor data and returns cooking parameters.
    Expected JSON format:
    {
        "EnvH": 56.11,
        "EnvT": 21.83,
        "InH": 60.33,
        "InT": 21.88,
        "IrA": 22.41,
        "IrO": 21.73,
        "timestamp": "2025-09-16 23:01:18.35"
    }
    
    Returns:
    {
        "start_temp": 22.41,
        "cook_temp": 85.50,
        "cool_temp": 18.75,
        "cook_time": 83,
        "timestamp": "2025-09-16 23:01:19.12",
        "status": "ok"
    }
    """
    try:
        # Get JSON data from request
        env_data = request.json
        if not env_data:
            raise HTTPError(400, "JSON data required")
        
        # Validate required fields
        required_fields = ['EnvH', 'EnvT', 'InH', 'InT', 'IrA', 'IrO']
        missing_fields = [field for field in required_fields if field not in env_data]
        if missing_fields:
            raise HTTPError(400, f"Missing required fields: {', '.join(missing_fields)}")
        
        # Calculate cooking parameters
        cooking_params = calculate_cooking_parameters(env_data)
        
        # Add timestamp and status to response
        now = datetime.now()
        ms_two_digits = f"{int(now.microsecond / 1000):03d}"[:2]
        cooking_params.update({
            "timestamp": now.strftime(f"%Y-%m-%d %H:%M:%S.{ms_two_digits}"),
            "status": "ok"
        })
        
        return cooking_params
        
    except HTTPError:
        raise
    except Exception as e:
        raise HTTPError(500, f"Internal error: {str(e)}")

# --- GET to "/" for status/info ---
@app.get("/")
def info():
    """
    Returns service information and example usage.
    """
    info_data = {
        "service": "Cotton Candy Prescriptive Model",
        "version": "1.0.0",
        "status": "running",
        "description": "Provides cooking parameters based on environmental sensor data",
        "endpoint": {
            "url": "/",
            "method": "POST",
            "content_type": "application/json"
        },
        "input_format": {
            "EnvH": "Environmental humidity (%)",
            "EnvT": "Environmental temperature (°C)",
            "InH": "Internal humidity (%)",
            "InT": "Internal temperature (°C)", 
            "IrA": "IR ambient temperature (°C)",
            "IrO": "IR object temperature (°C)",
            "timestamp": "Timestamp (optional)"
        },
        "output_format": {
            "start_temp": "Starting temperature (°C)",
            "cook_temp": "Cooking temperature (°C)",
            "cool_temp": "Cooling temperature (°C)",
            "cook_time": "Cooking time (seconds)",
            "timestamp": "Response timestamp",
            "status": "Response status"
        },
        "example_input": {
            "EnvH": 56.11,
            "EnvT": 21.83,
            "InH": 60.33,
            "InT": 21.88,
            "IrA": 22.41,
            "IrO": 21.73,
            "timestamp": "2025-09-16 23:01:18.35"
        },
        "example_output": {
            "start_temp": 22.41,
            "cook_temp": 85.50,
            "cool_temp": 18.75,
            "cook_time": 83,
            "timestamp": "2025-09-16 23:01:19.12",
            "status": "ok"
        }
    }
    return info_data

if __name__ == "__main__":
    print(f"Starting Cotton Candy Prescriptive Model Service on {HOST}:{PORT}")
    app.run(host=HOST, port=PORT, server="paste")