#!/usr/bin/env python3
# Cotton Candy Prescriptive Model Service
# POST to "/" with environmental sensor data to get cooking parameters
# Accepts JSON: {"EnvH": 56.11, "EnvT": 21.83, "InH": 60.33, "InT": 21.88, "IrA": 22.41, "IrO": 21.73, "timestamp": "2025-09-16 23:01:18.35"}
# Returns JSON: {"start_temp": 22.41, "cook_temp": 85.50, "cool_temp": 18.75, "cook_time": 83}

from bottle import Bottle, request, response, HTTPError
import json
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

# --- Optimized prescriptive model logic ---
def calculate_cooking_parameters(env_data):
    """
    Optimized function to calculate cooking parameters based on environmental data.
    Based on analysis of last 60 high-performing iterations from Complete_cc_dataset.csv
    
    Key findings:
    - Lower humidity correlates with higher quality (but cannot be controlled)
    - Cook temp should remain constant around 53°C for optimal results
    - Longer cook times preferred for better quality
    - Start temp varies based on IR temperature conditions
    
    Args:
        env_data (dict): Environmental sensor data
        
    Returns:
        dict: Cooking parameters (start_temp, cook_temp, cool_temp, cook_time)
    """
    env_h = env_data.get('EnvH', 50.0)  # Environmental humidity
    ir_o = env_data.get('IrO', 21.0)    # IR object temperature (before turn on)
    
    # Cook temp remains constant for optimal results (based on data analysis)
    cook_temp = 53.0
    cool_temp = 54.0  # Standard cooling temperature from high-performing iterations
    
    # Determine start_temp based on IR temperature conditions
    if ir_o >= 50.0:
        # Higher IR temp -> slightly higher start temp
        start_temp = 50.0
    elif ir_o >= 40.0:  
        # Medium IR temp -> moderate start temp
        start_temp = 48.0
    else:
        # Lower IR temp -> lower start temp
        start_temp = 45.0
    
    # Calculate cook_time based on environmental conditions
    # Base time from analysis: 73-80 seconds for high-performing iterations
    base_cook_time = 75
    
    # Adjust based on humidity (longer times preferred, especially in higher humidity)
    if env_h <= 45.0:
        # Low humidity (optimal conditions) -> standard time
        cook_time = base_cook_time
    elif env_h <= 55.0:
        # Normal humidity (50-60% is norm) -> slightly longer
        cook_time = base_cook_time + 3
    else:
        # Higher humidity -> compensate with longer cook time
        cook_time = base_cook_time + 5
    
    # Small adjustment based on IR temperature (warmer conditions need less time)
    if ir_o >= 50.0:
        cook_time += 2  # Slightly longer for consistency
    
    return {
        "start_temp": round(start_temp, 1),
        "cook_temp": round(cook_temp, 1), 
        "cool_temp": round(cool_temp, 1),
        "cook_time": int(cook_time)
    }

# --- Core: POST to "/" ---
@app.post("/")
def predict_parameters():
    """
    Accepts environmental sensor data and returns cooking parameters.
    Accepts:
      - form: value=<env_json_string>
      - JSON: {"value":<env_data_object>} or direct env data object
    
    Expected env data format:
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
        # Handle both form and JSON data, similar to cc_registry
        body = request.json or {}
        
        # Try to get env data from different sources
        env_data = None
        
        # 1. From form field 'value' (JSON string)
        form_value = request.forms.get("value")
        if form_value:
            try:
                import json
                env_data = json.loads(form_value)
            except json.JSONDecodeError:
                raise HTTPError(400, "Invalid JSON in form field 'value'")
        
        # 2. From JSON body 'value' field
        elif body.get("value"):
            env_data = body.get("value")
        
        # 3. Direct JSON body (backward compatibility)
        elif body and 'EnvH' in body:
            env_data = body
        
        if not env_data:
            raise HTTPError(400, "Environmental data required as form field 'value' or JSON 'value'")
        
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
        "version": "2.0.0",
        "status": "running",
        "description": "Provides optimized cooking parameters based on environmental sensor data (based on analysis of 60 high-performing iterations)",
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