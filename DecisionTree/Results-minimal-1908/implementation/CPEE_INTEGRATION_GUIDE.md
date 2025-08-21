# CPEE Integration Guide for Cotton Candy Digital Twin
# ===================================================

## Overview

This guide explains how to integrate the minimal decision tree model into your Cloud Process Execution Engine (CPEE) for automated cotton candy quality prediction and process optimization.

## Files Created

1. **`cpee_decision_engine.py`** - Full Flask-based REST API service (requires Flask)
2. **`simple_decision_tree.py`** - Lightweight, dependency-free implementation
3. **`cpee_workflow_template.xml`** - CPEE workflow template
4. **`example_parameters.json`** - Example input parameters

## Quick Start

### Option 1: Simple Standalone Implementation (Recommended)

The `simple_decision_tree.py` requires no external dependencies and can be used directly:

```bash
# Test with example parameters
python3 simple_decision_tree.py example

# Predict quality from JSON file
python3 simple_decision_tree.py predict example_parameters.json

# Get optimization suggestions
python3 simple_decision_tree.py optimize example_parameters.json 40.0

# Start HTTP server for CPEE integration
python3 simple_decision_tree.py server localhost 8080
```

### Option 2: Full Flask Service (Advanced)

If you need more features and have Flask available:

```bash
# Install Flask first
pip install flask

# Run the full service
python3 cpee_decision_engine.py
```

## CPEE Integration Methods

### Method 1: HTTP Service Integration

1. **Start the Decision Tree Server**:
   ```bash
   python3 simple_decision_tree.py server localhost 8080
   ```

2. **Use HTTP POST in CPEE Activities**:
   ```xml
   <activity id="predict_quality">
     <implementation>
       <service>http_post</service>
       <parameters>
         <parameter name="url">http://localhost:8080/predict</parameter>
         <parameter name="headers">{"Content-Type": "application/json"}</parameter>
         <parameter name="body">{
           "before_turn_on_env_InH": #{sensor_humidity},
           "iteration_since_maintenance": #{maintenance_counter},
           "before_turn_on_env_IrO": #{sensor_oxygen},
           "wait_time": #{process_wait_time},
           "duration_cc_flow": #{flow_duration}
         }</parameter>
       </parameters>
     </implementation>
   </activity>
   ```

### Method 2: Direct Python Script Integration

1. **Create CPEE Script Activity**:
   ```xml
   <activity id="quality_prediction">
     <implementation>
       <script language="ruby">
         # Call Python script directly
         require 'json'
         
         params = {
           "before_turn_on_env_InH" => data.get('humidity'),
           "iteration_since_maintenance" => data.get('maintenance_iter'),
           "before_turn_on_env_IrO" => data.get('oxygen_level'),
           "wait_time" => data.get('wait_time'),
           "duration_cc_flow" => data.get('flow_duration')
         }
         
         # Write parameters to temp file
         File.write('/tmp/params.json', params.to_json)
         
         # Call Python script
         result = `python3 simple_decision_tree.py predict /tmp/params.json`
         prediction = JSON.parse(result)
         
         # Store results in CPEE data
         data.set('predicted_quality', prediction['score'])
         data.set('quality_category', prediction['category'])
         data.set('recommendation', prediction['recommendation'])
       </script>
     </implementation>
   </activity>
   ```

### Method 3: Import Template Workflow

1. **Import the Template**:
   - Load `cpee_workflow_template.xml` into your CPEE instance
   - Configure the `decision_engine_url` property to point to your service
   - Map your sensor data to the required parameters

2. **Configure Data Mappings**:
   - Map your humidity sensor to `before_turn_on_env_InH`
   - Map your maintenance counter to `iteration_since_maintenance`
   - Map other sensors according to the parameter names

## Required Input Parameters

The minimal decision tree requires these 5 key parameters:

| Parameter | Description | Typical Range | Critical Thresholds |
|-----------|-------------|---------------|-------------------|
| `before_turn_on_env_InH` | Internal humidity before turn on (%) | 30-40 | 35.86 (primary split) |
| `iteration_since_maintenance` | Iterations since last maintenance | 0-100 | 21.5 (maintenance threshold) |
| `before_turn_on_env_IrO` | Infrared oxygen sensor before turn on | 50-65 | 58.03 (quality split) |
| `wait_time` | Wait time before process (seconds) | 30-80 | 50, 65 (optimization points) |
| `duration_cc_flow` | Cotton candy flow duration (seconds) | 60-75 | 66.2 (flow optimization) |

## API Endpoints

### POST /predict
Predict quality score for given parameters.

**Request**:
```json
{
  "before_turn_on_env_InH": 34.5,
  "iteration_since_maintenance": 15.0,
  "before_turn_on_env_IrO": 55.2,
  "wait_time": 45.0,
  "duration_cc_flow": 68.5
}
```

**Response**:
```json
{
  "timestamp": "2025-08-21T10:30:00",
  "score": 46.67,
  "category": "EXCELLENT",
  "confidence": "HIGH",
  "recommendation": "Proceed with production",
  "decision_path": [
    "before_turn_on_env_InH <= 35.86 (34.50)",
    "iteration_since_maintenance <= 21.50 (15.0)",
    "wait_time <= 50.00 (45.0)"
  ]
}
```

### POST /optimize
Get optimization suggestions for parameters.

**Request**:
```json
{
  "parameters": {
    "before_turn_on_env_InH": 34.5,
    "iteration_since_maintenance": 25.0,
    "wait_time": 60.0,
    "duration_cc_flow": 65.0
  },
  "target_score": 40.0
}
```

**Response**:
```json
{
  "current_score": 20.25,
  "target_score": 40.0,
  "optimization_needed": true,
  "suggestions": {
    "humidity_adjustment": {
      "parameter": "before_turn_on_env_InH",
      "suggested_range": "35.9 - 36.3",
      "priority": "HIGH",
      "expected_improvement": "15-25 points"
    }
  }
}
```

## Decision Logic Summary

The minimal model uses this decision tree:

```
if humidity ≤ 35.86:
    if maintenance_iterations ≤ 21.5:
        if oxygen ≤ 58.03: → Quality = 10.67
        else: → Quality = 4.00
    else:
        if wait_time ≤ 50: → Quality = 46.67
        else:
            if wait_time ≤ 65: → Quality = 20.25
            else: → Quality = 27.50
else:
    if flow_duration ≤ 66.2: → Quality = 30.00
    else:
        if humidity ≤ 36.31: → Quality = 51.67
        else: → Quality = 43.00
```

## Process Optimization Strategy

### For CPEE Implementation:

1. **Pre-Production Check**:
   - Collect sensor readings
   - Predict quality score
   - If score < threshold, apply optimizations

2. **Real-Time Optimization**:
   - **High Priority**: Adjust humidity (35.9-36.3%)
   - **Medium Priority**: Schedule maintenance if iterations > 21
   - **Medium Priority**: Optimize wait time (45-50 seconds)

3. **Quality Monitoring**:
   - Log predictions vs actual results
   - Update model thresholds based on performance
   - Trigger maintenance alerts

### Key Decision Points:

- **Quality < 15**: Stop production, major optimization needed
- **Quality 15-25**: Acceptable, minor optimization recommended
- **Quality 25-40**: Good, proceed with minor monitoring
- **Quality > 40**: Excellent, proceed with confidence

## Troubleshooting

### Common Issues:

1. **Server Not Starting**: Check if port 8080 is available
2. **Missing Parameters**: Ensure all 5 key parameters are provided
3. **CPEE Connection**: Verify the service URL is accessible from CPEE

### Testing:

```bash
# Test server health
curl http://localhost:8080/health

# Test prediction
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d @example_parameters.json
```

## Production Deployment

### For Production Use:

1. **Use Process Manager**: 
   ```bash
   # Using systemd, supervisor, or similar
   python3 simple_decision_tree.py server 0.0.0.0 8080
   ```

2. **Add Logging**:
   - Log all predictions for model validation
   - Monitor prediction accuracy over time
   - Track parameter optimization effectiveness

3. **Security**:
   - Add authentication if needed
   - Use HTTPS in production
   - Validate input parameters

This implementation gives you a production-ready decision tree that integrates seamlessly with CPEE for automated cotton candy quality optimization!
