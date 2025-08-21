import yaml
from typing import List, Dict, Optional

def extract_stream_data_only(yaml_file_path: str) -> Dict[str, List[Dict]]:
    """
    Extract only stream/data events for plug and environment data from YAML file.
    
    Args:
        yaml_file_path: Path to the YAML file
        
    Returns:
        Dict with 'plug_data' and 'environment_data' lists containing extracted measurements
    """
    plug_data = []
    environment_data = []
    
    with open(yaml_file_path, 'r') as f:
        for doc in yaml.safe_load_all(f):
            if 'event' not in doc:
                continue
                
            event = doc['event']
            activity_name = event.get('concept:name', '')
            lifecycle_transition = event.get('cpee:lifecycle:transition', '')
            
            # Only process stream/data events
            if lifecycle_transition != 'stream/data':
                continue
            
            timestamp = event.get('time:timestamp', '')
            
            # Extract plug data
            if activity_name == 'Get the Plug Data':
                plug_measurement = extract_plug_stream_data(event, timestamp)
                if plug_measurement:
                    plug_data.append(plug_measurement)
            
            # Extract environment data
            elif activity_name == 'Get the Environment Data':
                env_measurement = extract_environment_stream_data(event, timestamp)
                if env_measurement:
                    environment_data.append(env_measurement)
    
    return {
        'plug_data': plug_data,
        'environment_data': environment_data
    }

def extract_plug_stream_data(event: Dict, timestamp: str) -> Optional[Dict]:
    """Extract power and current measurements from plug stream/data event"""
    stream_datastream = event.get('stream:datastream', [])
    
    for stream_item in stream_datastream:
        if not isinstance(stream_item, dict):
            continue
            
        # Look for nested plug stream
        nested_stream = stream_item.get('stream:datastream', [])
        if not isinstance(nested_stream, list):
            continue
            
        # Check if this contains plug data
        has_plug = any(
            item.get('stream:name') == 'plug' 
            for item in nested_stream 
            if isinstance(item, dict)
        )
        
        if not has_plug:
            continue
            
        # Extract power and current values
        measurement = {'timestamp': timestamp}
        
        for nested_item in nested_stream:
            if not isinstance(nested_item, dict) or 'stream:point' not in nested_item:
                continue
                
            point = nested_item['stream:point']
            if not isinstance(point, dict):
                continue
                
            stream_id = point.get('stream:id')
            stream_value = point.get('stream:value')
            
            if stream_id == 'power' and stream_value is not None:
                try:
                    measurement['power'] = float(stream_value)
                except (ValueError, TypeError):
                    measurement['power'] = stream_value
            elif stream_id == 'current' and stream_value is not None:
                try:
                    measurement['current'] = float(stream_value)
                except (ValueError, TypeError):
                    measurement['current'] = stream_value
        
        if 'power' in measurement:
            return measurement
    
    return None

def extract_environment_stream_data(event: Dict, timestamp: str) -> Optional[Dict]:
    """Extract environmental sensor measurements from environment stream/data event"""
    stream_datastream = event.get('stream:datastream', [])
    measurement = {'timestamp': timestamp}
    
    for stream_item in stream_datastream:
        if not isinstance(stream_item, dict):
            continue
            
        # Look for nested environment streams
        nested_stream = stream_item.get('stream:datastream', [])
        if not isinstance(nested_stream, list):
            continue
            
        # Determine stream type
        stream_type = None
        for nested_item in nested_stream:
            if isinstance(nested_item, dict) and 'stream:name' in nested_item:
                stream_type = nested_item['stream:name']
                break
        
        # Extract data based on stream type
        if stream_type == 'environment':
            # External environmental sensors
            for nested_item in nested_stream:
                if not isinstance(nested_item, dict) or 'stream:point' not in nested_item:
                    continue
                    
                point = nested_item['stream:point']
                if not isinstance(point, dict):
                    continue
                    
                stream_id = point.get('stream:id')
                stream_value = point.get('stream:value')
                
                if stream_id and stream_value is not None:
                    try:
                        # Map humidity -> EnvH, temperature -> EnvT
                        if stream_id == 'humidity':
                            measurement['env_EnvH'] = float(stream_value)
                        elif stream_id == 'temperature':
                            measurement['env_EnvT'] = float(stream_value)
                    except (ValueError, TypeError):
                        pass
        
        elif stream_type == 'internal':
            # Internal sensors
            for nested_item in nested_stream:
                if isinstance(nested_item, dict) and 'stream:point' in nested_item:
                    point = nested_item['stream:point']
                    if isinstance(point, dict):
                        stream_id = point.get('stream:id')
                        stream_value = point.get('stream:value')
                        
                        if stream_id and stream_value is not None:
                            try:
                                # Map humidity -> InH, temperature -> InT
                                if stream_id == 'humidity':
                                    measurement['env_InH'] = float(stream_value)
                                elif stream_id == 'temperature':
                                    measurement['env_InT'] = float(stream_value)
                            except (ValueError, TypeError):
                                pass
                
                # Look for infrared nested stream
                elif isinstance(nested_item, dict) and 'stream:datastream' in nested_item:
                    infrared_stream = nested_item['stream:datastream']
                    if isinstance(infrared_stream, list):
                        # Check if this is infrared
                        is_infrared = any(
                            item.get('stream:name') == 'infrared' 
                            for item in infrared_stream 
                            if isinstance(item, dict)
                        )
                        
                        if is_infrared:
                            for ir_item in infrared_stream:
                                if isinstance(ir_item, dict) and 'stream:point' in ir_item:
                                    point = ir_item['stream:point']
                                    if isinstance(point, dict):
                                        stream_id = point.get('stream:id')
                                        stream_value = point.get('stream:value')
                                        
                                        if stream_id and stream_value is not None:
                                            try:
                                                # Map head -> IrO, ambient -> IrA
                                                if stream_id == 'head':
                                                    measurement['env_IrO'] = float(stream_value)
                                                elif stream_id == 'ambient':
                                                    measurement['env_IrA'] = float(stream_value)
                                            except (ValueError, TypeError):
                                                pass
    
    # Return measurement if we found any sensor data
    sensor_keys = [k for k in measurement.keys() if k != 'timestamp']
    return measurement if sensor_keys else None

# Example usage and testing
if __name__ == "__main__":
    import os
    
    # Test the function
    yaml_file = "Batches/test-batch-0/63693789-3e88-4a7e-91f0-2940bbe3aec0.xes.yaml"
    
    if os.path.exists(yaml_file):
        print(f"Processing {yaml_file}...")
        data = extract_stream_data_only(yaml_file)
        
        print(f"\nFound {len(data['plug_data'])} plug measurements")
        print(f"Found {len(data['environment_data'])} environment measurements")
        
        # Show first few samples
        if data['plug_data']:
            print(f"\nFirst plug measurement: {data['plug_data'][0]}")
            print(f"Last plug measurement: {data['plug_data'][-1]}")
            
            # Show power range
            powers = [m['power'] for m in data['plug_data'] if 'power' in m]
            if powers:
                print(f"Power range: {min(powers)} - {max(powers)} W")
        
        if data['environment_data']:
            print(f"\nFirst environment measurement: {data['environment_data'][0]}")
            print(f"Last environment measurement: {data['environment_data'][-1]}")
            
            # Show available sensors
            all_sensors = set()
            for measurement in data['environment_data']:
                all_sensors.update(k for k in measurement.keys() if k != 'timestamp')
            print(f"Available sensors: {sorted(all_sensors)}")
    else:
        print(f"File {yaml_file} not found")
