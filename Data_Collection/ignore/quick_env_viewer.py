#!/usr/bin/env python3
"""
Quick Environment Data Viewer - Interactive version
Usage: python quick_env_viewer.py <yaml_file>
"""

import yaml
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import sys
import os

def parse_timestamp(timestamp_str):
    """Parse timestamp string to datetime object."""
    try:
        # Handle different timestamp formats
        if '+' in timestamp_str:
            # Remove timezone offset for simplicity
            timestamp_str = timestamp_str.split('+')[0]
        
        # Try different formats
        formats = [
            '%Y-%m-%d %H:%M:%S.%f',
            '%Y-%m-%d %H:%M:%S'
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(timestamp_str, fmt)
            except ValueError:
                continue
        
        print(f"Could not parse timestamp: {timestamp_str}")
        return None
    except Exception as e:
        print(f"Error parsing timestamp {timestamp_str}: {e}")
        return None

def extract_environment_data(yaml_file_path):
    """Extract all environment data from the YAML file."""
    
    environment_data = []
    
    try:
        with open(yaml_file_path, 'r') as f:
            docs = list(yaml.safe_load_all(f))
        
        for doc in docs:
            if 'event' not in doc:
                continue
                
            event = doc['event']
            
            # Look for environment data events
            if (event.get('concept:name') == 'Get the Environment Data' and 
                event.get('cpee:lifecycle:transition') == 'stream/data'):
                
                # Extract data from stream:datastream
                if 'stream:datastream' in event:
                    datastreams = event['stream:datastream']
                    
                    def extract_from_datastream(ds_list, parent_name=""):
                        """Recursively extract data from datastream structure."""
                        current_name = parent_name
                        points = []
                        
                        for item in ds_list:
                            if isinstance(item, dict):
                                if 'stream:name' in item:
                                    current_name = item['stream:name']
                                elif 'stream:point' in item:
                                    point = item['stream:point']
                                    timestamp_str = point.get('stream:timestamp', '')
                                    sensor_id = point.get('stream:id', '')
                                    value = point.get('stream:value', 0)
                                    
                                    timestamp = parse_timestamp(timestamp_str)
                                    if timestamp and sensor_id:
                                        points.append({
                                            'timestamp': timestamp,
                                            'sensor_type': current_name,
                                            'sensor_id': sensor_id,
                                            'value': float(value),
                                            'measurement': f"{current_name}_{sensor_id}"
                                        })
                                elif 'stream:datastream' in item:
                                    # Recursive call for nested datastreams
                                    nested_points = extract_from_datastream(item['stream:datastream'], current_name)
                                    points.extend(nested_points)
                        
                        return points
                    
                    points = extract_from_datastream(datastreams)
                    environment_data.extend(points)
        
        return environment_data
    
    except Exception as e:
        print(f"Error reading YAML file: {e}")
        return []

def quick_view(yaml_file_path):
    """Quick interactive view of environment data."""
    
    print(f"Loading data from: {yaml_file_path}")
    environment_data = extract_environment_data(yaml_file_path)
    
    if not environment_data:
        print("No environment data found!")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(environment_data)
    df = df.sort_values('timestamp')
    
    # Create a simple time series plot
    plt.figure(figsize=(14, 8))
    
    # Plot all sensors in separate subplots
    sensors = df['measurement'].unique()
    n_sensors = len(sensors)
    
    if n_sensors > 6:
        cols = 3
        rows = (n_sensors + 2) // 3
    else:
        cols = 2
        rows = (n_sensors + 1) // 2
    
    for i, sensor in enumerate(sorted(sensors)):
        plt.subplot(rows, cols, i + 1)
        sensor_data = df[df['measurement'] == sensor]
        
        # Color code by sensor type
        if 'temperature' in sensor:
            color = 'red' if 'environment' in sensor else 'orange' if 'internal' in sensor else 'darkred'
        elif 'humidity' in sensor:
            color = 'blue' if 'environment' in sensor else 'lightblue' if 'internal' in sensor else 'darkblue'
        elif 'infrared' in sensor:
            color = 'purple' if 'head' in sensor else 'magenta'
        else:
            color = 'black'
        
        plt.plot(sensor_data['timestamp'], sensor_data['value'], color=color, linewidth=2)
        plt.title(sensor.replace('_', ' ').title())
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Add units
        if 'temperature' in sensor or 'infrared' in sensor:
            plt.ylabel('°C')
        elif 'humidity' in sensor:
            plt.ylabel('%')
    
    plt.suptitle(f'Environment Data: {os.path.basename(yaml_file_path)}', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Print quick stats
    print(f"\nQuick Stats:")
    print(f"Duration: {df['timestamp'].max() - df['timestamp'].min()}")
    print(f"Total measurements: {len(df)}")
    print(f"Measurement frequency: ~{len(df) / (df['timestamp'].max() - df['timestamp'].min()).total_seconds():.1f} per second")
    
    for measurement in sorted(df['measurement'].unique()):
        sensor_data = df[df['measurement'] == measurement]
        print(f"{measurement}: {sensor_data['value'].min():.1f} - {sensor_data['value'].max():.1f} (avg: {sensor_data['value'].mean():.1f})")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python quick_env_viewer.py <yaml_file>")
        sys.exit(1)
    
    yaml_file = sys.argv[1]
    if not os.path.exists(yaml_file):
        print(f"File not found: {yaml_file}")
        sys.exit(1)
    
    quick_view(yaml_file)
