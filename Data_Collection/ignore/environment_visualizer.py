#!/usr/bin/env python3
"""
Environment Data Visualization Script for Cotton Candy Process YAML files.
This script extracts and plots all environment sensor data over time.
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

def plot_environment_data(environment_data, output_file=None):
    """Create plots for all environment data."""
    
    if not environment_data:
        print("No environment data found!")
        return
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(environment_data)
    
    if df.empty:
        print("No valid environment data found!")
        return
    
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    # Create subplots for different sensor types
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Cotton Candy Process - Environment Data Over Time', fontsize=16)
    
    # Plot 1: Environment Temperature and Humidity
    ax1 = axes[0, 0]
    env_temp = df[(df['sensor_type'] == 'environment') & (df['sensor_id'] == 'temperature')]
    env_humidity = df[(df['sensor_type'] == 'environment') & (df['sensor_id'] == 'humidity')]
    
    if not env_temp.empty:
        ax1.plot(env_temp['timestamp'], env_temp['value'], 'r-', label='Temperature (°C)', linewidth=2)
    if not env_humidity.empty:
        ax1_twin = ax1.twinx()
        ax1_twin.plot(env_humidity['timestamp'], env_humidity['value'], 'b-', label='Humidity (%)', linewidth=2)
        ax1_twin.set_ylabel('Humidity (%)', color='b')
        ax1_twin.tick_params(axis='y', labelcolor='b')
    
    ax1.set_title('Environment Sensors')
    ax1.set_ylabel('Temperature (°C)', color='r')
    ax1.tick_params(axis='y', labelcolor='r')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Internal Temperature and Humidity
    ax2 = axes[0, 1]
    int_temp = df[(df['sensor_type'] == 'internal') & (df['sensor_id'] == 'temperature')]
    int_humidity = df[(df['sensor_type'] == 'internal') & (df['sensor_id'] == 'humidity')]
    
    if not int_temp.empty:
        ax2.plot(int_temp['timestamp'], int_temp['value'], 'r-', label='Temperature (°C)', linewidth=2)
    if not int_humidity.empty:
        ax2_twin = ax2.twinx()
        ax2_twin.plot(int_humidity['timestamp'], int_humidity['value'], 'b-', label='Humidity (%)', linewidth=2)
        ax2_twin.set_ylabel('Humidity (%)', color='b')
        ax2_twin.tick_params(axis='y', labelcolor='b')
    
    ax2.set_title('Internal Sensors')
    ax2.set_ylabel('Temperature (°C)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Infrared Sensors
    ax3 = axes[1, 0]
    ir_head = df[(df['sensor_type'] == 'infrared') & (df['sensor_id'] == 'head')]
    ir_ambient = df[(df['sensor_type'] == 'infrared') & (df['sensor_id'] == 'ambient')]
    
    if not ir_head.empty:
        ax3.plot(ir_head['timestamp'], ir_head['value'], 'orange', label='Head (°C)', linewidth=2)
    if not ir_ambient.empty:
        ax3.plot(ir_ambient['timestamp'], ir_ambient['value'], 'purple', label='Ambient (°C)', linewidth=2)
    
    ax3.set_title('Infrared Sensors')
    ax3.set_ylabel('Temperature (°C)')
    ax3.tick_params(axis='x', rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: All Temperature Sensors Combined
    ax4 = axes[1, 1]
    
    if not env_temp.empty:
        ax4.plot(env_temp['timestamp'], env_temp['value'], 'r-', label='Environment', linewidth=2)
    if not int_temp.empty:
        ax4.plot(int_temp['timestamp'], int_temp['value'], 'g-', label='Internal', linewidth=2)
    if not ir_head.empty:
        ax4.plot(ir_head['timestamp'], ir_head['value'], 'orange', label='IR Head', linewidth=2)
    if not ir_ambient.empty:
        ax4.plot(ir_ambient['timestamp'], ir_ambient['value'], 'purple', label='IR Ambient', linewidth=2)
    
    ax4.set_title('All Temperature Sensors')
    ax4.set_ylabel('Temperature (°C)')
    ax4.tick_params(axis='x', rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show the plot
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    else:
        plt.show()
    
    # Print data summary
    print(f"\nData Summary:")
    print(f"Total data points: {len(df)}")
    print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Duration: {df['timestamp'].max() - df['timestamp'].min()}")
    print(f"\nSensor measurements found:")
    for measurement in sorted(df['measurement'].unique()):
        count = len(df[df['measurement'] == measurement])
        print(f"  {measurement}: {count} data points")

def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python environment_visualizer.py <yaml_file_path> [output_image_file]")
        print("Example: python environment_visualizer.py 4-6-process.yaml environment_plot.png")
        return
    
    yaml_file_path = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(yaml_file_path):
        print(f"Error: File {yaml_file_path} not found!")
        return
    
    print(f"Extracting environment data from: {yaml_file_path}")
    environment_data = extract_environment_data(yaml_file_path)
    
    if environment_data:
        print(f"Found {len(environment_data)} environment data points")
        plot_environment_data(environment_data, output_file)
    else:
        print("No environment data found in the file!")

if __name__ == "__main__":
    main()
