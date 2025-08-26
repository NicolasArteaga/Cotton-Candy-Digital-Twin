#!/usr/bin/env python3
"""
Cotton Candy Digital Twin - Data Pipeline
This script processes CPEE log data from batch folders to create training data for Decision Tree models.

Usage:
    python cotton_candy_pipeline.py --process-all
    python cotton_candy_pipeline.py --batch 0
    python cotton_candy_pipeline.py --batch 0 --process 1
"""

import argparse
import re
import os
import csv
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import yaml

class CottonCandyPipeline:
    def __init__(self, data_dir: str = "Data_Collection/Batches", output_file: str = "Data_Collection/cotton_candy_dataset.csv"):
        self.data_dir = Path(data_dir)
        self.output_file = output_file
        self.processes_data = []
        
        # Define CSV columns - exactly 29 features as per feature_documentation.md
        self.columns = [
            'iteration',
            'batch_number', 
            'stick_number',
            'index_log',
            'stick_weight',
            'sugar_amount',
            # Core Process Parameters (4)
            'iteration_since_maintenance',
            'wait_time',
            'cook_time', 
            'cooldown_time',
            # Timing Metrics (7 total: 5 durations + 2 debugging times)
            'duration_till_handover',
            'duration_total',
            'show_start_time',  # For debugging - MM:SS format
            'show_end_time',    # For debugging - MM:SS format
            'duration_cc_flow',
            'diff_flow',
            'diff_flow_stop',
            # Environmental Baseline (2)
            'baseline_env_EnvH',
            'baseline_env_EnvT',
            # Internal Environmental Sensors - Before Turn On (4)
            'before_turn_on_env_InH',
            'before_turn_on_env_InT',
            'before_turn_on_env_IrO',
            'before_turn_on_env_IrA',
            # Internal Environmental Sensors - After Flow Start (4)
            'after_flow_start_env_InH',
            'after_flow_start_env_InT',
            'after_flow_start_env_IrO',
            'after_flow_start_env_IrA',
            # Internal Environmental Sensors - After Flow End (4)
            'after_flow_end_env_InH',
            'after_flow_end_env_InT',
            'after_flow_end_env_IrO',
            'after_flow_end_env_IrA',
            # Internal Environmental Sensors - Before Cooldown (4)
            'before_cooldown_env_InH',
            'before_cooldown_env_InT',
            'before_cooldown_env_IrO',
            'before_cooldown_env_IrA',
            # Internal Environmental Sensors - After Cooldown (4)
            'after_cooldown_env_InH',
            'after_cooldown_env_InT',
            'after_cooldown_env_IrO',
            'after_cooldown_env_IrA',
            # Quality Data (7)
            'touch_pos1',
            'touch_pos2', 
            'touch_pos3',
            'max_pos1',
            'max_pos2',
            'max_pos3',
            'cc_weight',
        ]
        
    def parse_index_file(self, index_path: Path) -> List[Dict]:
        """
        Parse index.txt file to extract 'Cottonbot - Run with Data Collection' processes
        
        Returns:
            List of dictionaries with process information
        """
        processes = []
        
        try:
            with open(index_path, 'r') as f:
                content = f.read()
            
            # Pattern to match "Cottonbot - Run with Data Collection" processes
            pattern = r'Cottonbot - Run with Data Collection \(([a-f0-9-]+)\) - (\d+)'
            matches = re.findall(pattern, content)
            
            for i, (uuid, index_log) in enumerate(matches):
                processes.append({
                    'uuid': uuid,
                    'index_log': int(index_log),
                    'stick_number': i,  # 0-indexed position within the batch
                })
                
        except Exception as e:
            print(f"Error parsing {index_path}: {e}")
            
                        
        return processes
    
    def extract_process_parameters(self, yaml_file_path: Path) -> Dict:
        """
        Extract all 29 process parameters from YAML log file
        
        Args:
            yaml_file_path: Path to the YAML log file
        
        Returns:
            Dictionary with all extracted parameters
        """
        # Initialize all parameters with None
        parameters = {
            'stick_weight': None,
            'sugar_amount': None,
            # Core Process Parameters (4)
            'iteration_since_maintenance': None,
            'wait_time': None,
            'cook_time': None,
            'cooldown_time': None,
            # Timing Metrics (7 total: 5 durations + 2 debugging times) - will be calculated from events
            'duration_till_handover': None,
            'duration_total': None,
            'show_start_time': None,  # For debugging - MM:SS format
            'show_end_time': None,    # For debugging - MM:SS format
            'duration_cc_flow': None,
            'diff_flow': None,
            'diff_flow_stop': None,
            # Environmental Baseline (2)
            'baseline_env_EnvH': None,
            'baseline_env_EnvT': None,
            # Internal Environmental Sensors - Before Turn On (4)
            'before_turn_on_env_InH': None,
            'before_turn_on_env_InT': None,
            'before_turn_on_env_IrO': None,
            'before_turn_on_env_IrA': None,
            # Internal Environmental Sensors - After Flow Start (4)
            'after_flow_start_env_InH': None,
            'after_flow_start_env_InT': None,
            'after_flow_start_env_IrO': None,
            'after_flow_start_env_IrA': None,
            # Internal Environmental Sensors - After Flow End (4)
            'after_flow_end_env_InH': None,
            'after_flow_end_env_InT': None,
            'after_flow_end_env_IrO': None,
            'after_flow_end_env_IrA': None,
            # Internal Environmental Sensors - Before Cooldown (4)
            'before_cooldown_env_InH': None,
            'before_cooldown_env_InT': None,
            'before_cooldown_env_IrO': None,
            'before_cooldown_env_IrA': None,
            # Internal Environmental Sensors - After Cooldown (4)
            'after_cooldown_env_InH': None,
            'after_cooldown_env_InT': None,
            'after_cooldown_env_IrO': None,
            'after_cooldown_env_IrA': None,
            # Quality Data (7)
            'touch_pos1': None,
            'touch_pos2': None,
            'touch_pos3': None,
            'max_pos1': None,
            'max_pos2': None,
            'max_pos3': None,
            'cc_weight': None,
        }
        
        # For storing events to calculate timing metrics
        events = []
        
        # For tracking process phases for environmental sensor assignment
        show_start_done = False
        show_end_done = False
        weigh_cc_done = False
        cooldown_time_set = False
        machine_cooled_off = False
        
        # Store the last recorded environmental sensors as fallback for after_cooldown
        last_environmental_reading = {}
        
        try:
            with open(yaml_file_path, 'r') as f:
                for doc in yaml.safe_load_all(f):
                    if 'event' in doc:
                        event = doc['event']
                        events.append(event)
                        
                        # Extract initial process parameters from dataelements/change
                        if (event.get('cpee:lifecycle:transition') == 'dataelements/change' and 
                            'data' in event and event['data'] is not None):
                            
                            for data_item in event['data']:
                                if isinstance(data_item, dict):
                                    name = data_item.get('name')
                                    value = data_item.get('value')
                                    
                                    if name == 'stick_weight' and value is not None:
                                        try:
                                            parameters['stick_weight'] = float(value)
                                        except (ValueError, TypeError):
                                            pass
                                    elif name == 'sugar_amount' and value is not None:
                                        try:
                                            parameters['sugar_amount'] = int(value)
                                        except (ValueError, TypeError):
                                            pass
                                    elif name == 'iteration_since_maintenance' and value is not None:
                                        try:
                                            parameters['iteration_since_maintenance'] = int(value)
                                        except (ValueError, TypeError):
                                            pass
                                    elif name == 'wait_time' and value is not None:
                                        try:
                                            parameters['wait_time'] = float(value)
                                        except (ValueError, TypeError):
                                            pass
                                    elif name == 'cook_time' and value is not None:
                                        try:
                                            parameters['cook_time'] = float(value)
                                        except (ValueError, TypeError):
                                            pass
                                    elif name == 'cooldown_time' and value is not None:
                                        try:
                                            parameters['cooldown_time'] = float(value)
                                        except (ValueError, TypeError):
                                            pass
                        
                        # Track Show Start activity/done for phase detection
                        elif (event.get('concept:name') == 'Show Start' and
                              event.get('cpee:lifecycle:transition') == 'activity/done'):
                            show_start_done = True
                        
                        # Track Show End activity/done for phase detection
                        elif (event.get('concept:name') == 'Show End' and
                              event.get('cpee:lifecycle:transition') == 'activity/done'):
                            show_end_done = True
                        
                        # Track Weigh the whole Cotton Candy activity/done for phase detection
                        elif (event.get('concept:name') == 'Weigh the whole Cotton Candy' and
                              event.get('cpee:lifecycle:transition') == 'activity/done'):
                            weigh_cc_done = True
                        
                        # Track Set cooldown time activity/done for phase detection
                        elif (event.get('concept:name') == 'Set cooldown time' and
                              event.get('cpee:lifecycle:transition') == 'activity/done'):
                            cooldown_time_set = True
                        
                        # Track Cools Down and Turns Off Machine activity/done for phase detection
                        elif (event.get('concept:name') == 'Cools Down and Turns Off Machine' and
                              event.get('cpee:lifecycle:transition') == 'activity/done'):
                            machine_cooled_off = True
                        
                        # Extract environmental sensor data from Get the Environment Data stream/data events
                        elif (event.get('concept:name') == 'Get the Environment Data' and
                              event.get('cpee:lifecycle:transition') == 'stream/data' and
                              'stream:datastream' in event):
                            
                            sensor_data = self._extract_environmental_sensors(event)
                            if sensor_data:
                                # Store this as the last recorded environmental reading (for fallback)
                                for sensor in ['InH', 'InT', 'IrO', 'IrA']:
                                    if sensor in sensor_data:
                                        last_environmental_reading[sensor] = sensor_data[sensor]
                                
                                # Handle baseline environmental sensors (external)
                                if 'EnvH' in sensor_data and parameters['baseline_env_EnvH'] is None:
                                    parameters['baseline_env_EnvH'] = sensor_data['EnvH']
                                if 'EnvT' in sensor_data and parameters['baseline_env_EnvT'] is None:
                                    parameters['baseline_env_EnvT'] = sensor_data['EnvT']
                                
                                # Handle internal sensors with phase detection
                                for sensor in ['InH', 'InT', 'IrO', 'IrA']:
                                    if sensor in sensor_data:
                                        # Determine phase based on process state
                                        if machine_cooled_off:
                                            # After cooldown phase
                                            full_key = f"after_cooldown_env_{sensor}"
                                            if full_key in parameters and parameters[full_key] is None:
                                                parameters[full_key] = sensor_data[sensor]
                                        elif weigh_cc_done:
                                            # Before cooldown phase (first env data after weighing)
                                            full_key = f"before_cooldown_env_{sensor}"
                                            if full_key in parameters and parameters[full_key] is None:
                                                parameters[full_key] = sensor_data[sensor]
                                        elif show_end_done:
                                            # After flow end phase
                                            full_key = f"after_flow_end_env_{sensor}"
                                            if full_key in parameters and parameters[full_key] is None:
                                                parameters[full_key] = sensor_data[sensor]
                                        elif show_start_done:
                                            # After flow start phase
                                            full_key = f"after_flow_start_env_{sensor}"
                                            if full_key in parameters and parameters[full_key] is None:
                                                parameters[full_key] = sensor_data[sensor]
                                        else:
                                            # Before turn on phase (default)
                                            full_key = f"before_turn_on_env_{sensor}"
                                            if full_key in parameters and parameters[full_key] is None:
                                                parameters[full_key] = sensor_data[sensor]
                        
                        # Extract quality data - cotton candy size measurements from stream/data events
                        elif (event.get('concept:name') == 'Measure the size of the Cotton Candy' and
                              event.get('cpee:lifecycle:transition') == 'stream/data' and
                              'stream:datastream' in event):
                            
                            touch_data = self._extract_touch_data(event)
                            if touch_data:
                                if 'pos1' in touch_data and parameters['touch_pos1'] is None:
                                    parameters['touch_pos1'] = touch_data['pos1']
                                if 'pos2' in touch_data and parameters['touch_pos2'] is None:
                                    parameters['touch_pos2'] = touch_data['pos2']
                                if 'pos3' in touch_data and parameters['touch_pos3'] is None:
                                    parameters['touch_pos3'] = touch_data['pos3']
                        
                        # Extract quality data - pressure measurements from stream/data events
                        elif (event.get('concept:name') == 'Measure the pressure of the three sides of the cotton candy' and
                              event.get('cpee:lifecycle:transition') == 'stream/data' and
                              'stream:datastream' in event):
                            
                            pressure_data = self._extract_pressure_data(event)
                            if pressure_data:
                                if 'pos1' in pressure_data and parameters['max_pos1'] is None:
                                    parameters['max_pos1'] = pressure_data['pos1']
                                if 'pos2' in pressure_data and parameters['max_pos2'] is None:
                                    parameters['max_pos2'] = pressure_data['pos2']
                                if 'pos3' in pressure_data and parameters['max_pos3'] is None:
                                    parameters['max_pos3'] = pressure_data['pos3']
                        
                        # Extract cotton candy weight from stream/data events
                        elif (event.get('concept:name') == 'Weigh the whole Cotton Candy' and
                              event.get('cpee:lifecycle:transition') == 'stream/data' and
                              'stream:datastream' in event):
                            
                            weight_data = self._extract_weight_data(event)
                            if weight_data is not None and weight_data >= 0:  # Only store positive weights
                                parameters['cc_weight'] = weight_data
            
            # Calculate timing metrics from events
            timing_metrics = self._calculate_timing_metrics(events, parameters.get('wait_time'), parameters.get('cook_time'))
            parameters.update(timing_metrics)
            
            # Fallback: Use last recorded environmental data for missing after_cooldown values
            for sensor in ['InH', 'InT', 'IrO', 'IrA']:
                full_key = f"after_cooldown_env_{sensor}"
                if full_key in parameters and parameters[full_key] is None:
                    if sensor in last_environmental_reading:
                        parameters[full_key] = last_environmental_reading[sensor]
                                
        except Exception as e:
            print(f"    Warning: Error parsing {yaml_file_path.name}: {e}")
            
        return parameters
    
    def _extract_environmental_sensors(self, event: Dict) -> Dict:
        """Extract environmental sensor values from Get the Environment Data stream/data event"""
        sensor_data = {}
        
        stream_datastream = event.get('stream:datastream', [])
        
        for stream_item in stream_datastream:
            if not isinstance(stream_item, dict):
                continue
                
            # Look for nested datastreams
            nested_datastream = stream_item.get('stream:datastream')
            if not isinstance(nested_datastream, list):
                continue
                
            # Find the stream type
            stream_name = None
            for item in nested_datastream:
                if isinstance(item, dict) and 'stream:name' in item:
                    stream_name = item['stream:name']
                    break
            
            # Handle environment sensors (external)
            if stream_name == 'environment':
                for item in nested_datastream:
                    if isinstance(item, dict) and 'stream:point' in item:
                        point = item['stream:point']
                        if isinstance(point, dict):
                            stream_id = point.get('stream:id')
                            stream_value = point.get('stream:value')
                            
                            if stream_id and stream_value is not None:
                                try:
                                    if stream_id == 'humidity':
                                        sensor_data['EnvH'] = float(stream_value)
                                    elif stream_id == 'temperature':
                                        sensor_data['EnvT'] = float(stream_value)
                                except (ValueError, TypeError):
                                    pass
            
            # Handle internal sensors
            elif stream_name == 'internal':
                for item in nested_datastream:
                    if isinstance(item, dict) and 'stream:point' in item:
                        point = item['stream:point']
                        if isinstance(point, dict):
                            stream_id = point.get('stream:id')
                            stream_value = point.get('stream:value')
                            
                            if stream_id and stream_value is not None:
                                try:
                                    if stream_id == 'humidity':
                                        sensor_data['InH'] = float(stream_value)
                                    elif stream_id == 'temperature':
                                        sensor_data['InT'] = float(str(stream_value))  # Handle string values
                                except (ValueError, TypeError):
                                    pass
                    
                    # Look for nested infrared datastream
                    elif isinstance(item, dict) and 'stream:datastream' in item:
                        infrared_datastream = item['stream:datastream']
                        if isinstance(infrared_datastream, list):
                            # Check if this is infrared stream
                            is_infrared = any(
                                ir_item.get('stream:name') == 'infrared' 
                                for ir_item in infrared_datastream 
                                if isinstance(ir_item, dict)
                            )
                            
                            if is_infrared:
                                for ir_item in infrared_datastream:
                                    if isinstance(ir_item, dict) and 'stream:point' in ir_item:
                                        ir_point = ir_item['stream:point']
                                        if isinstance(ir_point, dict):
                                            stream_id = ir_point.get('stream:id')
                                            stream_value = ir_point.get('stream:value')
                                            
                                            if stream_id and stream_value is not None:
                                                try:
                                                    if stream_id == 'head':
                                                        sensor_data['IrO'] = float(stream_value)
                                                    elif stream_id == 'ambient':
                                                        sensor_data['IrA'] = float(stream_value)
                                                except (ValueError, TypeError):
                                                    pass
        
        return sensor_data
    
    def _extract_touch_data(self, event: Dict) -> Dict:
        """Extract touch position values from cotton candy size measurement stream/data event"""
        touch_data = {}
        
        stream_datastream = event.get('stream:datastream', [])
        
        for stream_item in stream_datastream:
            if isinstance(stream_item, dict) and 'stream:point' in stream_item:
                point = stream_item['stream:point']
                if isinstance(point, dict):
                    stream_id = point.get('stream:id')
                    stream_value = point.get('stream:value')
                    
                    if stream_id and stream_value is not None:
                        try:
                            if stream_id in ['pos1', 'pos2', 'pos3']:
                                touch_data[stream_id] = round(float(stream_value), 2)
                        except (ValueError, TypeError):
                            pass
        
        return touch_data
    
    def _extract_pressure_data(self, event: Dict) -> Dict:
        """Extract max pressure values from pressure measurement stream/data event"""
        pressure_data = {}
        
        stream_datastream = event.get('stream:datastream', [])
        
        for stream_item in stream_datastream:
            if not isinstance(stream_item, dict):
                continue
                
            # Look for nested datastreams
            nested_datastream = stream_item.get('stream:datastream')
            if not isinstance(nested_datastream, list):
                continue
                
            # Find the stream type - look for max_pressures
            stream_name = None
            for item in nested_datastream:
                if isinstance(item, dict) and 'stream:name' in item:
                    stream_name = item['stream:name']
                    break
            
            # Handle max_pressures data
            if stream_name == 'max_pressures':
                for item in nested_datastream:
                    if isinstance(item, dict) and 'stream:point' in item:
                        point = item['stream:point']
                        if isinstance(point, dict):
                            stream_id = point.get('stream:id')
                            stream_value = point.get('stream:value')
                            
                            if stream_id and stream_value is not None:
                                try:
                                    if stream_id in ['pos1', 'pos2', 'pos3']:
                                        pressure_data[stream_id] = round(float(stream_value), 2)
                                except (ValueError, TypeError):
                                    pass
        
        return pressure_data
    
    def _extract_weight_data(self, event: Dict) -> Optional[float]:
        """Extract cotton candy weight from weight measurement stream/data event"""
        stream_datastream = event.get('stream:datastream', [])
        
        for stream_item in stream_datastream:
            if isinstance(stream_item, dict) and 'stream:point' in stream_item:
                point = stream_item['stream:point']
                if isinstance(point, dict):
                    stream_id = point.get('stream:id')
                    stream_value = point.get('stream:value')
                    
                    if stream_id == 'weight' and stream_value is not None:
                        try:
                            return round(float(stream_value), 2)
                        except (ValueError, TypeError):
                            pass
        
        return None
    
    def _determine_sensor_phase(self, event: Dict, events: List[Dict]) -> str:
        """Determine which phase this sensor reading belongs to"""
        # This is a simplified version - would need more complex logic
        # to accurately determine phases based on the event timeline
        activity_name = event.get('concept:name', '')
        
        if 'turn_on' in activity_name.lower():
            return 'before_turn_on'
        elif 'flow' in activity_name.lower() and 'start' in activity_name.lower():
            return 'after_flow_start'
        elif 'flow' in activity_name.lower() and 'end' in activity_name.lower():
            return 'after_flow_end'
        elif 'cooldown' in activity_name.lower() and ('before' in activity_name.lower() or 'start' in activity_name.lower()):
            return 'before_cooldown'
        elif 'cooldown' in activity_name.lower() and ('after' in activity_name.lower() or 'end' in activity_name.lower()):
            return 'after_cooldown'
        
        # Default to before_turn_on for now
        return 'before_turn_on'
    
    def _calculate_timing_metrics(self, events: List[Dict], wait_time: Optional[float] = None, cook_time: Optional[float] = None) -> Dict:
        """Calculate timing metrics from event timeline"""
        from datetime import datetime
        
        timing_metrics = {
            'duration_till_handover': None,
            'duration_total': None,
            'duration_cc_flow': None,
            'diff_flow': None,
            'diff_flow_stop': None,
            'show_start_time': None,
            'show_end_time': None,
        }
        
        # Find key timestamps
        first_timestamp = None
        create_cc_done_timestamp = None
        last_timestamp = None
        show_start_timestamp = None
        show_end_timestamp = None
        turn_machine_on_timestamp = None
        
        try:
            for event in events:
                timestamp_str = event.get('time:timestamp')
                if not timestamp_str:
                    continue
                    
                # Parse timestamp
                try:
                    # Handle timezone format
                    if timestamp_str.endswith('+02:00'):
                        timestamp_str = timestamp_str[:-6]
                    timestamp = datetime.fromisoformat(timestamp_str)
                except:
                    continue
                
                # Find first timestamp (from state/change ready event)
                if (first_timestamp is None and 
                    event.get('cpee:lifecycle:transition') == 'state/change' and
                    event.get('cpee:state') == 'ready'):
                    first_timestamp = timestamp
                
                # Find Create Cotton Candy done
                if (event.get('concept:name') == 'Create Cotton Candy' and
                    event.get('cpee:lifecycle:transition') == 'activity/done' and
                    event.get('lifecycle:transition') == 'complete'):
                    create_cc_done_timestamp = timestamp
                
                # Find last timestamp (state/change finished)
                if (event.get('cpee:lifecycle:transition') == 'state/change' and
                    event.get('cpee:state') == 'finished'):
                    last_timestamp = timestamp
                
                # Find Show Start done
                if (event.get('concept:name') == 'Show Start' and
                    event.get('cpee:lifecycle:transition') == 'activity/done' and
                    event.get('lifecycle:transition') == 'complete'):
                    show_start_timestamp = timestamp
                    timing_metrics['show_start_time'] = f"{timestamp.minute:02d}:{timestamp.second:02d}"
                
                # Find Turn the Machine On done
                if (event.get('concept:name') == 'Turn the Machine On' and
                    event.get('cpee:lifecycle:transition') == 'activity/done' and
                    event.get('lifecycle:transition') == 'complete'):
                    turn_machine_on_timestamp = timestamp
                
                # Find Show End done
                if (event.get('concept:name') == 'Show End' and
                    event.get('cpee:lifecycle:transition') == 'activity/done' and
                    event.get('lifecycle:transition') == 'complete'):
                    show_end_timestamp = timestamp
                    timing_metrics['show_end_time'] = f"{timestamp.minute:02d}:{timestamp.second:02d}"
            
            # Calculate durations
            if first_timestamp and create_cc_done_timestamp:
                duration_handover = (create_cc_done_timestamp - first_timestamp).total_seconds()
                timing_metrics['duration_till_handover'] = round(duration_handover, 2)
            
            if first_timestamp and last_timestamp:
                duration_total = (last_timestamp - first_timestamp).total_seconds()
                timing_metrics['duration_total'] = round(duration_total, 2)
            
            if show_start_timestamp and show_end_timestamp:
                duration_cc_flow = (show_end_timestamp - show_start_timestamp).total_seconds()
                timing_metrics['duration_cc_flow'] = round(duration_cc_flow, 2)
            
            # Calculate diff_flow: Show Start timestamp - (Turn Machine On timestamp + wait_time)
            if turn_machine_on_timestamp and show_start_timestamp and wait_time is not None:
                # Add wait_time (in seconds) to the Turn Machine On timestamp
                from datetime import timedelta
                machine_on_plus_wait = turn_machine_on_timestamp + timedelta(seconds=wait_time)
                
                # Calculate the difference: actual show start time - target time
                diff_flow_seconds = (show_start_timestamp - machine_on_plus_wait).total_seconds()
                timing_metrics['diff_flow'] = round(diff_flow_seconds, 2)
            
            # Calculate diff_flow_stop: Show End timestamp - (Turn Machine On timestamp + wait_time + cook_time)
            if turn_machine_on_timestamp and show_end_timestamp and wait_time is not None and cook_time is not None:
                # Add wait_time + cook_time (in seconds) to the Turn Machine On timestamp
                from datetime import timedelta
                machine_on_plus_wait_cook = turn_machine_on_timestamp + timedelta(seconds=wait_time + cook_time)
                
                # Calculate the difference: actual show end time - target time
                diff_flow_stop_seconds = (show_end_timestamp - machine_on_plus_wait_cook).total_seconds()
                timing_metrics['diff_flow_stop'] = round(diff_flow_stop_seconds, 2)
                                
        except Exception as e:
            print(f"    Warning: Error calculating timing metrics: {e}")
        
        return timing_metrics
    
    def get_batch_folders(self) -> List[Path]:
        """Get all batch-x folders (excluding test batches)"""
        batch_folders = []
        
        if not self.data_dir.exists():
            print(f"Data directory {self.data_dir} does not exist!")
            return batch_folders
            
        for folder in self.data_dir.iterdir():
            if folder.is_dir() and folder.name.startswith('batch-') and not folder.name.startswith('test-'):
                batch_folders.append(folder)
                
        # Sort by batch number
        batch_folders.sort(key=lambda x: int(x.name.split('-')[1]))
        return batch_folders
    
    def process_batch(self, batch_folder: Path, target_process: Optional[int] = None, 
                     global_iteration_start: int = 0) -> List[Dict]:
        """
        Process a single batch folder
        
        Args:
            batch_folder: Path to batch folder
            target_process: If specified, only process this specific process index
            global_iteration_start: Starting global iteration number for this batch
        
        Returns:
            List of processed data rows
        """
        batch_name = batch_folder.name
        batch_number = int(batch_name.split('-')[1])
        
        print(f"Processing {batch_name}...")
        
        index_file = batch_folder / 'index.txt'
        if not index_file.exists():
            print(f"  Warning: index.txt not found in {batch_name}")
            return []
            
        # Parse the index file to get processes
        processes = self.parse_index_file(index_file)
        
        if not processes:
            print(f"  Warning: No 'Cottonbot - Run with Data Collection' processes found in {batch_name}")
            return []
            
        # Filter to specific process if requested
        if target_process is not None:
            if 0 <= target_process < len(processes):
                processes = [processes[target_process]]
                print(f"  Processing only process {target_process}")
            else:
                print(f"  Error: Process {target_process} not found. Available: 0-{len(processes)-1}")
                return []
        
        batch_data = []
        
        for i, process in enumerate(processes):
            # Calculate global iteration number
            global_iteration = global_iteration_start + i
            
            # Extract process parameters from the YAML file
            yaml_file = batch_folder / f"{process['uuid']}.xes.yaml"
            process_params = {}
            
            if yaml_file.exists():
                process_params = self.extract_process_parameters(yaml_file)
            else:
                print(f"    Warning: YAML file not found for {process['uuid'][:8]}...")
            
            # Create row data with all 29 features
            row_data = {
                'iteration': global_iteration,
                'batch_number': batch_number,
                'stick_number': process['stick_number'],
                'index_log': process['index_log'],
                'stick_weight': process_params.get('stick_weight'),
                'sugar_amount': process_params.get('sugar_amount'),
                # Core Process Parameters (4)
                'iteration_since_maintenance': process_params.get('iteration_since_maintenance'),
                'wait_time': process_params.get('wait_time'),
                'cook_time': process_params.get('cook_time'),
                'cooldown_time': process_params.get('cooldown_time'),
                # Timing Metrics (7 total: 5 durations + 2 debugging times)
                'duration_till_handover': process_params.get('duration_till_handover'),
                'duration_total': process_params.get('duration_total'),
                'show_start_time': process_params.get('show_start_time'),
                'show_end_time': process_params.get('show_end_time'),
                'duration_cc_flow': process_params.get('duration_cc_flow'),
                'diff_flow': process_params.get('diff_flow'),
                'diff_flow_stop': process_params.get('diff_flow_stop'),
                # Environmental Baseline (2)
                'baseline_env_EnvH': process_params.get('baseline_env_EnvH'),
                'baseline_env_EnvT': process_params.get('baseline_env_EnvT'),
                # Internal Environmental Sensors - Before Turn On (4)
                'before_turn_on_env_InH': process_params.get('before_turn_on_env_InH'),
                'before_turn_on_env_InT': process_params.get('before_turn_on_env_InT'),
                'before_turn_on_env_IrO': process_params.get('before_turn_on_env_IrO'),
                'before_turn_on_env_IrA': process_params.get('before_turn_on_env_IrA'),
                # Internal Environmental Sensors - After Flow Start (4)
                'after_flow_start_env_InH': process_params.get('after_flow_start_env_InH'),
                'after_flow_start_env_InT': process_params.get('after_flow_start_env_InT'),
                'after_flow_start_env_IrO': process_params.get('after_flow_start_env_IrO'),
                'after_flow_start_env_IrA': process_params.get('after_flow_start_env_IrA'),
                # Internal Environmental Sensors - After Flow End (4)
                'after_flow_end_env_InH': process_params.get('after_flow_end_env_InH'),
                'after_flow_end_env_InT': process_params.get('after_flow_end_env_InT'),
                'after_flow_end_env_IrO': process_params.get('after_flow_end_env_IrO'),
                'after_flow_end_env_IrA': process_params.get('after_flow_end_env_IrA'),
                # Internal Environmental Sensors - Before Cooldown (4)
                'before_cooldown_env_InH': process_params.get('before_cooldown_env_InH'),
                'before_cooldown_env_InT': process_params.get('before_cooldown_env_InT'),
                'before_cooldown_env_IrO': process_params.get('before_cooldown_env_IrO'),
                'before_cooldown_env_IrA': process_params.get('before_cooldown_env_IrA'),
                # Internal Environmental Sensors - After Cooldown (4)
                'after_cooldown_env_InH': process_params.get('after_cooldown_env_InH'),
                'after_cooldown_env_InT': process_params.get('after_cooldown_env_InT'),
                'after_cooldown_env_IrO': process_params.get('after_cooldown_env_IrO'),
                'after_cooldown_env_IrA': process_params.get('after_cooldown_env_IrA'),
                # Quality Data (7)
                'touch_pos1': process_params.get('touch_pos1'),
                'touch_pos2': process_params.get('touch_pos2'),
                'touch_pos3': process_params.get('touch_pos3'),
                'max_pos1': process_params.get('max_pos1'),
                'max_pos2': process_params.get('max_pos2'),
                'max_pos3': process_params.get('max_pos3'),
                'cc_weight': process_params.get('cc_weight'),
            }
            
            batch_data.append(row_data)
            print(f"  -> Process {i}: Global Iteration {global_iteration}, UUID={process['uuid'][:8]}..., Index={process['index_log']}")
        
        return batch_data
    
    def load_existing_data(self) -> pd.DataFrame:
        """Load existing CSV data if it exists"""
        if os.path.exists(self.output_file):
            try:
                return pd.read_csv(self.output_file)
            except Exception as e:
                print(f"Error loading existing data: {e}")
                return pd.DataFrame(columns=self.columns)
        else:
            return pd.DataFrame(columns=self.columns)
    
    def update_csv_data(self, new_data: List[Dict], batch_number: Optional[int] = None, 
                       process_index: Optional[int] = None):
        """
        Update CSV file with new data
        
        Args:
            new_data: List of new data rows
            batch_number: If specified, only update rows for this batch
            process_index: If specified, only update this specific process within the batch
        """
        if not new_data:
            print("No data to update")
            return
            
        # Load existing data
        existing_df = self.load_existing_data()
        
        if batch_number is not None:
            if process_index is not None:
                # Remove specific row - identify by batch_number and stick_number (which matches process_index)
                mask = (existing_df['batch_number'] == batch_number) & \
                       (existing_df['stick_number'] == process_index)
                existing_df = existing_df[~mask]
                print(f"Updating batch {batch_number}, process {process_index}")
            else:
                # Remove entire batch
                existing_df = existing_df[existing_df['batch_number'] != batch_number]
                print(f"Updating entire batch {batch_number}")
        else:
            # Full update - clear all data
            existing_df = pd.DataFrame(columns=self.columns)
            print("Full update - replacing all data")
        
        # Add new data
        new_df = pd.DataFrame(new_data)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        
        # Sort by batch_number, then by iteration
        combined_df = combined_df.sort_values(['batch_number', 'iteration']).reset_index(drop=True)
        
        # Save to CSV with 2 decimal places formatting
        combined_df.to_csv(self.output_file, index=False, float_format='%.2f')
        print(f"Data saved to {self.output_file}")
        print(f"Total rows: {len(combined_df)}")
    
    def run(self, process_all: bool = False, batch_number: Optional[int] = None, 
            process_index: Optional[int] = None):
        """
        Main pipeline execution
        
        Args:
            process_all: Process all batches
            batch_number: Process specific batch
            process_index: Process specific process within batch
        """
        batch_folders = self.get_batch_folders()
        
        if not batch_folders:
            print("No batch folders found!")
            return
            
        print(f"Found {len(batch_folders)} batch folders:")
        for folder in batch_folders:
            print(f"  - {folder.name}")
        print()
        
        all_data = []
        
        if process_all:
            # Process all batches with global iteration counter
            global_iteration = 0
            for batch_folder in batch_folders:
                batch_data = self.process_batch(batch_folder, None, global_iteration)
                all_data.extend(batch_data)
                global_iteration += len(batch_data)
            
            self.update_csv_data(all_data)
            
        elif batch_number is not None:
            # Process specific batch - need to calculate global iteration start
            target_folder = None
            global_iteration_start = 0
            
            for folder in batch_folders:
                current_batch_num = int(folder.name.split('-')[1])
                if current_batch_num == batch_number:
                    target_folder = folder
                    break
                elif current_batch_num < batch_number:
                    # Count processes in previous batches
                    index_file = folder / 'index.txt'
                    if index_file.exists():
                        processes = self.parse_index_file(index_file)
                        global_iteration_start += len(processes)
                    
            if target_folder is None:
                print(f"Batch {batch_number} not found!")
                return
                
            # If processing specific process, add its local index to the start
            if process_index is not None:
                global_iteration_start += process_index
                
            batch_data = self.process_batch(target_folder, process_index, global_iteration_start)
            self.update_csv_data(batch_data, batch_number, process_index)
            
        else:
            print("Please specify --process-all, --batch, or --batch with --process")
            return
        
        print("\nPipeline completed successfully!")

def main():
    parser = argparse.ArgumentParser(
        description='Cotton Candy Digital Twin Data Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cotton_candy_pipeline.py --process-all
  python cotton_candy_pipeline.py --batch 0
  python cotton_candy_pipeline.py --batch 0 --process 1
        """
    )
    
    parser.add_argument('--process-all', action='store_true',
                       help='Process all batch folders')
    parser.add_argument('--batch', type=int,
                       help='Process specific batch number (e.g., 0 for batch-0)')
    parser.add_argument('--process', type=int,
                       help='Process specific process index within batch (use with --batch)')
    parser.add_argument('--data-dir', default='Data_Collection/Batches',
                       help='Path to batches directory (default: Data_Collection/Batches)')
    parser.add_argument('--output', default='Data_Collection/cotton_candy_dataset.csv',
                       help='Output CSV file (default: Data_Collection/cotton_candy_dataset.csv)')
    
    args = parser.parse_args()
    
    # Validation
    if not args.process_all and args.batch is None:
        parser.error('Must specify either --process-all or --batch')
    
    if args.process is not None and args.batch is None:
        parser.error('--process can only be used with --batch')
    
    # Initialize and run pipeline
    pipeline = CottonCandyPipeline(args.data_dir, args.output)
    pipeline.run(
        process_all=args.process_all,
        batch_number=args.batch,
        process_index=args.process
    )

if __name__ == "__main__":
    main()
