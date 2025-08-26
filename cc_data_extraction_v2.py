#!/usr/bin/env python3
"""
Cotton Candy Digital Twin - Data Pipeline V2
This script processes CPEE log data from batch folders to create training data for Decision Tree models.
V2 extracts most data from the final "For Simple Data Extraction" dataelements/change event.

Usage:
    python cc_data_extraction_v2.py --process-all
    python cc_data_extraction_v2.py --batch 0
    python cc_data_extraction_v2.py --batch 0 --process 1
"""

import argparse
import re
import os
import csv
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import yaml
from datetime import datetime, time

class CottonCandyPipelineV2:
    def __init__(self, data_dir: str = "Data_Collection/Batches", 
                 output_file: str = "Data_Collection/cotton_candy_dataset_v2.csv",
                 f_pressures_file: str = "Data_Collection/f_pressures_dataset.csv"):
        self.data_dir = Path(data_dir)
        self.output_file = output_file
        self.f_pressures_file = f_pressures_file
        self.processes_data = []
        
        # Define main CSV columns - parameters available from final event + calculated ones
        self.columns = [
            'iteration',
            'batch_number', 
            'stick_number',
            'index_log',
            'stick_weight',  # Need to extract from initial parameters
            'sugar_amount',  # Need to extract from initial parameters
            # Core Process Parameters (4)
            'iteration_since_maintenance',
            'wait_time',
            'cook_time', 
            'cooldown_time',
            # Temperature Parameters (3)
            'start_temp',
            'cook_temp',
            'cooled_temp',
            # Timing Metrics (7 total: 5 durations + 2 debugging times)
            'duration_till_handover',
            'duration_total',
            'show_start_time',  # For debugging - MM:SS format
            'show_end_time',    # For debugging - MM:SS format
            'duration_cc_flow',
            'diff_flow',
            'diff_flow_stop',
            # Environmental Baseline (2) - Need to extract from environment events
            'baseline_env_EnvH',
            'baseline_env_EnvT',
            # Internal Environmental Sensors - Before Turn On (4) - Need to extract
            'before_turn_on_env_InH',
            'before_turn_on_env_InT',
            'before_turn_on_env_IrO',
            'before_turn_on_env_IrA',
            # Internal Environmental Sensors - After Flow Start (4) - Need to extract
            'after_flow_start_env_InH',
            'after_flow_start_env_InT',
            'after_flow_start_env_IrO',
            'after_flow_start_env_IrA',
            # Internal Environmental Sensors - After Flow End (4) - Need to extract
            'after_flow_end_env_InH',
            'after_flow_end_env_InT',
            'after_flow_end_env_IrO',
            'after_flow_end_env_IrA',
            # Internal Environmental Sensors - Before Cooldown (4) - Need to extract
            'before_cooldown_env_InH',
            'before_cooldown_env_InT',
            'before_cooldown_env_IrO',
            'before_cooldown_env_IrA',
            # Internal Environmental Sensors - After Cooldown (4) - Need to extract
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
        
        # Define f_pressures CSV columns
        self.f_pressures_columns = [
            'iteration',
            'batch_number',
            'stick_number',
            'pos1_pressures',  # JSON string of pressure array
            'pos2_pressures',  # JSON string of pressure array
            'pos3_pressures',  # JSON string of pressure array
        ]
        
    def parse_index_file(self, index_path: Path) -> List[Dict]:
        """
        Parse index.txt file to extract all 'Cottonbot - Run with Data Collection*' processes
        
        Returns:
            List of dictionaries with process information
        """
        processes = []
        
        try:
            with open(index_path, 'r') as f:
                content = f.read()
            
            # Pattern to match all "Cottonbot - Run with Data Collection" processes (any variant)
            pattern = r'Cottonbot - Run with Data Collection[^(]*\(([a-f0-9-]+)\) - (\d+)'
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
    
    def extract_process_parameters_v2(self, yaml_file_path: Path) -> Tuple[Dict, Dict]:
        """
        Extract process parameters from YAML log file - V2 approach
        
        Args:
            yaml_file_path: Path to the YAML log file
        
        Returns:
            Tuple of (main_parameters_dict, f_pressures_dict)
        """
        # Initialize main parameters
        parameters = {col: None for col in self.columns if col not in ['iteration', 'batch_number', 'stick_number']}
        
        # Initialize f_pressures data
        f_pressures_data = {
            'pos1_pressures': None,
            'pos2_pressures': None,
            'pos3_pressures': None,
        }
        
        # Variables to track if we found the final extraction event
        found_final_event = False
        
        try:
            with open(yaml_file_path, 'r') as f:
                events = []
                
                for doc in yaml.safe_load_all(f):
                    if 'event' in doc:
                        event = doc['event']
                        events.append(event)
                        
                        # Extract initial process parameters (stick_weight, sugar_amount) from early dataelements/change
                        if (not found_final_event and
                            event.get('cpee:lifecycle:transition') == 'dataelements/change' and 
                            'data' in event and event['data'] is not None):
                            
                            for data_item in event['data']:
                                if isinstance(data_item, dict):
                                    name = data_item.get('name')
                                    value = data_item.get('value')
                                    
                                    if name == 'stick_weight' and value is not None and parameters['stick_weight'] is None:
                                        try:
                                            parameters['stick_weight'] = float(value)
                                        except (ValueError, TypeError):
                                            pass
                                    elif name == 'sugar_amount' and value is not None and parameters['sugar_amount'] is None:
                                        try:
                                            parameters['sugar_amount'] = int(value)
                                        except (ValueError, TypeError):
                                            pass
                        
                        # Extract from final "For Simple Data Extraction" event
                        if (event.get('concept:name') == 'For Simple Data Extraction' and
                              event.get('cpee:lifecycle:transition') == 'dataelements/change' and
                              'data' in event and event['data'] is not None):
                            
                            found_final_event = True
                            
                            for data_item in event['data']:
                                if isinstance(data_item, dict):
                                    name = data_item.get('name')
                                    value = data_item.get('value')
                                    
                                    # Direct mappings
                                    if name == 'iteration_since_maintenance' and value is not None:
                                        try:
                                            parameters['iteration_since_maintenance'] = int(value)
                                        except (ValueError, TypeError):
                                            pass
                                    elif name == 'wait_time' and value is not None:
                                        try:
                                            # Always store wait_time as positive value
                                            parameters['wait_time'] = abs(float(value))
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
                                    elif name == 'total_time' and value is not None:
                                        try:
                                            parameters['duration_total'] = float(value)
                                        except (ValueError, TypeError):
                                            pass
                                    elif name == 'show_start_diff' and value is not None:
                                        try:
                                            parameters['diff_flow'] = float(value)
                                        except (ValueError, TypeError):
                                            pass
                                    elif name == 'show_end_diff' and value is not None:
                                        try:
                                            parameters['diff_flow_stop'] = float(value)
                                        except (ValueError, TypeError):
                                            pass
                                    elif name == 'handover_time' and value is not None:
                                        try:
                                            parameters['duration_till_handover'] = float(value)
                                        except (ValueError, TypeError):
                                            pass
                                    elif name == 'weight' and value is not None:
                                        try:
                                            parameters['cc_weight'] = float(value)
                                        except (ValueError, TypeError):
                                            pass
                                    elif name == 'show_start_time' and value is not None:
                                        parameters['show_start_time'] = str(value)
                                    elif name == 'show_end_time' and value is not None:
                                        parameters['show_end_time'] = str(value)
                                    
                                    # Extract temperature parameters
                                    elif name == 'start_temp' and value is not None:
                                        try:
                                            parameters['start_temp'] = float(value)
                                        except (ValueError, TypeError):
                                            pass
                                    elif name == 'cook_temp' and value is not None:
                                        try:
                                            parameters['cook_temp'] = float(value)
                                        except (ValueError, TypeError):
                                            pass
                                    elif name == 'cooled_temp' and value is not None:
                                        try:
                                            parameters['cooled_temp'] = float(value)
                                        except (ValueError, TypeError):
                                            pass
                                    
                                    # Extract sizes (touch positions)
                                    elif name == 'sizes' and isinstance(value, dict):
                                        try:
                                            if 'pos1' in value:
                                                parameters['touch_pos1'] = float(value['pos1'])
                                            if 'pos2' in value:
                                                parameters['touch_pos2'] = float(value['pos2'])
                                            if 'pos3' in value:
                                                parameters['touch_pos3'] = float(value['pos3'])
                                        except (ValueError, TypeError):
                                            pass
                                    
                                    # Extract max_pressures
                                    elif name == 'max_pressures' and isinstance(value, dict):
                                        try:
                                            if 'pos1' in value:
                                                parameters['max_pos1'] = float(value['pos1'])
                                            if 'pos2' in value:
                                                parameters['max_pos2'] = float(value['pos2'])
                                            if 'pos3' in value:
                                                parameters['max_pos3'] = float(value['pos3'])
                                        except (ValueError, TypeError):
                                            pass
                                    
                                    # Extract f_pressures for separate CSV
                                    elif name == 'f_pressures' and isinstance(value, dict):
                                        try:
                                            import json
                                            if 'pos1' in value and isinstance(value['pos1'], list):
                                                f_pressures_data['pos1_pressures'] = json.dumps(value['pos1'])
                                            if 'pos2' in value and isinstance(value['pos2'], list):
                                                f_pressures_data['pos2_pressures'] = json.dumps(value['pos2'])
                                            if 'pos3' in value and isinstance(value['pos3'], list):
                                                f_pressures_data['pos3_pressures'] = json.dumps(value['pos3'])
                                        except Exception:
                                            pass
                
                # Calculate duration_cc_flow from show times
                if parameters['show_start_time'] and parameters['show_end_time']:
                    try:
                        start_time = datetime.strptime(parameters['show_start_time'], '%H:%M:%S').time()
                        end_time = datetime.strptime(parameters['show_end_time'], '%H:%M:%S').time()
                        
                        # Convert to datetime objects for calculation (using arbitrary date)
                        start_dt = datetime.combine(datetime.today(), start_time)
                        end_dt = datetime.combine(datetime.today(), end_time)
                        
                        # Handle case where end time is next day (unlikely but possible)
                        if end_dt < start_dt:
                            end_dt = end_dt.replace(day=end_dt.day + 1)
                        
                        duration_seconds = (end_dt - start_dt).total_seconds()
                        parameters['duration_cc_flow'] = round(duration_seconds, 2)
                    except Exception as e:
                        print(f"    Warning: Error calculating duration_cc_flow: {e}")
                
                # Extract environmental sensors using existing logic
                env_params = self._extract_environmental_data(events)
                parameters.update(env_params)
                                
        except Exception as e:
            print(f"    Warning: Error parsing {yaml_file_path.name}: {e}")
            
        return parameters, f_pressures_data
    
    def _extract_environmental_data(self, events: List[Dict]) -> Dict:
        """Extract environmental sensor data from events - reuse logic from v1"""
        env_parameters = {}
        
        # Initialize all environmental parameters
        env_keys = [
            'baseline_env_EnvH', 'baseline_env_EnvT',
            'before_turn_on_env_InH', 'before_turn_on_env_InT', 'before_turn_on_env_IrO', 'before_turn_on_env_IrA',
            'after_flow_start_env_InH', 'after_flow_start_env_InT', 'after_flow_start_env_IrO', 'after_flow_start_env_IrA',
            'after_flow_end_env_InH', 'after_flow_end_env_InT', 'after_flow_end_env_IrO', 'after_flow_end_env_IrA',
            'before_cooldown_env_InH', 'before_cooldown_env_InT', 'before_cooldown_env_IrO', 'before_cooldown_env_IrA',
            'after_cooldown_env_InH', 'after_cooldown_env_InT', 'after_cooldown_env_IrO', 'after_cooldown_env_IrA',
        ]
        
        for key in env_keys:
            env_parameters[key] = None
        
        # Track process phases for environmental sensor assignment
        show_start_done = False
        show_end_done = False
        weigh_cc_done = False
        run_until_cooled_done = False
        machine_cooled_off = False
        
        # Store the last recorded environmental sensors as fallback for after_cooldown
        last_environmental_reading = {}
        
        try:
            for event in events:
                # Track Show Start activity/done for phase detection
                if (event.get('concept:name') == 'Show Start' and
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
                
                # Track Run until CC Head is optimally cooled activity/done for phase detection
                elif (event.get('concept:name') == 'Run until CC Head is optimally cooled' and
                      event.get('cpee:lifecycle:transition') == 'activity/done'):
                    run_until_cooled_done = True
                
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
                        if 'EnvH' in sensor_data and env_parameters['baseline_env_EnvH'] is None:
                            env_parameters['baseline_env_EnvH'] = sensor_data['EnvH']
                        if 'EnvT' in sensor_data and env_parameters['baseline_env_EnvT'] is None:
                            env_parameters['baseline_env_EnvT'] = sensor_data['EnvT']
                        
                        # Handle internal sensors with phase detection
                        for sensor in ['InH', 'InT', 'IrO', 'IrA']:
                            if sensor in sensor_data:
                                # Determine phase based on process state
                                if machine_cooled_off:
                                    # After cooldown phase
                                    full_key = f"after_cooldown_env_{sensor}"
                                    if full_key in env_parameters and env_parameters[full_key] is None:
                                        env_parameters[full_key] = sensor_data[sensor]
                                elif run_until_cooled_done:
                                    # Before cooldown phase (first env data after "Run until CC Head is optimally cooled")
                                    full_key = f"before_cooldown_env_{sensor}"
                                    if full_key in env_parameters and env_parameters[full_key] is None:
                                        env_parameters[full_key] = sensor_data[sensor]
                                elif show_end_done:
                                    # After flow end phase
                                    full_key = f"after_flow_end_env_{sensor}"
                                    if full_key in env_parameters and env_parameters[full_key] is None:
                                        env_parameters[full_key] = sensor_data[sensor]
                                elif show_start_done:
                                    # After flow start phase
                                    full_key = f"after_flow_start_env_{sensor}"
                                    if full_key in env_parameters and env_parameters[full_key] is None:
                                        env_parameters[full_key] = sensor_data[sensor]
                                else:
                                    # Before turn on phase (default)
                                    full_key = f"before_turn_on_env_{sensor}"
                                    if full_key in env_parameters and env_parameters[full_key] is None:
                                        env_parameters[full_key] = sensor_data[sensor]
            
            # Fallback: Use last recorded environmental data for missing after_cooldown values
            for sensor in ['InH', 'InT', 'IrO', 'IrA']:
                full_key = f"after_cooldown_env_{sensor}"
                if full_key in env_parameters and env_parameters[full_key] is None:
                    if sensor in last_environmental_reading:
                        env_parameters[full_key] = last_environmental_reading[sensor]
                        
        except Exception as e:
            print(f"    Warning: Error extracting environmental data: {e}")
        
        return env_parameters
    
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
                     global_iteration_start: int = 0) -> Tuple[List[Dict], List[Dict]]:
        """
        Process a single batch folder
        
        Args:
            batch_folder: Path to batch folder
            target_process: If specified, only process this specific process index
            global_iteration_start: Starting global iteration number for this batch
        
        Returns:
            Tuple of (main_data_rows, f_pressures_data_rows)
        """
        batch_name = batch_folder.name
        batch_number = int(batch_name.split('-')[1])
        
        print(f"Processing {batch_name}...")
        
        index_file = batch_folder / 'index.txt'
        if not index_file.exists():
            print(f"  Warning: index.txt not found in {batch_name}")
            return [], []
            
        # Parse the index file to get processes
        processes = self.parse_index_file(index_file)
        
        if not processes:
            print(f"  Warning: No 'Cottonbot - Run with Data Collection*' processes found in {batch_name}")
            return [], []
            
        # Filter to specific process if requested
        if target_process is not None:
            if 0 <= target_process < len(processes):
                processes = [processes[target_process]]
                print(f"  Processing only process {target_process}")
            else:
                print(f"  Error: Process {target_process} not found. Available: 0-{len(processes)-1}")
                return [], []
        
        main_batch_data = []
        f_pressures_batch_data = []
        
        for i, process in enumerate(processes):
            # Calculate global iteration number
            global_iteration = global_iteration_start + i
            
            # Extract process parameters from the YAML file
            yaml_file = batch_folder / f"{process['uuid']}.xes.yaml"
            
            if yaml_file.exists():
                process_params, f_pressures_data = self.extract_process_parameters_v2(yaml_file)
            else:
                print(f"    Warning: YAML file not found for {process['uuid'][:8]}...")
                continue
            
            # Create main row data
            row_data = {
                'iteration': global_iteration,
                'batch_number': batch_number,
                'stick_number': process['stick_number'],
                'index_log': process['index_log'],
            }
            
            # Add all other parameters
            for col in self.columns:
                if col not in ['iteration', 'batch_number', 'stick_number', 'index_log']:
                    row_data[col] = process_params.get(col)
            
            main_batch_data.append(row_data)
            
            # Create f_pressures row data
            f_pressures_row = {
                'iteration': global_iteration,
                'batch_number': batch_number,
                'stick_number': process['stick_number'],
                'pos1_pressures': f_pressures_data.get('pos1_pressures'),
                'pos2_pressures': f_pressures_data.get('pos2_pressures'),
                'pos3_pressures': f_pressures_data.get('pos3_pressures'),
            }
            
            f_pressures_batch_data.append(f_pressures_row)
            
            print(f"  -> Process {i}: Global Iteration {global_iteration}, UUID={process['uuid'][:8]}..., Index={process['index_log']}")
        
        return main_batch_data, f_pressures_batch_data
    
    def load_existing_data(self, file_path: str, columns: List[str]) -> pd.DataFrame:
        """Load existing CSV data if it exists"""
        if os.path.exists(file_path):
            try:
                return pd.read_csv(file_path)
            except Exception as e:
                print(f"Error loading existing data from {file_path}: {e}")
                return pd.DataFrame(columns=columns)
        else:
            return pd.DataFrame(columns=columns)
    
    def update_csv_data(self, new_main_data: List[Dict], new_f_pressures_data: List[Dict], 
                       batch_number: Optional[int] = None, process_index: Optional[int] = None):
        """
        Update both CSV files with new data
        
        Args:
            new_main_data: List of new main data rows
            new_f_pressures_data: List of new f_pressures data rows
            batch_number: If specified, only update rows for this batch
            process_index: If specified, only update this specific process within the batch
        """
        if not new_main_data:
            print("No data to update")
            return
        
        # Update main CSV
        existing_main_df = self.load_existing_data(self.output_file, self.columns)
        
        if batch_number is not None:
            if process_index is not None:
                # Remove specific row - identify by batch_number and stick_number (which matches process_index)
                mask = (existing_main_df['batch_number'] == batch_number) & \
                       (existing_main_df['stick_number'] == process_index)
                existing_main_df = existing_main_df[~mask]
                print(f"Updating batch {batch_number}, process {process_index}")
            else:
                # Remove entire batch
                existing_main_df = existing_main_df[existing_main_df['batch_number'] != batch_number]
                print(f"Updating entire batch {batch_number}")
        else:
            # Full update - clear all data
            existing_main_df = pd.DataFrame(columns=self.columns)
            print("Full update - replacing all main data")
        
        # Add new main data
        new_main_df = pd.DataFrame(new_main_data)
        combined_main_df = pd.concat([existing_main_df, new_main_df], ignore_index=True)
        
        # Sort by batch_number, then by iteration
        combined_main_df = combined_main_df.sort_values(['batch_number', 'iteration']).reset_index(drop=True)
        
        # Save main CSV
        combined_main_df.to_csv(self.output_file, index=False, float_format='%.2f')
        print(f"Main data saved to {self.output_file}")
        print(f"Total main rows: {len(combined_main_df)}")
        
        # Update f_pressures CSV
        existing_f_pressures_df = self.load_existing_data(self.f_pressures_file, self.f_pressures_columns)
        
        if batch_number is not None:
            if process_index is not None:
                # Remove specific row
                mask = (existing_f_pressures_df['batch_number'] == batch_number) & \
                       (existing_f_pressures_df['stick_number'] == process_index)
                existing_f_pressures_df = existing_f_pressures_df[~mask]
            else:
                # Remove entire batch
                existing_f_pressures_df = existing_f_pressures_df[existing_f_pressures_df['batch_number'] != batch_number]
        else:
            # Full update - clear all data
            existing_f_pressures_df = pd.DataFrame(columns=self.f_pressures_columns)
            print("Full update - replacing all f_pressures data")
        
        # Add new f_pressures data
        new_f_pressures_df = pd.DataFrame(new_f_pressures_data)
        combined_f_pressures_df = pd.concat([existing_f_pressures_df, new_f_pressures_df], ignore_index=True)
        
        # Sort by batch_number, then by iteration
        combined_f_pressures_df = combined_f_pressures_df.sort_values(['batch_number', 'iteration']).reset_index(drop=True)
        
        # Save f_pressures CSV
        combined_f_pressures_df.to_csv(self.f_pressures_file, index=False)
        print(f"F_pressures data saved to {self.f_pressures_file}")
        print(f"Total f_pressures rows: {len(combined_f_pressures_df)}")
    
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
        
        all_main_data = []
        all_f_pressures_data = []
        
        if process_all:
            # Process all batches with global iteration counter
            global_iteration = 0
            for batch_folder in batch_folders:
                main_data, f_pressures_data = self.process_batch(batch_folder, None, global_iteration)
                all_main_data.extend(main_data)
                all_f_pressures_data.extend(f_pressures_data)
                global_iteration += len(main_data)
            
            self.update_csv_data(all_main_data, all_f_pressures_data)
            
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
                
            main_data, f_pressures_data = self.process_batch(target_folder, process_index, global_iteration_start)
            self.update_csv_data(main_data, f_pressures_data, batch_number, process_index)
            
        else:
            print("Please specify --process-all, --batch, or --batch with --process")
            return
        
        print("\nPipeline V2 completed successfully!")

def main():
    parser = argparse.ArgumentParser(
        description='Cotton Candy Digital Twin Data Pipeline V2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cc_data_extraction_v2.py --process-all
  python cc_data_extraction_v2.py --batch 0
  python cc_data_extraction_v2.py --batch 0 --process 1
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
    parser.add_argument('--output', default='Data_Collection/cotton_candy_dataset_v2.csv',
                       help='Output main CSV file (default: Data_Collection/cotton_candy_dataset_v2.csv)')
    parser.add_argument('--f-pressures-output', default='Data_Collection/f_pressures_dataset.csv',
                       help='Output f_pressures CSV file (default: Data_Collection/f_pressures_dataset.csv)')
    
    args = parser.parse_args()
    
    # Validation
    if not args.process_all and args.batch is None:
        parser.error('Must specify either --process-all or --batch')
    
    if args.process is not None and args.batch is None:
        parser.error('--process can only be used with --batch')
    
    # Redirect to v1 for batches < 20
    if args.batch is not None and args.batch < 20:
        print(f"Batch {args.batch} is < 20, redirecting to v1 script...")
        import subprocess
        import sys
        
        # Build command for v1 script
        v1_command = [sys.executable, 'cc_data_extraction_v1.py']
        v1_command.extend(['--batch', str(args.batch)])
        
        if args.process is not None:
            v1_command.extend(['--process', str(args.process)])
        
        if args.data_dir != 'Data_Collection/Batches':
            v1_command.extend(['--data-dir', args.data_dir])
        
        if args.f_pressures_output != 'Data_Collection/f_pressures_dataset.csv':
            v1_command.extend(['--f-pressures-output', args.f_pressures_output])
        
        # Execute v1 script
        try:
            result = subprocess.run(v1_command, check=True)
            sys.exit(result.returncode)
        except subprocess.CalledProcessError as e:
            print(f"Error running v1 script: {e}")
            sys.exit(e.returncode)
        except FileNotFoundError:
            print("Error: cc_data_extraction_v1.py not found in current directory")
            sys.exit(1)
    
    # Continue with v2 for batches >= 20 or process-all
    # Initialize and run pipeline
    pipeline = CottonCandyPipelineV2(args.data_dir, args.output, args.f_pressures_output)
    pipeline.run(
        process_all=args.process_all,
        batch_number=args.batch,
        process_index=args.process
    )

if __name__ == "__main__":
    main()
