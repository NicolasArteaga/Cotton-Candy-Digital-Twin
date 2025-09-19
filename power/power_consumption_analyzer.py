#!/usr/bin/env python3
"""
Cotton Candy Digital Twin - Power Consumption Analysis
This script analyzes power consumption from process logs using trapezoidal integration.

The script extracts plug stream data from CPEE log files and calculates total energy
consumption for each process using the trapezoidal integration method:

E = Σ(P[i-1] + P[i])/2 * Δt[i]

where P[i] is the measured power at time t[i] and Δt[i] is the interval between measurements.

Usage:
    python power_consumption_analyzer.py --process-all
    python power_consumption_analyzer.py --batch 0
    python power_consumption_analyzer.py --batch 0 --process 1
"""

import argparse
import re
import os
import csv
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import yaml
from datetime import datetime, timedelta
import json


class PowerConsumptionAnalyzer:
    def __init__(self, data_dir: str = "Data_Collection/Batches", 
                 output_file: str = "power/power_consumption_analysis.csv"):
        self.data_dir = Path(data_dir)
        self.output_file = output_file
        
        # Define CSV columns for power analysis
        self.columns = [
            'iteration',
            'batch_number', 
            'stick_number',
            'index_log',
            'process_uuid',
            'process_start_time',
            'process_end_time',
            'total_duration_seconds',
            'total_power_measurements',
            'avg_power_watts',
            'max_power_watts',
            'min_power_watts',
            'total_energy_watt_seconds',
            'total_energy_watt_hours',
            'energy_per_minute_watt_hours',
            'power_measurement_intervals_seconds',  # JSON array of intervals between measurements
            'power_values_watts',  # JSON array of power values
            'power_timestamps',  # JSON array of timestamps
            'current_values_amps',  # JSON array of current values (if available)
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
    
    def extract_plug_data(self, yaml_file_path: Path) -> Dict:
        """
        Extract all plug power measurements from YAML log file and calculate energy consumption
        
        Args:
            yaml_file_path: Path to the YAML log file
        
        Returns:
            Dictionary with power analysis results
        """
        
        power_data = {
            'process_start_time': None,
            'process_end_time': None,
            'total_duration_seconds': None,
            'total_power_measurements': 0,
            'avg_power_watts': None,
            'max_power_watts': None,
            'min_power_watts': None,
            'total_energy_watt_seconds': None,
            'total_energy_watt_hours': None,
            'energy_per_minute_watt_hours': None,
            'power_measurement_intervals_seconds': None,
            'power_values_watts': None,
            'power_timestamps': None,
            'current_values_amps': None,
        }
        
        # Lists to collect all power measurements
        power_measurements = []  # [(timestamp, power_watts, current_amps), ...]
        
        try:
            with open(yaml_file_path, 'r') as f:
                for doc in yaml.safe_load_all(f):
                    if 'event' in doc:
                        event = doc['event']
                        
                        # Track process start and end times
                        timestamp_str = event.get('time:timestamp')
                        if timestamp_str:
                            timestamp = self._parse_timestamp(timestamp_str)
                            if timestamp:
                                if power_data['process_start_time'] is None:
                                    power_data['process_start_time'] = timestamp
                                power_data['process_end_time'] = timestamp
                        
                        # Extract plug data from "Get the Plug Data" stream/data events
                        if (event.get('concept:name') == 'Get the Plug Data' and
                            event.get('cpee:lifecycle:transition') == 'stream/data' and
                            'stream:datastream' in event):
                            
                            plug_measurement = self._extract_plug_measurement(event)
                            if plug_measurement:
                                power_measurements.append(plug_measurement)
            
            # Calculate power consumption metrics
            if power_measurements:
                power_data.update(self._calculate_power_metrics(power_measurements))
                                
        except Exception as e:
            print(f"    Warning: Error parsing {yaml_file_path.name}: {e}")
            
        return power_data
    
    def _parse_timestamp(self, timestamp_str: str) -> Optional[datetime]:
        """Parse timestamp string to datetime object"""
        try:
            # Handle timezone format
            if timestamp_str.endswith('+02:00'):
                timestamp_str = timestamp_str[:-6]
            return datetime.fromisoformat(timestamp_str)
        except:
            return None
    
    def _extract_plug_measurement(self, event: Dict) -> Optional[Tuple[datetime, float, Optional[float]]]:
        """
        Extract power and current measurements from a plug stream/data event
        
        Returns:
            Tuple of (timestamp, power_watts, current_amps) or None if no valid data
        """
        stream_datastream = event.get('stream:datastream', [])
        
        power_value = None
        current_value = None
        timestamp = None
        
        for stream_item in stream_datastream:
            if not isinstance(stream_item, dict):
                continue
            
            # Look for stream points with power and current data
            stream_point = stream_item.get('stream:point')
            if isinstance(stream_point, dict):
                stream_id = stream_point.get('stream:id')
                stream_value = stream_point.get('stream:value')
                stream_timestamp = stream_point.get('stream:timestamp')
                
                if stream_id == 'power' and stream_value is not None:
                    try:
                        power_value = float(stream_value)
                        if stream_timestamp:
                            timestamp = self._parse_plug_timestamp(stream_timestamp)
                    except (ValueError, TypeError):
                        pass
                
                elif stream_id == 'current' and stream_value is not None:
                    try:
                        current_value = float(stream_value)
                    except (ValueError, TypeError):
                        pass
        
        # Return measurement if we have power data
        if power_value is not None and timestamp is not None:
            return (timestamp, power_value, current_value)
        
        return None
    
    def _parse_plug_timestamp(self, timestamp_str: str) -> Optional[datetime]:
        """Parse plug-specific timestamp format"""
        try:
            # Handle format like '2025-07-28 12:57:18.76'
            return datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f')
        except:
            try:
                # Fallback for different formats
                return datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
            except:
                return None
    
    def _calculate_power_metrics(self, power_measurements: List[Tuple[datetime, float, Optional[float]]]) -> Dict:
        """
        Calculate power consumption metrics using trapezoidal integration
        
        Args:
            power_measurements: List of (timestamp, power_watts, current_amps) tuples
        
        Returns:
            Dictionary with calculated metrics
        """
        if not power_measurements:
            return {}
        
        # Sort measurements by timestamp
        power_measurements.sort(key=lambda x: x[0])
        
        # Extract data arrays
        timestamps = [m[0] for m in power_measurements]
        power_values = [m[1] for m in power_measurements]
        current_values = [m[2] for m in power_measurements if m[2] is not None]
        
        # Calculate basic statistics
        total_measurements = len(power_measurements)
        avg_power = sum(power_values) / total_measurements
        max_power = max(power_values)
        min_power = min(power_values)
        
        # Calculate intervals between measurements
        intervals = []
        for i in range(1, len(timestamps)):
            interval = (timestamps[i] - timestamps[i-1]).total_seconds()
            intervals.append(interval)
        
        # Calculate total energy using trapezoidal integration
        # E = Σ(P[i-1] + P[i])/2 * Δt[i]
        total_energy_ws = 0.0  # Watt-seconds
        
        for i in range(1, len(power_values)):
            power_prev = power_values[i-1]
            power_curr = power_values[i]
            dt = intervals[i-1]  # Time interval in seconds
            
            # Trapezoidal rule: area = (height1 + height2) * width / 2
            energy_segment = (power_prev + power_curr) * dt / 2
            total_energy_ws += energy_segment
        
        # Convert to watt-hours
        total_energy_wh = total_energy_ws / 3600
        
        # Calculate total duration
        total_duration = (timestamps[-1] - timestamps[0]).total_seconds()
        
        # Calculate energy per minute
        energy_per_minute_wh = total_energy_wh / (total_duration / 60) if total_duration > 0 else None
        
        # Prepare JSON-serializable data
        power_timestamps_str = [ts.isoformat() for ts in timestamps]
        
        return {
            'total_duration_seconds': round(total_duration, 2),
            'total_power_measurements': total_measurements,
            'avg_power_watts': round(avg_power, 3),
            'max_power_watts': round(max_power, 3),
            'min_power_watts': round(min_power, 3),
            'total_energy_watt_seconds': round(total_energy_ws, 2),
            'total_energy_watt_hours': round(total_energy_wh, 4),
            'energy_per_minute_watt_hours': round(energy_per_minute_wh, 4) if energy_per_minute_wh else None,
            'power_measurement_intervals_seconds': json.dumps([round(i, 2) for i in intervals]),
            'power_values_watts': json.dumps([round(p, 3) for p in power_values]),
            'power_timestamps': json.dumps(power_timestamps_str),
            'current_values_amps': json.dumps([round(c, 3) for c in current_values]) if current_values else None,
        }
    
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
        Process a single batch folder for power consumption analysis
        
        Args:
            batch_folder: Path to batch folder
            target_process: If specified, only process this specific process index
            global_iteration_start: Starting global iteration number for this batch
        
        Returns:
            List of processed power analysis rows
        """
        batch_name = batch_folder.name
        batch_number = int(batch_name.split('-')[1])
        
        print(f"Processing {batch_name} for power analysis...")
        
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
                print(f"  Processing only process {target_process} for power analysis")
            else:
                print(f"  Error: Process {target_process} not found. Available: 0-{len(processes)-1}")
                return []
        
        batch_data = []
        
        for i, process in enumerate(processes):
            # Calculate global iteration number
            global_iteration = global_iteration_start + i
            
            # Extract power consumption data from the YAML file
            yaml_file = batch_folder / f"{process['uuid']}.xes.yaml"
            power_data = {}
            
            if yaml_file.exists():
                power_data = self.extract_plug_data(yaml_file)
                print(f"    -> Process {i}: {power_data.get('total_power_measurements', 0)} power measurements, "
                      f"Energy: {power_data.get('total_energy_watt_hours', 0):.4f} Wh")
            else:
                print(f"    Warning: YAML file not found for {process['uuid'][:8]}...")
            
            # Create row data
            row_data = {
                'iteration': global_iteration,
                'batch_number': batch_number,
                'stick_number': process['stick_number'],
                'index_log': process['index_log'],
                'process_uuid': process['uuid'],
                'process_start_time': power_data.get('process_start_time').isoformat() if power_data.get('process_start_time') else None,
                'process_end_time': power_data.get('process_end_time').isoformat() if power_data.get('process_end_time') else None,
                'total_duration_seconds': power_data.get('total_duration_seconds'),
                'total_power_measurements': power_data.get('total_power_measurements', 0),
                'avg_power_watts': power_data.get('avg_power_watts'),
                'max_power_watts': power_data.get('max_power_watts'),
                'min_power_watts': power_data.get('min_power_watts'),
                'total_energy_watt_seconds': power_data.get('total_energy_watt_seconds'),
                'total_energy_watt_hours': power_data.get('total_energy_watt_hours'),
                'energy_per_minute_watt_hours': power_data.get('energy_per_minute_watt_hours'),
                'power_measurement_intervals_seconds': power_data.get('power_measurement_intervals_seconds'),
                'power_values_watts': power_data.get('power_values_watts'),
                'power_timestamps': power_data.get('power_timestamps'),
                'current_values_amps': power_data.get('current_values_amps'),
            }
            
            batch_data.append(row_data)
        
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
        Update CSV file with new power consumption data
        
        Args:
            new_data: List of new data rows
            batch_number: If specified, only update rows for this batch
            process_index: If specified, only update this specific process within the batch
        """
        if not new_data:
            print("No power consumption data to update")
            return
            
        # Load existing data
        existing_df = self.load_existing_data()
        
        if batch_number is not None:
            if process_index is not None:
                # Remove specific row - identify by batch_number and stick_number
                mask = (existing_df['batch_number'] == batch_number) & \
                       (existing_df['stick_number'] == process_index)
                existing_df = existing_df[~mask]
                print(f"Updating power data for batch {batch_number}, process {process_index}")
            else:
                # Remove entire batch
                existing_df = existing_df[existing_df['batch_number'] != batch_number]
                print(f"Updating power data for entire batch {batch_number}")
        else:
            # Full update - clear all data
            existing_df = pd.DataFrame(columns=self.columns)
            print("Full power analysis update - replacing all data")
        
        # Add new data
        new_df = pd.DataFrame(new_data)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        
        # Sort by batch_number, then by iteration
        combined_df = combined_df.sort_values(['batch_number', 'iteration']).reset_index(drop=True)
        
        # Save to CSV
        combined_df.to_csv(self.output_file, index=False)
        print(f"Power consumption data saved to {self.output_file}")
        print(f"Total rows: {len(combined_df)}")
        
        # Generate summary statistics
        self._print_power_summary(combined_df)
    
    def _print_power_summary(self, df: pd.DataFrame):
        """Print summary statistics of power consumption analysis"""
        if df.empty:
            return
        
        print("\n=== POWER CONSUMPTION SUMMARY ===")
        print(f"Total processes analyzed: {len(df)}")
        
        # Filter out rows with actual measurements
        measured_df = df[df['total_power_measurements'] > 0]
        
        if not measured_df.empty:
            print(f"Processes with power measurements: {len(measured_df)}")
            print(f"Average power consumption: {measured_df['avg_power_watts'].mean():.3f} W")
            print(f"Average energy per process: {measured_df['total_energy_watt_hours'].mean():.4f} Wh")
            print(f"Total energy consumed (all processes): {measured_df['total_energy_watt_hours'].sum():.4f} Wh")
            print(f"Max single process energy: {measured_df['total_energy_watt_hours'].max():.4f} Wh")
            print(f"Min single process energy: {measured_df['total_energy_watt_hours'].min():.4f} Wh")
            print(f"Average process duration: {measured_df['total_duration_seconds'].mean():.1f} seconds")
        else:
            print("No processes found with valid power measurements")
        print("================================\n")
    
    def run(self, process_all: bool = False, batch_number: Optional[int] = None, 
            process_index: Optional[int] = None):
        """
        Main power analysis execution
        
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
        
        print("\nPower consumption analysis completed successfully!")


def main():
    parser = argparse.ArgumentParser(
        description='Cotton Candy Digital Twin Power Consumption Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python power_consumption_analyzer.py --process-all
  python power_consumption_analyzer.py --batch 0
  python power_consumption_analyzer.py --batch 0 --process 1
        
This script analyzes power consumption using trapezoidal integration:
E = Σ(P[i-1] + P[i])/2 * Δt[i]

Output includes total energy in Watt-hours, average power, and detailed measurements.
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
    parser.add_argument('--output', default='power/power_consumption_analysis.csv',
                       help='Output CSV file (default: power/power_consumption_analysis.csv)')
    
    args = parser.parse_args()
    
    # Validation
    if not args.process_all and args.batch is None:
        parser.error('Must specify either --process-all or --batch')
    
    if args.process is not None and args.batch is None:
        parser.error('--process can only be used with --batch')
    
    # Initialize and run power consumption analyzer
    analyzer = PowerConsumptionAnalyzer(args.data_dir, args.output)
    analyzer.run(
        process_all=args.process_all,
        batch_number=args.batch,
        process_index=args.process
    )


if __name__ == "__main__":
    main()