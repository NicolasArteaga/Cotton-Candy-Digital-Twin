import yaml
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np

class CottonCandyLogParser:
    def __init__(self, yaml_file_path: str):
        self.yaml_file_path = yaml_file_path
        self.events = []
        self.env_data = []
        self.weights = []
        self.timestamps = []
        
    def parse_yaml_efficiently(self):
        """Parse YAML file efficiently using streaming to handle large files"""
        with open(self.yaml_file_path, 'r') as f:
            for doc in yaml.safe_load_all(f):
                if 'event' in doc:
                    event = doc['event']
                    # Filter plug events - only keep the ones with actual data
                    if self._should_keep_event(event):
                        self.events.append(event)
    
    def _should_keep_event(self, event: Dict) -> bool:
        """Filter out unnecessary plug and environment events, keeping only the stream/data ones"""
        activity_name = event.get('concept:name', '')
        lifecycle_transition = event.get('cpee:lifecycle:transition', '')
        
        # For plug data events, only keep stream/data for real-time measurements
        if activity_name == 'Get the Plug Data':
            return lifecycle_transition == 'stream/data'
        
        # For environment data events, only keep stream/data
        elif activity_name == 'Get the Environment Data':
            return lifecycle_transition == 'stream/data'
        
        # Keep all other events
        return True
    
    def extract_environmental_data(self, event: Dict) -> Dict:
        """Extract environmental sensor data using simple stream filter strategy"""
        # Only process stream/data events
        if event.get('cpee:lifecycle:transition') != 'stream/data':
            return {}
            
        stream_datastream = event.get('stream:datastream', [])
        sensor_data = {}
        
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
                                sensor_data['env_EnvH'] = float(stream_value)
                            elif stream_id == 'temperature':
                                sensor_data['env_EnvT'] = float(stream_value)
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
                                        sensor_data['env_InH'] = float(stream_value)
                                    elif stream_id == 'temperature':
                                        sensor_data['env_InT'] = float(stream_value)
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
                                                        sensor_data['env_IrO'] = float(stream_value)
                                                    elif stream_id == 'ambient':
                                                        sensor_data['env_IrA'] = float(stream_value)
                                                except (ValueError, TypeError):
                                                    pass
        
        return sensor_data
    
    def extract_process_parameters(self, event: Dict) -> Dict:
        """Extract process parameters like cook_time, wait_time, etc."""
        if 'data' not in event or event['data'] is None:
            return {}
            
        params = {}
        
        # Handle different data structures
        data_items = event['data']
        if isinstance(data_items, list):
            for data_item in data_items:
                if data_item is None or not isinstance(data_item, dict):
                    continue
                name = data_item.get('name', '')
                value = data_item.get('value', '')

                if name in ['wait_time', 'cook_time', 'cooldown_time',
                           'iteration_since_maintenance']:
                    try:
                        params[name] = float(value) if value else None
                    except (ValueError, TypeError):
                        params[name] = None
                        
        return params
    
    def extract_power_data(self, event: Dict) -> Optional[Dict]:
        """Extract power consumption data using simple stream filter strategy"""
        # Only process stream/data events for real-time power measurements
        if event.get('cpee:lifecycle:transition') != 'stream/data':
            return None
            
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
            power_data = {}
            
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
                        power_data['power'] = float(stream_value)
                    except (ValueError, TypeError):
                        power_data['power'] = stream_value
                elif stream_id == 'current' and stream_value is not None:
                    try:
                        power_data['current'] = float(stream_value)
                    except (ValueError, TypeError):
                        power_data['current'] = stream_value
            
            if 'power' in power_data:
                return power_data
        
        return None

    def calculate_quality_metrics(self) -> Dict:
        """Calculate quality and performance metrics from weight, pressure, and other measurements"""
        quality_metrics = {}
        
        # Extract weight measurements for quality assessment
        weights = []
        max_pressures = []
        sizes = []
        
        for event in self.events:
            if 'data' not in event or event['data'] is None:
                continue
                
            # Handle different data structures
            data_items = event['data']
            if isinstance(data_items, list):
                for data_item in data_items:
                    if data_item is None or not isinstance(data_item, dict):
                        continue
                    name = data_item.get('name', '')
                    value = data_item.get('value', '')
                    
                    # Weight measurements
                    if name == 'weight':
                        try:
                            weight = float(value) if value else None
                            if weight is not None:
                                weights.append(weight)
                        except (ValueError, TypeError):
                            pass
                    
                    # Pressure measurements for quality
                    elif name == 'max_pressures' and isinstance(value, dict):
                        pressure_sum = 0
                        pressure_count = 0
                        for pressure_key, pressure_value in value.items():
                            try:
                                pressure = float(pressure_value) if pressure_value else None
                                if pressure is not None:
                                    pressure_sum += pressure
                                    pressure_count += 1
                            except (ValueError, TypeError):
                                pass
                        if pressure_count > 0:
                            max_pressures.append(pressure_sum / pressure_count)
                    
                    # Size measurements for quality
                    elif name == 'sizes' and isinstance(value, dict):
                        size_sum = 0
                        size_count = 0
                        for size_key, size_value in value.items():
                            try:
                                size = float(size_value) if size_value else None
                                if size is not None:
                                    size_sum += size
                                    size_count += 1
                            except (ValueError, TypeError):
                                pass
                        if size_count > 0:
                            sizes.append(size_sum / size_count)
        
        # Calculate quality metrics
        if weights:
            quality_metrics['final_weight'] = max(weights)  # Final weight
            quality_metrics['weight_consistency'] = 1.0 / (np.std(weights) + 0.1)  # Lower std = higher consistency
        
        if max_pressures:
            quality_metrics['avg_pressure'] = np.mean(max_pressures)
            quality_metrics['pressure_stability'] = 1.0 / (np.std(max_pressures) + 0.1)
        
        if sizes:
            quality_metrics['avg_size'] = np.mean(sizes)
            quality_metrics['size_consistency'] = 1.0 / (np.std(sizes) + 0.1)
        
        return quality_metrics

    def calculate_total_energy_consumption(self, unit: str = 'wh') -> float:
        """
        Calculate total energy consumption from power measurements
        
        Args:
            unit: 'wh' for Watt-hours, 'kwh' for kilowatt-hours, 'j' for Joules
            
        Returns:
            Energy consumption in specified unit
        """
        power_measurements = []
        
        for event in self.events:
            timestamp_str = event.get('time:timestamp', '')
            if not timestamp_str:
                continue
                
            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            except:
                continue
                
            power_data = self.extract_power_data(event)
            if power_data and 'power' in power_data:
                power_measurements.append((timestamp, power_data['power']))
        
        print(f"Found {len(power_measurements)} power measurements")
        if len(power_measurements) >= 2:
            print(f"Power range: {min(p[1] for p in power_measurements)} - {max(p[1] for p in power_measurements)} W")
        
        if len(power_measurements) < 2:
            return 0.0
        
        # Calculate energy using trapezoidal integration
        # For short processes (max 5 minutes), use seconds for better precision
        total_energy_ws = 0.0  # Watt-seconds (Joules)
        for i in range(1, len(power_measurements)):
            prev_time, prev_power = power_measurements[i-1]
            curr_time, curr_power = power_measurements[i]
            
            # Time difference in seconds
            time_diff_seconds = (curr_time - prev_time).total_seconds()
            
            # Average power over the interval (Watts)
            avg_power = (prev_power + curr_power) / 2.0
            
            # Energy in this interval (Watt-seconds = Joules)
            energy_interval = avg_power * time_diff_seconds
            total_energy_ws += energy_interval
        
        # Return in requested unit
        if unit.lower() == 'j' or unit.lower() == 'joules':
            return total_energy_ws  # Watt-seconds = Joules
        elif unit.lower() == 'wh':
            return total_energy_ws / 3600.0  # Convert to Watt-hours
        elif unit.lower() == 'kwh':
            return total_energy_ws / 3600000.0  # Convert to kWh
        else:
            # Default to Watt-hours for compatibility
            return total_energy_ws / 3600.0
    
    def identify_process_phases(self) -> Dict[str, datetime]:
        """Identify key timestamps in the cotton candy making process"""
        phases = {
            'process_start': None,
            'machine_turn_on': None,
            'machine_turn_off': None,
            'process_end': None,
            'handover': None,  # When "Creates Cotton Candy" completes
            'flow_start': None,  # When "Show Start" completes
            'flow_end': None,    # When "Show End" completes
            'weigh_start': None  # When "Weigh the whole Cotton Candy" starts
        }
        
        for event in self.events:
            timestamp_str = event.get('time:timestamp', '')
            if not timestamp_str:
                continue
                
            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            except:
                continue
                
            # Set process start (first event)
            if phases['process_start'] is None:
                phases['process_start'] = timestamp
                
            # Update process end (last event)
            phases['process_end'] = timestamp
            
            # Look for machine control events
            activity_name = str(event.get('concept:name', '')).lower()
            endpoint = str(event.get('concept:endpoint', '')).lower()
            
            if 'turn_on' in activity_name or 'turn_on' in endpoint:
                phases['machine_turn_on'] = timestamp
            elif 'turn_off' in activity_name or 'turn_off' in endpoint:
                phases['machine_turn_off'] = timestamp
            
            # Look for "Creates Cotton Candy" completion (handover point)
            elif activity_name == 'creates cotton candy' and event.get('lifecycle:transition') == 'complete':
                phases['handover'] = timestamp
            
            # Look for "Show Start" completion (cotton candy starts flowing)
            elif activity_name == 'show start' and event.get('lifecycle:transition') == 'complete':
                phases['flow_start'] = timestamp
            
            # Look for "Show End" completion (cotton candy stops flowing)
            elif activity_name == 'show end' and event.get('lifecycle:transition') == 'complete':
                phases['flow_end'] = timestamp
            
            # Look for "Weigh the whole Cotton Candy" start (cooling phase beginning)
            elif 'weigh the whole cotton candy' in activity_name and event.get('lifecycle:transition') == 'unknown':
                phases['weigh_start'] = timestamp
                    
        return phases
    
    def get_environmental_state_at_phase(self, target_timestamp: datetime) -> Dict:
        """Get the environmental conditions closest to a specific timestamp"""
        if target_timestamp is None:
            return {}
            
        closest_env = {}
        min_time_diff = float('inf')
        
        for event in self.events:
            timestamp_str = event.get('time:timestamp', '')
            if not timestamp_str:
                continue
                
            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            except:
                continue
                
            env_data = self.extract_environmental_data(event)
            if env_data:
                time_diff = abs((timestamp - target_timestamp).total_seconds())
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    closest_env = env_data.copy()
                    closest_env['timestamp_diff'] = time_diff
                    
        return closest_env
    
    def create_feature_vector(self) -> Dict:
        """Create a clean feature vector for decision tree modeling"""
        phases = self.identify_process_phases()
        
        # Get environmental conditions at 5 key phases
        env_before_turn_on = self.get_environmental_state_at_phase(phases['machine_turn_on'])
        env_after_flow_start = self.get_environmental_state_at_phase(phases['flow_start'])
        env_after_flow_end = self.get_environmental_state_at_phase(phases['flow_end'])
        env_after_weigh = self.get_environmental_state_at_phase(phases['weigh_start'])
        env_end = self.get_environmental_state_at_phase(phases['process_end'])
        
        # Extract process parameters - hardcoded to look in third event (index 2)
        process_params = {
            'wait_time': None,
            'cook_time': None,
            'cooldown_time': None,
            'iteration_since_maintenance': None
        }
        
        # Process parameters are always in the third event (index 2)
        if len(self.events) > 2:
            third_event = self.events[2]
            if 'data' in third_event and isinstance(third_event['data'], list):
                for item in third_event['data']:
                    if isinstance(item, dict) and 'name' in item and 'value' in item:
                        name = item['name']
                        if name in process_params:
                            value = item['value']
                            # Convert to appropriate type
                            if isinstance(value, (int, float)):
                                process_params[name] = value
                            elif isinstance(value, str) and value.isdigit():
                                process_params[name] = int(value)
                            else:
                                process_params[name] = value
        
        print(f"Process parameters from third event: {process_params}")
        
        # Calculate timing metrics (features)
        timing_metrics = {}
        if phases['process_start'] and phases['process_end']:
            timing_metrics['duration_total'] = (phases['process_end'] - phases['process_start']).total_seconds()
        if phases['process_start'] and phases['handover']:
            timing_metrics['duration_till_handover'] = (phases['handover'] - phases['process_start']).total_seconds()
        
        # Calculate targets (what we want to optimize)
        targets = {}
        # Energy consumption (MINIMIZE this - lower is better)
        targets['total_energy_wh'] = self.calculate_total_energy_consumption('wh')
        # Quality metrics (MAXIMIZE these - higher is better)
        targets.update(self.calculate_quality_metrics())
        
        # Build clean feature vector (X - input variables) in specific order
        feature_vector = {}
        
        # 1. First: iteration_since_maintenance
        if 'iteration_since_maintenance' in process_params:
            feature_vector['iteration_since_maintenance'] = process_params['iteration_since_maintenance']
        
        # 2. Then: core process parameters
        for param in ['wait_time', 'cook_time', 'cooldown_time']:
            if param in process_params:
                feature_vector[param] = process_params[param]
        
        # 3. Then: duration_till_handover
        if 'duration_till_handover' in timing_metrics:
            feature_vector['duration_till_handover'] = timing_metrics['duration_till_handover']
        
        # 4. Then: duration_total
        if 'duration_total' in timing_metrics:
            feature_vector['duration_total'] = timing_metrics['duration_total']
        
        # 5. Finally: environmental features - external baseline + internal dynamics
        # External conditions (baseline from before_turn_on phase only)
        for sensor_name, value in env_before_turn_on.items():
            if sensor_name not in ['timestamp_diff'] and sensor_name.startswith('env_Env'):
                # Only store external sensors (EnvH, EnvT) from the first phase
                feature_vector[f'baseline_{sensor_name}'] = value
        
        # Internal sensors across all 5 phases (dynamic measurements)
        for phase_name, env_data in [
            ('before_turn_on', env_before_turn_on),
            ('after_flow_start', env_after_flow_start),
            ('after_flow_end', env_after_flow_end),
            ('after_weigh', env_after_weigh),
            ('end', env_end)
        ]:
            for sensor_name, value in env_data.items():
                if sensor_name not in ['timestamp_diff'] and not sensor_name.startswith('env_Env'):
                    # Store internal sensors (InH, InT, IrA, IrO) for all phases
                    feature_vector[f'{phase_name}_{sensor_name}'] = value
        
        return feature_vector, targets
    
    def process_file(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Main method to process the file and return features and targets as DataFrames"""
        print(f"Parsing {self.yaml_file_path}...")
        self.parse_yaml_efficiently()
        print(f"Extracted {len(self.events)} events")
        
        features, targets = self.create_feature_vector()
        features_df = pd.DataFrame([features])
        targets_df = pd.DataFrame([targets])
        
        return features_df, targets_df

def process_multiple_files(file_paths: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Process multiple YAML files and combine into features and targets datasets"""
    all_features = []
    all_targets = []
    
    for file_path in file_paths:
        try:
            parser = CottonCandyLogParser(file_path)
            features_df, targets_df = parser.process_file()
            # Add file identifier
            features_df['source_file'] = file_path
            targets_df['source_file'] = file_path
            all_features.append(features_df)
            all_targets.append(targets_df)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    if all_features and all_targets:
        combined_features = pd.concat(all_features, ignore_index=True)
        combined_targets = pd.concat(all_targets, ignore_index=True)
        return combined_features, combined_targets
    else:
        return pd.DataFrame(), pd.DataFrame()

# Example usage
if __name__ == "__main__":
    # Process the YAML file from Test Batch 0 (has stream format for environment data)
    parser = CottonCandyLogParser('Batches/test-batch-0/63693789-3e88-4a7e-91f0-2940bbe3aec0.xes.yaml')
    features_df, targets_df = parser.process_file()
    
    print("Feature columns (X):", features_df.columns.tolist())
    print("Target columns (y):", targets_df.columns.tolist())
    print("Feature vector shape:", features_df.shape)
    print("Target vector shape:", targets_df.shape)
    print("\nSample features (X):")
    print(features_df.head())
    print("\nSample targets (y):")
    print(targets_df.head())
    
    # Save to CSV
    features_df.to_csv("features_X.csv", index=False)
    targets_df.to_csv("targets_y.csv", index=False)
    print("\nFeatures saved to features_X.csv")
    print("Targets saved to targets_y.csv")
