#!/usr/bin/env python3
"""
Batch Processor for Cotton Candy Digital Twin
Processes individual batches to extract feature vectors for Decision Tree training.
"""

import os
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
from optimized_log_parser import CottonCandyLogParser


class BatchProcessor:
    def __init__(self, batch_folder: str):
        """
        Initialize batch processor for a specific batch folder
        
        Args:
            batch_folder: Path to the batch folder containing YAML files and index.txt
        """
        self.batch_folder = Path(batch_folder)
        self.batch_name = self.batch_folder.name
        self.processes = []
        self.feature_vectors = []
        
    def parse_index_file(self) -> List[Dict]:
        """
        Parse the index.txt file to identify Cottonbot processes
        
        Returns:
            List of process information dictionaries
        """
        index_file = self.batch_folder / "index.txt"
        if not index_file.exists():
            print(f"Warning: No index.txt found in {self.batch_folder}")
            return []
        
        processes = []
        with open(index_file, 'r') as f:
            for line in f:
                line = line.strip()
                if "Cottonbot - Run with Data Collection" in line:
                    # Extract process ID from line like:
                    # "  Cottonbot - Run with Data Collection (8ad81af0-9da1-4e4a-b97f-1a72d35bc9bd) - 61055"
                    start_idx = line.find('(') + 1
                    end_idx = line.find(')')
                    if start_idx > 0 and end_idx > start_idx:
                        process_id = line[start_idx:end_idx]
                        # Extract the process number at the end
                        process_num = line.split(' - ')[-1]
                        processes.append({
                            'process_id': process_id,
                            'process_number': process_num,
                            'line': line.strip()
                        })
        
        print(f"Found {len(processes)} Cottonbot processes in {self.batch_name}")
        return processes
    
    def process_yaml_file(self, yaml_file: Path) -> Optional[Dict]:
        """
        Process a single YAML file and extract features using optimized_log_parser
        
        Args:
            yaml_file: Path to the YAML file
            
        Returns:
            Dictionary containing extracted features, or None if processing failed
        """
        try:
            parser = CottonCandyLogParser(str(yaml_file))
            parser.parse_yaml_efficiently()
            
            # Extract features
            features = {}
            
            # Process parameters (cook_time, wait_time, etc.)
            for event in parser.events:
                params = parser.extract_process_parameters(event)
                features.update(params)
            
            # Quality metrics (weight, pressure, size)
            quality_metrics = parser.calculate_quality_metrics()
            features.update(quality_metrics)
            
            # Energy consumption
            total_energy = parser.calculate_total_energy_consumption(unit='wh')
            features['total_energy_wh'] = total_energy
            
            # Environmental data (average across all measurements)
            env_data = []
            for event in parser.events:
                env = parser.extract_environmental_data(event)
                if env:
                    env_data.append(env)
            
            if env_data:
                # Calculate averages for environmental sensors
                env_df = pd.DataFrame(env_data)
                for col in env_df.columns:
                    if env_df[col].notna().any():
                        features[f'avg_{col}'] = env_df[col].mean()
                        features[f'std_{col}'] = env_df[col].std()
            
            # Add metadata
            features['source_file'] = yaml_file.name
            features['batch_name'] = self.batch_name
            features['process_id'] = yaml_file.stem.split('.')[0]  # Remove .xes.yaml
            
            return features
            
        except Exception as e:
            print(f"Error processing {yaml_file}: {e}")
            return None
    
    def process_batch(self) -> pd.DataFrame:
        """
        Process all Cottonbot processes in the batch
        
        Returns:
            DataFrame containing feature vectors for all processes in the batch
        """
        processes = self.parse_index_file()
        feature_list = []
        
        for process_info in processes:
            process_id = process_info['process_id']
            yaml_file = self.batch_folder / f"{process_id}.xes.yaml"
            
            if yaml_file.exists():
                print(f"Processing {yaml_file.name}...")
                features = self.process_yaml_file(yaml_file)
                if features:
                    features['process_number'] = process_info['process_number']
                    feature_list.append(features)
            else:
                print(f"Warning: YAML file not found for process {process_id}")
        
        if feature_list:
            df = pd.DataFrame(feature_list)
            print(f"Successfully processed {len(feature_list)} processes from {self.batch_name}")
            return df
        else:
            print(f"No valid processes found in {self.batch_name}")
            return pd.DataFrame()
    
    def save_batch_data(self, output_dir: str = None) -> str:
        """
        Process batch and save the results
        
        Args:
            output_dir: Directory to save the batch data (default: same as batch folder)
            
        Returns:
            Path to the saved CSV file
        """
        df = self.process_batch()
        
        if df.empty:
            print(f"No data to save for batch {self.batch_name}")
            return None
        
        # Set output directory
        if output_dir is None:
            output_dir = self.batch_folder.parent
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save batch data
        output_file = output_dir / f"{self.batch_name}_features.csv"
        df.to_csv(output_file, index=False)
        
        # Save summary
        summary = {
            'batch_name': self.batch_name,
            'num_processes': len(df),
            'feature_columns': list(df.columns),
            'file_path': str(output_file)
        }
        
        summary_file = output_dir / f"{self.batch_name}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Batch data saved to: {output_file}")
        print(f"Summary saved to: {summary_file}")
        
        return str(output_file)


def process_single_batch(batch_path: str, output_dir: str = None) -> str:
    """
    Convenience function to process a single batch
    
    Args:
        batch_path: Path to the batch folder
        output_dir: Directory to save results
        
    Returns:
        Path to the saved CSV file
    """
    processor = BatchProcessor(batch_path)
    return processor.save_batch_data(output_dir)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python batch_processor.py <batch_folder> [output_dir]")
        print("Example: python batch_processor.py Data_Collection/Batches/test-batch-0")
        sys.exit(1)
    
    batch_folder = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    result = process_single_batch(batch_folder, output_dir)
    if result:
        print(f"✅ Batch processing complete: {result}")
    else:
        print("❌ Batch processing failed")
