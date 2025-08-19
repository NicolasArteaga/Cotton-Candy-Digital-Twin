#!/usr/bin/env python3
"""
Smart Cotton Candy Feature Pipeline
This approach:
1. Directly parses original YAML files (preserving full context)
2. Caches intermediate results to avoid re-processing
3. Extracts comprehensive features with timing relationships
4. Provides feature selection for decision tree optimization
"""

import pandas as pd
import numpy as np
import yaml
import os
import pickle
import hashlib
from datetime import datetime
from pathlib import Path
import glob
from optimized_log_parser import CottonCandyLogParser

class SmartFeaturePipeline:
    def __init__(self, cache_dir="feature_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
    def get_file_hash(self, file_path):
        """Generate hash of file for cache validation"""
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def get_cache_path(self, file_path):
        """Generate cache file path for a given YAML file"""
        file_hash = self.get_file_hash(file_path)
        filename = Path(file_path).stem
        return self.cache_dir / f"{filename}_{file_hash}.pkl"
    
    def is_cached_valid(self, original_file, cache_file):
        """Check if cached features are still valid"""
        if not cache_file.exists():
            return False
        
        # Check if original file is newer than cache
        original_mtime = os.path.getmtime(original_file)
        cache_mtime = os.path.getmtime(cache_file)
        
        return cache_mtime > original_mtime
    
    def extract_comprehensive_features(self, yaml_file):
        """Extract full feature set directly from original YAML"""
        print(f"Processing original file: {yaml_file}")
        
        # Use your existing parser but enhance it for comprehensive extraction
        parser = CottonCandyLogParser(yaml_file)
        parser.parse_yaml_efficiently()
        
        # Get basic features
        basic_features = parser.create_feature_vector()
        if isinstance(basic_features, tuple):
            basic_features = basic_features[0]
        
        # Get quality metrics
        quality_metrics = parser.calculate_quality_metrics()
        
        # Extract timing relationships (this is the key advantage)
        timing_features = self.extract_timing_relationships(parser)
        
        # Extract process state transitions
        transition_features = self.extract_state_transitions(parser)
        
        # Extract environmental patterns
        env_features = self.extract_environmental_patterns(parser)
        
        # Combine all features
        comprehensive_features = {
            **basic_features,
            **quality_metrics,
            **timing_features,
            **transition_features,
            **env_features,
            'source_file': Path(yaml_file).name,
            'processing_timestamp': datetime.now().isoformat()
        }
        
        return comprehensive_features
    
    def extract_timing_relationships(self, parser):
        """Extract timing relationships that might be lost in cleaning"""
        timing_features = {}
        
        # Get the parsed events
        if hasattr(parser, 'events') and parser.events:
            events = parser.events
            
            # Time between key transitions
            start_events = [e for e in events if 'start' in str(e).lower()]
            end_events = [e for e in events if 'end' in str(e).lower()]
            
            if start_events and end_events:
                timing_features['total_process_duration'] = len(events)
                timing_features['start_to_end_ratio'] = len(start_events) / len(end_events) if end_events else 0
            
            # Detect process phases and their durations
            phases = self.detect_process_phases(events)
            for phase_name, duration in phases.items():
                timing_features[f'phase_duration_{phase_name}'] = duration
        
        return timing_features
    
    def extract_state_transitions(self, parser):
        """Extract state transition patterns"""
        transition_features = {}
        
        # This would analyze the sequence of state changes
        # Example: heating -> spinning -> cooling patterns
        if hasattr(parser, 'events'):
            # Count different types of transitions
            transition_features['total_transitions'] = len(parser.events) if parser.events else 0
            
            # Add more sophisticated transition analysis here
            # e.g., transition velocity, irregular patterns, etc.
        
        return transition_features
    
    def extract_environmental_patterns(self, parser):
        """Extract environmental sensor patterns and trends"""
        env_features = {}
        
        # This could include:
        # - Temperature/humidity stability over time
        # - Rate of change in environmental conditions
        # - Environmental variance during critical phases
        
        return env_features
    
    def detect_process_phases(self, events):
        """Detect and time different phases of the cotton candy process"""
        phases = {
            'preparation': 0,
            'heating': 0,
            'spinning': 0,
            'cooling': 0,
            'finishing': 0
        }
        
        # This is where you'd implement phase detection logic
        # based on your understanding of the process
        
        return phases
    
    def process_file(self, yaml_file):
        """Process a single YAML file with caching"""
        cache_path = self.get_cache_path(yaml_file)
        
        # Check if we have valid cached features
        if self.is_cached_valid(yaml_file, cache_path):
            print(f"Loading cached features for {Path(yaml_file).name}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        # Extract features from scratch
        features = self.extract_comprehensive_features(yaml_file)
        
        # Cache the results
        with open(cache_path, 'wb') as f:
            pickle.dump(features, f)
        
        return features
    
    def process_batch_directory(self, batch_dir):
        """Process all original YAML files in a batch directory"""
        # Look for original YAML files (not processed ones)
        yaml_pattern = os.path.join(batch_dir, "*.yaml")
        # Exclude already processed files
        yaml_files = [f for f in glob.glob(yaml_pattern) 
                     if not f.endswith('-process.yaml')]
        
        if not yaml_files:
            # Fallback to processed files if originals aren't available
            yaml_files = glob.glob(os.path.join(batch_dir, "*-process.yaml"))
        
        batch_features = []
        for yaml_file in yaml_files:
            try:
                features = self.process_file(yaml_file)
                batch_features.append(features)
            except Exception as e:
                print(f"Error processing {yaml_file}: {e}")
                continue
        
        return batch_features
    
    def create_full_dataset(self, batches_dir="Batches"):
        """Create comprehensive dataset from all batches"""
        all_features = []
        
        # Process each batch
        for batch_item in sorted(os.listdir(batches_dir)):
            if batch_item.startswith('batch-') and not batch_item.startswith('test-batch-'):
                batch_path = os.path.join(batches_dir, batch_item)
                if os.path.isdir(batch_path):
                    print(f"Processing {batch_item}...")
                    batch_features = self.process_batch_directory(batch_path)
                    all_features.extend(batch_features)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_features)
        
        # Format numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            df[col] = df[col].round(2)
        
        return df
    
    def select_decision_tree_features(self, df, feature_importance_threshold=0.01):
        """Select optimal features for decision tree training"""
        
        # Define feature categories
        core_process_features = [
            'wait_time', 'cook_time', 'cooldown_time', 
            'iteration_since_maintenance'
        ]
        
        timing_features = [col for col in df.columns 
                          if 'duration' in col or 'phase_' in col]
        
        environmental_features = [col for col in df.columns 
                                if 'env_' in col and not col.endswith('_EnvH')]
        
        quality_features = [col for col in df.columns 
                           if any(word in col for word in 
                                 ['consistency', 'stability', 'quality'])]
        
        # Start with core features
        selected_features = core_process_features.copy()
        
        # Add the most important timing features
        selected_features.extend(timing_features[:5])  # Top 5 timing features
        
        # Add environmental features that show variation
        env_features_with_variance = []
        for col in environmental_features:
            if col in df.columns and df[col].std() > 0.1:  # Has meaningful variance
                env_features_with_variance.append(col)
        selected_features.extend(env_features_with_variance[:8])  # Top 8 env features
        
        # Always include quality targets
        selected_features.extend([col for col in quality_features if col in df.columns])
        
        # Remove duplicates and ensure columns exist
        selected_features = list(set(selected_features))
        selected_features = [col for col in selected_features if col in df.columns]
        
        # Add source file for tracking
        if 'source_file' in df.columns:
            selected_features.append('source_file')
        
        return df[selected_features]


def main():
    import sys
    
    pipeline = SmartFeaturePipeline()
    
    if len(sys.argv) == 2 and sys.argv[1] == "--clear-cache":
        # Clear cache and rebuild
        import shutil
        if pipeline.cache_dir.exists():
            shutil.rmtree(pipeline.cache_dir)
        print("Cache cleared. Rebuilding features...")
    
    # Create comprehensive dataset
    print("Creating comprehensive feature dataset...")
    full_dataset = pipeline.create_full_dataset()
    
    # Save full dataset
    full_dataset.to_csv("cotton_candy_comprehensive_dataset.csv", index=False)
    print(f"Comprehensive dataset saved: {len(full_dataset)} runs, {len(full_dataset.columns)} features")
    
    # Create decision tree optimized dataset
    print("\nCreating decision tree optimized dataset...")
    dt_dataset = pipeline.select_decision_tree_features(full_dataset)
    dt_dataset.to_csv("cotton_candy_decision_tree_dataset.csv", index=False)
    print(f"Decision tree dataset saved: {len(dt_dataset)} runs, {len(dt_dataset.columns)} features")
    
    # Show feature breakdown
    print(f"\nFeature breakdown:")
    print(f"- Process parameters: {len([c for c in dt_dataset.columns if c in ['wait_time', 'cook_time', 'cooldown_time']])}")
    print(f"- Timing features: {len([c for c in dt_dataset.columns if 'duration' in c or 'phase_' in c])}")
    print(f"- Environmental: {len([c for c in dt_dataset.columns if 'env_' in c])}")
    print(f"- Quality metrics: {len([c for c in dt_dataset.columns if any(w in c for w in ['consistency', 'quality', 'stability'])])}")


if __name__ == "__main__":
    main()
