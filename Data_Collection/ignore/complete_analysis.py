#!/usr/bin/env python3
"""
Complete Cotton Candy Analysis Pipeline
This script demonstrates how to:
1. Extract clean features for decision tree input
2. Calculate quality metrics and energy consumption
3. Train decision trees for optimization
"""

import pandas as pd
import numpy as np
from optimized_log_parser import CottonCandyLogParser, process_multiple_files
import glob
import matplotlib.pyplot as plt

def extract_features_and_targets(yaml_files):
    """Extract both features and target metrics from log files"""
    all_data = []
    
    for file_path in yaml_files:
        try:
            parser = CottonCandyLogParser(file_path)
            parser.parse_yaml_efficiently()
            
            # Get features for decision tree
            feature_result = parser.create_feature_vector()
            if isinstance(feature_result, tuple):
                # If it returns a tuple, take the first element (features)
                features = feature_result[0]
            else:
                # If it returns just the features dictionary
                features = feature_result
            
            # Get quality metrics for targets
            quality_metrics = parser.calculate_quality_metrics()
            
            # Combine features and targets
            row_data = {**features, **quality_metrics}
            # Extract just the filename from the full path
            row_data['source_file'] = file_path.split('/')[-1]
            
            all_data.append(row_data)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    if all_data:
        return pd.DataFrame(all_data)
    else:
        return pd.DataFrame()

def create_optimization_targets(df):
    """Create optimization targets for decision tree modeling"""
    if df.empty:
        return df
    
    # Energy efficiency: Lower energy per unit weight is better
    if 'total_energy_wh' in df.columns and 'final_weight' in df.columns:
        df['energy_efficiency'] = df['final_weight'] / (df['total_energy_wh'] + 0.001)  # Weight per Wh
    
    # Overall quality score (you can adjust these weights)
    quality_components = []
    weights = []
    
    if 'weight_consistency' in df.columns:
        quality_components.append(df['weight_consistency'])
        weights.append(0.3)  # 30% weight
    
    if 'pressure_stability' in df.columns:
        quality_components.append(df['pressure_stability'])
        weights.append(0.3)  # 30% weight
    
    if 'size_consistency' in df.columns:
        quality_components.append(df['size_consistency'])
        weights.append(0.2)  # 20% weight
    
    if 'energy_efficiency' in df.columns:
        # Normalize energy efficiency for scoring
        energy_norm = (df['energy_efficiency'] - df['energy_efficiency'].min()) / (df['energy_efficiency'].max() - df['energy_efficiency'].min())
        quality_components.append(energy_norm)
        weights.append(0.2)  # 20% weight
    
    if quality_components:
        # Calculate weighted quality score
        quality_matrix = np.column_stack(quality_components)
        weights_array = np.array(weights) / np.sum(weights)  # Normalize weights
        df['overall_quality_score'] = np.dot(quality_matrix, weights_array)
        
        # Create binary quality classification
        quality_threshold = df['overall_quality_score'].median()
        df['high_quality'] = (df['overall_quality_score'] > quality_threshold).astype(int)
    
    # Energy efficiency classification
    if 'total_energy_wh' in df.columns:
        energy_threshold = df['total_energy_wh'].median()
        df['low_energy'] = (df['total_energy_wh'] < energy_threshold).astype(int)
    
    return df

def format_numeric_columns(df):
    """Format all numeric columns to exactly 2 decimal places"""
    if df.empty:
        return df
    
    # Get numeric columns (excluding source_file)
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    # Round all numeric columns to 2 decimal places
    for col in numeric_columns:
        df[col] = df[col].round(2)
    
    return df

def analyze_parameter_impacts(df):
    """Analyze which parameters most impact energy and quality"""
    if df.empty:
        return
    
    # Define feature columns (input parameters)
    feature_cols = ['radius', 'height', 'sugar_amount', 'wait_time', 'cook_time', 'cooldown_time']
    feature_cols += [col for col in df.columns if col.startswith('start_env_')]
    
    # Define target columns (what we want to optimize)
    target_cols = ['total_energy_wh', 'overall_quality_score', 'final_weight']
    target_cols = [col for col in target_cols if col in df.columns]
    
    print("Parameter Impact Analysis:")
    print("=" * 50)
    
    for target in target_cols:
        print(f"\nCorrelations with {target}:")
        correlations = []
        for feature in feature_cols:
            if feature in df.columns:
                corr = df[feature].corr(df[target])
                if not np.isnan(corr):
                    correlations.append((feature, abs(corr), corr))
        
        # Sort by absolute correlation
        correlations.sort(key=lambda x: x[1], reverse=True)
        
        for feature, abs_corr, corr in correlations[:5]:  # Top 5
            direction = "↑" if corr > 0 else "↓"
            print(f"  {feature:20}: {corr:6.3f} {direction}")

def main():
    import sys
    import os
    
    # Check command line arguments
    if len(sys.argv) == 1:
        # No arguments - process all batches (excluding test batches)
        print("Processing all available process files (excluding test batches)...")
        yaml_files = []
        
        # Get all batch directories and sort them numerically
        batch_dirs = []
        for item in sorted(os.listdir("Batches")):
            if item.startswith('batch-') and os.path.isdir(os.path.join("Batches", item)):
                # Extract batch number for sorting
                try:
                    batch_num = int(item.split('-')[1])
                    batch_dirs.append((batch_num, item))
                except (ValueError, IndexError):
                    continue
        
        # Sort by batch number
        batch_dirs.sort(key=lambda x: x[0])
        
        # Collect YAML files in order
        for batch_num, batch_dir in batch_dirs:
            batch_files = glob.glob(f"Batches/{batch_dir}/*-process.yaml")
            # Sort process files within each batch by stick number
            batch_files.sort(key=lambda x: (
                int(x.split('/')[-1].split('-')[0]),  # batch number
                int(x.split('/')[-1].split('-')[1])   # stick number
            ))
            yaml_files.extend(batch_files)
        
        output_prefix = "all_batches"
    elif len(sys.argv) == 2:
        # Batch number specified
        try:
            batch_number = int(sys.argv[1])
            batch_dir = f"Batches/batch-{batch_number}"
            
            if not os.path.exists(batch_dir):
                print(f"Error: batch-{batch_number} directory not found!")
                print("Available batches:")
                for item in sorted(os.listdir("Batches")):
                    if item.startswith('batch-') and os.path.isdir(os.path.join("Batches", item)):
                        print(f"  {item}")
                sys.exit(1)
            
            print(f"Processing batch-{batch_number}...")
            yaml_files = glob.glob(f"{batch_dir}/*-process.yaml")
            output_prefix = f"batch_{batch_number}"
            
        except ValueError:
            print("Error: Batch number must be an integer")
            print("Usage: python complete_analysis.py [batch_number]")
            print("Example: python complete_analysis.py 7")
            sys.exit(1)
    else:
        print("Usage: python complete_analysis.py [batch_number]")
        print("Examples:")
        print("  python complete_analysis.py      # Process all batches")
        print("  python complete_analysis.py 7    # Process only batch-7")
        sys.exit(1)
    
    if not yaml_files:
        print(f"No process YAML files found!")
        if len(sys.argv) == 2:
            print(f"Make sure you've run the batch processor on batch-{sys.argv[1]} first:")
            print(f"  python batch_cleaner.py {sys.argv[1]}")
        sys.exit(1)
    
    print(f"Found {len(yaml_files)} process files to analyze...")
    
    # Extract features and quality metrics
    dataset = extract_features_and_targets(yaml_files)
    
    if dataset.empty:
        print("No data extracted.")
        return
    
    print(f"Extracted data from {len(dataset)} runs")
    
    # Create optimization targets
    dataset = create_optimization_targets(dataset)
    
    # Format all numeric columns to 2 decimal places
    dataset = format_numeric_columns(dataset)
    
    # Save comprehensive dataset with batch-specific naming
    output_filename = f"cotton_candy_{output_prefix}_dataset.csv"
    dataset.to_csv(output_filename, index=False)
    print(f"Complete dataset saved to: {output_filename}")
    
    # Display summary
    print(f"\nDataset Summary:")
    print(f"- Runs: {len(dataset)}")
    print(f"- Features: {len([col for col in dataset.columns if not col.startswith('source_file')])}")
    
    # Key metrics summary
    if 'total_energy_wh' in dataset.columns:
        print(f"- Energy consumption: {dataset['total_energy_wh'].mean():.2f} ± {dataset['total_energy_wh'].std():.2f} Wh")
        print(f"- Energy consumption: {dataset['total_energy_joules'].mean():.0f} ± {dataset['total_energy_joules'].std():.0f} Joules")
    
    if 'overall_quality_score' in dataset.columns:
        print(f"- Quality score: {dataset['overall_quality_score'].mean():.3f} ± {dataset['overall_quality_score'].std():.3f}")
    
    if 'final_weight' in dataset.columns:
        print(f"- Final weight: {dataset['final_weight'].mean():.2f} ± {dataset['final_weight'].std():.2f} g")
    
    # Parameter impact analysis
    if len(dataset) > 1:
        analyze_parameter_impacts(dataset)
    
    print(f"\n{'='*60}")
    print("Ready for Decision Tree Training!")
    print("Use the complete dataset for:")
    print("1. Predicting optimal parameters for energy efficiency")
    print("2. Predicting quality outcomes")
    print("3. Multi-objective optimization")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
