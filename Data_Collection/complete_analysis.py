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
            features = parser.create_feature_vector()
            
            # Get quality metrics for targets
            quality_metrics = parser.calculate_quality_metrics()
            
            # Combine features and targets
            row_data = {**features, **quality_metrics}
            row_data['source_file'] = file_path
            
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
    # Find YAML files
    yaml_files = glob.glob("*.yaml") + glob.glob("*.xes.yaml")
    
    if not yaml_files:
        print("No YAML files found. Using single file for demonstration.")
        yaml_files = ['4651e371-2b19-42ba-af9c-90ea170ce564.xes.yaml']
    
    print(f"Processing {len(yaml_files)} files...")
    
    # Extract features and quality metrics
    dataset = extract_features_and_targets(yaml_files)
    
    if dataset.empty:
        print("No data extracted.")
        return
    
    print(f"Extracted data from {len(dataset)} runs")
    
    # Create optimization targets
    dataset = create_optimization_targets(dataset)
    
    # Save comprehensive dataset
    dataset.to_csv("cotton_candy_complete_dataset.csv", index=False)
    print(f"Complete dataset saved to: cotton_candy_complete_dataset.csv")
    
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
