"""
Cotton Candy Feature and Target Extractor

This script extracts features and targets from the cotton candy dataset and creates:
1. features_X.csv - Contains iteration + 29 features (30 columns total) from feature_documentation.md
2. target_y.csv - Contains iteration + cc_weight as the target variable

Usage:
    python feature_target_extractor.py --all
    python feature_target_extractor.py --range 0 10
    python feature_target_extractor.py --range 5 15
"""

import pandas as pd
import argparse
import os
from pathlib import Path

# Define the exact 29 features from feature_documentation.md
FEATURES_29 = [
    # Core Process Parameters (4 features)
    'iteration_since_maintenance',
    'wait_time',
    'cook_time',
    'cooldown_time',
    
    # Timing Metrics (3 features)
    'duration_till_handover',
    'duration_total',
    'duration_cc_flow',
    
    # Environmental Baseline (2 features)
    'baseline_env_EnvH',
    'baseline_env_EnvT',
    
    # Internal Environmental Sensors - Phase 1: Before Turn On (4 features)
    'before_turn_on_env_InH',
    'before_turn_on_env_InT',
    'before_turn_on_env_IrO',
    'before_turn_on_env_IrA',
    
    # Internal Environmental Sensors - Phase 2: After Flow Start (4 features)
    'after_flow_start_env_InH',
    'after_flow_start_env_InT',
    'after_flow_start_env_IrO',
    'after_flow_start_env_IrA',
    
    # Internal Environmental Sensors - Phase 3: After Flow End (4 features)
    'after_flow_end_env_InH',
    'after_flow_end_env_InT',
    'after_flow_end_env_IrO',
    'after_flow_end_env_IrA',
    
    # Internal Environmental Sensors - Phase 4: Before Cooldown (4 features)
    'before_cooldown_env_InH',
    'before_cooldown_env_InT',
    'before_cooldown_env_IrO',
    'before_cooldown_env_IrA',
    
    # Internal Environmental Sensors - Phase 5: After Cooldown (4 features)
    'after_cooldown_env_InH',
    'after_cooldown_env_InT',
    'after_cooldown_env_IrO',
    'after_cooldown_env_IrA'
]

# Target variable
TARGET_VARIABLE = 'calculated_score'

def load_dataset(file_path):
    """Load the cotton candy dataset from CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    print(f"Loading dataset from: {file_path}")
    df = pd.read_csv(file_path)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def validate_features(df):
    """Validate that all 29 features exist in the dataset."""
    missing_features = []
    for feature in FEATURES_29:
        if feature not in df.columns:
            missing_features.append(feature)
    
    if missing_features:
        print(f"❌ Missing features in dataset: {missing_features}")
        print(f"Available columns: {list(df.columns)}")
        raise ValueError(f"Dataset is missing {len(missing_features)} required features")
    
    print(f"✅ All 29 features found in dataset")
    return True

def validate_target(df):
    """Validate that target variable exists in the dataset."""
    if TARGET_VARIABLE not in df.columns:
        print(f"❌ Target variable '{TARGET_VARIABLE}' not found in dataset")
        print(f"Available columns: {list(df.columns)}")
        raise ValueError(f"Dataset is missing target variable: {TARGET_VARIABLE}")
    
    print(f"✅ Target variable '{TARGET_VARIABLE}' found in dataset")
    return True

def extract_features_and_target(df, start_iteration=None, end_iteration=None):
    """Extract features and target from the dataset with optional iteration range."""
    
    # Filter by iteration range if specified
    if start_iteration is not None and end_iteration is not None:
        print(f"Filtering data: iterations {start_iteration} to {end_iteration}")
        df_filtered = df[(df['iteration'] >= start_iteration) & (df['iteration'] <= end_iteration)].copy()
        print(f"Filtered dataset: {df_filtered.shape[0]} rows")
    else:
        print("Processing all iterations")
        df_filtered = df.copy()
    
    if df_filtered.empty:
        raise ValueError(f"No data found for iteration range {start_iteration}-{end_iteration}")
    
    # Extract features (X) - include iteration column + 29 features for 30 total columns
    features_columns = ['iteration'] + FEATURES_29
    features_X = df_filtered[features_columns].copy()
    
    # Extract target (y)
    target_y = df_filtered[['iteration', TARGET_VARIABLE]].copy()
    
    print(f"Features extracted: {features_X.shape[0]} rows, {features_X.shape[1]} columns (including iteration)")
    print(f"Target extracted: {target_y.shape[0]} rows, {target_y.shape[1]} columns")
    
    return features_X, target_y

def save_features_and_target(features_X, target_y, output_dir="Data_Collection"):
    """Save features and target to CSV files."""
    
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save features
    features_file = output_path / "features_X.csv"
    features_X.to_csv(features_file, index=False, float_format='%.2f')
    print(f"✅ Features saved to: {features_file}")
    
    # Save target
    target_file = output_path / "target_y.csv"
    target_y.to_csv(target_file, index=False, float_format='%.2f')
    print(f"✅ Target saved to: {target_file}")
    
    # Display summary statistics
    print("\n📊 FEATURES SUMMARY:")
    print(f"   • Total columns: {features_X.shape[1]} (iteration + {len(FEATURES_29)} features)")
    print(f"   • Rows processed: {features_X.shape[0]}")
    print(f"   • Missing values per feature:")
    # Check missing values excluding iteration column
    feature_cols = [col for col in features_X.columns if col != 'iteration']
    missing_counts = features_X[feature_cols].isnull().sum()
    if missing_counts.sum() == 0:
        print("     No missing values ✅")
    else:
        for feature, count in missing_counts[missing_counts > 0].items():
            print(f"     • {feature}: {count} missing values")
    
    print(f"\n🎯 TARGET SUMMARY:")
    print(f"   • Target variable: {TARGET_VARIABLE}")
    print(f"   • Rows with target: {target_y.shape[0]}")
    print(f"   • Missing target values: {target_y[TARGET_VARIABLE].isnull().sum()}")
    print(f"   • Target range: {target_y[TARGET_VARIABLE].min():.2f} - {target_y[TARGET_VARIABLE].max():.2f}")
    
    return features_file, target_file

def main():
    parser = argparse.ArgumentParser(description='Extract Cotton Candy features and target variables')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--all', action='store_true', 
                      help='Process all iterations in the dataset')
    group.add_argument('--range', nargs=2, type=int, metavar=('START', 'END'),
                      help='Process specific iteration range (inclusive)')
    
    parser.add_argument('--input', default='Data_Collection/my_cc_dataset.csv',
                       help='Input dataset file path (default: Data_Collection/my_cc_dataset_backup_optimized.csv)')
    parser.add_argument('--output-dir', default='Data_Collection',
                       help='Output directory for generated files (default: Data_Collection)')
    
    args = parser.parse_args()
    
    try:
        # Load dataset
        df = load_dataset(args.input)
        
        # Validate features and target
        validate_features(df)
        validate_target(df)
        
        # Extract features and target
        if args.all:
            features_X, target_y = extract_features_and_target(df)
        else:
            start_iter, end_iter = args.range
            features_X, target_y = extract_features_and_target(df, start_iter, end_iter)
        
        # Save results
        features_file, target_file = save_features_and_target(features_X, target_y, args.output_dir)
        
        print(f"\n🎉 EXTRACTION COMPLETE!")
        print(f"   Features: {features_file}")
        print(f"   Target:   {target_file}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
