#!/usr/bin/env python3
"""
Minimal Cotton Candy Feature Extractor
=====================================
Extracts specific features and targets from cotton candy dataset CSV files.
Allows row range selection and focuses on essential manufacturing parameters.
"""

import pandas as pd
import numpy as np
import argparse
import sys
from pathlib import Path

class MinimalFeatureExtractor:
    def __init__(self):
        """Initialize the minimal feature extractor."""
        # Define minimal essential features
        self.feature_columns = [
            'iteration_since_maintenance',
            'wait_time', 
            'cook_time', 
            'cooldown_time',
            'duration_cc_flow',
            'baseline_env_EnvH',
            'baseline_env_EnvT', 
            'before_turn_on_env_InH',
            'before_turn_on_env_InT',
            'before_turn_on_env_IrO',
            'before_turn_on_env_IrA'
        ]
        
        # Define target column
        self.target_column = 'my_score'
        
        # Always include iteration for reference
        self.id_column = 'iteration'
        
    def load_and_validate_csv(self, csv_path):
        """Load CSV and validate required columns exist."""
        print(f"🔍 Loading CSV file: {csv_path}")
        
        try:
            df = pd.read_csv(csv_path)
            print(f"   ✅ Loaded {len(df)} rows, {len(df.columns)} columns")
        except Exception as e:
            print(f"   ❌ Error loading CSV: {e}")
            return None
            
        # Check for required columns
        all_required = self.feature_columns + [self.target_column, self.id_column]
        missing_cols = [col for col in all_required if col not in df.columns]
        
        if missing_cols:
            print(f"   ❌ Missing required columns: {missing_cols}")
            print(f"   Available columns: {list(df.columns)}")
            return None
            
        print(f"   ✅ All required columns found")
        return df
        
    def filter_rows(self, df, start_row=None, end_row=None):
        """Filter rows based on iteration range."""
        print(f"\n📊 FILTERING ROWS...")
        
        # Show available iteration range
        iterations = df[self.id_column].dropna().sort_values()
        print(f"   Available iterations: {iterations.min():.0f} to {iterations.max():.0f}")
        print(f"   Total available rows: {len(df)}")
        
        # Apply row filtering if specified
        if start_row is not None or end_row is not None:
            if start_row is None:
                start_row = iterations.min()
            if end_row is None:
                end_row = iterations.max()
                
            print(f"   Filtering iterations {start_row} to {end_row}")
            
            # Filter by iteration range
            mask = (df[self.id_column] >= start_row) & (df[self.id_column] <= end_row)
            df_filtered = df[mask].copy()
            
            print(f"   ✅ Filtered to {len(df_filtered)} rows")
        else:
            df_filtered = df.copy()
            print(f"   ✅ Using all {len(df_filtered)} rows")
            
        return df_filtered
        
    def extract_features_and_targets(self, df):
        """Extract features (X) and targets (y) from filtered dataframe."""
        print(f"\n🎯 EXTRACTING FEATURES AND TARGETS...")
        
        # Extract features (X)
        X = df[self.feature_columns].copy()
        print(f"   Features (X): {len(self.feature_columns)} columns")
        for i, col in enumerate(self.feature_columns, 1):
            missing_count = X[col].isnull().sum()
            print(f"     {i:2d}. {col:<30} (missing: {missing_count})")
            
        # Extract targets (y) 
        y = df[self.target_column].copy()
        target_missing = y.isnull().sum()
        print(f"   Target (y): {self.target_column} (missing: {target_missing})")
        
        # Extract iteration IDs for reference
        iterations = df[self.id_column].copy()
        
        # Remove rows with missing targets
        if target_missing > 0:
            print(f"   🧹 Removing {target_missing} rows with missing targets")
            valid_mask = y.notna()
            X = X[valid_mask]
            y = y[valid_mask] 
            iterations = iterations[valid_mask]
            
        print(f"   ✅ Final dataset: {len(X)} samples with {len(self.feature_columns)} features")
        print(f"   Target range: {y.min():.1f} to {y.max():.1f}")
        print(f"   Target mean: {y.mean():.1f} ± {y.std():.1f}")
        
        return X, y, iterations
        
    def save_results(self, X, y, iterations, output_dir="Data_Collection"):
        """Save extracted features and targets to CSV files."""
        print(f"\n💾 SAVING RESULTS...")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Create features dataframe with iteration
        features_df = X.copy()
        features_df.insert(0, 'iteration', iterations.values)
        
        # Create targets dataframe 
        targets_df = pd.DataFrame({
            'iteration': iterations.values,
            self.target_column: y.values
        })
        
        # Save files
        features_file = output_path / "features_X.csv"
        targets_file = output_path / "targets_y.csv" 
        
        features_df.to_csv(features_file, index=False)
        targets_df.to_csv(targets_file, index=False)
        
        print(f"   ✅ Features saved to: {features_file}")
        print(f"   ✅ Targets saved to: {targets_file}")
        
        # Show summary
        print(f"\n📈 EXTRACTION SUMMARY:")
        print(f"   • Dataset: {len(X)} samples")
        print(f"   • Features: {len(self.feature_columns)} columns")
        print(f"   • Target: {self.target_column}")
        print(f"   • Iteration range: {iterations.min():.0f} to {iterations.max():.0f}")
        print(f"   • Missing values in features: {X.isnull().sum().sum()}")
        print(f"   • Files ready for machine learning!")
        
        return features_file, targets_file

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Extract minimal features from cotton candy dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 minimal_feature_extractor.py my_cc_dataset.csv
  python3 minimal_feature_extractor.py my_cc_dataset.csv --start 14 --end 52
  python3 minimal_feature_extractor.py data.csv --start 20 --output results/
        """
    )
    
    parser.add_argument('csv_file', help='Path to input CSV file')
    parser.add_argument('--start', type=int, help='Start iteration number (inclusive)')
    parser.add_argument('--end', type=int, help='End iteration number (inclusive)')
    parser.add_argument('--output', default='Data_Collection', help='Output directory (default: Data_Collection)')
    
    args = parser.parse_args()
    
    print("🍭 MINIMAL COTTON CANDY FEATURE EXTRACTOR")
    print("=" * 50)
    print(f"Input file: {args.csv_file}")
    if args.start is not None or args.end is not None:
        print(f"Row range: {args.start or 'start'} to {args.end or 'end'}")
    print(f"Output directory: {args.output}")
    print()
    
    # Check if input file exists
    if not Path(args.csv_file).exists():
        print(f"❌ Error: File '{args.csv_file}' not found!")
        return 1
        
    # Create extractor and process
    extractor = MinimalFeatureExtractor()
    
    # Load and validate CSV
    df = extractor.load_and_validate_csv(args.csv_file)
    if df is None:
        return 1
        
    # Filter rows
    df_filtered = extractor.filter_rows(df, args.start, args.end)
    
    # Extract features and targets
    X, y, iterations = extractor.extract_features_and_targets(df_filtered)
    
    if len(X) == 0:
        print("❌ No valid data found after filtering!")
        return 1
        
    # Save results
    features_file, targets_file = extractor.save_results(X, y, iterations, args.output)
    
    print(f"\n🎉 EXTRACTION COMPLETE!")
    print(f"Ready to train machine learning models with {len(X)} samples!")
    
    return 0

if __name__ == "__main__":
    exit(main())
