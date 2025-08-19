#!/usr/bin/env python3
"""
Batch processing script for cotton candy log files
Run this to process all YAML files in the Data_Collection directory
"""

import glob
import os
import pandas as pd
from optimized_log_parser import process_multiple_files

def main():
    # Find all YAML files in the current directory
    yaml_files = glob.glob("*.yaml") + glob.glob("*.xes.yaml")
    
    if not yaml_files:
        print("No YAML files found in the current directory.")
        print("Make sure you're in the Data_Collection directory with YAML log files.")
        return
    
    print(f"Found {len(yaml_files)} YAML files:")
    for f in yaml_files:
        print(f"  - {f}")
    
    print("\nProcessing files...")
    
    # Process all files
    dataset = process_multiple_files(yaml_files)
    
    if dataset.empty:
        print("No features extracted from any files.")
        return
    
    print(f"\nProcessed {len(dataset)} successful runs")
    print(f"Extracted {len(dataset.columns)} features")
    
    # Save combined dataset
    output_file = "combined_cotton_candy_features.csv"
    dataset.to_csv(output_file, index=False)
    print(f"\nCombined dataset saved to: {output_file}")
    
    # Show feature summary
    print(f"\nFeature Summary:")
    print(f"- Shape: {dataset.shape}")
    print(f"- Missing values per column:")
    missing_summary = dataset.isnull().sum()
    missing_summary = missing_summary[missing_summary > 0].sort_values(ascending=False)
    if len(missing_summary) > 0:
        print(missing_summary.head(10))
    else:
        print("  No missing values!")
    
    # Show sample features
    print(f"\nSample of extracted features:")
    feature_cols = [col for col in dataset.columns if col != 'source_file']
    print(dataset[feature_cols[:10]].head())
    
    print(f"\n{'='*60}")
    print("Next steps:")
    print("1. Add target variables (quality labels, success metrics)")
    print("2. Use improved_decision_tree.py to train models")
    print("3. Analyze feature importance to optimize your process")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
