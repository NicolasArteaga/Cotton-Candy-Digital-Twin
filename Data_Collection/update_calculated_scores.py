#!/usr/bin/env python3
"""
🍭 Cotton Candy Dataset Score Updater
====================================
Automatically calculates and updates the calculated_score column in my_cc_dataset.csv
using the calibrated quality function from cotton_candy_quality_final.py
"""

import pandas as pd
import numpy as np
from Data_Collection.cc_quality_score import calculate_cotton_candy_quality

def update_dataset_scores():
    """Update all calculated_score values in the dataset"""
    
    # Load the dataset
    df = pd.read_csv('my_cc_dataset.csv')
    print(f"📊 Loaded dataset with {len(df)} rows")
    
    # Initialize counters
    updated_count = 0
    skipped_count = 0
    
    # Update calculated scores for each row
    for index, row in df.iterrows():
        try:
            # Extract parameters for quality function
            touch_pos1 = row.get('touch_pos1', np.nan)
            touch_pos2 = row.get('touch_pos2', np.nan) 
            touch_pos3 = row.get('touch_pos3', np.nan)
            max_pos1 = row.get('max_pos1', np.nan)
            max_pos2 = row.get('max_pos2', np.nan)
            max_pos3 = row.get('max_pos3', np.nan)
            cc_weight = row.get('cc_weight', np.nan)
            
            # Calculate quality score using calibrated function
            calculated_score = calculate_cotton_candy_quality(
                touch_pos1, touch_pos2, touch_pos3,
                max_pos1, max_pos2, max_pos3, cc_weight
            )
            
            # Update the score in dataframe
            df.at[index, 'calculated_score'] = round(calculated_score, 1)
            updated_count += 1
            
            # Show progress for important updates
            if index >= 14:  # Rows 15+ have complete data
                my_score = row.get('my_score', 'N/A')
                print(f"Row {index}: {calculated_score:.1f} (subjective: {my_score})")
                
        except Exception as e:
            print(f"⚠️ Error processing row {index}: {e}")
            df.at[index, 'calculated_score'] = 0.0
            skipped_count += 1
    
    # Save updated dataset
    df.to_csv('my_cc_dataset.csv', index=False)
    
    # Summary
    print("\n🎯 UPDATE COMPLETE!")
    print(f"✅ Updated: {updated_count} rows")
    if skipped_count > 0:
        print(f"⚠️ Errors: {skipped_count} rows")
    print(f"💾 Saved to: my_cc_dataset.csv")
    
    # Show validation for key rows
    print("\n🔍 VALIDATION - Key corrected rows:")
    validation_rows = [19, 20, 24, 25, 37, 50]  # Previously problematic rows
    for row_idx in validation_rows:
        if row_idx < len(df):
            row = df.iloc[row_idx]
            calc_score = row['calculated_score']
            my_score = row.get('my_score', 'N/A')
            weight = row.get('cc_weight', 'N/A')
            print(f"Row {row_idx}: calculated={calc_score}, subjective={my_score}, weight={weight}g")

if __name__ == "__main__":
    print("🍭 COTTON CANDY SCORE UPDATER")
    print("=" * 35)
    print("Using calibrated quality function to update all calculated_score values...")
    print()
    
    update_dataset_scores()
    
    print()
    print("🎊 All calculated scores have been updated with the latest calibrated function!")
    print("✨ Your dataset is now ready with accurate quality assessments!")
