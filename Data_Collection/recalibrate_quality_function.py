#!/usr/bin/env python3
"""
🍭 Cotton Candy Quality Function - Calibrated Version 2.1
========================================================
Fixed calibration with proper touch_pos2 max value and better weight penalty
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def calibrated_quality_function(touch_pos1, touch_pos2, touch_pos3, 
                               max_pos1, max_pos2, max_pos3, cc_weight):
    """
    Calibrated quality function with proper touch_pos2 scaling and weight penalties.
    
    Key fixes:
    - touch_pos2 max value = 6 (not 11)
    - Harsher penalties for very low weights
    - Better calibration to match subjective scores
    """
    
    # 1. TOUCH POSITION SCORING (40% weight - increased importance)
    def touch_score_calibrated(pos, max_val):
        if pd.isna(pos) or pos is None:
            return 20  # Lower conservative estimate
        
        # Normalized score: max_val = 0 points, 0 = 100 points
        # More aggressive penalty curve
        normalized = min(pos / max_val, 1.0)
        score = 100 * (1 - normalized) ** 0.8  # More aggressive curve
        return max(0, score)
    
    # Individual touch scores with correct max values
    touch1_score = touch_score_calibrated(touch_pos1, 11.0)  # max = 11
    touch2_score = touch_score_calibrated(touch_pos2, 6.0)   # max = 6 (corrected!)
    touch3_score = touch_score_calibrated(touch_pos3, 11.0)  # max = 11
    
    avg_touch_score = np.mean([touch1_score, touch2_score, touch3_score])
    
    # 2. MAX POSITION SCORING (20% weight - reduced importance)
    def max_pos_score(pos, target_low, target_high, penalty_factor):
        if pd.isna(pos) or pos is None:
            return 20  # Lower conservative
        
        if target_low <= pos <= target_high:
            return 60  # Reduced from 80
        elif pos < target_low:
            return max(0, 60 - (target_low - pos) * penalty_factor)
        else:
            return max(0, 60 - (pos - target_high) * penalty_factor)
    
    max1_score = max_pos_score(max_pos1, 8, 20, 2.0)   
    max2_score = max_pos_score(max_pos2, 3, 10, 2.5)   
    max3_score = max_pos_score(max_pos3, 10, 25, 1.5) 
    
    avg_max_score = np.mean([max1_score, max2_score, max3_score])
    
    # 3. WEIGHT SCORING (40% weight - maintained importance)
    def weight_score_calibrated(weight):
        if pd.isna(weight) or weight is None:
            return 20  # Lower conservative
        
        if weight <= 0:
            return 0  # Complete failure
        
        # Much harsher penalties for very low weights
        if weight < 1.0:  # Less than 1g is basically failure
            return min(10, weight * 8)  # Max 8 points for < 1g
        elif weight < 3.0:  # 1-3g is poor
            return weight * 12  # 12-36 points
        elif 5.0 <= weight <= 10.0:  # Sweet spot
            return 60  # Good range
        elif weight < 5.0:  # 3-5g is below optimal
            return 30 + (weight - 3) * 15  # 30-60 points
        else:  # Over 10g gets penalized
            return max(0, 60 - (weight - 10) * 4)
    
    weight_score_val = weight_score_calibrated(cc_weight)
    
    # 4. COMBINE SCORES
    # Touch=40%, Weight=40%, Max=20%
    final_score = (
        avg_touch_score * 0.40 + 
        weight_score_val * 0.40 +
        avg_max_score * 0.20
    )
    
    # 5. CALIBRATION - more aggressive scaling
    calibrated_score = final_score * 0.5  # Reduced from 0.7
    
    return min(100, max(0, calibrated_score))

def recalibrate_dataset():
    """Recalibrate the dataset with the improved function."""
    
    # Load dataset
    df = pd.read_csv('my_cc_dataset.csv')
    print(f"🔧 RECALIBRATING QUALITY FUNCTION")
    print(f"Loading dataset with {len(df)} rows...")
    
    # Calculate new scores
    new_scores = []
    for idx, row in df.iterrows():
        score = calibrated_quality_function(
            row['touch_pos1'], row['touch_pos2'], row['touch_pos3'],
            row['max_pos1'], row['max_pos2'], row['max_pos3'], 
            row['cc_weight']
        )
        new_scores.append(round(score, 1))
    
    # Update the calculated_score column
    df['calculated_score'] = new_scores
    
    # Backup and save
    df.to_csv('my_cc_dataset_backup_v2.csv', index=False)
    df.to_csv('my_cc_dataset.csv', index=False)
    print(f"✅ Updated dataset saved (backup: my_cc_dataset_backup_v2.csv)")
    
    # Analysis
    comparison_df = df.dropna(subset=['my_score'])
    
    if len(comparison_df) > 0:
        mae = np.mean(np.abs(comparison_df['calculated_score'] - comparison_df['my_score']))
        corr = comparison_df['my_score'].corr(comparison_df['calculated_score'])
        
        print(f"\n📊 CALIBRATION RESULTS:")
        print(f"Mean Absolute Error: {mae:.2f} points")
        print(f"Correlation: {corr:.3f}")
        print(f"Subjective range: {comparison_df['my_score'].min():.1f} - {comparison_df['my_score'].max():.1f}")
        print(f"Calculated range: {comparison_df['calculated_score'].min():.1f} - {comparison_df['calculated_score'].max():.1f}")
        
        # Check specific problem rows
        print(f"\n🔍 CHECKING PROBLEM ROWS:")
        problem_rows = [20, 25]  # 0-indexed: rows 21, 26
        
        for row_num in problem_rows:
            if row_num < len(df):
                row = df.iloc[row_num]
                print(f"Row {row_num}: touch=({row['touch_pos1']}, {row['touch_pos2']}, {row['touch_pos3']}) "
                      f"weight={row['cc_weight']:.2f}g → my_score={row['my_score']:.0f}, "
                      f"calculated={row['calculated_score']:.1f}")
        
        # Show some examples of different score ranges
        print(f"\n📋 SAMPLE COMPARISONS:")
        print("Row | My | Calc | Touch1 | Touch2 | Touch3 | Weight | Description")
        print("-" * 70)
        
        # Sort by my_score for better overview
        sorted_df = comparison_df.sort_values('my_score')
        sample_rows = [0, 5, 10, 15, 20, 25, 30, -5, -1]  # Various positions
        
        for i in sample_rows:
            if abs(i) < len(sorted_df):
                row = sorted_df.iloc[i]
                desc = "Failed" if row['my_score'] == 0 else "Poor" if row['my_score'] <= 20 else "OK" if row['my_score'] <= 40 else "Good"
                weight = row['cc_weight'] if pd.notna(row['cc_weight']) else 0
                print(f"{row.name:3d} | {row['my_score']:2.0f} | {row['calculated_score']:4.1f} | "
                      f"{row['touch_pos1']:6.1f} | {row['touch_pos2']:6.1f} | {row['touch_pos3']:6.1f} | "
                      f"{weight:6.2f} | {desc}")

if __name__ == "__main__":
    recalibrate_dataset()
    
    print(f"\n🎯 CALIBRATION COMPLETE!")
    print(f"   • Fixed touch_pos2 max value: 6 (was 11)")
    print(f"   • Harsher weight penalties for < 1g cotton candy")
    print(f"   • Adjusted scoring weights: Touch=40%, Weight=40%, Max=20%")
    print(f"   • More aggressive calibration scaling")
