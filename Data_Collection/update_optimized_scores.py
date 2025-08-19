#!/usr/bin/env python3
"""
🔄 Update Calculated Scores with OPTIMIZED Max_Pos Ranges
========================================================
Update all calculated scores using the optimized cc_quality_score.py function
with the data-driven max_pos sweet spots.
"""

import pandas as pd
from cc_quality_score import calculate_cotton_candy_quality

def update_all_scores():
    """Update calculated scores with optimized function."""
    
    # Load dataset
    df = pd.read_csv('my_cc_dataset.csv')
    print(f"🔄 UPDATING WITH OPTIMIZED MAX_POS RANGES")
    print(f"Dataset: {len(df)} rows")
    print()
    
    # Backup current dataset
    df.to_csv('my_cc_dataset_backup_optimized.csv', index=False)
    print("✅ Backup saved as 'my_cc_dataset_backup_optimized.csv'")
    
    # Calculate new scores for all rows
    new_scores = []
    
    for idx, row in df.iterrows():
        # Use optimized function
        score = calculate_cotton_candy_quality(
            row['touch_pos1'], row['touch_pos2'], row['touch_pos3'],
            row['max_pos1'], row['max_pos2'], row['max_pos3'], 
            row['cc_weight']
        )
        new_scores.append(round(score, 1))
    
    # Update calculated_score column
    df['calculated_score'] = new_scores
    
    # Save updated dataset
    df.to_csv('my_cc_dataset.csv', index=False)
    print("✅ Dataset updated with optimized scores")
    
    # Quick analysis
    analysis_df = df[(df['iteration'] >= 14) & df['my_score'].notna()].copy()
    
    if len(analysis_df) > 0:
        # Calculate performance metrics
        analysis_df['score_diff'] = analysis_df['calculated_score'] - analysis_df['my_score']
        analysis_df['abs_diff'] = abs(analysis_df['score_diff'])
        
        mae = analysis_df['abs_diff'].mean()
        corr = analysis_df['my_score'].corr(analysis_df['calculated_score'])
        mean_diff = analysis_df['score_diff'].mean()
        
        print(f"\n📊 OPTIMIZATION RESULTS (Iteration 14+):")
        print(f"Mean Absolute Error: {mae:.2f} points (was ~13.4)")
        print(f"Correlation: {corr:.3f} (target: >0.6)")  
        print(f"Systematic bias: {mean_diff:+.2f} points")
        print()
        
        # Check key problem cases
        problem_rows = [20, 25, 31]  # Previously failed samples
        print("🔍 CHECKING PREVIOUSLY FAILED SAMPLES:")
        for row_idx in problem_rows:
            if row_idx < len(df):
                row = df.iloc[row_idx]
                if pd.notna(row['my_score']):
                    print(f"Row {row_idx}: Your={row['my_score']:.0f}, New Calc={row['calculated_score']:.1f}, "
                          f"Weight={row['cc_weight']:.2f}g")
        
        print()
        
        # Show some examples of optimized scoring
        print("📋 SAMPLE OPTIMIZED SCORES:")
        print("Row | Your | Old→New | Max_Pos (1/2/3)    | Notes")
        print("-" * 60)
        
        # Show a few key examples
        key_samples = [14, 17, 21, 28, 29, 45, 46]
        for row_idx in key_samples:
            if row_idx < len(df):
                row = df.iloc[row_idx] 
                if pd.notna(row['my_score']):
                    m1 = row['max_pos1'] if pd.notna(row['max_pos1']) else 0
                    m2 = row['max_pos2'] if pd.notna(row['max_pos2']) else 0
                    m3 = row['max_pos3'] if pd.notna(row['max_pos3']) else 0
                    
                    # Determine if this fits our sweet spots
                    in_sweet_spot = (13 <= m1 <= 25) and (0 <= m2 <= 5) and (18 <= m3 <= 35)
                    note = "✅ Sweet Spot" if in_sweet_spot else "⚠️ Outside Range"
                    
                    print(f"{row_idx:3d} | {row['my_score']:4.0f} | →{row['calculated_score']:4.1f} | "
                          f"{m1:4.1f}/{m2:4.1f}/{m3:4.1f} | {note}")

if __name__ == "__main__":
    update_all_scores()
    print(f"\n🎯 OPTIMIZATION COMPLETE!")
    print(f"   • Applied data-driven max_pos ranges:")
    print(f"     - max_pos1: 13-25 (sweet spot from best samples)")  
    print(f"     - max_pos2: 0-5 (critical - keep it LOW!)")
    print(f"     - max_pos3: 18-35 (strong quality correlation)")
    print(f"   • All scores updated and saved!")
