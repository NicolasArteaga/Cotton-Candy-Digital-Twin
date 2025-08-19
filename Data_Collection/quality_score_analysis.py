#!/usr/bin/env python3
"""
🔍 Cotton Candy Quality Score Analysis - Calculated vs Subjective
================================================================
Comprehensive analysis comparing my calculated quality scores with your 
subjective scores from iteration 14 onwards.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Note: Using existing calculated_score column from dataset
# Quality function is now in cc_quality_score.py

def analyze_quality_scores():
    """Analyze calculated vs subjective scores from iteration 14+"""
    
    # Load dataset
    df = pd.read_csv('my_cc_dataset.csv')
    
    # Filter for iteration 14+ (where we have complete touch/max data)
    analysis_df = df[df['iteration'] >= 14].copy()
    
    # Remove rows without calculated scores
    analysis_df = analysis_df.dropna(subset=['calculated_score', 'my_score'])
    
    print("🍭 COTTON CANDY QUALITY SCORE ANALYSIS")
    print("=" * 55)
    print(f"Analyzing {len(analysis_df)} samples from iteration 14+")
    print(f"Date range: Iterations {analysis_df['iteration'].min()} - {analysis_df['iteration'].max()}")
    print()
    
    # Calculate differences
    analysis_df['score_diff'] = analysis_df['calculated_score'] - analysis_df['my_score']
    analysis_df['abs_diff'] = abs(analysis_df['score_diff'])
    
    # Statistical summary
    mean_diff = analysis_df['score_diff'].mean()
    std_diff = analysis_df['score_diff'].std()
    mae = analysis_df['abs_diff'].mean()
    rmse = np.sqrt((analysis_df['score_diff']**2).mean())
    correlation = analysis_df['calculated_score'].corr(analysis_df['my_score'])
    
    print("📊 STATISTICAL SUMMARY:")
    print(f"• Mean difference (Calculated - Subjective): {mean_diff:.2f} points")
    print(f"• Standard deviation: {std_diff:.2f} points") 
    print(f"• Mean Absolute Error (MAE): {mae:.2f} points")
    print(f"• Root Mean Square Error (RMSE): {rmse:.2f} points")
    print(f"• Correlation coefficient: {correlation:.3f}")
    print()
    
    # Agreement analysis
    close_matches = len(analysis_df[analysis_df['abs_diff'] <= 10])
    reasonable_matches = len(analysis_df[analysis_df['abs_diff'] <= 20])
    
    print("🎯 AGREEMENT ANALYSIS:")
    print(f"• Within ±10 points: {close_matches}/{len(analysis_df)} ({close_matches/len(analysis_df)*100:.1f}%)")
    print(f"• Within ±20 points: {reasonable_matches}/{len(analysis_df)} ({reasonable_matches/len(analysis_df)*100:.1f}%)")
    print()
    
    # Quality category analysis
    def get_quality_category(score):
        if score >= 61: return "Good"
        elif score >= 41: return "Average"
        elif score >= 21: return "Below Average"
        else: return "Failed/Poor"
    
    analysis_df['calc_category'] = analysis_df['calculated_score'].apply(get_quality_category)
    analysis_df['subj_category'] = analysis_df['my_score'].apply(get_quality_category)
    analysis_df['category_match'] = analysis_df['calc_category'] == analysis_df['subj_category']
    
    category_agreement = analysis_df['category_match'].sum() / len(analysis_df) * 100
    print(f"📝 QUALITY CATEGORY AGREEMENT: {category_agreement:.1f}%")
    print()
    
    # Detailed breakdown
    print("🔍 DETAILED SAMPLE ANALYSIS:")
    print("-" * 85)
    print("Iter | Calc | Subj | Diff | Weight | Touch1/2/3 | Max1/2/3    | Notes")
    print("-" * 85)
    
    for _, row in analysis_df.iterrows():
        iter_num = int(row['iteration'])
        calc_score = row['calculated_score']
        subj_score = row['my_score'] 
        diff = row['score_diff']
        weight = row['cc_weight']
        
        # Handle missing touch/max data
        t1 = row['touch_pos1'] if pd.notna(row['touch_pos1']) else 'N/A'
        t2 = row['touch_pos2'] if pd.notna(row['touch_pos2']) else 'N/A'
        t3 = row['touch_pos3'] if pd.notna(row['touch_pos3']) else 'N/A'
        m1 = row['max_pos1'] if pd.notna(row['max_pos1']) else 'N/A'
        m2 = row['max_pos2'] if pd.notna(row['max_pos2']) else 'N/A' 
        m3 = row['max_pos3'] if pd.notna(row['max_pos3']) else 'N/A'
        
        # Special case notes
        notes = ""
        if abs(diff) > 25:
            notes += "LARGE_DIFF "
        if weight < 1.0:
            notes += "VERY_LOW_WEIGHT "
        if calc_score < 20:
            notes += "FAILED_CALC "
        if subj_score < 20:
            notes += "FAILED_SUBJ "
            
        print(f"{iter_num:4d} | {calc_score:4.1f} | {subj_score:4.0f} | {diff:+5.1f} | {weight:6.2f} | {t1}/{t2}/{t3} | {m1}/{m2}/{m3} | {notes}")
    
    print("-" * 85)
    print()
    
    # Problem case analysis
    large_diffs = analysis_df[analysis_df['abs_diff'] > 20]
    if len(large_diffs) > 0:
        print("⚠️ SAMPLES WITH LARGE DIFFERENCES (>20 points):")
        for _, row in large_diffs.iterrows():
            print(f"• Iteration {int(row['iteration'])}: Calculated={row['calculated_score']:.1f}, Subjective={row['my_score']:.0f} (diff: {row['score_diff']:+.1f})")
            print(f"  Weight: {row['cc_weight']:.2f}g, Touch: {row['touch_pos1']:.1f}/{row['touch_pos2']:.1f}/{row['touch_pos3']:.1f}")
        print()
    
    # Success stories
    good_matches = analysis_df[analysis_df['abs_diff'] <= 5]
    if len(good_matches) > 0:
        print("✅ EXCELLENT MATCHES (≤5 points difference):")
        for _, row in good_matches.iterrows():
            print(f"• Iteration {int(row['iteration'])}: Calculated={row['calculated_score']:.1f}, Subjective={row['my_score']:.0f} (diff: {row['score_diff']:+.1f})")
        print()
    
    # Model performance by weight range
    print("📈 PERFORMANCE BY WEIGHT RANGE:")
    weight_ranges = [
        (0, 2, "Very Light (<2g)"),
        (2, 5, "Light (2-5g)"), 
        (5, 8, "Medium (5-8g)"),
        (8, 12, "Heavy (8-12g)"),
        (12, float('inf'), "Very Heavy (>12g)")
    ]
    
    for min_w, max_w, label in weight_ranges:
        subset = analysis_df[(analysis_df['cc_weight'] >= min_w) & (analysis_df['cc_weight'] < max_w)]
        if len(subset) > 0:
            subset_mae = subset['abs_diff'].mean()
            subset_corr = subset['calculated_score'].corr(subset['my_score'])
            print(f"• {label}: {len(subset)} samples, MAE={subset_mae:.1f}, Corr={subset_corr:.3f}")
    
    print()
    print("🎯 OVERALL ASSESSMENT:")
    if mae <= 15:
        print("✅ EXCELLENT: Model performs very well with low average error")
    elif mae <= 25:
        print("✅ GOOD: Model performs well with acceptable error levels")  
    else:
        print("⚠️ NEEDS IMPROVEMENT: Higher than desired error levels")
    
    if correlation >= 0.4:
        print("✅ GOOD CORRELATION: Model captures subjective quality trends")
    else:
        print("⚠️ WEAK CORRELATION: Model may not capture subjective preferences well")
    
    return analysis_df

if __name__ == "__main__":
    analysis_results = analyze_quality_scores()
    
    print()
    print("Analysis complete! Key findings saved in analysis_results DataFrame.")
