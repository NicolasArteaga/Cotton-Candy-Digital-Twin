#!/usr/bin/env python3
"""
🎯 Max Position Analysis - Finding Optimal Max_Pos Ranges
========================================================
Analyze max_pos1, max_pos2, max_pos3 values to find optimal ranges
for high-quality cotton candy production.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_max_positions():
    """Analyze max_pos values and their correlation with quality scores."""
    
    # Load dataset
    df = pd.read_csv('my_cc_dataset.csv')
    
    # Filter for iteration 14+ where we have complete max_pos data
    analysis_df = df[(df['iteration'] >= 14) & df['my_score'].notna()].copy()
    
    print("🎯 MAX POSITION ANALYSIS")
    print("=" * 50)
    print(f"Analyzing {len(analysis_df)} samples from iteration 14+")
    print()
    
    # Basic statistics for each max_pos
    max_pos_cols = ['max_pos1', 'max_pos2', 'max_pos3']
    
    print("📊 MAX POSITION STATISTICS:")
    print("-" * 50)
    for col in max_pos_cols:
        values = analysis_df[col].dropna()
        print(f"{col}:")
        print(f"  Range: {values.min():.2f} - {values.max():.2f}")
        print(f"  Mean: {values.mean():.2f}, Std: {values.std():.2f}")
        print(f"  Median: {values.median():.2f}")
        print()
    
    # Correlation with quality scores
    print("🔗 CORRELATION WITH QUALITY SCORES:")
    print("-" * 50)
    for col in max_pos_cols:
        corr_my = analysis_df[col].corr(analysis_df['my_score'])
        corr_calc = analysis_df[col].corr(analysis_df['calculated_score'])
        print(f"{col}:")
        print(f"  vs Your Score: {corr_my:.3f}")
        print(f"  vs Calculated: {corr_calc:.3f}")
        print()
    
    # Quality category analysis
    def categorize_quality(score):
        if score >= 50:
            return "High (50+)"
        elif score >= 30:
            return "Medium (30-49)"
        elif score >= 15:
            return "Low (15-29)"
        else:
            return "Failed (<15)"
    
    analysis_df['quality_category'] = analysis_df['my_score'].apply(categorize_quality)
    
    print("🏆 OPTIMAL RANGES BY QUALITY CATEGORY:")
    print("-" * 70)
    
    # Analyze each quality category
    categories = ['High (50+)', 'Medium (30-49)', 'Low (15-29)', 'Failed (<15)']
    
    for category in categories:
        subset = analysis_df[analysis_df['quality_category'] == category]
        if len(subset) == 0:
            continue
            
        print(f"\n{category} Quality ({len(subset)} samples):")
        print("  Max_Pos Ranges:")
        
        for col in max_pos_cols:
            values = subset[col].dropna()
            if len(values) > 0:
                print(f"    {col}: {values.min():.1f} - {values.max():.1f} "
                      f"(avg: {values.mean():.1f}, median: {values.median():.1f})")
    
    # Find "sweet spots" - ranges that consistently produce good results
    print("\n🎯 SWEET SPOT ANALYSIS:")
    print("-" * 50)
    
    # Only look at samples with scores >= 40 (your good quality threshold)
    good_samples = analysis_df[analysis_df['my_score'] >= 40]
    poor_samples = analysis_df[analysis_df['my_score'] < 20]
    
    print(f"Analyzing {len(good_samples)} good samples (score ≥40) vs {len(poor_samples)} poor samples (<20)")
    print()
    
    for col in max_pos_cols:
        good_vals = good_samples[col].dropna()
        poor_vals = poor_samples[col].dropna()
        
        if len(good_vals) > 0 and len(poor_vals) > 0:
            print(f"{col} Sweet Spots:")
            print(f"  Good samples range: {good_vals.min():.1f} - {good_vals.max():.1f} (avg: {good_vals.mean():.1f})")
            print(f"  Poor samples range: {poor_vals.min():.1f} - {poor_vals.max():.1f} (avg: {poor_vals.mean():.1f})")
            
            # Find recommended range (where good samples cluster)
            q25, q75 = good_vals.quantile(0.25), good_vals.quantile(0.75)
            print(f"  🎯 RECOMMENDED RANGE: {q25:.1f} - {q75:.1f} (middle 50% of good samples)")
            print()
    
    # Detailed sample breakdown
    print("📋 DETAILED SAMPLE BREAKDOWN:")
    print("-" * 80)
    print("Iter | Score | Max1  | Max2  | Max3  | Quality Notes")
    print("-" * 80)
    
    # Sort by quality score for better visualization
    sorted_df = analysis_df.sort_values('my_score', ascending=False)
    
    for _, row in sorted_df.iterrows():
        iter_num = int(row['iteration'])
        score = row['my_score']
        m1 = row['max_pos1'] if pd.notna(row['max_pos1']) else 0
        m2 = row['max_pos2'] if pd.notna(row['max_pos2']) else 0  
        m3 = row['max_pos3'] if pd.notna(row['max_pos3']) else 0
        
        # Quality assessment
        if score >= 50:
            quality = "EXCELLENT"
        elif score >= 40:
            quality = "GOOD"
        elif score >= 25:
            quality = "FAIR"
        elif score >= 10:
            quality = "POOR"
        else:
            quality = "FAILED"
            
        print(f"{iter_num:4d} | {score:5.0f} | {m1:5.1f} | {m2:5.1f} | {m3:5.1f} | {quality}")
    
    # Pattern recognition
    print("\n🔍 PATTERN ANALYSIS:")
    print("-" * 50)
    
    # Look for patterns in the best samples
    excellent_samples = analysis_df[analysis_df['my_score'] >= 55]
    if len(excellent_samples) > 0:
        print(f"EXCELLENT samples (≥55 points) - {len(excellent_samples)} cases:")
        for _, row in excellent_samples.iterrows():
            print(f"  Iter {int(row['iteration'])}: max_pos = {row['max_pos1']:.1f}/{row['max_pos2']:.1f}/{row['max_pos3']:.1f}")
    
    # Look for patterns in failed samples  
    failed_samples = analysis_df[analysis_df['my_score'] <= 10]
    if len(failed_samples) > 0:
        print(f"\nFAILED samples (≤10 points) - {len(failed_samples)} cases:")
        for _, row in failed_samples.iterrows():
            m1 = row['max_pos1'] if pd.notna(row['max_pos1']) else 0
            m2 = row['max_pos2'] if pd.notna(row['max_pos2']) else 0
            m3 = row['max_pos3'] if pd.notna(row['max_pos3']) else 0
            print(f"  Iter {int(row['iteration'])}: max_pos = {m1:.1f}/{m2:.1f}/{m3:.1f}")
    
    return analysis_df

def create_max_pos_visualization(analysis_df):
    """Create visualization of max_pos relationships."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Max Position Analysis vs Quality Scores', fontsize=16)
    
    # Individual max_pos vs quality
    max_pos_cols = ['max_pos1', 'max_pos2', 'max_pos3']
    
    for i, col in enumerate(max_pos_cols):
        ax = axes[i//2, i%2] if i < 3 else axes[1, 1]
        
        # Scatter plot
        scatter = ax.scatter(analysis_df[col], analysis_df['my_score'], 
                           c=analysis_df['calculated_score'], cmap='viridis', 
                           alpha=0.7, s=60)
        ax.set_xlabel(f'{col}')
        ax.set_ylabel('Your Quality Score')
        ax.set_title(f'{col} vs Quality')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar for calculated scores
        plt.colorbar(scatter, ax=ax, label='Calculated Score')
    
    # Combined analysis in the 4th subplot
    if len(max_pos_cols) == 3:
        ax = axes[1, 1]
        ax.clear()
        
        # Create combined max_pos score (simple average)
        analysis_df['avg_max_pos'] = analysis_df[max_pos_cols].mean(axis=1)
        
        scatter = ax.scatter(analysis_df['avg_max_pos'], analysis_df['my_score'],
                           c=analysis_df['cc_weight'], cmap='plasma',
                           alpha=0.7, s=60)
        ax.set_xlabel('Average Max Position')
        ax.set_ylabel('Your Quality Score')  
        ax.set_title('Average Max Position vs Quality')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Cotton Candy Weight (g)')
    
    plt.tight_layout()
    plt.savefig('max_pos_analysis.png', dpi=300, bbox_inches='tight')
    print("📈 Visualization saved as 'max_pos_analysis.png'")
    
    plt.show()

if __name__ == "__main__":
    print("🍭 Starting Max Position Analysis...")
    analysis_df = analyze_max_positions()
    create_max_pos_visualization(analysis_df)
    print("\n✅ Analysis complete!")
