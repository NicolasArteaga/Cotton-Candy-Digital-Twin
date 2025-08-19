#!/usr/bin/env python3
"""
🍭 Ultra Harsh Weight Penalty Update
===================================
Apply ZERO weight points for cotton candy under 6g
Gradual increase from 6g-7g, then full points 7g+
"""

import pandas as pd
import numpy as np
from cc_quality_score import calculate_cotton_candy_quality

def main():
    print("🍭 ULTRA HARSH Weight Penalty Implementation")
    print("=" * 50)
    print("New weight scoring:")
    print("• < 6g: 0 points (ZERO weight contribution)")
    print("• 6-7g: Gradual increase (0 to 65 points)")
    print("• 7-10g: Full 65 points") 
    print("• > 10g: Penalty for overweight")
    print()
    
    # Load dataset
    df = pd.read_csv('/Users/nicolas/Desktop/Cotton-Candy-Digital-Twin/Data_Collection/my_cc_dataset.csv')
    print(f"Loaded {len(df)} cotton candy samples")
    
    # Track samples by weight categories
    under_6g = df[df['cc_weight'] < 6.0]
    between_6_7g = df[(df['cc_weight'] >= 6.0) & (df['cc_weight'] < 7.0)]
    over_7g = df[df['cc_weight'] >= 7.0]
    
    print(f"\nWeight Distribution:")
    print(f"• < 6g: {len(under_6g)} samples (will get 0 weight points)")
    print(f"• 6-7g: {len(between_6_7g)} samples (gradual scaling)")
    print(f"• ≥ 7g: {len(over_7g)} samples (full/overweight points)")
    
    # Calculate new scores for rows with complete data
    print(f"\nCalculating ultra-harsh scores...")
    
    scores_changed = 0
    for index, row in df.iterrows():
        # Skip rows with missing essential data
        if pd.isna(row['touch_pos1']) or pd.isna(row['cc_weight']):
            continue
            
        # Calculate new ultra-harsh score
        new_score = calculate_cotton_candy_quality(
            row['touch_pos1'], row['touch_pos2'], row['touch_pos3'],
            row['max_pos1'], row['max_pos2'], row['max_pos3'], 
            row['cc_weight']
        )
        
        old_score = row.get('calculated_score', 0)
        
        # Update calculated_score column
        df.at[index, 'calculated_score'] = round(new_score, 1)
        
        # Track if score changed significantly  
        if abs(new_score - old_score) > 0.1:
            scores_changed += 1
    
    print(f"✅ Updated {scores_changed} calculated scores")
    
    # Analyze the ultra-harsh impact
    print(f"\n🔍 ULTRA-HARSH WEIGHT PENALTY ANALYSIS:")
    print(f"{'Weight (g)':<10} {'Old Score':<10} {'New Score':<10} {'Change':<8}")
    print("-" * 40)
    
    # Show examples from different weight categories
    examples_shown = 0
    for _, row in df.iterrows():
        if pd.isna(row['cc_weight']) or examples_shown >= 8:
            continue
            
        weight = row['cc_weight']
        new_score = row['calculated_score']
        my_score = row.get('my_score', 'N/A')
        
        # Show key examples
        if weight < 1.0 or (6.0 <= weight < 7.0) or weight == 8.98:
            print(f"{weight:<10.2f} {'prev':<10} {new_score:<10.1f} {'harsh!':<8}")
            examples_shown += 1
    
    # Save updated dataset
    df.to_csv('/Users/nicolas/Desktop/Cotton-Candy-Digital-Twin/Data_Collection/my_cc_dataset.csv', index=False)
    print(f"\n✅ Saved ultra-harsh penalty scores to my_cc_dataset.csv")
    
    # Final validation
    complete_rows = df.dropna(subset=['my_score', 'calculated_score'])
    if len(complete_rows) > 0:
        correlation = complete_rows['my_score'].corr(complete_rows['calculated_score'])
        mae = np.mean(np.abs(complete_rows['my_score'] - complete_rows['calculated_score']))
        
        print(f"\n📊 ULTRA-HARSH PERFORMANCE METRICS:")
        print(f"• Correlation with subjective scores: {correlation:.3f}")
        print(f"• Mean Absolute Error: {mae:.2f} points")
        
        # Check ultra-low weight samples
        ultra_low = complete_rows[complete_rows['cc_weight'] < 6.0]
        if len(ultra_low) > 0:
            print(f"\n⚡ ULTRA-LOW WEIGHT SAMPLES (<6g): {len(ultra_low)} samples")
            for _, sample in ultra_low.iterrows():
                print(f"  Weight: {sample['cc_weight']:.2f}g → Score: {sample['calculated_score']:.1f} (was my_score: {sample['my_score']})")
                
    print(f"\n🎯 ULTRA-HARSH SYSTEM DEPLOYED!")
    print(f"Cotton candy under 6g now gets ZERO weight points!")

if __name__ == "__main__":
    main()
