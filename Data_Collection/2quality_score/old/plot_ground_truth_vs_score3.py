#!/usr/bin/env python3
"""
Simple Graph: Ground Truth vs Bias-Corrected Score3 over Iterations

Shows how both scores change over time as you improved your cotton candy making technique.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def create_iteration_graph():
    """Create a clean graph showing scores vs iteration"""
    
    # Load data
    df = pd.read_csv('cotton_candy_with_improved_score3.csv')
    
    # Filter data to show only iterations 15-96
    df_filtered = df[(df['iteration'] >= 15) & (df['iteration'] <= 96)]
    
    # Create the plot
    plt.figure(figsize=(12, 7))
    
    # Plot ground truth
    plt.plot(df_filtered['iteration'], df_filtered['my_score'], 'o-', 
             linewidth=2.5, markersize=6, alpha=0.8, 
             color='#2E86C1', label='Ground Truth', markeredgecolor='#1B4F72')
    
    # Plot bias-corrected Score3
    plt.plot(df_filtered['iteration'], df_filtered['calculated_score3_improved'], 's-', 
             linewidth=2.5, markersize=5, alpha=0.8, 
             color='#E74C3C', label='Predicted Quality', markeredgecolor='#A93226')
    
    # Styling
    plt.xlabel('Production Iteration', fontsize=12, fontweight='bold')
    plt.ylabel('Quality Score', fontsize=12, fontweight='bold')
    
    # Clean grid and legend
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    plt.legend(fontsize=11, loc='lower right', framealpha=0.9)
    
    # Set axis limits
    plt.ylim(0, 75)
    plt.xlim(15, 96)
    
    # Calculate correlation and sample count
    correlation = df_filtered['my_score'].corr(df_filtered['calculated_score3_improved'])
    n_samples = len(df_filtered)
    
    # Add clean performance metric with sample count
    textstr = f'r = {correlation:.3f}\nn = {n_samples}'
    props = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='gray')
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=11,
             verticalalignment='top', bbox=props, fontweight='bold')
    
    plt.tight_layout()
    return plt

def print_analysis(df):
    """Print detailed analysis of the comparison"""
    
    print("📊 GROUND TRUTH vs BIAS-CORRECTED SCORE3 ANALYSIS")
    print("="*60)
    
    # Overall statistics
    correlation = df['my_score'].corr(df['calculated_score3_improved'])
    mae = np.mean(np.abs(df['my_score'] - df['calculated_score3_improved']))
    rmse = np.sqrt(np.mean((df['my_score'] - df['calculated_score3_improved'])**2))
    
    print(f"📈 Overall Performance:")
    print(f"   Correlation: {correlation:.4f} ({correlation:.1%})")
    print(f"   Mean Absolute Error: {mae:.2f}")
    print(f"   Root Mean Square Error: {rmse:.2f}")
    
    # Score ranges
    print(f"\n📋 Data Summary:")
    print(f"   Total Iterations: {len(df)}")
    print(f"   Iteration Range: {df['iteration'].min()} to {df['iteration'].max()}")
    print(f"   Ground Truth Range: {df['my_score'].min():.0f} to {df['my_score'].max():.0f}")
    print(f"   Score3 Range: {df['calculated_score3_improved'].min():.1f} to {df['calculated_score3_improved'].max():.1f}")
    
    # Learning curve analysis
    early_iterations = df[df['iteration'] <= 30]
    late_iterations = df[df['iteration'] >= 70]
    
    if len(early_iterations) > 0 and len(late_iterations) > 0:
        print(f"\n📚 Learning Progress:")
        print(f"   Early iterations (≤30): Avg ground truth = {early_iterations['my_score'].mean():.1f}")
        print(f"   Late iterations (≥70): Avg ground truth = {late_iterations['my_score'].mean():.1f}")
        print(f"   Improvement: +{late_iterations['my_score'].mean() - early_iterations['my_score'].mean():.1f} points")
    
    # Best predictions
    errors = np.abs(df['my_score'] - df['calculated_score3_improved'])
    best_predictions = df[errors <= 5]  # Within 5 points
    
    print(f"\n🎯 Prediction Accuracy:")
    print(f"   Predictions within 5 points: {len(best_predictions)}/{len(df)} ({len(best_predictions)/len(df):.1%})")
    print(f"   Perfect predictions (±1 point): {len(df[errors <= 1])}/{len(df)} ({len(df[errors <= 1])/len(df):.1%})")
    
    # Worst prediction
    worst_idx = errors.idxmax()
    worst_error = errors.max()
    print(f"   Largest error: {worst_error:.1f} points (iteration {df.loc[worst_idx, 'iteration']})")

def main():
    """Main function"""
    
    print("📈 CREATING CLEAN ITERATION GRAPH: MANUAL vs PREDICTED QUALITY")
    print("="*60)
    
    # Load data
    df = pd.read_csv('cotton_candy_with_improved_score3.csv')
    
    # Filter data
    df_filtered = df[(df['iteration'] >= 15) & (df['iteration'] <= 96)]
    print(f"📊 Using iterations 15-96: {len(df_filtered)} samples")
    
    # Create graph
    print(f"\n🎨 Creating clean visualization...")
    plt = create_iteration_graph()
    
    # Save graph
    plt.savefig('ground_truth_vs_score3_iterations.png', dpi=300, bbox_inches='tight')
    print("✅ Graph saved as: ground_truth_vs_score3_iterations.png")
    
    # Show correlation
    correlation = df_filtered['my_score'].corr(df_filtered['calculated_score3_improved'])
    print(f"📈 Correlation: {correlation:.3f}")
    
    # Show graph
    plt.show()
    
    print(f"\n🎉 Clean graph complete!")

if __name__ == "__main__":
    main()
