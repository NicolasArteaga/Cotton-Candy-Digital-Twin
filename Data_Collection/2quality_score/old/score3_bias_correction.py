#!/usr/bin/env python3
"""
Score3 Bias Correction - Addressing Low Score Over-Prediction

This script creates an improved calculated_score3 that handles low scores better
by training on ALL data with proper bias correction and range constraints.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load the final scores data"""
    df = pd.read_csv('cotton_candy_final_scores.csv')
    
    # Clean data
    df = df.dropna(subset=['my_score', 'cc_weight'])
    df = df[df['max_pos3'] < 100]  # Remove extreme outliers
    
    print(f"Loaded {len(df)} clean samples")
    print(f"Score range: {df['my_score'].min():.0f} to {df['my_score'].max():.0f}")
    
    return df

def create_enhanced_features(df):
    """Create enhanced features that work better across all score ranges"""
    
    # Weight features
    df['weight_optimality'] = np.exp(-((df['cc_weight'] - 10)**2) / 8)  # Gaussian around 10
    df['weight_too_low'] = np.maximum(0, (7 - df['cc_weight'])) / 7  # Penalty for very low weight
    df['weight_too_high'] = np.maximum(0, (df['cc_weight'] - 13)) / 5  # Penalty for very high weight
    
    # Touch quality features - more nuanced
    for i in [1, 2, 3]:
        col = f'touch_pos{i}'
        # No-touch penalty (11 = didn't touch)
        df[f'no_touch_{i}'] = (df[col] == 11).astype(float)
        
        # Touch quality score (3-6 is good, further from this range is worse)
        ideal_touch = 4.5  # Middle of 3-6 range
        df[f'touch_quality_{i}'] = 1.0 / (1.0 + abs(df[col] - ideal_touch) / 4.0)
        df[f'touch_quality_{i}'] = np.where(df[col] == 11, 0.1, df[f'touch_quality_{i}'])
    
    # Max position features - normalized and capped
    for i in [1, 2, 3]:
        col = f'max_pos{i}'
        df[f'max_pos_{i}_norm'] = np.minimum(df[col], 50) / 50  # Cap at 50, normalize
    
    # Interaction features
    df['total_no_touch'] = df['no_touch_1'] + df['no_touch_2'] + df['no_touch_3']
    df['avg_touch_quality'] = (df['touch_quality_1'] + df['touch_quality_2'] + df['touch_quality_3']) / 3
    df['avg_max_pos'] = (df['max_pos_1_norm'] + df['max_pos_2_norm'] + df['max_pos_3_norm']) / 3
    
    # Experience factor (later iterations might have better technique)
    df['experience'] = (df['iteration'] - df['iteration'].min()) / (df['iteration'].max() - df['iteration'].min())
    
    return df

def train_improved_score3(df):
    """Train an improved Score3 that handles all score ranges better"""
    
    # Create features
    df = create_enhanced_features(df)
    
    # Feature columns
    feature_cols = [
        'weight_optimality', 'weight_too_low', 'weight_too_high',
        'touch_quality_1', 'touch_quality_2', 'touch_quality_3',
        'no_touch_1', 'no_touch_2', 'no_touch_3',
        'max_pos_1_norm', 'max_pos_2_norm', 'max_pos_3_norm',
        'total_no_touch', 'avg_touch_quality', 'avg_max_pos',
        'experience'
    ]
    
    X = df[feature_cols]
    y = df['my_score']
    
    print(f"\nTraining improved Score3 on ALL {len(df)} samples...")
    print("Features used:", feature_cols)
    
    # Try different algorithms
    algorithms = {
        'Ridge (conservative)': Ridge(alpha=10.0),  # More conservative
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    }
    
    best_model = None
    best_score = -np.inf
    best_name = ""
    
    for name, model in algorithms.items():
        # Cross-validation to avoid overfitting
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
        cv_mean = cv_scores.mean()
        
        print(f"{name}: CV MAE = {-cv_mean:.2f} (±{cv_scores.std():.2f})")
        
        if cv_mean > best_score:
            best_score = cv_mean
            best_model = model
            best_name = name
    
    print(f"\nBest model: {best_name}")
    
    # Train best model on all data
    best_model.fit(X, y)
    
    # Make predictions
    predictions_raw = best_model.predict(X)
    
    # Apply bounds and bias correction
    predictions_bounded = np.clip(predictions_raw, 0, 75)
    
    # Bias correction: Check if we're systematically over/under predicting
    bias = np.mean(predictions_bounded - y)
    predictions_corrected = predictions_bounded - bias
    predictions_final = np.clip(predictions_corrected, 0, 75)
    
    print(f"Bias detected: {bias:.2f}")
    print(f"Applied bias correction and bounds [0, 75]")
    
    return predictions_final, best_model, feature_cols

def evaluate_improved_score3(df, predictions):
    """Evaluate the improved Score3"""
    
    df['calculated_score3_improved'] = predictions
    
    print(f"\n" + "="*60)
    print("IMPROVED SCORE3 EVALUATION")
    print("="*60)
    
    # Overall performance
    overall_corr = df['my_score'].corr(df['calculated_score3_improved'])
    overall_mae = mean_absolute_error(df['my_score'], df['calculated_score3_improved'])
    
    print(f"Overall Performance:")
    print(f"  Correlation: {overall_corr:.3f}")
    print(f"  MAE: {overall_mae:.2f}")
    
    # Compare with original Score3
    if 'calculated_score3' in df.columns:
        orig_corr = df['my_score'].corr(df['calculated_score3'])
        orig_mae = mean_absolute_error(df['my_score'], df['calculated_score3'])
        
        print(f"\nComparison with Original Score3:")
        print(f"  Correlation: {overall_corr:.3f} vs {orig_corr:.3f} ({((overall_corr/orig_corr-1)*100):+.1f}%)")
        print(f"  MAE: {overall_mae:.2f} vs {orig_mae:.2f} ({((orig_mae-overall_mae)/orig_mae*100):+.1f}%)")
    
    # Performance by score ranges
    print(f"\nPerformance by Score Ranges:")
    score_ranges = [
        (0, 15, "Very Low"),
        (16, 30, "Low"), 
        (31, 45, "Medium"),
        (46, 60, "High"),
        (61, 75, "Very High")
    ]
    
    for min_score, max_score, range_name in score_ranges:
        mask = (df['my_score'] >= min_score) & (df['my_score'] <= max_score)
        subset = df[mask]
        
        if len(subset) == 0:
            continue
            
        corr = subset['my_score'].corr(subset['calculated_score3_improved'])
        mae = mean_absolute_error(subset['my_score'], subset['calculated_score3_improved'])
        bias = (subset['calculated_score3_improved'] - subset['my_score']).mean()
        
        print(f"  {range_name} ({min_score}-{max_score}): n={len(subset)}, r={corr:.3f}, MAE={mae:.2f}, bias={bias:+.1f}")
    
    return df

def create_comparison_visualization(df):
    """Create visualization comparing original vs improved Score3"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Overall comparison scatter
    ax1 = axes[0, 0]
    ax1.scatter(df['my_score'], df['calculated_score3'], alpha=0.6, s=40, 
               color='red', label='Original Score3')
    ax1.scatter(df['my_score'], df['calculated_score3_improved'], alpha=0.6, s=40, 
               color='green', label='Improved Score3')
    
    # Perfect correlation line
    ax1.plot([0, 75], [0, 75], 'k--', alpha=0.3, label='Perfect')
    
    ax1.set_xlabel('Ground Truth')
    ax1.set_ylabel('Predicted Score')
    ax1.set_title('Overall: Original vs Improved Score3')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Low scores focus
    ax2 = axes[0, 1]
    low_scores = df[df['my_score'] <= 30]
    
    if len(low_scores) > 0:
        ax2.scatter(low_scores['my_score'], low_scores['calculated_score3'], alpha=0.7, s=60, 
                   color='red', label=f'Original (n={len(low_scores)})')
        ax2.scatter(low_scores['my_score'], low_scores['calculated_score3_improved'], alpha=0.7, s=60, 
                   color='green', label='Improved')
        
        ax2.plot([0, 30], [0, 30], 'k--', alpha=0.3, label='Perfect')
        
        # Calculate correlations
        orig_corr = low_scores['my_score'].corr(low_scores['calculated_score3'])
        impr_corr = low_scores['my_score'].corr(low_scores['calculated_score3_improved'])
        
        ax2.set_title(f'Low Scores (≤30)\nOrig r={orig_corr:.3f}, Impr r={impr_corr:.3f}')
    else:
        ax2.set_title('Low Scores (≤30) - No samples')
    
    ax2.set_xlabel('Ground Truth')
    ax2.set_ylabel('Predicted Score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Error distribution comparison
    ax3 = axes[0, 2]
    
    errors_orig = df['calculated_score3'] - df['my_score']
    errors_impr = df['calculated_score3_improved'] - df['my_score']
    
    ax3.hist(errors_orig, bins=15, alpha=0.6, color='red', label='Original')
    ax3.hist(errors_impr, bins=15, alpha=0.6, color='green', label='Improved')
    ax3.axvline(x=0, color='black', linestyle='--', alpha=0.8)
    ax3.axvline(x=errors_orig.mean(), color='red', linestyle=':', alpha=0.8, 
               label=f'Orig bias: {errors_orig.mean():+.1f}')
    ax3.axvline(x=errors_impr.mean(), color='green', linestyle=':', alpha=0.8, 
               label=f'Impr bias: {errors_impr.mean():+.1f}')
    
    ax3.set_xlabel('Prediction Error')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Error Distribution Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plots 4-6: Range-specific comparisons
    ranges = [(0, 30, 'Low (0-30)'), (31, 60, 'Med (31-60)'), (61, 75, 'High (61-75)')]
    
    for i, (min_s, max_s, title) in enumerate(ranges):
        ax = axes[1, i]
        
        subset = df[(df['my_score'] >= min_s) & (df['my_score'] <= max_s)]
        
        if len(subset) > 0:
            ax.scatter(subset['my_score'], subset['calculated_score3'], alpha=0.6, s=40, 
                      color='red', label='Original')
            ax.scatter(subset['my_score'], subset['calculated_score3_improved'], alpha=0.6, s=40, 
                      color='green', label='Improved')
            
            ax.plot([min_s, max_s], [min_s, max_s], 'k--', alpha=0.3)
            
            # Calculate MAE for both
            mae_orig = mean_absolute_error(subset['my_score'], subset['calculated_score3'])
            mae_impr = mean_absolute_error(subset['my_score'], subset['calculated_score3_improved'])
            
            ax.set_title(f'{title}\nMAE: {mae_orig:.1f}→{mae_impr:.1f}')
        else:
            ax.set_title(f'{title} - No samples')
        
        ax.set_xlabel('Ground Truth')
        ax.set_ylabel('Predicted')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def main():
    """Main function to create improved Score3"""
    
    print("🔧 CREATING IMPROVED SCORE3 - BIAS CORRECTED")
    print("="*60)
    print("Goal: Better performance on low scores, reduced bias")
    
    # Load data
    df = load_data()
    
    # Train improved model
    predictions, model, feature_cols = train_improved_score3(df)
    
    # Evaluate
    df = evaluate_improved_score3(df, predictions)
    
    # Create visualizations
    fig = create_comparison_visualization(df)
    
    # Save results
    fig.savefig('score3_bias_correction_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✓ Analysis saved: score3_bias_correction_analysis.png")
    
    # Save improved dataset
    df.to_csv('cotton_candy_with_improved_score3.csv', index=False)
    print("✓ Data with improved Score3 saved: cotton_candy_with_improved_score3.csv")
    
    # Feature importance (if available)
    if hasattr(model, 'feature_importances_'):
        print(f"\nTop 5 Feature Importances:")
        importances = list(zip(feature_cols, model.feature_importances_))
        importances.sort(key=lambda x: x[1], reverse=True)
        for feature, importance in importances[:5]:
            print(f"  {feature}: {importance:.3f}")
    elif hasattr(model, 'coef_'):
        print(f"\nTop 5 Feature Coefficients (absolute):")
        coeffs = list(zip(feature_cols, np.abs(model.coef_)))
        coeffs.sort(key=lambda x: x[1], reverse=True)
        for feature, coeff in coeffs[:5]:
            print(f"  {feature}: {coeff:.3f}")
    
    # Final summary
    print(f"\n🎯 IMPROVEMENT SUMMARY:")
    
    low_scores = df[df['my_score'] <= 30]
    if len(low_scores) > 0:
        orig_mae_low = mean_absolute_error(low_scores['my_score'], low_scores['calculated_score3'])
        impr_mae_low = mean_absolute_error(low_scores['my_score'], low_scores['calculated_score3_improved'])
        
        print(f"📉 Low Scores MAE: {orig_mae_low:.1f} → {impr_mae_low:.1f} ({((orig_mae_low-impr_mae_low)/orig_mae_low*100):+.1f}%)")
        
        orig_bias_low = (low_scores['calculated_score3'] - low_scores['my_score']).mean()
        impr_bias_low = (low_scores['calculated_score3_improved'] - low_scores['my_score']).mean()
        
        print(f"📉 Low Scores Bias: {orig_bias_low:+.1f} → {impr_bias_low:+.1f}")
    
    overall_orig_mae = mean_absolute_error(df['my_score'], df['calculated_score3'])
    overall_impr_mae = mean_absolute_error(df['my_score'], df['calculated_score3_improved'])
    
    print(f"🎯 Overall MAE: {overall_orig_mae:.1f} → {overall_impr_mae:.1f} ({((overall_orig_mae-overall_impr_mae)/overall_orig_mae*100):+.1f}%)")
    
    plt.show()

if __name__ == "__main__":
    main()
