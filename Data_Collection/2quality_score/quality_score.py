#!/usr/bin/env python3
"""
Quality Score - quality score for my_cc_dataset.csv
Usage: python quality_score.py [--range start end]
"""

import pandas as pd
import numpy as np
import argparse
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

def create_enhanced_features(df):
    """Create enhanced features for quality prediction"""
    
    # Weight features
    df['weight_optimality'] = np.exp(-((df['cc_weight'] - 10)**2) / 8)
    df['weight_too_low'] = np.maximum(0, (7 - df['cc_weight'])) / 7
    df['weight_too_high'] = np.maximum(0, (df['cc_weight'] - 13)) / 5
    
    # Touch quality features
    for i in [1, 2, 3]:
        col = f'touch_pos{i}'
        df[col] = df[col].fillna(11)
        df[f'no_touch_{i}'] = (df[col] == 11).astype(float)
        ideal_touch = 4.5
        df[f'touch_quality_{i}'] = 1.0 / (1.0 + abs(df[col] - ideal_touch) / 4.0)
        df[f'touch_quality_{i}'] = np.where(df[col] == 11, 0.1, df[f'touch_quality_{i}'])
    
    # Max position features
    for i in [1, 2, 3]:
        col = f'max_pos{i}'
        df[col] = df[col].fillna(50)
        df[f'max_pos_{i}_norm'] = np.minimum(df[col], 50) / 50
    
    # Interaction features
    df['total_no_touch'] = df['no_touch_1'] + df['no_touch_2'] + df['no_touch_3']
    df['avg_touch_quality'] = (df['touch_quality_1'] + df['touch_quality_2'] + df['touch_quality_3']) / 3
    df['avg_max_pos'] = (df['max_pos_1_norm'] + df['max_pos_2_norm'] + df['max_pos_3_norm']) / 3
    
    # Experience factor
    if 'iteration' in df.columns:
        df['experience'] = (df['iteration'] - df['iteration'].min()) / (df['iteration'].max() - df['iteration'].min())
    else:
        df['experience'] = 0.5
    
    return df

def train_model(df):
    """Train the calculated score model"""
    
    training_data = df[df['my_score'].notna()].copy()
    if len(training_data) == 0:
        raise ValueError("No training data available")
    
    training_data = create_enhanced_features(training_data)
    
    feature_cols = [
        'weight_optimality', 'weight_too_low', 'weight_too_high',
        'touch_quality_1', 'touch_quality_2', 'touch_quality_3',
        'no_touch_1', 'no_touch_2', 'no_touch_3',
        'max_pos_1_norm', 'max_pos_2_norm', 'max_pos_3_norm',
        'total_no_touch', 'avg_touch_quality', 'avg_max_pos',
        'experience'
    ]
    
    X = training_data[feature_cols]
    y = training_data['my_score']
    
    # Try models silently
    algorithms = {
        'Ridge': Ridge(alpha=10.0),
        'Linear': LinearRegression(),
        'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    }
    
    best_model = None
    best_score = -np.inf
    
    for name, model in algorithms.items():
        try:
            cv_scores = cross_val_score(model, X, y, cv=min(5, len(training_data)), 
                                      scoring='neg_mean_absolute_error')
            if cv_scores.mean() > best_score:
                best_score = cv_scores.mean()
                best_model = model
        except:
            continue
    
    if best_model is None:
        best_model = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42)
    
    best_model.fit(X, y)
    
    # Calculate bias correction
    training_predictions = np.clip(best_model.predict(X), 0, 75)
    bias = np.mean(training_predictions - y)
    
    return best_model, feature_cols, bias

def calculate_scores(df, model, feature_cols, bias):
    """Calculate scores for all rows"""
    
    df_with_features = create_enhanced_features(df.copy())
    
    for col in feature_cols:
        if col not in df_with_features.columns:
            df_with_features[col] = 0.5
    
    X = df_with_features[feature_cols]
    predictions_raw = model.predict(X)
    predictions_bounded = np.clip(predictions_raw, 0, 75)
    predictions_corrected = predictions_bounded - bias
    predictions_final = np.clip(predictions_corrected, 0, 75)
    
    return np.round(predictions_final, 2)

def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description='Update quality scores in my_cc_dataset.csv')
    parser.add_argument('--range', nargs=2, type=int, metavar=('START', 'END'),
                       help='Only update rows within iteration range (e.g., --range 5 12)')
    args = parser.parse_args()
    
    # Load dataset
    df = pd.read_csv('my_cc_dataset.csv')
    
    # Check required columns
    required_cols = ['touch_pos1', 'touch_pos2', 'touch_pos3', 
                    'max_pos1', 'max_pos2', 'max_pos3', 'cc_weight']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"ERROR: Missing columns: {missing_cols}")
        return
    
    # Filter by range if specified
    original_df = df.copy()
    if args.range:
        start_iter, end_iter = args.range
        update_mask = (df['iteration'] >= start_iter) & (df['iteration'] <= end_iter)
        rows_to_update = df[update_mask].index.tolist()
        print(f"Updating iterations {start_iter} to {end_iter}: {len(rows_to_update)} rows")
    else:
        rows_to_update = df.index.tolist()
        print(f"Updating all {len(rows_to_update)} rows")
    
    try:
        # Train model
        model, feature_cols, bias = train_model(df)
        
        # Calculate new scores
        new_scores = calculate_scores(df, model, feature_cols, bias)
        
        # Update only specified rows
        for idx in rows_to_update:
            original_df.loc[idx, 'quality_score'] = new_scores[idx]
        
        # Save updated dataset
        original_df.to_csv('my_cc_dataset.csv', index=False)
        
        # Show 5 random updated rows
        if len(rows_to_update) > 0:
            sample_indices = np.random.choice(rows_to_update, min(5, len(rows_to_update)), replace=False)
            print(f"\n5 random updated rows:")
            print("Iter | Manual | Quality | Weight | Touch1 | Touch2 | Touch3")
            print("-" * 60)
            
            for idx in sample_indices:
                row = original_df.loc[idx]
                manual = f"{row['my_score']:.0f}" if pd.notna(row['my_score']) else "None"
                calc = f"{row['quality_score']:.2f}"
                weight = f"{row['cc_weight']:.1f}" if pd.notna(row['cc_weight']) else "None"
                t1 = f"{row['touch_pos1']:.0f}" if pd.notna(row['touch_pos1']) else "None"
                t2 = f"{row['touch_pos2']:.0f}" if pd.notna(row['touch_pos2']) else "None"
                t3 = f"{row['touch_pos3']:.0f}" if pd.notna(row['touch_pos3']) else "None"
                
                print(f"{row['iteration']:4.0f} | {manual:6} | {calc:10} | {weight:6} | {t1:6} | {t2:6} | {t3:6}")
        
        print(f"\n✅ Updated quality_score column successfully!")
        
    except Exception as e:
        print(f"ERROR: {str(e)}")

if __name__ == "__main__":
    main()
