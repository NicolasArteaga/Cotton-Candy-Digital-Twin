#!/usr/bin/env python3
"""
Feature Selector for Decision Tree Optimization
This utility helps select the most relevant features from your comprehensive dataset
"""

import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
import matplotlib.pyplot as plt

class DecisionTreeFeatureSelector:
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.feature_importance = {}
        
    def get_feature_categories(self):
        """Categorize features for better understanding"""
        categories = {
            'process_params': [],
            'timing': [],
            'environmental': [],
            'quality': [],
            'other': []
        }
        
        for col in self.df.columns:
            if col in ['wait_time', 'cook_time', 'cooldown_time', 'iteration_since_maintenance']:
                categories['process_params'].append(col)
            elif any(word in col for word in ['duration', 'time', 'phase']):
                categories['timing'].append(col)
            elif 'env_' in col or any(word in col for word in ['temperature', 'humidity', 'pressure']):
                categories['environmental'].append(col)
            elif any(word in col for word in ['consistency', 'stability', 'quality', 'weight']):
                categories['quality'].append(col)
            elif col not in ['source_file']:
                categories['other'].append(col)
                
        return categories
    
    def analyze_feature_importance(self, target_column='final_weight', top_k=15):
        """Analyze feature importance using Random Forest"""
        # Prepare data
        feature_cols = [col for col in self.df.columns 
                       if col not in ['source_file', target_column] and 
                       self.df[col].dtype in ['int64', 'float64']]
        
        # Remove rows with missing target
        clean_df = self.df.dropna(subset=[target_column])
        if len(clean_df) == 0:
            print(f"No valid data for target '{target_column}'")
            return []
        
        X = clean_df[feature_cols].fillna(0)
        y = clean_df[target_column]
        
        # Random Forest for feature importance
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        # Get feature importance
        importance_scores = list(zip(feature_cols, rf.feature_importances_))
        importance_scores.sort(key=lambda x: x[1], reverse=True)
        
        self.feature_importance[target_column] = importance_scores
        
        print(f"\nTop {top_k} features for predicting '{target_column}':")
        print("-" * 60)
        for i, (feature, importance) in enumerate(importance_scores[:top_k]):
            print(f"{i+1:2d}. {feature:30s}: {importance:.4f}")
        
        return importance_scores[:top_k]
    
    def create_minimal_dataset(self, target_features=None, max_features=20):
        """Create a minimal dataset with most important features"""
        if target_features is None:
            # Auto-select based on importance analysis
            important_features = []
            
            # Analyze importance for different targets
            targets_to_analyze = ['final_weight', 'overall_quality_score']
            targets_to_analyze = [t for t in targets_to_analyze if t in self.df.columns]
            
            for target in targets_to_analyze:
                top_features = self.analyze_feature_importance(target, top_k=10)
                important_features.extend([f[0] for f in top_features[:5]])  # Top 5 for each target
            
            # Remove duplicates and add essential features
            essential_features = ['wait_time', 'cook_time', 'cooldown_time']
            target_features = list(set(important_features + essential_features))
            target_features = target_features[:max_features]
        
        # Always include source file for tracking
        if 'source_file' in self.df.columns:
            target_features.append('source_file')
        
        # Include target columns
        target_cols = ['final_weight', 'overall_quality_score', 'high_quality']
        for col in target_cols:
            if col in self.df.columns and col not in target_features:
                target_features.append(col)
        
        # Create minimal dataset
        available_features = [f for f in target_features if f in self.df.columns]
        minimal_df = self.df[available_features].copy()
        
        return minimal_df, available_features
    
    def remove_correlated_features(self, correlation_threshold=0.9):
        """Remove highly correlated features to reduce redundancy"""
        numeric_df = self.df.select_dtypes(include=[np.number])
        correlation_matrix = numeric_df.corr().abs()
        
        # Find pairs of highly correlated features
        highly_correlated = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                if correlation_matrix.iloc[i, j] > correlation_threshold:
                    col1, col2 = correlation_matrix.columns[i], correlation_matrix.columns[j]
                    highly_correlated.append((col1, col2, correlation_matrix.iloc[i, j]))
        
        # Remove features with high correlation (keep the first one in each pair)
        features_to_remove = set()
        for col1, col2, corr in highly_correlated:
            print(f"High correlation: {col1} <-> {col2} ({corr:.3f})")
            features_to_remove.add(col2)  # Remove the second feature
        
        print(f"\nRemoving {len(features_to_remove)} highly correlated features:")
        for feature in features_to_remove:
            print(f"  - {feature}")
        
        # Create dataset without highly correlated features
        remaining_features = [col for col in self.df.columns if col not in features_to_remove]
        return self.df[remaining_features], list(features_to_remove)
    
    def create_decision_tree_ready_dataset(self, output_file="cotton_candy_dt_ready.csv"):
        """Create a decision tree ready dataset with optimal features"""
        print("Creating Decision Tree Ready Dataset")
        print("=" * 50)
        
        # Step 1: Remove highly correlated features
        print("Step 1: Removing highly correlated features...")
        df_uncorrelated, removed_features = self.remove_correlated_features()
        print(f"Features after correlation removal: {len(df_uncorrelated.columns)}")
        
        # Update working dataframe
        self.df = df_uncorrelated
        
        # Step 2: Select most important features
        print("\nStep 2: Selecting most important features...")
        minimal_df, selected_features = self.create_minimal_dataset(max_features=15)
        print(f"Selected {len(selected_features)} features for decision tree")
        
        # Step 3: Clean data
        print("\nStep 3: Cleaning data...")
        # Remove rows where all feature values are 0 or NaN
        feature_cols = [col for col in selected_features 
                       if col not in ['source_file', 'high_quality', 'overall_quality_score']]
        
        # Fill NaN values with column median for numeric columns
        for col in feature_cols:
            if minimal_df[col].dtype in ['int64', 'float64']:
                minimal_df[col] = minimal_df[col].fillna(minimal_df[col].median())
        
        # Remove rows with no meaningful data
        valid_rows = minimal_df[feature_cols].sum(axis=1) > 0
        clean_df = minimal_df[valid_rows].copy()
        
        print(f"Clean dataset: {len(clean_df)} rows, {len(clean_df.columns)} columns")
        
        # Step 4: Save dataset
        clean_df.to_csv(output_file, index=False)
        print(f"\nDecision tree ready dataset saved to: {output_file}")
        
        # Step 5: Show summary
        self.show_dataset_summary(clean_df)
        
        return clean_df
    
    def show_dataset_summary(self, df):
        """Show summary of the final dataset"""
        categories = self.get_feature_categories()
        
        print("\nFinal Dataset Summary:")
        print("-" * 30)
        print(f"Total rows: {len(df)}")
        print(f"Total features: {len(df.columns)}")
        
        for category, features in categories.items():
            category_features = [f for f in features if f in df.columns]
            if category_features:
                print(f"{category.replace('_', ' ').title()}: {len(category_features)} features")
                for feature in category_features[:3]:  # Show first 3
                    print(f"  - {feature}")
                if len(category_features) > 3:
                    print(f"  ... and {len(category_features)-3} more")


def main():
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python feature_selector.py <csv_file>")
        print("Example: python feature_selector.py cotton_candy_all_batches_dataset.csv")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    if not os.path.exists(csv_file):
        print(f"Error: File '{csv_file}' not found!")
        sys.exit(1)
    
    print(f"Analyzing features in {csv_file}...")
    selector = DecisionTreeFeatureSelector(csv_file)
    
    # Create decision tree ready dataset
    dt_ready_df = selector.create_decision_tree_ready_dataset()
    
    print("\n" + "="*60)
    print("Your dataset is now ready for Decision Tree training!")
    print("="*60)


if __name__ == "__main__":
    main()
