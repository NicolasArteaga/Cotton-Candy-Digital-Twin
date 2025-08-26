#!/usr/bin/env python3
"""
Data Exploration and Analysis
Comprehensive exploration of cotton candy quality features and targets
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load features and target data"""
    print("Loading data...")
    features_df = pd.read_csv('xy/features_X.csv')
    targets_df = pd.read_csv('xy/targets_Y.csv')
    
    # Merge on iteration
    data = pd.merge(features_df, targets_df, on='iteration', how='inner')
    print(f"Data shape: {data.shape}")
    print(f"Features: {features_df.columns.tolist()}")
    
    return data, features_df.columns[1:].tolist()  # Exclude 'iteration' from features

def explore_data_distribution(data, features):
    """Explore data distribution and basic statistics"""
    print("\n" + "="*50)
    print("DATA DISTRIBUTION ANALYSIS")
    print("="*50)
    
    # Basic statistics
    print("\nBasic Statistics:")
    print(data[features + ['quality_score']].describe())
    
    # Quality score distribution
    print(f"\nQuality Score Range: {data['quality_score'].min():.2f} - {data['quality_score'].max():.2f}")
    print(f"Quality Score Mean: {data['quality_score'].mean():.2f} ± {data['quality_score'].std():.2f}")
    
    # Create distribution plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Quality score distribution
    axes[0, 0].hist(data['quality_score'], bins=20, edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('Quality Score Distribution')
    axes[0, 0].set_xlabel('Quality Score')
    axes[0, 0].set_ylabel('Frequency')
    
    # Quality score vs iteration
    axes[0, 1].scatter(data['iteration'], data['quality_score'], alpha=0.6)
    axes[0, 1].set_title('Quality Score vs Iteration')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Quality Score')
    
    # Feature correlation heatmap (top correlations)
    corr_with_target = data[features].corrwith(data['quality_score']).abs().sort_values(ascending=False)
    top_features = corr_with_target.head(8).index.tolist()
    
    corr_matrix = data[top_features + ['quality_score']].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 0])
    axes[1, 0].set_title('Top 8 Features Correlation with Quality Score')
    
    # Feature importance preview
    axes[1, 1].barh(corr_with_target.head(10).index, corr_with_target.head(10).values)
    axes[1, 1].set_title('Top 10 Feature Correlations with Quality Score')
    axes[1, 1].set_xlabel('Absolute Correlation')
    
    plt.tight_layout()
    plt.savefig('data_exploration.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return corr_with_target

def correlation_analysis(data, features):
    """Detailed correlation analysis"""
    print("\n" + "="*50)
    print("CORRELATION ANALYSIS")
    print("="*50)
    
    correlations = []
    
    for feature in features:
        # Pearson correlation
        pearson_r, pearson_p = pearsonr(data[feature], data['quality_score'])
        
        # Spearman correlation (rank-based)
        spearman_r, spearman_p = spearmanr(data[feature], data['quality_score'])
        
        correlations.append({
            'feature': feature,
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'abs_pearson': abs(pearson_r),
            'abs_spearman': abs(spearman_r)
        })
    
    corr_df = pd.DataFrame(correlations)
    corr_df = corr_df.sort_values('abs_pearson', ascending=False)
    
    print("\nTop Features by Absolute Pearson Correlation:")
    print(corr_df[['feature', 'pearson_r', 'pearson_p', 'abs_pearson']].head(10))
    
    print("\nTop Features by Absolute Spearman Correlation:")
    corr_df_spearman = corr_df.sort_values('abs_spearman', ascending=False)
    print(corr_df_spearman[['feature', 'spearman_r', 'spearman_p', 'abs_spearman']].head(10))
    
    # Save correlation results
    corr_df.to_csv('correlation_analysis.csv', index=False)
    
    return corr_df

def feature_range_analysis(data, features):
    """Analyze feature ranges and quality score relationships"""
    print("\n" + "="*50)
    print("FEATURE RANGE ANALYSIS")
    print("="*50)
    
    # Find features with high correlation
    corr_with_target = data[features].corrwith(data['quality_score']).abs().sort_values(ascending=False)
    top_features = corr_with_target.head(6).index.tolist()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, feature in enumerate(top_features):
        # Scatter plot with trend line
        x = data[feature]
        y = data['quality_score']
        
        axes[i].scatter(x, y, alpha=0.6)
        
        # Add trend line
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        axes[i].plot(x, p(x), "r--", alpha=0.8)
        
        # Add correlation info
        corr = pearsonr(x, y)[0]
        axes[i].set_title(f'{feature}\nCorrelation: {corr:.3f}')
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('Quality Score')
    
    plt.tight_layout()
    plt.savefig('feature_relationships.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Analyze quality score ranges
    print("\nQuality Score Range Analysis:")
    
    # Divide quality scores into ranges
    data['quality_range'] = pd.cut(data['quality_score'], 
                                 bins=[0, 20, 40, 60, 80, 100], 
                                 labels=['Very Low (0-20)', 'Low (20-40)', 'Medium (40-60)', 
                                        'High (60-80)', 'Very High (80-100)'])
    
    range_analysis = data.groupby('quality_range').agg({
        'quality_score': ['count', 'mean', 'std'],
        **{feature: 'mean' for feature in top_features}
    }).round(2)
    
    print(range_analysis)
    
    return top_features

def main():
    """Main analysis function"""
    print("Cotton Candy Quality Score - Data Exploration")
    print("="*60)
    
    # Load data
    data, features = load_data()
    
    # Data distribution analysis
    corr_with_target = explore_data_distribution(data, features)
    
    # Correlation analysis
    corr_df = correlation_analysis(data, features)
    
    # Feature range analysis
    top_features = feature_range_analysis(data, features)
    
    # Summary insights
    print("\n" + "="*50)
    print("KEY INSIGHTS")
    print("="*50)
    
    print(f"\n1. Dataset contains {len(data)} samples with {len(features)} features")
    print(f"2. Quality scores range from {data['quality_score'].min():.1f} to {data['quality_score'].max():.1f}")
    print(f"3. Mean quality score: {data['quality_score'].mean():.2f} ± {data['quality_score'].std():.2f}")
    
    print(f"\n4. Top 5 most correlated features with quality score:")
    for i, (feature, corr) in enumerate(corr_with_target.head(5).items(), 1):
        print(f"   {i}. {feature}: {corr:.3f}")
    
    # Identify best quality samples
    best_quality = data.nlargest(5, 'quality_score')
    print(f"\n5. Best quality samples (top 5):")
    for _, row in best_quality.iterrows():
        print(f"   Iteration {row['iteration']}: Quality = {row['quality_score']:.2f}")
    
    # Identify worst quality samples
    worst_quality = data.nsmallest(5, 'quality_score')
    print(f"\n6. Worst quality samples (bottom 5):")
    for _, row in worst_quality.iterrows():
        print(f"   Iteration {row['iteration']}: Quality = {row['quality_score']:.2f}")
    
    print("\n" + "="*50)
    print("Files generated:")
    print("- data_exploration.png")
    print("- feature_relationships.png") 
    print("- correlation_analysis.csv")
    print("="*50)

if __name__ == "__main__":
    main()
