#!/usr/bin/env python3
"""
Data Exploration and Analysis - Full Dataset
Comprehensive exploration of cotton candy quality features and targets (29 features)
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
    """Load the preprocessed cotton candy dataset"""
    print("📥 Loading preprocessed cotton candy dataset...")
    
    # Load the cleaned data
    features = pd.read_csv('processed_features_X.csv')
    target = pd.read_csv('processed_target_y.csv')['quality_score']
    
    print(f"Dataset shape: {features.shape[0]} samples × {features.shape[1]} features")
    print(f"Quality score range: {target.min():.1f} to {target.max():.1f}")
    
    return features, target, features.columns.tolist()

def explore_data_distribution(data, features):
    """Explore data distribution and basic statistics"""
    print("\n" + "="*50)
    print("DATA DISTRIBUTION ANALYSIS - FULL DATASET")
    print("="*50)
    
    # Basic statistics
    print("\nBasic Statistics:")
    print(data[features[:10] + ['quality_score']].describe())  # Show first 10 features + target
    
    # Quality score distribution
    print(f"\nQuality Score Range: {data['quality_score'].min():.2f} - {data['quality_score'].max():.2f}")
    print(f"Quality Score Mean: {data['quality_score'].mean():.2f} ± {data['quality_score'].std():.2f}")
    
    # Create distribution plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Quality score distribution
    axes[0, 0].hist(data['quality_score'], bins=20, edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('Quality Score Distribution (Full Dataset)')
    axes[0, 0].set_xlabel('Quality Score')
    axes[0, 0].set_ylabel('Frequency')
    
    # Quality score vs iteration since maintenance
    axes[0, 1].scatter(data['iteration_since_maintenance'], data['quality_score'], alpha=0.6)
    axes[0, 1].set_title('Quality Score vs Iteration Since Maintenance')
    axes[0, 1].set_xlabel('Iteration Since Maintenance')
    axes[0, 1].set_ylabel('Quality Score')
    
    # Feature correlation heatmap (top correlations)
    corr_with_target = data[features].corrwith(data['quality_score']).abs().sort_values(ascending=False)
    top_features = corr_with_target.head(8).index.tolist()
    
    corr_matrix = data[top_features + ['quality_score']].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 0], fmt='.2f')
    axes[1, 0].set_title('Top 8 Features Correlation with Quality Score')
    
    # Feature importance preview
    axes[1, 1].barh(corr_with_target.head(15).index, corr_with_target.head(15).values)
    axes[1, 1].set_title('Top 15 Feature Correlations with Quality Score')
    axes[1, 1].set_xlabel('Absolute Correlation')
    
    plt.tight_layout()
    plt.savefig('data_exploration_full.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return corr_with_target

def correlation_analysis(data, features):
    """Detailed correlation analysis"""
    print("\n" + "="*50)
    print("CORRELATION ANALYSIS - FULL DATASET")
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
    
    print("\nTop 15 Features by Absolute Pearson Correlation:")
    print(corr_df[['feature', 'pearson_r', 'pearson_p', 'abs_pearson']].head(15))
    
    print("\nTop 15 Features by Absolute Spearman Correlation:")
    corr_df_spearman = corr_df.sort_values('abs_spearman', ascending=False)
    print(corr_df_spearman[['feature', 'spearman_r', 'spearman_p', 'abs_spearman']].head(15))
    
    # Save correlation results
    corr_df.to_csv('correlation_analysis_full.csv', index=False)
    
    return corr_df

def feature_range_analysis(data, features):
    """Analyze feature ranges and quality score relationships"""
    print("\n" + "="*50)
    print("FEATURE RANGE ANALYSIS - FULL DATASET")
    print("="*50)
    
    # Find features with high correlation
    corr_with_target = data[features].corrwith(data['quality_score']).abs().sort_values(ascending=False)
    top_features = corr_with_target.head(12).index.tolist()
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
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
        axes[i].set_title(f'{feature[:20]}...\nCorr: {corr:.3f}', fontsize=10)
        axes[i].set_xlabel(feature, fontsize=8)
        axes[i].set_ylabel('Quality Score', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('feature_relationships_full.png', dpi=300, bbox_inches='tight')
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
        **{feature: 'mean' for feature in top_features[:6]}  # Only top 6 to avoid clutter
    }).round(2)
    
    print(range_analysis)
    
    return top_features

def feature_categories_analysis(features):
    """Categorize features for better understanding"""
    print("\n" + "="*50)
    print("FEATURE CATEGORIES ANALYSIS")
    print("="*50)
    
    categories = {
        'Process Parameters': [],
        'Environmental Baseline': [],
        'Pre-Process Environment': [],
        'During Process': [],
        'Post Process': [],
        'Weight Measurements': [],
        'System State': [],
        'Quality Metrics': [],
        'Other': []
    }
    
    for feature in features:
        if any(x in feature.lower() for x in ['wait', 'cook', 'duration', 'flow']):
            categories['Process Parameters'].append(feature)
        elif 'baseline_env' in feature.lower():
            categories['Environmental Baseline'].append(feature)
        elif 'before_turn_on' in feature.lower():
            categories['Pre-Process Environment'].append(feature)
        elif any(x in feature.lower() for x in ['during', 'active']):
            categories['During Process'].append(feature)
        elif any(x in feature.lower() for x in ['after', 'end', 'final']):
            categories['Post Process'].append(feature)
        elif any(x in feature.lower() for x in ['weight', 'mass', 'gram']):
            categories['Weight Measurements'].append(feature)
        elif any(x in feature.lower() for x in ['iteration', 'maintenance', 'state']):
            categories['System State'].append(feature)
        elif any(x in feature.lower() for x in ['quality', 'score', 'rating']):
            categories['Quality Metrics'].append(feature)
        else:
            categories['Other'].append(feature)
    
    print("Feature Categories:")
    for category, feature_list in categories.items():
        if feature_list:
            print(f"\n{category} ({len(feature_list)} features):")
            for feature in feature_list[:5]:  # Show first 5
                print(f"  - {feature}")
            if len(feature_list) > 5:
                print(f"  ... and {len(feature_list) - 5} more")
    
    return categories

def main():
    """Main analysis function"""
    print("Cotton Candy Quality Score - Data Exploration (Full Dataset)")
    print("="*70)
    
    # Load data
    features, target, feature_names = load_data()
    
    # Combine features and target into single dataframe for compatibility
    data = features.copy()
    data['quality_score'] = target
    
    # Feature categories analysis
    feature_categories = feature_categories_analysis(feature_names)
    
    # Data distribution analysis
    corr_with_target = explore_data_distribution(data, feature_names)
    
    # Correlation analysis
    corr_df = correlation_analysis(data, feature_names)
    
    # Feature range analysis
    top_features = feature_range_analysis(data, feature_names)
    
    # Summary insights
    print("\n" + "="*60)
    print("KEY INSIGHTS - FULL DATASET")
    print("="*60)
    
    print(f"\n1. Dataset contains {len(data)} samples with {len(features)} features")
    print(f"2. Quality scores range from {data['quality_score'].min():.1f} to {data['quality_score'].max():.1f}")
    print(f"3. Mean quality score: {data['quality_score'].mean():.2f} ± {data['quality_score'].std():.2f}")
    
    print(f"\n4. Top 10 most correlated features with quality score:")
    for i, (feature, corr) in enumerate(corr_with_target.head(10).items(), 1):
        print(f"   {i:2d}. {feature[:40]:40}: {corr:.3f}")
    
    # Identify best quality samples
    best_quality = data.nlargest(5, 'quality_score')
    print(f"\n5. Best quality samples (top 5):")
    for _, row in best_quality.iterrows():
        print(f"   Maintenance Cycle {row['iteration_since_maintenance']}: Quality = {row['quality_score']:.2f}")
    
    # Identify worst quality samples
    worst_quality = data.nsmallest(5, 'quality_score')
    print(f"\n6. Worst quality samples (bottom 5):")
    for _, row in worst_quality.iterrows():
        print(f"   Maintenance Cycle {row['iteration_since_maintenance']}: Quality = {row['quality_score']:.2f}")
    
    # Feature completeness analysis
    print(f"\n7. Feature completeness:")
    missing_counts = data[feature_names].isnull().sum()
    if missing_counts.sum() > 0:
        print("   Features with missing values:")
        for feature, count in missing_counts[missing_counts > 0].items():
            print(f"     {feature}: {count} missing values")
    else:
        print("   ✅ No missing values found in any feature")
    
    print("\n" + "="*60)
    print("Files generated:")
    print("- data_exploration_full.png")
    print("- feature_relationships_full.png") 
    print("- correlation_analysis_full.csv")
    print("="*60)

if __name__ == "__main__":
    main()
