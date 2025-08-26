#!/usr/bin/env python3
"""
Comparison Analysis: Full Dataset (29 features) vs Basic Dataset (10 features)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def compare_feature_rankings():
    """Compare feature rankings between full and basic datasets"""
    print("🔍 COMPARISON: Full Dataset (29 features) vs Basic Dataset (10 features)")
    print("="*80)
    
    try:
        # Load full dataset results
        full_ranking = pd.read_csv('ensemble_feature_ranking_full.csv')
        print(f"✅ Loaded full dataset results: {len(full_ranking)} features")
        
        # Try to load basic dataset results
        basic_ranking = pd.read_csv('../ensemble_feature_ranking.csv')
        print(f"✅ Loaded basic dataset results: {len(basic_ranking)} features")
        
        print(f"\n📊 TOP 10 FEATURES COMPARISON:")
        print(f"{'Rank':<4} {'Full Dataset (29 features)':<35} {'Basic Dataset (10 features)':<35}")
        print("-" * 80)
        
        for i in range(min(10, len(full_ranking), len(basic_ranking))):
            full_feature = full_ranking.iloc[i]['feature'][:30] + "..." if len(full_ranking.iloc[i]['feature']) > 30 else full_ranking.iloc[i]['feature']
            basic_feature = basic_ranking.iloc[i]['feature'][:30] + "..." if len(basic_ranking.iloc[i]['feature']) > 30 else basic_ranking.iloc[i]['feature']
            print(f"{i+1:<4} {full_feature:<35} {basic_feature:<35}")
        
        # Find common features
        full_features = set(full_ranking['feature'].tolist())
        basic_features = set(basic_ranking['feature'].tolist())
        common_features = full_features.intersection(basic_features)
        
        print(f"\n🔗 COMMON FEATURES ANALYSIS:")
        print(f"Features in both datasets: {len(common_features)} out of {len(basic_features)} basic features")
        
        if common_features:
            print(f"\nCommon features ranking comparison:")
            print(f"{'Feature':<30} {'Full Rank':<10} {'Basic Rank':<12} {'Difference':<10}")
            print("-" * 65)
            
            for feature in list(common_features)[:10]:  # Top 10 common
                full_rank = full_ranking[full_ranking['feature'] == feature]['avg_rank'].iloc[0] if len(full_ranking[full_ranking['feature'] == feature]) > 0 else 999
                basic_rank = basic_ranking[basic_ranking['feature'] == feature]['avg_rank'].iloc[0] if len(basic_ranking[basic_ranking['feature'] == feature]) > 0 else 999
                
                diff = abs(full_rank - basic_rank)
                feature_short = feature[:25] + "..." if len(feature) > 25 else feature
                print(f"{feature_short:<30} {full_rank:<10.1f} {basic_rank:<12.1f} {diff:<10.1f}")
        
        print(f"\n💡 KEY INSIGHTS:")
        print(f"1. Full dataset provides {len(full_ranking) - len(basic_ranking)} additional features")
        print(f"2. Common features may have different rankings due to feature interactions")
        print(f"3. New features in full dataset may reveal hidden quality drivers")
        print(f"4. Model performance likely improved with extended feature set")
        
    except FileNotFoundError as e:
        print(f"⚠️  Could not find ranking files for comparison: {e}")
        print("Run both analyses first to enable comparison")
    except Exception as e:
        print(f"❌ Error during comparison: {e}")

def main():
    compare_feature_rankings()

if __name__ == "__main__":
    main()
