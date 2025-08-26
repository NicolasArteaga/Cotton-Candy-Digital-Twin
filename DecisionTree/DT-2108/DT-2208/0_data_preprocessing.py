#!/usr/bin/env python3
"""
Data Preprocessing for Full Dataset (DT-2208)
Handle missing values and prepare data for analysis
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
import matplotlib.pyplot as plt
import seaborn as sns

def get_feature_metadata():
    """Define feature categories and expected ranges based on documentation"""
    return {
        'core_process': {
            'features': ['iteration_since_maintenance', 'wait_time', 'cook_time', 'cooldown_time'],
            'units': ['count', 'seconds', 'seconds', 'seconds'],
            'expected_ranges': [(0, 60), (30, 110), (30, 115), (30, 120)]
        },
        'timing_metrics': {
            'features': ['duration_till_handover', 'duration_total', 'duration_cc_flow'],
            'units': ['seconds', 'seconds', 'seconds'],
            'expected_ranges': [(200, 400), (400, 600), (20, 150)]
        },
        'environmental_baseline': {
            'features': ['baseline_env_EnvH', 'baseline_env_EnvT'],
            'units': ['percentage', 'celsius'],
            'expected_ranges': [(30, 80), (15, 30)]
        },
        'internal_environmental': {
            'humidity_features': [f for f in ['before_turn_on_env_InH', 'after_flow_start_env_InH', 
                                            'after_flow_end_env_InH', 'before_cooldown_env_InH', 
                                            'after_cooldown_env_InH']],
            'temperature_features': [f for f in ['before_turn_on_env_InT', 'after_flow_start_env_InT',
                                               'after_flow_end_env_InT', 'before_cooldown_env_InT',
                                               'after_cooldown_env_InT']],
            'infrared_object_features': [f for f in ['before_turn_on_env_IrO', 'after_flow_start_env_IrO',
                                                    'after_flow_end_env_IrO', 'before_cooldown_env_IrO',
                                                    'after_cooldown_env_IrO']],
            'infrared_ambient_features': [f for f in ['before_turn_on_env_IrA', 'after_flow_start_env_IrA',
                                                     'after_flow_end_env_IrA', 'before_cooldown_env_IrA',
                                                     'after_cooldown_env_IrA']],
            'units': ['percentage', 'celsius', 'celsius', 'celsius'],
            'expected_ranges': [(20, 90), (15, 50), (20, 100), (20, 50)]
        }
    }

def load_and_examine_data():
    """Load data and examine missing values with feature context"""
    print("🔍 LOADING AND EXAMINING FULL DATASET")
    print("="*60)
    
    # Load data
    features = pd.read_csv('xy-full/features_X.csv')
    target = pd.read_csv('xy-full/target_y.csv')
    
    print(f"Original data shape:")
    print(f"Features: {features.shape}")
    print(f"Target: {target.shape}")
    
    # Get feature metadata
    metadata = get_feature_metadata()
    
    # Check for missing values
    print(f"\n📊 MISSING VALUES ANALYSIS:")
    missing_counts = features.isnull().sum()
    missing_percentages = (missing_counts / len(features)) * 100
    
    missing_summary = pd.DataFrame({
        'Feature': features.columns,
        'Missing_Count': missing_counts,
        'Missing_Percentage': missing_percentages
    }).sort_values('Missing_Percentage', ascending=False)
    
    print("Missing values by feature type:")
    print(missing_summary[missing_summary['Missing_Count'] > 0])
    
    # Analyze which sensors/phases have missing data
    print(f"\n🔬 MISSING DATA PATTERN ANALYSIS:")
    
    # Check environmental sensor phases
    phases = ['before_turn_on', 'after_flow_start', 'after_flow_end', 'before_cooldown', 'after_cooldown']
    sensors = ['InH', 'InT', 'IrO', 'IrA']
    
    for phase in phases:
        phase_features = [f'{phase}_env_{sensor}' for sensor in sensors]
        phase_missing = features[phase_features].isnull().sum().sum()
        if phase_missing > 0:
            print(f"  {phase}: {phase_missing} missing values across {len(phase_features)} sensors")
    
    # Data quality checks based on expected ranges
    print(f"\n🎯 DATA QUALITY VALIDATION:")
    quality_issues = []
    
    # Check core process parameters
    for i, feature in enumerate(metadata['core_process']['features']):
        if feature in features.columns:
            min_val, max_val = metadata['core_process']['expected_ranges'][i]
            unit = metadata['core_process']['units'][i]
            
            actual_min = features[feature].min()
            actual_max = features[feature].max()
            
            if actual_min < min_val or actual_max > max_val:
                quality_issues.append(f"{feature} ({unit}): Range {actual_min:.1f}-{actual_max:.1f}, Expected {min_val}-{max_val}")
    
    if quality_issues:
        print("⚠️  Data quality issues detected:")
        for issue in quality_issues:
            print(f"  {issue}")
    else:
        print("✅ All features within expected ranges")
    
    print(f"\n📋 FEATURE CATEGORIES SUMMARY:")
    all_features = []
    for category, info in metadata.items():
        if category == 'internal_environmental':
            category_features = (info['humidity_features'] + info['temperature_features'] + 
                                info['infrared_object_features'] + info['infrared_ambient_features'])
        else:
            category_features = info['features']
        
        all_features.extend(category_features)
        found_features = [f for f in category_features if f in features.columns]
        missing_features = [f for f in category_features if f not in features.columns]
        
        print(f"  {category}: {len(found_features)}/{len(category_features)} features present")
        if missing_features:
            print(f"    Missing: {missing_features}")
    
    return features, target, missing_summary
    
    # Visualize missing values pattern
    plt.figure(figsize=(15, 8))
    
    # Missing values heatmap
    plt.subplot(2, 2, 1)
    missing_matrix = features.isnull()
    sns.heatmap(missing_matrix, yticklabels=False, cbar=True, cmap='viridis')
    plt.title('Missing Values Pattern')
    plt.xlabel('Features')
    
    # Missing values by column
    plt.subplot(2, 2, 2)
    missing_by_col = features.isnull().sum().sort_values(ascending=False)
    missing_by_col = missing_by_col[missing_by_col > 0]
    if len(missing_by_col) > 0:
        missing_by_col.plot(kind='bar')
        plt.title('Missing Values by Feature')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
    
    # Missing values by row
    plt.subplot(2, 2, 3)
    missing_by_row = features.isnull().sum(axis=1)
    missing_by_row.hist(bins=20)
    plt.title('Distribution of Missing Values per Row')
    plt.xlabel('Number of Missing Values')
    plt.ylabel('Frequency')
    
    # Complete cases analysis
    plt.subplot(2, 2, 4)
    complete_cases = (~features.isnull()).all(axis=1).sum()
    incomplete_cases = len(features) - complete_cases
    
    plt.pie([complete_cases, incomplete_cases], 
            labels=[f'Complete Cases\n({complete_cases})', f'Incomplete Cases\n({incomplete_cases})'], 
            autopct='%1.1f%%')
    plt.title('Data Completeness')
    
    plt.tight_layout()
    plt.savefig('missing_values_analysis_full.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return features, target, missing_summary

def preprocess_data_strategies(features, target):
    """Apply different preprocessing strategies with feature-aware methods"""
    print(f"\n🔧 DATA PREPROCESSING STRATEGIES")
    print("="*60)
    
    strategies = {}
    metadata = get_feature_metadata()
    
    # Strategy 1: Complete Case Analysis (Remove rows with any missing values)
    print("Strategy 1: Complete Case Analysis (Remove rows with missing values)")
    complete_mask = features.notnull().all(axis=1)
    features_complete = features[complete_mask].copy()
    target_complete = target[complete_mask].copy()
    
    print(f"Remaining samples: {len(features_complete)} (removed {len(features) - len(features_complete)})")
    strategies['complete_case'] = (features_complete, target_complete)
    
    # Strategy 2: Smart sensor-based filtering
    print(f"\nStrategy 2: Smart Sensor-Based Preprocessing")
    
    # Identify core vs environmental features
    core_features = metadata['core_process']['features'] + metadata['timing_metrics']['features']
    baseline_features = metadata['environmental_baseline']['features']
    
    # Remove rows where core features are missing (these are critical)
    core_complete_mask = features[core_features].notnull().all(axis=1)
    features_smart = features[core_complete_mask].copy()
    target_smart = target[core_complete_mask].copy()
    
    print(f"After removing samples with missing core features: {len(features_smart)} samples")
    
    # For environmental sensors, use phase-based imputation
    # If an entire sensor phase is missing, remove those features
    phases = ['before_turn_on', 'after_flow_start', 'after_flow_end', 'before_cooldown', 'after_cooldown']
    sensors = ['InH', 'InT', 'IrO', 'IrA']
    
    features_to_keep = core_features + baseline_features
    
    for phase in phases:
        phase_features = [f'{phase}_env_{sensor}' for sensor in sensors]
        phase_available = [f for f in phase_features if f in features_smart.columns]
        
        # If more than 50% of phase measurements are available, keep the phase
        if len(phase_available) > 0:
            phase_missing_pct = features_smart[phase_available].isnull().sum().sum() / (len(features_smart) * len(phase_available))
            if phase_missing_pct < 0.5:  # Less than 50% missing
                features_to_keep.extend(phase_available)
                print(f"  Keeping {phase} phase: {len(phase_available)} features, {phase_missing_pct:.1%} missing")
            else:
                print(f"  Excluding {phase} phase: {phase_missing_pct:.1%} missing data")
    
    features_smart = features_smart[features_to_keep]
    print(f"Final feature count: {len(features_to_keep)} features")
    
    # Apply KNN imputation to remaining missing values
    if features_smart.isnull().sum().sum() > 0:
        knn_imputer = KNNImputer(n_neighbors=3)
        features_smart = pd.DataFrame(
            knn_imputer.fit_transform(features_smart),
            columns=features_smart.columns,
            index=features_smart.index
        )
        print(f"Applied KNN imputation for remaining missing values")
    
    strategies['smart_sensor'] = (features_smart, target_smart)
    
    # Strategy 3: Conservative approach - median imputation
    print(f"\nStrategy 3: Conservative Median Imputation")
    
    # Remove features with >40% missing data
    missing_percentages = (features.isnull().sum() / len(features)) * 100
    low_missing_features = features.columns[missing_percentages <= 40]
    
    features_conservative = features[low_missing_features].copy()
    print(f"Features after filtering high-missing: {len(low_missing_features)} (removed {len(features.columns) - len(low_missing_features)})")
    
    # Group features by type for targeted imputation
    median_imputer = SimpleImputer(strategy='median')
    features_conservative = pd.DataFrame(
        median_imputer.fit_transform(features_conservative),
        columns=features_conservative.columns,
        index=features_conservative.index
    )
    
    strategies['conservative_median'] = (features_conservative, target)
    
    # Strategy 4: Advanced feature engineering
    print(f"\nStrategy 4: Feature Engineering + Imputation")
    
    # Start with smart sensor approach
    features_engineered = features_smart.copy()
    
    # Add derived features that might be useful
    if all(f in features_engineered.columns for f in ['wait_time', 'cook_time', 'cooldown_time']):
        features_engineered['total_control_time'] = (features_engineered['wait_time'] + 
                                                    features_engineered['cook_time'] + 
                                                    features_engineered['cooldown_time'])
        print(f"  Added total_control_time feature")
    
    if all(f in features_engineered.columns for f in ['duration_cc_flow', 'cook_time']):
        # Handle division by zero and create safe ratio
        cook_time_safe = features_engineered['cook_time'].replace(0, 0.01)  # Replace 0 with small value
        efficiency_ratio = features_engineered['duration_cc_flow'] / cook_time_safe
        # Cap extreme values
        efficiency_ratio = np.clip(efficiency_ratio, 0, 10)
        features_engineered['efficiency_ratio'] = efficiency_ratio
        print(f"  Added efficiency_ratio feature (safely handled division by zero)")
    
    # Add temperature stability metrics if we have multiple temperature readings
    temp_features = [f for f in features_engineered.columns if '_InT' in f]
    if len(temp_features) >= 2:
        temp_stability = features_engineered[temp_features].std(axis=1)
        # Handle any infinite or NaN values
        temp_stability = temp_stability.fillna(0)
        temp_stability = np.clip(temp_stability, 0, 50)  # Cap extreme values
        features_engineered['temp_stability'] = temp_stability
        print(f"  Added temperature stability feature from {len(temp_features)} sensors")
    
    strategies['feature_engineering'] = (features_engineered, target_smart)
    
    # Summary
    print(f"\n📊 PREPROCESSING SUMMARY:")
    for strategy_name, (X, y) in strategies.items():
        quality_score = 100 - (X.isnull().sum().sum() / (X.shape[0] * X.shape[1]) * 100)
        print(f"{strategy_name:20} -> Samples: {X.shape[0]:2d}, Features: {X.shape[1]:2d}, Quality: {quality_score:.1f}%")
    
    return strategies

def evaluate_preprocessing_impact(strategies):
    """Evaluate the impact of different preprocessing strategies"""
    print(f"\n🧪 PREPROCESSING STRATEGY EVALUATION")
    print("="*60)
    
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    
    results = {}
    
    for strategy_name, (X, y) in strategies.items():
        print(f"\nEvaluating: {strategy_name}")
        
        # Align target with features (in case of complete case analysis)
        if len(X) != len(y):
            y_aligned = y.loc[X.index]
        else:
            y_aligned = y['quality_score'] if 'quality_score' in y.columns else y
        
        # Quick model evaluation
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        
        try:
            # Cross-validation
            scores = cross_val_score(rf, X, y_aligned, cv=5, scoring='r2')
            results[strategy_name] = {
                'samples': len(X),
                'features': X.shape[1],
                'cv_r2_mean': scores.mean(),
                'cv_r2_std': scores.std(),
                'data_completeness': (1 - X.isnull().sum().sum() / (X.shape[0] * X.shape[1])) * 100
            }
            
            print(f"  Samples: {len(X)}")
            print(f"  Features: {X.shape[1]}")
            print(f"  CV R²: {scores.mean():.3f} ± {scores.std():.3f}")
            print(f"  Data Completeness: {results[strategy_name]['data_completeness']:.1f}%")
            
        except Exception as e:
            print(f"  Error: {e}")
            results[strategy_name] = {
                'samples': len(X),
                'features': X.shape[1],
                'cv_r2_mean': np.nan,
                'cv_r2_std': np.nan,
                'data_completeness': (1 - X.isnull().sum().sum() / (X.shape[0] * X.shape[1])) * 100
            }
    
    # Create evaluation visualization
    plt.figure(figsize=(15, 10))
    
    # R² scores comparison
    plt.subplot(2, 3, 1)
    strategy_names = list(results.keys())
    r2_scores = [results[name]['cv_r2_mean'] for name in strategy_names]
    r2_stds = [results[name]['cv_r2_std'] for name in strategy_names]
    
    bars = plt.bar(range(len(strategy_names)), r2_scores, yerr=r2_stds, capsize=5)
    plt.xlabel('Preprocessing Strategy')
    plt.ylabel('Cross-Validation R²')
    plt.title('Model Performance by Strategy')
    plt.xticks(range(len(strategy_names)), strategy_names, rotation=45, ha='right')
    
    # Color bars by performance
    for i, bar in enumerate(bars):
        if not np.isnan(r2_scores[i]):
            if r2_scores[i] > 0.6:
                bar.set_color('green')
            elif r2_scores[i] > 0.4:
                bar.set_color('orange')
            else:
                bar.set_color('red')
    
    # Sample size comparison
    plt.subplot(2, 3, 2)
    sample_sizes = [results[name]['samples'] for name in strategy_names]
    plt.bar(range(len(strategy_names)), sample_sizes)
    plt.xlabel('Strategy')
    plt.ylabel('Number of Samples')
    plt.title('Sample Size by Strategy')
    plt.xticks(range(len(strategy_names)), strategy_names, rotation=45, ha='right')
    
    # Feature count comparison
    plt.subplot(2, 3, 3)
    feature_counts = [results[name]['features'] for name in strategy_names]
    plt.bar(range(len(strategy_names)), feature_counts)
    plt.xlabel('Strategy')
    plt.ylabel('Number of Features')
    plt.title('Feature Count by Strategy')
    plt.xticks(range(len(strategy_names)), strategy_names, rotation=45, ha='right')
    
    # Data completeness
    plt.subplot(2, 3, 4)
    completeness = [results[name]['data_completeness'] for name in strategy_names]
    plt.bar(range(len(strategy_names)), completeness)
    plt.xlabel('Strategy')
    plt.ylabel('Data Completeness (%)')
    plt.title('Data Completeness by Strategy')
    plt.xticks(range(len(strategy_names)), strategy_names, rotation=45, ha='right')
    plt.ylim([90, 100])
    
    # Performance vs Sample Size
    plt.subplot(2, 3, 5)
    valid_indices = [i for i, score in enumerate(r2_scores) if not np.isnan(score)]
    if valid_indices:
        plt.scatter([sample_sizes[i] for i in valid_indices], 
                   [r2_scores[i] for i in valid_indices])
        for i in valid_indices:
            plt.annotate(strategy_names[i], 
                        (sample_sizes[i], r2_scores[i]),
                        xytext=(5, 5), textcoords='offset points')
        plt.xlabel('Sample Size')
        plt.ylabel('CV R²')
        plt.title('Performance vs Sample Size')
    
    # Summary table
    plt.subplot(2, 3, 6)
    plt.axis('off')
    
    # Create summary table
    summary_data = []
    for name in strategy_names:
        r = results[name]
        summary_data.append([
            name,
            f"{r['samples']}",
            f"{r['features']}",
            f"{r['cv_r2_mean']:.3f}" if not np.isnan(r['cv_r2_mean']) else "N/A",
            f"{r['data_completeness']:.1f}%"
        ])
    
    table = plt.table(cellText=summary_data,
                     colLabels=['Strategy', 'Samples', 'Features', 'CV R²', 'Completeness'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    plt.title('Strategy Comparison Summary')
    
    plt.tight_layout()
    plt.savefig('preprocessing_evaluation_full.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return results

def select_best_strategy(results, strategies):
    """Select the best preprocessing strategy"""
    print(f"\n🏆 BEST STRATEGY SELECTION")
    print("="*60)
    
    # Score strategies based on multiple criteria
    strategy_scores = {}
    
    for name, metrics in results.items():
        if np.isnan(metrics['cv_r2_mean']):
            strategy_scores[name] = -999  # Penalize failed strategies
            continue
            
        # Weighted scoring: R² (60%), Sample size (20%), Features (10%), Completeness (10%)
        r2_normalized = metrics['cv_r2_mean']  # Already 0-1
        sample_normalized = min(metrics['samples'] / 100, 1.0)  # Cap at 100 samples = 1.0
        feature_normalized = min(metrics['features'] / 30, 1.0)  # Cap at 30 features = 1.0
        completeness_normalized = metrics['data_completeness'] / 100
        
        score = (0.6 * r2_normalized + 
                0.2 * sample_normalized + 
                0.1 * feature_normalized + 
                0.1 * completeness_normalized)
        
        strategy_scores[name] = score
    
    # Find best strategy
    best_strategy = max(strategy_scores, key=strategy_scores.get)
    
    print(f"Strategy Scores:")
    for name, score in sorted(strategy_scores.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name:20}: {score:.3f}")
    
    print(f"\n🥇 BEST STRATEGY: {best_strategy}")
    print(f"Performance Score: {strategy_scores[best_strategy]:.3f}")
    
    # Get best data
    best_X, best_y = strategies[best_strategy]
    
    if len(best_X) != len(best_y):
        best_y = best_y.loc[best_X.index]
    
    return best_strategy, best_X, best_y

def save_processed_data(best_strategy, best_X, best_y):
    """Save the processed data"""
    print(f"\n💾 SAVING PROCESSED DATA")
    print("="*60)
    
    # Save processed features
    best_X.to_csv('processed_features_X.csv', index=False)
    print(f"✅ Saved processed features: processed_features_X.csv")
    print(f"   Shape: {best_X.shape}")
    
    # Save aligned target
    if 'quality_score' in best_y.columns:
        target_series = best_y['quality_score']
        best_y[['quality_score']].to_csv('processed_target_y.csv', index=False)
    else:
        target_series = best_y
        pd.DataFrame({'quality_score': best_y}).to_csv('processed_target_y.csv', index=False)
    
    print(f"✅ Saved processed target: processed_target_y.csv")
    
    # Save preprocessing summary
    summary = f"""# Data Preprocessing Summary - Full Dataset (DT-2208)

## Selected Strategy: {best_strategy}

### Final Dataset:
- **Samples**: {len(best_X)}
- **Features**: {best_X.shape[1]}
- **Missing Values**: {best_X.isnull().sum().sum()}
- **Data Completeness**: {((1 - best_X.isnull().sum().sum() / (best_X.shape[0] * best_X.shape[1])) * 100):.1f}%

### Feature List:
"""
    
    for i, feature in enumerate(best_X.columns):
        summary += f"{i+1:2d}. {feature}\n"
    
    summary += f"""
### Quality Scores:
- **Range**: {target_series.min():.2f} to {target_series.max():.2f}
- **Mean**: {target_series.mean():.2f}
- **Std**: {target_series.std():.2f}

### Preprocessing Applied:
- Strategy: {best_strategy}
- Missing value handling: Applied based on strategy selection
- Feature engineering: Included if applicable
- Data validation: Completed successfully

### Files Generated:
- `processed_features_X.csv`: Clean feature matrix
- `processed_target_y.csv`: Quality scores aligned with features
- `missing_values_analysis_full.png`: Missing values visualization
- `preprocessing_evaluation_full.png`: Strategy comparison
- `PREPROCESSING_SUMMARY.md`: This summary document

Ready for machine learning analysis!
"""
    
    with open('PREPROCESSING_SUMMARY.md', 'w') as f:
        f.write(summary)
    
    print(f"✅ Saved preprocessing summary: PREPROCESSING_SUMMARY.md")

def main():
    """Main preprocessing pipeline"""
    print("🧹 FULL DATASET PREPROCESSING PIPELINE")
    print("="*80)
    print("This will clean and prepare the 29-feature dataset for analysis")
    print("="*80)
    
    # Load and examine data
    features, target, missing_summary = load_and_examine_data()
    
    # Apply preprocessing strategies
    strategies = preprocess_data_strategies(features, target)
    
    # Evaluate strategies
    results = evaluate_preprocessing_impact(strategies)
    
    # Select best strategy
    best_strategy, best_X, best_y = select_best_strategy(results, strategies)
    
    # Save processed data
    save_processed_data(best_strategy, best_X, best_y)
    
    print(f"\n🎉 PREPROCESSING COMPLETE!")
    print(f"Best strategy: {best_strategy}")
    print(f"Final dataset: {best_X.shape[0]} samples × {best_X.shape[1]} features")
    print(f"Ready for analysis with clean, complete data!")

if __name__ == "__main__":
    main()
