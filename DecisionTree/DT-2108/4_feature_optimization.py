#!/usr/bin/env python3
"""
Feature Effect Analysis and Quality Optimization
Analyze how changing individual features affects quality score and provide optimization guidelines
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class FeatureEffectAnalyzer:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.feature_stats = {}
        self.optimization_results = {}
        
    def load_data(self):
        """Load and prepare data"""
        print("Loading data...")
        features_df = pd.read_csv('xy/features_X.csv')
        targets_df = pd.read_csv('xy/targets_Y.csv')
        
        # Merge datasets
        data = pd.merge(features_df, targets_df, on='iteration', how='inner')
        
        # Prepare features and target
        self.feature_names = features_df.columns[1:].tolist()  # Exclude 'iteration'
        X = data[self.feature_names]
        y = data['quality_score']
        
        # Store feature statistics
        for feature in self.feature_names:
            self.feature_stats[feature] = {
                'min': X[feature].min(),
                'max': X[feature].max(),
                'mean': X[feature].mean(),
                'std': X[feature].std(),
                'q25': X[feature].quantile(0.25),
                'q75': X[feature].quantile(0.75)
            }
        
        print(f"Data shape: {X.shape}")
        print(f"Target range: {y.min():.2f} - {y.max():.2f}")
        
        return X, y, data
    
    def train_best_model(self, X, y):
        """Train the best performing model for analysis"""
        print("\nTraining best model for feature effect analysis...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Try both RandomForest and GradientBoosting
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
        }
        
        best_score = -np.inf
        best_model = None
        best_name = None
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = r2_score(y_test, y_pred)
            
            print(f"{name} - R²: {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_model = model
                best_name = name
        
        self.model = best_model
        print(f"\nSelected {best_name} as best model (R² = {best_score:.4f})")
        
        return best_model, best_name, best_score
    
    def analyze_feature_effects(self, X, y):
        """Analyze how each feature affects quality score"""
        print("\n" + "="*60)
        print("INDIVIDUAL FEATURE EFFECT ANALYSIS")
        print("="*60)
        
        feature_effects = []
        
        for feature in self.feature_names:
            print(f"\nAnalyzing {feature}...")
            
            # Get feature statistics
            feature_min = self.feature_stats[feature]['min']
            feature_max = self.feature_stats[feature]['max']
            feature_mean = self.feature_stats[feature]['mean']
            
            # Create test points across feature range
            test_points = np.linspace(feature_min, feature_max, 50)
            
            # Use mean values for all other features
            baseline_sample = X.mean().values
            feature_idx = self.feature_names.index(feature)
            
            predictions = []
            for test_value in test_points:
                sample = baseline_sample.copy()
                sample[feature_idx] = test_value
                pred = self.model.predict(sample.reshape(1, -1))[0]
                predictions.append(pred)
            
            predictions = np.array(predictions)
            
            # Calculate effect metrics
            max_pred = np.max(predictions)
            min_pred = np.min(predictions)
            effect_range = max_pred - min_pred
            
            # Find optimal value
            optimal_idx = np.argmax(predictions)
            optimal_value = test_points[optimal_idx]
            optimal_quality = predictions[optimal_idx]
            
            # Calculate correlation with actual data
            correlation, p_value = stats.pearsonr(X[feature], y)
            
            # Store results
            feature_effects.append({
                'feature': feature,
                'effect_range': effect_range,
                'optimal_value': optimal_value,
                'optimal_quality': optimal_quality,
                'correlation': correlation,
                'p_value': p_value,
                'min_prediction': min_pred,
                'max_prediction': max_pred,
                'current_mean': feature_mean,
                'test_points': test_points,
                'predictions': predictions
            })
            
            print(f"   Effect range: {effect_range:.2f} quality points")
            print(f"   Optimal value: {optimal_value:.2f}")
            print(f"   Optimal quality: {optimal_quality:.2f}")
            print(f"   Current mean: {feature_mean:.2f}")
            print(f"   Correlation: {correlation:.3f} (p={p_value:.3f})")
        
        # Convert to DataFrame and sort by effect range
        effects_df = pd.DataFrame([{k: v for k, v in effect.items() 
                                  if k not in ['test_points', 'predictions']} 
                                 for effect in feature_effects])
        effects_df = effects_df.sort_values('effect_range', ascending=False)
        
        print(f"\nTop 10 Features by Effect Range:")
        print(effects_df[['feature', 'effect_range', 'optimal_value', 'optimal_quality', 'correlation']].head(10))
        
        # Save results
        effects_df.to_csv('feature_effects_analysis.csv', index=False)
        
        return feature_effects, effects_df
    
    def create_feature_effect_plots(self, feature_effects, top_n=8):
        """Create plots showing feature effects"""
        print(f"\nCreating feature effect plots...")
        
        # Sort by effect range and take top features
        sorted_effects = sorted(feature_effects, key=lambda x: x['effect_range'], reverse=True)
        top_effects = sorted_effects[:top_n]
        
        # Create subplots
        fig, axes = plt.subplots(2, 4, figsize=(20, 12))
        axes = axes.flatten()
        
        for i, effect in enumerate(top_effects):
            ax = axes[i]
            
            # Plot feature effect curve
            ax.plot(effect['test_points'], effect['predictions'], 
                   'b-', linewidth=2, label='Predicted Quality')
            
            # Mark optimal point
            ax.scatter([effect['optimal_value']], [effect['optimal_quality']], 
                      color='red', s=100, zorder=5, label=f'Optimal: {effect["optimal_value"]:.1f}')
            
            # Mark current mean
            current_pred = np.interp(effect['current_mean'], effect['test_points'], effect['predictions'])
            ax.scatter([effect['current_mean']], [current_pred], 
                      color='orange', s=80, zorder=5, label=f'Current: {effect["current_mean"]:.1f}')
            
            ax.set_xlabel(effect['feature'])
            ax.set_ylabel('Quality Score')
            ax.set_title(f'{effect["feature"]}\nEffect Range: {effect["effect_range"]:.1f}')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            
            # Add correlation info
            ax.text(0.02, 0.98, f'r = {effect["correlation"]:.3f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, pad=0.2))
        
        plt.tight_layout()
        plt.savefig('feature_effect_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def multi_feature_optimization(self, X, y, top_features=5):
        """Optimize multiple features simultaneously"""
        print(f"\n" + "="*60)
        print(f"MULTI-FEATURE OPTIMIZATION (Top {top_features} features)")
        print("="*60)
        
        # Get feature importance from trained model
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_importance = list(zip(self.feature_names, importances))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            # Select top features for optimization
            selected_features = [f[0] for f in feature_importance[:top_features]]
        else:
            # Fallback: use correlation
            correlations = [abs(stats.pearsonr(X[f], y)[0]) for f in self.feature_names]
            feature_corr = list(zip(self.feature_names, correlations))
            feature_corr.sort(key=lambda x: x[1], reverse=True)
            selected_features = [f[0] for f in feature_corr[:top_features]]
        
        print(f"Selected features for optimization: {selected_features}")
        
        # Define optimization function
        def objective(x):
            # Create sample with optimized features and mean values for others
            sample = X.mean().values.copy()
            for i, feature in enumerate(selected_features):
                feature_idx = self.feature_names.index(feature)
                sample[feature_idx] = x[i]
            
            # Predict quality (negative because we want to maximize)
            pred = self.model.predict(sample.reshape(1, -1))[0]
            return -pred
        
        # Set bounds for optimization (within observed ranges)
        bounds = []
        initial_guess = []
        for feature in selected_features:
            stats = self.feature_stats[feature]
            # Use 10th to 90th percentile as bounds to avoid extreme values
            lower_bound = stats['q25'] - (stats['q75'] - stats['q25']) * 0.5
            upper_bound = stats['q75'] + (stats['q75'] - stats['q25']) * 0.5
            
            # Ensure bounds are within actual data range
            lower_bound = max(lower_bound, stats['min'])
            upper_bound = min(upper_bound, stats['max'])
            
            bounds.append((lower_bound, upper_bound))
            initial_guess.append(stats['mean'])
        
        # Run optimization
        print("\nRunning optimization...")
        result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
        
        if result.success:
            optimal_values = result.x
            optimal_quality = -result.fun
            
            print(f"\nOptimization Results:")
            print(f"Predicted Optimal Quality: {optimal_quality:.2f}")
            print(f"Improvement over current mean: {optimal_quality - y.mean():.2f}")
            
            print(f"\nOptimal Feature Values:")
            optimization_recommendations = []
            for i, (feature, optimal_val) in enumerate(zip(selected_features, optimal_values)):
                current_val = self.feature_stats[feature]['mean']
                change = optimal_val - current_val
                change_pct = (change / current_val) * 100 if current_val != 0 else 0
                
                optimization_recommendations.append({
                    'feature': feature,
                    'current_value': current_val,
                    'optimal_value': optimal_val,
                    'change': change,
                    'change_percent': change_pct
                })
                
                print(f"   {feature}: {optimal_val:.2f} (current: {current_val:.2f}, change: {change:+.2f}, {change_pct:+.1f}%)")
            
            # Save optimization results
            opt_df = pd.DataFrame(optimization_recommendations)
            opt_df.to_csv('optimization_recommendations.csv', index=False)
            
            return optimization_recommendations, optimal_quality
        else:
            print("Optimization failed!")
            return None, None
    
    def sensitivity_analysis(self, X, y, feature_effects):
        """Perform sensitivity analysis to understand feature interactions"""
        print(f"\n" + "="*60)
        print("SENSITIVITY ANALYSIS")
        print("="*60)
        
        # Get top 5 most impactful features
        sorted_effects = sorted(feature_effects, key=lambda x: x['effect_range'], reverse=True)
        top_5_features = [effect['feature'] for effect in sorted_effects[:5]]
        
        print(f"Analyzing sensitivity for: {top_5_features}")
        
        sensitivity_results = []
        baseline_sample = X.mean().values
        baseline_pred = self.model.predict(baseline_sample.reshape(1, -1))[0]
        
        for feature in top_5_features:
            feature_idx = self.feature_names.index(feature)
            stats = self.feature_stats[feature]
            
            # Test different percentage changes
            changes = [-20, -10, -5, 5, 10, 20]  # Percentage changes
            
            for change_pct in changes:
                test_sample = baseline_sample.copy()
                
                # Calculate new value with percentage change
                current_val = stats['mean']
                new_val = current_val * (1 + change_pct / 100.0)
                
                # Ensure within bounds
                new_val = max(stats['min'], min(stats['max'], new_val))
                
                test_sample[feature_idx] = new_val
                new_pred = self.model.predict(test_sample.reshape(1, -1))[0]
                
                quality_change = new_pred - baseline_pred
                
                sensitivity_results.append({
                    'feature': feature,
                    'change_percent': change_pct,
                    'original_value': current_val,
                    'new_value': new_val,
                    'baseline_quality': baseline_pred,
                    'new_quality': new_pred,
                    'quality_change': quality_change,
                    'sensitivity': quality_change / abs(change_pct) if change_pct != 0 else 0
                })
        
        # Convert to DataFrame
        sensitivity_df = pd.DataFrame(sensitivity_results)
        
        # Calculate average sensitivity per feature
        avg_sensitivity = sensitivity_df.groupby('feature')['sensitivity'].mean().abs().sort_values(ascending=False)
        
        print(f"\nAverage Sensitivity (Quality change per % feature change):")
        for feature, sens in avg_sensitivity.items():
            print(f"   {feature}: {sens:.3f}")
        
        # Save results
        sensitivity_df.to_csv('sensitivity_analysis.csv', index=False)
        
        return sensitivity_df, avg_sensitivity
    
    def create_optimization_dashboard(self, optimization_recs, feature_effects):
        """Create optimization dashboard visualization"""
        print("\nCreating optimization dashboard...")
        
        if optimization_recs is None:
            print("No optimization results to visualize")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Feature effect ranges
        sorted_effects = sorted(feature_effects, key=lambda x: x['effect_range'], reverse=True)
        top_10_effects = sorted_effects[:10]
        
        features = [e['feature'] for e in top_10_effects]
        ranges = [e['effect_range'] for e in top_10_effects]
        
        bars = axes[0, 0].barh(features, ranges, color='skyblue', alpha=0.7)
        axes[0, 0].set_xlabel('Quality Score Range')
        axes[0, 0].set_title('Top 10 Features by Effect Range')
        
        # Add value labels
        for bar, value in zip(bars, ranges):
            axes[0, 0].text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                           f'{value:.1f}', va='center', ha='left')
        
        # 2. Optimization recommendations
        opt_df = pd.DataFrame(optimization_recs)
        
        x_pos = np.arange(len(opt_df))
        width = 0.35
        
        bars1 = axes[0, 1].bar(x_pos - width/2, opt_df['current_value'], width, 
                              label='Current', alpha=0.7)
        bars2 = axes[0, 1].bar(x_pos + width/2, opt_df['optimal_value'], width, 
                              label='Optimal', alpha=0.7)
        
        axes[0, 1].set_xlabel('Features')
        axes[0, 1].set_ylabel('Feature Value')
        axes[0, 1].set_title('Current vs Optimal Feature Values')
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(opt_df['feature'], rotation=45, ha='right')
        axes[0, 1].legend()
        
        # 3. Percentage changes required
        colors = ['red' if x < 0 else 'green' for x in opt_df['change_percent']]
        bars = axes[1, 0].bar(opt_df['feature'], opt_df['change_percent'], color=colors, alpha=0.7)
        axes[1, 0].set_ylabel('Change Required (%)')
        axes[1, 0].set_title('Required Changes for Optimization')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Add value labels
        for bar, value in zip(bars, opt_df['change_percent']):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, 
                           bar.get_height() + (1 if value >= 0 else -2),
                           f'{value:.1f}%', ha='center', va='bottom' if value >= 0 else 'top')
        
        # 4. Correlation vs Effect Range scatter
        correlations = [abs(e['correlation']) for e in sorted_effects[:15]]
        effect_ranges = [e['effect_range'] for e in sorted_effects[:15]]
        feature_labels = [e['feature'] for e in sorted_effects[:15]]
        
        scatter = axes[1, 1].scatter(correlations, effect_ranges, alpha=0.7, s=80)
        
        # Add feature labels
        for i, txt in enumerate(feature_labels):
            if i < 8:  # Only label top 8 to avoid clutter
                axes[1, 1].annotate(txt, (correlations[i], effect_ranges[i]),
                                   xytext=(5, 5), textcoords='offset points',
                                   fontsize=8, alpha=0.8)
        
        axes[1, 1].set_xlabel('Absolute Correlation with Quality')
        axes[1, 1].set_ylabel('Effect Range')
        axes[1, 1].set_title('Correlation vs Effect Range')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('optimization_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_actionable_insights(self, feature_effects, optimization_recs, optimal_quality):
        """Generate comprehensive actionable insights"""
        print("\n" + "="*80)
        print("ACTIONABLE INSIGHTS FOR QUALITY OPTIMIZATION")
        print("="*80)
        
        # Sort features by effect range
        sorted_effects = sorted(feature_effects, key=lambda x: x['effect_range'], reverse=True)
        
        print(f"\n🎯 KEY FINDINGS:")
        print(f"   • Current average quality score: {np.mean([e['current_mean'] for e in feature_effects]):.2f}")
        if optimal_quality:
            print(f"   • Predicted optimal quality score: {optimal_quality:.2f}")
            print(f"   • Potential improvement: {optimal_quality - 50:.2f} points")  # Assuming 50 as baseline
        
        print(f"\n📈 TOP 5 HIGH-IMPACT FEATURES:")
        for i, effect in enumerate(sorted_effects[:5], 1):
            impact = "HIGH" if effect['effect_range'] > 10 else "MEDIUM" if effect['effect_range'] > 5 else "LOW"
            direction = "↗️" if effect['correlation'] > 0 else "↘️"
            
            print(f"   {i}. {effect['feature']} {direction}")
            print(f"      • Impact: {impact} (±{effect['effect_range']:.1f} quality points)")
            print(f"      • Optimal value: {effect['optimal_value']:.2f}")
            print(f"      • Current average: {effect['current_mean']:.2f}")
            
            if abs(effect['optimal_value'] - effect['current_mean']) > 0.1:
                change_direction = "INCREASE" if effect['optimal_value'] > effect['current_mean'] else "DECREASE"
                print(f"      • Recommendation: {change_direction} by {abs(effect['optimal_value'] - effect['current_mean']):.2f}")
        
        if optimization_recs:
            print(f"\n🔧 SPECIFIC OPTIMIZATION RECOMMENDATIONS:")
            for i, rec in enumerate(optimization_recs, 1):
                if abs(rec['change_percent']) > 2:  # Only show significant changes
                    action = "INCREASE" if rec['change'] > 0 else "DECREASE"
                    print(f"   {i}. {rec['feature']}: {action} by {abs(rec['change_percent']):.1f}%")
                    print(f"      From {rec['current_value']:.2f} to {rec['optimal_value']:.2f}")
        
        print(f"\n⚡ PRIORITY ACTION ITEMS:")
        priority_features = [e for e in sorted_effects if e['effect_range'] > 5][:3]
        
        for i, feature in enumerate(priority_features, 1):
            print(f"   {i}. Monitor and optimize {feature['feature']}")
            print(f"      Target range: {feature['optimal_value'] - 2:.1f} - {feature['optimal_value'] + 2:.1f}")
        
        print(f"\n💡 IMPLEMENTATION STRATEGY:")
        print(f"   1. Start with the highest impact feature: {sorted_effects[0]['feature']}")
        print(f"   2. Make gradual adjustments (5-10% changes)")
        print(f"   3. Monitor quality score response after each change")
        print(f"   4. Focus on features with effect range > 5.0")
        print(f"   5. Consider feature interactions when making multiple changes")
        
        # Feature categories for easier implementation
        process_params = [f for f in self.feature_names if any(x in f.lower() for x in ['wait', 'cook', 'duration', 'iteration'])]
        env_params = [f for f in self.feature_names if 'env' in f.lower()]
        
        if process_params:
            process_effects = [e for e in sorted_effects if e['feature'] in process_params][:3]
            if process_effects:
                print(f"\n🏭 PROCESS PARAMETER OPTIMIZATION:")
                for effect in process_effects:
                    print(f"   • {effect['feature']}: Target {effect['optimal_value']:.1f}")
        
        if env_params:
            env_effects = [e for e in sorted_effects if e['feature'] in env_params][:3]
            if env_effects:
                print(f"\n🌡️ ENVIRONMENTAL PARAMETER OPTIMIZATION:")
                for effect in env_effects:
                    print(f"   • {effect['feature']}: Target {effect['optimal_value']:.1f}")

def main():
    """Main analysis function"""
    print("Cotton Candy Quality - Feature Effect Analysis & Optimization")
    print("="*80)
    
    # Initialize analyzer
    analyzer = FeatureEffectAnalyzer()
    
    # Load data
    X, y, data = analyzer.load_data()
    
    # Train best model
    model, model_name, model_score = analyzer.train_best_model(X, y)
    
    # Analyze individual feature effects
    feature_effects, effects_df = analyzer.analyze_feature_effects(X, y)
    
    # Create feature effect plots
    analyzer.create_feature_effect_plots(feature_effects)
    
    # Multi-feature optimization
    optimization_recs, optimal_quality = analyzer.multi_feature_optimization(X, y)
    
    # Sensitivity analysis
    sensitivity_df, avg_sensitivity = analyzer.sensitivity_analysis(X, y, feature_effects)
    
    # Create optimization dashboard
    analyzer.create_optimization_dashboard(optimization_recs, feature_effects)
    
    # Generate actionable insights
    analyzer.generate_actionable_insights(feature_effects, optimization_recs, optimal_quality)
    
    print("\n" + "="*80)
    print("FILES GENERATED:")
    print("- feature_effects_analysis.csv")
    print("- optimization_recommendations.csv") 
    print("- sensitivity_analysis.csv")
    print("- feature_effect_curves.png")
    print("- optimization_dashboard.png")
    print("="*80)

if __name__ == "__main__":
    main()
