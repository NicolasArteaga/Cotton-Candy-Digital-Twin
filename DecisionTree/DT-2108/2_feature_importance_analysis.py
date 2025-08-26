#!/usr/bin/env python3
"""
Feature Importance Analysis using Multiple Models
Comprehensive analysis to identify the most important features for quality prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class FeatureImportanceAnalyzer:
    def __init__(self):
        self.models = {}
        self.feature_importances = {}
        self.model_performances = {}
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load and prepare data"""
        print("Loading data...")
        features_df = pd.read_csv('xy/features_X.csv')
        targets_df = pd.read_csv('xy/targets_Y.csv')
        
        # Merge datasets
        data = pd.merge(features_df, targets_df, on='iteration', how='inner')
        
        # Prepare features and target
        feature_columns = features_df.columns[1:].tolist()  # Exclude 'iteration'
        X = data[feature_columns]
        y = data['quality_score']
        
        print(f"Data shape: {X.shape}")
        print(f"Target range: {y.min():.2f} - {y.max():.2f}")
        
        return X, y, feature_columns
    
    def univariate_feature_selection(self, X, y, feature_names):
        """Univariate feature selection methods"""
        print("\n" + "="*50)
        print("UNIVARIATE FEATURE SELECTION")
        print("="*50)
        
        # F-regression test
        f_scores, f_pvalues = f_regression(X, y)
        
        # Mutual information
        mi_scores = mutual_info_regression(X, y, random_state=42)
        
        # Create dataframe
        univariate_results = pd.DataFrame({
            'feature': feature_names,
            'f_score': f_scores,
            'f_pvalue': f_pvalues,
            'mutual_info': mi_scores
        })
        
        # Rank by different methods
        univariate_results['f_score_rank'] = univariate_results['f_score'].rank(ascending=False)
        univariate_results['mi_rank'] = univariate_results['mutual_info'].rank(ascending=False)
        
        # Sort by f_score
        univariate_results = univariate_results.sort_values('f_score', ascending=False)
        
        print("Top 10 Features by F-Score:")
        print(univariate_results[['feature', 'f_score', 'f_pvalue']].head(10))
        
        print("\nTop 10 Features by Mutual Information:")
        print(univariate_results.sort_values('mutual_info', ascending=False)[['feature', 'mutual_info']].head(10))
        
        # Save results
        univariate_results.to_csv('univariate_feature_selection.csv', index=False)
        
        return univariate_results
    
    def tree_based_importance(self, X, y, feature_names):
        """Tree-based feature importance methods"""
        print("\n" + "="*50)
        print("TREE-BASED FEATURE IMPORTANCE")
        print("="*50)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        tree_models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'ExtraTrees': ExtraTreesRegressor(n_estimators=100, random_state=42)
        }
        
        importances_df = pd.DataFrame({'feature': feature_names})
        
        for name, model in tree_models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)
            
            # Predictions and performance
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            print(f"{name} Performance:")
            print(f"  R² Score: {r2:.4f}")
            print(f"  MSE: {mse:.4f}")
            print(f"  MAE: {mae:.4f}")
            
            # Feature importances
            importances = model.feature_importances_
            importances_df[f'{name}_importance'] = importances
            
            # Store model and performance
            self.models[name] = model
            self.model_performances[name] = {'r2': r2, 'mse': mse, 'mae': mae}
        
        # Calculate average importance
        importance_cols = [col for col in importances_df.columns if 'importance' in col]
        importances_df['avg_importance'] = importances_df[importance_cols].mean(axis=1)
        
        # Sort by average importance
        importances_df = importances_df.sort_values('avg_importance', ascending=False)
        
        print(f"\nTop 10 Features by Average Tree-Based Importance:")
        print(importances_df[['feature', 'avg_importance'] + importance_cols].head(10))
        
        # Save results
        importances_df.to_csv('tree_based_importances.csv', index=False)
        
        return importances_df
    
    def linear_model_importance(self, X, y, feature_names):
        """Linear model feature importance (coefficients)"""
        print("\n" + "="*50)
        print("LINEAR MODEL FEATURE IMPORTANCE")
        print("="*50)
        
        # Split and scale data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        linear_models = {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0, random_state=42),
            'Lasso': Lasso(alpha=0.1, random_state=42),
            'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
        }
        
        coefficients_df = pd.DataFrame({'feature': feature_names})
        
        for name, model in linear_models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train_scaled, y_train)
            
            # Predictions and performance
            y_pred = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            print(f"{name} Performance:")
            print(f"  R² Score: {r2:.4f}")
            print(f"  MSE: {mse:.4f}")
            print(f"  MAE: {mae:.4f}")
            
            # Coefficients (absolute values for importance)
            coeffs = np.abs(model.coef_)
            coefficients_df[f'{name}_coef'] = coeffs
            
            # Store model and performance
            self.models[name] = (model, self.scaler)
            self.model_performances[name] = {'r2': r2, 'mse': mse, 'mae': mae}
        
        # Calculate average absolute coefficient
        coef_cols = [col for col in coefficients_df.columns if 'coef' in col]
        coefficients_df['avg_abs_coef'] = coefficients_df[coef_cols].mean(axis=1)
        
        # Sort by average coefficient
        coefficients_df = coefficients_df.sort_values('avg_abs_coef', ascending=False)
        
        print(f"\nTop 10 Features by Average Absolute Coefficient:")
        print(coefficients_df[['feature', 'avg_abs_coef'] + coef_cols].head(10))
        
        # Save results
        coefficients_df.to_csv('linear_model_coefficients.csv', index=False)
        
        return coefficients_df
    
    def ensemble_importance_ranking(self, univariate_df, tree_df, linear_df):
        """Combine all importance measures for final ranking"""
        print("\n" + "="*50)
        print("ENSEMBLE FEATURE IMPORTANCE RANKING")
        print("="*50)
        
        # Merge all importance measures
        ensemble_df = pd.DataFrame({'feature': univariate_df['feature']})
        
        # Add normalized rankings (lower rank = more important)
        ensemble_df['f_score_rank'] = univariate_df['f_score'].rank(ascending=False)
        ensemble_df['mi_rank'] = univariate_df['mutual_info'].rank(ascending=False)
        ensemble_df['tree_importance_rank'] = tree_df['avg_importance'].rank(ascending=False)
        ensemble_df['linear_coef_rank'] = linear_df['avg_abs_coef'].rank(ascending=False)
        
        # Calculate average rank
        rank_cols = [col for col in ensemble_df.columns if 'rank' in col]
        ensemble_df['avg_rank'] = ensemble_df[rank_cols].mean(axis=1)
        
        # Add actual values for reference
        ensemble_df = pd.merge(ensemble_df, univariate_df[['feature', 'f_score', 'mutual_info']], on='feature')
        ensemble_df = pd.merge(ensemble_df, tree_df[['feature', 'avg_importance']], on='feature')
        ensemble_df = pd.merge(ensemble_df, linear_df[['feature', 'avg_abs_coef']], on='feature')
        
        # Sort by average rank
        ensemble_df = ensemble_df.sort_values('avg_rank')
        
        print("Top 15 Features by Ensemble Ranking:")
        display_cols = ['feature', 'avg_rank', 'f_score', 'mutual_info', 'avg_importance', 'avg_abs_coef']
        print(ensemble_df[display_cols].head(15))
        
        # Save results
        ensemble_df.to_csv('ensemble_feature_ranking.csv', index=False)
        
        return ensemble_df
    
    def create_visualizations(self, ensemble_df, tree_df):
        """Create comprehensive visualization plots"""
        print("\nCreating visualizations...")
        
        # Top features for visualization
        top_features = ensemble_df.head(12)
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # 1. Feature importance comparison
        methods = ['f_score', 'mutual_info', 'avg_importance', 'avg_abs_coef']
        method_names = ['F-Score', 'Mutual Info', 'Tree Importance', 'Linear Coef']
        
        x_pos = np.arange(len(top_features))
        width = 0.2
        
        for i, (method, name) in enumerate(zip(methods, method_names)):
            # Normalize to 0-1 scale for comparison
            values = top_features[method] / top_features[method].max()
            axes[0, 0].bar(x_pos + i*width, values, width, label=name, alpha=0.8)
        
        axes[0, 0].set_xlabel('Features')
        axes[0, 0].set_ylabel('Normalized Importance')
        axes[0, 0].set_title('Feature Importance Comparison (Top 12 Features)')
        axes[0, 0].set_xticks(x_pos + width * 1.5)
        axes[0, 0].set_xticklabels(top_features['feature'], rotation=45, ha='right')
        axes[0, 0].legend()
        
        # 2. Average ranking
        axes[0, 1].barh(top_features['feature'][::-1], top_features['avg_rank'][::-1])
        axes[0, 1].set_xlabel('Average Rank (lower is better)')
        axes[0, 1].set_title('Top Features by Average Ranking')
        
        # 3. Model performance comparison
        model_names = list(self.model_performances.keys())
        r2_scores = [self.model_performances[name]['r2'] for name in model_names]
        
        bars = axes[1, 0].bar(model_names, r2_scores)
        axes[1, 0].set_ylabel('R² Score')
        axes[1, 0].set_title('Model Performance Comparison')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, score in zip(bars, r2_scores):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{score:.3f}', ha='center', va='bottom')
        
        # 4. Feature importance heatmap (top features across methods)
        top_10_features = ensemble_df.head(10)
        heatmap_data = top_10_features[['f_score', 'mutual_info', 'avg_importance', 'avg_abs_coef']].T
        heatmap_data.columns = top_10_features['feature']
        
        # Normalize each row to 0-1
        heatmap_data_norm = heatmap_data.div(heatmap_data.max(axis=1), axis=0)
        
        sns.heatmap(heatmap_data_norm, annot=True, fmt='.2f', cmap='YlOrRd', ax=axes[1, 1])
        axes[1, 1].set_title('Feature Importance Heatmap (Top 10 Features)')
        axes[1, 1].set_ylabel('Importance Method')
        
        plt.tight_layout()
        plt.savefig('comprehensive_feature_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Additional plot: Feature importance by category
        self.plot_feature_categories(ensemble_df)
    
    def plot_feature_categories(self, ensemble_df):
        """Create categorized feature importance plot"""
        # Categorize features
        categories = {
            'Process Parameters': ['wait_time', 'cook_time', 'duration_cc_flow', 'iteration_since_maintenance'],
            'Environmental Baseline': ['baseline_env_EnvH', 'baseline_env_EnvT'],
            'Pre-Process Environment': ['before_turn_on_env_InH', 'before_turn_on_env_InT', 
                                      'before_turn_on_env_IrO', 'before_turn_on_env_IrA'],
            'System State': ['iteration']
        }
        
        # Assign categories
        ensemble_df['category'] = 'Unknown'
        for category, features in categories.items():
            mask = ensemble_df['feature'].isin(features)
            ensemble_df.loc[mask, 'category'] = category
        
        # Create category-based plot
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Sort by average rank within categories
        ensemble_df_sorted = ensemble_df.sort_values(['category', 'avg_rank'])
        
        # Color map for categories
        category_colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
        color_map = dict(zip(categories.keys(), category_colors))
        
        # Create horizontal bar plot
        y_pos = np.arange(len(ensemble_df_sorted))
        colors = [color_map.get(cat, 'gray') for cat in ensemble_df_sorted['category']]
        
        bars = ax.barh(y_pos, 1 / ensemble_df_sorted['avg_rank'], color=colors, alpha=0.7)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(ensemble_df_sorted['feature'])
        ax.set_xlabel('Importance Score (1/avg_rank)')
        ax.set_title('Feature Importance by Category')
        
        # Add legend
        legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=color_map[cat], alpha=0.7, label=cat)
                          for cat in categories.keys()]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        plt.savefig('feature_importance_by_category.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_insights(self, ensemble_df):
        """Generate actionable insights"""
        print("\n" + "="*60)
        print("ACTIONABLE INSIGHTS FOR QUALITY IMPROVEMENT")
        print("="*60)
        
        top_5 = ensemble_df.head(5)
        
        print(f"\n🎯 TOP 5 MOST IMPORTANT FEATURES:")
        for i, row in top_5.iterrows():
            print(f"   {int(row['avg_rank'])}. {row['feature']}")
            print(f"      - Average Rank: {row['avg_rank']:.1f}")
            print(f"      - F-Score: {row['f_score']:.2f}")
            print(f"      - Tree Importance: {row['avg_importance']:.3f}")
        
        print(f"\n📊 MODEL PERFORMANCE SUMMARY:")
        best_model = max(self.model_performances.items(), key=lambda x: x[1]['r2'])
        print(f"   Best Model: {best_model[0]} (R² = {best_model[1]['r2']:.4f})")
        
        for name, metrics in self.model_performances.items():
            print(f"   {name}: R² = {metrics['r2']:.4f}, MAE = {metrics['mae']:.2f}")
        
        print(f"\n🔧 RECOMMENDATIONS FOR QUALITY IMPROVEMENT:")
        print("   1. Focus on optimizing the top 5 features identified above")
        print("   2. Monitor and control environmental conditions carefully")
        print("   3. Optimize process timing parameters (wait_time, cook_time)")
        print("   4. Consider maintenance scheduling impact")
        print("   5. Implement real-time monitoring of critical features")
        
        return top_5['feature'].tolist()

def main():
    """Main analysis function"""
    print("Cotton Candy Quality - Feature Importance Analysis")
    print("="*60)
    
    # Initialize analyzer
    analyzer = FeatureImportanceAnalyzer()
    
    # Load data
    X, y, feature_names = analyzer.load_data()
    
    # Univariate feature selection
    univariate_results = analyzer.univariate_feature_selection(X, y, feature_names)
    
    # Tree-based importance
    tree_importance = analyzer.tree_based_importance(X, y, feature_names)
    
    # Linear model importance
    linear_importance = analyzer.linear_model_importance(X, y, feature_names)
    
    # Ensemble ranking
    ensemble_ranking = analyzer.ensemble_importance_ranking(
        univariate_results, tree_importance, linear_importance)
    
    # Create visualizations
    analyzer.create_visualizations(ensemble_ranking, tree_importance)
    
    # Generate insights
    top_features = analyzer.generate_insights(ensemble_ranking)
    
    print("\n" + "="*60)
    print("FILES GENERATED:")
    print("- univariate_feature_selection.csv")
    print("- tree_based_importances.csv") 
    print("- linear_model_coefficients.csv")
    print("- ensemble_feature_ranking.csv")
    print("- comprehensive_feature_analysis.png")
    print("- feature_importance_by_category.png")
    print("="*60)
    
    return top_features

if __name__ == "__main__":
    main()
