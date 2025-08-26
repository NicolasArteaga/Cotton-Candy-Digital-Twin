#!/usr/bin/env python3
"""
Advanced Machine Learning Models Comparison
Testing multiple advanced ML models to find the best approach for quality prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, 
                             ExtraTreesRegressor, AdaBoostRegressor, BaggingRegressor)
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet, 
                                 BayesianRidge, HuberRegressor)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

class AdvancedMLComparison:
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_models = {}
        self.scalers = ['standard', 'robust', 'minmax', 'none']
        
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
    
    def get_model_configs(self):
        """Define all models to test"""
        models = {
            # Tree-based models
            'RandomForest': {
                'model': RandomForestRegressor(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 5, 10, 15],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'scaling': False
            },
            'GradientBoosting': {
                'model': GradientBoostingRegressor(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0]
                },
                'scaling': False
            },
            'ExtraTrees': {
                'model': ExtraTreesRegressor(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 5, 10, 15],
                    'min_samples_split': [2, 5, 10]
                },
                'scaling': False
            },
            'AdaBoost': {
                'model': AdaBoostRegressor(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 1.0],
                    'loss': ['linear', 'square', 'exponential']
                },
                'scaling': False
            },
            
            # Linear models
            'Ridge': {
                'model': Ridge(random_state=42),
                'params': {
                    'alpha': [0.1, 1.0, 10.0, 100.0]
                },
                'scaling': True
            },
            'Lasso': {
                'model': Lasso(random_state=42, max_iter=2000),
                'params': {
                    'alpha': [0.01, 0.1, 1.0, 10.0]
                },
                'scaling': True
            },
            'ElasticNet': {
                'model': ElasticNet(random_state=42, max_iter=2000),
                'params': {
                    'alpha': [0.01, 0.1, 1.0],
                    'l1_ratio': [0.1, 0.5, 0.9]
                },
                'scaling': True
            },
            'BayesianRidge': {
                'model': BayesianRidge(),
                'params': {
                    'alpha_1': [1e-6, 1e-5, 1e-4],
                    'alpha_2': [1e-6, 1e-5, 1e-4],
                    'lambda_1': [1e-6, 1e-5, 1e-4],
                    'lambda_2': [1e-6, 1e-5, 1e-4]
                },
                'scaling': True
            },
            
            # Support Vector Machine
            'SVR': {
                'model': SVR(),
                'params': {
                    'C': [0.1, 1.0, 10.0, 100.0],
                    'kernel': ['linear', 'rbf', 'poly'],
                    'gamma': ['scale', 'auto'],
                    'epsilon': [0.01, 0.1, 1.0]
                },
                'scaling': True
            },
            
            # K-Nearest Neighbors
            'KNN': {
                'model': KNeighborsRegressor(),
                'params': {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan']
                },
                'scaling': True
            },
            
            # Neural Network
            'MLP': {
                'model': MLPRegressor(random_state=42, max_iter=2000),
                'params': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                    'activation': ['relu', 'tanh'],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate_init': [0.001, 0.01]
                },
                'scaling': True
            }
        }
        
        return models
    
    def get_scaler(self, scaler_type):
        """Get scaler object"""
        scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler(),
            'minmax': MinMaxScaler(),
            'none': None
        }
        return scalers[scaler_type]
    
    def evaluate_model(self, X, y, model_name, model_config, cv_folds=5):
        """Evaluate a single model with hyperparameter tuning"""
        print(f"\nEvaluating {model_name}...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        best_score = -np.inf
        best_config = None
        best_model = None
        
        # Test with and without scaling if applicable
        scaling_options = [True, False] if model_config['scaling'] else [False]
        
        for use_scaling in scaling_options:
            for scaler_type in (['standard', 'robust'] if use_scaling else ['none']):
                scaler = self.get_scaler(scaler_type)
                
                # Create pipeline
                if scaler is not None:
                    pipeline = Pipeline([
                        ('scaler', scaler),
                        ('model', model_config['model'])
                    ])
                    param_grid = {f'model__{k}': v for k, v in model_config['params'].items()}
                else:
                    pipeline = model_config['model']
                    param_grid = model_config['params']
                
                # Grid search with cross-validation
                try:
                    grid_search = GridSearchCV(
                        pipeline, param_grid, cv=cv_folds, 
                        scoring='r2', n_jobs=-1, verbose=0
                    )
                    grid_search.fit(X_train, y_train)
                    
                    # Evaluate on test set
                    y_pred = grid_search.predict(X_test)
                    test_r2 = r2_score(y_test, y_pred)
                    
                    if test_r2 > best_score:
                        best_score = test_r2
                        best_config = {
                            'scaler': scaler_type,
                            'params': grid_search.best_params_,
                            'cv_score': grid_search.best_score_
                        }
                        best_model = grid_search.best_estimator_
                        
                        # Calculate additional metrics
                        test_mse = mean_squared_error(y_test, y_pred)
                        test_mae = mean_absolute_error(y_test, y_pred)
                        
                except Exception as e:
                    print(f"   Error with {scaler_type} scaling: {e}")
                    continue
        
        if best_model is not None:
            print(f"   Best R²: {best_score:.4f}")
            print(f"   Best config: {best_config}")
            
            return {
                'model': best_model,
                'test_r2': best_score,
                'test_mse': test_mse,
                'test_mae': test_mae,
                'config': best_config,
                'y_pred': y_pred,
                'y_test': y_test
            }
        else:
            print(f"   Failed to train {model_name}")
            return None
    
    def run_comprehensive_comparison(self, X, y):
        """Run comprehensive model comparison"""
        print("\n" + "="*60)
        print("COMPREHENSIVE MODEL COMPARISON")
        print("="*60)
        
        model_configs = self.get_model_configs()
        results_list = []
        
        for model_name, model_config in model_configs.items():
            result = self.evaluate_model(X, y, model_name, model_config)
            if result is not None:
                self.results[model_name] = result
                results_list.append({
                    'Model': model_name,
                    'Test_R2': result['test_r2'],
                    'Test_MSE': result['test_mse'],
                    'Test_MAE': result['test_mae'],
                    'CV_R2': result['config']['cv_score'],
                    'Scaler': result['config']['scaler']
                })
        
        # Create results DataFrame
        results_df = pd.DataFrame(results_list)
        results_df = results_df.sort_values('Test_R2', ascending=False)
        
        print("\nModel Performance Summary:")
        print(results_df.round(4))
        
        # Save results
        results_df.to_csv('model_comparison_results.csv', index=False)
        
        return results_df
    
    def analyze_best_models(self, results_df, top_n=5):
        """Detailed analysis of top performing models"""
        print(f"\n" + "="*60)
        print(f"TOP {top_n} MODEL ANALYSIS")
        print("="*60)
        
        top_models = results_df.head(top_n)
        
        for idx, row in top_models.iterrows():
            model_name = row['Model']
            result = self.results[model_name]
            
            print(f"\n🏆 {idx+1}. {model_name}")
            print(f"   Test R²: {result['test_r2']:.4f}")
            print(f"   Test MSE: {result['test_mse']:.4f}")
            print(f"   Test MAE: {result['test_mae']:.4f}")
            print(f"   CV R²: {result['config']['cv_score']:.4f}")
            print(f"   Scaling: {result['config']['scaler']}")
            print(f"   Best params: {result['config']['params']}")
            
            # Feature importance if available
            model = result['model']
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                print(f"   Top 3 features: {np.argsort(importances)[-3:][::-1]}")
            elif hasattr(model, 'coef_'):
                coef = model.coef_ if hasattr(model.coef_, '__len__') else [model.coef_]
                print(f"   Top 3 coef indices: {np.argsort(np.abs(coef))[-3:][::-1]}")
    
    def create_model_comparison_plots(self, results_df):
        """Create comprehensive visualization plots"""
        print("\nCreating visualization plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # 1. Model performance comparison
        top_10 = results_df.head(10)
        
        x_pos = np.arange(len(top_10))
        width = 0.35
        
        bars1 = axes[0, 0].bar(x_pos - width/2, top_10['Test_R2'], width, 
                              label='Test R²', alpha=0.8)
        bars2 = axes[0, 0].bar(x_pos + width/2, top_10['CV_R2'], width, 
                              label='CV R²', alpha=0.8)
        
        axes[0, 0].set_xlabel('Models')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].set_title('Model Performance Comparison (Top 10)')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(top_10['Model'], rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.005,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 2. Error comparison
        axes[0, 1].scatter(top_10['Test_MAE'], top_10['Test_R2'], 
                          s=100, alpha=0.7, c=range(len(top_10)), cmap='viridis')
        
        for i, row in top_10.iterrows():
            axes[0, 1].annotate(row['Model'], 
                               (row['Test_MAE'], row['Test_R2']),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8, alpha=0.8)
        
        axes[0, 1].set_xlabel('Test MAE')
        axes[0, 1].set_ylabel('Test R²')
        axes[0, 1].set_title('R² vs MAE Trade-off')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Prediction vs Actual for best model
        best_model_name = results_df.iloc[0]['Model']
        best_result = self.results[best_model_name]
        
        axes[1, 0].scatter(best_result['y_test'], best_result['y_pred'], 
                          alpha=0.7, s=50)
        
        # Perfect prediction line
        min_val = min(best_result['y_test'].min(), best_result['y_pred'].min())
        max_val = max(best_result['y_test'].max(), best_result['y_pred'].max())
        axes[1, 0].plot([min_val, max_val], [min_val, max_val], 
                       'r--', alpha=0.8, label='Perfect Prediction')
        
        axes[1, 0].set_xlabel('Actual Quality Score')
        axes[1, 0].set_ylabel('Predicted Quality Score')
        axes[1, 0].set_title(f'Best Model Predictions: {best_model_name}')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add R² and MAE to the plot
        r2 = best_result['test_r2']
        mae = best_result['test_mae']
        axes[1, 0].text(0.05, 0.95, f'R² = {r2:.4f}\nMAE = {mae:.2f}', 
                       transform=axes[1, 0].transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 4. Model complexity vs performance
        model_complexity = {
            'LinearRegression': 1, 'Ridge': 1, 'Lasso': 1, 'ElasticNet': 1, 'BayesianRidge': 1,
            'KNN': 2, 'SVR': 3, 'DecisionTree': 3, 'MLP': 4,
            'RandomForest': 5, 'ExtraTrees': 5, 'GradientBoosting': 5, 'AdaBoost': 4
        }
        
        complexity_scores = [model_complexity.get(model, 3) for model in top_10['Model']]
        
        scatter = axes[1, 1].scatter(complexity_scores, top_10['Test_R2'], 
                                    s=100, alpha=0.7, c=range(len(top_10)), cmap='viridis')
        
        for i, row in top_10.iterrows():
            axes[1, 1].annotate(row['Model'], 
                               (model_complexity.get(row['Model'], 3), row['Test_R2']),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8, alpha=0.8)
        
        axes[1, 1].set_xlabel('Model Complexity (1=Simple, 5=Complex)')
        axes[1, 1].set_ylabel('Test R²')
        axes[1, 1].set_title('Model Complexity vs Performance')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('advanced_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def ensemble_model_analysis(self, X, y, top_models_list):
        """Create ensemble of top models"""
        print(f"\n" + "="*60)
        print("ENSEMBLE MODEL ANALYSIS")
        print("="*60)
        
        if len(top_models_list) < 2:
            print("Not enough models for ensemble analysis")
            return
        
        # Use top 3-5 models for ensemble
        ensemble_models = top_models_list[:min(5, len(top_models_list))]
        print(f"Creating ensemble from: {[model['Model'] for model in ensemble_models]}")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Get predictions from each model
        predictions = []
        weights = []
        
        for model_info in ensemble_models:
            model_name = model_info['Model']
            model_result = self.results[model_name]
            model = model_result['model']
            
            # Get predictions
            pred = model.predict(X_test)
            predictions.append(pred)
            weights.append(model_result['test_r2'])  # Use R² as weight
        
        # Convert to array
        predictions = np.array(predictions)
        weights = np.array(weights)
        weights = weights / np.sum(weights)  # Normalize weights
        
        # Simple average ensemble
        ensemble_pred_avg = np.mean(predictions, axis=0)
        
        # Weighted average ensemble
        ensemble_pred_weighted = np.average(predictions, axis=0, weights=weights)
        
        # Evaluate ensembles
        avg_r2 = r2_score(y_test, ensemble_pred_avg)
        avg_mae = mean_absolute_error(y_test, ensemble_pred_avg)
        
        weighted_r2 = r2_score(y_test, ensemble_pred_weighted)
        weighted_mae = mean_absolute_error(y_test, ensemble_pred_weighted)
        
        print(f"\nEnsemble Results:")
        print(f"Simple Average - R²: {avg_r2:.4f}, MAE: {avg_mae:.2f}")
        print(f"Weighted Average - R²: {weighted_r2:.4f}, MAE: {weighted_mae:.2f}")
        
        # Compare with best individual model
        best_individual = ensemble_models[0]
        print(f"Best Individual ({best_individual['Model']}) - R²: {best_individual['Test_R2']:.4f}, MAE: {best_individual['Test_MAE']:.2f}")
        
        return {
            'simple_avg': {'r2': avg_r2, 'mae': avg_mae, 'predictions': ensemble_pred_avg},
            'weighted_avg': {'r2': weighted_r2, 'mae': weighted_mae, 'predictions': ensemble_pred_weighted},
            'y_test': y_test
        }
    
    def generate_final_recommendations(self, results_df, ensemble_results=None):
        """Generate final recommendations"""
        print("\n" + "="*80)
        print("FINAL MODEL RECOMMENDATIONS")
        print("="*80)
        
        best_model = results_df.iloc[0]
        
        print(f"\n🏆 BEST SINGLE MODEL: {best_model['Model']}")
        print(f"   Performance: R² = {best_model['Test_R2']:.4f}, MAE = {best_model['Test_MAE']:.2f}")
        print(f"   Cross-validation R²: {best_model['CV_R2']:.4f}")
        print(f"   Preprocessing: {best_model['Scaler']} scaling")
        
        if ensemble_results:
            best_ensemble = max(ensemble_results.items(), key=lambda x: x[1]['r2'])
            print(f"\n🎯 BEST ENSEMBLE: {best_ensemble[0]}")
            print(f"   Performance: R² = {best_ensemble[1]['r2']:.4f}, MAE = {best_ensemble[1]['mae']:.2f}")
        
        print(f"\n📊 TOP 5 MODELS SUMMARY:")
        for i, row in results_df.head(5).iterrows():
            print(f"   {i+1}. {row['Model']}: R² = {row['Test_R2']:.4f}, MAE = {row['Test_MAE']:.2f}")
        
        print(f"\n💡 RECOMMENDATIONS:")
        print(f"   1. Use {best_model['Model']} as primary model")
        print(f"   2. Implement proper data scaling ({best_model['Scaler']})")
        print(f"   3. Consider ensemble approach for improved robustness")
        print(f"   4. Monitor model performance with cross-validation")
        print(f"   5. Focus on top features identified in feature importance analysis")

def main():
    """Main analysis function"""
    print("Cotton Candy Quality - Advanced ML Model Comparison")
    print("="*70)
    
    # Initialize analyzer
    analyzer = AdvancedMLComparison()
    
    # Load data
    X, y, feature_names = analyzer.load_data()
    
    # Run comprehensive comparison
    results_df = analyzer.run_comprehensive_comparison(X, y)
    
    # Analyze best models
    analyzer.analyze_best_models(results_df, top_n=5)
    
    # Create visualizations
    analyzer.create_model_comparison_plots(results_df)
    
    # Ensemble analysis
    ensemble_results = analyzer.ensemble_model_analysis(X, y, results_df.to_dict('records'))
    
    # Final recommendations
    analyzer.generate_final_recommendations(results_df, ensemble_results)
    
    print("\n" + "="*70)
    print("FILES GENERATED:")
    print("- model_comparison_results.csv")
    print("- advanced_model_comparison.png")
    print("="*70)
    
    return results_df

if __name__ == "__main__":
    main()
