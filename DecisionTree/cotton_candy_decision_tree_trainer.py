#!/usr/bin/env python3
"""
Cotton Candy Decision Tree Trainer

This scrip        # Load target data
        if targets_path.exists():
            targets_df = pd.read_csv(targets_path)
            print(f"   Targets loaded: {len(targets_df)} samples")
            
            # Check if we have weight or quality score
            if 'cc_weight' in targets_df.columns:
                target_col = 'cc_weight'
                target_name = "Cotton Candy Weight (grams)"
            elif 'cc_quality_score' in targets_df.columns:
                target_col = 'cc_quality_score'
                target_name = "Cotton Candy Quality Score (0-100)"
            else:
                raise ValueError("No valid target column found. Expected 'cc_weight' or 'cc_quality_score'")
                
            print(f"   Target variable: {target_col}")
        else:
            raise FileNotFoundError(f"Targets file not found: {targets_path}")ins a Decision Tree model to predict cotton candy weight based on
manufacturing process parameters and environmental conditions.

The Decision Tree is a Digital Twin that can:
1. Predict cotton candy weight given process parameters
2. Identify the most important features affecting quality
3. Provide interpretable rules for manufacturing optimization

Usage:
    python cotton_candy_decision_tree_trainer.py
    python cotton_candy_decision_tree_trainer.py --max-depth 10 --min-samples 3
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor, plot_tree, export_text
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib
import argparse
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class CottonCandyDigitalTwin:
    """
    Digital Twin for Cotton Candy Manufacturing using Decision Trees
    
    This class encapsulates the entire machine learning pipeline:
    - Data loading and preprocessing
    - Model training and hyperparameter tuning
    - Model evaluation and interpretation
    - Prediction and optimization
    """
    
    def __init__(self, features_file="Data_Collection/features_X.csv", 
                 target_file="Data_Collection/target_y.csv",
                 output_dir="DecisionTree"):
        self.features_file = features_file
        self.target_file = target_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Model components
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Results storage
        self.training_results = {}
        
    def load_and_prepare_data(self):
        """Load features and targets, handle missing values, and prepare for training."""
        print("🔍 LOADING DATA...")
        
        # Load features (X)
        if not os.path.exists(self.features_file):
            raise FileNotFoundError(f"Features file not found: {self.features_file}")
        features_df = pd.read_csv(self.features_file)
        print(f"   Features loaded: {features_df.shape[0]} samples, {features_df.shape[1]} features")
        
        # Load targets (y)
        if not os.path.exists(self.target_file):
            raise FileNotFoundError(f"Target file not found: {self.target_file}")
        target_df = pd.read_csv(self.target_file)
        print(f"   Targets loaded: {target_df.shape[0]} samples")
        
        # Merge on iteration if present, otherwise use index
        if 'iteration' in target_df.columns and 'iteration' in features_df.columns:
            # Both have iteration columns - merge properly
            print(f"   Target iterations: {sorted(target_df['iteration'].tolist())}")
            print(f"   Features iterations: {sorted(features_df['iteration'].tolist())}")
            merged_df = pd.merge(features_df, target_df, on='iteration', how='inner')
            print(f"   Merged on iteration: {merged_df.shape[0]} samples matched")
        elif 'iteration' in target_df.columns:
            # Only targets have iteration - add to features
            features_df['iteration'] = range(len(features_df))
            merged_df = pd.merge(features_df, target_df, on='iteration', how='inner')
            merged_df = merged_df.drop('iteration', axis=1)
        else:
            # Assume same order and length
            merged_df = features_df.copy()
            merged_df['calculated_score'] = target_df['calculated_score'].values
        
        # Remove rows with missing targets
        before_dropna = len(merged_df)
        merged_df = merged_df.dropna(subset=['cc_weight'])
        after_dropna = len(merged_df)
        print(f"   Samples with valid targets: {after_dropna} (removed {before_dropna - after_dropna} missing)")
        
        if len(merged_df) == 0:
            raise ValueError("No valid samples found after removing missing targets")
        
        # Separate features and target
        self.feature_names = [col for col in merged_df.columns if col != 'cc_weight']
        X = merged_df[self.feature_names]
        y = merged_df['cc_weight']
        
        # Handle missing values in features
        print(f"   Missing values per feature:")
        missing_counts = X.isnull().sum()
        if missing_counts.sum() > 0:
            for feature, count in missing_counts[missing_counts > 0].items():
                print(f"     • {feature}: {count} missing")
            
            # Fill missing values with median (robust to outliers)
            X = X.fillna(X.median())
            print(f"   ✅ Missing values filled with median values")
        else:
            print(f"     No missing values found ✅")
        
        print(f"\n📊 DATA SUMMARY:")
        print(f"   • Features: {X.shape[1]}")
        print(f"   • Samples: {X.shape[0]}")
        print(f"   • Target range: {y.min():.2f} - {y.max():.2f}")
        print(f"   • Target mean: {y.mean():.2f} ± {y.std():.2f}")
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into training and testing sets."""
        print(f"\n🔀 SPLITTING DATA...")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=None
        )
        
        print(f"   Training set: {self.X_train.shape[0]} samples")
        print(f"   Testing set: {self.X_test.shape[0]} samples")
        print(f"   Train target range: {self.y_train.min():.2f} - {self.y_train.max():.2f}")
        print(f"   Test target range: {self.y_test.min():.2f} - {self.y_test.max():.2f}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_decision_tree(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, 
                           random_state=42, tune_hyperparameters=True):
        """Train the Decision Tree model with optional hyperparameter tuning."""
        print(f"\n🌳 TRAINING DECISION TREE...")
        
        if tune_hyperparameters:
            print("   Performing hyperparameter tuning...")
            
            # Define hyperparameter grid
            param_grid = {
                'max_depth': [None, 3, 5, 7, 10, 15],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
            
            # Grid search with cross-validation
            dt_base = DecisionTreeRegressor(random_state=random_state)
            grid_search = GridSearchCV(
                dt_base, param_grid, cv=5, scoring='r2', 
                n_jobs=-1, verbose=0
            )
            grid_search.fit(self.X_train, self.y_train)
            
            # Use best parameters
            self.model = grid_search.best_estimator_
            print(f"   ✅ Best parameters found:")
            for param, value in grid_search.best_params_.items():
                print(f"      • {param}: {value}")
            print(f"   ✅ Best CV R² score: {grid_search.best_score_:.4f}")
            
        else:
            # Use provided parameters
            self.model = DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state
            )
            self.model.fit(self.X_train, self.y_train)
            print(f"   ✅ Model trained with provided parameters")
        
        # Store training results
        self.training_results['model_params'] = self.model.get_params()
        
        return self.model
    
    def evaluate_model(self):
        """Evaluate the trained model and compute performance metrics."""
        print(f"\n📈 EVALUATING MODEL...")
        
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        # Make predictions
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)
        
        # Calculate metrics
        train_r2 = r2_score(self.y_train, y_train_pred)
        test_r2 = r2_score(self.y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
        train_mae = mean_absolute_error(self.y_train, y_train_pred)
        test_mae = mean_absolute_error(self.y_test, y_test_pred)
        
        # Cross-validation scores (only for datasets with enough samples)
        if len(self.X_train) >= 5:
            cv_folds = min(5, len(self.X_train))
            cv_scores = cross_val_score(self.model, self.X_train, self.y_train, cv=cv_folds, scoring='r2')
            cv_r2_mean = cv_scores.mean()
            cv_r2_std = cv_scores.std()
        elif len(self.X_train) >= 3:
            # Use smaller number of folds for small datasets
            cv_folds = min(3, len(self.X_train))
            cv_scores = cross_val_score(self.model, self.X_train, self.y_train, cv=cv_folds, scoring='r2')
            cv_r2_mean = cv_scores.mean()
            cv_r2_std = cv_scores.std()
        else:
            print("   Skipping cross-validation (dataset too small)")
            cv_r2_mean = train_r2  # Use training R² as proxy
            cv_r2_std = 0.0
        
        # Store results
        metrics = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'cv_r2_mean': cv_r2_mean,
            'cv_r2_std': cv_r2_std
        }
        
        self.training_results['metrics'] = metrics
        
        # Print results
        print(f"   📊 PERFORMANCE METRICS:")
        print(f"      • Training R²: {train_r2:.4f}")
        print(f"      • Testing R²:  {test_r2:.4f}")
        print(f"      • Training RMSE: {train_rmse:.4f}")
        print(f"      • Testing RMSE:  {test_rmse:.4f}")
        print(f"      • Training MAE: {train_mae:.4f}")
        print(f"      • Testing MAE:  {test_mae:.4f}")
        print(f"      • Cross-Val R²: {cv_r2_mean:.4f} ± {cv_r2_std:.4f}")
        
        # Model interpretation
        feature_importances = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.training_results['feature_importances'] = feature_importances
        
        print(f"\n   🔍 TOP 10 MOST IMPORTANT FEATURES:")
        for i, (_, row) in enumerate(feature_importances.head(10).iterrows()):
            print(f"      {i+1:2d}. {row['feature']:<25}: {row['importance']:.4f}")
        
        return metrics, feature_importances
    
    def visualize_results(self):
        """Create visualizations of model performance and interpretability."""
        print(f"\n📊 CREATING VISUALIZATIONS...")
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Predictions vs Actual
        ax1 = plt.subplot(2, 3, 1)
        y_test_pred = self.model.predict(self.X_test)
        plt.scatter(self.y_test, y_test_pred, alpha=0.6, s=50)
        plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Cotton Candy Weight')
        plt.ylabel('Predicted Cotton Candy Weight')
        plt.title('Predictions vs Actual Values')
        r2 = self.training_results['metrics']['test_r2']
        plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax1.transAxes, fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # 2. Feature Importance
        ax2 = plt.subplot(2, 3, 2)
        top_features = self.training_results['feature_importances'].head(15)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title('Top 15 Feature Importances')
        plt.gca().invert_yaxis()
        
        # 3. Residuals Plot
        ax3 = plt.subplot(2, 3, 3)
        residuals = self.y_test - y_test_pred
        plt.scatter(y_test_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residuals Plot')
        
        # 4. Decision Tree Visualization (simplified)
        ax4 = plt.subplot(2, 3, 4)
        if self.model.tree_.max_depth <= 3:  # Only plot if tree is small enough
            plot_tree(self.model, max_depth=3, feature_names=self.feature_names, 
                     filled=True, rounded=True, fontsize=8)
            plt.title('Decision Tree Structure (Max Depth 3)')
        else:
            plt.text(0.5, 0.5, f'Tree too complex to visualize\n(Max Depth: {self.model.tree_.max_depth})', 
                    ha='center', va='center', fontsize=12, transform=ax4.transAxes)
            plt.title('Decision Tree Structure')
            
        # 5. Performance Metrics Bar Chart
        ax5 = plt.subplot(2, 3, 5)
        metrics = self.training_results['metrics']
        metric_names = ['Train R²', 'Test R²', 'CV R²']
        metric_values = [metrics['train_r2'], metrics['test_r2'], metrics['cv_r2_mean']]
        bars = plt.bar(metric_names, metric_values, color=['skyblue', 'lightcoral', 'lightgreen'])
        plt.ylabel('R² Score')
        plt.title('Model Performance Comparison')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom')
        
        # 6. Target Distribution
        ax6 = plt.subplot(2, 3, 6)
        plt.hist(self.y_train, bins=10, alpha=0.7, label='Training', color='skyblue')
        plt.hist(self.y_test, bins=10, alpha=0.7, label='Testing', color='lightcoral')
        plt.xlabel('Cotton Candy Weight')
        plt.ylabel('Frequency')
        plt.title('Target Variable Distribution')
        plt.legend()
        
        plt.tight_layout()
        
        # Save the plot
        plot_file = self.output_dir / "cotton_candy_decision_tree_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"   ✅ Visualizations saved to: {plot_file}")
        
        plt.show()
    
    def save_model_and_results(self):
        """Save the trained model and results."""
        print(f"\n💾 SAVING MODEL AND RESULTS...")
        
        # Save the model
        model_file = self.output_dir / "cotton_candy_digital_twin.joblib"
        joblib.dump(self.model, model_file)
        print(f"   ✅ Model saved to: {model_file}")
        
        # Save feature importances
        importance_file = self.output_dir / "feature_importances.csv"
        self.training_results['feature_importances'].to_csv(importance_file, index=False)
        print(f"   ✅ Feature importances saved to: {importance_file}")
        
        # Save model rules as text
        rules_file = self.output_dir / "decision_tree_rules.txt"
        with open(rules_file, 'w') as f:
            f.write("COTTON CANDY DIGITAL TWIN - DECISION TREE RULES\n")
            f.write("=" * 50 + "\n\n")
            f.write("Model Parameters:\n")
            for param, value in self.training_results['model_params'].items():
                f.write(f"  {param}: {value}\n")
            f.write(f"\nPerformance Metrics:\n")
            for metric, value in self.training_results['metrics'].items():
                f.write(f"  {metric}: {value:.4f}\n")
            f.write(f"\nDecision Tree Rules:\n")
            f.write(export_text(self.model, feature_names=self.feature_names))
        print(f"   ✅ Decision tree rules saved to: {rules_file}")
    
    def predict_cotton_candy_weight(self, input_features):
        """Make predictions using the trained model."""
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Convert input to DataFrame if it's a dict or list
        if isinstance(input_features, dict):
            input_df = pd.DataFrame([input_features])
        elif isinstance(input_features, list):
            input_df = pd.DataFrame([input_features], columns=self.feature_names)
        else:
            input_df = input_features
        
        # Make prediction
        prediction = self.model.predict(input_df)
        return prediction[0] if len(prediction) == 1 else prediction

def main():
    parser = argparse.ArgumentParser(description='Train Cotton Candy Digital Twin Decision Tree')
    parser.add_argument('--features', default='Data_Collection/features_X.csv',
                       help='Features CSV file path')
    parser.add_argument('--target', default='Data_Collection/target_y.csv',
                       help='Target CSV file path')
    parser.add_argument('--output-dir', default='DecisionTree',
                       help='Output directory for results')
    parser.add_argument('--max-depth', type=int, default=None,
                       help='Maximum depth of decision tree (None for unlimited)')
    parser.add_argument('--min-samples', type=int, default=2,
                       help='Minimum samples required to split internal node')
    parser.add_argument('--no-tuning', action='store_true',
                       help='Skip hyperparameter tuning')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Fraction of data to use for testing (0.0-1.0)')
    
    args = parser.parse_args()
    
    try:
        print("🍭 COTTON CANDY DIGITAL TWIN - DECISION TREE TRAINER")
        print("=" * 60)
        
        # Initialize the Digital Twin
        digital_twin = CottonCandyDigitalTwin(
            features_file=args.features,
            target_file=args.target,
            output_dir=args.output_dir
        )
        
        # Load and prepare data
        X, y = digital_twin.load_and_prepare_data()
        
        # Split data
        digital_twin.split_data(X, y, test_size=args.test_size)
        
        # Train model
        digital_twin.train_decision_tree(
            max_depth=args.max_depth,
            min_samples_split=args.min_samples,
            tune_hyperparameters=not args.no_tuning
        )
        
        # Evaluate model
        digital_twin.evaluate_model()
        
        # Create visualizations
        digital_twin.visualize_results()
        
        # Save results
        digital_twin.save_model_and_results()
        
        print(f"\n🎉 COTTON CANDY DIGITAL TWIN TRAINING COMPLETE!")
        print(f"   Model files saved in: {args.output_dir}/")
        print(f"   You can now use this model to predict cotton candy quality!")
        
        # Example prediction
        print(f"\n🔮 EXAMPLE PREDICTION:")
        example_features = X.iloc[0].to_dict()
        predicted_weight = digital_twin.predict_cotton_candy_weight(example_features)
        actual_weight = y.iloc[0]
        print(f"   Input: First sample from dataset")
        print(f"   Predicted weight: {predicted_weight:.2f}")
        print(f"   Actual weight: {actual_weight:.2f}")
        print(f"   Prediction error: {abs(predicted_weight - actual_weight):.2f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
