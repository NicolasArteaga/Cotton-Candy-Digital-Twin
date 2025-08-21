#!/usr/bin/env python3
"""
Cotton Candy Decision Tree Trainer - Updated for Quality Scores
==============================================================
This script trains a decision tree to predict cotton candy quality based on
manufacturing process parameters. Now supports both weight and quality score targets.
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class CottonCandyDigitalTwin:
    def __init__(self, features_path="/Users/nicolas/Desktop/Cotton-Candy-Digital-Twin/Data_Collection/xy/features_X.csv", 
                 targets_path="/Users/nicolas/Desktop/Cotton-Candy-Digital-Twin/Data_Collection/xy/target_y.csv",
                 preferred_target=None):
        """Initialize the Cotton Candy Digital Twin.
        
        Args:
            features_path: Path to features CSV file
            targets_path: Path to targets CSV file  
            preferred_target: Preferred target column ('cc_quality_score', 'calculated_score', 'cc_weight', or None for auto-detect)
        """
        self.features_path = Path(features_path)
        self.targets_path = Path(targets_path)
        self.preferred_target = preferred_target
        self.model = None
        self.feature_names = None
        self.target_name = None
        self.target_col = None
        
    def load_and_prepare_data(self):
        """Load and prepare features and targets data."""
        print("🔍 LOADING DATA...")
        
        # Load features
        if self.features_path.exists():
            features_df = pd.read_csv(self.features_path)
            print(f"   Features loaded: {len(features_df)} samples, {len(features_df.columns)} features")
        else:
            raise FileNotFoundError(f"Features file not found: {self.features_path}")
        
        # Load targets
        if self.targets_path.exists():
            targets_df = pd.read_csv(self.targets_path)
            print(f"   Targets loaded: {len(targets_df)} samples")
            
            # Detect target type (use preferred if specified and available)
            available_targets = []
            if 'cc_quality_score' in targets_df.columns:
                available_targets.append('cc_quality_score')
            if 'calculated_score' in targets_df.columns:
                available_targets.append('calculated_score')
            if 'cc_weight' in targets_df.columns:
                available_targets.append('cc_weight')
            
            if not available_targets:
                raise ValueError("No valid target column found. Expected 'cc_quality_score', 'calculated_score', or 'cc_weight'")
            
            # Use preferred target if specified and available
            if self.preferred_target and self.preferred_target in available_targets:
                self.target_col = self.preferred_target
            else:
                # Auto-detect priority: cc_quality_score > calculated_score > cc_weight
                if 'cc_quality_score' in available_targets:
                    self.target_col = 'cc_quality_score'
                elif 'calculated_score' in available_targets:
                    self.target_col = 'calculated_score'
                else:
                    self.target_col = 'cc_weight'
            
            # Set target name based on selected column
            if self.target_col == 'cc_quality_score':
                self.target_name = "Cotton Candy Quality Score (0-100)"
                print("   Target type: Quality Score (0-100)")
            elif self.target_col == 'calculated_score':
                self.target_name = "Cotton Candy Calculated Score (0-100)"
                print("   Target type: Calculated Score (0-100)")
            elif self.target_col == 'cc_weight':
                self.target_name = "Cotton Candy Weight (grams)"
                print("   Target type: Weight (grams)")
                
            print(f"   Available targets: {available_targets}")
            print(f"   Selected target: {self.target_col}")
        else:
            raise FileNotFoundError(f"Targets file not found: {self.targets_path}")
        
        # Merge on iteration
        print(f"   Target iterations: {list(targets_df['iteration'].values)}")
        print(f"   Features iterations: {list(features_df['iteration'].values)}")
        
        merged_df = features_df.merge(targets_df, on='iteration', how='inner')
        print(f"   Merged on iteration: {len(merged_df)} samples matched")
        
        # Handle missing target values
        initial_count = len(merged_df)
        merged_df = merged_df.dropna(subset=[self.target_col])
        print(f"   Samples with valid targets: {len(merged_df)} (removed {initial_count - len(merged_df)} missing)")
        
        # Clean data for quality scores
        if self.target_col in ['cc_quality_score', 'calculated_score']:
            # Fix obvious data entry errors
            merged_df[self.target_col] = merged_df[self.target_col].replace(0.2, 0)  # Likely meant to be 0
            
        # Check for missing values in features
        print("   Missing values per feature:")
        missing_counts = merged_df.isnull().sum()
        missing_features = missing_counts[missing_counts > 0]
        if len(missing_features) == 0:
            print("     No missing values found ✅")
        else:
            for feature, count in missing_features.items():
                if feature != self.target_col:  # Don't report target missing values
                    print(f"     {feature}: {count} missing values")
        
        # Separate features and target
        self.feature_names = [col for col in merged_df.columns if col != self.target_col]
        X = merged_df[self.feature_names]
        y = merged_df[self.target_col]
        
        return X, y
    
    def analyze_data(self, X, y):
        """Analyze the loaded data."""
        print("\n📊 DATA SUMMARY:")
        print(f"   • Features: {len(self.feature_names)}")
        print(f"   • Samples: {len(X)}")
        print(f"   • Target range: {y.min():.2f} - {y.max():.2f}")
        print(f"   • Target mean: {y.mean():.2f} ± {y.std():.2f}")
        
        return X, y
    
    def split_data(self, X, y):
        """Split data into training and testing sets."""
        print("\n🔀 SPLITTING DATA...")
        
        # Use smaller test size for small datasets
        test_size = 0.3 if len(X) < 30 else 0.2
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=None
        )
        
        print(f"   Training set: {len(X_train)} samples")
        print(f"   Testing set: {len(X_test)} samples")
        print(f"   Train target range: {y_train.min():.2f} - {y_train.max():.2f}")
        print(f"   Test target range: {y_test.min():.2f} - {y_test.max():.2f}")
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train, y_train):
        """Train the decision tree with hyperparameter tuning."""
        print("\n🌳 TRAINING DECISION TREE...")
        print("   Performing hyperparameter tuning...")
        
        # Define parameter grid for small dataset
        param_grid = {
            'max_depth': [None, 3, 4, 5],
            'min_samples_split': [2, 3, 4],
            'min_samples_leaf': [1, 2, 3],
            'max_features': ['sqrt', 'log2', None]
        }
        
        # Use fewer folds for small datasets
        cv_folds = min(3, len(X_train) // 5) if len(X_train) < 50 else 5
        cv_folds = max(2, cv_folds)  # Ensure at least 2 folds
        
        # Grid search with cross-validation
        dt = DecisionTreeRegressor(random_state=42)
        grid_search = GridSearchCV(dt, param_grid, cv=cv_folds, 
                                 scoring='r2', n_jobs=-1, verbose=0)
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        
        print("   ✅ Best parameters found:")
        for param, value in grid_search.best_params_.items():
            print(f"      • {param}: {value}")
        print(f"   ✅ Best CV R² score: {grid_search.best_score_:.4f}")
        
        return self.model
    
    def evaluate_model(self, X_train, X_test, y_train, y_test):
        """Evaluate the trained model."""
        print("\n📈 EVALUATING MODEL...")
        
        # Make predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # Calculate metrics
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        # Cross-validation
        cv_folds = min(3, len(X_train) // 5) if len(X_train) < 50 else 5
        cv_folds = max(2, cv_folds)
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=cv_folds, scoring='r2')
        
        print("   📊 PERFORMANCE METRICS:")
        print(f"      • Training R²: {train_r2:.4f}")
        print(f"      • Testing R²:  {test_r2:.4f}")
        print(f"      • Training RMSE: {train_rmse:.4f}")
        print(f"      • Testing RMSE:  {test_rmse:.4f}")
        print(f"      • Training MAE: {train_mae:.4f}")
        print(f"      • Testing MAE:  {test_mae:.4f}")
        print(f"      • Cross-Val R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        return {
            'train_r2': train_r2, 'test_r2': test_r2,
            'train_rmse': train_rmse, 'test_rmse': test_rmse,
            'train_mae': train_mae, 'test_mae': test_mae,
            'cv_r2_mean': cv_scores.mean(), 'cv_r2_std': cv_scores.std()
        }
    
    def analyze_feature_importance(self):
        """Analyze and display feature importance."""
        print("\n   🔍 TOP 10 MOST IMPORTANT FEATURES:")
        
        # Get feature importances
        importances = self.model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Display top 10
        for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
            print(f"       {i+1:2d}. {row['feature']:<25} : {row['importance']:.4f}")
        
        return feature_importance
    
    def create_visualizations(self, save_path="cotton_candy_decision_tree_analysis.png"):
        """Create visualizations of model performance and feature importance."""
        print("\n📊 CREATING VISUALIZATIONS...")
        
        # Get feature importances
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Feature importance plot
        top_features = feature_importance.head(10)
        ax1.barh(range(len(top_features)), top_features['importance'])
        ax1.set_yticks(range(len(top_features)))
        ax1.set_yticklabels(top_features['feature'])
        ax1.set_xlabel('Feature Importance')
        ax1.set_title('Top 10 Most Important Features')
        ax1.invert_yaxis()
        
        # Tree structure plot (simplified)
        try:
            plot_tree(self.model, feature_names=self.feature_names, 
                     filled=True, max_depth=3, fontsize=8, ax=ax2)
            ax2.set_title('Decision Tree Structure (Depth ≤ 3)')
        except:
            ax2.text(0.5, 0.5, 'Tree too complex to visualize', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Decision Tree Structure')
        
        plt.suptitle(f'Cotton Candy Digital Twin - {self.target_name}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Visualizations saved to: {save_path}")
        
        return fig
    
    def save_results(self, metrics, feature_importance):
        """Save model and results to files."""
        print("\n💾 SAVING MODEL AND RESULTS...")
        
        # Save model
        model_path = "cotton_candy_digital_twin.joblib"
        joblib.dump(self.model, model_path)
        print(f"   ✅ Model saved to: {model_path}")
        
        # Save feature importance
        importance_path = "feature_importances.csv"
        feature_importance.to_csv(importance_path, index=False)
        print(f"   ✅ Feature importances saved to: {importance_path}")
        
        # Save decision tree rules
        rules_path = "decision_tree_rules.txt"
        tree_rules = export_text(self.model, feature_names=self.feature_names)
        
        with open(rules_path, 'w') as f:
            f.write("COTTON CANDY DIGITAL TWIN - DECISION TREE RULES\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Target: {self.target_name}\n\n")
            f.write("Model Parameters:\n")
            for param, value in self.model.get_params().items():
                f.write(f"  {param}: {value}\n")
            f.write(f"\nPerformance Metrics:\n")
            for metric, value in metrics.items():
                f.write(f"  {metric}: {value}\n")
            f.write(f"\nDecision Tree Rules:\n{tree_rules}")
        
        print(f"   ✅ Decision tree rules saved to: {rules_path}")
    
    def make_example_prediction(self, X):
        """Make an example prediction."""
        print("\n🔮 EXAMPLE PREDICTION:")
        if len(X) > 0:
            sample = X.iloc[0:1]
            prediction = self.model.predict(sample)[0]
            print(f"   Input: First sample from dataset")
            print(f"   Predicted {self.target_col}: {prediction:.2f}")
            
            return prediction
        return None
    
    def train_complete_pipeline(self):
        """Execute the complete training pipeline."""
        try:
            # Load and prepare data
            X, y = self.load_and_prepare_data()
            X, y = self.analyze_data(X, y)
            
            # Split data
            X_train, X_test, y_train, y_test = self.split_data(X, y)
            
            # Train model
            model = self.train_model(X_train, y_train)
            
            # Evaluate model
            metrics = self.evaluate_model(X_train, X_test, y_train, y_test)
            
            # Analyze features
            feature_importance = self.analyze_feature_importance()
            
            # Create visualizations
            self.create_visualizations()
            
            # Save results
            self.save_results(metrics, feature_importance)
            
            # Example prediction
            self.make_example_prediction(X)
            
            return True
            
        except Exception as e:
            print(f"❌ Error in training pipeline: {e}")
            return False

def main():
    """Main function to train the Cotton Candy Digital Twin."""
    print("🍭 COTTON CANDY DIGITAL TWIN - DECISION TREE TRAINER")
    print("=" * 60)
    
    # You can specify which target to use:
    # - None: Auto-detect (priority: cc_quality_score > calculated_score > cc_weight)
    # - 'cc_quality_score': Use subjective quality scores
    # - 'calculated_score': Use calculated quality scores from function
    # - 'cc_weight': Use cotton candy weight
    preferred_target = None  # Change this to specify target
    
    # Create and train the model
    twin = CottonCandyDigitalTwin(preferred_target=preferred_target)
    success = twin.train_complete_pipeline()
    
    if success:
        print("\n🎉 COTTON CANDY DIGITAL TWIN TRAINING COMPLETE!")
        print("   Model files saved in: DecisionTree/")
        print("   You can now use this model to predict cotton candy quality!")
        print(f"   Target used: {twin.target_col}")
    else:
        print("\n❌ Training failed. Please check the error messages above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
