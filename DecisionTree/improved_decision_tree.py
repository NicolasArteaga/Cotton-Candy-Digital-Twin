import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from sklearn import tree
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple
import joblib

class CottonCandyDecisionTree:
    def __init__(self, task_type: str = 'classification'):
        """
        Initialize the decision tree model
        
        Args:
            task_type: 'classification' for quality prediction, 'regression' for continuous metrics
        """
        self.task_type = task_type
        self.model = None
        self.feature_names = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_importance = None
        
    def prepare_features(self, df: pd.DataFrame, target_column: str = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for training by handling missing values and encoding
        """
        # Make a copy to avoid modifying original
        df_processed = df.copy()
        
        # Remove non-numeric columns that shouldn't be features
        columns_to_drop = ['source_file', 'timestamp_diff']
        df_processed = df_processed.drop(columns=[col for col in columns_to_drop if col in df_processed.columns])
        
        # Separate features and target
        if target_column and target_column in df_processed.columns:
            X = df_processed.drop(columns=[target_column])
            y = df_processed[target_column]
        else:
            X = df_processed
            y = None
            
        # Handle missing values
        # For numeric columns, fill with median
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        X[numeric_columns] = X[numeric_columns].fillna(X[numeric_columns].median())
        
        # For categorical columns, fill with mode
        categorical_columns = X.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'Unknown')
            
        # Convert categorical variables to numeric
        for col in categorical_columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        
        self.feature_names = X.columns.tolist()
        
        return X, y
    
    def train(self, df: pd.DataFrame, target_column: str, 
              max_depth: int = 10, min_samples_split: int = 5, 
              test_size: float = 0.2, random_state: int = 42):
        """
        Train the decision tree model
        """
        X, y = self.prepare_features(df, target_column)
        
        if y is None:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        # Handle target variable based on task type
        if self.task_type == 'classification':
            if y.dtype == 'object':
                y = self.label_encoder.fit_transform(y)
            self.model = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=random_state
            )
        else:  # regression
            y = pd.to_numeric(y, errors='coerce')
            y = y.fillna(y.median())
            self.model = DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=random_state
            )
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features (optional for decision trees, but can help with interpretation)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        print("Training decision tree...")
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        if self.task_type == 'classification':
            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test)
            y_pred = self.model.predict(X_test)
            
            print(f"Training Accuracy: {train_score:.3f}")
            print(f"Testing Accuracy: {test_score:.3f}")
            print("\nClassification Report:")
            
            # Handle label names
            if hasattr(self.label_encoder, 'classes_'):
                target_names = self.label_encoder.classes_
            else:
                target_names = None
                
            print(classification_report(y_test, y_pred, target_names=target_names))
            
        else:  # regression
            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test)
            y_pred = self.model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            
            print(f"Training R²: {train_score:.3f}")
            print(f"Testing R²: {test_score:.3f}")
            print(f"Test MSE: {mse:.3f}")
            print(f"Test RMSE: {np.sqrt(mse):.3f}")
        
        # Store feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 10 Most Important Features:")
        print(self.feature_importance.head(10))
        
        return X_train, X_test, y_train, y_test
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
            
        X, _ = self.prepare_features(df)
        predictions = self.model.predict(X)
        
        # Convert back to original labels for classification
        if self.task_type == 'classification' and hasattr(self.label_encoder, 'classes_'):
            predictions = self.label_encoder.inverse_transform(predictions)
            
        return predictions
    
    def plot_tree(self, max_depth_display: int = 3, figsize: Tuple[int, int] = (15, 10)):
        """Visualize the decision tree"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
            
        plt.figure(figsize=figsize)
        
        # Limit display depth for readability
        if self.task_type == 'classification':
            class_names = None
            if hasattr(self.label_encoder, 'classes_'):
                class_names = self.label_encoder.classes_.astype(str)
        else:
            class_names = None
            
        tree.plot_tree(
            self.model,
            feature_names=self.feature_names,
            class_names=class_names,
            filled=True,
            max_depth=max_depth_display,
            fontsize=8
        )
        plt.title(f"Cotton Candy Decision Tree ({self.task_type.title()})")
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self, top_n: int = 15):
        """Plot feature importance"""
        if self.feature_importance is None:
            raise ValueError("Model has not been trained yet")
            
        plt.figure(figsize=(10, 8))
        top_features = self.feature_importance.head(top_n)
        
        sns.barplot(data=top_features, y='feature', x='importance')
        plt.title(f'Top {top_n} Feature Importances')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
            
        model_data = {
            'model': self.model,
            'task_type': self.task_type,
            'feature_names': self.feature_names,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_importance': self.feature_importance
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.task_type = model_data['task_type']
        self.feature_names = model_data['feature_names']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_importance = model_data['feature_importance']
        print(f"Model loaded from {filepath}")

def create_sample_dataset_with_targets(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a sample dataset with target variables for demonstration
    This is just for testing - replace with your actual target labels
    """
    df = features_df.copy()
    
    # Add some synthetic target variables for demonstration
    np.random.seed(42)
    
    # Example classification target: Quality (Good/Bad)
    # This would normally come from your actual quality assessments
    quality_probs = np.random.random(len(df))
    df['quality'] = ['Good' if p > 0.6 else 'Bad' for p in quality_probs]
    
    # Example regression target: Final weight
    # This would normally be your actual measured final weight
    if 'weight' in df.columns:
        base_weight = df['weight'].fillna(5.0)
    else:
        base_weight = np.random.normal(5.0, 1.0, len(df))
    df['final_weight'] = base_weight + np.random.normal(0, 0.5, len(df))
    
    return df

# Example usage
if __name__ == "__main__":
    # This assumes you have feature data from the optimized parser
    try:
        # Load features (you would get this from the optimized parser)
        features_df = pd.read_csv("optimized_features.csv")
        
        # Add synthetic targets for demonstration
        dataset = create_sample_dataset_with_targets(features_df)
        
        print("Dataset shape:", dataset.shape)
        print("Columns:", dataset.columns.tolist())
        
        # Example 1: Classification (Quality prediction)
        print("\n" + "="*50)
        print("CLASSIFICATION EXAMPLE: Quality Prediction")
        print("="*50)
        
        classifier = CottonCandyDecisionTree(task_type='classification')
        classifier.train(dataset, target_column='quality')
        classifier.plot_tree()
        classifier.plot_feature_importance()
        
        # Example 2: Regression (Final weight prediction)
        print("\n" + "="*50)
        print("REGRESSION EXAMPLE: Final Weight Prediction")
        print("="*50)
        
        regressor = CottonCandyDecisionTree(task_type='regression')
        regressor.train(dataset, target_column='final_weight')
        regressor.plot_tree()
        regressor.plot_feature_importance()
        
        # Save models
        classifier.save_model("cotton_candy_quality_model.joblib")
        regressor.save_model("cotton_candy_weight_model.joblib")
        
    except FileNotFoundError:
        print("Please run the optimized_log_parser.py first to generate features")
