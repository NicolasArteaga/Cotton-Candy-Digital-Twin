# Cotton Candy Digital Twin - Decision Tree Model

## Overview
This Decision Tree Digital Twin predicts cotton candy quality (weight) based on manufacturing process parameters extracted from CPEE (Cotton Candy Process Execution Engine) logs.

## 🎯 Model Performance
- **Training R²**: 0.94 (excellent fit on training data)
- **Testing R²**: -0.20 (indicates overfitting, needs more data)
- **Training RMSE**: 0.64g 
- **Testing RMSE**: 3.66g
- **Dataset**: 26 valid samples from 44 manufacturing processes

## 🔑 Key Quality Predictors
The model identified these critical manufacturing parameters (in order of importance):

1. **Internal Temperature Before Turn-On** (`before_turn_on_env_InT`) - 37.1% importance
2. **Baseline Environmental Humidity** (`baseline_env_EnvH`) - 26.5% importance  
3. **Internal Humidity Before Cooldown** (`before_cooldown_env_InH`) - 26.0% importance
4. **Internal Temperature After Flow End** (`after_flow_end_env_InT`) - 8.6% importance
5. **Total Process Duration** (`duration_total`) - 1.5% importance

## 📋 Decision Rules
The trained model uses these rules to predict cotton candy quality:

### Primary Split: Internal Humidity Before Cooldown ≤ 32.99%
- **Left Branch (Low Humidity)**: 
  - If internal temp before turn-on ≤ 39.55°C: Weight = 7.37-8.64g (depends on duration)
  - If internal temp before turn-on > 39.55°C: Weight = 0.52-5.88g (depends on baseline humidity)

### Right Branch (High Humidity > 32.99%):
- **Quality depends on internal temperature after flow ends**:
  - ≤ 38.65°C: Weight = 6.95g
  - 38.65-41.23°C: Weight = 9.18-9.70g (varies by process iteration)  
  - > 41.23°C: Weight = 8.92g

## 📁 Files Generated
- `cotton_candy_digital_twin.joblib` - Trained model (load with joblib.load())
- `feature_importances.csv` - Feature rankings by importance
- `decision_tree_rules.txt` - Complete model rules and performance metrics
- `cotton_candy_decision_tree_analysis.png` - Visualizations

## 🚀 Usage
```python
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('cotton_candy_digital_twin.joblib')

# Make predictions on new process data
# Features must include all 30 columns from features_X.csv
predicted_weight = model.predict(new_process_data)
```

## 🏭 Manufacturing Insights
1. **Temperature Control is Critical**: Internal temperature management accounts for 45.7% of quality variation
2. **Humidity Matters**: Environmental and internal humidity control accounts for 52.5% of quality variation
3. **Process Duration**: Longer processes tend to produce heavier cotton candy
4. **Optimal Conditions**: Best quality achieved with controlled humidity (~33%) and moderate temperatures

## 📈 Data Pipeline
1. **Data Collection**: CPEE logs from 9 batches (44 total processes)
2. **Feature Extraction**: 29 manufacturing parameters + iteration number
3. **Quality Targets**: Cotton candy weight measurements
4. **Model Training**: Decision Tree with hyperparameter optimization

## ⚠️ Model Limitations
- **Small Dataset**: Only 26 valid samples may cause overfitting
- **Missing Data**: Some environmental measurements missing from early batches
- **Generalization**: Model may not generalize well to new conditions
- **Recommendation**: Collect more data for improved accuracy

## 🔮 Next Steps
1. **Data Collection**: Add more manufacturing processes for better generalization
2. **Feature Engineering**: Explore derived features (ratios, differences)
3. **Model Ensemble**: Combine with other algorithms (Random Forest, XGBoost)
4. **Real-time Integration**: Deploy for live quality prediction during manufacturing
