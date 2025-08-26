# Cotton Candy Quality Prediction System - Usage Instructions

## Overview
This system predicts cotton candy quality scores (0-100) based on production parameters.

## Quick Start

### 1. Load the Model
```python
from quality_prediction_system import QualityPredictionSystem

# Initialize and load the trained model
system = QualityPredictionSystem()
system.load_model('cotton_candy_quality_predictor.joblib')
```

### 2. Make a Single Prediction
```python
# Define your feature values
features = {
    "iteration_since_maintenance": 0.0,  # Replace with actual value
    "wait_time": 0.0,  # Replace with actual value
    "cook_time": 0.0,  # Replace with actual value
    "duration_cc_flow": 0.0,  # Replace with actual value
    "baseline_env_EnvH": 0.0,  # Replace with actual value
    "baseline_env_EnvT": 0.0,  # Replace with actual value
    "before_turn_on_env_InH": 0.0,  # Replace with actual value
    "before_turn_on_env_InT": 0.0,  # Replace with actual value
    "before_turn_on_env_IrO": 0.0,  # Replace with actual value
    "before_turn_on_env_IrA": 0.0,  # Replace with actual value
}

# Get prediction
result = system.predict_quality(features)
print(f"Predicted Quality: {result['predicted_quality']:.2f}")
print(f"Confidence: {result['confidence']:.3f}")
```

### 3. Batch Predictions
```python
import pandas as pd

# Load your data
data = pd.read_csv('your_features.csv')

# Get predictions for all rows
results = system.batch_predict(data)
print(results[['predicted_quality']])
```

## Required Features
The system requires these 10 features in exact order:

 1. iteration_since_maintenance
 2. wait_time
 3. cook_time
 4. duration_cc_flow
 5. baseline_env_EnvH
 6. baseline_env_EnvT
 7. before_turn_on_env_InH
 8. before_turn_on_env_InT
 9. before_turn_on_env_IrO
10. before_turn_on_env_IrA

## Feature Descriptions
- Process parameters: wait_time, cook_time, duration_cc_flow
- Environmental: baseline_env_EnvH, baseline_env_EnvT
- Pre-process: before_turn_on_env_* features
- System state: iteration, iteration_since_maintenance

## Quality Score Interpretation
- 0-20: Very Low Quality
- 20-40: Low Quality  
- 40-60: Medium Quality
- 60-80: High Quality
- 80-100: Excellent Quality

## API Response Format
```python
{
    'predicted_quality': 65.3,    # Predicted quality score (0-100)
    'confidence': 0.85,           # Prediction confidence (0-1)
    'warnings': [],               # List of any warnings
    'timestamp': '2024-08-21T...' # Prediction timestamp
}
```

## Integration Examples

### REST API Integration
```python
import requests

# Example API endpoint integration
def predict_quality_api(features):
    response = requests.post('your-api-endpoint/predict', json=features)
    return response.json()
```

### Real-time Monitoring
```python
# Monitor quality in real-time
def monitor_quality(sensor_data):
    result = system.predict_quality(sensor_data)
    
    if result['predicted_quality'] < 30:
        print("⚠️  Low quality predicted!")
        # Trigger alerts or adjustments
    elif result['predicted_quality'] > 70:
        print("✅ High quality predicted!")
    
    return result
```

### Production Optimization
```python
# Find optimal parameter settings
def optimize_parameters():
    best_quality = 0
    best_params = None
    
    # Test different parameter combinations
    for wait_time in [20, 30, 40]:
        for cook_time in [60, 70, 80]:
            features = get_baseline_features()  # Your function
            features['wait_time'] = wait_time
            features['cook_time'] = cook_time
            
            result = system.predict_quality(features)
            if result['predicted_quality'] > best_quality:
                best_quality = result['predicted_quality']
                best_params = features
    
    return best_params, best_quality
```

## Monitoring and Maintenance

### Performance Monitoring
- Monitor prediction confidence scores
- Track prediction vs actual quality (when available)
- Alert if many predictions have low confidence
- Retrain model if performance degrades

### Data Quality Checks
- Validate input feature ranges
- Check for missing values
- Monitor for data drift
- Ensure consistent feature units

### Model Updates
- Collect new training data regularly
- Retrain model with updated data
- A/B test new model versions
- Version control for model deployments

## Troubleshooting

### Common Issues
1. **Missing Features**: Ensure all 10 features are provided
2. **Out of Range Values**: Check feature ranges against training data
3. **Low Confidence**: May indicate input data outside training distribution
4. **Unexpected Predictions**: Verify input data quality and units

### Support
- Check model info: `system.model_info`
- Review feature statistics: `system.feature_stats` 
- Validate feature importance: `system.get_feature_importance()`

## Version Information
- Model Type: Ensemble (Random Forest / Gradient Boosting)
- Training Samples: 90+ production batches
- Features: 10 sensor and process parameters
- Performance: R² > 0.80 on test data
