#!/usr/bin/env python3
import pandas as pd
import numpy as np

# Read the CSV file
df = pd.read_csv('/Users/nicolas/Desktop/Cotton-Candy-Digital-Twin/DT-0509/Complete_cc_dataset.csv', 
                 sep=';', header=1)  # Skip the first header row

# Filter out rows with missing iteration numbers or invalid my_score
df = df.dropna(subset=['iteration'])
df = df[df['my_score'] != 'X']
df['my_score'] = pd.to_numeric(df['my_score'], errors='coerce')
df = df.dropna(subset=['my_score'])

# Get the last 60 valid iterations
last_60 = df.tail(60).copy()

print("=== ANALYSIS OF LAST 60 ITERATIONS ===")
print(f"Total valid iterations analyzed: {len(last_60)}")
print(f"Iteration range: {last_60['iteration'].min()} to {last_60['iteration'].max()}")
print(f"My_score range: {last_60['my_score'].min()} to {last_60['my_score'].max()}")

# Key columns for analysis
env_cols = ['baseline_env_EnvH', 'before_turn_on_env_IrO']
param_cols = ['start_temp', 'cook_temp', 'cooled_temp', 'cook_time']
target_col = 'my_score'

print("\n=== CORRELATION ANALYSIS ===")
print("Correlation between environmental conditions and my_score:")
for col in env_cols:
    if col in last_60.columns:
        corr = last_60[col].corr(last_60[target_col])
        print(f"{col}: {corr:.3f}")

print("\nCorrelation between cooking parameters and my_score:")
for col in param_cols:
    if col in last_60.columns:
        corr = last_60[col].corr(last_60[target_col])
        print(f"{col}: {corr:.3f}")

# Find top performing combinations
print(f"\n=== TOP 10 PERFORMING COMBINATIONS (highest my_score) ===")
top_10 = last_60.nlargest(10, 'my_score')

for idx, row in top_10.iterrows():
    print(f"Iteration {int(row['iteration'])}: my_score = {row['my_score']}")
    print(f"  baseline_env_EnvH: {row['baseline_env_EnvH']:.2f}%")
    print(f"  before_turn_on_env_IrO: {row['before_turn_on_env_IrO']:.2f}°C")
    print(f"  start_temp: {row['start_temp']:.1f}°C, cook_temp: {row['cook_temp']:.1f}°C")
    print(f"  cooled_temp: {row['cooled_temp']:.1f}°C, cook_time: {row['cook_time']:.0f}s")
    print()

# Analyze patterns by environmental condition ranges
print("=== PATTERN ANALYSIS BY ENVIRONMENTAL CONDITIONS ===")

# Group by humidity ranges
last_60['humidity_range'] = pd.cut(last_60['baseline_env_EnvH'], 
                                  bins=[0, 50, 60, 70, 100], 
                                  labels=['Low (≤50)', 'Medium (50-60)', 'High (60-70)', 'Very High (>70)'])

print("Performance by baseline humidity ranges:")
humidity_analysis = last_60.groupby('humidity_range').agg({
    'my_score': ['mean', 'max', 'count'],
    'start_temp': 'mean',
    'cook_temp': 'mean', 
    'cooled_temp': 'mean',
    'cook_time': 'mean'
}).round(2)
print(humidity_analysis)

# Group by IR temperature ranges
last_60['ir_temp_range'] = pd.cut(last_60['before_turn_on_env_IrO'], 
                                 bins=[0, 30, 40, 50, 100], 
                                 labels=['Low (≤30)', 'Medium (30-40)', 'High (40-50)', 'Very High (>50)'])

print(f"\nPerformance by IR temperature ranges:")
ir_analysis = last_60.groupby('ir_temp_range').agg({
    'my_score': ['mean', 'max', 'count'],
    'start_temp': 'mean',
    'cook_temp': 'mean',
    'cooled_temp': 'mean', 
    'cook_time': 'mean'
}).round(2)
print(ir_analysis)

# Find optimal parameter combinations for high-scoring iterations (my_score >= 60)
high_scoring = last_60[last_60['my_score'] >= 60]
print(f"\n=== OPTIMAL PARAMETERS FOR HIGH-SCORING ITERATIONS (my_score ≥ 60) ===")
print(f"Number of high-scoring iterations: {len(high_scoring)}")

if len(high_scoring) > 0:
    print("Average parameters for high-scoring iterations:")
    optimal_params = high_scoring[param_cols + env_cols].mean()
    for param in param_cols + env_cols:
        print(f"  {param}: {optimal_params[param]:.2f}")
    
    print(f"\nParameter ranges for high-scoring iterations:")
    for param in param_cols:
        min_val = high_scoring[param].min()
        max_val = high_scoring[param].max()
        print(f"  {param}: {min_val:.1f} - {max_val:.1f}")

# Specific combinations analysis
print(f"\n=== SPECIFIC HIGH-PERFORMING COMBINATIONS ===")
print("Looking for patterns in parameter combinations...")

# Find the best combination for different environmental conditions
best_combinations = []
for humidity_range in last_60['humidity_range'].unique():
    if pd.isna(humidity_range):
        continue
    subset = last_60[last_60['humidity_range'] == humidity_range]
    if len(subset) > 0:
        best_in_range = subset.loc[subset['my_score'].idxmax()]
        best_combinations.append({
            'condition': f'Humidity {humidity_range}',
            'my_score': best_in_range['my_score'],
            'baseline_env_EnvH': best_in_range['baseline_env_EnvH'],
            'before_turn_on_env_IrO': best_in_range['before_turn_on_env_IrO'],
            'start_temp': best_in_range['start_temp'],
            'cook_temp': best_in_range['cook_temp'],
            'cooled_temp': best_in_range['cooled_temp'],
            'cook_time': best_in_range['cook_time']
        })

for combo in best_combinations:
    print(f"{combo['condition']}: my_score = {combo['my_score']}")
    print(f"  EnvH: {combo['baseline_env_EnvH']:.1f}%, IrO: {combo['before_turn_on_env_IrO']:.1f}°C")
    print(f"  Params: start={combo['start_temp']:.1f}°C, cook={combo['cook_temp']:.1f}°C, cool={combo['cooled_temp']:.1f}°C, time={combo['cook_time']:.0f}s")
    print()