#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Read the complete dataset
df = pd.read_csv('/Users/nicolas/Desktop/Cotton-Candy-Digital-Twin/DT-0509/Complete_cc_dataset.csv', 
                 sep=';', header=1)

# Clean the data
df = df.dropna(subset=['iteration'])
df = df[df['my_score'] != 'X']
df['my_score'] = pd.to_numeric(df['my_score'], errors='coerce')
df = df.dropna(subset=['my_score'])

print("=== COMPLETE DATASET ANALYSIS vs CURRENT PRESCRIPTIVE MODEL ===")
print(f"Total valid iterations in complete dataset: {len(df)}")
print(f"Quality score range: {df['my_score'].min():.1f} - {df['my_score'].max():.1f}")
print(f"Mean quality score (all data): {df['my_score'].mean():.1f}")

# Get last 60 iterations (our model basis)
last_60 = df.tail(60).copy()
print(f"Mean quality score (last 60 - model basis): {last_60['my_score'].mean():.1f}")

# === CURRENT MODEL LOGIC (for comparison) ===
def current_model_logic(env_h, ir_o):
    """Our current prescriptive model logic"""
    # Cook temp remains constant
    cook_temp = 53.0
    cool_temp = 54.0
    
    # Start temp based on IR
    if ir_o >= 50.0:
        start_temp = 50.0
    elif ir_o >= 40.0:
        start_temp = 48.0
    else:
        start_temp = 45.0
    
    # Cook time based on humidity
    base_cook_time = 75
    if env_h <= 45.0:
        cook_time = base_cook_time
    elif env_h <= 55.0:
        cook_time = base_cook_time + 3
    else:
        cook_time = base_cook_time + 5
    
    # IR adjustment
    if ir_o >= 50.0:
        cook_time += 2
        
    return start_temp, cook_temp, cool_temp, cook_time

# === ANALYSIS 1: CORRELATION COMPARISON ===
print("\n=== CORRELATION ANALYSIS: COMPLETE DATASET vs LAST 60 ===")

# Environmental correlations
env_cols = ['baseline_env_EnvH', 'before_turn_on_env_IrO']
param_cols = ['start_temp', 'cook_temp', 'cooled_temp', 'cook_time']

print("\nEnvironmental conditions vs Quality:")
for col in env_cols:
    if col in df.columns:
        corr_all = df[col].corr(df['my_score'])
        corr_60 = last_60[col].corr(last_60['my_score'])
        print(f"{col}:")
        print(f"  Complete dataset: {corr_all:.3f}")
        print(f"  Last 60:          {corr_60:.3f}")
        print(f"  Difference:       {abs(corr_all - corr_60):.3f}")

print("\nCooking parameters vs Quality:")
for col in param_cols:
    if col in df.columns:
        corr_all = df[col].corr(df['my_score'])
        corr_60 = last_60[col].corr(last_60['my_score'])
        print(f"{col}:")
        print(f"  Complete dataset: {corr_all:.3f}")
        print(f"  Last 60:          {corr_60:.3f}")
        print(f"  Difference:       {abs(corr_all - corr_60):.3f}")

# === ANALYSIS 2: HIGH PERFORMERS COMPARISON ===
print("\n=== HIGH PERFORMERS ANALYSIS (my_score >= 60) ===")

high_perf_all = df[df['my_score'] >= 60]
high_perf_60 = last_60[last_60['my_score'] >= 60]

print(f"High performers in complete dataset: {len(high_perf_all)} ({len(high_perf_all)/len(df)*100:.1f}%)")
print(f"High performers in last 60: {len(high_perf_60)} ({len(high_perf_60)/len(last_60)*100:.1f}%)")

if len(high_perf_all) > 0:
    print(f"\nOptimal parameters (Complete dataset):")
    for param in param_cols:
        if param in high_perf_all.columns:
            mean_all = high_perf_all[param].mean()
            mean_60 = high_perf_60[param].mean() if len(high_perf_60) > 0 else 0
            print(f"  {param}: All={mean_all:.1f}, Last60={mean_60:.1f}")

# === ANALYSIS 3: MODEL VALIDATION ON COMPLETE DATASET ===
print("\n=== CURRENT MODEL VALIDATION ON COMPLETE DATASET ===")

# Apply current model to all data points where we have the required inputs
valid_data = df.dropna(subset=['baseline_env_EnvH', 'before_turn_on_env_IrO'])
print(f"Valid data points for model testing: {len(valid_data)}")

if len(valid_data) > 0:
    # Apply model predictions
    predictions = []
    for idx, row in valid_data.iterrows():
        env_h = row['baseline_env_EnvH']
        ir_o = row['before_turn_on_env_IrO']
        pred_start, pred_cook, pred_cool, pred_time = current_model_logic(env_h, ir_o)
        predictions.append({
            'iteration': row['iteration'],
            'actual_my_score': row['my_score'],
            'actual_start_temp': row.get('start_temp', np.nan),
            'actual_cook_temp': row.get('cook_temp', np.nan),
            'actual_cooled_temp': row.get('cooled_temp', np.nan),
            'actual_cook_time': row.get('cook_time', np.nan),
            'pred_start_temp': pred_start,
            'pred_cook_temp': pred_cook,
            'pred_cool_temp': pred_cool,
            'pred_cook_time': pred_time,
            'env_h': env_h,
            'ir_o': ir_o
        })
    
    pred_df = pd.DataFrame(predictions)
    
    # Calculate parameter differences
    param_diffs = {}
    param_mapping = {
        'start_temp': 'start_temp',
        'cook_temp': 'cook_temp', 
        'cooled_temp': 'cool_temp',  # Fix the column name mismatch
        'cook_time': 'cook_time'
    }
    
    for param, pred_param in param_mapping.items():
        actual_col = f'actual_{param}'
        pred_col = f'pred_{pred_param}'
        if actual_col in pred_df.columns:
            valid_comparisons = pred_df.dropna(subset=[actual_col, pred_col])
            if len(valid_comparisons) > 0:
                diff = np.abs(valid_comparisons[actual_col] - valid_comparisons[pred_col])
                param_diffs[param] = {
                    'mean_abs_diff': diff.mean(),
                    'std_diff': diff.std(),
                    'valid_comparisons': len(valid_comparisons)
                }
    
    print("\nParameter prediction accuracy (Mean Absolute Difference):")
    for param, stats in param_diffs.items():
        print(f"  {param}: {stats['mean_abs_diff']:.2f} ± {stats['std_diff']:.2f} (n={stats['valid_comparisons']})")

# === ANALYSIS 4: ENVIRONMENTAL CONDITIONS EVOLUTION ===
print("\n=== ENVIRONMENTAL CONDITIONS EVOLUTION ===")

# Split dataset into early and recent periods
mid_point = len(df) // 2
early_data = df.iloc[:mid_point]
recent_data = df.iloc[mid_point:]

print(f"Early period: iterations {early_data['iteration'].min():.0f}-{early_data['iteration'].max():.0f} (n={len(early_data)})")
print(f"Recent period: iterations {recent_data['iteration'].min():.0f}-{recent_data['iteration'].max():.0f} (n={len(recent_data)})")

for period_name, period_data in [("Early", early_data), ("Recent", recent_data)]:
    print(f"\n{period_name} period statistics:")
    print(f"  Mean quality score: {period_data['my_score'].mean():.1f}")
    print(f"  Mean env humidity: {period_data['baseline_env_EnvH'].mean():.1f}%")
    print(f"  Mean IR temp: {period_data['before_turn_on_env_IrO'].mean():.1f}°C")
    if 'cook_temp' in period_data.columns:
        cook_temps = period_data['cook_temp'].dropna()
        if len(cook_temps) > 0:
            print(f"  Mean cook temp: {cook_temps.mean():.1f}°C")

# === ANALYSIS 5: MODEL ROBUSTNESS TEST ===
print("\n=== MODEL ROBUSTNESS ACROSS ALL DATA ===")

# Test model performance on different quality ranges
quality_ranges = [(0, 30, "Low"), (30, 50, "Medium"), (50, 70, "Good"), (70, 100, "Excellent")]

for min_q, max_q, label in quality_ranges:
    subset = df[(df['my_score'] >= min_q) & (df['my_score'] < max_q)]
    if len(subset) > 0:
        print(f"\n{label} Quality Range ({min_q}-{max_q}): n={len(subset)}")
        if 'baseline_env_EnvH' in subset.columns:
            print(f"  Avg humidity: {subset['baseline_env_EnvH'].mean():.1f}%")
        if 'before_turn_on_env_IrO' in subset.columns:
            print(f"  Avg IR temp: {subset['before_turn_on_env_IrO'].mean():.1f}°C")
        
        # Check how our model would perform
        optimal_conditions = subset['baseline_env_EnvH'] <= 50  # Our model's optimal zone
        if len(subset) > 0:
            optimal_pct = optimal_conditions.sum() / len(subset) * 100
            print(f"  % in optimal conditions (≤50% humidity): {optimal_pct:.1f}%")

# === VISUAL COMPARISON ===
print("\n=== CREATING COMPARISON VISUALIZATIONS ===")

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# 1. Correlation comparison
correlations_all = [df[col].corr(df['my_score']) for col in env_cols + param_cols if col in df.columns]
correlations_60 = [last_60[col].corr(last_60['my_score']) for col in env_cols + param_cols if col in last_60.columns]
param_names = [col.replace('_', ' ').title() for col in env_cols + param_cols if col in df.columns]

x = np.arange(len(param_names))
width = 0.35

bars1 = ax1.bar(x - width/2, correlations_all, width, label='Complete Dataset', alpha=0.8)
bars2 = ax1.bar(x + width/2, correlations_60, width, label='Last 60 (Model Basis)', alpha=0.8)

ax1.set_ylabel('Correlation with Quality Score')
ax1.set_title('Correlation Comparison: Complete Dataset vs Model Basis')
ax1.set_xticks(x)
ax1.set_xticklabels(param_names, rotation=45, ha='right')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)

# 2. Quality distribution over time
ax2.scatter(df['iteration'], df['my_score'], alpha=0.6, s=30)
ax2.axhline(y=df['my_score'].mean(), color='red', linestyle='--', label=f'Overall Mean: {df["my_score"].mean():.1f}')
ax2.axhline(y=60, color='green', linestyle='--', label='High Quality Threshold')
ax2.set_xlabel('Iteration Number')
ax2.set_ylabel('Quality Score')
ax2.set_title('Quality Score Evolution Over All Iterations')
ax2.legend()

# Highlight last 60 iterations
last_60_iterations = df.tail(60)
ax2.scatter(last_60_iterations['iteration'], last_60_iterations['my_score'], 
           color='orange', s=50, alpha=0.8, label='Last 60 (Model Basis)')
ax2.legend()

# 3. Environmental conditions distribution
ax3.hist(df['baseline_env_EnvH'], bins=20, alpha=0.7, label='Complete Dataset', density=True)
ax3.hist(last_60['baseline_env_EnvH'], bins=15, alpha=0.7, label='Last 60', density=True)
ax3.axvline(x=50, color='red', linestyle='--', label='Model Threshold (50%)')
ax3.set_xlabel('Environmental Humidity (%)')
ax3.set_ylabel('Density')
ax3.set_title('Environmental Humidity Distribution Comparison')
ax3.legend()

# 4. Model prediction zones on all data
if len(valid_data) > 0:
    scatter = ax4.scatter(valid_data['baseline_env_EnvH'], valid_data['before_turn_on_env_IrO'], 
                         c=valid_data['my_score'], cmap='RdYlGn', alpha=0.7, s=40)
    plt.colorbar(scatter, ax=ax4, label='Quality Score')
    
    # Add model decision boundaries
    ax4.axvline(x=45, color='blue', linestyle='--', alpha=0.8, label='Humidity Thresholds')
    ax4.axvline(x=55, color='blue', linestyle='--', alpha=0.8)
    ax4.axhline(y=40, color='red', linestyle='--', alpha=0.8, label='IR Temp Thresholds')
    ax4.axhline(y=50, color='red', linestyle='--', alpha=0.8)
    
    ax4.set_xlabel('Environmental Humidity (%)')
    ax4.set_ylabel('IR Object Temperature (°C)')
    ax4.set_title('Model Decision Zones vs All Data Points')
    ax4.legend()

plt.tight_layout()
plt.savefig('/Users/nicolas/Desktop/Cotton-Candy-Digital-Twin/graphs/complete_dataset_vs_model_comparison.png', 
           dpi=300, bbox_inches='tight')
plt.show()

print(f"\n=== SUMMARY ===")
print(f"✅ Analysis complete! Comparison visualization saved as:")
print(f"   graphs/complete_dataset_vs_model_comparison.png")
print(f"\n📊 Key findings will be displayed above.")