#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Read the CSV file
df = pd.read_csv('/Users/nicolas/Desktop/Cotton-Candy-Digital-Twin/DT-0509/Complete_cc_dataset.csv', 
                 sep=';', header=1)

# Filter out rows with missing iteration numbers or invalid my_score
df = df.dropna(subset=['iteration'])
df = df[df['my_score'] != 'X']
df['my_score'] = pd.to_numeric(df['my_score'], errors='coerce')
df = df.dropna(subset=['my_score'])

# Get the last 60 valid iterations for analysis
last_60 = df.tail(60).copy()

print("Creating visualizations from the last 60 iterations...")

# Create figure with multiple subplots
fig = plt.figure(figsize=(16, 12))

# 1. Humidity vs Quality Score Scatter Plot
plt.subplot(2, 3, 1)
scatter = plt.scatter(last_60['baseline_env_EnvH'], last_60['my_score'], 
                     c=last_60['my_score'], cmap='RdYlGn', alpha=0.7, s=60)
plt.colorbar(scatter, label='Quality Score')
plt.xlabel('Environmental Humidity (%)')
plt.ylabel('Quality Score (my_score)')
plt.title('Environmental Humidity vs Quality Score\n(r = -0.548)')
z = np.polyfit(last_60['baseline_env_EnvH'], last_60['my_score'], 1)
p = np.poly1d(z)
plt.plot(last_60['baseline_env_EnvH'], p(last_60['baseline_env_EnvH']), "r--", alpha=0.8)

# 2. Cook Time Distribution by Quality Ranges
plt.subplot(2, 3, 2)
quality_ranges = ['Low (0-30)', 'Medium (30-50)', 'High (50-70)', 'Excellent (70+)']
last_60['quality_category'] = pd.cut(last_60['my_score'], 
                                    bins=[0, 30, 50, 70, 100], 
                                    labels=quality_ranges)
cook_times_by_quality = [last_60[last_60['quality_category'] == cat]['cook_time'].values 
                        for cat in quality_ranges]
cook_times_by_quality = [ct for ct in cook_times_by_quality if len(ct) > 0]
quality_labels = [label for i, label in enumerate(quality_ranges) 
                 if len(last_60[last_60['quality_category'] == label]) > 0]

plt.boxplot(cook_times_by_quality, labels=quality_labels)
plt.ylabel('Cook Time (seconds)')
plt.title('Cook Time Distribution by Quality Category')
plt.xticks(rotation=45)

# 3. Parameter Optimization: High vs Low Performers
plt.subplot(2, 3, 3)
high_performers = last_60[last_60['my_score'] >= 60]
low_performers = last_60[last_60['my_score'] < 40]

params = ['start_temp', 'cook_temp', 'cooled_temp', 'cook_time']
high_means = [high_performers[param].mean() for param in params]
low_means = [low_performers[param].mean() for param in params]

x = np.arange(len(params))
width = 0.35

plt.bar(x - width/2, high_means, width, label='High Performers (≥60)', alpha=0.8)
plt.bar(x + width/2, low_means, width, label='Low Performers (<40)', alpha=0.8)

plt.ylabel('Parameter Values')
plt.title('Parameter Comparison: High vs Low Performers')
plt.xticks(x, ['Start Temp\n(°C)', 'Cook Temp\n(°C)', 'Cool Temp\n(°C)', 'Cook Time\n(s)'])
plt.legend()

# 4. Environmental Conditions Heatmap
plt.subplot(2, 3, 4)
# Create humidity and IR temperature bins
last_60['humidity_bin'] = pd.cut(last_60['baseline_env_EnvH'], 
                                bins=[0, 45, 55, 65, 100], 
                                labels=['Low\n(≤45%)', 'Normal\n(45-55%)', 'High\n(55-65%)', 'Very High\n(>65%)'])
last_60['ir_bin'] = pd.cut(last_60['before_turn_on_env_IrO'], 
                          bins=[0, 40, 50, 60, 100], 
                          labels=['Low\n(≤40°C)', 'Medium\n(40-50°C)', 'High\n(50-60°C)', 'Very High\n(>60°C)'])

# Create pivot table for heatmap
heatmap_data = last_60.pivot_table(values='my_score', 
                                  index='humidity_bin', 
                                  columns='ir_bin', 
                                  aggfunc='mean')
sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn', 
           cbar_kws={'label': 'Average Quality Score'})
plt.title('Quality Score by Environmental Conditions')
plt.xlabel('IR Temperature Before Turn-on')
plt.ylabel('Environmental Humidity')

# 5. Time Series of Quality Scores
plt.subplot(2, 3, 5)
plt.plot(last_60['iteration'], last_60['my_score'], 'o-', alpha=0.7, markersize=4)
plt.axhline(y=60, color='green', linestyle='--', alpha=0.7, label='High Quality Threshold')
plt.axhline(y=last_60['my_score'].mean(), color='red', linestyle='--', alpha=0.7, label='Average')
plt.xlabel('Iteration Number')
plt.ylabel('Quality Score')
plt.title('Quality Score Trend Over Last 60 Iterations')
plt.legend()

# 6. Optimal Parameter Combinations
plt.subplot(2, 3, 6)
# Show optimal combinations for different humidity ranges
humidity_ranges = last_60['humidity_bin'].unique()
colors = plt.cm.Set3(np.linspace(0, 1, len(humidity_ranges)))

for i, hum_range in enumerate(humidity_ranges):
    if pd.isna(hum_range):
        continue
    subset = last_60[last_60['humidity_bin'] == hum_range]
    if len(subset) > 0:
        plt.scatter(subset['cook_time'], subset['my_score'], 
                   c=[colors[i]], label=f'{hum_range}', alpha=0.7, s=60)

plt.xlabel('Cook Time (seconds)')
plt.ylabel('Quality Score')
plt.title('Cook Time vs Quality by Humidity Range')
plt.legend(title='Humidity Range', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig('/Users/nicolas/Desktop/Cotton-Candy-Digital-Twin/prescriptive_model_analysis.png', 
           dpi=300, bbox_inches='tight')
plt.show()

# Create a second figure with correlation matrix
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Correlation Matrix
correlation_vars = ['baseline_env_EnvH', 'before_turn_on_env_IrO', 
                   'start_temp', 'cook_temp', 'cooled_temp', 'cook_time', 'my_score']
corr_matrix = last_60[correlation_vars].corr()

sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
           square=True, ax=ax1, cbar_kws={'label': 'Correlation Coefficient'})
ax1.set_title('Correlation Matrix: Environmental Conditions vs Parameters vs Quality')

# Performance Distribution
ax2.hist(last_60['my_score'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
ax2.axvline(last_60['my_score'].mean(), color='red', linestyle='--', 
           label=f'Mean: {last_60["my_score"].mean():.1f}')
ax2.axvline(last_60['my_score'].median(), color='green', linestyle='--', 
           label=f'Median: {last_60["my_score"].median():.1f}')
ax2.set_xlabel('Quality Score')
ax2.set_ylabel('Frequency')
ax2.set_title('Quality Score Distribution (Last 60 Iterations)')
ax2.legend()

plt.tight_layout()
plt.savefig('/Users/nicolas/Desktop/Cotton-Candy-Digital-Twin/correlation_and_distribution.png', 
           dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics
print("\n=== SUMMARY STATISTICS ===")
print(f"Total iterations analyzed: {len(last_60)}")
print(f"Quality score range: {last_60['my_score'].min():.1f} - {last_60['my_score'].max():.1f}")
print(f"Mean quality score: {last_60['my_score'].mean():.1f}")
print(f"High performers (≥60): {len(last_60[last_60['my_score'] >= 60])} ({len(last_60[last_60['my_score'] >= 60])/len(last_60)*100:.1f}%)")

print(f"\nOptimal conditions for high performers:")
high_perf = last_60[last_60['my_score'] >= 60]
if len(high_perf) > 0:
    print(f"- Humidity range: {high_perf['baseline_env_EnvH'].min():.1f}% - {high_perf['baseline_env_EnvH'].max():.1f}%")
    print(f"- IR temp range: {high_perf['before_turn_on_env_IrO'].min():.1f}°C - {high_perf['before_turn_on_env_IrO'].max():.1f}°C")
    print(f"- Cook time range: {high_perf['cook_time'].min():.0f}s - {high_perf['cook_time'].max():.0f}s")

print("\nGraphs saved as:")
print("- prescriptive_model_analysis.png")
print("- correlation_and_distribution.png")