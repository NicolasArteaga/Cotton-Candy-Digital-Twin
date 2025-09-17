#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

# Set style for publication-quality plots
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

# Get last 60 iterations
last_60 = df.tail(60).copy()

print("Creating clean correlation comparison graph...")

# Clean correlation comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left plot: Complete dataset
scatter1 = ax1.scatter(df['baseline_env_EnvH'], df['my_score'], 
                      c=df['my_score'], cmap='RdYlGn', alpha=0.7, s=50, edgecolors='black', linewidth=0.3)
ax1.set_xlabel('Environmental Humidity (%)', fontsize=13)
ax1.set_ylabel('Cotton Candy Quality Score', fontsize=13)
ax1.set_title('Complete Dataset (n=150)', fontsize=15, fontweight='bold')

# Add trend line for complete dataset
z1 = np.polyfit(df['baseline_env_EnvH'], df['my_score'], 1)
p1 = np.poly1d(z1)
ax1.plot(df['baseline_env_EnvH'], p1(df['baseline_env_EnvH']), "r-", alpha=0.8, linewidth=2.5)

# Calculate correlation
corr_all = df['baseline_env_EnvH'].corr(df['my_score'])
ax1.text(0.95, 0.95, f'r = {corr_all:.3f}', transform=ax1.transAxes, 
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
         verticalalignment='top', horizontalalignment='right', fontsize=13, fontweight='bold')

ax1.grid(True, alpha=0.3)
ax1.set_xlim(40, 75)
ax1.set_ylim(20, 80)

# Right plot: Last 60 iterations
scatter2 = ax2.scatter(last_60['baseline_env_EnvH'], last_60['my_score'], 
                      c=last_60['my_score'], cmap='RdYlGn', alpha=0.8, s=60, edgecolors='black', linewidth=0.4)
ax2.set_xlabel('Environmental Humidity (%)', fontsize=13)
ax2.set_ylabel('Cotton Candy Quality Score', fontsize=13)
ax2.set_title('Recent 60 Iterations', fontsize=15, fontweight='bold')

# Add trend line for last 60
z2 = np.polyfit(last_60['baseline_env_EnvH'], last_60['my_score'], 1)
p2 = np.poly1d(z2)
ax2.plot(last_60['baseline_env_EnvH'], p2(last_60['baseline_env_EnvH']), "r-", alpha=0.8, linewidth=2.5)

# Calculate correlation
corr_60 = last_60['baseline_env_EnvH'].corr(last_60['my_score'])
ax2.text(0.95, 0.95, f'r = {corr_60:.3f}', transform=ax2.transAxes, 
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
         verticalalignment='top', horizontalalignment='right', fontsize=13, fontweight='bold')

ax2.grid(True, alpha=0.3)
ax2.set_xlim(40, 75)
ax2.set_ylim(20, 80)

# Strong arrow between plots
ax1.annotate('', xy=(1.05, 0.5), xytext=(0.95, 0.5), xycoords='axes fraction',
            arrowprops=dict(arrowstyle='->', lw=5, color='darkblue'))

plt.tight_layout()
plt.savefig('/Users/nicolas/Desktop/Cotton-Candy-Digital-Twin/graphs/clean_correlation_comparison.png', 
           dpi=300, bbox_inches='tight')
plt.close()

print("✅ Clean correlation comparison graph created!")
print(f"📊 Correlations: Complete dataset r = {corr_all:.3f}, Recent 60 r = {corr_60:.3f}")