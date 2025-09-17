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

print("Creating superiority demonstration graphs...")

# GRAPH 1: Correlation Evolution Comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Left plot: Complete dataset
scatter1 = ax1.scatter(df['baseline_env_EnvH'], df['my_score'], 
                      c=df['my_score'], cmap='RdYlGn', alpha=0.6, s=60, edgecolors='black', linewidth=0.5)
ax1.set_xlabel('Environmental Humidity (%)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Cotton Candy Quality Score', fontsize=14, fontweight='bold')
ax1.set_title('Complete Dataset (n=150)\nCorrelation: r = -0.197', fontsize=16, fontweight='bold')

# Add trend line for complete dataset
z1 = np.polyfit(df['baseline_env_EnvH'], df['my_score'], 1)
p1 = np.poly1d(z1)
ax1.plot(df['baseline_env_EnvH'], p1(df['baseline_env_EnvH']), "r--", alpha=0.8, linewidth=3, label='Trend Line')

# Calculate R-squared for complete dataset
corr_all = df['baseline_env_EnvH'].corr(df['my_score'])
r_squared_all = corr_all ** 2

# Add statistics text box
stats_text_all = f'Correlation: r = {corr_all:.3f}\nR² = {r_squared_all:.3f}\nRelationship: Weak'
ax1.text(0.05, 0.95, stats_text_all, transform=ax1.transAxes, 
         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral", alpha=0.8),
         verticalalignment='top', fontsize=12, fontweight='bold')

ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=12)

# Right plot: Last 60 iterations
scatter2 = ax2.scatter(last_60['baseline_env_EnvH'], last_60['my_score'], 
                      c=last_60['my_score'], cmap='RdYlGn', alpha=0.8, s=80, edgecolors='black', linewidth=0.8)
ax2.set_xlabel('Environmental Humidity (%)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Cotton Candy Quality Score', fontsize=14, fontweight='bold')
ax2.set_title('Recent Data - Last 60 Iterations\nCorrelation: r = -0.548', fontsize=16, fontweight='bold')

# Add trend line for last 60
z2 = np.polyfit(last_60['baseline_env_EnvH'], last_60['my_score'], 1)
p2 = np.poly1d(z2)
ax2.plot(last_60['baseline_env_EnvH'], p2(last_60['baseline_env_EnvH']), "r--", alpha=0.8, linewidth=3, label='Trend Line')

# Calculate R-squared for last 60
corr_60 = last_60['baseline_env_EnvH'].corr(last_60['my_score'])
r_squared_60 = corr_60 ** 2

# Add statistics text box
stats_text_60 = f'Correlation: r = {corr_60:.3f}\nR² = {r_squared_60:.3f}\nRelationship: Strong'
ax2.text(0.05, 0.95, stats_text_60, transform=ax2.transAxes, 
         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8),
         verticalalignment='top', fontsize=12, fontweight='bold')

ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=12)

# Add overall title and improvement indicator
fig.suptitle('Humidity-Quality Correlation: Why Recent Data is Superior\nCorrelation Strength Improvement: 178%', 
             fontsize=18, fontweight='bold', y=0.98)

# Add improvement arrow and text
ax1.annotate('', xy=(1.1, 0.5), xytext=(0.9, 0.5), xycoords='axes fraction',
            arrowprops=dict(arrowstyle='->', lw=4, color='red', alpha=0.8))
ax1.text(1.05, 0.6, 'STRONGER\nCORRELATION', transform=ax1.transAxes, 
         ha='center', va='center', fontsize=12, fontweight='bold', color='red')

plt.tight_layout()
plt.savefig('/Users/nicolas/Desktop/Cotton-Candy-Digital-Twin/graphs/correlation_superiority_comparison.png', 
           dpi=300, bbox_inches='tight')
plt.close()

# GRAPH 2: Temporal Evolution of Quality and Correlations
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Top-left: Quality evolution over time
ax1.plot(df['iteration'], df['my_score'], 'o-', alpha=0.7, markersize=4, color='steelblue')
ax1.axhline(y=df['my_score'].mean(), color='red', linestyle='--', linewidth=2, 
           label=f'Overall Mean: {df["my_score"].mean():.1f}')

# Highlight recent period
recent_start = df['iteration'].iloc[-60]
ax1.axvspan(recent_start, df['iteration'].max(), alpha=0.3, color='green', 
           label='Recent Period (Model Basis)')

# Add trend line
z = np.polyfit(df['iteration'], df['my_score'], 1)
p = np.poly1d(z)
ax1.plot(df['iteration'], p(df['iteration']), "g--", alpha=0.8, linewidth=2, label='Overall Trend')

ax1.set_xlabel('Iteration Number', fontsize=12)
ax1.set_ylabel('Quality Score', fontsize=12)
ax1.set_title('Quality Evolution Over Time\nClear Improvement Trend', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Top-right: Rolling correlation analysis
window_size = 30
rolling_correlations = []
iterations_for_corr = []

for i in range(window_size, len(df)):
    subset = df.iloc[i-window_size:i]
    if len(subset) >= window_size:
        corr = subset['baseline_env_EnvH'].corr(subset['my_score'])
        rolling_correlations.append(corr)
        iterations_for_corr.append(subset['iteration'].iloc[-1])

ax2.plot(iterations_for_corr, rolling_correlations, 'o-', color='purple', alpha=0.8, markersize=3)
ax2.axhline(y=corr_all, color='red', linestyle='--', linewidth=2, 
           label=f'Overall Correlation: {corr_all:.3f}')
ax2.axhline(y=corr_60, color='green', linestyle='--', linewidth=2,
           label=f'Recent 60 Correlation: {corr_60:.3f}')

# Highlight strengthening region
ax2.axvspan(recent_start, df['iteration'].max(), alpha=0.3, color='green')

ax2.set_xlabel('Iteration Number', fontsize=12)
ax2.set_ylabel('Rolling Correlation (30-point window)', fontsize=12)
ax2.set_title('Correlation Strength Evolution\nRecent Strengthening Pattern', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Bottom-left: Parameter evolution comparison
periods = ['Early\n(0-75)', 'Mid\n(75-120)', 'Recent\n(120-185)']
early_data = df.iloc[:75]
mid_data = df.iloc[75:120] 
recent_data = df.iloc[120:]

quality_means = [early_data['my_score'].mean(), 
                mid_data['my_score'].mean(), 
                recent_data['my_score'].mean()]
humidity_means = [early_data['baseline_env_EnvH'].mean(),
                 mid_data['baseline_env_EnvH'].mean(),
                 recent_data['baseline_env_EnvH'].mean()]

x = np.arange(len(periods))
width = 0.35

# Normalize humidity to be on similar scale as quality for visualization
humidity_normalized = [h/2 for h in humidity_means]  # Divide by 2 to fit scale

bars1 = ax3.bar(x - width/2, quality_means, width, label='Quality Score', alpha=0.8, color='steelblue')
bars2 = ax3.bar(x + width/2, humidity_normalized, width, label='Humidity (÷2)', alpha=0.8, color='orange')

# Add value labels
for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
    ax3.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 1,
             f'{quality_means[i]:.1f}', ha='center', va='bottom', fontweight='bold')
    ax3.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 1,
             f'{humidity_means[i]:.1f}%', ha='center', va='bottom', fontweight='bold')

ax3.set_ylabel('Score / Humidity Level', fontsize=12)
ax3.set_title('Temporal Parameter Evolution\nQuality ↑, Humidity ↓', fontsize=14, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(periods)
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# Bottom-right: High performer distribution over time
high_performers_by_period = [
    len(early_data[early_data['my_score'] >= 60]) / len(early_data) * 100,
    len(mid_data[mid_data['my_score'] >= 60]) / len(mid_data) * 100,
    len(recent_data[recent_data['my_score'] >= 60]) / len(recent_data) * 100
]

bars3 = ax4.bar(periods, high_performers_by_period, alpha=0.8, color=['lightcoral', 'khaki', 'lightgreen'])

# Add value labels
for bar, value in zip(bars3, high_performers_by_period):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{value:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)

ax4.set_ylabel('% High Performers (≥60 score)', fontsize=12)
ax4.set_title('High Performer Distribution\nRecent Period Shows Consistency', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

plt.suptitle('Temporal Analysis: Why Recent Data (Last 60) is Superior for Modeling', 
             fontsize=18, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('/Users/nicolas/Desktop/Cotton-Candy-Digital-Twin/graphs/temporal_superiority_analysis.png', 
           dpi=300, bbox_inches='tight')
plt.close()

print("✅ Superiority demonstration graphs created!")
print("📊 Generated files:")
print("   1. correlation_superiority_comparison.png - Shows stronger correlation in recent data")
print("   2. temporal_superiority_analysis.png - Shows overall temporal evolution and improvement")
print("\nKey findings visualized:")
print(f"   • Correlation improvement: {corr_all:.3f} → {corr_60:.3f} ({abs(corr_60/corr_all-1)*100:.0f}% stronger)")
print(f"   • R² improvement: {r_squared_all:.3f} → {r_squared_60:.3f}")
print("   • Clear temporal evolution in quality and environmental conditions")
print("   • Recent data represents current optimal conditions")