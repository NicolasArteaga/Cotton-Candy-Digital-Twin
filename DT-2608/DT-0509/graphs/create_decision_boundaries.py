#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8')

# Create decision boundary visualization
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# 1. Start Temperature Decision Logic
humidity_range = np.linspace(40, 80, 100)
ir_temp_range = np.linspace(20, 70, 100)
H, IR = np.meshgrid(humidity_range, ir_temp_range)

# Apply our model logic for start_temp
start_temp_grid = np.where(IR >= 50, 50, 
                          np.where(IR >= 40, 48, 45))

contour1 = ax1.contourf(H, IR, start_temp_grid, levels=[44, 46, 48, 50, 52], 
                       colors=['lightblue', 'lightgreen', 'yellow', 'orange'], alpha=0.7)
ax1.contour(H, IR, start_temp_grid, levels=[44, 46, 48, 50, 52], colors='black', linewidths=1)
ax1.set_xlabel('Environmental Humidity (%)')
ax1.set_ylabel('IR Object Temperature (°C)')
ax1.set_title('Prescriptive Model: Start Temperature Decision Boundaries')
cbar1 = plt.colorbar(contour1, ax=ax1)
cbar1.set_label('Start Temperature (°C)')

# Add decision boundary lines
ax1.axhline(y=50, color='red', linestyle='--', linewidth=2, alpha=0.8)
ax1.axhline(y=40, color='red', linestyle='--', linewidth=2, alpha=0.8)
ax1.text(42, 52, '50°C', fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor="white"))
ax1.text(42, 45, '48°C', fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor="white"))
ax1.text(42, 35, '45°C', fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor="white"))

# 2. Cook Time Decision Logic
cook_time_grid = np.where(H <= 45, 75,
                         np.where(H <= 55, 78, 80))
# Add IR adjustment
cook_time_grid = np.where(IR >= 50, cook_time_grid + 2, cook_time_grid)

contour2 = ax2.contourf(H, IR, cook_time_grid, levels=[74, 76, 78, 80, 82], 
                       colors=['lightcoral', 'khaki', 'lightgreen', 'lightsalmon'], alpha=0.7)
ax2.contour(H, IR, cook_time_grid, levels=[74, 76, 78, 80, 82], colors='black', linewidths=1)
ax2.set_xlabel('Environmental Humidity (%)')
ax2.set_ylabel('IR Object Temperature (°C)')
ax2.set_title('Prescriptive Model: Cook Time Decision Boundaries')
cbar2 = plt.colorbar(contour2, ax=ax2)
cbar2.set_label('Cook Time (seconds)')

# Add humidity boundary lines
ax2.axvline(x=45, color='blue', linestyle='--', linewidth=2, alpha=0.8)
ax2.axvline(x=55, color='blue', linestyle='--', linewidth=2, alpha=0.8)
ax2.axhline(y=50, color='red', linestyle='--', linewidth=1, alpha=0.6)

# 3. Quality Prediction Zones (based on our analysis)
# Higher quality expected in lower humidity zones
quality_zones = np.where(H <= 50, 3,  # High quality zone
                        np.where(H <= 60, 2,  # Medium quality zone
                                1))  # Lower quality zone

zone_colors = ['lightcoral', 'khaki', 'lightgreen']
contour3 = ax3.contourf(H, IR, quality_zones, levels=[0.5, 1.5, 2.5, 3.5], 
                       colors=zone_colors, alpha=0.7)
ax3.set_xlabel('Environmental Humidity (%)')
ax3.set_ylabel('IR Object Temperature (°C)')
ax3.set_title('Expected Quality Zones Based on Environmental Conditions')

# Add zone labels
ax3.text(45, 35, 'HIGH QUALITY\nEXPECTED', fontweight='bold', ha='center',
         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
ax3.text(57, 35, 'MEDIUM\nQUALITY', fontweight='bold', ha='center',
         bbox=dict(boxstyle="round,pad=0.5", facecolor="khaki", alpha=0.8))
ax3.text(70, 35, 'LOWER\nQUALITY', fontweight='bold', ha='center',
         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral", alpha=0.8))

# 4. Complete Parameter Summary
# Create a parameter summary chart
conditions = ['Low Humidity\n(≤45%)\nLow IR (≤40°C)', 
             'Low Humidity\n(≤45%)\nHigh IR (≥50°C)',
             'Normal Humidity\n(45-55%)\nMedium IR', 
             'High Humidity\n(>55%)\nAny IR']
start_temps = [45, 50, 48, 48]
cook_times = [75, 77, 78, 80]
cook_temps = [53, 53, 53, 53]  # Always constant
cool_temps = [54, 54, 54, 54]  # Always constant

x_pos = np.arange(len(conditions))
width = 0.2

bars1 = ax4.bar(x_pos - 1.5*width, start_temps, width, label='Start Temp', alpha=0.8)
bars2 = ax4.bar(x_pos - 0.5*width, cook_temps, width, label='Cook Temp', alpha=0.8)
bars3 = ax4.bar(x_pos + 0.5*width, cool_temps, width, label='Cool Temp', alpha=0.8)
bars4 = ax4.bar(x_pos + 1.5*width, cook_times, width, label='Cook Time', alpha=0.8)

ax4.set_xlabel('Environmental Conditions')
ax4.set_ylabel('Temperature (°C) / Time (s)')
ax4.set_title('Complete Parameter Set by Environmental Conditions')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(conditions, rotation=45, ha='right')
ax4.legend()

# Add value labels on bars
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('/Users/nicolas/Desktop/Cotton-Candy-Digital-Twin/prescriptive_model_decision_boundaries.png', 
           dpi=300, bbox_inches='tight')
plt.show()

print("Decision boundary visualization saved as: prescriptive_model_decision_boundaries.png")
print("\nModel Summary:")
print("- Start Temperature: 45-50°C (based on IR temperature)")
print("- Cook Temperature: 53°C (constant optimal)")
print("- Cool Temperature: 54°C (constant optimal)")  
print("- Cook Time: 75-80s (humidity + IR compensated)")
print("\nOptimal conditions: Humidity ≤50%, any IR temperature")