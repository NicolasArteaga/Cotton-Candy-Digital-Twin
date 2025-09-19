#!/usr/bin/env python3
"""
Power Consumption Analysis Summary
Simple script to display power consumption results in a readable format.
"""

import pandas as pd
import json
from pathlib import Path

def display_power_summary(csv_file: str = "power/power_consumption_analysis.csv"):
    """Display a summary of power consumption analysis results"""
    
    csv_path = Path(csv_file)
    if not csv_path.exists():
        print(f"Error: {csv_file} not found!")
        print("Run the power consumption analyzer first:")
        print("python power/power_consumption_analyzer.py --process-all")
        return
    
    # Load the data
    df = pd.read_csv(csv_file)
    
    if df.empty:
        print("No power consumption data found.")
        return
    
    print("=" * 80)
    print("COTTON CANDY PRODUCTION - POWER CONSUMPTION ANALYSIS")
    print("=" * 80)
    
    # Overall statistics
    total_processes = len(df)
    valid_processes = len(df[df['total_power_measurements'] > 0])
    total_energy_wh = df['total_energy_watt_hours'].sum()
    avg_energy_per_process = df['total_energy_watt_hours'].mean()
    
    print(f"\nOVERALL STATISTICS:")
    print(f"  Total processes analyzed: {total_processes}")
    print(f"  Processes with valid power data: {valid_processes}")
    print(f"  Total energy consumed: {total_energy_wh:.2f} Wh ({total_energy_wh/1000:.3f} kWh)")
    print(f"  Average energy per process: {avg_energy_per_process:.2f} Wh")
    print(f"  Average duration per process: {df['total_duration_seconds'].mean():.1f} seconds")
    
    # Energy distribution
    print(f"\nENERGY DISTRIBUTION:")
    print(f"  Minimum energy: {df['total_energy_watt_hours'].min():.2f} Wh")
    print(f"  Maximum energy: {df['total_energy_watt_hours'].max():.2f} Wh")
    print(f"  Median energy: {df['total_energy_watt_hours'].median():.2f} Wh")
    print(f"  Standard deviation: {df['total_energy_watt_hours'].std():.2f} Wh")
    
    # Power statistics
    print(f"\nPOWER STATISTICS:")
    print(f"  Average power consumption: {df['avg_power_watts'].mean():.1f} W")
    print(f"  Maximum recorded power: {df['max_power_watts'].max():.1f} W")
    print(f"  Minimum recorded power: {df['min_power_watts'].min():.1f} W")
    
    # Batch-wise summary
    print(f"\nBATCH-WISE SUMMARY:")
    batch_summary = df.groupby('batch_number').agg({
        'total_energy_watt_hours': ['count', 'sum', 'mean'],
        'avg_power_watts': 'mean',
        'total_duration_seconds': 'mean'
    }).round(2)
    
    batch_summary.columns = ['Processes', 'Total Energy (Wh)', 'Avg Energy (Wh)', 'Avg Power (W)', 'Avg Duration (s)']
    print(batch_summary.to_string())
    
    # Top 10 highest energy consuming processes
    print(f"\nTOP 10 HIGHEST ENERGY CONSUMERS:")
    top_consumers = df.nlargest(10, 'total_energy_watt_hours')[
        ['batch_number', 'stick_number', 'total_energy_watt_hours', 'avg_power_watts', 'total_duration_seconds']
    ]
    top_consumers.columns = ['Batch', 'Stick', 'Energy (Wh)', 'Avg Power (W)', 'Duration (s)']
    print(top_consumers.to_string(index=False))
    
    # Energy efficiency analysis (Energy per minute)
    print(f"\nENERGY EFFICIENCY (Energy per minute):")
    df_efficiency = df.copy()
    df_efficiency['efficiency_wh_per_min'] = df_efficiency['total_energy_watt_hours'] / (df_efficiency['total_duration_seconds'] / 60)
    
    print(f"  Most efficient process: {df_efficiency['efficiency_wh_per_min'].min():.2f} Wh/min")
    print(f"  Least efficient process: {df_efficiency['efficiency_wh_per_min'].max():.2f} Wh/min")
    print(f"  Average efficiency: {df_efficiency['efficiency_wh_per_min'].mean():.2f} Wh/min")
    
    print("\n" + "=" * 80)
    
    # Show sample detailed data
    print("\nSAMPLE DETAILED DATA (First 3 processes):")
    sample_data = df.head(3)[['batch_number', 'stick_number', 'total_energy_watt_hours', 
                             'avg_power_watts', 'max_power_watts', 'min_power_watts', 
                             'total_power_measurements', 'total_duration_seconds']]
    
    for idx, row in sample_data.iterrows():
        print(f"\nBatch {int(row['batch_number'])}, Stick {int(row['stick_number'])}:")
        print(f"  Energy consumed: {row['total_energy_watt_hours']:.4f} Wh")
        print(f"  Power range: {row['min_power_watts']:.1f} - {row['max_power_watts']:.1f} W (avg: {row['avg_power_watts']:.1f} W)")
        print(f"  Duration: {row['total_duration_seconds']:.1f} seconds")
        print(f"  Power measurements: {int(row['total_power_measurements'])}")
    
    print("\n" + "=" * 80)
    print(f"Data saved in: {csv_file}")
    print("For detailed analysis, load the CSV file into your preferred data analysis tool.")
    print("=" * 80)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Display power consumption analysis summary')
    parser.add_argument('--csv-file', default='power/power_consumption_analysis.csv',
                       help='Path to power consumption CSV file')
    
    args = parser.parse_args()
    display_power_summary(args.csv_file)