import pandas as pd
import matplotlib.pyplot as plt

def plot_data(filepath, selection):
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    if selection == "temperature":
        columns = ['EnvT', 'InT', 'IrA2', 'IrO2']
        title = "Temperature Readings Over Time"
        ylabel = "Temperature (°C)"
    elif selection == "humidity":
        columns = ['EnvH', 'InH']
        title = "Humidity Readings Over Time"
        ylabel = "Humidity (%)"
    else:
        print("Invalid selection.")
        return

    plt.figure(figsize=(12, 6))
    for col in columns:
        if col in df:
            plt.plot(df['timestamp'], df[col], label=col)

    plt.xlabel("Timestamp")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    filepath = input("Enter the path to your CSV file: ").strip()
    selection = input("What do you want to visualize? (temperature / humidity): ").strip().lower()
    plot_data(filepath, selection)