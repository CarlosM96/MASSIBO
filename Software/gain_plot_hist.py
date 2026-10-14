import tkinter as tk
from tkinter import filedialog, messagebox
import csv
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_histograms(results_per_channel, filename):
    """
    Plots the Gain Histograms per channel (3 subplots).
    results_per_channel: dict { 'CH0': [g1, g2...], 'CH1': [...], ... }
    """
    # Check if we have any gain data
    has_gain = any(len(vals) > 0 for vals in results_per_channel.values())
    
    if has_gain:
        # Plot Gain Histograms per channel (3 subplots)
        fig_g, axes_g = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
        fig_g.suptitle(f"Gain Analysis Results per Channel (Source: {os.path.basename(filename)})", fontsize=16)
        
        channels = ['CH0', 'CH1', 'CH2']
        
        for i, ch_id in enumerate(channels):
            if i < 3: # Limit to 3 subplots for now
                ax = axes_g[i]
                data = results_per_channel.get(ch_id, [])
                if len(data) > 0:
                    mean_g = np.mean(data)
                    std_g = np.std(data)
                    ax.hist(data, bins=50, color=f'C{i}', alpha=0.7, edgecolor='black')
                    ax.axvline(mean_g, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_g:.2f}')
                    ax.set_title(f"Gain Distribution - {ch_id}")
                    ax.set_xlabel("Gain (ADU x samples)")
                    ax.set_ylabel("Counts")
                    ax.legend()
                    
                    # Also print basic stats
                    print(f"[{ch_id}] Count: {len(data)}, Mean: {mean_g:.2f}, Std: {std_g:.2f}")
                else:
                    ax.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f"Gain Distribution - {ch_id}")
                
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    else:
        print("No valid gain data found to plot.")

def main():
    root = tk.Tk()
    root.withdraw() # Hide main window

    print("Please select the CSV file containing the results (e.g., offline_results_GAIN.csv).")
    filepath = filedialog.askopenfilename(
        title="Select Results CSV",
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
    )

    if not filepath:
        print("No file selected. Exiting.")
        return

    print(f"Reading file: {filepath}")

    results = {'CH0': [], 'CH1': [], 'CH2': []}

    try:
        with open(filepath, 'r', newline='') as f:
            reader = csv.DictReader(f)
            
            # Verify columns
            if 'Channel' not in reader.fieldnames or 'Gain' not in reader.fieldnames:
                print("Error: CSV missing required columns (Channel, Gain)")
                return
                
            print(f"CSV Headers found: {reader.fieldnames}")
            
            # Robust column gathering
            col_corrected = next((h for h in reader.fieldnames if h.strip() == 'Corrected'), None)
            has_corrected = col_corrected is not None
            if has_corrected:
                print(f"Found 'Corrected' column as: '{col_corrected}'")
            else:
                print("WARNING: 'Corrected' column NOT found.")
            
            row_count = 0
            for row in reader:
                ch = row['Channel']
                gain_str = row['Gain']
                corrected_str = row[col_corrected] if has_corrected else ""
                
                # Logic: Use Corrected if available, else Gain
                final_val = None
                
                if corrected_str and corrected_str.strip():
                    try:
                        final_val = float(corrected_str)
                    except ValueError:
                        print(f"Warning: Invalid Corrected value '{corrected_str}' in row {row_count+2}")
                        continue
                elif gain_str and gain_str.strip():
                    try:
                        final_val = float(gain_str)
                    except ValueError:
                        print(f"Warning: Invalid Gain value '{gain_str}' in row {row_count+2}")
                        continue
                
                if final_val is not None:
                    # Debug print for first few rows or when corrected is used
                    if (corrected_str and corrected_str.strip()) or row_count < 5:
                         source = "CORRECTED" if (corrected_str and corrected_str.strip()) else "GAIN"
                         print(f"Row {row_count+2}: Channel={ch}, Corrected='{corrected_str}', Gain='{gain_str}' -> Used {final_val} ({source})")

                    # Filter outlier/invalid data (optional, but consistent with analysis)
                    # if final_val > 0: 
                    if ch in results:
                        results[ch].append(final_val)
                
                row_count += 1
                
        print(f"Processed {row_count} rows.")
        plot_histograms(results, filepath)
        
    except Exception as e:
        print(f"Error reading CSV: {e}")
        messagebox.showerror("Error", f"Failed to read CSV:\n{e}")

if __name__ == "__main__":
    main()
