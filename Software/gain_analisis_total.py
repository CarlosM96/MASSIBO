import os
import csv
import tkinter as tk
from tkinter import filedialog, messagebox
import re
import matplotlib.pyplot as plt
import numpy as np
from gain_function import analyze_gain_file

def sanitize_filename(name):
    return "".join(c if c.isalnum() or c in (" ", "_") else "_" for c in name)

def parse_filename(filename):
    """
    Parses the filename to extract metadata.
    Expected format: set{X}_meas{Y}_{V}_{Ver}_{Type}_{CH}_{Run}_{Type}.npy
    Example: set6_meas1_48V_V2_G_CH0_3_G.npy
    """
    try:
        # Remove extension
        name = os.path.splitext(filename)[0]
        parts = name.split('_')
        
        # Robust parsing:
        # Expected structure tail: ..._Type_Channel_Run_Type.npy
        # Example: set62_meas1_44_5V_V2_5_G_CH1_10_G.npy
        
        # Check if we have enough parts
        if len(parts) < 4:
            raise ValueError("Filename too short")

        # Parsing from the right side for consistent parts
        type_suffix = parts[-1]        # G
        run_str = parts[-2]            # 10
        channel_str = parts[-3]        # CH1
        type_prefix = parts[-4]        # G
        
        # Determine Set and Meas from the left side
        set_str = parts[0]             # set62
        meas_str = parts[1]            # meas1
        
        # Everything in between is Voltage/Version/etc info
        # e.g. ['44', '5V', 'V2', '5'] -> "44_5V_V2_5"
        intermediate_parts = parts[2:-4]
        voltage_version_info = "_".join(intermediate_parts) if intermediate_parts else "Unknown"
        
        # Extract numeric/string values
        set_num = set_str.replace('set', '')
        meas_num = meas_str.replace('meas', '')
        
        return {
            'Set': set_num,
            'Meas': meas_num,
            'Voltage': voltage_version_info, # Using this field for combined info
            'Version': "",                   # Empty as we combined it
            'Type': type_suffix,
            'Channel': channel_str,
            'Run': int(run_str)
        }
    except Exception as e:
        print(f"Error parsing filename {filename}: {e}")
        return None

def get_socket_from_run(run_counter):
    # Socket Logic: 
    # Run 0-5 -> Socket 1
    # Run 6-11 -> Socket 2
    # Run 12-17 -> Socket 3
    socket_idx = int(run_counter // 6)
    socket_num = socket_idx + 1
    return socket_num

def plot_gain_results(results_per_channel):
    """
    Plots the Gain Histograms per channel (3 subplots).
    results_per_channel: dict { 'CH0': [g1, g2...], 'CH1': [...], ... }
    """
    # Check if we have any gain data
    has_gain = any(len(vals) > 0 for vals in results_per_channel.values())
    
    if has_gain:
        # Plot Gain Histograms per channel (3 subplots)
        fig_g, axes_g = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
        fig_g.suptitle("Gain Analysis Results per Channel (Offline)", fontsize=16)
        
        channels = ['CH0', 'CH1', 'CH2']
        
        for i, ch_id in enumerate(channels):
            if i < 3: # Limit to 3 subplots for now
                ax = axes_g[i]
                data = results_per_channel.get(ch_id, [])
                if len(data) > 0:
                    mean_g = np.mean(data)
                    # std_g = np.std(data)
                    ax.hist(data, bins='auto', color=f'C{i}', alpha=0.7, edgecolor='black')
                    ax.axvline(mean_g, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_g:.2f}')
                    ax.set_title(f"Gain Distribution - {ch_id}")
                    ax.set_xlabel("Gain (ADU x samples)")
                    ax.set_ylabel("Counts")
                    ax.legend()
                else:
                    ax.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f"Gain Distribution - {ch_id}")
                
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def main():
    root = tk.Tk()
    root.withdraw() # Hide main window

    print("Please select the directory containing the .npy files using the dialog window.")
    selected_dir = filedialog.askdirectory(title="Select Data Directory")

    if not selected_dir:
        print("No directory selected. Exiting.")
        return

    print(f"Scanning directory: {selected_dir}")

    # Prepare specific CSV file
    # We try to extract set number from directory name or default to 'offline'
    dir_name = os.path.basename(selected_dir) 
    # Example dir_name: SET_6. Try to extract '6'
    set_match = re.search(r'SET_(\d+)', dir_name, re.IGNORECASE)
    if set_match:
        set_num = set_match.group(1)
        csv_filename = f"offline_results_GAIN_SET_{set_num}.csv"
    else:
        csv_filename = "offline_results_GAIN.csv"
        
    csv_path = os.path.join(selected_dir, csv_filename)
    
    # Headers matching DAQ_5GDBI.py + Corrected
    # ['Date', 'Operator', 'Brand', 'Set', 'Meas', 'Voltage', 'Version', 'Type', 'Run', 'Channel', 'Socket', 'BoardID', 'Duration_s', 'Wait_35min', 'Gain', 'Corrected']
    headers = ['Date', 'Operator', 'Brand', 'Set', 'Meas', 'Voltage', 'Version', 'Type', 'Run', 'Channel', 'Socket', 'BoardID', 'Duration_s', 'Wait_35min', 'Gain', 'Corrected']

    # Initialize CSV if it doesn't exist
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
        print(f"Created new CSV file: {csv_path}")
    else:
        print(f"Appending to existing CSV file: {csv_path}")

    files = [f for f in os.listdir(selected_dir) if f.endswith('.npy') and '_G_' in f]
    
    # Sort files to process in order (optional but nice)
    # Try to sort by Run number if possible
    def get_run_sort_key(f):
        meta = parse_filename(f)
        return meta['Run'] if meta else -1
        
    files.sort(key=get_run_sort_key)

    print(f"Found {len(files)} GAIN data files.")

    processed_count = 0
    
    # Store results for plotting
    gain_results = {'CH0': [], 'CH1': [], 'CH2': []}

    for filename in files:
        filepath = os.path.join(selected_dir, filename)
        
        # Parse Metadata
        meta = parse_filename(filename)
        if not meta:
            print(f"Skipping {filename} (could not parse filename)")
            continue
            
        print(f"Analyzing {filename} ...")
        
        # Analyze Gain
        try:
            gain = analyze_gain_file(filepath)
        except Exception as e:
            print(f"  Error analyzing file: {e}")
            gain = None
            
        if gain is not None:
            print(f"  -> GAIN = {gain:.2f}")
            
            # Store for plotting
            if meta['Channel'] in gain_results:
                gain_results[meta['Channel']].append(gain)
            
            # Construct CSV Row
            # Metadata placeholders for offline analysis
            date_val = "Offline"
            operator_val = "Offline"
            brand_val = "Offline"
            board_id_val = "N/A" # Cannot know board ID without the config map
            duration_val = "N/A"
            wait_val = "N/A"
            
            socket_num = get_socket_from_run(meta['Run'])
            
            row = [
                date_val,
                operator_val,
                brand_val,
                meta['Set'],
                meta['Meas'],
                meta['Voltage'],
                meta['Version'],
                meta['Type'],
                str(meta['Run']),
                meta['Channel'],
                str(socket_num),
                board_id_val,
                duration_val,
                wait_val,
                gain,
                "" # Corrected placeholder
            ]
            
            # Append to CSV
            try:
                with open(csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
                processed_count += 1
            except Exception as e:
                print(f"  Error writing to CSV: {e}")
        else:
            print("  -> Analysis failed (not enough peaks or bad data)")

    print("\n" + "="*40)
    print(f"Analysis Complete.")
    print(f"Processed {processed_count}/{len(files)} files successfully.")
    print(f"Results saved to: {csv_path}")
    print("="*40)
    
    messagebox.showinfo("Offline Analysis", f"Analysis Complete.\nSaved to {csv_filename}\n\nClick OK to see plots.")
    
    # Plot results
    plot_gain_results(gain_results)

if __name__ == "__main__":
    main()
