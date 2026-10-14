import os
import csv
import tkinter as tk
from tkinter import filedialog, messagebox
import re
import matplotlib.pyplot as plt
import numpy as np
from analisis_functions import analyze_file

def sanitize_filename(name):
    return "".join(c if c.isalnum() or c in (" ", "_") else "_" for c in name)

def parse_filename(filename):
    """
    Parses the filename to extract metadata.
    Expected format: set{X}_meas{Y}_{V}_{Ver}_{Type}_{CH}_{Run}_{Type}.npy
    Example: set6_meas1_48V_V2_G_CH0_3_G.npy (or DCR)
    """
    try:
        # Remove extension
        name = os.path.splitext(filename)[0]
        parts = name.split('_')
        
        # Robust parsing:
        # Expected structure tail: ..._Type_Channel_Run_Type.npy
        
        # Check if we have enough parts
        if len(parts) < 4:
            raise ValueError("Filename too short")

        # Parsing from the right side for consistent parts
        type_suffix = parts[-1]        # DCR, G, or W
        run_str = parts[-2]            # Run Number
        channel_str = parts[-3]        # CH1
        type_prefix = parts[-4]        # DCR or G
        
        # Determine Set and Meas from the left side
        set_str = parts[0]             # set62
        meas_str = parts[1]            # meas1
        
        # Everything in between is Voltage/Version/etc info
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

def plot_metric_results(results_per_channel, metric_name, unit_label):
    """
    Plots Histograms for a given metric per channel (3 subplots).
    """
    # Check if we have any data
    has_data = any(len(vals) > 0 for vals in results_per_channel.values())
    
    if has_data:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
        fig.suptitle(f"{metric_name} Analysis Results per Channel (Offline)", fontsize=16)
        
        channels = ['CH0', 'CH1', 'CH2']
        
        for i, ch_id in enumerate(channels):
            if i < 3: # Limit to 3 subplots for now
                ax = axes[i]
                data = results_per_channel.get(ch_id, [])
                if len(data) > 0:
                    data_plot = data 
                    
                    mean_val = np.mean(data_plot)
                    # std_val = np.std(data_plot)
                    
                    ax.hist(data_plot, bins=30, color=f'C{i}', alpha=0.7, edgecolor='black')
                    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
                    ax.set_title(f"{metric_name} - {ch_id}")
                    ax.set_xlabel(f"{metric_name} ({unit_label})")
                    ax.set_ylabel("Counts")
                    ax.legend()
                else:
                    ax.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f"{metric_name} - {ch_id}")
                
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def plot_dcr_results(results_per_channel):
    plot_metric_results(results_per_channel, "DCR (No Burst)", "mHz/mm²")

def main():
    root = tk.Tk()
    root.withdraw() # Hide main window

    print("Please select the directory containing the .npy files (DCR analysis) using the dialog window.")
    selected_dir = filedialog.askdirectory(title="Select Data Directory")

    if not selected_dir:
        print("No directory selected. Exiting.")
        return

    print(f"Scanning directory: {selected_dir}")

    # Prepare specific CSV file
    dir_name = os.path.basename(selected_dir) 
    set_match = re.search(r'SET_(\d+)', dir_name, re.IGNORECASE)
    if set_match:
        set_num = set_match.group(1)
        csv_filename = f"offline_results_DCR_SET_{set_num}.csv"
    else:
        csv_filename = "offline_results_DCR.csv"
        
    csv_path = os.path.join(selected_dir, csv_filename)
    
    # Headers
    # Base Headers: 'Date', 'Operator', 'Brand', 'Set', 'Meas', 'Voltage', 'Version', 'Type', 'Run', 'Channel', 'Socket', 'BoardID', 'Duration_s', 'Wait_35min'
    # Extra Headers: 'DCR_NoBurst_mHz_mm2', 'DCR_Raw_mHz_mm2', 'XTP_percent', 'APP_percent', 'XTP_Alt_percent', 'APP_Alt_percent'
    
    base_headers = ['Date', 'Operator', 'Brand', 'Set', 'Meas', 'Voltage', 'Version', 'Type', 'Run', 'Channel', 'Socket', 'BoardID', 'Duration_s', 'Wait_35min']
    analysis_headers = ['DCR_NoBurst_mHz_mm2', 'DCR_Raw_mHz_mm2', 'XTP_percent', 'APP_percent', 'XTP_Alt_percent', 'APP_Alt_percent']
    
    headers = base_headers + analysis_headers

    # Initialize CSV if it doesn't exist
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
        print(f"Created new CSV file: {csv_path}")
    else:
        print(f"Appending to existing CSV file: {csv_path}")

    # Look for files with '_DCR_' or '_W_' 
    # 'W' is the identifier used in DAQ_5GDBI.py for DCR/Waveform mode
    files = [f for f in os.listdir(selected_dir) if f.endswith('.npy') and ('DCR' in f or 'dcr' in f or '_W' in f or '_W_' in f)]
    
    # Sort files
    def get_run_sort_key(f):
        meta = parse_filename(f)
        return meta['Run'] if meta else -1
        
    files.sort(key=get_run_sort_key)

    print(f"Found {len(files)} DCR data files.")

    processed_count = 0
    
    # Store results for plotting
    dcr_results = {'CH0': [], 'CH1': [], 'CH2': []}
    # Store results for plotting
    dcr_results = {'CH0': [], 'CH1': [], 'CH2': []}
    xtp_results = {'CH0': [], 'CH1': [], 'CH2': []}
    app_alt_results = {'CH0': [], 'CH1': [], 'CH2': []}
    app_alt_results = {'CH0': [], 'CH1': [], 'CH2': []}

    for filename in files:
        filepath = os.path.join(selected_dir, filename)
        
        # Parse Metadata
        meta = parse_filename(filename)
        if not meta:
            print(f"Skipping {filename} (could not parse filename)")
            continue
            
        print(f"Analyzing {filename} ...")
        
        # Analyze DCR
        # analyze_file returns: dcr, dcr_raw, crosstalk_prob, ap_prob, APP_alternative, XTP_alternative
        try:
            dcr, dcr_raw, xtp, app, app_alt, xtp_alt = analyze_file(filepath, plot=False)
        except Exception as e:
            print(f"  Error analyzing file: {e}")
            dcr = None
            
        if dcr is not None:
            # Convert units as in DAQ_5GDBI.py logic
            # analyze_file returns DCR in Hz/mm2 (if area was mm2)
            # We want mHz/mm2 => * 1e3
            dcr_mHz = dcr * 1e3
            dcr_raw_mHz = dcr_raw * 1e3
            xtp_pct = xtp * 100
            app_pct = app * 100
            xtp_alt_pct = xtp_alt * 100
            app_alt_pct = app_alt * 100
            
            print(f"  -> DCR (NoBurst) = {dcr_mHz:.2f} mHz/mm2")
            
            # Store for plotting
            if meta['Channel'] in dcr_results:
                dcr_results[meta['Channel']].append(dcr_mHz)
                xtp_results[meta['Channel']].append(xtp_pct)
                app_alt_results[meta['Channel']].append(app_alt_pct)
            
            # Construct CSV Row
            date_val = "Offline"
            operator_val = "Offline"
            brand_val = "Offline"
            board_id_val = "N/A"
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
                dcr_mHz,
                dcr_raw_mHz,
                xtp_pct,
                app_pct,
                xtp_alt_pct,
                app_alt_pct
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
            print("  -> Analysis failed or returned None")

    print("\n" + "="*40)
    print(f"Analysis Complete.")
    print(f"Processed {processed_count}/{len(files)} files successfully.")
    print(f"Results saved to: {csv_path}")
    print("="*40)
    
    messagebox.showinfo("Offline Analysis", f"Analysis Complete.\nSaved to {csv_filename}\n\nClick OK to see DCR, XTP, and APP plots.")
    
    # Plot results
    plot_metric_results(dcr_results, "DCR (No Burst)", "mHz/mm²")
    plot_metric_results(xtp_results, "XTP (Standard)", "%")
    plot_metric_results(app_alt_results, "APP Alternative", "%")

if __name__ == "__main__":
    main()
