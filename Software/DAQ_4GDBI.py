import tkinter as tk
from tkinter import ttk, messagebox

try:
    import serial.tools.list_ports
    HAS_SERIAL_TOOLS = True
except ImportError:
    HAS_SERIAL_TOOLS = False
import os
import struct
import time
import copy
import datetime
import csv
import serial
import io
import sys
from logger import logger
from multiprocessing import Process, Manager
from multiprocessing.managers import ValueProxy
from typing import Tuple, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from ctypes import c_bool
from oei import OEI
from daphne_channel import DaphneChannel
import registers as reg
from analisis_functions import analyze_file
from gain_function import analyze_gain_file
from tkcalendar import DateEntry
from datetime import datetime

# --- Global Defaults (can be overridden by GUI) ---
DAPHNE_IP="192.168.0.200"
PLOT_HISTOGRAMS = False
STORE_WAVEFORMS = True
STORE_WAVEFORMS_DIR = "/home/dunelab/massibo/data"
UPLOAD_TO_DB = False
CH_0_ENABLE = True
CH_1_ENABLE = True
CH_2_ENABLE = True
CH_0_THRESHOLD_ADC_UNITS = 8130
CH_1_THRESHOLD_ADC_UNITS = 8142
CH_2_THRESHOLD_ADC_UNITS = 8175
CH_0_BASELINE = 8146
CH_1_BASELINE = 8184
CH_2_BASELINE = 8159
DB_USER="root"
DB_PASSWORD=""
DB_HOST="127.0.0.1"
DB_NAME=""
DB_TABLE=""


MAX_TIMESTAMP = 2**32
TIME_INCREMENT_NS = 16 
OVERFLOW_LIMIT = MAX_TIMESTAMP * TIME_INCREMENT_NS 
previous_timestamp = 0
overflow_count = 0

# --- Helper Functions from DAQ_V4 ---

def calculate_extended_timestamp(current_timestamp):
   global previous_timestamp, overflow_count
   if current_timestamp < previous_timestamp:
       overflow_count += 1
   extended_timestamp = (overflow_count * OVERFLOW_LIMIT) + (current_timestamp * TIME_INCREMENT_NS)
   previous_timestamp = current_timestamp
   return extended_timestamp

def reset_timestamp_overflow_state():
   global previous_timestamp, overflow_count
   previous_timestamp = 0
   overflow_count = 0

def sanitize_filename(name):
   return "".join(c if c.isalnum() or c in (" ", "_") else "_" for c in name)

def write_to_file_binary(channel, pc_timestamp, comment=None, run_n=None, typo=None):
   sanitized_comment = sanitize_filename(comment or "default")
   filename = f"{sanitized_comment}_{channel.IDENTIFIER}_{typo}_{run_n}.bin"
   filepath = os.path.join(STORE_WAVEFORMS_DIR, filename)

   wf = channel.waveform_data
   ts = channel.timestamp_data
   cuentas = len(wf) // channel.WAVEFORM_LENGTH
   segmentos = np.array([
       wf[i:i + channel.WAVEFORM_LENGTH]
       for i in range(0, cuentas * channel.WAVEFORM_LENGTH, channel.WAVEFORM_LENGTH)
   ], dtype=np.uint16)

   ts_array = np.array(ts[:cuentas], dtype=np.uint32).reshape(-1, 1)
   data_with_timestamps = np.hstack((ts_array, segmentos)).astype(np.uint32)

   # Metadata
   VERTICAL_UNITS = 'ADU'
   HORIZONTAL_UNITS = 'ns'
   SAMPLE_INTERVAL = 16.0
   WAVEFORM_LENGTH = channel.WAVEFORM_LENGTH
   NUMBER_OF_WAVEFORMS = cuentas
   PRETRIGGER_LENGTH = 64
   TIMESTAMP_BYTES = 8

   metadata_buffer = io.BytesIO()

   def write_string(s):
       b = s.encode('utf-8')
       metadata_buffer.write(struct.pack('<H', len(b)))
       metadata_buffer.write(b)

   write_string(VERTICAL_UNITS)
   write_string(HORIZONTAL_UNITS)
   metadata_buffer.write(struct.pack('<f', SAMPLE_INTERVAL))
   metadata_buffer.write(struct.pack('<I', WAVEFORM_LENGTH))
   metadata_buffer.write(struct.pack('<I', NUMBER_OF_WAVEFORMS))
   metadata_buffer.write(struct.pack('<I', PRETRIGGER_LENGTH))
   metadata_buffer.write(struct.pack('<H', TIMESTAMP_BYTES))

   metadata = metadata_buffer.getvalue()
   metadata_length_bytes = len(metadata)

   with open(filepath, 'wb') as file:
       file.write(struct.pack('<H', metadata_length_bytes))
       file.write(metadata)
       file.write(data_with_timestamps.tobytes())

MAX_WAVEFORMS = 3000

def write_to_file_numpy(channel, pc_timestamp, comment=None, run_n=None, typo=None, output_dir=None):
   sanitized_comment = sanitize_filename(comment or "default")
   filename = f"{sanitized_comment}_{typo}_{channel.IDENTIFIER}_{run_n}_{typo}.npy"
   filepath = os.path.join(output_dir if output_dir else STORE_WAVEFORMS_DIR, filename)

   wf = channel.waveform_data
   ts = channel.timestamp_data
   cuentas = len(wf) // channel.WAVEFORM_LENGTH

   if cuentas == 0:
       return

   segmentos = np.array([
       wf[i:i + channel.WAVEFORM_LENGTH]
       for i in range(0, cuentas * channel.WAVEFORM_LENGTH, channel.WAVEFORM_LENGTH)
   ], dtype=np.uint16)



   ts_array = np.array(ts[:cuentas], dtype=np.uint64).reshape(-1, 1)

   # Concatenamos: [timestamp_hardware, timestamp_pc_ms, waveform...]
   data_with_timestamps = np.hstack((ts_array, segmentos))

   if os.path.exists(filepath):
       existing_data = np.load(filepath)
       current_waveforms = existing_data.shape[0]

       if current_waveforms >= MAX_WAVEFORMS:
           return
      
       remaining_space = MAX_WAVEFORMS - current_waveforms
       if data_with_timestamps.shape[0] > remaining_space:
           data_with_timestamps = data_with_timestamps[:remaining_space]

       data_with_timestamps = np.vstack((existing_data, data_with_timestamps))

   else:
       if data_with_timestamps.shape[0] > MAX_WAVEFORMS:
           data_with_timestamps = data_with_timestamps[:MAX_WAVEFORMS]

   np.save(filepath, data_with_timestamps)


def plot_analysis_results(results_per_channel):
    """
    Plots the histograms of DCR, XTP, and APP for each channel.
    results_per_channel: dict { 'CH0': {'DCR': [], 'XTP': [], 'APP': []}, ... }
    Plots the histograms of DCR, XTP, and APP for each channel, and GAIN if available.
    results_per_channel: dict { 'CH0': {'DCR': [], 'XTP': [], 'APP': [], 'GAIN': []}, ... }
    """
    # Check if we have DCR data (MEAS_TYPE 0) or GAIN data (MEAS_TYPE 1)
    
    # Check for DCR data
    has_dcr = any(len(res['DCR']) > 0 for res in results_per_channel.values())
    has_gain = any(len(res['GAIN']) > 0 for res in results_per_channel.values())

    if has_dcr:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("DCR, XTP, APP Analysis Summary", fontsize=16)
        
        # 1. DCR Histogram
        ax = axes[0]
        for ch_id, res in results_per_channel.items():
            data = [val * 1e3 for val in res['DCR'] if val is not None] # Convert to mHz/mm2
            if len(data) > 0:
                ax.hist(data, bins=10, alpha=0.5, label=f'{ch_id} (Mean: {np.mean(data):.2f})')
        ax.set_title("Dark Count Rate (DCR)")
        ax.set_xlabel("DCR (mHz/mm^2)")
        ax.set_ylabel("Counts")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. XTP Histogram
        ax = axes[1]
        for ch_id, res in results_per_channel.items():
            data = [val * 100 for val in res['XTP'] if val is not None] # Convert to %
            if len(data) > 0:
                ax.hist(data, bins=10, alpha=0.5, label=f'{ch_id} (Mean: {np.mean(data):.2f})')
        ax.set_title("Crosstalk Probability (XTP)")
        ax.set_xlabel("XTP (%)")
        ax.set_ylabel("Counts")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. APP Histogram
        ax = axes[2]
        for ch_id, res in results_per_channel.items():
            data = [val * 100 for val in res['APP'] if val is not None] # Convert to %
            if len(data) > 0:
                ax.hist(data, bins=10, alpha=0.5, label=f'{ch_id} (Mean: {np.mean(data):.2f})')
        ax.set_title("Afterpulsing Probability (APP)")
        ax.set_xlabel("APP (%)")
        ax.set_ylabel("Counts")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show(block=False)

        # --- NUEVO: Gráfico de XTP_Alt y APP_Alt ---
        fig_alt, axes_alt = plt.subplots(1, 2, figsize=(12, 5))
        fig_alt.suptitle("Alternative XTP & APP Analysis (Burst-Free / Strict)", fontsize=16)

        # 1. XTP_Alt Histogram
        ax = axes_alt[0]
        for ch_id, res in results_per_channel.items():
            if 'XTP_Alt' in res:
                data = [val * 100 for val in res['XTP_Alt'] if val is not None] 
                if len(data) > 0:
                    ax.hist(data, bins=10, alpha=0.5, label=f'{ch_id} (Mean: {np.mean(data):.2f})')
        ax.set_title("Alternative XTP (>1.5 p.e., No Bursts)")
        ax.set_xlabel("XTP_Alt (%)")
        ax.set_ylabel("Counts")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. APP_Alt Histogram
        ax = axes_alt[1]
        for ch_id, res in results_per_channel.items():
            if 'APP_Alt' in res:
                data = [val * 100 for val in res['APP_Alt'] if val is not None] 
                if len(data) > 0:
                    ax.hist(data, bins=10, alpha=0.5, label=f'{ch_id} (Mean: {np.mean(data):.2f})')
        ax.set_title("Alternative APP (<5us, Bursts Cleaned)")
        ax.set_xlabel("APP_Alt (%)")
        ax.set_ylabel("Counts")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show(block=False)

    if has_gain:
        # Plot Gain Histograms per channel (3 subplots)
        fig_g, axes_g = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
        fig_g.suptitle("Gain Analysis Results per Channel", fontsize=16)
        
        channels = list(results_per_channel.keys()) # ['CH0', 'CH1', 'CH2'] ideally
        
        # Ensure we handle up to 3 channels properly even if keys are different
        sorted_ch_ids = sorted(channels)
        
        for i, ch_id in enumerate(sorted_ch_ids):
            if i < 3: # Limit to 3 subplots for now
                ax = axes_g[i]
                data = results_per_channel[ch_id]['GAIN']
                if len(data) > 0:
                    mean_g = np.mean(data)
                    std_g = np.std(data)
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
        plt.show(block=False)
    
    # Keep plots open until user closes them
    if has_dcr or has_gain:
        input("Press Enter to close plots and exit...")





# --- Acquisition Function ---

def run_acquisition(config):
    """
    Executes the data acquisition process using the parameters from `config`.
    """
    STORE_HEADER_COMMENT = config['filename']
    start_run = config['start_run']
    end_run = config['end_run']
    RUN_DURATION_SECONDS = config['duration']
    MEAS_TYPE = config['meas_type'] # 0 for DCR, 1 for GAIN
    
    arduino_port = config['arduino1_port']
    arduino_port2 = config['arduino2_port']
    arduino_port3 = config['arduino3_port']
    
    # Board IDs are available in config['board_ids'] if needed in the future
    # e.g., print(f"Using Boards: {config['board_ids']}")

    if (MEAS_TYPE == 0):
       TYPE = "W"
    elif (MEAS_TYPE == 1):
       TYPE = "G"

    # Identify channels first
    channel_0 = DaphneChannel("CH0", reg.FIFO_ch0_ADDR, reg.FIFO_ch0_TS_ADDR, reg.FIFO_ch0_WR_ADDR, reg.ch0_THRESHOLD_ADDR, CH_0_THRESHOLD_ADC_UNITS, CH_0_BASELINE)
    channel_1 = DaphneChannel("CH1", reg.FIFO_ch1_ADDR, reg.FIFO_ch1_TS_ADDR, reg.FIFO_ch1_WR_ADDR, reg.ch1_THRESHOLD_ADDR, CH_1_THRESHOLD_ADC_UNITS, CH_1_BASELINE)
    channel_2 = DaphneChannel("CH2", reg.FIFO_ch2_ADDR, reg.FIFO_ch2_TS_ADDR, reg.FIFO_ch2_WR_ADDR, reg.ch2_THRESHOLD_ADDR, CH_2_THRESHOLD_ADC_UNITS, CH_2_BASELINE)       
  
    channels_obj = []
    if CH_0_ENABLE: channels_obj.append(channel_0)
    if CH_1_ENABLE: channels_obj.append(channel_1)
    if CH_2_ENABLE: channels_obj.append(channel_2)
   
    channels: Tuple[DaphneChannel] = tuple(channels_obj)

    analysis_results = {ch.IDENTIFIER: {'DCR': [], 'XTP': [], 'APP': [], 'GAIN': [], 'XTP_Alt': [], 'APP_Alt': []} for ch in channels}
    files_created_in_this_run = []

    # --- Hardware Connections ---
    try:
        arduino = serial.Serial(arduino_port, 9600, timeout=1)
        time.sleep(2)
        print(f"Conectado al Arduino 1 en {arduino_port}.")
    except Exception as e:
        print(f"Error al conectar con el Arduino 1: {e}")
        return

    try:
        arduino2 = serial.Serial(arduino_port2, 9600, timeout=1)
        time.sleep(2)
        print(f"Conectado al Arduino 2 en {arduino_port2}.")
    except Exception as e:
        print(f"Error al conectar con el Arduino 2: {e}")
        return

    try:
        arduino3 = serial.Serial(arduino_port3, 9600, timeout=1)
        time.sleep(2)
        print(f"Conectado al Arduino 3 en {arduino_port3}.")
    except Exception as e:
        print(f"Error al conectar con el Arduino 3: {e}")
        return

    start_time = time.time()
    trigger_mode_addr = reg.EXT_TRIGGER_MODE_ADDR if MEAS_TYPE == 1 else reg.SELF_TRIGGER_MODE_ADDR
    
    # --- DAPHNE Setup ---
    thing = OEI(DAPHNE_IP)
    thing.write(reg.SOFT_TRIGGER_MODE_ADDR, [1234])
    for channel in channels:
        channel.write_threshold_value(thing)
        channel.empty_fifos(thing)

    logger.debug(f"Setting trigger mode to {trigger_mode_addr}")
    thing.write(trigger_mode_addr, [1234])

    # --- Create directory structure for this specific dataset ---
    set_num = config.get('set_number', '0')
    meas_num = config.get('meas_number', '0')
    run_output_dir = os.path.join(STORE_WAVEFORMS_DIR, f"SET_{set_num}", f"MEAS_{meas_num}")
    
    if STORE_WAVEFORMS:
        try:
            os.makedirs(run_output_dir, exist_ok=True)
            print(f"Data will be saved to: {run_output_dir}")
        except OSError as e:
            print(f"Error creating directory {run_output_dir}: {e}")

    # --- Run Loop ---
    for run_counter in range(end_run, start_run - 1, -1):
        os.system("clear")
        print("======= MASSIBO/DAPHNE Acquisition System =======")
        print(f"Run: {run_counter} | Type: {TYPE}")
        print(f"Duration: {RUN_DURATION_SECONDS}s")
        print(f"Storing Waveforms: {STORE_WAVEFORMS}")
        print()
        print(f"Canal\tThresh\tRate\tNoVal\tCeros\tSumAmp\tAvgAmp")
        for ch in channels:
            print(f"{ch.IDENTIFIER}\t{ch.threshold_adc_units}\t{ch.cuentas}\t{ch.contador_no_validos}\t{ch.contador_ceros}\t{ch.suma_amplitudes}\t{ch.suma_amplitudes//ch.cuentas if ch.cuentas != 0 else 0}")
        print()   
        
        # Trigger Arduinos (Sequentially as per original script)
        arduino.write(f"{12}\n".encode())
        time.sleep(1) 
        arduino2.write(f"{14}\n".encode())
        time.sleep(1)
        arduino3.write(f"{17}\n".encode())
        time.sleep(1)

        run_start_time = time.time()

        for channel in channels:
            channel.empty_fifos(thing)

        # Reset accumulators
        for channel in channels:
            channel.contador_ceros = 0
            channel.contador_no_validos = 0
            channel.cuentas = 0
            channel.waveform_data = []
            channel.timestamp_data = []
            channel.suma_amplitudes = 0

        # Acquisition Loop
        while time.time() - run_start_time < RUN_DURATION_SECONDS:
            inicio = time.time()
            while time.time() - inicio < 1.0:
                try:
                    readable_flags = thing.readf(reg.READABLE_FLAG_ADDR, 1)[2]
                    # flags = (readable_flags & 0b1), ((readable_flags >> 1) & 0b1), ((readable_flags >> 2) & 0b1)

                    for ch in channels:
                        ch.fifo_last_write_address = thing.readf(ch.FIFO_WRITE_ADDRESS, 1)[2]
                        if ch.write_flag_firmware:
                            doutrec = thing.readf(ch.DATA_ADDRESS, DaphneChannel.WAVEFORM_LENGTH)[2:]
                            if 0 in doutrec:
                                ch.contador_ceros += 1
                            else:
                                for sample in doutrec:
                                    ch.waveform_data.append(sample & 0x3FFF)
                                doutts = thing.readf(ch.TIMESTAMP_ADDRESS, 1)[2]  & 0xFFFFFFFFFF
                                ch.timestamp_data.append(doutts)
                                ch.cuentas += 1
                except TimeoutError as te:
                    logger.warning(f"Timeout error: {te}")
                    continue

        # Save Data
        for ch in channels:
            if len(ch.waveform_data) > 0:
                write_to_file_numpy(ch, run_start_time, STORE_HEADER_COMMENT, run_counter, TYPE, output_dir=run_output_dir)
                
                # Collect filename for analysis
                sanitized_comment = sanitize_filename(STORE_HEADER_COMMENT or "default")
                filename = f"{sanitized_comment}_{TYPE}_{ch.IDENTIFIER}_{run_counter}_{TYPE}.npy"
                filepath = os.path.join(run_output_dir, filename)
                files_created_in_this_run.append((ch.IDENTIFIER, filepath))

        print(f"Run {run_counter} completado en {int(time.time() - start_time)} segundos desde inicio.")

    print("Proceso de adquisición finalizado.")

    csv_headers_common = ['Date', 'Operator', 'Brand', 'Set', 'Meas', 'Voltage', 'Version', 'Type', 'Run', 'Channel', 'Socket', 'BoardID']
    csv_headers_dcr = csv_headers_common + ['DCR_mHz_mm2', 'XTP_percent', 'APP_percent', 'XTP_Alt_percent', 'APP_Alt_percent']
    csv_headers_gain = csv_headers_common + ['Gain']

    results_dir = STORE_WAVEFORMS_DIR
    dcr_csv_path = os.path.join(results_dir, "results_DCR.csv")
    gain_csv_path = os.path.join(results_dir, "results_GAIN.csv")
    


    def get_metadata(ch_id, run_cnt):
        socket_num = run_cnt + 1
        try:
           # config['board_ids'] is { 'CH0': ['id_s1', 'id_s2', 'id_s3'], ... }
           board_id = config['board_ids'].get(ch_id, ["","",""])[run_cnt]
        except IndexError:
           board_id = "N/A"
        
        return [
            config.get('date', ''),
            config.get('operator', ''),
            config.get('brand', ''),
            config['filename'].split('_')[0].replace('set',''),
            config['filename'].split('_')[1].replace('meas',''),
            config['filename'].split('_')[2],
            config['filename'].split('_')[3],
            TYPE,
            str(run_cnt),
            ch_id,
            str(socket_num),
            board_id
        ]
        
    def append_to_csv(filepath, headers, row_data):
        file_exists = os.path.exists(filepath)
        try:
            with open(filepath, mode='a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(headers)
                writer.writerow(row_data)
        except Exception as e:
            print(f"Error writing to CSV: {e}")

    def check_and_alert_outliers(results, metric_name):
        """
        Checks for outliers in a list of values (per channel).
        If a value deviates more than 3 sigma from the mean, adds a warning.
        results: dict { 'CH0': [val1, val2...], ... }
        """
        alerts = []
        for ch_id, values in results.items():
            data = [v for v in values if v is not None]
            if len(data) < 3: continue # Need minimum stats to compute meaningful std
            
            mean = np.mean(data)
            std = np.std(data)
            
            if std == 0: continue
            
            for i, val in enumerate(data):
                z_score = abs(val - mean) / std
                if z_score > 3.0: # 3 Sigma threshold
                    alerts.append(f"OUTLIER [{metric_name}]\nChannel: {ch_id} | Run: {i}\nValue: {val:.4f} (Mean: {mean:.4f}, Std: {std:.4f})")

        return alerts

    # --- Analysis ---
    if MEAS_TYPE == 0 and len(files_created_in_this_run) > 0:
        print("\n" + "="*40)
        print("       STARTING AUTOMATED ANALYSIS")
        print("="*40)
        
        for ch_id, filepath in files_created_in_this_run:
            if os.path.exists(filepath):
                print(f"Analyzing {os.path.basename(filepath)}...")
                try:
                    dcr, xtp, app, app_alt, xtp_alt = analyze_file(filepath, plot=False)
                    if dcr is not None:
                        analysis_results[ch_id]['DCR'].append(dcr)
                        analysis_results[ch_id]['XTP'].append(xtp)
                        analysis_results[ch_id]['APP'].append(app)
                        analysis_results[ch_id]['XTP_Alt'].append(xtp_alt)
                        analysis_results[ch_id]['APP_Alt'].append(app_alt)
                        print(f" -> Results [ {ch_id} ]: DCR={dcr*1e3:.2f} mHz/mm2 | XTP={xtp*100:.2f}% | APP={app*100:.2f}%")
                        print(f"    (Alt): APP_Alt={app_alt*100:.2f}% | XTP_Alt={xtp_alt*100:.2f}%")
                        
                        # Save to CSV
                        meta = get_metadata(ch_id, run_counter)
                        data_row = meta + [dcr*1e3, xtp*100, app*100, xtp_alt*100, app_alt*100]
                        append_to_csv(dcr_csv_path, csv_headers_dcr, data_row)
                        


                    else:
                        print(f" -> Analysis failed or no valid pulses found.")
                except Exception as e:
                    print(f" -> Error analyzing file: {e}")
            else:
                print(f"File not found: {filepath}")
        


        
        print("\n" + "="*40)
        print("          PLOTTING SUMMARY")
        print("="*40)

        # Outlier Detection
        warnings = []
        warnings.extend(check_and_alert_outliers({k:v['DCR'] for k,v in analysis_results.items()}, "DCR"))
        warnings.extend(check_and_alert_outliers({k:v['XTP'] for k,v in analysis_results.items()}, "XTP"))
        warnings.extend(check_and_alert_outliers({k:v['APP'] for k,v in analysis_results.items()}, "APP"))
        
        if warnings:
            folder_warning = "\n".join(warnings)
            messagebox.showwarning("Hardware Stability Warning", f"Potential issues detected:\n\n{folder_warning}")

        plot_analysis_results(analysis_results)
        input("Press Enter to close plots and exit...")

    # --- Gain Analysis (MEAS_TYPE == 1) ---
    elif MEAS_TYPE == 1 and len(files_created_in_this_run) > 0:
        print("\n" + "="*40)
        print("       STARTING AUTOMATED GAIN ANALYSIS")
        print("="*40)
        

        for ch_id, filepath in files_created_in_this_run:
            if os.path.exists(filepath):
                print(f"Analyzing Gain for {os.path.basename(filepath)}...")
                try:
                    gain = analyze_gain_file(filepath)
                    if gain is not None:
                        analysis_results[ch_id]['GAIN'].append(gain)
                        print(f" -> Results [ {ch_id} ]: GAIN={gain:.2f} (ADU x samples)/p.e.")
                        
                        # Save to CSV
                        meta = get_metadata(ch_id, run_counter)
                        data_row = meta + [gain]
                        append_to_csv(gain_csv_path, csv_headers_gain, data_row)
                        

                    else:
                        print(f" -> Gain analysis failed.")
                except Exception as e:
                    print(f" -> Error analyzing file: {e}")
        


        
        print("\n" + "="*40)
        print("          PLOTTING GAIN SUMMARY")
        print("="*40)
        
        # Outlier Detection for Gain
        warnings = []
        warnings.extend(check_and_alert_outliers({k:v['GAIN'] for k,v in analysis_results.items()}, "GAIN"))
        
        if warnings:
            folder_warning = "\n".join(warnings)
            messagebox.showwarning("Gain Stability Warning", f"Potential Gain issues detected:\n\n{folder_warning}")
            
        plot_analysis_results(analysis_results)
        input("Press Enter to close plots and exit...")


# --- GUI Implementation ---

class DAQ_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("DAPHNE/MASSIBO DAQ Configuration")
        self.root.geometry("800x900")

        # --- Styles ---
        style = ttk.Style()
        style.configure("TLabel", font=("Helvetica", 11))
        style.configure("TButton", font=("Helvetica", 11, "bold"))
        style.configure("TEntry", font=("Helvetica", 11))
        
        
        # --- Variables ---
        # Filename construction parts
        self.set_var = tk.StringVar(value="1")
        self.meas_var = tk.StringVar(value="1")
        self.vop_var = tk.StringVar(value="48V")
        self.version_var = tk.StringVar(value="V2")

        # Start Date default to today
        self.date_var = tk.StringVar(value=datetime.now().strftime("%d/%m/%Y"))
        
        # Brand
        self.brand_var = tk.StringVar(value="HPK")
        
        # Operator
        self.operator_var = tk.StringVar(value="Carlos Benitez")
        self.other_operator_var = tk.StringVar()

        self.start_run_var = tk.IntVar(value=0)
        self.end_run_var = tk.IntVar(value=1)
        self.duration_var = tk.IntVar(value=60)
        self.meas_type_var = tk.IntVar(value=0) # 0=DCR, 1=GAIN
        
        self.arduino1_port_var = tk.StringVar()
        self.arduino2_port_var = tk.StringVar()
        self.arduino3_port_var = tk.StringVar()

        # Board ID Variables (Matrix 3x3)
        self.board_ids = {} # { 'CH0': [var1, var2, var3], ... }

        self.create_widgets()
        self.scan_ports()

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 1. Hardware Connection Section
        ttk.Label(main_frame, text="Connections (Detected Ports)", font=("Helvetica", 12, "bold")).pack(anchor=tk.W, pady=(0, 10))
        
        conn_frame = ttk.Frame(main_frame)
        conn_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(conn_frame, text="Arduino 1:").grid(row=0, column=0, padx=5, sticky=tk.W)
        self.combo_a1 = ttk.Combobox(conn_frame, textvariable=self.arduino1_port_var, width=30)
        self.combo_a1.grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(conn_frame, text="Arduino 2:").grid(row=1, column=0, padx=5, sticky=tk.W)
        self.combo_a2 = ttk.Combobox(conn_frame, textvariable=self.arduino2_port_var, width=30)
        self.combo_a2.grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Label(conn_frame, text="Arduino 3:").grid(row=2, column=0, padx=5, sticky=tk.W)
        self.combo_a3 = ttk.Combobox(conn_frame, textvariable=self.arduino3_port_var, width=30)
        self.combo_a3.grid(row=2, column=1, padx=5, pady=2)
        
        ttk.Button(conn_frame, text="Rescan Ports", command=self.scan_ports).grid(row=3, column=1, pady=5, sticky=tk.E)

        ttk.Separator(main_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=15)

        # 2. Measurement Parameters
        ttk.Label(main_frame, text="Run Parameters", font=("Helvetica", 12, "bold")).pack(anchor=tk.W, pady=(0, 10))
        
        param_frame = ttk.Frame(main_frame)
        param_frame.pack(fill=tk.X, pady=5)
        
        # --- Metadata Inputs ---
        # Date
        ttk.Label(param_frame, text="Date:").grid(row=0, column=0, padx=5, sticky=tk.W)
        self.date_entry = DateEntry(param_frame, textvariable=self.date_var, width=12, background='darkblue',
                                    foreground='white', borderwidth=2, date_pattern='dd/mm/yyyy')
        self.date_entry.grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)

        # Brand
        ttk.Label(param_frame, text="Brand:").grid(row=0, column=2, padx=5, sticky=tk.W)
        brand_combo = ttk.Combobox(param_frame, textvariable=self.brand_var, values=["HPK", "FBK"], width=8, state="readonly")
        brand_combo.grid(row=0, column=3, padx=5, pady=2, sticky=tk.W)

        # Operator
        ttk.Label(param_frame, text="Operator:").grid(row=1, column=0, padx=5, sticky=tk.W)
        self.operator_combo = ttk.Combobox(param_frame, textvariable=self.operator_var, 
                                           values=["Carlos Benitez", "Jesús Alvarez", "Andrea Roche", "Other"], 
                                           width=15, state="readonly")
        self.operator_combo.grid(row=1, column=1, padx=5, pady=2, sticky=tk.W)
        self.operator_combo.bind("<<ComboboxSelected>>", self.on_operator_change)

        # Other Operator Name
        self.other_entry = ttk.Entry(param_frame, textvariable=self.other_operator_var, width=15, state="disabled")
        self.other_entry.grid(row=1, column=2, padx=5, pady=2, sticky=tk.W)
        
        # Separator
        ttk.Label(param_frame, text="-----------------").grid(row=2, column=0, columnspan=4, pady=5)

        # --- Structured Filename Inputs ---
        # Set Number
        ttk.Label(param_frame, text="Set Number (X):").grid(row=3, column=0, padx=5, sticky=tk.W)
        ttk.Entry(param_frame, textvariable=self.set_var, width=10).grid(row=3, column=1, padx=5, pady=2, sticky=tk.W)
        
        # Meas Number
        ttk.Label(param_frame, text="Meas Number (Y):").grid(row=4, column=0, padx=5, sticky=tk.W)
        meas_combo = ttk.Combobox(param_frame, textvariable=self.meas_var, values=["1", "2", "3"], width=8, state="readonly")
        meas_combo.grid(row=4, column=1, padx=5, pady=2, sticky=tk.W)

        # Vop
        ttk.Label(param_frame, text="Voltage (Vop):").grid(row=5, column=0, padx=5, sticky=tk.W)
        ttk.Entry(param_frame, textvariable=self.vop_var, width=10).grid(row=5, column=1, padx=5, pady=2, sticky=tk.W)

        # Version
        ttk.Label(param_frame, text="System Version:").grid(row=6, column=0, padx=5, sticky=tk.W)
        version_combo = ttk.Combobox(param_frame, textvariable=self.version_var, values=["V1", "V1.5", "V2", "V2.5"], width=8, state="readonly")
        version_combo.grid(row=6, column=1, padx=5, pady=2, sticky=tk.W)

        # Separator for other params
        ttk.Label(param_frame, text="-----------------").grid(row=7, column=0, columnspan=4, pady=5)

        ttk.Label(param_frame, text="Start Run:").grid(row=8, column=0, padx=5, sticky=tk.W)
        ttk.Entry(param_frame, textvariable=self.start_run_var, width=10).grid(row=8, column=1, padx=5, pady=2, sticky=tk.W)
        
        ttk.Label(param_frame, text="End Run:").grid(row=9, column=0, padx=5, sticky=tk.W)
        ttk.Entry(param_frame, textvariable=self.end_run_var, width=10).grid(row=9, column=1, padx=5, pady=2, sticky=tk.W)
        
        ttk.Label(param_frame, text="Duration (s):").grid(row=10, column=0, padx=5, sticky=tk.W)
        ttk.Entry(param_frame, textvariable=self.duration_var, width=10).grid(row=10, column=1, padx=5, pady=2, sticky=tk.W)

        # Measurement Type Radio Buttons
        ttk.Label(param_frame, text="Meas Type:").grid(row=11, column=0, padx=5, sticky=tk.W, pady=5)
        radio_frame = ttk.Frame(param_frame)
        radio_frame.grid(row=11, column=1, columnspan=3, sticky=tk.W)
        ttk.Radiobutton(radio_frame, text="DCR (Self-Trigger)", variable=self.meas_type_var, value=0).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(radio_frame, text="GAIN (Ext-Trigger)", variable=self.meas_type_var, value=1).pack(side=tk.LEFT, padx=5)

        ttk.Separator(main_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=15)

        # 3. Board IDs (SiPM Board Identification)
        ttk.Label(main_frame, text="SiPM Board IDs (3 Boards per Channel)", font=("Helvetica", 12, "bold")).pack(anchor=tk.W, pady=(0, 10))
        
        grid_frame = ttk.Frame(main_frame)
        grid_frame.pack(fill=tk.X, pady=5)
        
        # Add Header for Socket
        ttk.Label(grid_frame, text="Socket", font=("Helvetica", 10, "bold")).grid(row=0, column=0, padx=10, pady=5)
        
        # Add Row labels for Sockets
        for row_idx in range(3):
             ttk.Label(grid_frame, text=f"Socket {row_idx+1}", font=("Helvetica", 10)).grid(row=row_idx+1, column=0, padx=5, pady=3)

        channels = ['CH0', 'CH1', 'CH2']
        for col_idx, ch in enumerate(channels):
            # Shift columns by +1 because column 0 is now "Socket"
            grid_col = col_idx + 1
            
            lbl = ttk.Label(grid_frame, text=ch, font=("Helvetica", 10, "bold"))
            lbl.grid(row=0, column=grid_col, padx=10, pady=5)
            self.board_ids[ch] = []
            for row_idx in range(3):
                var = tk.StringVar()
                entry = ttk.Entry(grid_frame, textvariable=var, width=15) # , justify='center'
                entry.grid(row=row_idx+1, column=grid_col, padx=10, pady=3)
                self.board_ids[ch].append(var)

        ttk.Separator(main_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=20)

        # 4. Start Actions
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X)
        
        ttk.Button(btn_frame, text="START ACQUISITION", command=self.on_start, width=20).pack(side=tk.RIGHT, padx=5)
        ttk.Button(btn_frame, text="EXIT", command=self.root.quit, width=10).pack(side=tk.RIGHT, padx=5)

    def on_operator_change(self, event):
        """Enable/Disable Other Operator Entry"""
        if self.operator_var.get() == "Other":
            self.other_entry.config(state="normal")
            self.other_entry.focus()
        else:
            self.other_entry.config(state="disabled")
            self.other_operator_var.set("") # Clear input

    def scan_ports(self):
        """Scans for available serial ports and populates the comboboxes."""
        if HAS_SERIAL_TOOLS:
            try:
                ports = serial.tools.list_ports.comports()
                port_list = [port.device for port in ports]
            except Exception:
                port_list = []
        else:
            # Fallback list for Linux
            port_list = [f"/dev/ttyACM{i}" for i in range(10)] + [f"/dev/ttyUSB{i}" for i in range(10)]
        
        # Try to intelligently pre-select if known
        default_ports = ['/dev/ttyACM3', '/dev/ttyACM1', '/dev/ttyACM2']
        
        self.combo_a1['values'] = port_list
        self.combo_a2['values'] = port_list
        self.combo_a3['values'] = port_list
        
        if len(port_list) > 0:
            # Set defaults if they exist in the list (or just set them anyway as suggestions)
            self.arduino1_port_var.set(default_ports[0] if default_ports[0] in port_list else (port_list[0] if port_list else ""))
            if len(port_list) > 1:
                self.arduino2_port_var.set(default_ports[1] if default_ports[1] in port_list else (port_list[1] if len(port_list)>1 else ""))
            if len(port_list) > 2:
                self.arduino3_port_var.set(default_ports[2] if default_ports[2] in port_list else (port_list[2] if len(port_list)>2 else ""))
            
            # If fallback mode, force defaults if they weren't set above? 
            # Actually, standard defaults are better than empty
            if not self.arduino1_port_var.get(): self.arduino1_port_var.set(default_ports[0])
            if not self.arduino2_port_var.get(): self.arduino2_port_var.set(default_ports[1])
            if not self.arduino3_port_var.get(): self.arduino3_port_var.set(default_ports[2])

        else:
            self.arduino1_port_var.set(default_ports[0])
            self.arduino2_port_var.set(default_ports[1])
            self.arduino3_port_var.set(default_ports[2])


    def on_start(self):
        """Validates inputs and starts the acquisition logic."""
        
        # Validation
        try:
            # Construct formatted filename
            # setX_measY_Vop_Version
            set_val = self.set_var.get().strip()
            meas_val = self.meas_var.get().strip()
            vop_val = self.vop_var.get().strip()
            version_val = self.version_var.get().strip()
            
            if not all([set_val, meas_val, vop_val, version_val]):
                messagebox.showerror("Error", "Please fill in all file parameters (Set, Meas, Voltage, Version).")
                return
                
            final_filename = f"set{set_val}_meas{meas_val}_{vop_val}_{version_val}"

            # Determine final operator name
            operator_name = self.operator_var.get()
            if operator_name == "Other":
                operator_name = self.other_operator_var.get().strip()
                if not operator_name:
                    messagebox.showerror("Error", "Please enter the Other Operator's name.")
                    return

            config = {
                'filename': final_filename,
                'date': self.date_entry.get_date().strftime("%d-%m-%Y"),
                'brand': self.brand_var.get(),
                'operator': operator_name,
                'set_number': set_val,
                'meas_number': meas_val,
                'start_run': self.start_run_var.get(),
                'end_run': self.end_run_var.get(),
                'duration': self.duration_var.get(),
                'meas_type': self.meas_type_var.get(),
                'arduino1_port': self.arduino1_port_var.get(),
                'arduino2_port': self.arduino2_port_var.get(),
                'arduino3_port': self.arduino3_port_var.get(),
                'board_ids': {
                    ch: [v.get().strip() for v in vars] 
                    for ch, vars in self.board_ids.items()
                }
            }
            
            if not all([config['arduino1_port'], config['arduino2_port'], config['arduino3_port']]):
                 messagebox.showwarning("Warning", "Please select ports for all 3 Arduinos.")
                 return

            # If validation passes, close window and run
            self.root.destroy()
            
            # RUN LOGIC
            run_acquisition(config)

        except Exception as e:
            messagebox.showerror("Configuration Error", f"Invalid Input: {e}")

def main():
    root = tk.Tk()
    app = DAQ_GUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
