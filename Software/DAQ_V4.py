import os
import struct
import time
import copy
import datetime
import csv
import serial
import io
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

#
DAPHNE_IP="192.168.0.200"




PLOT_HISTOGRAMS =False
STORE_WAVEFORMS=True
STORE_HEADER_COMMENT=input("FILE NAME: ")
STORE_WAVEFORMS_DIR="/home/dunelab/massibo/data"
UPLOAD_TO_DB=False
UPLOAD_PERIOD_SECONDS=10
#UPLOAD_BUFFER_PATH="/home/cbenitez/Documents"
#UPLOAD_BUFFER_HEADER="timestamp,top,mid,bot,top_amp,mid_amp,bot_amp\n"
LOG_TO_FILE=False
LOG_FILE_PATH=".log"
TYPE = "default"


  
CH_0_ENABLE=True   ##Channel 0/9 Massibo
CH_1_ENABLE=True   ##Channel 1/10 Massibo
CH_2_ENABLE=True   ##Channel 2/11 Massibo


CH_0_THRESHOLD_ADC_UNITS=8130  #8138-7 8126 hpk   30 fbk
CH_1_THRESHOLD_ADC_UNITS= 8142
CH_2_THRESHOLD_ADC_UNITS= 8175 #8175 hpk  #8184-6 73  77 fbk


CH_0_BASELINE=8146
CH_1_BASELINE=8184 ##Channel 0/9 Massibo   Verificar con electrónica fria
CH_2_BASELINE=8159

DB_USER="root"
DB_PASSWORD=""
DB_HOST="127.0.0.1"
DB_NAME=""
DB_TABLE=""


MAX_TIMESTAMP = 2**32
TIME_INCREMENT_NS = 24 
OVERFLOW_LIMIT = MAX_TIMESTAMP * TIME_INCREMENT_NS 
previous_timestamp = 0
overflow_count = 0


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




def write_to_file_binary(channel, pc_timestamp, comment=None, run_n=None, typo=None):
   sanitized_comment = sanitize_filename(comment or "default")
  
   """if channel.IDENTIFIER == "CH0":
       filename = f"{sanitized_comment}_{channel.IDENTIFIER}_{typo}_{run_n}.bin"
   elif channel.IDENTIFIER == "CH2":
       filename = f"{sanitized_comment}_{channel.IDENTIFIER}_{typo}_{run_n + 9}.bin"""

   filename = f"{sanitized_comment}_{channel.IDENTIFIER}_{typo}_{run_n}.bin"
   filepath = os.path.join(STORE_WAVEFORMS_DIR, filename)


   # Datos
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
   SAMPLE_INTERVAL = 1000 / 41.66
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


"""
def write_to_file_numpy(channel, pc_timestamp, comment=None, run_n=None, typo=None):
   sanitized_comment = sanitize_filename(comment or "default")
   #filename = f"{sanitized_comment}_{typo}_channel_{run_n}_{channel.IDENTIFIER}.npy"
   if (channel.IDENTIFIER == "CH0"):
       filename = f"{sanitized_comment}_{typo}_{channel.IDENTIFIER}_{run_n}_{typo}.npy"
   elif (channel.IDENTIFIER == "CH2"):
       filename = f"{sanitized_comment}_{typo}_{channel.IDENTIFIER}_{run_n + 9}_{typo}.npy"
   filepath = os.path.join(STORE_WAVEFORMS_DIR, filename)




   wf = channel.waveform_data
   ts = channel.timestamp_data
   cuentas = len(wf) // channel.WAVEFORM_LENGTH
   segmentos = np.array([
       wf[i:i + channel.WAVEFORM_LENGTH]
       for i in range(0, cuentas * channel.WAVEFORM_LENGTH, channel.WAVEFORM_LENGTH)
   ], dtype=np.uint16)


   ts_array = np.array(ts[:cuentas], dtype=np.uint32).reshape(-1, 1)
   data_with_timestamps = np.hstack((ts_array, segmentos))


   if os.path.exists(filepath):
       existing_data = np.load(filepath)
       data_with_timestamps = np.vstack((existing_data, data_with_timestamps))


   np.save(filepath,data_with_timestamps)"""
MAX_WAVEFORMS = 3000


def write_to_file_numpy(channel, pc_timestamp, comment=None, run_n=None, typo=None):
   sanitized_comment = sanitize_filename(comment or "default")


   """if channel.IDENTIFIER == "CH0":
       filename = f"{sanitized_comment}_{typo}_{channel.IDENTIFIER}_{run_n}_{typo}.npy"
   elif channel.IDENTIFIER == "CH2":
       filename = f"{sanitized_comment}_{typo}_{channel.IDENTIFIER}_{run_n + 9}_{typo}.npy"
   else:
       return"""

   filename = f"{sanitized_comment}_{typo}_{channel.IDENTIFIER}_{run_n}_{typo}.npy" 
   filepath = os.path.join(STORE_WAVEFORMS_DIR, filename)


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


"""
def write_to_file_numpy(channel, pc_timestamp, comment=None, run_n=None, typo=None):
   sanitized_comment = sanitize_filename(comment or "default")


   if channel.IDENTIFIER == "CH0":
       filename = f"{sanitized_comment}_{typo}_{channel.IDENTIFIER}_{run_n}_{typo}.npy"
   elif channel.IDENTIFIER == "CH2":
       filename = f"{sanitized_comment}_{typo}_{channel.IDENTIFIER}_{run_n + 9}_{typo}.npy"
   else:
       return


   filepath = os.path.join(STORE_WAVEFORMS_DIR, filename)


   wf = channel.waveform_data
   ts = channel.timestamp_data
   cuentas = len(wf) // channel.WAVEFORM_LENGTH


   if cuentas == 0:
       return


   segmentos = np.array([
       wf[i:i + channel.WAVEFORM_LENGTH]
       for i in range(0, cuentas * channel.WAVEFORM_LENGTH, channel.WAVEFORM_LENGTH)
   ], dtype=np.uint16)


   # Agregamos timestamp del sistema (en milisegundos)
   sys_ts_ms = np.array([
       time.time() * 1000  # segundos → milisegundos como float
       for _ in range(cuentas)
   ], dtype=np.float64).reshape(-1, 1)


   ts_array = np.array(ts[:cuentas], dtype=np.uint64).reshape(-1, 1)


   # Concatenamos: [timestamp_hardware, timestamp_pc_ms, waveform...]
   data_with_timestamps = np.hstack((ts_array, sys_ts_ms, segmentos))


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
"""












def sanitize_filename(name):
   return "".join(c if c.isalnum() or c in (" ", "_") else "_" for c in name)


def write_to_file_csv(channel, pc_timestamp, comment=None, run_n=None, typo=None):


   sanitized_comment = "".join(c if c.isalnum() or c in (" ", "_") else "_" for c in (comment or "default"))
   filename = f"{sanitized_comment}_{typo}_channel_{run_n}_{channel.IDENTIFIER}.csv"




   filepath = os.path.join(STORE_WAVEFORMS_DIR, filename)




   wf = channel.waveform_data
   ts = channel.timestamp_data
   pc_datetime = datetime.datetime.fromtimestamp(pc_timestamp)


   cuentas = len(wf) // channel.WAVEFORM_LENGTH
   segmentos = [wf[i:i + channel.WAVEFORM_LENGTH] for i in range(0, cuentas * channel.WAVEFORM_LENGTH, channel.WAVEFORM_LENGTH)]


   file_exists = os.path.exists(filepath)


   with open(filepath, mode='a', newline='') as csvfile:
       writer = csv.writer(csvfile)


       if not file_exists:
           writer.writerow([
               "Source", "AcquisitionDatetime", "TriggerLevelADU",
               "BaselineLevelADU", "Timestamp", "WaveformData", "Comment"
           ])
      


       for i in range(len(segmentos)):
           ts_c = calculate_extended_timestamp(ts[i])
           row = [
               channel.IDENTIFIER,
               pc_datetime,
               channel.threshold_adc_units,
               channel.baseline_adc_units,
               ts_c,
               ','.join(map(str, segmentos[i])),  # Convertir segmento a cadena separada por comas
               comment or ""
           ]
           writer.writerow(row)




def plot_data(channels_last, update_plot: ValueProxy):
       x_plot = np.array(range(DaphneChannel.WAVEFORM_LENGTH))
       # global channels_last
       with open("plot.log", "at") as f:
           # f.write(f"{x_plot}\n")
           try:
               plt.ion()
               figure, (ax1, ax2) = plt.subplots(1, 2)
               figure.tight_layout()
               # ax.plot(x_plot, np.zeros(DaphneChannel.WAVEFORM_LENGTH))
               figure.canvas.draw()
               figure.canvas.flush_events()
               ax1.set_xlim(CH_0_THRESHOLD_ADC_UNITS-100,2**14)
               ax1.set_ylim(0, 0.1)
               # ax1.set_yscale("log")


               ax2.set_xlim(0, DaphneChannel.WAVEFORM_LENGTH)
               ax2.set_ylim(4000, 15000)
               xlim1 = ax1.get_xlim()
               ylim1 = ax1.get_ylim()
              
               xlim2 = ax2.get_xlim()
               ylim2 = ax2.get_ylim()


               bin= list(range(CH_0_THRESHOLD_ADC_UNITS,2**14,1))


               frecuencias = np.array(2**14, dtype=np.int32)
               aux         = np.zeros(len(bin)-1)
               counts, bins = np.histogram(aux,bin)
               frecuencias_acumuladas=aux


               def on_press(event):
                   if event.key == "g":
                       try:
                           times = int(time.time())
                           f.write("Saving histogram..."+"\n")
                           np.save(f"histogramas/{times}_bin", bin)
                           np.save(f"histogramas/{times}_histograma", frecuencias_acumuladas)
                       except Exception as e:
                           f.write(str(e) + "\n")


               figure.canvas.mpl_connect('key_press_event', on_press)


               while True:


                   _update_plot = update_plot.value
                   if not _update_plot:
                       continue


                   len(channels_last[0].waveform_data) and ax1.cla()
                   ax1.set_xlim(*xlim1)
                   ax1.set_ylim(*ylim1)
                   # ax1.set_yscale("log")


                   ax2.cla()
                   ax2.set_xlim(*xlim2)
                   ax2.set_ylim(*ylim2)
                   # plt.cla()


                   for ii in range(len(channels_last)):
                       cuentas = len(channels_last[ii].waveform_data)//DaphneChannel.WAVEFORM_LENGTH
                       if cuentas == 0:
                           continue
                       wf = channels_last[ii].waveform_data
                       segmentos = [wf[i:i+DaphneChannel.WAVEFORM_LENGTH] for i in range(0, cuentas*DaphneChannel.WAVEFORM_LENGTH, DaphneChannel.WAVEFORM_LENGTH)]
                       amplitudes, undershoots=[],[]


                       max_waves = 10 if cuentas >= 10 else cuentas
                       i = 0
                       filtrados = 0
                       for segmento in segmentos[:]:
                           if len(segmento) == 0:
                               continue
                           a1 = np.array(segmento[:22])


                           if  any(np.array(segmento[:24]) > 5960):
                               filtrados += 1
                               continue
                           if any(np.array(segmento) < 5900):
                               filtrados += 1
                               continue
                           undershoot = max(segmento)
                           undershoots.append(undershoot)
                           # if i < 50:
                           ax2.plot(x_plot, segmento, lw=0.05, color='b', alpha=0.8)
                       print(f"{filtrados}/{cuentas}")
                       counts, bins = np.histogram(undershoots, bin, density=False)
                       frecuencias_acumuladas = frecuencias_acumuladas + counts


                       ax1.plot(bins[:-1], frecuencias_acumuladas, lw=0.5)
                       ax1.set_ylim(0, max(frecuencias_acumuladas))


                  
                   ax2.text(0.95, 0.01, datetime.datetime.fromtimestamp(time.time()),
                   verticalalignment='bottom', horizontalalignment='right',
                   transform=ax2.transAxes,
                   #fontsize=15
                   )


                   figure.canvas.draw()
                   figure.canvas.flush_events()
                   ylim1 = ax1.get_ylim()
                   xlim1 = ax1.get_xlim()
                   ylim2 = ax2.get_ylim()
                   xlim2 = ax2.get_xlim()
                   f.flush()
                   # time.sleep(0.05)
           except Exception as e:
               print(str(e))
               f.write(str(e)+"\n")


def store_data_to_buffer(
       timestamp,
       trigger_rate_top,
       trigger_rate_mid,
       trigger_rate_bot,
       sum_amplitude_top,
       sum_amplitude_mid,
       sum_amplitude_bot):
   #logger.debug(f"Storing data to buffer: {UPLOAD_BUFFER_PATH}")
   try:
       average_amplitude_top = 0 if trigger_rate_top == 0 else sum_amplitude_top//trigger_rate_top
       average_amplitude_mid = 0 if trigger_rate_mid == 0 else sum_amplitude_mid//trigger_rate_mid
       average_amplitude_bot = 0 if trigger_rate_bot == 0 else sum_amplitude_bot//trigger_rate_top
       with open(UPLOAD_BUFFER_PATH, "a") as f:
           f.write(f"{timestamp},{trigger_rate_top},{trigger_rate_mid},{trigger_rate_bot},{average_amplitude_top},{average_amplitude_mid},{average_amplitude_bot}\n")
   except Exception as e:
       logger.error(f"store_data_to_buffer({UPLOAD_BUFFER_PATH}): {e}")


SQL_COMMAND = f"""\
INSERT INTO {DB_NAME}.{DB_TABLE}
(`timestamp`, TOP, MID, BOT, TOP_AMPLITUDE_SUM, MID_AMPLITUDE_SUM, BOT_AMPLITUDE_SUM)
VALUES(%s, %s, %s, %s, %s, %s, %s);
"""


# --- NEW: Function to plot analysis results ---
def plot_analysis_results(results_per_channel):
    """
    Plots the histograms of DCR, XTP, and APP for each channel.
    results_per_channel: dict { 'CH0': {'DCR': [], 'XTP': [], 'APP': []}, ... }
    """
    for ch_name, metrics in results_per_channel.items():
        if not metrics or not any(metrics.values()):
            print(f"No results to plot for {ch_name}")
            continue

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"{ch_name} Analysis Summary", fontsize=16)

        # Plot DCR
        if metrics['DCR']:
            dcr_vals = [v * 1e3 for v in metrics['DCR'] if v is not None] # Convert to mHz/mm2
            if dcr_vals:
                axs[0].hist(dcr_vals, bins='auto', color='blue', alpha=0.7)
                axs[0].set_title("DCR Distribution")
                axs[0].set_xlabel("DCR (mHz/mm²)")
                axs[0].set_ylabel("Counts")
                axs[0].text(0.05, 0.95, f"Mean: {np.mean(dcr_vals):.2f}\nStd: {np.std(dcr_vals):.2f}", 
                            transform=axs[0].transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

        # Plot XTP
        if metrics['XTP']:
            xtp_vals = [v * 100 for v in metrics['XTP'] if v is not None] # Convert to %
            if xtp_vals:
                axs[1].hist(xtp_vals, bins='auto', color='green', alpha=0.7)
                axs[1].set_title("Crosstalk Prob. Distribution")
                axs[1].set_xlabel("XTP (%)")
                axs[1].set_ylabel("Counts")
                axs[1].text(0.05, 0.95, f"Mean: {np.mean(xtp_vals):.2f}\nStd: {np.std(xtp_vals):.2f}", 
                            transform=axs[1].transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

        # Plot APP
        if metrics['APP']:
            app_vals = [v * 100 for v in metrics['APP'] if v is not None] # Convert to %
            if app_vals:
                axs[2].hist(app_vals, bins='auto', color='red', alpha=0.7)
                axs[2].set_title("Afterpulsing Prob. Distribution")
                axs[2].set_xlabel("APP (%)")
                axs[2].set_ylabel("Counts")
                axs[2].text(0.05, 0.95, f"Mean: {np.mean(app_vals):.2f}\nStd: {np.std(app_vals):.2f}", 
                            transform=axs[2].transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show() # Can block if non-interactive, but safe at end of script


def main(arduino_port='/dev/ttyACM3',arduino_port2='/dev/ttyACM1',arduino_port3='/dev/ttyACM2'):
   input(f"Is the file name: <<{STORE_HEADER_COMMENT}>> correct?")


   try:
       start_run = int(input("Introduce el número de inicio del run (start_run, entre 0 y 17): "))
       end_run = int(input("Introduce el número final del run (end_run, entre 0 y 17): "))


       # Validación de rango de inicio y fin para evitar errores
       if start_run < 0 or end_run > 17 or start_run > end_run:
           print("Error: Asegúrate de que el rango esté entre 0 y 17 y que start_run sea menor o igual a end_run.")
           return
   except ValueError:
       print("Error: Asegúrate de ingresar números enteros válidos para start_run y end_run.")
       return
   ##############
   try:
       RUN_DURATION_SECONDS = int(input("Introduce la duración de la medida en segundos (RUN_DURATION_SECONDS), entre 10 y 600): "))


       # Validación de rango de inicio y fin para evitar errores
       if RUN_DURATION_SECONDS < 0 or RUN_DURATION_SECONDS > 600 :
           print("Error: Asegúrate de que el rango esté entre 0 y 600.")
           return
   except ValueError:
       print("Error: Asegúrate de ingresar números enteros válidos para RUN_DURATION_SECONDS.")
       return


   try:
       MEAS_TYPE = int(input("Introduce el tipo de medida, DCR = 0 o GAIN = 1): "))


       # Validación de rango de inicio y fin para evitar errores
       if MEAS_TYPE < 0 or MEAS_TYPE > 1 :
           print("Error: Asegúrate de que el valor sea 0(DCR) o 1(GAIN).")
           return
   except ValueError:
       print("Error: Asegúrate de ingresar números enteros válidos para RUN_DURATION_SECONDS.")
       return


   if (MEAS_TYPE == 0):
       TYPE = "W"
   elif (MEAS_TYPE == 1):
       TYPE = "G"


   # Identify channels first
   channel_0 = DaphneChannel(
       "CH0",
       reg.FIFO_ch0_ADDR,
       reg.FIFO_ch0_TS_ADDR,
       reg.FIFO_ch0_WR_ADDR,
       reg.ch0_THRESHOLD_ADDR,
       CH_0_THRESHOLD_ADC_UNITS,
       CH_0_BASELINE)


   channel_1 = DaphneChannel(
       "CH1",
       reg.FIFO_ch1_ADDR,
       reg.FIFO_ch1_TS_ADDR,
       reg.FIFO_ch1_WR_ADDR,
       reg.ch1_THRESHOLD_ADDR,
       CH_1_THRESHOLD_ADC_UNITS,
       CH_1_BASELINE)


   channel_2 = DaphneChannel(
       "CH2",
       reg.FIFO_ch2_ADDR,
       reg.FIFO_ch2_TS_ADDR,
       reg.FIFO_ch2_WR_ADDR,
       reg.ch2_THRESHOLD_ADDR,
       CH_2_THRESHOLD_ADC_UNITS,
       CH_2_BASELINE)       
  
   channels_obj = []
   if CH_0_ENABLE:
       channels_obj.append(channel_0)
   if CH_1_ENABLE:
       channels_obj.append(channel_1)
   if CH_2_ENABLE:
       channels_obj.append(channel_2)
   
   channels: Tuple[DaphneChannel] = tuple(channels_obj)

   # --- NEW: Dictionary to store analysis results per channel ---
   analysis_results = {ch.IDENTIFIER: {'DCR': [], 'XTP': [], 'APP': []} for ch in channels}
   files_created_in_this_run = []


   try:
       arduino = serial.Serial(arduino_port, 9600, timeout=1)
       time.sleep(2)  # Espera para inicializar conexión
       print(f"Conectado al Arduino en {arduino_port}.")
   except Exception as e:
       print(f"Error al conectar con el Arduino: {e}")
       return


   try:
       arduino2 = serial.Serial(arduino_port2, 9600, timeout=1)
       time.sleep(2)  # Espera para inicializar conexión
       print(f"Conectado al Arduino2 en {arduino_port2}.")
   except Exception as e:
       print(f"Error al conectar con el Arduino2: {e}")
       return

   try:
       arduino3 = serial.Serial(arduino_port3, 9600, timeout=1)
       time.sleep(2)  # Espera para inicializar conexión
       print(f"Conectado al Arduino3 en {arduino_port3}.")
   except Exception as e:
       print(f"Error al conectar con el Arduino3: {e}")
       return


   RUN_DURATION_SECONDS_2 = 600  # 10 minutos en segundos
  
   start_time = time.time()  # Tiempo de inicio del primer run


   if PLOT_HISTOGRAMS:
       manager = Manager()
       channels_last = manager.list()
       update_plot = manager.Value("i", 0)
       for ch in channels:
           channels_last.append(copy.deepcopy(ch))
       #plot_process = Process(target=plot_data, args=(channels_last, update_plot))
       #plot_process.start()


   if (MEAS_TYPE == 0):
       thing = OEI(DAPHNE_IP)
       thing.write(reg.SOFT_TRIGGER_MODE_ADDR, [1234])
       for channel in channels:
           channel.write_threshold_value(thing)
           channel.empty_fifos(thing)


       logger.debug("Setting self-trigger mode")
       thing.write(reg.SELF_TRIGGER_MODE_ADDR, [1234])


       for run_counter in range(start_run, end_run + 1):
           os.system("clear")
           print("======= MASSIBO/DAPHNE Acquisition System =======")
           print()
           print(f"Uploading to DB:\t{UPLOAD_TO_DB}")
           print(f"Storing Waveforms:\t{STORE_WAVEFORMS}")
           print()
           print(f"Canal\tThresh\tRate\tNoVal\tCeros\tSumAmp\tAvgAmp")
           for ch in channels:
               print(f"{ch.IDENTIFIER}\t{ch.threshold_adc_units}\t{ch.cuentas}\t{ch.contador_no_validos}\t{ch.contador_ceros}\t{ch.suma_amplitudes}\t{ch.suma_amplitudes//ch.cuentas if ch.cuentas != 0 else 0}")
           print()   
           

           #print(f"Current Timestamp: {last_upload_time}")
           #print(f"Delta time:\t{time.time()-inicio}")
           arduino.write(f"{run_counter}\n".encode())
           time.sleep(1) 
           arduino2.write(f"{run_counter}\n".encode())
           time.sleep(1)  # Ajusta el tiempo según sea necesario para asegurar transmisión
           arduino3.write(f"{run_counter}\n".encode())
           time.sleep(1)  # Ajusta el tiempo según sea necesario para asegurar transmisión
           run_start_time = time.time()  # Tiempo de inicio de cada run

           for channel in channels:
               channel.empty_fifos(thing)


           # Reiniciar acumuladores UNA vez al inicio del run
           for channel in channels:
               channel.contador_ceros = 0
               channel.contador_no_validos = 0
               channel.cuentas = 0
               channel.waveform_data = []
               channel.timestamp_data = []
               channel.suma_amplitudes = 0


           while time.time() - run_start_time < RUN_DURATION_SECONDS:
               inicio = time.time()
               #update_plot.set(0)


               while time.time() - inicio < 1.0:
                   try:
                       readable_flags = thing.readf(reg.READABLE_FLAG_ADDR, 1)[2]
                       flags = (readable_flags & 0b1), ((readable_flags >> 1) & 0b1), ((readable_flags >> 2) & 0b1)


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


           # Al finalizar el run, escribir una vez por canal lo acumulado
           for ch in channels:
               if len(ch.waveform_data) > 0:
                   write_to_file_numpy(ch, run_start_time, STORE_HEADER_COMMENT, run_counter, TYPE)
                   
                   # --- Collect filename for analysis ---
                   sanitized_comment = sanitize_filename(STORE_HEADER_COMMENT or "default")
                   filename = f"{sanitized_comment}_{TYPE}_{ch.IDENTIFIER}_{run_counter}_{TYPE}.npy"
                   filepath = os.path.join(STORE_WAVEFORMS_DIR, filename)
                   files_created_in_this_run.append((ch.IDENTIFIER, filepath))


           print(f"Run {run_counter} completado en {int(time.time() - start_time)} segundos desde inicio.")


       print("Proceso de adquisición finalizado.")


   elif (MEAS_TYPE == 1):
       thing = OEI(DAPHNE_IP)
       thing.write(reg.SOFT_TRIGGER_MODE_ADDR, [1234])
       for channel in channels:
           channel.write_threshold_value(thing)
           channel.empty_fifos(thing)


       logger.debug("Setting self-trigger mode")
       thing.write(reg.EXT_TRIGGER_MODE_ADDR, [1234])


       for run_counter in range(start_run, end_run + 1):
           os.system("clear")
           print("======= MASSIBO/DAPHNE Acquisition System =======")
           print()
           print(f"Uploading to DB:\t{UPLOAD_TO_DB}")
           print(f"Storing Waveforms:\t{STORE_WAVEFORMS}")
           print()
           print(f"Canal\tThresh\tRate\tNoVal\tCeros\tSumAmp\tAvgAmp")
           for ch in channels:
               print(f"{ch.IDENTIFIER}\t{ch.threshold_adc_units}\t{ch.cuentas}\t{ch.contador_no_validos}\t{ch.contador_ceros}\t{ch.suma_amplitudes}\t{ch.suma_amplitudes//ch.cuentas if ch.cuentas != 0 else 0}")
           print()   
           #print(f"Current Timestamp: {last_upload_time}")
           #print(f"Delta time:\t{time.time()-inicio}")
           arduino.write(f"{run_counter}\n".encode())
           time.sleep(1)  # Ajusta el tiempo según sea necesario para asegurar transmisión
           arduino2.write(f"{run_counter}\n".encode())
           time.sleep(1)  # Ajusta el tiempo según sea necesario para asegurar transmisión
           arduino3.write(f"{run_counter}\n".encode())
           time.sleep(1)  # Ajusta el tiempo según sea necesario para asegurar transmisión
           run_start_time = time.time()  # Tiempo de inicio de cada run


           # Reiniciar acumuladores UNA vez al inicio del run
           for channel in channels:
               channel.contador_ceros = 0
               channel.contador_no_validos = 0
               channel.cuentas = 0
               channel.waveform_data = []
               channel.timestamp_data = []
               channel.suma_amplitudes = 0


           while time.time() - run_start_time < RUN_DURATION_SECONDS:
               # Iniciar adquisición de datos
               inicio = time.time()
               #update_plot.set(0)


               while time.time() - inicio < 1.0:
                   try:
                       readable_flags = thing.readf(reg.READABLE_FLAG_ADDR, 1)[2]
                       flags = (readable_flags & 0b1), ((readable_flags >> 1) & 0b1), ((readable_flags >> 2) & 0b1)


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


           # Al finalizar el run, escribir una vez por canal lo acumulado
           for ch in channels:
               if len(ch.waveform_data) > 0:
                   write_to_file_numpy(ch, run_start_time, STORE_HEADER_COMMENT, run_counter, TYPE)
                   
                   # --- Collect filename for analysis ---
                   sanitized_comment = sanitize_filename(STORE_HEADER_COMMENT or "default")
                   filename = f"{sanitized_comment}_{TYPE}_{ch.IDENTIFIER}_{run_counter}_{TYPE}.npy"
                   filepath = os.path.join(STORE_WAVEFORMS_DIR, filename)
                   files_created_in_this_run.append((ch.IDENTIFIER, filepath))


           print(f"Run {run_counter} completado en {int(time.time() - start_time)} segundos desde inicio.")


       print("Proceso de adquisición finalizado.")

   
   # --- POST-PROCESSING: ANALYZE FILES ---
   if len(files_created_in_this_run) > 0:
       print("\n" + "="*40)
       print("       STARTING AUTOMATED ANALYSIS")
       print("="*40)
       
       for ch_id, filepath in files_created_in_this_run:
           if os.path.exists(filepath):
               print(f"Analyzing {os.path.basename(filepath)}...")
               try:
                   # Call analysis function (plot=False to avoid blocking figures during loop)
                   dcr, xtp, app = analyze_file(filepath, plot=False)
                   
                   if dcr is not None:
                       analysis_results[ch_id]['DCR'].append(dcr)
                       analysis_results[ch_id]['XTP'].append(xtp)
                       analysis_results[ch_id]['APP'].append(app)
                       print(f" -> Results [ {ch_id} ]: DCR={dcr*1e3:.2f} mHz/mm2 | XTP={xtp*100:.2f}% | APP={app*100:.2f}%")
                   else:
                       print(f" -> Analysis failed or no valid pulses found.")
               except Exception as e:
                   print(f" -> Error analyzing file: {e}")
           else:
               print(f"File not found: {filepath}")
       
       print("\n" + "="*40)
       print("          PLOTTING SUMMARY")
       print("="*40)
       plot_analysis_results(analysis_results)
       input("Press Enter to close plots and exit...")


if __name__ == "__main__":
   main()
