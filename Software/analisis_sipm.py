import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy.ndimage import gaussian_filter1d
import os
from filter_waveform import (
    apply_waveform_filter,
    IDX_SIPM_MANUAL,
    DEFAULT_THRESHOLD_CORR,
    DEFAULT_ENABLE_ITERATIVE_REFINEMENT,
    DEFAULT_ENABLE_PHYSICS_DISCRIMINATION,
    TIME_PER_SAMPLE_NS, # Use this from filter_waveform for consistency if it's the digitizer's sample time
    MIN_PULSE_WIDTH_NS,
    NOISE_WIDTH_NS,
    SIPM_RISE_TIME_NS,
    SIPM_FALL_TIME_NS
)

# --- Constantes de Análisis ---

# Período del reloj de la FPGA en ns. Usamos TIME_PER_SAMPLE_NS del filtro.
CLOCK_PERIOD_NS = TIME_PER_SAMPLE_NS 

# Muestras al inicio de la waveform para calcular el baseline
BASELINE_SAMPLES = 40
# Ancho de la ventana de integración alrededor del pico del pulso (en muestras)
INTEGRATION_WINDOW = 30
# Umbral de altura inicial para la detección de picos (en ADU).
# Este valor debería ser ~0.5 p.e. pero se usa como punto de partida.
PEAK_HEIGHT_THRESHOLD = 5

# Área activa del SiPM en mm^2 (valor de ejemplo, ajustar según el SiPM)
SIPM_ACTIVE_AREA_MM2 = 36.0 # Placeholder, user should adjust this value


# --- Funciones de Carga y Procesamiento Básico ---

def load_waveforms_from_npy(filepath):
    """
    Carga waveforms y timestamps desde un archivo .npy.
    Retorna timestamps y waveforms como arrays de numpy.
    """
    if not os.path.exists(filepath):
        print(f"Error: El archivo no existe en la ruta: {filepath}")
        return None, None

    data = np.load(filepath)
    if data.shape[0] == 0:
        print("Advertencia: El archivo está vacío.")
        return None, None

    timestamps = data[:, 0].astype(np.uint64)
    waveforms = data[:, 1:].astype(np.int16)
    
    print(f"Cargadas {waveforms.shape[0]} waveforms de longitud {waveforms.shape[1]}.")
    return timestamps, waveforms

def process_single_waveform(waveform):
    """
    Invierte una waveform, sustrae el baseline y encuentra picos.
    Retorna una lista de diccionarios, cada uno con 'amplitude', 'charge' y 'position' de un pulso.
    """
    # 1. Invertir la señal (los pulsos de SiPM son negativos)
    inverted_wf = -waveform

    # 2. Calcular y sustraer el baseline
    baseline = np.mean(inverted_wf[:BASELINE_SAMPLES])
    corrected_wf = inverted_wf - baseline

    # 3. Encontrar picos (pulsos)
    peaks, properties = signal.find_peaks(corrected_wf, height=PEAK_HEIGHT_THRESHOLD, distance=7, prominence=4.0, width=4.0)
    
    pulses = []
    for i, peak_idx in enumerate(peaks):
        # 4. Calcular carga (integral) y amplitud para cada pulso
        start = max(0, peak_idx - INTEGRATION_WINDOW // 2)
        end = min(len(corrected_wf), peak_idx + INTEGRATION_WINDOW // 2)
        
        charge = np.sum(corrected_wf[start:end])
        amplitude = properties['peak_heights'][i]
        
        pulses.append({
            'charge': charge,
            'amplitude': amplitude,
            'position': peak_idx
        })
        
    return pulses

# --- Funciones para Análisis de Carga (Ganancia y Crosstalk) ---

def multi_gaussian_fit(charges, num_peaks=5, plot=False, show_xtp=False, xlabel='Integrated Charge (ADU x samples)', title='Charge Histogram (Finger Plot) - DCR Trigger (Starts at 1 p.e.)', unit_label='(ADU * muestras)'):
    """
    Ajusta un modelo de múltiples gaussianas al histograma de cargas.
    Retorna los parámetros del ajuste, la ganancia y la información del pedestal.
    """
    counts, bin_edges = np.histogram(charges, bins=200, range=(np.percentile(charges, 1), np.percentile(charges, 99)))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Función de la suma de N gaussianas
    def multi_gauss(x, *params):
        y = np.zeros_like(x, dtype=float)
        for i in range(0, len(params), 3):
            amp, mean, std = params[i], params[i+1], params[i+2]
            y += amp * norm.pdf(x, mean, std)
        return y

    # Estimación inicial de parámetros
    # 1. Suavizar el histograma (Sigma=2.0 para mantener forma pero reducir ruido)
    counts_smoothed = gaussian_filter1d(counts, sigma=2.0)
    
    # 2. Estimar Ganancia usando Autocorrelación para encontrar la periodicidad
    # Esto hace que el código se adapte automáticamente al espaciado de los fingers
    autocorr = signal.correlate(counts_smoothed, counts_smoothed, mode='full')
    autocorr = autocorr[len(autocorr)//2:] # Quedarse con lags positivos
    
    # Buscar el primer pico de la autocorrelación (que corresponde al periodo fundamental/ganancia)
    # Ignoramos el lag 0 (que es el máximo de auto-correlación trivial) buscando desde lag 10
    ac_peaks, _ = signal.find_peaks(autocorr[10:], distance=10, prominence=np.max(autocorr)*0.05)
    
    if len(ac_peaks) > 0:
        estimated_gain_bins = ac_peaks[0] + 10 # Sumar el offset de 10
        min_distance = int(estimated_gain_bins * 0.6) # Usar 60% de la ganancia como distancia mínima
        print(f"  [Auto-Calibration] Estimated Gain period: {estimated_gain_bins} bins. Setting min_distance={min_distance}.")
    else:
        # Fallback si falla la autocorrelación
        print("  [Auto-Calibration] Could not determine period from autocorrelation. Using default distance.")
        min_distance = 15 

    # 3. Buscar picos usando la distancia dinámica calculada
    # Prominence baja (2.0) para detectar fingers pequeños, confiando en el suavizado y la distancia correcta
    peak_indices, _ = signal.find_peaks(counts_smoothed, height=3, distance=min_distance, prominence=2.0, width=2.0)
    
    if len(peak_indices) < 2:
        print("Error: No se encontraron suficientes picos en el histograma para el ajuste multi-gaussiano.")
        # Fallback: try to find at least one peak and estimate gain
        if len(peak_indices) == 1:
            print("Intentando ajuste con un solo pico.")
            mean_est = bin_centers[peak_indices[0]] # Assuming the single peak is the 1 p.e. peak
            std_est = np.std(charges) / 5 # Rough estimate based on overall charge spread
            gain_est = std_est * 3 # Rough estimate
            pedestal_est = {'mean': mean_est, 'std': std_est}
            print(f"Ganancia estimada (aproximada): {gain_est:.2f} (ADU * muestras) / p.e.")
            return gain_est, pedestal_est
        return None, None

    
    initial_params = []
    for i in peak_indices[:num_peaks]:
        amp = counts[i]
        mean = bin_centers[i]
        std = np.std(charges) / (2 * num_peaks) # Estimación gruesa
        initial_params.extend([amp, mean, std])

    try:
        popt, _ = curve_fit(multi_gauss, bin_centers, counts, p0=initial_params, maxfev=10000, bounds=(0, np.inf))
    except RuntimeError:
        print("Error: El ajuste de curva gaussiana no convergió. Intentando con menos picos.")
        try:
            popt, _ = curve_fit(multi_gauss, bin_centers, counts, p0=initial_params[:6], maxfev=10000, bounds=(0, np.inf)) # Try with 2 peaks
        except RuntimeError:
            print("Error: El ajuste de curva gaussiana no convergió incluso con menos picos.")
            return None, None

    # Extraer centroides (medias) y ordenarlos
    means = sorted([popt[i+1] for i in range(0, len(popt), 3)])
    
    if len(means) < 2:
        print("Error: No se pudieron extraer suficientes medias del ajuste para calcular la ganancia.")
        return None, None

    gain = np.mean(np.diff(means))
    pedestal_mean = means[0] - gain # Assume first peak is 1 p.e., so 0 p.e. is one gain unit behind
    # Find the std corresponding to the first peak (now 1 p.e.)
    first_peak_std_idx = next((i for i in range(0, len(popt), 3) if abs(popt[i+1] - means[0]) < (popt[i+2] / 2)), -1)
    pedestal_std = abs(popt[first_peak_std_idx + 2]) if first_peak_std_idx != -1 else np.std(charges) / 10 


    
    # Plot
    # Plot
    if plot:
        plt.figure(figsize=(12, 7))
        plt.hist(charges, bins=200, range=(np.percentile(charges, 1), np.percentile(charges, 99)), label='Data', alpha=0.6)
        plt.plot(bin_centers, multi_gauss(bin_centers, *popt), 'r-', label='Multi-Gaussian Fit')
        for i, mean in enumerate(means):
            # i=0 is 1 p.e., i=1 is 2 p.e., etc.
            label = f'Peak {i+1} p.e.' if i < 4 else None
            plt.axvline(mean, color='k', linestyle='--', label=label)
        
        if show_xtp:
            # Threshold at 1.5 p.e. 
            xtp_threshold_val = pedestal_mean + 1.5 * gain
            # Linea de corte a 1.5 p.e.
            plt.axvline(xtp_threshold_val, color='magenta', linestyle='-.', linewidth=2, label='XTP Threshold (1.5 p.e.)')
            # Sombrear la región de Crosstalk
            plt.fill_between(bin_centers, 0, counts, where=(bin_centers > xtp_threshold_val), color='magenta', alpha=0.3, label='Crosstalk Region')

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel('Counts')
        plt.legend()
        plt.grid(True)
        plt.show()

    print(f"Ganancia estimada: {gain:.2f} {unit_label} / p.e.")
    print(f"Pedestal (Extrapolado 0 p.e.): Media={pedestal_mean:.2f}")

    return gain, {'mean': pedestal_mean, 'std': pedestal_std}

# --- Nuevas funciones de cálculo de key parameters ---

def calculate_dcr_no_bursts(timestamps, area_mm2, delta_t_max=0.1, n_min=5):
    """
    Calcula el Dark Count Rate (DCR) eliminando el componente de 'bursts' 
    mediante el 'tag method' descrito en el paper.
    
    Parámetros:
    -----------
    timestamps : array-like
        Tiempos de llegada de los pulsos en segundos (ordenados).
    area_mm2 : float
        Área activa del sensor en mm^2.
    delta_t_max : float
        Límite superior de tiempo entre eventos para identificar ráfagas. 
        Valor optimizado en el paper: ~100 ms.
    n_min : int
        Número mínimo de eventos consecutivos para considerar una ráfaga. 
        Valor optimizado en el paper: 5.
        
    Retorna:
    --------
    dcr_no_bursts : float
        Tasa de conteo oscuro en Hz/mm^2 (mHz/mm^2 si se multiplica por 1000).
    """
    n_events = len(timestamps) # Total number of individual pulses
    if n_events < 2:
        return 0.0

    # Calcular intervalos de tiempo entre eventos consecutivos (Delta t)
    delta_t = np.diff(timestamps)
    
    # Máscara para identificar eventos que pertenecen a una ráfaga 
    is_burst_event = np.zeros(n_events, dtype=bool)
    
    i = 0
    while i < len(delta_t):
        if delta_t[i] < delta_t_max:
            # Encontrar la longitud de la secuencia de eventos rápidos
            j = i
            while j < len(delta_t) and delta_t[j] < delta_t_max:
                j += 1
            
            # Si la secuencia es >= n_min, marcar como ráfaga 
            count = (j - i) + 1
            if count >= n_min:
                is_burst_event[i:j+1] = True
            i = j
        else:
            i += 1
    
    non_burst_pulse_indices = np.where(~is_burst_event)[0] # Indices of pulses that are NOT bursts
    standard_events_count = len(non_burst_pulse_indices)
    total_time = timestamps[-1] - timestamps[0] # Total acquisition time from first to last pulse
    
    if total_time == 0 or area_mm2 == 0:
        return 0.0

    dcr = standard_events_count / (total_time * area_mm2)
    return dcr, non_burst_pulse_indices # Return DCR and indices of non-burst pulses

def calculate_dcr_raw(timestamps, area_mm2):
    """
    Calcula el DCR Total sin rechazo de ráfagas (Raw DCR).
    
    Parámetros:
    -----------
    timestamps : array-like
        Tiempos de llegada de todos los pulsos en segundos.
    area_mm2 : float
        Área activa del sensor en mm^2.
        
    Retorna:
    --------
    dcr_raw : float
        Tasa de conteo total en Hz/mm^2.
    """
    n_events = len(timestamps)
    if n_events < 2 or area_mm2 == 0:
        return 0.0
        
    total_time = timestamps[-1] - timestamps[0]
    
    if total_time == 0:
        return 0.0
        
    dcr_raw = n_events / (total_time * area_mm2)
    return dcr_raw

def calculate_crosstalk_probability(amplitudes_pe):
    """
    Calcula la probabilidad de Cross-talk (XT).
    
    Definición: Número de eventos con amplitud > 1.5 p.e. dividido 
    por el número total de eventos.
    
    Parámetros:
    -----------
    amplitudes_pe : array-like
        Amplitudes de los pulsos detectados expresadas en unidades de 
        fotoelectrones (p.e.).
        
    Retorna:
    --------
    xt_probability : float
        Probabilidad de cross-talk (0 a 1).
    """
    if len(amplitudes_pe) == 0:
        return 0.0
        
    # Eventos por encima del umbral de 1.5 p.e. 
    events_above_threshold = np.sum(np.array(amplitudes_pe) > 1.5)
    total_events = len(amplitudes_pe)
    
    return events_above_threshold / total_events

def calculate_afterpulse_probability(timestamps):
    """
    Calcula la probabilidad de After-pulse (AP).
    
    Definición: Número de pulsos con un retraso temporal inferior a 5 microsegundos 
    dividido por el número total de pulsos[cite: 819].
    
    Parámetros:
    -----------
    timestamps : array-like
        Tiempos de llegada de los pulsos en segundos.
        
    Retorna:
    --------
    ap_probability : float
        Probabilidad de after-pulse (0 a 1).
    """
    if len(timestamps) < 2:
        return 0.0
        
    # Calcular el tiempo de retraso con respecto al pulso anterior [cite: 809]
    delta_t = np.diff(sorted(timestamps)) # Ensure timestamps are sorted for diff
    
    # Umbral de 5 microsegundos establecido según requisitos de DAQ 
    ap_threshold = 5e-6 
    
    afterpulse_events = np.sum(delta_t < ap_threshold)
    total_pulses = len(timestamps)

    # --- Visualización APP (Añadido) ---
    delta_t_us = delta_t * 1e6 # Convertir a microsegundos para el plot
    
    plt.figure(figsize=(10, 6))
    # Usar escala logarítmica en ambos ejes suele ser útil para Delta t
    # Pero para visualizar el corte de 5us linear-x log-y puede ser más claro en la zona corta
    # Vamos a probar Log-Log para ver todo el rango dinámico
    
    # Crear bins logarítmicos para abarcar desde tiempos muy cortos hasta largos
    if len(delta_t_us) > 0:
        min_dt = max(np.min(delta_t_us), 1e-4) # Evitar 0 para log
        max_dt = np.max(delta_t_us)
        bins = np.logspace(np.log10(min_dt), np.log10(max_dt), 100)
        
        plt.hist(delta_t_us, bins=bins, alpha=0.7, color='teal', label='Delta t (pulse interval)')
        plt.axvline(ap_threshold * 1e6, color='red', linestyle='--', linewidth=2, label='APP Threshold (5 µs)')
        
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Delta t (µs)')
        plt.ylabel('Counts')
        plt.title('Pulse Time Difference Distribution (Delta t) - APP Calculation')
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.legend()
        plt.show()
    
    return afterpulse_events / total_pulses

# --- Nueva función para plotear waveforms de DCR ---
def plot_random_dcr_waveforms(accepted_waveforms, selected_wf_indices_to_plot):
    """
    Plotea 30 waveforms aleatorias que contribuyen al DCR, marcando los picos encontrados.
    """
    if len(selected_wf_indices_to_plot) == 0: # Corrected: Check if the NumPy array is empty
        print("No hay waveforms seleccionadas para plotear.")
        return

    plt.figure(figsize=(15, 10))
    plt.suptitle("Random Waveforms Contributing to DCR (Peaks Marked with 'x')", fontsize=16)
    
    num_plots = len(selected_wf_indices_to_plot)
    rows = int(np.ceil(num_plots / 5)) # 5 waveforms per row
    cols = min(num_plots, 5) # Max 5 columns
    
    for i, wf_idx in enumerate(selected_wf_indices_to_plot):
        ax = plt.subplot(rows, cols, i + 1)
        waveform = accepted_waveforms[wf_idx]
        
        # Re-procesar la waveform para obtener los picos para el ploteo
        pulses = process_single_waveform(waveform)
        
        # Invertir y corregir baseline para el ploteo
        inverted_wf = -waveform
        baseline = np.mean(inverted_wf[:BASELINE_SAMPLES])
        corrected_wf = inverted_wf - baseline
        
        x_axis = np.arange(len(waveform))
        ax.plot(x_axis, corrected_wf, color='blue', alpha=0.7)
        
        # Marcar picos
        for p in pulses:
            ax.plot(p['position'], corrected_wf[p['position']], 'rx', markersize=8)
            
        ax.set_title(f"WF {wf_idx}")
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.tick_params(axis='x', labelbottom=False) # Hide x-axis labels for cleaner look
        ax.tick_params(axis='y', labelleft=False) # Hide y-axis labels
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make space for suptitle
    plt.show()

# --- Nueva función para plotear resultados del filtro ---
def plot_results(acc, rej, t_sipm, t_leak):
    """
    Plotea los resultados del filtro: templates, waveforms aceptadas y rechazadas.
    """
    if t_sipm is None or t_leak is None:
        print("Advertencia: No se pueden plotear los resultados del filtro porque faltan los templates.")
        return
        
    wave_len = len(t_sipm)
    x = np.arange(wave_len)
    
    # Resample templates for plotting if needed
    if len(t_leak) != wave_len: t_leak = signal.resample(t_leak, wave_len)

    fig, ax = plt.subplots(3, 1, figsize=(10, 12))
    
    # 1. Templates Ideales
    ax[0].set_title("References")
    ax[0].plot(x, t_sipm, 'b', linewidth=2, label='SiPM Template (Dinamic Average)')
    ax[0].plot(x, t_leak, 'r--', linewidth=2, label='Ideal Noise (Syntetic Bipolar)')
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)
    
    # 2. Aceptadas
    ax[1].set_title(f"Accepted ({len(acc)})")
    if len(acc) > 0:
        # Plotear solo las primeras 200 para velocidad
        for w in acc[:200]: ax[1].plot(x, w, color='blue', alpha=0.05)
        ax[1].plot(x, np.mean(acc, axis=0), color='cyan', linewidth=2, label='Real Average')
        ax[1].legend()
    else:
        ax[1].text(0.5, 0.5, "NO SIGNALS ACCEPTED\n(Correct if file is noise only)", 
                   transform=ax[1].transAxes, ha='center', va='center', fontsize=14, color='red')
    ax[1].grid(True, alpha=0.3)
    
    # 3. Rechazadas
    ax[2].set_title(f"Rejected ({len(rej)})")
    if len(rej) > 0:
        for w in rej[:200]: ax[2].plot(x, w, color='red', alpha=0.05)
        ax[2].plot(x, np.mean(rej, axis=0), color='orange', linewidth=2, label='Noise Average')
        ax[2].legend()
    ax[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# --- Función Principal ---

def main():
    """
    Función principal que orquesta el análisis.
    """
    filepath = "/home/cbenitez/Desktop/setpre/setfbk_meas2_31_3_v2_W_CH0_/setfbk_meas2_31_3_v2_W_CH0_0_W.npy"

    raw_timestamps, raw_waveforms = load_waveforms_from_npy(filepath)
    if raw_waveforms is None:
        return

    # Convert waveforms to float32 as expected by filter_waveform.py (filter_waveform expects original polarity)
    raw_waveforms_float = raw_waveforms.astype(np.float32)

    # --- Aplicar filtro de waveforms para seleccionar eventos de SiPM ---
    print("\n--- Ejecutando filtro de waveforms para seleccionar eventos de SiPM ---")
    print("Asegúrate de que los índices manuales (IDX_SIPM_MANUAL) en filter_waveform.py")
    print("estén ajustados a tu dataset para una clasificación precisa.")
    
    accepted_timestamps, accepted_waveforms, rejected_timestamps, rejected_waveforms, tmpl_sipm, tmpl_leak = apply_waveform_filter(
        raw_timestamps, 
        raw_waveforms_float, 
        IDX_SIPM_MANUAL,
        # Pass all filter parameters
        DEFAULT_THRESHOLD_CORR, DEFAULT_ENABLE_ITERATIVE_REFINEMENT, DEFAULT_ENABLE_PHYSICS_DISCRIMINATION,
        TIME_PER_SAMPLE_NS, MIN_PULSE_WIDTH_NS, NOISE_WIDTH_NS, SIPM_RISE_TIME_NS, SIPM_FALL_TIME_NS
    )

    # --- Plotear resultados del filtro ---
    print("\nMostrando resultados del filtro (Aceptadas vs. Rechazadas)...")
    plot_results(accepted_waveforms, rejected_waveforms, tmpl_sipm, tmpl_leak)
    
    if len(accepted_waveforms) == 0:
        print("No se encontraron waveforms aceptadas después del filtrado. Finalizando análisis.")
        return

    # --- Procesar solo las waveforms aceptadas ---
    all_pulses = []
    all_pulses_info = [] # To store pulse data along with original waveform index and timestamp
    all_pulse_timestamps_ns = []
    
    print("\nProcesando waveforms aceptadas para encontrar pulsos...")
    
    for wf_idx, waveform in enumerate(accepted_waveforms):
        pulses = process_single_waveform(waveform)
        if pulses:
            all_pulses.extend(pulses)
            for p in pulses:
                # Use accepted_timestamps for pulse time calculation
                pulse_time_ns = accepted_timestamps[wf_idx] * CLOCK_PERIOD_NS + p['position'] * CLOCK_PERIOD_NS
                all_pulse_timestamps_ns.append(pulse_time_ns)
                all_pulses_info.append({'pulse_data': p, 'waveform_idx': wf_idx, 'timestamp_ns': pulse_time_ns})
        
    if not all_pulses:
        print("No se encontraron pulsos en las waveforms aceptadas. Finalizando análisis.")
        return
        
    print(f"Se encontraron un total de {len(all_pulses)} pulsos en las waveforms aceptadas.")

    # Convert all pulse timestamps to seconds for DCR and Afterpulsing calculations
    all_pulse_timestamps_s = np.array(all_pulse_timestamps_ns) * 1e-9

    # --- DCR Calculation ---
    # The new DCR function requires timestamps in seconds and area_mm2
    dcr, non_burst_pulse_indices = calculate_dcr_no_bursts(all_pulse_timestamps_s, SIPM_ACTIVE_AREA_MM2)
    dcr_raw = calculate_dcr_raw(all_pulse_timestamps_s, SIPM_ACTIVE_AREA_MM2)

    all_charges = [p['charge'] for p in all_pulses]
    # Usar plot=True para mostrar el plot de Carga
    gain, pedestal = multi_gaussian_fit(all_charges, plot=True)

    if gain is None:
        print("No se pudo determinar la ganancia (Carga). Se omiten los cálculos de crosstalk y afterpulsing.")
        return

    # --- Amplitude Analysis (NEW) ---
    print("\n--- Realizando Análisis de Amplitud ---")
    all_amplitudes = [p['amplitude'] for p in all_pulses]
    # Ajustar gaussianas a las amplitudes y plotear
    gain_amp, pedestal_amp = multi_gaussian_fit(
        all_amplitudes, 
        plot=True, 
        show_xtp=True,
        xlabel='Amplitude (ADU)', 
        title='Amplitude Histogram (Finger Plot)', 
        unit_label='(ADU)'
    )

    if gain_amp is not None:
        # Calcular XTP usando Amplitud
        # Convertir amplitudes a p.e.
        amplitudes_val_pe = [(amp - pedestal_amp['mean']) / gain_amp for amp in all_amplitudes]
        crosstalk_prob_amp = calculate_crosstalk_probability(amplitudes_val_pe)
        
        # Calculate XTP without bursts
        if len(non_burst_pulse_indices) > 0:
            amplitudes_no_bursts = [all_amplitudes[i] for i in non_burst_pulse_indices]
            amplitudes_val_pe_no_bursts = [(amp - pedestal_amp['mean']) / gain_amp for amp in amplitudes_no_bursts]
            crosstalk_prob_amp_no_bursts = calculate_crosstalk_probability(amplitudes_val_pe_no_bursts)
        else:
            crosstalk_prob_amp_no_bursts = 0.0
    else:
        crosstalk_prob_amp = 0.0
        crosstalk_prob_amp_no_bursts = 0.0
        print("No se pudo determinar la ganancia (Amplitud).")

    # --- Afterpulsing Calculation ---
    # The new afterpulse function requires timestamps in seconds
    ap_prob = calculate_afterpulse_probability(all_pulse_timestamps_s)
    
    # Calculate APP without bursts (using the indices found during DCR calculation)
    if len(non_burst_pulse_indices) > 0:
        non_burst_timestamps = all_pulse_timestamps_s[non_burst_pulse_indices]
        ap_prob_no_bursts = calculate_afterpulse_probability(non_burst_timestamps)
    else:
        ap_prob_no_bursts = 0.0

    # --- Plotear 30 waveforms aleatorias de DCR ---
    if len(non_burst_pulse_indices) > 0:
        if all_pulses_info:
            # Obtener los índices únicos de las waveforms que contienen pulsos no-burst
            unique_dcr_waveform_indices = list(set([all_pulses_info[i]['waveform_idx'] for i in non_burst_pulse_indices]))
            
            if len(unique_dcr_waveform_indices) > 0:
                # Seleccionar hasta 30 waveforms aleatorias para plotear
                num_to_plot = min(50, len(unique_dcr_waveform_indices))
                selected_wf_indices_to_plot = np.random.choice(unique_dcr_waveform_indices, num_to_plot, replace=False)
                plot_random_dcr_waveforms(accepted_waveforms, selected_wf_indices_to_plot)
            else:
                print("No hay waveforms únicas que contribuyan al DCR para plotear.")
        else:
            print("No se generó información de pulsos para el ploteo de DCR.")
    else:
        print("No hay pulsos que contribuyan al DCR para plotear.")

    print("\n--- Realizando Cálculos Alternativos (Paper Definitions) ---")
    
# 1. Preparar datos limpios (Sin Bursts)
    amps_clean = [all_amplitudes[i] for i in non_burst_pulse_indices]
    # Calcular amplitudes en p.e.
    amps_pe_clean = np.array([(amp - pedestal_amp['mean']) / gain_amp for amp in amps_clean])
    
    times_clean = all_pulse_timestamps_s[non_burst_pulse_indices]
    
    # --- CORRECCIÓN CRÍTICA AQUÍ ---
    # Filtramos para quedarnos SOLO con pulsos que sean > 0.5 p.e.
    # Esto elimina el ruido de la cola (undershoot/ringing) que finge ser Afterpulse.
    valid_pulse_mask = amps_pe_clean > 0.5
    
    times_clean_filtered = times_clean[valid_pulse_mask]
    
    # 2. Cálculo XTP Alternative (Strict > 1.5 p.e.)
    # Nota: XTP se calcula sobre el total de pulsos válidos (>0.5 pe)
    amps_pe_clean_filtered = amps_pe_clean[valid_pulse_mask]
    count_xtp = np.sum(amps_pe_clean_filtered > 1.5)
    total_clean = len(amps_pe_clean_filtered)
    XTP_alternative = count_xtp / total_clean if total_clean > 0 else 0.0
    
    # 3. Cálculo APP Alternative (< 5 us)
    # Ahora calculamos el tiempo SOLO entre pulsos válidos (> 0.5 p.e.)
    if len(times_clean_filtered) > 1:
        delta_t_clean = np.diff(sorted(times_clean_filtered))
        count_app = np.sum((delta_t_clean < 5e-6) & (delta_t_clean > 0.15e-6))
            
        APP_alternative = count_app / (len(times_clean_filtered) - 1)
    else:
        APP_alternative = 0.0
        
    

    print("\n" + "="*30)
    print("      RESUMEN DEL ANÁLISIS")
    print("="*30)
    print(f"Archivo: {os.path.basename(filepath)}")
    print(f"Waveforms totales cargadas: {len(raw_waveforms)}")
    print(f"Waveforms aceptadas por el filtro: {len(accepted_waveforms)}")
    print("-" * 20)
    print(f"DCR (No Bursts): {dcr * 1e3:.2f} mHz/mm²") 
    print(f"DCR (Raw/Total): {dcr_raw * 1e3:.2f} mHz/mm²")
    print("-" * 20)
    #print(f"Ganancia (Carga): {gain:.2f} (ADU x muestras)/p.e.")
    if gain_amp is not None:
        #print(f"Ganancia (Amplitud): {gain_amp:.2f} (ADU)/p.e.")
        #print("-" * 20)
        # Tus cálculos originales
        print(f"XTP : {crosstalk_prob_amp * 100:.2f} %")
        #print(f"APP (Original): {ap_prob_no_bursts * 100:.2f} %")
        print("-" * 20)
        # LOS NUEVOS CÁLCULOS
        #print(f"XTP alternative: {XTP_alternative * 100:.2f} %")
        print(f"APP : {APP_alternative * 100:.2f} %")
    print("="*30)

if __name__ == "__main__":
    main()