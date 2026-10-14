import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy.ndimage import gaussian_filter1d
import os

# --- CONFIGURACIÓN MANUAL ---
FILEPATH = "/home/dunelab/massibo/data/SET_6/MEAS_3/set6_meas3_44_5V_V2_5_G_CH0_4_G.npy"

# Rango de integración (en número de muestra)
# Ajusta estos valores según donde esté el pulso en tus waveforms
SAMPLE_START = 110
SAMPLE_END = 140

# Muestras para baseline (al inicio de la waveform)
BASELINE_SAMPLES = 40

# --- FUNCIONES ---

def load_waveforms_from_npy(filepath):
    """
    Carga waveforms desde un archivo .npy.
    """
    if not os.path.exists(filepath):
        print(f"Error: El archivo no existe: {filepath}")
        return None, None
    data = np.load(filepath)
    timestamps = data[:, 0].astype(np.uint64)
    waveforms = data[:, 1:].astype(np.int16)
    print(f"Cargadas {len(waveforms)} waveforms.")
    return timestamps, waveforms

def multi_gaussian_fit(charges, num_peaks=6):
    """
    Ajusta múltiples gaussianas al histograma de cargas.
    Copiado y adaptado de analisis_functions.py para ser robusto.
    """
    # 1. Crear histograma
    counts, bin_edges = np.histogram(charges, bins=200, range=(np.percentile(charges, 1), np.percentile(charges, 99)))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # 2. Definir función de ajuste
    def multi_gauss(x, *params):
        y = np.zeros_like(x, dtype=float)
        for i in range(0, len(params), 3):
            amp, mean, std = params[i], params[i+1], params[i+2]
            y += amp * norm.pdf(x, mean, std)
        return y

    # 3. Estimación inicial de picos
    counts_smoothed = gaussian_filter1d(counts, sigma=2.5)
    
    # Autocorrelación para encontrar periodo (ganancia) inicial
    autocorr = signal.correlate(counts_smoothed, counts_smoothed, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    ac_peaks, _ = signal.find_peaks(autocorr[10:], distance=10, prominence=np.max(autocorr)*0.05)
    
    if len(ac_peaks) > 0:
        estimated_gain = ac_peaks[0] + 10
        min_distance = int(estimated_gain * 0.6)
    else:
        min_distance = 15

    peak_indices, _ = signal.find_peaks(counts_smoothed, height=5, distance=min_distance, prominence=4.0)

    if len(peak_indices) < 2:
        print("Advertencia: No se encontraron suficientes picos automáticos. Intentando ajuste básico.")
        if len(peak_indices) == 0: return None
        # Fallback simple
        gain_est = 100 # Default dummy
    
    initial_params = []
    # Tomar hasta num_peaks
    # Tomar hasta num_peaks
    for i in peak_indices[:num_peaks]:
        # Ignorar picos iniciales muy negativos
        if bin_centers[i] < -50: continue
        initial_params.extend([counts[i], bin_centers[i], np.std(charges)/10])

    # 4. Ajuste
    try:
        popt, _ = curve_fit(multi_gauss, bin_centers, counts, p0=initial_params, maxfev=10000)
    except:
        print("El ajuste falló.")
        return None

    # 5. Extraer Ganancia
    # Filtramos means muy negativos (<-50) que puedan ser ruido
    raw_means = sorted([popt[i+1] for i in range(0, len(popt), 3)])
    means = [m for m in raw_means if m >= -50]
    
    if len(means) < 2: 
        print(f"Warning: Only {len(means)} valid peaks found > -50. Raw means: {raw_means}")
        return None
    
    gain = np.mean(np.diff(means))
    
    # 6. Plotear
    plt.figure(figsize=(10, 6))
    plt.hist(charges, bins=200, range=(np.percentile(charges, 1), np.percentile(charges, 99)), alpha=0.6, label='Data')
    plt.plot(bin_centers, multi_gauss(bin_centers, *popt), 'r-', label='Fit')
    for m in means:
        plt.axvline(m, color='k', linestyle='--', alpha=0.5)
    
    plt.title(f"Gain Analysis (Window [{SAMPLE_START}:{SAMPLE_END}])")
    plt.xlabel("Integrated Charge (ADU)")
    plt.ylabel("Counts")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return gain

def main():
    print(f"Propcesando archivo: {FILEPATH}")
    print(f"Ventana de integración manual: Sample {SAMPLE_START} a {SAMPLE_END}")
    
    timestamps, waveforms = load_waveforms_from_npy(FILEPATH)
    if waveforms is None: return

    charges = []
    processed_waveforms = []
    
    # Procesar waveforms
    for wf in waveforms:
        # Invertir (SiPM son negativos)
        wf_inv = -wf
        
        # Baseline
        baseline = np.mean(wf_inv[:BASELINE_SAMPLES])
        wf_corr = wf_inv - baseline
        
        # Integrar ventana fija
        # Asegurar límites
        start = max(0, SAMPLE_START)
        end = min(len(wf_corr), SAMPLE_END)
        
        charge = np.sum(wf_corr[start:end])
        
        # Filtro: Ignorar integrales < -100
        if charge >= -100:
            charges.append(charge)
            processed_waveforms.append(wf_corr)
            
    print(f"Total waveforms: {len(waveforms)}")
    print(f"Accepted waveforms: {len(charges)}")
    print(f"Rejected waveforms: {len(waveforms) - len(charges)}")
    if charges:
        print(f"Charge Range: [{min(charges):.2f}, {max(charges):.2f}]")

    # Calcular Ganancia
    gain = multi_gaussian_fit(charges)
    
    # Plot Waveforms
    if processed_waveforms:
        plt.figure(figsize=(10, 6))
        for wf in processed_waveforms:
            plt.plot(wf, color='blue', alpha=0.05) # Low alpha for superposition
            
        plt.axvspan(SAMPLE_START, SAMPLE_END, color='red', alpha=0.2, label='Integration Window')
        plt.title(f"Superimposed Waveforms ({len(processed_waveforms)} events)")
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude (ADU)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()

    if gain:
        print("\n" + "="*30)
        print(f"GANANCIA ESTIMADA: {gain:.2f} (ADU x muestras)/p.e.")
        print("="*30)
    else:
        print("No se pudo estimar la ganancia.")

if __name__ == "__main__":
    main()
