import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy.ndimage import gaussian_filter1d
import os

# --- Default Integration Window (Manual) ---
# Can be overridden if needed, but keeping fixed for now as requested
SAMPLE_START = 110
SAMPLE_END = 150
BASELINE_SAMPLES = 40

def load_waveforms_from_npy(filepath):
    """
    Carga waveforms desde un archivo .npy.
    """
    if not os.path.exists(filepath):
        # Silent error in automated mode, or log it
        return None, None
    data = np.load(filepath)
    timestamps = data[:, 0].astype(np.uint64)
    waveforms = data[:, 1:].astype(np.int16)
    return timestamps, waveforms

def multi_gaussian_fit(charges, num_peaks=6, plot=False):
    """
    Ajusta múltiples gaussianas al histograma de cargas.
    """
    if len(charges) < 10: return None

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
    counts_smoothed = gaussian_filter1d(counts, sigma=2.5) # User tuned value
    
    # Autocorrelación para encontrar periodo (ganancia) inicial
    autocorr = signal.correlate(counts_smoothed, counts_smoothed, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    ac_peaks, _ = signal.find_peaks(autocorr[10:], distance=10, prominence=np.max(autocorr)*0.05)
    
    if len(ac_peaks) > 0:
        estimated_gain = ac_peaks[0] + 10
        min_distance = int(estimated_gain * 0.6)
    else:
        min_distance = 15

    peak_indices, _ = signal.find_peaks(counts_smoothed, height=2, distance=min_distance, prominence=4.0) # User tuned value

    if len(peak_indices) < 2:
        return None
    
    initial_params = []
    # Tomar hasta num_peaks
    for i in peak_indices[:num_peaks]:
        initial_params.extend([counts[i], bin_centers[i], np.std(charges)/10])

    # 4. Ajuste
    try:
        popt, _ = curve_fit(multi_gauss, bin_centers, counts, p0=initial_params, maxfev=10000)
    except:
        return None

    # 5. Extraer Ganancia
    means = sorted([popt[i+1] for i in range(0, len(popt), 3)])
    if len(means) < 2: return None
    
    gain = np.mean(np.diff(means))
    
    if plot:
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

def analyze_gain_file(filepath):
    """
    Analiza un archivo individual y retorna la ganancia estimada.
    Diseñado para ser llamado desde DAQ_4G.py.
    """
    timestamps, waveforms = load_waveforms_from_npy(filepath)
    if waveforms is None: return None

    charges = []
    
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
        charges.append(charge)

    # Calcular Ganancia sin plotear
    gain = multi_gaussian_fit(charges, plot=False)
    
    return gain
