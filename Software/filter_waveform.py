import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os

# =============================================================================
# 1. CONFIGURACIÓN DEL USUARIO (CONSTANTES EXPORTABLES)
#    Estas constantes pueden ser importadas por otros scripts.
# =============================================================================

# --- ARCHIVO DE DATOS ---
# FILENAME for standalone testing (will be defined in main_filter_script)

# --- OPCIONES DE ALGORITMO (TRUE / FALSE) ---
DEFAULT_ENABLE_PHYSICS_DISCRIMINATION = True 
DEFAULT_ENABLE_ITERATIVE_REFINEMENT = True

# --- PARÁMETROS DE AJUSTE ---
DEFAULT_THRESHOLD_CORR = 0.50      # Umbral mínimo de correlación (0.0 a 1.0)
TIME_PER_SAMPLE_NS = 16            # Nanosegundos por muestra (ajusta según tu digitizer)
MIN_PULSE_WIDTH_NS = 120            # Ancho mínimo para ser considerado SiPM (descarta fugas SSR)

# --- PARÁMETROS PARA RUIDO SINTÉTICO (NUEVO) ---
NOISE_WIDTH_NS = 50                # Anchura aproximada del ruido bipolar (muy rápido)

# --- PARÁMETROS PARA TEMPLATE SiPM SINTÉTICO ---
SIPM_RISE_TIME_NS = 20             # Tiempo de subida del pulso SiPM ideal
SIPM_FALL_TIME_NS = 600            # Tiempo de bajada del pulso SiPM ideal

# --- ÍNDICES PARA ENTRENAMIENTO MANUAL (OBLIGATORIO AL INICIO) ---
# Mira tus datos crudos y elige índices de ejemplos claros
IDX_SIPM_MANUAL = [0, 1, 2, 3, 4]      # Índices de señales que SABES que son SiPM

# Para compatibilidad con la versión anterior de analisis_sipm.py, aunque ya no se usa
# IDX_LEAK_MANUAL = []


# =============================================================================
# 2. FUNCIONES DE CARGA Y UTILIDADES
# =============================================================================

def load_sipm_data(filepath):
    """Carga datos separando timestamps y waveforms."""
    if not os.path.exists(filepath): # This function is for standalone use, so it can generate dummy
        # Generar datos dummy si no existe el archivo para probar el script
        print("¡AVISO! Archivo no encontrado. Generando datos simulados para demostración...")
        return generate_dummy_data()
        
    raw_data = np.load(filepath)
    timestamps = raw_data[:, 0].astype(np.uint64)
    waveforms = raw_data[:, 1:].astype(np.float32)
    return timestamps, waveforms

def generate_dummy_data(n=200, waveform_length=251):
    """Generador de datos falsos para pruebas."""
    # Use waveform_length from DaphneChannel if available, otherwise default to 251
    # This is a placeholder, in a real scenario, DaphneChannel.WAVEFORM_LENGTH would be imported
    # For now, hardcode 251 as it's common in the context files.
    t = np.linspace(0, waveform_length * TIME_PER_SAMPLE_NS, waveform_length)
    waves = []
    # Generar SiPM (lentos) y Fugas (rápidas)
    for i in range(n):
        if i < 100: # SiPM
            w = -1 * np.exp(-(t-100)/50) * (t > 100)
            w += np.random.normal(0, 0.05, len(t))
        else: # Leakage
            w = -1 * np.exp(-(t-100)/5) * (t > 100)
            w += np.random.normal(0, 0.05, len(t))
        waves.append(w) # Original dummy data had issues with SIPM_RISE/FALL_TIME_NS
    # Scale to typical ADC values (e.g., 8000 baseline, 1000 ADU pulse)
    waves_scaled = (np.array(waves) * 1000) + 8000
    return np.arange(n).astype(np.uint64), waves_scaled.astype(np.float32)

# =============================================================================
# 3. LÓGICA CORE (TEMPLATES Y FÍSICA)
# =============================================================================

def create_template(waveforms_subset):
    """Genera una forma de onda promedio alineada."""
    if len(waveforms_subset) == 0: return None # Return None if no waveforms
    aligned_waves = []
    
    for w in waveforms_subset:
        peak_idx = np.argmin(w) # Asume pulsos negativos
        shift = int(len(w)/2) - peak_idx
        aligned_waves.append(np.roll(w, shift))
            
    template = np.mean(aligned_waves, axis=0)
    # Normalización Min-Max para comparar solo forma
    rng = np.max(template) - np.min(template)
    if rng == 0: return template
    return (template - np.min(template)) / rng

def generate_synthetic_sipm_template(length_samples, rise_ns, fall_ns, time_per_sample):
    """
    Genera la forma ideal de un SiPM: Doble Exponencial.
    Independiente de los datos del archivo (evita contaminación por ruido).
    """
    t = np.arange(length_samples) * time_per_sample
    # Centramos el pulso
    t_start = length_samples * time_per_sample * 0.2 # Start pulse earlier than center
    
    mask = t > t_start
    dt = t[mask] - t_start
    
    tau_rise = max(1e-9, rise_ns)
    tau_fall = max(1e-9, fall_ns)
    
    val = np.zeros_like(t, dtype=np.float32)
    # Forma analítica: I(t) = exp(-t/fall) - exp(-t/rise)
    # Multiplicamos por -1 porque tus pulsos son negativos
    val[mask] = -1 * (np.exp(-dt/tau_fall) - np.exp(-dt/tau_rise))
    
    rng = np.max(val) - np.min(val)
    if rng == 0: return val
    return (val - np.min(val)) / rng

def generate_synthetic_bipolar_template(length_samples, width_ns, time_per_sample):
    """
    NUEVO: Genera una forma de onda BIPOLAR analítica (Derivada de Gaussiana).
    Simula la descarga capacitiva rápida: Pico negativo fuerte + Rebote positivo.
    """
    # Eje temporal centrado
    t = np.arange(length_samples) * time_per_sample
    center_time = length_samples * time_per_sample / 2
    
    # Sigma controla el ancho del pulso
    sigma = width_ns / 2.355  # Relación FWHM a Sigma
    
    # Fórmula: Primera derivada de Gaussiana (forma bipolar perfecta)
    # El signo menos inicial es para que el primer pico sea NEGATIVO (tu caso)
    template = (t - center_time) * np.exp(-((t - center_time)**2) / (2 * sigma**2))
    
    # Normalizar (Min-Max)
    rng = np.max(template) - np.min(template)
    if rng == 0: return np.zeros(length_samples)
    norm_template = (template - np.min(template)) / rng
    
    return norm_template

def check_pulse_width(waveform, min_width_ns, time_per_sample):
    """
    DISCRIMINADOR FÍSICO: Mide el ancho a media altura (FWHM).
    Las fugas del multiplexor SSR suelen ser muy estrechas (<50ns).
    Las señales SiPM tienen cola de recuperación (>100ns).
    """
    peak_idx = np.argmin(waveform)
    peak_val = waveform[peak_idx]
    baseline = np.median(waveform[:15]) # Línea base robusta
    # If the signal is too small (thermal noise), reject by minimum amplitude
    amplitude = baseline - peak_val
    if amplitude < 0.001: return False # Señal plana/muerta
    
    half_max = peak_val + (amplitude * 0.5)
    
    # Buscar cruces por la izquierda y derecha
    left = peak_idx
    while left > 0 and waveform[left] < half_max: left -= 1
        
    right = peak_idx
    while right < len(waveform)-1 and waveform[right] < half_max: right += 1
        
    width_samples = right - left # This is the FWHM in samples
    width_ns = width_samples * time_per_sample
    
    return width_ns > min_width_ns

def classify_waveform(wave, tmpl_sipm, tmpl_leak, threshold, use_physics, min_pulse_width_ns, time_per_sample):
    """Decide si una onda es SiPM o Ruido basándose en correlación y física."""
    
    # 1. Preparar onda (Normalizar y Alinear)
    rng = np.max(wave) - np.min(wave)
    if rng == 0: return 'rejected'
    norm_wave = (wave - np.min(wave)) / rng # Normalize to 0-1 range
    
    peak_idx = np.argmin(wave)
    shift = int(len(wave)/2) - peak_idx
    aligned_wave = np.roll(norm_wave, shift)
    
    # 2. Correlación Matemática
    # Aseguramos que los templates tengan el mismo tamaño (resample si es necesario)
    # Ensure templates are not None and have correct length
    if tmpl_sipm is None or tmpl_leak is None: return 'rejected'
    if len(tmpl_sipm) != len(aligned_wave): tmpl_sipm = signal.resample(tmpl_sipm, len(aligned_wave))
    if len(tmpl_leak) != len(aligned_wave): tmpl_leak = signal.resample(tmpl_leak, len(aligned_wave))
    
    corr_sipm = np.corrcoef(aligned_wave, tmpl_sipm)[0, 1]
    corr_leak = np.corrcoef(aligned_wave, tmpl_leak)[0, 1]
    
    # Criterio base: Se parece más al SiPM que al ruido Y supera umbral
    is_mathematically_sipm = (corr_sipm > threshold) and (corr_sipm > corr_leak)
    
    # 3. Filtro Físico (Si está activado)
    if use_physics: # Check pulse width
        is_physically_valid = check_pulse_width(wave, min_pulse_width_ns, time_per_sample)
        
        # Combined logic:
        if is_mathematically_sipm and is_physically_valid:
            return 'accepted'
        # "Rescate": Si la correlación es dudosa pero físicamente es perfecta, aceptar
        elif (corr_sipm > threshold - 0.1) and is_physically_valid and (corr_sipm > corr_leak):
            return 'accepted'
        else:
            return 'rejected'
    else:
        # Solo criterio matemático
        return 'accepted' if is_mathematically_sipm else 'rejected'

# =============================================================================
# 4. FUNCIÓN PRINCIPAL DE FILTRADO (EXPORTABLE)
# =============================================================================

def apply_waveform_filter(
    raw_timestamps: np.ndarray,
    raw_waveforms: np.ndarray,
    idx_sipm_manual: list,
    threshold_corr: float = DEFAULT_THRESHOLD_CORR,
    enable_iterative_refinement: bool = DEFAULT_ENABLE_ITERATIVE_REFINEMENT,
    enable_physics_discrimination: bool = DEFAULT_ENABLE_PHYSICS_DISCRIMINATION,
    time_per_sample_ns: float = TIME_PER_SAMPLE_NS,
    min_pulse_width_ns: float = MIN_PULSE_WIDTH_NS,
    noise_width_ns: float = NOISE_WIDTH_NS,
    sipm_rise_time_ns: float = SIPM_RISE_TIME_NS,
    sipm_fall_time_ns: float = SIPM_FALL_TIME_NS
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Aplica un filtro de correlación y discriminación física a las waveforms.
    Retorna los timestamps y waveforms que fueron aceptados y rechazados,
    así como los templates utilizados.
    """
    print("--- INICIANDO PROCESO DE FILTRADO DE WAVEFORMS ---")
    if len(raw_waveforms) == 0:
        print("Advertencia: No hay waveforms para filtrar.")
        return np.array([]), np.array([]), np.array([]), np.array([]), None, None

    n_samples = raw_waveforms.shape[1]
    print(f"Total eventos recibidos: {len(raw_waveforms)}, Longitud de waveform: {n_samples}")

    # A) Generar Templates (SiPM real, Ruido sintético)
    print("Generando templates...")
    
    # 1. Template SiPM: Basado en datos reales (índices manuales)
    # Asegurarse de que los índices manuales son válidos
    valid_sipm_idx = [i for i in idx_sipm_manual if i < len(raw_waveforms)]
    if not valid_sipm_idx:
        print("Advertencia: No se proporcionaron índices manuales válidos para el template SiPM. Usando template sintético.")
        tmpl_sipm = generate_synthetic_sipm_template(n_samples, sipm_rise_time_ns, sipm_fall_time_ns, time_per_sample_ns)
    else:
        tmpl_sipm = create_template(raw_waveforms[valid_sipm_idx])
        if tmpl_sipm is None: # Fallback if create_template returns None
             print("Advertencia: No se pudo crear el template SiPM a partir de los índices manuales. Usando template sintético.")
             tmpl_sipm = generate_synthetic_sipm_template(n_samples, sipm_rise_time_ns, sipm_fall_time_ns, time_per_sample_ns)

    # 2. Template Ruido: GENERADO SINTÉTICAMENTE (Bipolar)
    print("-> Generando template de ruido sintético bipolar...")
    tmpl_leak = generate_synthetic_bipolar_template(
        length_samples=n_samples,
        width_ns=noise_width_ns,
        time_per_sample=time_per_sample_ns
    )

    # C) Primera Pasada de Clasificación
    print(f"Clasificando (Física={enable_physics_discrimination})...")
    accepted_indices_p1 = []
    rejected_indices_p1 = []
    
    for i, w in enumerate(raw_waveforms):
        res = classify_waveform(w, tmpl_sipm, tmpl_leak, threshold_corr, 
                                enable_physics_discrimination, min_pulse_width_ns, time_per_sample_ns)
        if res == 'accepted':
            accepted_indices_p1.append(i)
        else:
            rejected_indices_p1.append(i)
            
    print(f"Pasada 1: {len(accepted_indices_p1)} aceptadas, {len(rejected_indices_p1)} rechazadas.")

    # D) Refinamiento Iterativo (Opcional)
    final_accepted_indices = accepted_indices_p1
    final_rejected_indices = rejected_indices_p1
    
    if enable_iterative_refinement and len(accepted_indices_p1) > 10:
        print("Refinando template (Iterative Refinement)...")
        # Crear Super-Template con las aceptadas
        super_tmpl_sipm = create_template(raw_waveforms[accepted_indices_p1])
        
        if super_tmpl_sipm is not None:
            # Segunda clasificación más estricta/precisa
            refined_accepted = []
            refined_rejected = []
            for i, w in enumerate(raw_waveforms):
                # Nota: Usamos el super template, pero mantenemos el template de ruido sintético
                res = classify_waveform(w, super_tmpl_sipm, tmpl_leak, threshold_corr, 
                                        enable_physics_discrimination, min_pulse_width_ns, time_per_sample_ns)
                if res == 'accepted':
                    refined_accepted.append(i)
                else:
                    refined_rejected.append(i)
            
            final_accepted_indices = refined_accepted
            final_rejected_indices = refined_rejected
            print(f"Pasada 2 (Refinada): {len(final_accepted_indices)} aceptadas, {len(final_rejected_indices)} rechazadas.")
        else:
            print("Advertencia: No se pudo crear el super-template. Saltando refinamiento.")

    # E) Separar datos finales
    accepted_timestamps = raw_timestamps[final_accepted_indices]
    accepted_waveforms = raw_waveforms[final_accepted_indices]
    rejected_timestamps = raw_timestamps[final_rejected_indices]
    rejected_waveforms = raw_waveforms[final_rejected_indices]
    
    print(f"Filtrado completado. {len(accepted_waveforms)} waveforms aceptadas.")
    return accepted_timestamps, accepted_waveforms, rejected_timestamps, rejected_waveforms, tmpl_sipm, tmpl_leak

# =============================================================================
# 5. VISUALIZACIÓN (Para uso standalone del filtro)
# =============================================================================

def plot_filter_results(raw_waveforms, accepted_waveforms, rejected_waveforms, tmpl_sipm, tmpl_leak,
                        threshold_corr, enable_physics_discrimination, enable_iterative_refinement,
                        min_pulse_width_ns, noise_width_ns):
    """
    Función para visualizar los resultados del filtro.
    """
    if len(raw_waveforms) > 0:
        wave_len = raw_waveforms.shape[1]
    elif tmpl_sipm is not None:
        wave_len = len(tmpl_sipm)
    else:
        print("No hay datos para plotear.")
        return
        
    x = np.arange(wave_len)
    
    # Ajustar templates para plotear si difieren en tamaño
    if tmpl_sipm is not None and len(tmpl_sipm) != wave_len:
        tmpl_sipm = signal.resample(tmpl_sipm, wave_len)
    if tmpl_leak is not None and len(tmpl_leak) != wave_len:
        tmpl_leak = signal.resample(tmpl_leak, wave_len)
    
    fig, ax = plt.subplots(2, 2, figsize=(14, 8))
    
    # 1. Templates
    if tmpl_sipm is not None:
        ax[0,0].plot(x, tmpl_sipm, 'b', label='Template SiPM')
    if tmpl_leak is not None:
        ax[0,0].plot(x, tmpl_leak, 'r--', label='Template Ruido')
    ax[0,0].set_title("Templates Utilizados")
    ax[0,0].legend()
    ax[0,0].grid(True, alpha=0.3)
    
    # 2. Aceptadas
    ax[0,1].set_title(f"Aceptadas ({len(accepted_waveforms)})")
    if len(accepted_waveforms) > 0:
        # Plotear un subconjunto para no saturar memoria
        step = max(1, len(accepted_waveforms)//200)
        for w in accepted_waveforms[::step]:
            ax[0,1].plot(x, w, color='blue', alpha=0.1)
        ax[0,1].plot(x, np.mean(accepted_waveforms, axis=0), color='cyan', linewidth=2, label='Promedio')
    ax[0,1].grid(True, alpha=0.3)

    # 3. Rechazadas
    ax[1,1].set_title(f"Rechazadas ({len(rejected_waveforms)})")
    if len(rejected_waveforms) > 0:
        step = max(1, len(rejected_waveforms)//200)
        for w in rejected_waveforms[::step]:
            ax[1,1].plot(x, w, color='red', alpha=0.1)
        ax[1,1].plot(x, np.mean(rejected_waveforms, axis=0), color='orange', linewidth=2, label='Promedio')
    ax[1,1].grid(True, alpha=0.3)
    
    # 4. Texto informativo
    info_text = ( # Use the passed parameters for info text
        f"Filtro Físico (Ancho): {enable_physics_discrimination}\n"
        f"Refinamiento: {enable_iterative_refinement}\n"
        f"Umbral Corr: {threshold_corr}\n"
        f"Min Width: {min_pulse_width_ns}ns\n"
        f"Ruido Sintético: {noise_width_ns}ns"
    )
    ax[1,0].text(0.1, 0.5, info_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    ax[1,0].axis('off')

    plt.tight_layout()
    plt.show()
    
def main_filter_script():
    """
    Función principal para ejecutar el filtro como script independiente.
    """
    # FILENAME for standalone testing
    FILENAME = "/home/cbenitez/Desktop/setpre/set5_meas2_45_8_v2_2_/set5_meas2_45_8_v2_2_W_CH1_5_W.npy" 

    print("--- EJECUTANDO FILTRO DE WAVEFORMS (MODO STANDALONE) ---")
    raw_timestamps, raw_waveforms = load_sipm_data(FILENAME) # Use the original load_sipm_data
    
    accepted_ts, accepted_wf, rejected_ts, rejected_wf, tmpl_sipm, tmpl_leak = apply_waveform_filter(
        raw_timestamps, raw_waveforms, IDX_SIPM_MANUAL,
        DEFAULT_THRESHOLD_CORR, DEFAULT_ENABLE_ITERATIVE_REFINEMENT, DEFAULT_ENABLE_PHYSICS_DISCRIMINATION,
        TIME_PER_SAMPLE_NS, MIN_PULSE_WIDTH_NS, NOISE_WIDTH_NS, SIPM_RISE_TIME_NS, SIPM_FALL_TIME_NS
    )
    
    plot_filter_results(raw_waveforms, accepted_wf, rejected_wf, tmpl_sipm, tmpl_leak,
                        DEFAULT_THRESHOLD_CORR, DEFAULT_ENABLE_PHYSICS_DISCRIMINATION, DEFAULT_ENABLE_ITERATIVE_REFINEMENT,
                        MIN_PULSE_WIDTH_NS, NOISE_WIDTH_NS)

if __name__ == "__main__":
    main_filter_script()