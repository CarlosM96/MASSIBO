import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Load the file
filepath = "/home/dunelab/massibo/data/SET_5/MEAS_3/set5_meas3_48V_V2_5_W_CH0_12_W.npy"
try:
    data = np.load(filepath)
    timestamps = data[:, 0]
    waveforms = data[:, 1:]
    print(f"Loaded {waveforms.shape[0]} waveforms. Length: {waveforms.shape[1]}")
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

# Parameters from analysis
BASELINE_SAMPLES = 40
PEAK_HEIGHT_THRESHOLD = 3
PROMINENCE = 2.0
WIDTH = 1.0
DISTANCE = 5

# Plot 5 random waveforms
indices = np.random.choice(len(waveforms), 5, replace=False)

for i in indices:
    wf = waveforms[i]
    
    # Process exactly as in finding peaks
    inverted_wf = -wf
    baseline = np.mean(inverted_wf[:BASELINE_SAMPLES])
    corrected_wf = inverted_wf - baseline
    
    # Run find_peaks
    peaks, properties = signal.find_peaks(corrected_wf, height=PEAK_HEIGHT_THRESHOLD, distance=DISTANCE, prominence=PROMINENCE, width=WIDTH)
    
    print(f"Running on WF {i}: Min={np.min(wf)}, Max={np.max(wf)}, Baseline={baseline:.2f}")
    if len(peaks) > 0:
        print(f" -> Found {len(peaks)} peaks at {peaks} with heights {properties['peak_heights']}")
    else:
        print(f" -> No peaks found. Max corrected value: {np.max(corrected_wf):.2f}")

    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(corrected_wf, label='Corrected Waveform (-WF - Baseline)')
    plt.axhline(PEAK_HEIGHT_THRESHOLD, color='r', linestyle='--', label='Threshold')
    if len(peaks) > 0:
        plt.plot(peaks, corrected_wf[peaks], "x", color='k', label='Peaks')
    plt.title(f"Waveform {i} Debug - Max Amp: {np.max(corrected_wf):.2f}")
    plt.legend()
    plt.grid(True)
    plt.show()
