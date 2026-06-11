import h5py
import numpy as np
import time
import os

# =========================================================================
# GENERAL FIBER AND FILE CONFIGURATION
# =========================================================================
OUTPUT_FILENAME = "Simulated_FiberTest_TSB_2km_0.1m_res_noise_1_5.h5"

CONFIG = {
    "fiber_length_m": 2000.0,          # 2 km
    "sampling_resolution_m": 0.1,      # Distance between points (0.1m)
    "measurement_count": 60,           # Number of temporal traces
    "measurement_interval_s": 60,      # 1 minute between each trace
    "base_temp_c": 25.0,               # Resting temperature
    "base_strain_ue": 0.0,             # Resting strain (microstrain)
    "base_brillouin_mhz": 10850.0,     # Central reference frequency
}

# =========================================================================
# TEMPERATURE EVENT CONFIGURATION (EDIT HERE)
# =========================================================================
# Evolution Types: 
#   "sinusoidal" (rises and falls), 
#   "linear" (constant growth), 
#   "peak" (occurs at a specific minute)

TEMPERATURE_EVENTS = [
    # { "meter": position, "width": m, "amplitude": ºC, "evolution": type, "param": optional }
    {"meter": 1200, "width": 3,   "amplitude": 80.0,  "evolution": "sinusoidal", "param": None},
    {"meter": 1500, "width": 5,   "amplitude": 45.0,  "evolution": "sinusoidal", "param": None},
    {"meter": 1200, "width": 1.5, "amplitude": 100.0, "evolution": "peak",       "param": 40}, # Peak at minute 40
]

# =========================================================================
# STRAIN EVENT CONFIGURATION (EDIT HERE)
# =========================================================================
STRAIN_EVENTS = [
    {"meter": 1400, "width": 4,   "amplitude": 1300.0,  "evolution": "sinusoidal", "param": None},
    {"meter": 1800, "width": 8,   "amplitude": -1200.0, "evolution": "linear",     "param": None}, # Compression
    {"meter": 1100, "width": 3.5, "amplitude": 800.0,   "evolution": "wave",       "param": 2}    # 2 vibration cycles
]

# =========================================================================
# PHYSICAL PROCESSING (NO NEED TO EDIT BELOW)
# =========================================================================

if os.path.exists(OUTPUT_FILENAME): os.remove(OUTPUT_FILENAME)

# Axis Preparation
n_distances = int(CONFIG["fiber_length_m"] / CONFIG["sampling_resolution_m"])
distances = np.linspace(0, CONFIG["fiber_length_m"], n_distances, endpoint=False)
base_timestamp = time.time()

# Real Febus Sensitivities
temp_sens = 1.07    # MHz / ºC
strain_sens = 0.046 # MHz / ue

# Matrix Allocation
temp_data = np.zeros((CONFIG["measurement_count"], n_distances), dtype=np.float32)
strain_data = np.zeros((CONFIG["measurement_count"], n_distances), dtype=np.float32)
bsl_data = np.zeros((CONFIG["measurement_count"], n_distances), dtype=np.float32)
start_times = np.zeros(CONFIG["measurement_count"], dtype=np.float64)
end_times = np.zeros(CONFIG["measurement_count"], dtype=np.float64)
hw_temperatures = np.zeros(CONFIG["measurement_count"], dtype=np.float32)

print(f"Simulating Physics: {n_distances} points x {CONFIG['measurement_count']} traces...")

for t in range(CONFIG["measurement_count"]):
    start_times[t] = base_timestamp + (t * CONFIG["measurement_interval_s"])
    end_times[t] = start_times[t] + CONFIG["measurement_interval_s"]
    hw_temperatures[t] = 24.58 + np.sin(t * 0.1) * 0.5 # Internal hardware temp
    
    # Base arrays with Gaussian noise
    t_arr = np.full(n_distances, CONFIG["base_temp_c"]) + np.random.normal(0, 1, n_distances)
    s_arr = np.full(n_distances, CONFIG["base_strain_ue"]) + np.random.normal(0, 5.0, n_distances)

    progress = t / (CONFIG["measurement_count"] - 1) if CONFIG["measurement_count"] > 1 else 1
    sin_factor = np.sin(progress * np.pi)
    
    # Process Temperature Events
    for ev in TEMPERATURE_EVENTS:
        factor = 1.0
        if ev["evolution"] == "sinusoidal": factor = sin_factor
        elif ev["evolution"] == "linear": factor = progress
        elif ev["evolution"] == "peak":   factor = np.exp(-0.5 * ((t - ev["param"]) / 5)**2)
        
        # Gaussian Kernel for the Hotspot
        idx_center = int(ev["meter"] / CONFIG["sampling_resolution_m"])
        std_pts = ev["width"] / CONFIG["sampling_resolution_m"]
        t_arr += ev["amplitude"] * factor * np.exp(-0.5 * ((np.arange(n_distances) - idx_center) / std_pts)**2)

    # Process Strain Events
    for ev in STRAIN_EVENTS:
        factor = 1.0
        if ev["evolution"] == "sinusoidal": factor = sin_factor
        elif ev["evolution"] == "linear": factor = progress
        elif ev["evolution"] == "wave":    factor = np.sin(progress * 2 * np.pi * ev["param"])
        
        idx_center = int(ev["meter"] / CONFIG["sampling_resolution_m"])
        std_pts = ev["width"] / CONFIG["sampling_resolution_m"]
        s_arr += ev["amplitude"] * factor * np.exp(-0.5 * ((np.arange(n_distances) - idx_center) / std_pts)**2)

    # Calculate Brillouin Frequency Shift (BSL)
    bsl_arr = CONFIG["base_brillouin_mhz"] + \
              (t_arr - CONFIG["base_temp_c"]) * temp_sens + \
              (s_arr - CONFIG["base_strain_ue"]) * strain_sens
    
    temp_data[t, :] = t_arr
    strain_data[t, :] = s_arr
    bsl_data[t, :] = bsl_arr

# =========================================================================
# HDF5 EXPORT (WITH ALL ORIGINAL ROOT ATTRIBUTES)
# =========================================================================
print(f"Writing file with full metadata: {OUTPUT_FILENAME}...")
with h5py.File(OUTPUT_FILENAME, 'w') as f:
    
    # 1. ROOT Attributes (Official FEBUS Metadata)
    f.attrs['acq_res'] = np.array([20], dtype=np.int32)
    f.attrs['ampliPower'] = np.array([20], dtype=np.int32)
    f.attrs['average'] = np.array([1201], dtype=np.int32)
    f.attrs['channel'] = np.array([1], dtype=np.int32)
    f.attrs['end_time'] = np.array([end_times[-1]], dtype=np.float64)
    f.attrs['fiberBreak'] = np.array([int(CONFIG["fiber_length_m"])], dtype=np.int32)
    f.attrs['fiberFrom'] = np.array([0], dtype=np.int32)
    f.attrs['fiberLength'] = np.array([int(CONFIG["fiber_length_m"])], dtype=np.int32)
    f.attrs['fiberTo'] = np.array([int(CONFIG["fiber_length_m"])], dtype=np.int32)
    f.attrs['formatVersion'] = np.array([1], dtype=np.int32)
    f.attrs['freq_fiber'] = np.array([CONFIG["base_brillouin_mhz"]], dtype=np.int32)
    f.attrs['freq_offset'] = np.array([330.954], dtype=np.float32)
    f.attrs['freq_offset_abs'] = np.array([11175.0], dtype=np.float32)
    f.attrs['freq_ref'] = np.array([10975], dtype=np.int32)
    f.attrs['freq_step'] = np.array([-1.953125], dtype=np.float32)
    f.attrs['mode'] = np.array([2], dtype=np.int32)
    f.attrs['sampling_resolution'] = np.array([CONFIG["sampling_resolution_m"]], dtype=np.float32)
    f.attrs['signal_size'] = np.array([1024], dtype=np.int32)
    f.attrs['spatial_resolution'] = np.array([5], dtype=np.float32)
    f.attrs['start_time'] = np.array([start_times[0]], dtype=np.float64)
    f.attrs['strain_amp_sensitivity'] = np.array([-0.00082], dtype=np.float32)
    f.attrs['strain_freq_sensitivity'] = np.array([strain_sens], dtype=np.float32)
    f.attrs['temp_amp_sensitivity'] = np.array([0.003], dtype=np.float32)
    f.attrs['temp_freq_sensitivity'] = np.array([temp_sens], dtype=np.float32)
    f.attrs['temperature'] = np.array([24.58], dtype=np.float32)
    f.attrs['zoneCount'] = np.array([0], dtype=np.int32)
    f.attrs['zones'] = b''
    
    # 2. Official Datasets
    f.create_dataset('distances', data=distances, dtype='float64')
    f.create_dataset('start_times', data=start_times, dtype='float64')
    f.create_dataset('end_times', data=end_times, dtype='float64')
    f.create_dataset('temperatures', data=hw_temperatures, dtype='float32')
    f.create_dataset('temp_data', data=temp_data, dtype='float32')
    f.create_dataset('strain_data', data=strain_data, dtype='float32')
    f.create_dataset('bsl_data', data=bsl_data, dtype='float32')
    
print(">>> Success! Full TSB mock file created with synced metadata and physics.")