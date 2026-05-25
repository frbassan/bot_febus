import h5py
import numpy as np
import time
import datetime

# --- Simulation Settings ---
filename = "mock_febus_data_10k.h5"
num_points = 10000    # 10km of fiber with 1m resolution
num_measures = 1000   # 1000 acquisitions
pulse_width_ns = 10   # 10ns = 1m of resolution

# --- Signal-to-Noise Ratio (SNR) ---
# Smaller values = MORE noise (dirtier signal). Larger values = LESS noise.

# 1. Creating distance and time axes
distances = np.linspace(0, 10000, num_points)

# Start time: 01/01/2026 12:00 (standard Unix timestamp)
start_time_utc = datetime.datetime(2026, 1, 1, 12, 0, tzinfo=datetime.timezone.utc)
start_timestamp = start_time_utc.timestamp()
step_time = 60  # 1 trace each minute (60 seconds)
measure_duration = 2  # Each measurement lasts 2s

start_times = start_timestamp + np.arange(num_measures) * step_time
end_times = start_times + measure_duration

# ----------------------------------------------------
# 2. Generating Temperature (Base 25°C + Heavy noise)
# ----------------------------------------------------
base_temp = 25.0
noise_amplitude_temp = 5
temp_data = base_temp + np.random.normal(0, noise_amplitude_temp, (num_measures, num_points))

# Injecting 5 Elevated Temperature Zones (Hot Spots) with varying widths and intensities
hotspots = [
    {"start": 1000, "end": 1050, "intensity": -8.0},   # Snow
    {"start": 3200, "end": 3300, "intensity": 45.0},   # Critical leak
    {"start": 4200, "end": 4250, "intensity": 25.0},   # Fire in nearby vegetation
    {"start": 5500, "end": 5520, "intensity": 15.0},   # Hotspot at splice
    {"start": 7800, "end": 8000, "intensity": 5.0},    # Solar heating in exposed section
    {"start": 9100, "end": 9105, "intensity": 60.0}    # Extremely high point short-circuit
]

for hs in hotspots:
    # Adding the hotspot with static intensity across the whole time block
    temp_data[:, hs["start"]:hs["end"]] += hs["intensity"]

# ----------------------------------------------------
# 3. Generating Strain (Base 0 + Heavy noise)
# ----------------------------------------------------
base_strain = 0.0
noise_amplitude_strain = 100.0
strain_data = base_strain + np.random.normal(0, noise_amplitude_strain, (num_measures, num_points))

# Injecting 5 Elevated Strain Zones (Traction/Compression)
strain_spots = [
    {"start": 500, "end": 600, "intensity": 350.0},     # Soil displacement
    {"start": 2500, "end": 2550, "intensity": -150.0},  # Compression
    {"start": 4800, "end": 4850, "intensity": 800.0},   # Suddenly opening crack
    {"start": 6200, "end": 6300, "intensity": 400.0},   # Landslide tensioning cable
    {"start": 8500, "end": 8550, "intensity": -300.0}   # Strong compression
]

for ss in strain_spots:
    # Strain event with static intensity in space and time (rectangular block)
    strain_data[:, ss["start"]:ss["end"]] += ss["intensity"]


# ----------------------------------------------------
# 4. Creating the HDF5 file
# ----------------------------------------------------
with h5py.File(filename, 'w') as f:
    f.create_dataset("distances", data=distances)
    f.create_dataset("start_times", data=start_times)
    f.create_dataset("end_times", data=end_times)

    
    # Chunking enabled to optimize Slice reading in Streamlit for large files
    f.create_dataset("extractedTemperature", data=temp_data, chunks=(1, num_points), compression="gzip")
    f.create_dataset("extractedDeformation", data=strain_data, chunks=(1, num_points), compression="gzip")
    
    f.attrs['interrogator_model'] = "FEBUS G2-R"
    f.attrs['location'] = "TS Conductor Mega Test Site"
    f.attrs['pulse_width_ns'] = float(pulse_width_ns)
    f.attrs['noise_temp_amplitude_°C'] = float(noise_amplitude_temp)
    f.attrs['noise_strain_amplitude_µε'] = float(noise_amplitude_strain)

print(f"File '{filename}' (10,000 points) generated successfully!")
