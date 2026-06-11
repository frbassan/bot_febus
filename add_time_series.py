import re

# The new function code
new_func_code = """
def render_time_series_module(struct, section_num, section_title, y_dataset, distance_dataset="distances", ymin=None, ymax=None):
    from scipy.signal import butter, filtfilt, savgol_filter, medfilt
    import numpy as np
    import pandas as pd
    import h5py
    import matplotlib.pyplot as plt

    st.header(f"{section_num}. {section_title}")

    if y_dataset not in struct["datasets"]:
        st.warning(f"Dataset '{y_dataset}' not found in the file.")
        return
        
    y_shape = struct["datasets"][y_dataset]["shape"]
    if len(y_shape) < 2:
        st.warning("Time-series extraction requires a 2D dataset.")
        return
        
    num_traces = y_shape[0]
    num_distances = y_shape[1]

    with st.expander("X Axis Time Range", expanded=True):
        times_raw = None
        try:
            with h5py.File(st.session_state['file_path'], 'r') as f:
                if 'start_times' in f:
                    times_raw = f['start_times'][:]
        except: pass
        
        st.markdown(f"**Traces Available:** {num_traces}")
        use_slider = st.checkbox("Use Index Slider (Time)", value=True, key=f"t_slide_{section_num}")
        if use_slider:
            slice_range = st.slider("Select Trace Range:", 0, num_traces, (0, num_traces), key=f"idx_t_{section_num}")
        else:
            c1, c2 = st.columns(2)
            with c1: s_in = st.number_input("Start Trace", min_value=0, max_value=max(0, num_traces-1), value=0, key=f"t_s_{section_num}")
            with c2: e_in = st.number_input("End Trace", min_value=1, max_value=num_traces, value=num_traces, key=f"t_e_{section_num}")
            slice_range = (0, num_traces) if s_in >= e_in else (int(s_in), int(e_in))
            
        t_slice = slice(slice_range[0], slice_range[1])
        x_dim_size = slice_range[1] - slice_range[0]
        st.info(f"The plot will have {x_dim_size} time points on the X Axis.")

    with st.expander(f"Y Axis Distance Selection ({distance_dataset})", expanded=True):
        dist_mapping = None
        try:
            with h5py.File(st.session_state['file_path'], 'r') as f:
                if distance_dataset in f:
                    dist_mapping = f[distance_dataset][:]
        except: pass
        
        if dist_mapping is not None and len(dist_mapping) >= num_distances:
            min_dist, max_dist = float(dist_mapping[0]), float(dist_mapping[num_distances-1])
            sel_dist = st.slider("📏 Select Distance (m):", min_value=min_dist, max_value=max_dist, value=min_dist, step=0.1, key=f"d_sel_{section_num}")
            dist_idx = int((np.abs(dist_mapping[:num_distances] - sel_dist)).argmin())
            st.write(f"**Selected Index:** {dist_idx} (Exact Distance: {dist_mapping[dist_idx]:.2f} m)")
        else:
            dist_idx = st.number_input(f"📏 Select Distance Index (0 to {num_distances-1}):", min_value=0, max_value=num_distances-1, value=0, key=f"d_num_{section_num}")
            
        d_slice = slice(dist_idx, dist_idx + 1)
        
    y_slice = (t_slice, d_slice)

    st.subheader(f"Digital Filters (DSP) 〰️")
    tab_smooth, tab_freq, tab_spike = st.tabs(["Smoothing", "Frequency", "Spikes"])

    with tab_smooth:
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            use_moving_avg = st.checkbox("Enable Moving Average", value=False, key=f"ma_{section_num}")
            if use_moving_avg:
                ma_window = st.number_input("Window Size", min_value=2, max_value=max(2, x_dim_size//2), value=min(10, max(2, x_dim_size//10)), step=1, key=f"maw_{section_num}")
        with col_s2:
            use_savgol = st.checkbox("Enable Savitzky-Golay", value=False, key=f"sg_{section_num}")
            if use_savgol:
                sg_window = st.number_input("Window Size (Odd)", min_value=3, max_value=max(3, x_dim_size//2), value=min(11, max(3, x_dim_size//10)|1), step=2, key=f"sgw_{section_num}")
                if sg_window % 2 == 0: sg_window += 1
                sg_order = st.number_input("Polynomial Order", min_value=1, max_value=min(5, sg_window-1), value=2, key=f"sgo_{section_num}")

    with tab_freq:
        c_f1, c_f2 = st.columns(2)
        with c_f1:
            use_bw_low = st.checkbox("Low-Pass Filter", value=False, key=f"bwl_{section_num}")
            if use_bw_low:
                bw_low_order = st.number_input("Order (Low)", min_value=1, max_value=10, value=3, key=f"bwlo_{section_num}")
                bw_low_cutoff = st.slider("Normalized Cutoff", 0.01, 0.99, 0.1, 0.01, key=f"bwlc_{section_num}")
        with c_f2:
            use_bw_high = st.checkbox("High-Pass Filter", value=False, key=f"bwh_{section_num}")
            if use_bw_high:
                bw_high_order = st.number_input("Order (High)", min_value=1, max_value=10, value=3, key=f"bwho_{section_num}")
                bw_high_cutoff = st.slider("Normalized Cutoff", 0.01, 0.99, 0.02, 0.01, key=f"bwhc_{section_num}")

    with tab_spike:
        use_median = st.checkbox("Enable Median Filter", value=False, key=f"med_{section_num}")
        if use_median:
            med_kernel = st.number_input("Kernel Size (Odd)", min_value=3, max_value=99, value=3, step=2, key=f"medk_{section_num}")
            if med_kernel % 2 == 0: med_kernel += 1

    if st.button(f"Extract and Plot 2D Time-Series {y_dataset}", type="primary", key=f"btn_{section_num}"):
        try:
            with h5py.File(st.session_state['file_path'], 'r') as f:
                if times_raw is not None and len(times_raw) > 0:
                    data_x = times_raw[t_slice] - times_raw[0]
                    x_label = "Time (Seconds from start)"
                else:
                    data_x = np.arange(slice_range[0], slice_range[1])
                    x_label = "Trace Index"
                data_y = np.array(f[y_dataset][y_slice]).flatten()
    
            if len(data_x) != len(data_y):
                st.error(f"Dimension Error: X has {len(data_x)} and Y has {len(data_y)}.")
                return
        
            st.session_state[f'raw_x_{section_num}'] = data_x
            st.session_state[f'raw_y_{section_num}'] = data_y
            st.session_state[f'xlabel_{section_num}'] = x_label
            st.success("Successfully processed slice!")
        except Exception as e:
            st.error(f"HDF5 Slicing Error: {e}")
            return
            
    if f'raw_x_{section_num}' in st.session_state and f'raw_y_{section_num}' in st.session_state:
        data_x = st.session_state[f'raw_x_{section_num}']
        data_y = st.session_state[f'raw_y_{section_num}']
        x_label = st.session_state.get(f'xlabel_{section_num}', "Time")

        plots_to_make = []
        try:
            if use_moving_avg:
                ma = np.convolve(data_y, np.ones(ma_window)/ma_window, mode='same')
                plots_to_make.append((ma, 'Moving Avg', 'tab:orange'))
            if use_savgol:
                sg = savgol_filter(data_y, window_length=sg_window, polyorder=sg_order)
                plots_to_make.append((sg, 'Savitzky-Golay', 'tab:green'))
            if use_median:
                med = medfilt(data_y, kernel_size=med_kernel)
                plots_to_make.append((med, 'Median', 'tab:purple'))
            if use_bw_low:
                b, a = butter(bw_low_order, bw_low_cutoff, btype='low', analog=False)
                plots_to_make.append((filtfilt(b, a, data_y), 'Low-Pass', 'tab:red'))
            if use_bw_high:
                b, a = butter(bw_high_order, bw_high_cutoff, btype='high', analog=False)
                plots_to_make.append((filtfilt(b, a, data_y), 'High-Pass', 'tab:cyan'))
        except Exception as e:
            st.warning(f"Filter application warning: {e}")

        try:
            fig, ax = plt.subplots(figsize=(10, 4))
            
            MAX_POINTS = 1500
            if len(data_x) > MAX_POINTS:
                step = len(data_x) // MAX_POINTS
                plot_x = data_x[::step]
                plot_y = data_y[::step]
                downsample_msg = f" (Downsampled for plot: {len(plot_x)} pts)"
            else:
                plot_x = data_x
                plot_y = data_y
                downsample_msg = ""
                
            ax.plot(plot_x, plot_y, label='Raw Time-Series', color='tab:blue', alpha=0.6 if plots_to_make else 1.0, linewidth=1.5)
            
            for p_data, p_name, p_color in plots_to_make:
                if len(p_data) > MAX_POINTS:
                    p_data = p_data[::step]
                ax.plot(plot_x, p_data, label=f'Filter: {p_name}', color=p_color, linewidth=1.5)
                
            ax.set_title(f"{section_title}{downsample_msg}", fontweight='bold')
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_dataset)
            if ymin is not None and ymax is not None: ax.set_ylim([ymin, ymax])
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend(loc='best')
            fig.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Plotting Error: {e}")
"""

with open("combined_app.py", "r") as f:
    app_code = f.read()

# Append the new function after plot_3d_surface
# Find def plot_3d_surface and its end
plot_3d_end_idx = app_code.find("def plot_3d_surface")
# Find the next # --- UNIFIED SIDEBAR --- which is immediately after
sidebar_idx = app_code.find("# --- UNIFIED SIDEBAR ---")

app_code = app_code[:sidebar_idx] + new_func_code + "\n\n" + app_code[sidebar_idx:]

# Now update the tabs
old_tabs_code = '''    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "1. Metadata", 
        "2. Temp (2D)", 
        "3. Strain (2D)", 
        "4. Temp (3D)", 
        "5. Strain (3D)",
        "6. Intelligent Assistant"
    ])'''

new_tabs_code = '''    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "1. Metadata", 
        "2. Temp (2D)", 
        "3. Strain (2D)", 
        "4. Temp (2D-Time)",
        "5. Strain (2D-Time)",
        "6. Temp (3D)", 
        "7. Strain (3D)",
        "8. Intelligent Assistant"
    ])'''

app_code = app_code.replace(old_tabs_code, new_tabs_code)

# Shift existing tab usages
app_code = app_code.replace('with tab6:', 'with tab8:')
app_code = app_code.replace('with tab5:', 'with tab7:')
app_code = app_code.replace('with tab4:', 'with tab6:')

# Add new usages for tab4 and tab5
new_tabs_impl = '''    with tab4:
        st.info("💡 **Temperature Evolution (Time-Series)**: Plots the temperature changes at a specific meter over the entire duration of the monitoring.")
        render_time_series_module(struct, 4, "Temperature Time-Series", "temp_data")
        
    with tab5:
        st.info("💡 **Strain Evolution (Time-Series)**: Plots the strain changes at a specific meter over the entire duration of the monitoring.")
        render_time_series_module(struct, 5, "Strain Time-Series", "strain_data")
        
    with tab6:'''

app_code = app_code.replace('    with tab6:', new_tabs_impl)

with open("combined_app.py", "w") as f:
    f.write(app_code)

print("Updates applied to combined_app.py!")
