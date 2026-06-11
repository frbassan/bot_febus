import streamlit as st
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os
import sys
from pathlib import Path
from PIL import Image

def get_asset_path(filename):
    if getattr(sys, 'frozen', False):
        return Path(sys._MEIPASS) / filename
    return Path(__file__).parent.absolute() / filename

st.set_page_config(page_title="HDF5 Data Viewer", layout="wide", page_icon=get_asset_path("logo.png"))

c1, c2 = st.columns([1, 8])
with c1:
    try:
        st.image(get_asset_path("logo.png"), use_container_width=True)
    except:
        pass
with c2:
    st.title("HDF5 Data Viewer 📊")
    st.markdown("Explore, slice, plot, and export HDF5 data directly from your browser.")

# ---------------------------------------------------------
# Sidebar: File Loading (Versão Corrigida para Web/Cloud)
# ---------------------------------------------------------
st.sidebar.header("1. Load File")

# Em vez de procurar um caminho no seu PC, o usuário faz o upload do arquivo
uploaded_file = st.sidebar.file_uploader("Upload HDF5 File", type=['h5', 'hdf5'])

if uploaded_file is not None:
    # O Streamlit salva o arquivo temporariamente na memória/disco do servidor
    # Precisamos salvar esse arquivo temporário para o h5py conseguir ler o caminho
    temp_path = Path("temp_uploaded_file.h5")
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.session_state['file_path'] = str(temp_path)
    st.sidebar.success("File uploaded successfully!")
else:
    # Caso queira manter um arquivo padrão (ex: se ele estiver no seu GitHub)
    default_file = "Simulated_FiberTest_TSB.h5"
    if os.path.exists(default_file):
        st.session_state['file_path'] = default_file
        st.sidebar.info("Using default simulated file.")
    else:
        st.sidebar.warning("Please upload a .h5 file to begin.")

# A partir daqui, o resto do seu código usa st.session_state['file_path']
if 'file_path' in st.session_state:
    file_path = st.session_state['file_path']
    # Segue a lógica do get_h5_structure...

def get_h5_structure(filepath):
    """Reads the H5 file structure and returns groups, datasets, and attributes."""
    from typing import Any
    structure: dict[str, Any] = {"datasets": {}, "groups": [], "attributes": {}}
    
    if not os.path.exists(filepath):
        return None, f"File not found: {filepath}"
        
    try:
        def clean_attrs(attrs):
            cleaned = {}
            for k, v in attrs.items():
                if isinstance(v, (np.ndarray, np.generic)):
                    if v.size == 1:
                        val = v.item()
                        cleaned[k] = val.decode('utf-8', 'ignore') if isinstance(val, bytes) else val
                    else:
                        val = v.tolist()
                        cleaned[k] = [b.decode('utf-8', 'ignore') if isinstance(b, bytes) else b for b in val] if len(val) > 0 and isinstance(val[0], bytes) else val
                elif isinstance(v, bytes):
                    cleaned[k] = v.decode('utf-8', 'ignore')
                else:
                    cleaned[k] = v
            return cleaned

        def visitor(name, node):
            if isinstance(node, h5py.Dataset):
                structure["datasets"][name] = {
                    "shape": node.shape,
                    "dtype": str(node.dtype),
                    "attrs": clean_attrs(node.attrs)
                }
            elif isinstance(node, h5py.Group):
                structure["groups"].append(name)
                
        with h5py.File(filepath, 'r') as f:
            structure["attributes"] = clean_attrs(f.attrs)
            f.visititems(visitor)
            
        return structure, "Success"
    except Exception as e:
        return None, str(e)

def parse_slice_string(slice_str):
    """Converts a string like '0, 450:550' into a Python slice object."""
    if not slice_str or slice_str.strip() == ":":
        return slice(None)
        
    parts = slice_str.split(',')
    slices = []
    
    for p in parts:
        p = p.strip()
        if p == ':':
            slices.append(slice(None))
        elif ':' in p:
            start, stop = p.split(':')
            start = int(start) if start else None
            stop = int(stop) if stop else None
            slices.append(slice(start, stop))
        else:
            slices.append(int(p))
            
    return tuple(slices) if len(slices) > 1 else slices[0]

# ---------------------------------------------------------
# Main App
# ---------------------------------------------------------

if st.sidebar.button("Load / Refresh"):
    if file_path:
        st.session_state['file_path'] = file_path

if 'file_path' in st.session_state:
    struct, msg = get_h5_structure(st.session_state['file_path'])
    
    if struct is None:
        st.error(f"Error reading file: {msg}")
    else:
        st.success(f"File loaded successfully! ({st.session_state['file_path']})")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "1. Metadata & Structure", 
            "2. Temperature (2D)", 
            "3. Strain (2D)", 
            "4. Temperature (3D Map)", 
            "5. Strain (3D Map)"
        ])
        
        with tab1:
            # --- Section 1: Metadata and Structure ---
            with st.expander("View File Structure & Metadata", expanded=True):
                st.subheader("Global Attributes (Root)")
                if struct["attributes"]:
                    import datetime
                    attrs_to_show = struct["attributes"].copy()
                    ordered_attrs = {}
                    
                    for key_time in ['start_time', 'end_time']:
                        raw_val = attrs_to_show.pop(key_time, None)
                        if raw_val is not None:
                            try:
                                ts = float(raw_val[0] if isinstance(raw_val, list) else raw_val)
                                ordered_attrs[key_time] = f"{ts} ---> ({datetime.datetime.fromtimestamp(ts).strftime('%d/%m/%Y %H:%M:%S')})"
                            except:
                                ordered_attrs[key_time] = raw_val
                    
                    # Coloca o restante dos atributos abaixo
                    for k, v in attrs_to_show.items():
                        ordered_attrs[k] = v
                        
                    st.json(ordered_attrs)
                else:
                    st.info("No global attributes found.")
                    
                st.subheader("Available Datasets")
                for ds_name, info in struct["datasets"].items():
                    st.markdown(f"**`{ds_name}`** | Shape: `{info['shape']}` | Type: `{info['dtype']}`")
                    if info['attrs']:
                        st.json(info['attrs'])
        

        def render_2d_analysis_module(struct, section_num, section_title, y_dataset, x_dataset="distances", ymin=None, ymax=None):
            import datetime
            from scipy.signal import butter, filtfilt, savgol_filter, medfilt
            import numpy as np
            import pandas as pd
            import h5py
            import matplotlib.pyplot as plt
    
            st.header(f"{section_num}. {section_title}")
    
            if y_dataset not in struct["datasets"]:
                st.warning(f"Dataset '{y_dataset}' not found in the file.")
                return
        
            x_shape = struct["datasets"][x_dataset]["shape"] if x_dataset in struct["datasets"] else "N/A"
            x_dim_size = x_shape[0] if x_shape != "N/A" and len(x_shape) > 0 else 1
            y_shape = struct["datasets"][y_dataset]["shape"]
    
            x_slice = slice(0, x_dim_size)
            y_slice_list = []
    
            with st.expander(f"X Axis Physical Range ({x_dataset})", expanded=True):
                x_mapping = None
                try:
                    with h5py.File(st.session_state['file_path'], 'r') as f:
                        if x_dataset in f:
                            arr = f[x_dataset][:]
                            if len(arr.shape) == 1: x_mapping = arr
                except: pass
        
                if x_mapping is not None and len(x_mapping) > 1:
                    min_val, max_val = float(x_mapping[0]), float(x_mapping[-1])
                    st.markdown(f"**Physical Domain:** {min_val:.1f} to {max_val:.1f} (Array Size: {x_dim_size})")
            
                    use_slider = st.checkbox("Use Slider (Quick Adjust)", value=True, key=f"x_slide_{section_num}")
                    if use_slider:
                        slice_m = st.slider("Select Physical Range:", min_value=min_val, max_value=max_val, value=(min_val, max_val), key=f"m_slide_{section_num}")
                    else:
                        c1, c2 = st.columns(2)
                        with c1: s_in = st.number_input("Start (Physical)", min_value=min_val, max_value=max_val, value=min_val, key=f"m_s_{section_num}")
                        with c2: e_in = st.number_input("End (Physical)", min_value=min_val, max_value=max_val, value=max_val, key=f"m_e_{section_num}")
                        slice_m = (min_val, max_val) if s_in >= e_in else (s_in, e_in)
                    
                    start_idx = int((np.abs(x_mapping - slice_m[0])).argmin())
                    end_idx = int((np.abs(x_mapping - slice_m[1])).argmin())
                    if end_idx == start_idx: end_idx += 1
                    slice_range = (start_idx, end_idx)
                else:
                    st.markdown(f"**Max Array Size (Indices):** {x_dim_size}")
                    use_slider = st.checkbox("Use Slider (Quick Adjust)", value=True, key=f"x_slide_{section_num}")
                    if use_slider:
                        slice_range = st.slider("Select Index Range Visually:", 0, x_dim_size, (0, x_dim_size), key=f"idx_slide_{section_num}")
                    else:
                        c1, c2 = st.columns(2)
                        with c1: s_in = st.number_input("Start Index", min_value=0, max_value=x_dim_size-1, value=0, key=f"idx_s_{section_num}")
                        with c2: e_in = st.number_input("End Index", min_value=1, max_value=x_dim_size, value=x_dim_size, key=f"idx_e_{section_num}")
                        slice_range = (0, x_dim_size) if s_in >= e_in else (int(s_in), int(e_in))
                
                x_slice = slice(slice_range[0], slice_range[1])
                st.info(f"The plot will have {slice_range[1] - slice_range[0]} points on the X Axis.")

            with st.expander(f"Y Axis Time Selection ({y_dataset})", expanded=True):
                if len(y_shape) <= 1:
                    st.write("1D Dataset. Interval automatically synchronized with X Axis.")
                    y_slice_list = [x_slice]
                else:
                    axis_dim = next((i for i, size in enumerate(y_shape) if size == x_dim_size), len(y_shape) - 1)
                    for dim_idx, dim_size in enumerate(y_shape):
                        if dim_idx == axis_dim:
                            st.write(f"🧭 **Dim {dim_idx} ({dim_size} pts):** Locked to Distance/Space Axis.")
                            y_slice_list.append(x_slice)
                        else:
                            if dim_size > 1:
                                time_opts = []
                                try:
                                    with h5py.File(st.session_state['file_path'], 'r') as f:
                                        if 'start_times' in f:
                                            times_raw = f['start_times'][:]
                                            time_opts = [f"Trace {i+1} ➔ {datetime.datetime.fromtimestamp(t).strftime('%d/%m/%y %H:%M')}" for i, t in enumerate(times_raw)]
                                except: pass
                            
                                if len(time_opts) == dim_size:
                                    sel_str = st.selectbox("⏱️ Select a Specific Trace Time:", time_opts, key=f"y_sel_{section_num}")
                                    trace_selection = time_opts.index(sel_str) + 1
                                else:
                                    trace_selection = st.number_input(f"⏱️ Select a Specific Trace (1 to {dim_size}):", min_value=1, max_value=dim_size, value=1, key=f"y_num_{section_num}")
                            
                                y_slice_list.append(slice(trace_selection - 1, trace_selection))
                            else:
                                y_slice_list.append(slice(0, 1))
            y_slice = tuple(y_slice_list) if len(y_slice_list) > 1 else y_slice_list[0] if y_slice_list else slice(None)

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

            if st.button(f"Extract and Plot 2D {y_dataset}", type="primary", key=f"btn_{section_num}"):
                try:
                    with h5py.File(st.session_state['file_path'], 'r') as f:
                        data_x = np.array(f[x_dataset][x_slice]).flatten()
                        data_y = np.array(f[y_dataset][y_slice]).flatten()
            
                    if len(data_x) != len(data_y):
                        st.error(f"Dimension Error: X has {len(data_x)} and Y has {len(data_y)}.")
                        return
                
                    st.session_state[f'raw_x_{section_num}'] = data_x
                    st.session_state[f'raw_y_{section_num}'] = data_y
                    st.success("Successfully processed slice!")
                except Exception as e:
                    st.error(f"HDF5 Slicing Error: {e}")
            
            if f'raw_x_{section_num}' in st.session_state and f'raw_y_{section_num}' in st.session_state:
                data_x = st.session_state[f'raw_x_{section_num}']
                data_y = st.session_state[f'raw_y_{section_num}']
        
                plots_to_make = []
                try:
                    if use_moving_avg:
                        ma = np.convolve(data_y, np.ones(ma_window)/ma_window, mode='same')
                        plots_to_make.append((ma, f'Moving Avg', 'tab:orange'))
                    if use_savgol:
                        sg = savgol_filter(data_y, window_length=sg_window, polyorder=sg_order)
                        plots_to_make.append((sg, f'Savitzky-Golay', 'tab:green'))
                    if use_median:
                        med = medfilt(data_y, kernel_size=med_kernel)
                        plots_to_make.append((med, f'Median', 'tab:purple'))
                    if use_bw_low:
                        b, a = butter(bw_low_order, bw_low_cutoff, btype='low', analog=False)
                        plots_to_make.append((filtfilt(b, a, data_y), f'Low-Pass', 'tab:red'))
                    if use_bw_high:
                        b, a = butter(bw_high_order, bw_high_cutoff, btype='high', analog=False)
                        plots_to_make.append((filtfilt(b, a, data_y), f'High-Pass', 'tab:cyan'))
                
                    fig, ax = plt.subplots(figsize=(10, 5))
                    alpha_orig = 0.3 if len(plots_to_make) > 0 else 1.0
                    ax.plot(data_x, data_y, label=f'Raw {y_dataset}', color='black', alpha=alpha_orig, linestyle='-' if alpha_orig == 1.0 else ':')
                    for pt in plots_to_make: ax.plot(data_x, pt[0], label=pt[1], color=pt[2], linewidth=1.5)
            
                    # Matplotlib Configuration
                    ax.set_xlabel(x_dataset, labelpad=15, loc='center') # O loc center e labelpad que colocamos
                    ax.set_ylabel(y_dataset, labelpad=20, loc='center')
                    ax.set_title(f"2D Signal Profile: {section_title} ({y_dataset} vs {x_dataset})", pad=20)
                    if ymin is not None and ymax is not None:
                        ax.set_ylim([ymin, ymax])
                    ax.grid(True, linestyle='--', alpha=0.7)
                    ax.legend()
                    st.pyplot(fig)
            
                    # Export
                    df_export = pd.DataFrame({x_dataset: data_x, y_dataset: data_y})
                    for pt in plots_to_make: df_export[pt[1]] = pt[0]
                    st.download_button(label="📥 Download Data as CSV", data=df_export.to_csv(index=False).encode('utf-8'), file_name=f"export_{y_dataset}.csv", mime="text/csv", key=f"csv_{section_num}")
            
                except Exception as e:
                    st.error(f"Plotting Error: {e}")

        # Iniciar as renderizações independentes nas Abas!
        with tab2:
            render_2d_analysis_module(struct, section_num=2, section_title="Temperature 2D Analysis", y_dataset="temp_data", ymin=-50, ymax=200)
            
        with tab3:
            render_2d_analysis_module(struct, section_num=3, section_title="Strain 2D Analysis", y_dataset="strain_data", ymin=-2000, ymax=2000)

        # --- Helper for 3D Surface Plots ---
        def plot_3d_surface(file_path, dataset_name, x_dataset, title, z_unit, colorscale, zmin=None, zmax=None):
            with st.spinner(f"Generating 3D Surface Plot for {dataset_name}..."):
                try:
                    with h5py.File(file_path, 'r') as f:
                        if dataset_name not in f:
                            st.warning(f"Dataset {dataset_name} not found in this file.")
                            return
                        z_data = f[dataset_name][()]
                        try:
                            x_raw = np.squeeze(f[x_dataset][()])
                        except Exception:
                            x_raw = None
                        
                        time_raw = None
                        if 'start_times' in f:
                            time_raw = np.squeeze(f['start_times'][()])
                        elif 'end_times' in f:
                            time_raw = np.squeeze(f['end_times'][()])
                        elif 'times' in f:
                            time_raw = np.squeeze(f['times'][()])
                    
                    z_sq = np.squeeze(np.array(z_data))
                    
                    if len(z_sq.shape) >= 2:
                        import datetime
                        max_dim1 = 150
                        max_dim2 = 800
                        
                        step_dim1 = max(1, z_sq.shape[0] // max_dim1)
                        step_dim2 = max(1, z_sq.shape[1] // max_dim2)
                        
                        z_down = z_sq[::step_dim1, ::step_dim2]
                        
                        if x_raw is not None and len(x_raw.shape) == 1 and x_raw.shape[0] == z_sq.shape[1]:
                            x_down = x_raw[::step_dim2]
                        else:
                            x_down = np.arange(z_sq.shape[1])[::step_dim2]
                            
                        if time_raw is not None and len(time_raw.shape) == 1 and time_raw.shape[0] == z_sq.shape[0]:
                            y_raw_down = time_raw[::step_dim1]
                            
                            # Computes interval to force exactly 10 min (600s) skips
                            trace_interval = float(y_raw_down[1] - y_raw_down[0]) if len(y_raw_down) > 1 else 60.0
                            dtick_10min = max(1, int(round(600.0 / trace_interval))) if trace_interval > 0 else 10
                            
                            y_down = []
                            for t_val in y_raw_down:
                                if t_val > 100000000:
                                    dt = datetime.datetime.fromtimestamp(t_val)
                                    date_str = dt.strftime('%d/%m/%y')
                                    time_str = dt.strftime('%H:%M')
                                    
                                    # '04/04/26' (8 chars) vs '00:50' (5 chars).
                                    # Inserimos 4 \u00A0 invisíveis na Hora para que ela fique alinhada à direita da data!
                                    aligned_time = f"{chr(160)*4}{time_str}"
                                    
                                    # Padding GIGANTE SÓ NA ESQUERDA:
                                    # Isso joga o centro matemático da string lá pra esquerda, 
                                    # o que força o motor 3D a renderizar os textos reais todos pra DIREITA 
                                    # (livrando completamente do título "Time" e alinhando-se perfeitamente).
                                    pad_left = chr(160) * 25
                                    pad_right = ""
                                    
                                    y_down.append(f"{pad_left}{date_str}{pad_right}<br>{pad_left}{aligned_time}{pad_right}")
                                else:
                                    y_down.append(f"{chr(160)*25}{t_val:.1f}s")
                        else:
                            y_down = np.arange(z_sq.shape[0])[::step_dim1]
                        
                        # Set up layout kwargs
                        surface_kwargs = dict(z=z_down, x=x_down, y=y_down, colorscale=colorscale)
                        if zmin is not None: surface_kwargs['cmin'] = zmin
                        if zmax is not None: surface_kwargs['cmax'] = zmax

                        fig_3d = go.Figure(data=[go.Surface(**surface_kwargs)])
                        
                        scene_dict = dict(
                            xaxis=dict(title="Distance (m)"),
                            yaxis=dict(title="Time", tickmode='linear', dtick=dtick_10min) if 'dtick_10min' in locals() else dict(title="Time"),
                            zaxis=dict(title=z_unit)
                        )
                        if zmin is not None and zmax is not None:
                            scene_dict['zaxis'].update(dict(range=[zmin, zmax]))

                        fig_3d.update_layout(
                            title=title,
                            autosize=True,
                            height=700,
                            # A Margem 'b' (bottom) FOI AUMENTADA para abrigar a palavra isolada lá no fundo:
                            margin=dict(l=65, r=50, b=150, t=90),
                            scene=scene_dict
                        )
                        st.plotly_chart(fig_3d, use_container_width=True)
                    else:
                        st.warning("Data squeezed to less than 2D, cannot plot surface.")
                except Exception as e:
                    st.error(f"Error drawing 3D surface: {e}")

        with tab4:
            st.header("4. 3D Surface Graph ( Temperature )")
            if "temp_data" in struct["datasets"]:
                st.write("Real-time 3D topographic visualization of Temperature.")
                if st.checkbox("Show / Generate Temperature 3D Surface", value=False):
                    plot_3d_surface(st.session_state['file_path'], "temp_data", "distances", "3D Surface Topography: Temperature", "Temperature (°C)", "Turbo", zmin=-50, zmax=200)
            else:
                 st.info("The 'temp_data' dataset was not found in this file.")

        with tab5:
            st.header("5. 3D Surface Graph ( Strain )")
            if "strain_data" in struct["datasets"]:
                st.write("Real-time 3D topographic visualization of Strain.")
                if st.checkbox("Show / Generate Strain 3D Surface", value=False):
                    plot_3d_surface(st.session_state['file_path'], "strain_data", "distances", "3D Surface Topography: Strain", "Strain (µe)", "Viridis", zmin=-2000, zmax=2000)
            else:
                 st.info("The 'strain_data' dataset was not found in this file.")
            
else:
    st.info("Enter the file path in the sidebar and click 'Load'.")
