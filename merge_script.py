import re
import os

with open("h5_viewer.py", "r") as f:
    h5_content = f.read()

with open("app.py", "r") as f:
    app_content = f.read()

h5_imports = re.search(r"^(?:import .*?\n|from .*?\n)+", h5_content, re.MULTILINE).group(0)
h5_get_h5_structure = re.search(r"def get_h5_structure.*?return None, str\(e\)", h5_content, re.DOTALL).group(0)
h5_parse_slice_string = re.search(r"def parse_slice_string.*?return tuple\(slices\) if len\(slices\) > 1 else slices\[0\]", h5_content, re.DOTALL).group(0)
h5_render_2d = re.search(r"def render_2d_analysis_module.*?st\.pyplot\(fig\).*?except Exception as e:.*?st\.error\(f\"Plotting Error: {e}\"\)", h5_content, re.DOTALL).group(0)
h5_plot_3d = re.search(r"def plot_3d_surface.*?st\.error\(f\"Error drawing 3D surface: {e}\"\)", h5_content, re.DOTALL).group(0)

app_imports = re.search(r"^(?:import .*?\n|from .*?\n)+", app_content, re.MULTILINE).group(0)
app_css = re.search(r"# Premium CSS Styling.*?unsafe_allow_html=True\)", app_content, re.DOTALL).group(0)
app_helpers = re.search(r"# --- HDF5 FILE READING ---.*?def generate_natural_language_response.*?return None", app_content, re.DOTALL).group(0)

combined = []
combined.append("# Unified Imports")
# Combine imports uniquely
all_imports = set(h5_imports.split("\n") + app_imports.split("\n"))
imports_clean = ["import streamlit as st"]
for imp in sorted(list(all_imports)):
    imp = imp.strip()
    if imp and "streamlit" not in imp:
        imports_clean.append(imp)
        
combined.extend(imports_clean)

combined.append("\n# --- CONFIGURATION & PAGE ---")
combined.append("""
def get_asset_path(filename):
    if getattr(sys, 'frozen', False):
        return Path(sys._MEIPASS) / filename
    return Path(__file__).parent.absolute() / filename

st.set_page_config(
    page_title="FEBUS DTSS Viewer & Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)
""")

combined.append(app_css)

combined.append("\n# --- HELPER FUNCTIONS ---")
combined.append(h5_get_h5_structure)
combined.append("\n")
combined.append(h5_parse_slice_string)
combined.append("\n")
combined.append(app_helpers)

combined.append("\n# --- 2D & 3D RENDERING FUNCTIONS ---")
combined.append(h5_render_2d)
combined.append("\n")
combined.append(h5_plot_3d)

combined.append("""
# --- UNIFIED SIDEBAR ---
st.sidebar.markdown("### 📁 1. Load File")
uploaded_file = st.sidebar.file_uploader("Upload HDF5 File", type=['h5', 'hdf5'])

if uploaded_file is not None:
    temp_path = Path("uploaded_" + uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.session_state['file_path'] = str(temp_path)
    st.sidebar.success("File uploaded successfully!")
else:
    default_file = "Simulated_FiberTest_TSB_2km_noise_1_5.h5"
    if os.path.exists(default_file):
        if 'file_path' not in st.session_state:
            st.session_state['file_path'] = default_file
        st.sidebar.info("Using default simulated file.")
    else:
        st.sidebar.warning("Please upload a .h5 file to begin.")

# Load Metadata globally if file exists
metadata = {}
if 'file_path' in st.session_state:
    file_path = st.session_state['file_path']
    metadata = get_sensor_metadata(file_path)
    
    if "error" in metadata:
        st.sidebar.error(f"File Error: {metadata['error']}")
        st.stop()

# LLM Config & Metadata Sidebar
llm_provider = "Ollama (Local)"
api_key = None
gemini_model = None
ollama_host = "http://localhost:11434"
ollama_model = "llama3"

with st.sidebar:
    st.markdown("### ⚙️ 2. IA Configuration")
    llm_provider = st.selectbox(
        "LLM Provider",
        ["Rules (Local/Fast)", "Google Gemini", "Ollama (Local)"],
        index=2,
        help="Select the AI provider to interpret your questions."
    )
    
    if llm_provider == "Google Gemini":
        default_key = os.environ.get("GEMINI_API_KEY", "")
        if not default_key and os.path.exists(".env"):
            try:
                with open(".env", "r") as env_f:
                    for line in env_f:
                        if line.strip().startswith("GEMINI_API_KEY="):
                            default_key = line.strip().split("=", 1)[1].strip().strip('"').strip("'")
                            break
            except: pass
        api_key = st.text_input("Gemini API Key", value=default_key, type="password")
        gemini_model = st.selectbox("Gemini Model", ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"], index=0)
        if not api_key: st.warning("⚠️ Enter API Key to enable Gemini.")
    elif llm_provider == "Ollama (Local)":
        ollama_host = st.text_input("Ollama Host URL", value="http://localhost:11434")
        ollama_model = st.text_input("Model Name", value="llama3")

    if metadata and "interrogator_model" in metadata:
        st.markdown("---")
        st.markdown("### 📊 3. Fiber Optic Details (DTSS)")
        time_range_html = ""
        if metadata.get('start_times') is not None and len(metadata['start_times']) > 0:
            start_dt = datetime.fromtimestamp(metadata['start_times'][0]).strftime('%Y-%m-%d %H:%M:%S')
            end_dt = datetime.fromtimestamp(metadata['end_times'][-1]).strftime('%Y-%m-%d %H:%M:%S')
            time_range_html = f'''
            <div class="meta-card">
                <div class="meta-title">Monitoring Period</div>
                <div class="meta-value" style="font-size: 0.8rem; font-weight: 600; line-height: 1.2; margin-top: 2px;">{start_dt}<br/>to<br/>{end_dt} (UTC)</div>
            </div>'''
            
        st.markdown(f'''
        <div class="meta-card"><div class="meta-title">Interrogator</div><div class="meta-value">{metadata.get('interrogator_model')}</div></div>
        <div class="meta-card"><div class="meta-title">Location</div><div class="meta-value">{metadata.get('location')}</div></div>
        <div class="meta-card"><div class="meta-title">Cable Length</div><div class="meta-value">{metadata.get('cable_length_m', 0):.1f} m</div></div>
        <div class="meta-card"><div class="meta-title">Channels</div><div class="meta-value">{metadata.get('num_channels', 0):,}</div></div>
        <div class="meta-card"><div class="meta-title">Measurements</div><div class="meta-value">{metadata.get('num_measurements', 0)}</div></div>
        {time_range_html}
        ''', unsafe_allow_html=True)
""")

combined.append("""
# --- MAIN PAGE ---
if 'file_path' in st.session_state:
    file_path = st.session_state['file_path']
    struct, msg = get_h5_structure(file_path)
    
    st.markdown('<h1 class="main-title">🧠 FEBUS HDF5 Viewer & Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Explore, slice, plot, and ask questions about the optical fiber HDF5 directly.</p>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "1. Metadata", 
        "2. Temp (2D)", 
        "3. Strain (2D)", 
        "4. Temp (3D)", 
        "5. Strain (3D)",
        "6. Intelligent Assistant"
    ])
""")

h5_tab1 = re.search(r"        with tab1:(.*?)def render_2d_analysis_module", h5_content, re.DOTALL).group(1)
h5_tab2 = re.search(r"        with tab2:(.*?)with tab3:", h5_content, re.DOTALL).group(1)
h5_tab3 = re.search(r"        with tab3:(.*?)def plot_3d_surface", h5_content, re.DOTALL).group(1)
h5_tab4 = re.search(r"        with tab4:(.*?)with tab5:", h5_content, re.DOTALL).group(1)
h5_tab5 = re.search(r"        with tab5:(.*?)else:\n    st\.info", h5_content, re.DOTALL).group(1)

combined.append("    with tab1:\n" + h5_tab1)
combined.append("    with tab2:\n" + h5_tab2)
combined.append("    with tab3:\n" + h5_tab3)
combined.append("    with tab4:\n" + h5_tab4)
combined.append("    with tab5:\n" + h5_tab5)

app_main_code = re.search(r"col1, col2 = st\.columns\(\[1\.2, 1\.0\], gap=\"large\"\).*?st\.rerun\(\)", app_content, re.DOTALL).group(0)
app_main_code = app_main_code.replace("HDF5_FILE_PATH", "file_path")
app_main_code_indented = "\n".join("        " + line for line in app_main_code.split("\n"))

combined.append("    with tab6:\n" + app_main_code_indented)

combined.append("""
else:
    st.info("Please wait while the file is loading or upload a new file from the sidebar.")
""")

with open("combined_app.py", "w") as f:
    f.write("\n".join(combined))

print("combined_app.py generated successfully!")
