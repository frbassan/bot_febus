# 🧠 FEBUS DTSS Viewer & Assistant

An interactive, high-performance Streamlit dashboard and AI-powered assistant designed to explore, visualize, and query Distributed Temperature and Strain Sensing (DTSS) data stored in HDF5 format.

## 📊 Project Block Diagram

The diagram below outlines the data flow, digital signal processing (DSP) pipeline, layout presentation, and AI Assistant subsystem:

```mermaid
flowchart TD
    %% Define Styles %%
    classDef datasource fill:#e1f5fe,stroke:#039be5,stroke-width:2px,color:#01579b;
    classDef processing fill:#fbe9e7,stroke:#ff5722,stroke-width:2px,color:#bf360c;
    classDef ui fill:#ede7f6,stroke:#7e57c2,stroke-width:2px,color:#311b92;
    classDef ai fill:#e8f5e9,stroke:#4caf50,stroke-width:2px,color:#1b5e20;

    %% Data Inputs %%
    subgraph DataSources["📁 Data Inputs"]
        A["Real FEBUS HDF5 File"]:::datasource
        B["generate_tsb_febus.py<br>(Physics Simulator)"]:::datasource -.-> C["Simulated HDF5 File (*.h5)"]:::datasource
    end

    %% Streamlit Core App %%
    subgraph CombinedApp["🧠 combined_app.py (Streamlit Application)"]
        %% File Loading & Processing %%
        C & A --> D["File Uploader & Metadata Loader"]:::processing
        D --> E["Data Slicing & Downsampling (< 1500 pts)"]:::processing
        
        %% DSP Filters %%
        subgraph DSPOptions["〰️ Digital Signal Processing (DSP)"]
            E --> F["Moving Avg / Savitzky-Golay"]:::processing
            E --> G["Butterworth Filter (Low/High-Pass)"]:::processing
            E --> H["Median Spike Filter"]:::processing
        end

        %% Tabs Layout %%
        subgraph VisualTabs["🖥️ Dashboard Visualizations (Plotly)"]
            F & G & H --> T1["Tab 1: Metadata & Structure Explorer"]:::ui
            F & G & H --> T2["Tabs 2 & 3: 2D Spatial Profiles"]:::ui
            F & G & H --> T3["Tabs 4 & 5: 2D Time-Series Evolution"]:::ui
            F & G & H --> T4["Tabs 6 & 7: 3D Surface Topography"]:::ui
        end

        %% AI Assistant Subsystem %%
        subgraph AIAssistant["🤖 Intelligent Assistant Subsystem (Tab 8)"]
            T5["Chat Interface & Interactive Viewer"]:::ui
            T5 -->|User Query| IE["Intent Extractor"]:::ai
            
            subgraph AIEngine["AI Parsing Options"]
                IE -->|Option 1| IE1["Google Gemini API"]:::ai
                IE -->|Option 2| IE2["Ollama Local LLM"]:::ai
                IE -->|Option 3| IE3["Rules-based Regex Parser"]:::ai
            end
            
            IE1 & IE2 & IE3 -->|Structured JSON Query| QE["HDF5 Query Core<br>(LlmBotCore)"]:::processing
            QE -->|Retrieved Data| RG["NL Response Generator<br>(Gemini / Ollama / Rules)"]:::ai
            RG -->|Markdown Explanation| T5
        end
    end
```

---

## 🚀 Key Features

*   **📊 Multi-Dimensional Visualization**:
    *   **2D Spatial Analysis**: Slice temperature and strain profiles across the fiber length for specific measurement times.
    *   **2D Time-Series Evolution**: Track temperature and strain changes at any individual fiber location over the entire monitoring period.
    *   **3D Surface Topography**: Generate topographic interactive plots displaying space-time thermal and mechanical dynamics.
*   **〰️ Advanced Digital Signal Processing (DSP)**:
    *   **Smoothing**: Moving Average and Savitzky-Golay filtering.
    *   **Frequency Filters**: Butterworth low-pass and high-pass filters to reduce high-frequency noise or capture trends.
    *   **Spike Removal**: Median filtering for sensor spike mitigation.
*   **🧠 Intelligent AI Assistant**:
    *   Natural language query extractor (parses questions like *"What is the peak temperature in scan 3?"* or *"Hottest spot in the cable history?"*).
    *   Support for multiple LLM backends: **Google Gemini**, local **Ollama** models, and a fast regex-based **Rules Engine** fallback.
    *   Side-by-side interactive data panel for manual cross-verification.
*   **⚙️ Physics-Based Fiber Simulator**:
    *   Generates custom synthetic FEBUS DTSS HDF5 datasets.
    *   Simulates realistic temperature/strain events (sinusoidal, linear, transient peaks).
    *   Computes Brillouin Frequency Shifts (BSL) based on official physical sensitivities.
*   **⚡ Performance Optimization**:
    *   Automatic signal downsampling for high-resolution datasets (>1500 spatial points) to maintain fluid rendering rates in Plotly.
    *   Optimized low-overhead HDF5 file-reading utilities.

---

## 📁 Project Structure

After cleaning up obsolete files and utilities, the project is cleanly structured around 4 core Python files and organized folders:

| File / Folder | Description |
|------|-------------|
| **`HDF5_files/`** | Directory where all the `.hdf5` and `.h5` sensor data files should be placed for easy loading via the interface. |
| **`combined_app.py`** | The main application logic and user interface. It is written using the Streamlit framework and contains all the code for reading `.hdf5` data, rendering the Plotly 2D/3D charts, applying DSP filters, and integrating the LLM assistant. |
| **`run_desktop.py`** | The entry point for running the application natively on your desktop. It silently starts the Streamlit server from `combined_app.py` in the background and opens a `pywebview` window so you get a native desktop app experience without needing a browser. |
| **`build_linux.py`** | A build script used to package the entire application into a single standalone executable using `PyInstaller`. It configures all the necessary hooks and metadata to bundle Streamlit and the desktop wrapper together. |
| **`generate_tsb_febus.py`** | A simulation and generation script. It programmatically generates the dummy `.hdf5` test files to mimic the real output structure of a FEBUS DTSS system for testing and development purposes. |

---

## 🗄️ HDF5 Data Structure

The application expects HDF5 files (`.h5` or `.hdf5`) configured with the following structure, aligned with official FEBUS DTSS interrogator outputs:

### Root Attributes (Metadata)
| Attribute | Type | Description |
|---|---|---|
| `acq_res` | `int` | Acquisition resolution settings |
| `average` | `int` | Number of averaged sweeps |
| `fiberLength` | `int` | Total physical length of the fiber (m) |
| `sampling_resolution`| `float` | Distance step between sampling channels (m) |
| `temp_freq_sensitivity` | `float` | Fiber frequency sensitivity to Temperature (~1.07 MHz/°C) |
| `strain_freq_sensitivity`| `float` | Fiber frequency sensitivity to Strain (~0.046 MHz/µε) |
| `start_time` / `end_time` | `float` | Timestamps indicating monitoring period limits |

### Datasets
*   **`distances`** `[N_distances]`: 1D array mapping each sampling point to its physical distance along the fiber (in meters).
*   **`start_times`** `[N_traces]`: 1D array containing unix timestamps for when each trace measurement began.
*   **`end_times`** `[N_traces]`: 1D array containing unix timestamps for when each trace measurement finished.
*   **`temp_data`** `[N_traces x N_distances]`: 2D array of temperature readings (°C).
*   **`strain_data`** `[N_traces x N_distances]`: 2D array of strain/deformation readings (microstrain, µε).
*   **`bsl_data`** `[N_traces x N_distances]`: 2D array of raw Brillouin Shift frequencies (MHz).

---

## ⚙️ How it Works: DSP Filtering & Physical Simulation

### Digital Signal Processing (DSP)
The app includes interactive scipy-based filtering tabs for 2D profile graphs:
1.  **Moving Average**: Smooths local spatial fluctuations using a rolling mean window.
2.  **Savitzky-Golay**: Applies a local polynomial regression to preserve signal peaks while filtering high-frequency noise.
3.  **Butterworth Filter**: Performs zero-phase digital filtering (`filtfilt`) with customizable low-pass or high-pass frequency thresholds.
4.  **Median Filter**: Replaces samples with their neighborhood median to reject short-lived sensor spikes.

### Synthetic Data Generation
`generate_tsb_febus.py` creates physical events on the fiber using a Gaussian kernel:
$$\text{Event}(x) = A \cdot f(t) \cdot \exp\left(-0.5 \cdot \left(\frac{x - x_{\text{center}}}{\sigma}\right)^2\right)$$
where $A$ is the amplitude, $f(t)$ represents the temporal evolution factor (sinusoidal, linear, or peak), and $\sigma$ represents the event width. 
It then calculates the corresponding Brillouin Frequency Shift:
$$\text{BSL}(x, t) = \text{BSL}_{\text{ref}} + (T(x,t) - T_{\text{ref}}) \cdot C_T + (\epsilon(x,t) - \epsilon_{\text{ref}}) \cdot C_\epsilon$$
where $C_T$ and $C_\epsilon$ are the temperature and strain frequency sensitivities, respectively.

---

## 🛠️ Setup & Running

### 1. Installation
Ensure you have Python 3.9+ installed. Clone the repository and install the dependencies listed in `requirements.txt`. You will also need `scipy` for digital filters:

```bash
pip install -r requirements.txt scipy
```

### 2. Generate Simulated Data (Optional)
If you do not have a real FEBUS HDF5 file, run the physics simulator to build a synthetic dataset. The resulting file will be placed in your root, which you can move to `HDF5_files/`:

```bash
python generate_tsb_febus.py
```

### 3. Run the Dashboard

**Option A: Native Desktop App (Recommended)**
To run the application natively in a standalone window without opening your browser:
```bash
python run_desktop.py
```

**Option B: Compiled Linux Executable**
If you have built the executable via `build_linux.py`, you can run the standalone binary directly:
```bash
./dist/FEBUS_Viewer
```

**Option C: Standard Web Browser**
Launch the classic Streamlit web application:
```bash
streamlit run combined_app.py
```

---

## 🤖 AI Assistant Configuration
Under the **Intelligent Assistant** tab, configure the desired backend in the sidebar:
1.  **Rules (Local/Fast)**: Regular expression parsing. Fast and requires no API keys or servers.
2.  **Google Gemini**: Highly capable generative AI parsing. Requires a `GEMINI_API_KEY` (configured via sidebar or a `.env` file).
3.  **Ollama (Local)**: Queries a locally run Ollama model (e.g. `llama3` or `mistral`) at `http://localhost:11434`.
