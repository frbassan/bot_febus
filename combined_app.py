# Unified Imports
import streamlit as st
from PIL import Image
from datetime import datetime, timedelta
from pathlib import Path
import google.generativeai as genai
import h5py
import json
import plotly.graph_objects as go
import numpy as np
import os
import pandas as pd
import plotly.graph_objects as go
import re
import requests
import sys

# --- CONFIGURATION & PAGE ---

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

# Premium CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Main title gradient */
    .main-title {
        background: linear-gradient(135deg, #FF4B4B 0%, #1A73E8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 2.2rem;
        margin-bottom: 0.2rem;
        padding-top: 0px;
    }
    
    .subtitle {
        color: #666;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    
    /* Metadata Cards in Sidebar */
    .meta-card {
        background-color: #f0f2f6;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 8px;
        border-left: 4px solid #1A73E8;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    .meta-title {
        font-size: 0.75rem;
        color: #555;
        text-transform: uppercase;
        font-weight: 600;
        margin-bottom: 2px;
    }
    
    .meta-value {
        font-size: 0.95rem;
        color: #111;
        font-weight: 700;
    }
    
    /* Result Card in Chat */
    .result-card {
        background: linear-gradient(145deg, #ffffff, #f7f9fc);
        border-radius: 10px;
        padding: 15px;
        margin-top: 8px;
        margin-bottom: 8px;
        border: 1px solid #e1e4e8;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
    }
    
    .result-header {
        font-size: 1.05rem;
        font-weight: 600;
        color: #1A73E8;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    
    .result-body {
        font-size: 0.95rem;
        color: #333;
        line-height: 1.4;
    }
    
    /* Soft scrollbar */
    ::-webkit-scrollbar {
        width: 5px;
        height: 5px;
    }
    ::-webkit-scrollbar-thumb {
        background: #bbb;
        border-radius: 3px;
    }
</style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
@st.cache_data
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


# --- HDF5 FILE READING ---
@st.cache_data
def get_sensor_metadata(file_path):
    """Loads global metadata and dimensions from the HDF5 file."""
    try:
        with h5py.File(file_path, 'r') as f:
            # Validate structure
            if "distances" not in f or "temp_data" not in f or "strain_data" not in f:
                return {
                    "error": "The HDF5 file does not have the expected structure (datasets 'distances', 'temp_data', and 'strain_data')."
                }
            
            distances = f['distances'][:]
            num_measurements = f['temp_data'].shape[0]
            
            # Load attributes with fallbacks
            interrogator = f.attrs.get("interrogator_model")
            if isinstance(interrogator, bytes):
                interrogator = interrogator.decode('utf-8')
            elif interrogator is None:
                interrogator = "FEBUS G2-R"
                
            location = f.attrs.get("location")
            if isinstance(location, bytes):
                location = location.decode('utf-8')
            elif location is None:
                location = "TS Conductor Mega Test Site"
                
            pulse_width = f.attrs.get("pulse_width_ns", 10.0)
            if isinstance(pulse_width, (str, bytes)):
                try:
                    pulse_width = float(pulse_width)
                except ValueError:
                    pulse_width = 10.0
            
            # Read all global attributes dynamically
            all_global_attrs = {}
            for k, v in f.attrs.items():
                if isinstance(v, bytes):
                    v = v.decode('utf-8', errors='ignore')
                all_global_attrs[k] = v
                
            # Read all datasets metadata dynamically
            datasets_info = {}
            for k in f.keys():
                ds = f[k]
                ds_attrs = {}
                for ak, av in ds.attrs.items():
                    if isinstance(av, bytes):
                        av = av.decode('utf-8', errors='ignore')
                    ds_attrs[ak] = av
                datasets_info[k] = {
                    "shape": ds.shape,
                    "dtype": str(ds.dtype),
                    "attrs": ds_attrs
                }
            
            start_times = f['start_times'][:] if 'start_times' in f else None
            end_times = f['end_times'][:] if 'end_times' in f else None
            
            meta = {
                "interrogator_model": interrogator,
                "location": location,
                "pulse_width_ns": pulse_width,
                "cable_length_m": float(distances[-1]),
                "num_channels": len(distances),
                "num_measurements": num_measurements,
                "all_global_attrs": all_global_attrs,
                "datasets_info": datasets_info,
                "start_times": start_times,
                "end_times": end_times
            }
            return meta
    except Exception as e:
        return {"error": str(e)}


# Load initial metadata
HDF5_FILE_PATH = "HDF5_files/Simulated_FiberTest_TSB_2km_noise_1_5.h5"
metadata = get_sensor_metadata(HDF5_FILE_PATH)

if "error" in metadata:
    st.sidebar.error(f"File Error: {metadata['error']}")
    st.info("Please select an HDF5 file compatible with the DTSS structure (containing 'distances', 'temp_data', and 'strain_data').")
    st.stop()


# --- CACHED DATA LOADING ---
@st.cache_data
def load_h5_dataset(file_path, dataset_name):
    try:
        with h5py.File(file_path, 'r') as f:
            if dataset_name in f:
                return f[dataset_name][:]
    except Exception:
        pass
    return None

@st.cache_data
def get_cached_profile(file_path, quantity, idx_m):
    try:
        with h5py.File(file_path, 'r') as f:
            distances = f['distances'][:]
            temp_data = f['temp_data'][idx_m, :] if quantity in ['temperature', 'both'] else None
            strain_data = f['strain_data'][idx_m, :] if quantity in ['deformation', 'both'] else None
            return distances, temp_data, strain_data
    except Exception:
        return np.array([]), None, None

# --- HDF5 QUERY LOGIC ---
class LlmBotCore:
    """Core interface for querying raw data from the HDF5 file."""
    def __init__(self, file_path):
        self.file_path = file_path
        try:
            meta = get_sensor_metadata(file_path)
            if "error" not in meta:
                self.num_measurements = meta.get("num_measurements", 8)
            else:
                self.num_measurements = 8
        except Exception:
            self.num_measurements = 8

    def _parse_measurement_index(self, index):
        """Maps index input (including literal strings or timestamps) to a valid index."""
        max_idx = self.num_measurements - 1
        if index == "latest" or index is None:
            return max_idx
        try:
            val = int(index)
            if val < 0:
                return 0
            if val > max_idx:
                return max_idx
            return val
        except (ValueError, TypeError):
            pass
            
        # If it's a string, it might be a timestamp or datetime query
        if isinstance(index, str):
            try:
                start_times = load_h5_dataset(self.file_path, 'start_times')
                if start_times is not None:
                    # Parse time/date formats
                    query_time = None
                    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%H:%M:%S", "%H:%M", "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M"):
                        try:
                            parsed = datetime.strptime(index, fmt)
                            # Default to base date if only time is queried
                            if "%Y" not in fmt and "%y" not in fmt:
                                parsed = parsed.replace(year=2026, month=1, day=1)
                            query_time = parsed
                            break
                        except ValueError:
                            continue
                            
                    if query_time is not None:
                        from datetime import timezone
                        query_time = query_time.replace(tzinfo=timezone.utc)
                        query_ts = query_time.timestamp()
                        
                        # Find index of closest start_time
                        closest_idx = (np.abs(start_times - query_ts)).argmin()
                        return int(closest_idx)
            except Exception:
                pass
                
        return max_idx

    def query_profile(self, quantity, measurement_index):
        """Returns distance array and data for a given measurement index."""
        idx_m = self._parse_measurement_index(measurement_index)
        distances, temp_data, def_data = get_cached_profile(self.file_path, quantity, idx_m)
        return distances, temp_data, def_data, idx_m

    def query_value_at_distance(self, quantity, measurement_index, target_distance):
        """Finds values closest to the target distance."""
        idx_m = self._parse_measurement_index(measurement_index)
        distances, temp_data, def_data = get_cached_profile(self.file_path, 'both', idx_m)
        
        if len(distances) == 0:
            return target_distance, None, None, idx_m
            
        dist_idx = (np.abs(distances - target_distance)).argmin()
        actual_distance = float(distances[dist_idx])
        
        temp_val = float(temp_data[dist_idx]) if temp_data is not None else None
        def_val = float(def_data[dist_idx]) if def_data is not None else None
        
        return actual_distance, temp_val, def_val, idx_m

    def query_peak(self, quantity, measurement_index, peak_type='max'):
        """Finds max or min peak values and their spatial locations."""
        idx_m = self._parse_measurement_index(measurement_index)
        distances, temp_data, def_data = get_cached_profile(self.file_path, quantity, idx_m)
        
        res = {}
        if quantity in ['temperature', 'both'] and temp_data is not None and len(temp_data) > 0:
            p_idx = np.argmax(temp_data) if peak_type == 'max' else np.argmin(temp_data)
            res['temp'] = {
                "value": float(temp_data[p_idx]),
                "distance": float(distances[p_idx])
            }
        if quantity in ['deformation', 'both'] and def_data is not None and len(def_data) > 0:
            p_idx = np.argmax(def_data) if peak_type == 'max' else np.argmin(def_data)
            res['def'] = {
                "value": float(def_data[p_idx]),
                "distance": float(distances[p_idx])
            }
        return res, idx_m

    def query_history(self, quantity, target_distance):
        """Returns the historical evolution (all measurements) at a fixed distance."""
        distances = load_h5_dataset(self.file_path, 'distances')
        if distances is None or len(distances) == 0:
            return target_distance, None, None
            
        dist_idx = (np.abs(distances - target_distance)).argmin()
        actual_distance = float(distances[dist_idx])
        
        temp_history = None
        def_history = None
        
        if quantity in ['temperature', 'both']:
            temp_data_all = load_h5_dataset(self.file_path, 'temp_data')
            if temp_data_all is not None:
                temp_history = [float(x) for x in temp_data_all[:, dist_idx]]
                
        if quantity in ['deformation', 'both']:
            strain_data_all = load_h5_dataset(self.file_path, 'strain_data')
            if strain_data_all is not None:
                def_history = [float(x) for x in strain_data_all[:, dist_idx]]
                
        return actual_distance, temp_history, def_history

    def query_global_peak(self, quantity, peak_type='max'):
        """Finds the global maximum or minimum value, distance, and measurement index across all measurements."""
        distances = load_h5_dataset(self.file_path, 'distances')
        start_times = load_h5_dataset(self.file_path, 'start_times')
        
        if distances is None or len(distances) == 0:
            return {}
            
        res = {}
        if quantity in ['temperature', 'both']:
            temp_data_all = load_h5_dataset(self.file_path, 'temp_data')
            if temp_data_all is not None:
                flat_idx = np.argmax(temp_data_all) if peak_type == 'max' else np.argmin(temp_data_all)
                m_idx, d_idx = np.unravel_index(flat_idx, temp_data_all.shape)
                timestamp = float(start_times[m_idx]) if start_times is not None else None
                res['temp'] = {
                    "value": float(temp_data_all[m_idx, d_idx]),
                    "distance": float(distances[d_idx]),
                    "measurement_index": int(m_idx),
                    "timestamp": timestamp
                }
                
        if quantity in ['deformation', 'both']:
            strain_data_all = load_h5_dataset(self.file_path, 'strain_data')
            if strain_data_all is not None:
                flat_idx = np.argmax(strain_data_all) if peak_type == 'max' else np.argmin(strain_data_all)
                m_idx, d_idx = np.unravel_index(flat_idx, strain_data_all.shape)
                timestamp = float(start_times[m_idx]) if start_times is not None else None
                res['def'] = {
                    "value": float(strain_data_all[m_idx, d_idx]),
                    "distance": float(distances[d_idx]),
                    "measurement_index": int(m_idx),
                    "timestamp": timestamp
                }
                
        return res


# --- INTENT EXTRACTION (LLM & FALLBACK) ---
class IntentExtractor:
    """Parses user natural language queries into a structured JSON query format."""
    def __init__(self, provider, api_key=None, gemini_model=None, ollama_host=None, ollama_model=None, num_measurements=8, cable_length=10000.0, num_channels=10000):
        self.provider = provider
        self.api_key = api_key
        self.gemini_model = gemini_model or "gemini-1.5-flash"
        self.ollama_host = ollama_host
        self.ollama_model = ollama_model
        
        self.system_prompt = f"""
You are an expert assistant specialized in analyzing questions about an HDF5 file containing measurement data from a DTSS (Distributed Temperature and Strain Sensor) optical fiber.
The sensor configuration:
- Distance: 0 to {cable_length:.1f} meters along the cable ({num_channels} sampling points).
- Two physical quantities: Temperature (temperature) and Deformation/Strain (deformation).
- {num_measurements} temporal measurements saved (indices 0 to {num_measurements - 1}, where {num_measurements - 1} is the latest or most recent).

Your task is to analyze the user's query and extract the query parameters into a valid JSON object.
Respond ONLY with the JSON object. Do not include Markdown wrapping (like ```json), introductions, or any other characters.

The JSON structure must match this schema:
{{
  "quantity": "temperature" | "deformation" | "both" | "metadata" | null,
  "analysis_type": "plot_profile" | "plot_history" | "value_at_distance" | "find_peak" | "find_global_peak" | "metadata_info" | "help" | null,
  "measurement_index": integer | "latest" | "YYYY-MM-DD HH:MM:SS" (or "HH:MM" time string) | null,
  "distance_m": float | null,
  "peak_type": "max" | "min" | null
}}

Extraction Rules:
- "quantity": Choose "temperature" for temperature queries, "deformation" for strain/deformation queries, "both" if both are requested, or "metadata" for general sensor metadata.
- "analysis_type":
  - "plot_profile": If the user wants to draw/plot/graph the whole cable profile (e.g. "plot temperature of test 2", "strain graph for scan 0").
  - "plot_history": If the user wants a history or time-series evolution at a specific distance (e.g. "history at point 500m", "temperature evolution at 2500m over time").
  - "value_at_distance": If the user wants the exact value at a specific location (e.g. "what is the temperature at 1200 meters in measurement 3?", "strain value at point 500m").
  - "find_peak": If the user wants the maximum/minimum or peak of a SINGLE measurement (e.g. "hottest spot in measurement 4", "what is the lowest strain in test 1?", "peak temperature").
  - "find_global_peak": If the user wants the global maximum/minimum or peak across the ENTIRE history/all measurements (e.g. "what is the highest temperature, what is the location in the cable and when its happen?", "overall peak temperature ever").
  - "metadata_info": If the user asks about specifications, location, interrogator model, etc. (e.g. "sensor info", "specifications", "where is it installed?").
  - "help": If they ask for help or how to use the bot.
- "measurement_index": Integer from 0 to {num_measurements - 1}. If the user queries a specific time or date (e.g. "at 12:05", "profile of trace at 2026-01-01 12:10"), extract that time/date string (e.g. "12:05" or "2026-01-01 12:10"). If they say "last", "latest", or "recent", use "latest". If not specified, default to "latest".
- "distance_m": Convert the distance to a float representing meters. E.g., "1.5km" or "1.5 kilometers" should be 1500.0.
- "peak_type": "max" for maximum/highest/peak/hottest, "min" for minimum/lowest/coldest.

Examples:
- "What is the peak temperature in scan {min(5, num_measurements - 1)}?" -> {{"quantity": "temperature", "analysis_type": "find_peak", "measurement_index": {min(5, num_measurements - 1)}, "distance_m": null, "peak_type": "max"}}
- "Whats the highest temperature, what is the location in the cable and when its happen" -> {{"quantity": "temperature", "analysis_type": "find_global_peak", "measurement_index": null, "distance_m": null, "peak_type": "max"}}
- "Plot the temperature profile of trace at 12:05" -> {{"quantity": "temperature", "analysis_type": "plot_profile", "measurement_index": "12:05", "distance_m": null, "peak_type": null}}
- "Plot deformation for the latest test" -> {{"quantity": "deformation", "analysis_type": "plot_profile", "measurement_index": "latest", "distance_m": null, "peak_type": null}}
- "What is the temperature at 3000 meters in test 0?" -> {{"quantity": "temperature", "analysis_type": "value_at_distance", "measurement_index": 0, "distance_m": 3000.0, "peak_type": null}}
- "Show strain history at 1200m" -> {{"quantity": "deformation", "analysis_type": "plot_history", "measurement_index": null, "distance_m": 1200.0, "peak_type": null}}
"""

    def extract(self, user_query):
        if self.provider == "Rules (Local/Fast)":
            return self._extract_regex(user_query)
        elif self.provider == "Google Gemini":
            return self._extract_gemini(user_query)
        elif self.provider == "Ollama (Local)":
            return self._extract_ollama(user_query)
        return None

    def _extract_regex(self, user_query):
        """Fallback deterministic parsing via Regex (supporting both English and Portuguese)."""
        query = user_query.lower().strip()
        
        quantity = None
        analysis_type = None
        measurement_index = "latest"
        distance_m = None
        peak_type = None
        
        # 1. Quantity
        if any(w in query for w in ["temp", "heat", "hot", "warm", "calor", "grau", "quente"]):
            quantity = "temperature"
        elif any(w in query for w in ["def", "strain", "stretch", "deform", "tension", "tensa", "tensã"]):
            quantity = "deformation"
        elif any(w in query for w in ["info", "metadata", "metadado", "local", "model", "sensor", "spec"]):
            quantity = "metadata"
            analysis_type = "metadata_info"
            
        # 2. Measurement index
        med_match = re.search(r'\b(?:measurement|medi[cç]ao|teste|test|scan|tempo|time)\b\s*(\d+)|\bt\s*(\d+)|\bt(\d+)\b', query)
        if med_match:
            measurement_index = int(next(g for g in med_match.groups() if g is not None))
        elif any(w in query for w in ["first", "initial", "primeir", "inicial"]):
            measurement_index = 0
        elif any(w in query for w in ["last", "latest", "recent", "ultim", "últim", "recente"]):
            measurement_index = "latest"
        else:
            # Check for time/timestamp match (e.g. 12:05, 12:05:00, 2026-01-01 12:05)
            time_match = re.search(r'\b(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}(?::\d{2})?|\d{2}:\d{2}(?::\d{2})?)\b', query)
            if time_match:
                measurement_index = time_match.group(1)
            
        # 3. Distance (m, km, meter, kilometer)
        dist_match = re.search(r'(\d+(?:[.,]\d+)?)\s*(m|meter|metro|km|kilometer|quilometro|kilometro|q?m)', query)
        if dist_match:
            val = float(dist_match.group(1).replace(',', '.'))
            unit = dist_match.group(2)
            if unit.startswith('k'):
                distance_m = val * 1000.0
            else:
                distance_m = val
        else:
            num_match = re.search(r'(?:at|in|on|no|ponto|em)\s*(\d+(?:[.,]\d+)?)', query)
            if num_match:
                distance_m = float(num_match.group(1).replace(',', '.'))
                
        if any(w in query for w in ["grafico", "gráfico", "plot", "plote", "plotar", "curve", "curva", "profile", "perfil"]):
            if any(w in query for w in ["history", "historico", "histórico", "evoluc", "evoluç", "over time", "along time", "along the time", "time series", "série temporal", "serie temporal", "ao longo do tempo"]) or (any(w in query for w in ["time", "tempo"]) and (measurement_index == "latest" or measurement_index is None)):
                analysis_type = "plot_history"
            else:
                analysis_type = "plot_profile"
        elif any(w in query for w in ["max", "peak", "highest", "hottest", "maior", "máx", "pico", "quente"]):
            if any(w in query for w in ["when", "happened", "history", "ever", "global", "all measurement", "overall", "todo", "sempre", "histórico", "aconteceu"]):
                analysis_type = "find_global_peak"
            else:
                analysis_type = "find_peak"
            peak_type = "max"
        elif any(w in query for w in ["min", "lowest", "coldest", "menor", "mín", "frio"]):
            if any(w in query for w in ["when", "happened", "history", "ever", "global", "all measurement", "overall", "todo", "sempre", "histórico", "aconteceu"]):
                analysis_type = "find_global_peak"
            else:
                analysis_type = "find_peak"
            peak_type = "min"
        elif distance_m is not None:
            if any(w in query for w in ["history", "historico", "histórico", "evoluc", "evoluç", "over time", "along time", "along the time", "time series", "série temporal", "serie temporal", "ao longo do tempo"]) or (any(w in query for w in ["time", "tempo"]) and (measurement_index == "latest" or measurement_index is None)):
                analysis_type = "plot_history"
            else:
                analysis_type = "value_at_distance"
                
        if any(w in query for w in ["help", "ajuda", "socorro", "how to use"]):
            analysis_type = "help"
            
        if quantity is None and analysis_type is None:
            return None
            
        return {
            "quantity": quantity,
            "analysis_type": analysis_type,
            "measurement_index": measurement_index,
            "distance_m": distance_m,
            "peak_type": peak_type
        }

    def _extract_gemini(self, user_query):
        if not self.api_key:
            raise Exception("Gemini API Key is not set. Please insert it in the sidebar.")
        
        try:
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(
                model_name=self.gemini_model,
                system_instruction=self.system_prompt
            )
            generation_config = genai.GenerationConfig(
                temperature=0.0,
                response_mime_type="application/json"
            )
            response = model.generate_content(
                user_query,
                generation_config=generation_config
            )
            return self._parse_json(response.text)
        except Exception as e:
            raise Exception(f"Gemini API error: {e}")

    def _extract_ollama(self, user_query):
        if not self.ollama_host:
            raise Exception("Ollama Host URL is not configured.")
        
        url = f"{self.ollama_host.rstrip('/')}/api/generate"
        payload = {
            "model": self.ollama_model or "llama3",
            "prompt": user_query,
            "system": self.system_prompt,
            "stream": False,
            "options": {
                "temperature": 0.0
            }
        }
        try:
            response = requests.post(url, json=payload, timeout=180)
            if response.status_code == 200:
                raw_text = response.json().get("response", "")
                return self._parse_json(raw_text)
            else:
                err_msg = ""
                try:
                    err_msg = f": {response.json().get('error')}"
                except:
                    pass
                raise Exception(f"Invalid response from Ollama (HTTP Status {response.status_code}){err_msg}")
        except Exception as e:
            raise Exception(f"Error connecting to Ollama at {url}: {e}")

    def _parse_json(self, raw_text):
        raw_text = raw_text.strip()
        if raw_text.startswith("```"):
            lines = raw_text.splitlines()
            content_lines = []
            for line in lines:
                if not line.startswith("```"):
                    content_lines.append(line)
            raw_text = "\n".join(content_lines)
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
        
        try:
            return json.loads(raw_text.strip())
        except json.JSONDecodeError:
            json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    pass
            raise Exception(f"The LLM did not return a valid JSON: {raw_text[:100]}...")


def generate_natural_language_response(user_query, analysis_type, retrieved_data, provider, api_key=None, gemini_model=None, ollama_host=None, ollama_model=None):
    """Generates a natural language response explaining the retrieved data using the selected AI provider."""
    prompt = f"""
You are an intelligent assistant for a Distributed Temperature and Strain Sensing (DTSS) system.
The user asked a question: "{user_query}"

Here is the data queried from the sensor's HDF5 file:
- Analysis Type: {analysis_type}
- Data Details: {json.dumps(retrieved_data, indent=2)}

Please answer the user's question directly, clearly, and concisely in natural language using the data above.
If the data shows anomalies (e.g. temperatures far from 25°C or high strain/deformation values), highlight them.
Maintain a professional and helpful engineering tone.
Respond directly in markdown (do not output JSON or code blocks unless relevant).
"""
    if provider == "Google Gemini" and api_key:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name=gemini_model or "gemini-1.5-flash")
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"*(AI Explanation Error: {e})*"
    elif provider == "Ollama (Local)" and ollama_host:
        url = f"{ollama_host.rstrip('/')}/api/generate"
        payload = {
            "model": ollama_model or "llama3",
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2}
        }
        try:
            response = requests.post(url, json=payload, timeout=180)
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                err_msg = ""
                try:
                    err_msg = f": {response.json().get('error')}"
                except:
                    pass
                return f"*(Ollama Explanation Error (HTTP Status {response.status_code}){err_msg})*"
        except Exception as e:
            return f"*(Ollama Explanation Error: {e})*"
    return None

# --- 2D & 3D RENDERING FUNCTIONS ---
def render_2d_analysis_module(struct, section_num, section_title, y_dataset, x_dataset="distances", ymin=None, ymax=None):
            from scipy.signal import butter, filtfilt, savgol_filter, medfilt
            import numpy as np
            import pandas as pd
            import h5py
            import plotly.graph_objects as go
    
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
                    arr = load_h5_dataset(st.session_state['file_path'], x_dataset)
                    if arr is not None and len(arr.shape) == 1:
                        x_mapping = arr
                except: pass
        
                if x_mapping is not None and len(x_mapping) > 1:
                    min_val, max_val = float(x_mapping[0]), float(x_mapping[-1])
                    step_val = float(x_mapping[1] - x_mapping[0])
                    if step_val <= 0: step_val = 1.0
                    st.markdown(f"**Physical Domain:** {min_val:.1f} to {max_val:.1f} (Array Size: {x_dim_size}, Step: {step_val:.2f})")
            
                    use_slider = st.checkbox("Use Slider (Quick Adjust)", value=True, key=f"x_slide_{section_num}")
                    if use_slider:
                        slice_m = st.slider("Select Physical Range:", min_value=min_val, max_value=max_val, value=(min_val, max_val), step=step_val, key=f"m_slide_{section_num}")
                    else:
                        c1, c2 = st.columns(2)
                        with c1: s_in = st.number_input("Start (Physical)", min_value=min_val, max_value=max_val, value=min_val, step=step_val, key=f"m_s_{section_num}")
                        with c2: e_in = st.number_input("End (Physical)", min_value=min_val, max_value=max_val, value=max_val, step=step_val, key=f"m_e_{section_num}")
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
                                    times_raw = load_h5_dataset(st.session_state['file_path'], 'start_times')
                                    if times_raw is not None:
                                        time_opts = [f"Trace {i+1} ➔ {datetime.fromtimestamp(t).strftime('%d/%m/%y %H:%M')}" for i, t in enumerate(times_raw)]
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
                
                    # Downsample for Plotly to maintain performance
                    MAX_POINTS = 1500
                    if len(data_x) > MAX_POINTS:
                        step = len(data_x) // MAX_POINTS
                        plot_x = data_x[::step]
                        plot_y = data_y[::step]
                        downsample_msg = f" (Downsampled for plot: {len(plot_x)} pts)"
                    else:
                        step = 1
                        plot_x = data_x
                        plot_y = data_y
                        downsample_msg = ""

                    fig = go.Figure()
                    alpha_orig = 0.4 if len(plots_to_make) > 0 else 1.0
                    fig.add_trace(go.Scatter(
                        x=plot_x, y=plot_y, mode='lines',
                        name=f'Raw Data',
                        line=dict(color=f'rgba(0, 100, 255, {alpha_orig})', dash='dot' if alpha_orig < 1.0 else 'solid')
                    ))
                    
                    color_map = {'tab:orange': 'orange', 'tab:green': 'green', 'tab:purple': 'purple', 'tab:red': 'red', 'tab:cyan': 'cyan'}
                    for pt in plots_to_make:
                        p_data = pt[0][::step] if step > 1 else pt[0]
                        c_str = color_map.get(pt[2], 'blue')
                        fig.add_trace(go.Scatter(
                            x=plot_x, y=p_data, mode='lines',
                            name=f'Filter: {pt[1]}',
                            line=dict(color=c_str, width=2)
                        ))
                        
                    label_map = {"temp_data": "Temperature (°C)", "strain_data": "Strain (uE)"}
                    display_y = label_map.get(y_dataset, y_dataset)
                    fig.update_layout(
                        title=f"2D Signal Profile: {section_title} {downsample_msg}",
                        xaxis_title=x_dataset, yaxis_title=display_y,
                        hovermode="x unified", template="plotly_dark",
                        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    fig.update_xaxes(showgrid=True, gridcolor='rgba(255, 255, 255, 0.3)', gridwidth=1, zeroline=True, zerolinecolor='rgba(255, 255, 255, 0.6)')
                    fig.update_yaxes(showgrid=True, gridcolor='rgba(255, 255, 255, 0.3)', gridwidth=1, zeroline=True, zerolinecolor='rgba(255, 255, 255, 0.6)')
                    
                    if ymin is not None and ymax is not None:
                        fig.update_yaxes(range=[ymin, ymax])
                        
                    st.plotly_chart(fig, use_container_width=True)
            
                    # Export
                    df_export = pd.DataFrame({x_dataset: data_x, y_dataset: data_y})
                    for pt in plots_to_make: df_export[pt[1]] = pt[0]
                    st.download_button(label="📥 Download Data as CSV", data=df_export.to_csv(index=False).encode('utf-8'), file_name=f"export_{y_dataset}.csv", mime="text/csv", key=f"csv_{section_num}")
            
                except Exception as e:
                    st.error(f"Plotting Error: {e}")


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
                                    dt = datetime.fromtimestamp(t_val)
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


def render_time_series_module(struct, section_num, section_title, y_dataset, distance_dataset="distances", ymin=None, ymax=None):
    from scipy.signal import butter, filtfilt, savgol_filter, medfilt
    import numpy as np
    import pandas as pd
    import h5py
    import plotly.graph_objects as go

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
            times_raw = load_h5_dataset(st.session_state['file_path'], 'start_times')
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
            dist_mapping = load_h5_dataset(st.session_state['file_path'], distance_dataset)
        except: pass
        
        if dist_mapping is not None and len(dist_mapping) >= num_distances:
            min_dist, max_dist = float(dist_mapping[0]), float(dist_mapping[num_distances-1])
            step_val = float(dist_mapping[1] - dist_mapping[0]) if num_distances > 1 else 1.0
            if step_val <= 0: step_val = 1.0
            st.markdown(f"**Physical Domain:** {min_dist:.1f} to {max_dist:.1f} m (Array Size: {num_distances}, Step: {step_val:.2f} m)")
            sel_dist = st.slider("📏 Select Distance (m):", min_value=min_dist, max_value=max_dist, value=min_dist, step=step_val, key=f"d_sel_{section_num}")
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
                    data_x = np.array([datetime.fromtimestamp(t) for t in times_raw[t_slice]])
                    x_label = "Time"
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
            MAX_POINTS = 1500
            if len(data_x) > MAX_POINTS:
                step = len(data_x) // MAX_POINTS
                plot_x = data_x[::step]
                plot_y = data_y[::step]
                downsample_msg = f" (Downsampled for plot: {len(plot_x)} pts)"
            else:
                step = 1
                plot_x = data_x
                plot_y = data_y
                downsample_msg = ""
                
            fig = go.Figure()
            alpha_orig = 0.4 if len(plots_to_make) > 0 else 1.0
            fig.add_trace(go.Scatter(
                x=plot_x, y=plot_y, mode='lines',
                name='Raw Time-Series',
                line=dict(color=f'rgba(0, 100, 255, {alpha_orig})')
            ))
            
            color_map = {'tab:orange': 'orange', 'tab:green': 'green', 'tab:purple': 'purple', 'tab:red': 'red', 'tab:cyan': 'cyan'}
            for pt in plots_to_make:
                p_data = pt[0][::step] if step > 1 else pt[0]
                c_str = color_map.get(pt[2], 'blue')
                fig.add_trace(go.Scatter(
                    x=plot_x, y=p_data, mode='lines',
                    name=f'Filter: {pt[1]}',
                    line=dict(color=c_str, width=2)
                ))
                
            label_map = {"temp_data": "Temperature (°C)", "strain_data": "Strain (uE)"}
            display_y = label_map.get(y_dataset, y_dataset)
            x_title = "Time (dd/mm/yy hh:mm)" if len(plot_x) > 0 and isinstance(plot_x[0], datetime) else x_label
            fig.update_layout(
                title=f"{section_title}{downsample_msg}",
                xaxis_title=x_title, yaxis_title=display_y,
                hovermode="x unified", template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            fig.update_xaxes(showgrid=True, gridcolor='rgba(255, 255, 255, 0.3)', gridwidth=1, zeroline=True, zerolinecolor='rgba(255, 255, 255, 0.6)')
            fig.update_yaxes(showgrid=True, gridcolor='rgba(255, 255, 255, 0.3)', gridwidth=1, zeroline=True, zerolinecolor='rgba(255, 255, 255, 0.6)')
            
            if len(plot_x) > 0 and isinstance(plot_x[0], datetime):
                fig.update_xaxes(tickformat="%d/%m/%y %H:%M")
                
            if ymin is not None and ymax is not None:
                fig.update_yaxes(range=[ymin, ymax])
                
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Plotting Error: {e}")


# --- UNIFIED SIDEBAR ---
st.sidebar.markdown("### 📁 1. Load File")
uploaded_file = st.sidebar.file_uploader("Upload HDF5 File")

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

@st.cache_data(show_spinner=False, ttl=10)
def get_ollama_models(host_url):
    try:
        response = requests.get(f"{host_url.rstrip('/')}/api/tags", timeout=1.5)
        if response.status_code == 200:
            models_data = response.json().get("models", [])
            return [m["name"] for m in models_data]
    except Exception:
        pass
    return []

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
        
        # Try to dynamically list models
        available_models = get_ollama_models(ollama_host)
        if available_models:
            default_index = 0
            for i, name in enumerate(available_models):
                if "llama3.1" in name.lower() or "llama-3.1" in name.lower():
                    default_index = i
                    break
                elif "llama3" in name.lower() or "llama-3" in name.lower():
                    default_index = i
            
            ollama_model = st.selectbox(
                "Model Name (Auto-detected)", 
                available_models, 
                index=default_index,
                help="Select one of the models currently downloaded in Ollama."
            )
        else:
            ollama_model = st.text_input(
                "Model Name", 
                value="llama3", 
                help="No active models detected. Type the name of the model installed in Ollama."
            )
            st.warning("⚠️ Could not retrieve local models list. Make sure Ollama is running.")

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


# --- MAIN PAGE ---
if 'file_path' in st.session_state:
    file_path = st.session_state['file_path']
    struct, msg = get_h5_structure(file_path)
    
    st.markdown('<h1 class="main-title">🧠 FEBUS HDF5 Viewer & Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Explore, slice, plot, and ask questions about the optical fiber HDF5 directly.</p>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "1. Metadata", 
        "2. Temp (2D)", 
        "3. Strain (2D)", 
        "4. Temp (2D-Time)",
        "5. Strain (2D-Time)",
        "6. Temp (3D)", 
        "7. Strain (3D)",
        "8. Intelligent Assistant"
    ])

    with tab1:

            # --- Section 1: Metadata and Structure ---
            with st.expander("View File Structure & Metadata", expanded=True):
                st.subheader("Global Attributes (Root)")
                if struct["attributes"]:
                    attrs_to_show = struct["attributes"].copy()
                    ordered_attrs = {}
                    
                    for key_time in ['start_time', 'end_time']:
                        raw_val = attrs_to_show.pop(key_time, None)
                        if raw_val is not None:
                            try:
                                ts = float(raw_val[0] if isinstance(raw_val, list) else raw_val)
                                ordered_attrs[key_time] = f"{ts} ---> ({datetime.fromtimestamp(ts).strftime('%d/%m/%Y %H:%M:%S')})"
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
        

        
    with tab2:

            render_2d_analysis_module(struct, section_num=2, section_title="Temperature 2D Analysis", y_dataset="temp_data", ymin=-50, ymax=200)
            
        
    with tab3:

            render_2d_analysis_module(struct, section_num=3, section_title="Strain 2D Analysis", y_dataset="strain_data", ymin=-2000, ymax=2000)

        # --- Helper for 3D Surface Plots ---
        
    with tab4:
        st.info("💡 **Temperature Evolution (Time-Series)**: Plots the temperature changes at a specific meter over the entire duration of the monitoring.")
        render_time_series_module(struct, 4, "Temperature Time-Series", "temp_data", ymin=-50, ymax=200)
        
    with tab5:
        st.info("💡 **Strain Evolution (Time-Series)**: Plots the strain changes at a specific meter over the entire duration of the monitoring.")
        render_time_series_module(struct, 5, "Strain Time-Series", "strain_data", ymin=-2000, ymax=2000)
        
    with tab6:

            st.header("4. 3D Surface Graph ( Temperature )")
            if "temp_data" in struct["datasets"]:
                st.write("Real-time 3D topographic visualization of Temperature.")
                if st.checkbox("Show / Generate Temperature 3D Surface", value=False):
                    plot_3d_surface(st.session_state['file_path'], "temp_data", "distances", "3D Surface Topography: Temperature", "Temperature (°C)", "Turbo", zmin=-50, zmax=200)
            else:
                 st.info("The 'temp_data' dataset was not found in this file.")

        
    with tab7:

            st.header("5. 3D Surface Graph ( Strain )")
            if "strain_data" in struct["datasets"]:
                st.write("Real-time 3D topographic visualization of Strain.")
                if st.checkbox("Show / Generate Strain 3D Surface", value=False):
                    plot_3d_surface(st.session_state['file_path'], "strain_data", "distances", "3D Surface Topography: Strain", "Strain (µe)", "Viridis", zmin=-2000, zmax=2000)
            else:
                 st.info("The 'strain_data' dataset was not found in this file.")
            

    with tab8:
        col1, col2 = st.columns([1.2, 1.0], gap="large")
        
        # COLUMN 2: INTERACTIVE VISUALIZATION PANEL
        with col2:
            st.markdown("### 🖥️ Interactive Fiber Viewer")
            st.write("Select parameters below to inspect the fiber profiles manually:")
            
            v_qty = st.selectbox("Physical Quantity", ["Temperature (°C)", "Deformation (µε)"], key="v_qty")
            v_idx = st.slider("Measurement Index", 0, metadata['num_measurements'] - 1, metadata['num_measurements'] - 1, key="v_idx")
            
            # Query HDF5 for manual view
            core = LlmBotCore(file_path)
            q_key = 'temperature' if "Temp" in v_qty else 'deformation'
            distances, temp_data, def_data, actual_idx = core.query_profile(q_key, v_idx)
            
            # Show active measurement time if available
            if metadata.get('start_times') is not None and actual_idx < len(metadata['start_times']):
                t_start = datetime.fromtimestamp(metadata['start_times'][actual_idx]).strftime('%Y-%m-%d %H:%M:%S')
                t_end = datetime.fromtimestamp(metadata['end_times'][actual_idx]).strftime('%Y-%m-%d %H:%M:%S')
                st.markdown(f"""
                <div style="background-color: #f0f4f8; padding: 10px 14px; border-radius: 8px; border-left: 4px solid #1A73E8; margin-bottom: 12px; font-size: 0.9rem;">
                    📅 <b>Measurement Time:</b> {t_start} — {t_end} (UTC)
                </div>
                """, unsafe_allow_html=True)
                
            y_data = temp_data if q_key == 'temperature' else def_data
            
            # Downsample for faster plotting in the UI
            max_plot_points = 1500
            step = max(1, len(distances) // max_plot_points)
            
            df_plot = pd.DataFrame({
                "Distance (m)": distances[::step],
                v_qty: y_data[::step]
            })
            
            color_hex = "#FF4B4B" if q_key == 'temperature' else "#1A73E8"
            
            st.line_chart(
                df_plot,
                x="Distance (m)",
                y=v_qty,
                color=color_hex,
                width="stretch"
            )
            
            # Summary stats for the active profile
            max_val = float(y_data.max())
            min_val = float(y_data.min())
            mean_val = float(y_data.mean())
            max_dist = float(distances[np.argmax(y_data)])
            min_dist = float(distances[np.argmin(y_data)])
            
            st.markdown("<p style='font-weight:600; font-size: 0.9rem; margin-bottom: 2px;'>Active Measurement Metrics:</p>", unsafe_allow_html=True)
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                st.metric("Maximum", f"{max_val:.2f}", f"at {max_dist:.1f} m", delta_color="off")
            with sc2:
                st.metric("Minimum", f"{min_val:.2f}", f"at {min_dist:.1f} m", delta_color="off")
            with sc3:
                st.metric("Average", f"{mean_val:.2f}", delta_color="off")
        
        
        # COLUMN 1: CHATBOT INTERFACE
        with col1:
            st.markdown("### 💬 Intelligent Assistant")
            
            # Initialize message history
            if "messages" not in st.session_state:
                st.session_state.messages = [
                    {
                        "role": "assistant",
                        "type": "text",
                        "content": "Hello! I am the **FEBUS Optics** intelligent assistant. I am ready to extract any information from the optical fiber HDF5 file.\n\n"
                                   "Try asking:\n"
                                   "- *'What is the maximum temperature in the last measurement?'*\n"
                                   "- *'Plot the strain in measurement 3'*\n"
                                   "- *'What is the temperature at 2500m in test 0?'*\n"
                                   "- *'Deformation evolution at 1.2km over time'*"
                    }
                ]
                
            # Render previous messages
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    if msg["type"] == "text":
                        st.markdown(msg["content"])
                    elif msg["type"] == "card":
                        st.markdown(msg["content"], unsafe_allow_html=True)
                    elif msg["type"] == "chart":
                        st.markdown(msg["content"]["text"])
                        st.line_chart(
                            data=msg["content"]["df"],
                            x=msg["content"]["x"],
                            y=msg["content"]["y"],
                            color=msg["content"]["color"]
                        )
                        
            # Process user input
            if user_input := st.chat_input("Ask the bot about the optical fiber..."):
                with st.chat_message("user"):
                    st.markdown(user_input)
                st.session_state.messages.append({"role": "user", "type": "text", "content": user_input})
                
                # Assistant response
                with st.chat_message("assistant"):
                    try:
                        # Initialize extractor
                        extractor = IntentExtractor(
                            provider=llm_provider,
                            api_key=api_key,
                            gemini_model=gemini_model if llm_provider == "Google Gemini" else None,
                            ollama_host=ollama_host,
                            ollama_model=ollama_model,
                            num_measurements=metadata.get('num_measurements', 8),
                            cable_length=metadata.get('cable_length_m', 10000.0),
                            num_channels=metadata.get('num_channels', 10000)
                        )
                        
                        with st.spinner("Interpreting query..."):
                            params = extractor.extract(user_input)
                            
                        if params is None:
                            resp_fail = "Sorry, I did not understand which physical quantity (temperature or strain) or analysis you want. Try rephrasing your question (e.g., 'what is the temperature at 500m?')."
                            st.markdown(resp_fail)
                            st.session_state.messages.append({"role": "assistant", "type": "text", "content": resp_fail})
                        else:
                            core = LlmBotCore(file_path)
                            
                            analysis_type = params.get("analysis_type")
                            qty = params.get("quantity")
                            idx_m = params.get("measurement_index")
                            dist = params.get("distance_m")
                            peak = params.get("peak_type")
                            
                            summary_data = None
                            
                            # 1. Help
                            if analysis_type == "help" or (qty is None and analysis_type is None):
                                help_text = (
                                    "I can help you query the following DTSS information from the HDF5 file:\n\n"
                                    "1. **Spatial Profile Charts (whole cable)**:\n"
                                    "   - *'Plot temperature profile at measurement 5'*\n"
                                    "   - *'Show deformation chart for the latest scan'*\n\n"
                                    "2. **Point Queries**:\n"
                                    "   - *'What is the temperature at 4300m in measurement 2?'*\n"
                                    "   - *'Strain value at 1.5km in the latest test'*\n\n"
                                    "3. **Peak & Extreme Values**:\n"
                                    "   - *'Where is the hottest spot in measurement 0?'*\n"
                                    "   - *'What is the minimum deformation in test 6?'*\n\n"
                                    "4. **Temporal History of a Point**:\n"
                                    "   - *'Show temperature history at point 3000 meters'*\n"
                                    "   - *'Deformation evolution at 5km over time'*"
                                )
                                st.markdown(help_text)
                                st.session_state.messages.append({"role": "assistant", "type": "text", "content": help_text})
                                
                            # 2. Sensor Metadata
                            elif analysis_type == "metadata_info" or qty == "metadata":
                                meta_text = (
                                    f"### 📋 Sensor Technical Specifications\n\n"
                                    f"- **Interrogator Model:** {metadata['interrogator_model']}\n"
                                    f"- **Monitoring Location:** {metadata['location']}\n"
                                    f"- **Fiber Extension:** {metadata['cable_length_m']:.2f} meters\n"
                                    f"- **Spatial Channels:** {metadata['num_channels']:,} points\n"
                                    f"- **Measurements in History:** {metadata['num_measurements']} scans\n"
                                    f"- **Optical Pulse Width:** {metadata['pulse_width_ns']} ns (spatial resolution)"
                                )
                                st.markdown(meta_text)
                                st.session_state.messages.append({"role": "assistant", "type": "text", "content": meta_text})
                                
                            # 3. Spatial Profile Charts (plot_profile)
                            elif analysis_type == "plot_profile":
                                distances, temp_data, def_data, actual_idx = core.query_profile(qty, idx_m)
                                
                                max_plot_points = 1500
                                step = max(1, len(distances) // max_plot_points)
                                
                                if qty == "temperature":
                                    df = pd.DataFrame({"Distance (m)": distances[::step], "Temperature (°C)": temp_data[::step]})
                                    y_col = "Temperature (°C)"
                                    color = "#FF4B4B"
                                    desc = f"I generated the **Temperature** profile for **Measurement {actual_idx}**:"
                                elif qty == "deformation":
                                    df = pd.DataFrame({"Distance (m)": distances[::step], "Deformation (µε)": def_data[::step]})
                                    y_col = "Deformation (µε)"
                                    color = "#1A73E8"
                                    desc = f"I generated the **Deformation** profile for **Measurement {actual_idx}**:"
                                else: # both
                                    df = pd.DataFrame({
                                        "Distance (m)": distances[::step], 
                                        "Temperature (°C)": temp_data[::step],
                                        "Deformation (µε)": def_data[::step]
                                    })
                                    y_col = ["Temperature (°C)", "Deformation (µε)"]
                                    color = ["#FF4B4B", "#1A73E8"]
                                    desc = f"I generated the **Temperature & Deformation** profiles for **Measurement {actual_idx}**:"
                                    
                                st.markdown(desc)
                                st.line_chart(df, x="Distance (m)", y=y_col, color=color)
                                
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "type": "chart",
                                    "content": {
                                        "text": desc,
                                        "df": df,
                                        "x": "Distance (m)",
                                        "y": y_col,
                                        "color": color
                                    }
                                })
                                
                                # Populate summary_data
                                if qty == "temperature":
                                    summary_data = {
                                        "quantity": qty,
                                        "measurement_index": actual_idx,
                                        "max_value": float(temp_data.max()),
                                        "max_distance_m": float(distances[np.argmax(temp_data)]),
                                        "min_value": float(temp_data.min()),
                                        "min_distance_m": float(distances[np.argmin(temp_data)]),
                                        "mean_value": float(temp_data.mean())
                                    }
                                elif qty == "deformation":
                                    summary_data = {
                                        "quantity": qty,
                                        "measurement_index": actual_idx,
                                        "max_value": float(def_data.max()),
                                        "max_distance_m": float(distances[np.argmax(def_data)]),
                                        "min_value": float(def_data.min()),
                                        "min_distance_m": float(distances[np.argmin(def_data)]),
                                        "mean_value": float(def_data.mean())
                                    }
                                else:
                                    summary_data = {
                                        "quantity": qty,
                                        "measurement_index": actual_idx,
                                        "temperature_summary": {
                                            "max_value": float(temp_data.max()),
                                            "max_distance_m": float(distances[np.argmax(temp_data)]),
                                            "min_value": float(temp_data.min()),
                                            "min_distance_m": float(distances[np.argmin(temp_data)]),
                                            "mean_value": float(temp_data.mean())
                                        },
                                        "deformation_summary": {
                                            "max_value": float(def_data.max()),
                                            "max_distance_m": float(distances[np.argmax(def_data)]),
                                            "min_value": float(def_data.min()),
                                            "min_distance_m": float(distances[np.argmin(def_data)]),
                                            "mean_value": float(def_data.mean())
                                        }
                                    }
                                
                            # 4. Point Query (value_at_distance)
                            elif analysis_type == "value_at_distance":
                                if dist is None:
                                    resp_err = "Please specify a distance to query (e.g., 'temperature at 1500m')."
                                    st.markdown(resp_err)
                                    st.session_state.messages.append({"role": "assistant", "type": "text", "content": resp_err})
                                else:
                                    act_d, temp_v, def_v, actual_idx = core.query_value_at_distance(qty, idx_m, dist)
                                    
                                    card_html = f"""
                                    <div class="result-card">
                                        <div class="result-header">📍 Spot Measurement at {act_d:.1f} m (Measurement {actual_idx})</div>
                                        <div class="result-body">
                                    """
                                    
                                    if temp_v is not None:
                                        card_html += f"🔥 <b>Temperature:</b> {temp_v:.2f} °C<br/>"
                                    if def_v is not None:
                                        card_html += f"🌀 <b>Deformation:</b> {def_v:.2f} µε<br/>"
                                        
                                    card_html += "</div></div>"
                                    
                                    st.markdown(card_html, unsafe_allow_html=True)
                                    st.session_state.messages.append({"role": "assistant", "type": "card", "content": card_html})
                                    
                                    summary_data = {
                                        "quantity": qty,
                                        "measurement_index": actual_idx,
                                        "requested_distance_m": dist,
                                        "actual_distance_m": act_d,
                                        "temperature_c": temp_v,
                                        "deformation_ue": def_v
                                    }
                                    
                            # 5. Peak/Extreme Values (find_peak)
                            elif analysis_type == "find_peak":
                                res, actual_idx = core.query_peak(qty, idx_m, peak or "max")
                                p_word = "maximum" if (peak or "max") == "max" else "minimum"
                                p_emoji = "🔥" if (peak or "max") == "max" else "❄️"
                                
                                card_html = f"""
                                <div class="result-card">
                                    <div class="result-header">📈 Peak {p_word.capitalize()} Detected (Measurement {actual_idx})</div>
                                    <div class="result-body">
                                """
                                
                                if 'temp' in res:
                                    t_val = res['temp']['value']
                                    t_dist = res['temp']['distance']
                                    card_html += f"{p_emoji} <b>Peak Temperature ({p_word}):</b> {t_val:.2f} °C at <b>{t_dist:.1f} m</b><br/>"
                                if 'def' in res:
                                    d_val = res['def']['value']
                                    d_dist = res['def']['distance']
                                    card_html += f"🌀 <b>Peak Deformation ({p_word}):</b> {d_val:.2f} µε at <b>{d_dist:.1f} m</b><br/>"
                                    
                                card_html += "</div></div>"
                                
                                st.markdown(card_html, unsafe_allow_html=True)
                                st.session_state.messages.append({"role": "assistant", "type": "card", "content": card_html})
                                
                                summary_data = {
                                    "quantity": qty,
                                    "measurement_index": actual_idx,
                                    "peak_type": peak or "max",
                                    "peak_results": res
                                }
                                
                            # 5b. Global Peak Query (find_global_peak)
                            elif analysis_type == "find_global_peak":
                                res = core.query_global_peak(qty, peak or "max")
                                p_word = "maximum" if (peak or "max") == "max" else "minimum"
                                p_emoji = "🔥" if (peak or "max") == "max" else "❄️"
                                
                                card_html = f"""
                                <div class="result-card">
                                    <div class="result-header">🌐 Global Peak {p_word.capitalize()} Detected</div>
                                    <div class="result-body">
                                """
                                
                                if 'temp' in res:
                                    t_val = res['temp']['value']
                                    t_dist = res['temp']['distance']
                                    t_idx = res['temp']['measurement_index']
                                    t_ts = res['temp']['timestamp']
                                    t_time_str = datetime.fromtimestamp(t_ts).strftime('%Y-%m-%d %H:%M:%S') if t_ts is not None else f"Index {t_idx}"
                                    card_html += f"{p_emoji} <b>Global Peak Temperature ({p_word}):</b> {t_val:.2f} °C at <b>{t_dist:.1f} m</b><br/>"
                                    card_html += f"📅 <b>Time occurred:</b> {t_time_str} (UTC) (Measurement {t_idx})<br/><br/>"
                                if 'def' in res:
                                    d_val = res['def']['value']
                                    d_dist = res['def']['distance']
                                    d_idx = res['def']['measurement_index']
                                    d_ts = res['def']['timestamp']
                                    d_time_str = datetime.fromtimestamp(d_ts).strftime('%Y-%m-%d %H:%M:%S') if d_ts is not None else f"Index {d_idx}"
                                    card_html += f"🌀 <b>Global Peak Deformation ({p_word}):</b> {d_val:.2f} µε at <b>{d_dist:.1f} m</b><br/>"
                                    card_html += f"📅 <b>Time occurred:</b> {d_time_str} (UTC) (Measurement {d_idx})<br/>"
                                    
                                card_html += "</div></div>"
                                
                                st.markdown(card_html, unsafe_allow_html=True)
                                st.session_state.messages.append({"role": "assistant", "type": "card", "content": card_html})
                                
                                summary_data = {
                                    "quantity": qty,
                                    "peak_type": peak or "max",
                                    "global_peak_results": res
                                }
                                
                            # 6. Historical Evolution (plot_history)
                            elif analysis_type == "plot_history":
                                if dist is None:
                                    resp_err = "Please specify a distance for the history query (e.g., 'temperature history at 3500m')."
                                    st.markdown(resp_err)
                                    st.session_state.messages.append({"role": "assistant", "type": "text", "content": resp_err})
                                else:
                                    act_d, temp_h, def_h = core.query_history(qty, dist)
                                    
                                    # Create time labels based on actual timestamps in HDF5 if available
                                    n_meas = len(temp_h) if temp_h is not None else len(def_h)
                                    time_desc = ""
                                    if metadata.get('start_times') is not None and len(metadata['start_times']) >= n_meas:
                                        start_dt = datetime.fromtimestamp(metadata['start_times'][0]).strftime('%Y-%m-%d %H:%M')
                                        end_dt = datetime.fromtimestamp(metadata['start_times'][-1]).strftime('%Y-%m-%d %H:%M')
                                        time_desc = f" from **{start_dt}** to **{end_dt}** (UTC)"
                                        
                                        times = [datetime.fromtimestamp(metadata['start_times'][i]).strftime('%H:%M') for i in range(n_meas)]
                                        med_labels = [f"M{i} ({times[i]})" for i in range(n_meas)]
                                    else:
                                        base_time = datetime.now() - timedelta(hours=n_meas)
                                        if n_meas <= 24:
                                            times = [(base_time + timedelta(hours=i)).strftime("%H:%M") for i in range(n_meas)]
                                            med_labels = [f"M{i} ({times[i]})" for i in range(n_meas)]
                                        else:
                                            med_labels = [f"M{i}" for i in range(n_meas)]
                                    
                                    if qty == "temperature":
                                        df = pd.DataFrame({"Measurement": med_labels, "Temperature (°C)": temp_h})
                                        y_col = "Temperature (°C)"
                                        color = "#FF4B4B"
                                        desc = f"**Temperature** evolution at **{act_d:.1f} meters** over {n_meas} measurements{time_desc}:"
                                    elif qty == "deformation":
                                        df = pd.DataFrame({"Measurement": med_labels, "Deformation (µε)": def_h})
                                        y_col = "Deformation (µε)"
                                        color = "#1A73E8"
                                        desc = f"**Deformation** evolution at **{act_d:.1f} meters** over {n_meas} measurements{time_desc}:"
                                    else: # both
                                        df = pd.DataFrame({
                                            "Measurement": med_labels, 
                                            "Temperature (°C)": temp_h,
                                            "Deformation (µε)": def_h
                                        })
                                        y_col = ["Temperature (°C)", "Deformation (µε)"]
                                        color = ["#FF4B4B", "#1A73E8"]
                                        desc = f"**Temperature & Deformation** evolution at **{act_d:.1f} meters** over {n_meas} measurements{time_desc}:"
                                        
                                    st.markdown(desc)
                                    st.line_chart(df, x="Measurement", y=y_col, color=color)
                                    
                                    st.session_state.messages.append({
                                        "role": "assistant",
                                        "type": "chart",
                                        "content": {
                                            "text": desc,
                                            "df": df,
                                            "x": "Measurement",
                                            "y": y_col,
                                            "color": color
                                        }
                                    })
                                    
                                    # Summarize history
                                    temp_summary = {
                                        "max": float(np.max(temp_h)), "min": float(np.min(temp_h)), "mean": float(np.mean(temp_h))
                                    } if temp_h is not None else None
                                    def_summary = {
                                        "max": float(np.max(def_h)), "min": float(np.min(def_h)), "mean": float(np.mean(def_h))
                                    } if def_h is not None else None
                                    
                                    summary_data = {
                                        "quantity": qty,
                                        "distance_m": act_d,
                                        "num_measurements": n_meas,
                                        "temperature_history_summary": temp_summary,
                                        "deformation_history_summary": def_summary
                                    }
                            else:
                                resp_unsupp = f"Query parsed but display format not supported: {params}"
                                st.markdown(resp_unsupp)
                                st.session_state.messages.append({"role": "assistant", "type": "text", "content": resp_unsupp})
                                
                            # Generate natural language explanation using LLM if selected
                            if summary_data is not None and llm_provider in ["Google Gemini", "Ollama (Local)"]:
                                with st.spinner("Generating AI explanation..."):
                                    explanation = generate_natural_language_response(
                                        user_query=user_input,
                                        analysis_type=analysis_type,
                                        retrieved_data=summary_data,
                                        provider=llm_provider,
                                        api_key=api_key,
                                        gemini_model=gemini_model,
                                        ollama_host=ollama_host,
                                        ollama_model=ollama_model
                                    )
                                if explanation:
                                    st.markdown(explanation)
                                    st.session_state.messages.append({"role": "assistant", "type": "text", "content": explanation})
                                    
                    except Exception as e:
                        resp_err = f"⚠️ Query error: {str(e)}"
                        st.error(resp_err)
                        st.session_state.messages.append({"role": "assistant", "type": "text", "content": resp_err})
                        
                    st.rerun()

else:
    st.info("Please wait while the file is loading or upload a new file from the sidebar.")
