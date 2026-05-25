import streamlit as st
import h5py
import numpy as np
import pandas as pd
import json
import re
import requests
from datetime import datetime, timedelta
import google.generativeai as genai

# --- CONFIGURATION AND UPLOAD ---
import os
DEFAULT_HDF5_PATH = "mock_febus_data_10k.h5" if os.path.exists("mock_febus_data_10k.h5") else ("mock_febus_data.h5" if os.path.exists("mock_febus_data.h5") else "mock_febus_data_10k_rotating.h5")
HDF5_FILE_PATH = DEFAULT_HDF5_PATH

# Render file uploader in sidebar
st.sidebar.markdown("### 📁 Load New File")
uploaded_file = st.sidebar.file_uploader(
    "Select an HDF5 file (.h5)", 
    type=["h5", "hdf5"],
    help="Upload your DTSS data file to query."
)

if uploaded_file is not None:
    # Save the file temporarily
    HDF5_FILE_PATH = f"uploaded_{uploaded_file.name}"
    try:
        with open(HDF5_FILE_PATH, "wb") as f:
            f.write(uploaded_file.getbuffer())
    except Exception as e:
        st.sidebar.error(f"Error saving file: {e}")

st.set_page_config(
    page_title="FEBUS DTSS Intelligent Assistant",
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


# --- HDF5 FILE READING ---
def get_sensor_metadata(file_path):
    """Loads global metadata and dimensions from the HDF5 file."""
    try:
        with h5py.File(file_path, 'r') as f:
            # Validate structure
            if "distances" not in f or "extractedTemperature" not in f or "extractedDeformation" not in f:
                return {
                    "error": "The HDF5 file does not have the expected structure (datasets 'distances', 'extractedTemperature', and 'extractedDeformation')."
                }
            
            distances = f['distances'][:]
            num_measurements = f['extractedTemperature'].shape[0]
            
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
metadata = get_sensor_metadata(HDF5_FILE_PATH)

if "error" in metadata:
    st.sidebar.error(f"File Error: {metadata['error']}")
    st.info("Please select an HDF5 file compatible with the DTSS structure (containing 'distances', 'extractedTemperature', and 'extractedDeformation').")
    st.stop()


# --- HDF5 QUERY LOGIC ---
class LlmBotCore:
    """Core interface for querying raw data from the HDF5 file."""
    def __init__(self, file_path):
        self.file_path = file_path
        try:
            with h5py.File(self.file_path, 'r') as f:
                self.num_measurements = f['extractedTemperature'].shape[0]
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
                with h5py.File(self.file_path, 'r') as f:
                    if 'start_times' in f:
                        start_times = f['start_times'][:]
                        
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
        with h5py.File(self.file_path, 'r') as f:
            distances = f['distances'][:]
            idx_m = self._parse_measurement_index(measurement_index)
            
            temp_data = None
            def_data = None
            
            if quantity in ['temperature', 'both']:
                temp_data = f['extractedTemperature'][idx_m, :]
            if quantity in ['deformation', 'both']:
                def_data = f['extractedDeformation'][idx_m, :]
                
        return distances, temp_data, def_data, idx_m

    def query_value_at_distance(self, quantity, measurement_index, target_distance):
        """Finds values closest to the target distance."""
        with h5py.File(self.file_path, 'r') as f:
            distances = f['distances'][:]
            idx_m = self._parse_measurement_index(measurement_index)
            
            # Find closest distance index
            dist_idx = (np.abs(distances - target_distance)).argmin()
            actual_distance = distances[dist_idx]
            
            temp_val = None
            def_val = None
            
            if quantity in ['temperature', 'both']:
                temp_val = float(f['extractedTemperature'][idx_m, dist_idx])
            if quantity in ['deformation', 'both']:
                def_val = float(f['extractedDeformation'][idx_m, dist_idx])
                
        return actual_distance, temp_val, def_val, idx_m

    def query_peak(self, quantity, measurement_index, peak_type='max'):
        """Finds max or min peak values and their spatial locations."""
        with h5py.File(self.file_path, 'r') as f:
            distances = f['distances'][:]
            idx_m = self._parse_measurement_index(measurement_index)
            
            res = {}
            if quantity in ['temperature', 'both']:
                data = f['extractedTemperature'][idx_m, :]
                p_idx = np.argmax(data) if peak_type == 'max' else np.argmin(data)
                res['temp'] = {
                    "value": float(data[p_idx]),
                    "distance": float(distances[p_idx])
                }
            if quantity in ['deformation', 'both']:
                data = f['extractedDeformation'][idx_m, :]
                p_idx = np.argmax(data) if peak_type == 'max' else np.argmin(data)
                res['def'] = {
                    "value": float(data[p_idx]),
                    "distance": float(distances[p_idx])
                }
        return res, idx_m

    def query_history(self, quantity, target_distance):
        """Returns the historical evolution (all 8 measurements) at a fixed distance."""
        with h5py.File(self.file_path, 'r') as f:
            distances = f['distances'][:]
            dist_idx = (np.abs(distances - target_distance)).argmin()
            actual_distance = distances[dist_idx]
            
            temp_history = None
            def_history = None
            
            if quantity in ['temperature', 'both']:
                temp_history = [float(x) for x in f['extractedTemperature'][:, dist_idx]]
            if quantity in ['deformation', 'both']:
                def_history = [float(x) for x in f['extractedDeformation'][:, dist_idx]]
                
        return actual_distance, temp_history, def_history

    def query_global_peak(self, quantity, peak_type='max'):
        """Finds the global maximum or minimum value, distance, and measurement index across all measurements."""
        with h5py.File(self.file_path, 'r') as f:
            distances = f['distances'][:]
            start_times = f['start_times'][:] if 'start_times' in f else None
            
            res = {}
            if quantity in ['temperature', 'both']:
                data = f['extractedTemperature'][:, :]
                flat_idx = np.argmax(data) if peak_type == 'max' else np.argmin(data)
                m_idx, d_idx = np.unravel_index(flat_idx, data.shape)
                
                timestamp = float(start_times[m_idx]) if start_times is not None else None
                res['temp'] = {
                    "value": float(data[m_idx, d_idx]),
                    "distance": float(distances[d_idx]),
                    "measurement_index": int(m_idx),
                    "timestamp": timestamp
                }
                
            if quantity in ['deformation', 'both']:
                data = f['extractedDeformation'][:, :]
                flat_idx = np.argmax(data) if peak_type == 'max' else np.argmin(data)
                m_idx, d_idx = np.unravel_index(flat_idx, data.shape)
                
                timestamp = float(start_times[m_idx]) if start_times is not None else None
                res['def'] = {
                    "value": float(data[m_idx, d_idx]),
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
            response = requests.post(url, json=payload, timeout=12)
            if response.status_code == 200:
                raw_text = response.json().get("response", "")
                return self._parse_json(raw_text)
            else:
                raise Exception(f"Invalid response from Ollama (HTTP Status {response.status_code})")
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
            response = requests.post(url, json=payload, timeout=12)
            if response.status_code == 200:
                return response.json().get("response", "")
        except Exception as e:
            return f"*(Ollama Explanation Error: {e})*"
    return None


# --- VISUAL INTERFACE (STREAMLIT) ---

# SIDEBAR (CONFIGURATIONS AND SENSOR METADATA)
with st.sidebar:
    st.markdown("### ⚙️ IA Configuration")
    llm_provider = st.selectbox(
        "LLM Provider",
        ["Rules (Local/Fast)", "Google Gemini", "Ollama (Local)"],
        index=1,
        help="Select the AI provider to interpret your questions. Choose 'Rules' for immediate local parser."
    )
    
    api_key = None
    gemini_model = None
    ollama_host = None
    ollama_model = None
    
    if llm_provider == "Google Gemini":
        # Check Streamlit secrets or environment variables for pre-configuration
        default_key = ""
        try:
            if "GEMINI_API_KEY" in st.secrets:
                default_key = st.secrets["GEMINI_API_KEY"]
            elif "gemini_api_key" in st.secrets:
                default_key = st.secrets["gemini_api_key"]
        except Exception:
            pass
            
        if not default_key:
            default_key = os.environ.get("GEMINI_API_KEY", "")
            
        if not default_key:
            # Fallback to local .env file
            if os.path.exists(".env"):
                try:
                    with open(".env", "r") as env_f:
                        for line in env_f:
                            if line.strip().startswith("GEMINI_API_KEY="):
                                default_key = line.strip().split("=", 1)[1].strip().strip('"').strip("'")
                                break
                except Exception:
                    pass
            
        api_key = st.text_input(
            "Gemini API Key", 
            value=default_key, 
            type="password", 
            help="Pre-configured via secrets or enter your API Key from Google AI Studio."
        )
        
        gemini_model = st.selectbox(
            "Gemini Model",
            ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.5-pro"],
            index=0,
            help="Select the Gemini model to use. If one model fails (e.g. 404), try another."
        )
        
        if not api_key:
            st.warning("⚠️ Enter your API Key to enable Gemini.")
    elif llm_provider == "Ollama (Local)":
        ollama_host = st.text_input("Ollama Host URL", value="http://localhost:11434")
        ollama_model = st.text_input("Model Name", value="llama3")
        
    st.markdown("---")
    st.markdown("### 📊 Fiber Optic Details (DTSS)")
    
    # Prepare monitoring period HTML
    time_range_html = ""
    if metadata.get('start_times') is not None and len(metadata['start_times']) > 0:
        start_dt = datetime.fromtimestamp(metadata['start_times'][0]).strftime('%Y-%m-%d %H:%M:%S')
        end_dt = datetime.fromtimestamp(metadata['end_times'][-1]).strftime('%Y-%m-%d %H:%M:%S')
        time_range_html = f"""
    <div class="meta-card">
        <div class="meta-title">Monitoring Period</div>
        <div class="meta-value" style="font-size: 0.8rem; font-weight: 600; line-height: 1.2; margin-top: 2px;">{start_dt}<br/>to<br/>{end_dt} (UTC)</div>
    </div>"""

    # Styled metadata cards
    st.markdown(f"""
    <div class="meta-card">
        <div class="meta-title">Interrogator Model</div>
        <div class="meta-value">{metadata['interrogator_model']}</div>
    </div>
    <div class="meta-card">
        <div class="meta-title">Location</div>
        <div class="meta-value">{metadata['location']}</div>
    </div>
    <div class="meta-card">
        <div class="meta-title">Cable Length</div>
        <div class="meta-value">{metadata['cable_length_m']:.1f} m</div>
    </div>
    <div class="meta-card">
        <div class="meta-title">Spatial Resolution</div>
        <div class="meta-value">{metadata['num_channels']:,} fiber points</div>
    </div>
    <div class="meta-card">
        <div class="meta-title">Saved History</div>
        <div class="meta-value">{metadata['num_measurements']} measurements</div>
    </div>{time_range_html}
    <div class="meta-card">
        <div class="meta-title">Pulse Width</div>
        <div class="meta-value">{metadata['pulse_width_ns']:.1f} ns</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    with st.expander("🔍 Complete HDF5 Schema & Attributes", expanded=True):
        st.markdown("<p style='font-size:0.85rem; font-weight:600; text-transform:uppercase; margin-bottom:5px; color:#555;'>Global Attributes</p>", unsafe_allow_html=True)
        if "all_global_attrs" in metadata and metadata["all_global_attrs"]:
            for k, v in metadata["all_global_attrs"].items():
                st.markdown(f"**`{k}`**: `{v}`")
        else:
            st.info("No global attributes found.")
            
        st.markdown("<p style='font-size:0.85rem; font-weight:600; text-transform:uppercase; margin-top:15px; margin-bottom:5px; color:#555;'>Datasets & Shapes</p>", unsafe_allow_html=True)
        if "datasets_info" in metadata and metadata["datasets_info"]:
            for ds_name, ds_meta in metadata["datasets_info"].items():
                st.markdown(f"📁 **`{ds_name}`**")
                st.markdown(f"- Shape: `{ds_meta['shape']}`")
                st.markdown(f"- Type: `{ds_meta['dtype']}`")
                if ds_meta["attrs"]:
                    for ak, av in ds_meta["attrs"].items():
                        st.markdown(f"  - `{ak}`: `{av}`")
        else:
            st.info("No datasets found.")


# MAIN PAGE (SIDE-BY-SIDE LAYOUT)
st.markdown('<h1 class="main-title">🧠 FEBUS DTSS Intelligent Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Ask questions about temperature and strain along the optical fiber. The bot queries the HDF5 directly.</p>', unsafe_allow_html=True)

col1, col2 = st.columns([1.2, 1.0], gap="large")

# COLUMN 2: INTERACTIVE VISUALIZATION PANEL
with col2:
    st.markdown("### 🖥️ Interactive Fiber Viewer")
    st.write("Select parameters below to inspect the fiber profiles manually:")
    
    v_qty = st.selectbox("Physical Quantity", ["Temperature (°C)", "Deformation (µε)"], key="v_qty")
    v_idx = st.slider("Measurement Index", 0, metadata['num_measurements'] - 1, metadata['num_measurements'] - 1, key="v_idx")
    
    # Query HDF5 for manual view
    core = LlmBotCore(HDF5_FILE_PATH)
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
    df_plot = pd.DataFrame({
        "Distance (m)": distances,
        v_qty: y_data
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
                    core = LlmBotCore(HDF5_FILE_PATH)
                    
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
                        
                        if qty == "temperature":
                            df = pd.DataFrame({"Distance (m)": distances, "Temperature (°C)": temp_data})
                            y_col = "Temperature (°C)"
                            color = "#FF4B4B"
                            desc = f"I generated the **Temperature** profile for **Measurement {actual_idx}**:"
                        elif qty == "deformation":
                            df = pd.DataFrame({"Distance (m)": distances, "Deformation (µε)": def_data})
                            y_col = "Deformation (µε)"
                            color = "#1A73E8"
                            desc = f"I generated the **Deformation** profile for **Measurement {actual_idx}**:"
                        else: # both
                            df = pd.DataFrame({
                                "Distance (m)": distances, 
                                "Temperature (°C)": temp_data,
                                "Deformation (µε)": def_data
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