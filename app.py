import streamlit as st
import h5py
import numpy as np
import pandas as pd
import json
import re
import requests
from datetime import datetime, timedelta
import google.generativeai as genai

# --- CONFIGURAÇÕES E ESTILOS ---
HDF5_FILE_PATH = "mock_febus_data_10k_rotating.h5"

st.set_page_config(
    page_title="FEBUS DTSS Intelligent Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização CSS Premium
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Gradiente para o título principal */
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
    
    /* Cards de Metadados na Barra Lateral */
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
    
    /* Card de Resultado da Consulta no Chat */
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


# --- PERSISTÊNCIA E LEITURA DO ARQUIVO HDF5 ---
@st.cache_resource
def get_sensor_metadata(file_path):
    """Carrega os metadados globais e dimensões do HDF5."""
    try:
        with h5py.File(file_path, 'r') as f:
            distances = f['distances'][:]
            num_measurements = f['extractedTemperature'].shape[0] if 'extractedTemperature' in f else 0
            
            # Carrega atributos, com fallbacks caso não existam
            interrogator = f.attrs.get("interrogator_model")
            if isinstance(interrogator, bytes):
                interrogator = interrogator.decode('utf-8')
            elif interrogator is None:
                interrogator = "FEBUS G2-R (Live Rotating)"
                
            location = f.attrs.get("location")
            if isinstance(location, bytes):
                location = location.decode('utf-8')
            elif location is None:
                location = "TS Conductor Mega Test Site"
                
            pulse_width = f.attrs.get("pulse_width_ns", 10.0)
            
            meta = {
                "interrogator_model": interrogator,
                "location": location,
                "pulse_width_ns": pulse_width,
                "cable_length_m": float(distances[-1]),
                "num_channels": len(distances),
                "num_measurements": num_measurements
            }
            return meta
    except Exception as e:
        return {"error": str(e)}


# Carrega metadados iniciais
metadata = get_sensor_metadata(HDF5_FILE_PATH)

if "error" in metadata:
    st.error(f"Erro crítico ao acessar o arquivo HDF5 '{HDF5_FILE_PATH}': {metadata['error']}")
    st.info("Certifique-se de que o arquivo HDF5 está na pasta raiz do projeto.")
    st.stop()


# --- LÓGICA DE QUERIES NO HDF5 ---
class LlmBotCore:
    """Núcleo de consultas diretas do HDF5."""
    def __init__(self, file_path):
        self.file_path = file_path

    def _parse_measurement_index(self, index):
        """Mapeia índices literais ou strings ('latest') para o índice correto (0 a 7)."""
        if index == "latest" or index is None:
            return 7
        try:
            val = int(index)
            if val < 0:
                return 0
            if val > 7:
                return 7
            return val
        except (ValueError, TypeError):
            return 7

    def query_profile(self, quantity, measurement_index):
        """Retorna distâncias e dados de temperatura/deformação de uma medição específica."""
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
        """Retorna os valores mais próximos no cabo óptico para uma distância fornecida."""
        with h5py.File(self.file_path, 'r') as f:
            distances = f['distances'][:]
            idx_m = self._parse_measurement_index(measurement_index)
            
            # Acha o índice da distância mais próxima do cabo
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
        """Retorna os valores máximos ou mínimos e em que distância ocorrem."""
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
        """Retorna o comportamento temporal (todas as 8 medições) em um ponto fixo da fibra."""
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


# --- ANÁLISE DE PERGUNTAS E INTENÇÕES (LLM & FALLBACK) ---
class IntentExtractor:
    """Interpreta perguntas do usuário para extrair intenções estruturadas (JSON)."""
    def __init__(self, provider, api_key=None, ollama_host=None, ollama_model=None):
        self.provider = provider
        self.api_key = api_key
        self.ollama_host = ollama_host
        self.ollama_model = ollama_model
        
        self.system_prompt = """
Você é um assistente especializado em analisar perguntas sobre um arquivo HDF5 contendo medições de um sensor óptico DTSS (temperatura e deformação distribuída).
O sensor possui:
- Distâncias (de 0 a 10.000 metros, totalizando 10.000 pontos de amostragem).
- Duas grandezas medidas: Temperatura (temperature) e Deformação/Deformation/Strain (deformation).
- 8 medições salvas (índices de 0 a 7, onde 7 é a última ou mais recente).

Sua tarefa é analisar a frase do usuário em português e extrair os parâmetros de consulta na forma de um objeto JSON válido.
Responda EXCLUSIVAMENTE com o objeto JSON. Não inclua Markdown (como ```json), explicações ou caracteres adicionais.

O JSON deve seguir exatamente a seguinte estrutura:
{
  "quantity": "temperature" | "deformation" | "both" | "metadata" | null,
  "analysis_type": "plot_profile" | "plot_history" | "value_at_distance" | "find_peak" | "metadata_info" | "help" | null,
  "measurement_index": 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | "latest" | null,
  "distance_m": float | null,
  "peak_type": "max" | "min" | null
}

Regras de Extração:
- "quantity": Escolha "temperature" se a pergunta for sobre temperatura, "deformation" se for sobre deformação ou strain (deformação), "both" se envolver ambos, ou "metadata" para informações gerais do sensor.
- "analysis_type":
  - "plot_profile": Se o usuário quer plotar, desenhar ou ver um gráfico do cabo inteiro (ex: "plote a temperatura do teste 2", "gráfico de deformação da medição 0").
  - "plot_history": Se o usuário quer ver o gráfico temporal ou histórico de um ponto específico (ex: "histórico no ponto 500m", "evolução da temperatura a 2500m ao longo do tempo").
  - "value_at_distance": Se o usuário quer saber o valor exato em uma distância específica (ex: "qual a temperatura a 1200 metros na medição 3?", "valor de deformação no ponto 500m").
  - "find_peak": Se o usuário quer saber o valor máximo/mínimo ou pico (ex: "onde está a maior temperatura da medição 4?", "qual a menor deformação no teste 1?", "máxima temperatura").
  - "metadata_info": Se o usuário quer saber metadados, modelo do interrogador, localização ou características do sensor (ex: "info do sensor", "metadados", "onde está instalado o sensor?").
  - "help": Se for uma saudação ou pedido de ajuda.
- "measurement_index": O índice da medição de 0 a 7. Se falarem "medição 3" ou "teste 3", use 3. Se falarem "primeira medição", use 0. Se falarem "última medição", "último teste" ou "recente", use "latest". Se não especificado e aplicável, use "latest".
- "distance_m": A distância numérica convertida para metros (float). Se disserem "1.5km" ou "1,5 quilômetros", converta para 1500.0.
- "peak_type": "max" se pedirem pelo máximo/maior/pico, ou "min" se pedirem pelo mínimo/menor.

Exemplos de Saída:
- "Qual a temperatura máxima na medição 5?" -> {"quantity": "temperature", "analysis_type": "find_peak", "measurement_index": 5, "distance_m": null, "peak_type": "max"}
- "Plote o gráfico da deformação para o último teste" -> {"quantity": "deformation", "analysis_type": "plot_profile", "measurement_index": "latest", "distance_m": null, "peak_type": null}
- "Qual o valor de temperatura no ponto de 3000 metros da medição 0?" -> {"quantity": "temperature", "analysis_type": "value_at_distance", "measurement_index": 0, "distance_m": 3000.0, "peak_type": null}
- "Mostre o histórico de deformação a 1200m" -> {"quantity": "deformation", "analysis_type": "plot_history", "measurement_index": null, "distance_m": 1200.0, "peak_type": null}
- "Quais são as especificações do sensor?" -> {"quantity": "metadata", "analysis_type": "metadata_info", "measurement_index": null, "distance_m": null, "peak_type": null}
"""

    def extract(self, user_query):
        if self.provider == "Regras (Local/Rápido)":
            return self._extract_regex(user_query)
        elif self.provider == "Google Gemini":
            return self._extract_gemini(user_query)
        elif self.provider == "Ollama (Local)":
            return self._extract_ollama(user_query)
        return None

    def _extract_regex(self, user_query):
        """Fallback deterministic parsing via Regex."""
        query = user_query.lower().strip()
        
        quantity = None
        analysis_type = None
        measurement_index = "latest"
        distance_m = None
        peak_type = None
        
        # 1. Quantity
        if any(w in query for w in ["temp", "calor", "grau", "quente"]):
            quantity = "temperature"
        elif any(w in query for w in ["def", "strain", "tensa", "tensã", "deforma"]):
            quantity = "deformation"
        elif any(w in query for w in ["info", "metadado", "local", "modelo", "sensor", "especific"]):
            quantity = "metadata"
            analysis_type = "metadata_info"
            
        # 2. Measurement index
        med_match = re.search(r'(?:medi[cç]ao|teste|scan)\s*(\d+)', query)
        if med_match:
            measurement_index = int(med_match.group(1))
        elif any(w in query for w in ["primeir", "inicial"]):
            measurement_index = 0
        elif any(w in query for w in ["ultim", "últim", "recente"]):
            measurement_index = "latest"
            
        # 3. Distance (m, km)
        dist_match = re.search(r'(\d+(?:[.,]\d+)?)\s*(m|metro|km|quilometro|kilometro|q?m)', query)
        if dist_match:
            val = float(dist_match.group(1).replace(',', '.'))
            unit = dist_match.group(2)
            if unit.startswith('k'):
                distance_m = val * 1000.0
            else:
                distance_m = val
        else:
            num_match = re.search(r'(?:a|em|no|ponto)\s*(\d+(?:[.,]\d+)?)', query)
            if num_match:
                distance_m = float(num_match.group(1).replace(',', '.'))
                
        # 4. Analysis type & peak type
        if any(w in query for w in ["grafico", "gráfico", "plote", "plotar", "curva", "perfil"]):
            if any(w in query for w in ["historico", "histórico", "tempo", "evoluc", "evoluç"]):
                analysis_type = "plot_history"
            else:
                analysis_type = "plot_profile"
        elif any(w in query for w in ["maior", "max", "máx", "pico", "quente"]):
            analysis_type = "find_peak"
            peak_type = "max"
        elif any(w in query for w in ["menor", "min", "mín", "frio"]):
            analysis_type = "find_peak"
            peak_type = "min"
        elif distance_m is not None:
            if any(w in query for w in ["historico", "histórico", "tempo", "evoluc", "evoluç"]):
                analysis_type = "plot_history"
            else:
                analysis_type = "value_at_distance"
                
        if any(w in query for w in ["ajuda", "socorro", "como usar", "help"]):
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
            raise Exception("Chave da API do Gemini não fornecida. Insira a chave na barra lateral.")
        
        try:
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(
                model_name='gemini-1.5-flash',
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
            raise Exception(f"Falha na API do Gemini: {e}")

    def _extract_ollama(self, user_query):
        if not self.ollama_host:
            raise Exception("URL do Ollama não configurada na barra lateral.")
        
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
                raise Exception(f"Retorno inválido do Ollama (Status HTTP {response.status_code})")
        except Exception as e:
            raise Exception(f"Erro de conexão com o Ollama em {url}: {e}")

    def _parse_json(self, raw_text):
        raw_text = raw_text.strip()
        # Remove eventuais delimitadores markdown de código
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
            raise Exception(f"A IA não retornou um JSON válido: {raw_text[:100]}...")


# --- INTERFACE VISUAL (STREAMLIT) ---

# BARRA LATERAL (CONFIGURAÇÕES E METADADOS DO SENSOR)
with st.sidebar:
    st.markdown("### ⚙️ Configuração da IA")
    llm_provider = st.selectbox(
        "Provedor LLM",
        ["Regras (Local/Rápido)", "Google Gemini", "Ollama (Local)"],
        index=0,
        help="Escolha 'Regras' para testar localmente de forma instantânea sem precisar de chaves de API."
    )
    
    api_key = None
    ollama_host = None
    ollama_model = None
    
    if llm_provider == "Google Gemini":
        api_key = st.text_input("Gemini API Key", type="password", help="Gere sua chave gratuita no Google AI Studio.")
        if not api_key:
            st.warning("⚠️ Insira a chave da API para habilitar o Gemini.")
    elif llm_provider == "Ollama (Local)":
        ollama_host = st.text_input("URL do Host Ollama", value="http://localhost:11434")
        ollama_model = st.text_input("Nome do Modelo", value="llama3")
        
    st.markdown("---")
    st.markdown("### 📊 Detalhes da Fibra Óptica (DTSS)")
    
    # Exibição estilizada de metadados
    st.markdown(f"""
    <div class="meta-card">
        <div class="meta-title">Modelo do Interrogador</div>
        <div class="meta-value">{metadata['interrogator_model']}</div>
    </div>
    <div class="meta-card">
        <div class="meta-title">Local de Instalação</div>
        <div class="meta-value">{metadata['location']}</div>
    </div>
    <div class="meta-card">
        <div class="meta-title">Comprimento do Cabo</div>
        <div class="meta-value">{metadata['cable_length_m']:.1f} m (10 km)</div>
    </div>
    <div class="meta-card">
        <div class="meta-title">Resolução Espacial</div>
        <div class="meta-value">{metadata['num_channels']:,} pontos de fibra</div>
    </div>
    <div class="meta-card">
        <div class="meta-title">Histórico Salvo</div>
        <div class="meta-value">{metadata['num_measurements']} medições consecutivas</div>
    </div>
    <div class="meta-card">
        <div class="meta-title">Largura de Pulso</div>
        <div class="meta-value">{metadata['pulse_width_ns']:.1f} ns</div>
    </div>
    """, unsafe_allow_html=True)


# TELA PRINCIPAL (LAYOUT LADO A LADO)
st.markdown('<h1 class="main-title">🧠 FEBUS DTSS Intelligent Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Faça perguntas sobre temperatura e deformação ao longo da fibra óptica. O bot responderá consultando o HDF5.</p>', unsafe_allow_html=True)

col1, col2 = st.columns([1.2, 1.0], gap="large")

# COLUNA 2: PAINEL DE VISUALIZAÇÃO INTERATIVO
with col2:
    st.markdown("### 🖥️ Visualizador de Fibra Interativo")
    st.write("Selecione os parâmetros abaixo para inspecionar os perfis da fibra manualmente:")
    
    v_qty = st.selectbox("Grandeza Física", ["Temperatura (°C)", "Deformação (µε)"], key="v_qty")
    v_idx = st.slider("Medição Temporal", 0, metadata['num_measurements'] - 1, metadata['num_measurements'] - 1, key="v_idx")
    
    # Consulta HDF5 para o dashboard manual
    core = LlmBotCore(HDF5_FILE_PATH)
    q_key = 'temperature' if "Temp" in v_qty else 'deformation'
    distances, temp_data, def_data, actual_idx = core.query_profile(q_key, v_idx)
    
    y_data = temp_data if q_key == 'temperature' else def_data
    df_plot = pd.DataFrame({
        "Distância (m)": distances,
        v_qty: y_data
    })
    
    color_hex = "#FF4B4B" if q_key == 'temperature' else "#1A73E8"
    
    st.line_chart(
        df_plot,
        x="Distância (m)",
        y=v_qty,
        color=color_hex,
        use_container_width=True
    )
    
    # Estatísticas resumidas da medição ativa
    max_val = float(y_data.max())
    min_val = float(y_data.min())
    mean_val = float(y_data.mean())
    max_dist = float(distances[np.argmax(y_data)])
    min_dist = float(distances[np.argmin(y_data)])
    
    st.markdown("<p style='font-weight:600; font-size: 0.9rem; margin-bottom: 2px;'>Métricas da Medição Selecionada:</p>", unsafe_allow_html=True)
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        st.metric("Máximo", f"{max_val:.2f}", f"em {max_dist:.1f} m", delta_color="off")
    with sc2:
        st.metric("Mínimo", f"{min_val:.2f}", f"em {min_dist:.1f} m", delta_color="off")
    with sc3:
        st.metric("Média Geral", f"{mean_val:.2f}", delta_color="off")


# COLUNA 1: CHATBOT E INTEGRAÇÃO DE INTELIGÊNCIA
with col1:
    st.markdown("### 💬 Assistente Inteligente")
    
    # Inicializa o histórico de conversa
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "type": "text",
                "content": "Olá! Sou o assistente inteligente da **FEBUS Optics**. Estou pronto para extrair qualquer informação do arquivo HDF5 de fibra óptica.\n\n"
                           "Experimente perguntar:\n"
                           "- *'Qual a maior temperatura na última medição?'*\n"
                           "- *'Plote a deformação na medição 3'*\n"
                           "- *'Qual a temperatura a 2500m no teste 0?'*\n"
                           "- *'Evolução da deformação a 1.2km ao longo do tempo'*"
            }
        ]
        
    # Renderiza mensagens anteriores do histórico
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
                
    # Processa nova entrada do usuário
    if user_input := st.chat_input("Pergunte ao bot sobre a fibra óptica..."):
        # Mostra mensagem do usuário imediatamente
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "type": "text", "content": user_input})
        
        # Resposta do assistente
        with st.chat_message("assistant"):
            try:
                # Inicializa extrator
                extractor = IntentExtractor(
                    provider=llm_provider,
                    api_key=api_key,
                    ollama_host=ollama_host,
                    ollama_model=ollama_model
                )
                
                with st.spinner("Interpretando pergunta..."):
                    params = extractor.extract(user_input)
                    
                if params is None:
                    resp_fail = "Desculpe, não entendi qual análise ou grandeza (temperatura ou deformação) você deseja. Tente reformular a pergunta (ex: 'qual a temperatura a 500m?')."
                    st.markdown(resp_fail)
                    st.session_state.messages.append({"role": "assistant", "type": "text", "content": resp_fail})
                else:
                    # Carrega motor de busca do HDF5
                    core = LlmBotCore(HDF5_FILE_PATH)
                    
                    analysis_type = params.get("analysis_type")
                    qty = params.get("quantity")
                    idx_m = params.get("measurement_index")
                    dist = params.get("distance_m")
                    peak = params.get("peak_type")
                    
                    # 1. Ajuda
                    if analysis_type == "help" or (qty is None and analysis_type is None):
                        help_text = (
                            "Posso ajudar você a consultar as seguintes informações sobre o sensor DTSS:\n\n"
                            "1. **Gráficos de Perfil (cabo inteiro)**:\n"
                            "   - *'Mostre o gráfico de temperatura na medição 5'*\n"
                            "   - *'Plote a deformação no último teste'*\n\n"
                            "2. **Consulta a Pontos Específicos**:\n"
                            "   - *'Qual a temperatura a 4300m na medição 2?'*\n"
                            "   - *'Qual a deformação a 1.5km na última medição?'*\n\n"
                            "3. **Picos e Valores Extremos**:\n"
                            "   - *'Qual o ponto mais quente da medição 0?'*\n"
                            "   - *'Onde ocorreu a menor deformação no teste 6?'*\n\n"
                            "4. **Histórico Temporal de um Ponto**:\n"
                            "   - *'Evolução da temperatura no ponto de 3000 metros'*\n"
                            "   - *'Mostre o histórico de deformação a 5km'*"
                        )
                        st.markdown(help_text)
                        st.session_state.messages.append({"role": "assistant", "type": "text", "content": help_text})
                        
                    # 2. Metadados do Sensor
                    elif analysis_type == "metadata_info" or qty == "metadata":
                        meta_text = (
                            f"### 📋 Especificações Técnicas do Sensor DTSS\n\n"
                            f"- **Modelo do Interrogador:** {metadata['interrogator_model']}\n"
                            f"- **Local de Monitoramento:** {metadata['location']}\n"
                            f"- **Extensão da Fibra:** {metadata['cable_length_m']:.2f} metros\n"
                            f"- **Número de Canais de Amostragem:** {metadata['num_channels']:,} pontos de fibra\n"
                            f"- **Medições no Histórico:** {metadata['num_measurements']} medições salvas\n"
                            f"- **Largura de Pulso Óptico:** {metadata['pulse_width_ns']} ns (resolução métrica)"
                        )
                        st.markdown(meta_text)
                        st.session_state.messages.append({"role": "assistant", "type": "text", "content": meta_text})
                        
                    # 3. Gráficos de Perfil Espacial (plot_profile)
                    elif analysis_type == "plot_profile":
                        distances, temp_data, def_data, actual_idx = core.query_profile(qty, idx_m)
                        
                        if qty == "temperature":
                            df = pd.DataFrame({"Distância (m)": distances, "Temperatura (°C)": temp_data})
                            y_col = "Temperatura (°C)"
                            color = "#FF4B4B"
                            desc = f"Gerei o perfil de **Temperatura** para a **Medição {actual_idx}**:"
                        elif qty == "deformation":
                            df = pd.DataFrame({"Distância (m)": distances, "Deformação (µε)": def_data})
                            y_col = "Deformação (µε)"
                            color = "#1A73E8"
                            desc = f"Gerei o perfil de **Deformação** para a **Medição {actual_idx}**:"
                        else: # both
                            df = pd.DataFrame({
                                "Distância (m)": distances, 
                                "Temperatura (°C)": temp_data,
                                "Deformação (µε)": def_data
                            })
                            y_col = ["Temperatura (°C)", "Deformação (µε)"]
                            color = ["#FF4B4B", "#1A73E8"]
                            desc = f"Gerei os perfis de **Temperatura e Deformação** para a **Medição {actual_idx}**:"
                            
                        st.markdown(desc)
                        st.line_chart(df, x="Distância (m)", y=y_col, color=color)
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "type": "chart",
                            "content": {
                                "text": desc,
                                "df": df,
                                "x": "Distância (m)",
                                "y": y_col,
                                "color": color
                            }
                        })
                        
                    # 4. Valor em uma distância específica (value_at_distance)
                    elif analysis_type == "value_at_distance":
                        if dist is None:
                            resp_err = "Para buscar o valor em um ponto específico, informe a distância (ex: 'temperatura no ponto 1500m')."
                            st.markdown(resp_err)
                            st.session_state.messages.append({"role": "assistant", "type": "text", "content": resp_err})
                        else:
                            act_d, temp_v, def_v, actual_idx = core.query_value_at_distance(qty, idx_m, dist)
                            
                            card_html = f"""
                            <div class="result-card">
                                <div class="result-header">📍 Medição no Ponto de {act_d:.1f} m (Medição {actual_idx})</div>
                                <div class="result-body">
                            """
                            
                            if temp_v is not None:
                                card_html += f"🔥 <b>Temperatura:</b> {temp_v:.2f} °C<br/>"
                            if def_v is not None:
                                card_html += f"🌀 <b>Deformação:</b> {def_v:.2f} µε<br/>"
                                
                            card_html += "</div></div>"
                            
                            st.markdown(card_html, unsafe_allow_html=True)
                            st.session_state.messages.append({"role": "assistant", "type": "card", "content": card_html})
                            
                    # 5. Localizar Pico Máximo ou Mínimo (find_peak)
                    elif analysis_type == "find_peak":
                        res, actual_idx = core.query_peak(qty, idx_m, peak or "max")
                        p_word = "máximo" if (peak or "max") == "max" else "mínimo"
                        p_emoji = "🔥" if (peak or "max") == "max" else "❄️"
                        
                        card_html = f"""
                        <div class="result-card">
                            <div class="result-header">📈 Pico {p_word.capitalize()} Detectado (Medição {actual_idx})</div>
                            <div class="result-body">
                        """
                        
                        if 'temp' in res:
                            t_val = res['temp']['value']
                            t_dist = res['temp']['distance']
                            card_html += f"{p_emoji} <b>Temperatura {p_word}a:</b> {t_val:.2f} °C na distância <b>{t_dist:.1f} m</b><br/>"
                        if 'def' in res:
                            d_val = res['def']['value']
                            d_dist = res['def']['distance']
                            card_html += f"🌀 <b>Deformação {p_word}a:</b> {d_val:.2f} µε na distância <b>{d_dist:.1f} m</b><br/>"
                            
                        card_html += "</div></div>"
                        
                        st.markdown(card_html, unsafe_allow_html=True)
                        st.session_state.messages.append({"role": "assistant", "type": "card", "content": card_html})
                        
                    # 6. Histórico Temporal de um ponto (plot_history)
                    elif analysis_type == "plot_history":
                        if dist is None:
                            resp_err = "Informe a distância para plotar o histórico temporal (ex: 'histórico de deformação a 3500m')."
                            st.markdown(resp_err)
                            st.session_state.messages.append({"role": "assistant", "type": "text", "content": resp_err})
                        else:
                            act_d, temp_h, def_h = core.query_history(qty, dist)
                            
                            # Cria carimbos de horas fictícias e realistas (espaçados de 1h) para fins estéticos de gráfico
                            base_time = datetime.now() - timedelta(hours=8)
                            times = [(base_time + timedelta(hours=i)).strftime("%H:%M") for i in range(8)]
                            med_labels = [f"M{i} ({times[i]})" for i in range(8)]
                            
                            if qty == "temperature":
                                df = pd.DataFrame({"Medição": med_labels, "Temperatura (°C)": temp_h})
                                y_col = "Temperatura (°C)"
                                color = "#FF4B4B"
                                desc = f"Evolução de **Temperatura** no ponto de **{act_d:.1f} metros** ao longo das 8 medições:"
                            elif qty == "deformation":
                                df = pd.DataFrame({"Medição": med_labels, "Deformação (µε)": def_h})
                                y_col = "Deformação (µε)"
                                color = "#1A73E8"
                                desc = f"Evolução de **Deformação** no ponto de **{act_d:.1f} metros** ao longo das 8 medições:"
                            else: # both
                                df = pd.DataFrame({
                                    "Medição": med_labels, 
                                    "Temperatura (°C)": temp_h,
                                    "Deformação (µε)": def_h
                                })
                                y_col = ["Temperatura (°C)", "Deformação (µε)"]
                                color = ["#FF4B4B", "#1A73E8"]
                                desc = f"Evolução de **Temperatura e Deformação** no ponto de **{act_d:.1f} metros**:"
                                
                            st.markdown(desc)
                            st.line_chart(df, x="Medição", y=y_col, color=color)
                            
                            st.session_state.messages.append({
                                "role": "assistant",
                                "type": "chart",
                                "content": {
                                    "text": desc,
                                    "df": df,
                                    "x": "Medição",
                                    "y": y_col,
                                    "color": color
                                }
                            })
                    else:
                        resp_unsupp = f"Consulta compreendida, mas o formato de exibição não é suportado: {params}"
                        st.markdown(resp_unsupp)
                        st.session_state.messages.append({"role": "assistant", "type": "text", "content": resp_unsupp})
                        
            except Exception as e:
                resp_err = f"⚠️ Erro ao executar consulta: {str(e)}"
                st.error(resp_err)
                st.session_state.messages.append({"role": "assistant", "type": "text", "content": resp_err})
                
            # Força o Streamlit a sincronizar a interface
            st.rerun()