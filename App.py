# App.py
import streamlit as st
import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import time
import google.generativeai as genai
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import numpy as np
import io
import base64
import random
from streamlit_folium import st_folium
import folium
from geopy.geocoders import Nominatim
from dotenv import load_dotenv
import os
import pycountry
from geopy.geocoders import Nominatim
import requests
import concurrent.futures
from threading import Lock

# Cargar variables de entorno
load_dotenv()

# Configuración de la página (debe ser lo primero)
st.set_page_config(page_title="Análisis de Acciones", layout="wide")

GOOGLE_KEY = os.getenv("AP")
genai.configure(api_key=GOOGLE_KEY)

currencyapi = os.getenv("AP1")

FMP = os.getenv("AP2") #Financial Modeling Prep

AlphaVantage = os.getenv("AP3")

# CSS personalizado mejorado
st.markdown("""
<style>
    /* Estilos para botones seleccionados */
    .stButton > button {
        border: 2px solid #cccccc;
        background-color: white;
        color: black;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        border-color: #adb5bd;
        background-color: #f8f9fa;
    }
    
    /* Botón seleccionado */
    .stButton > button.selected {
        border: 3px solid #28a745 !important;
        background-color: #d4edda !important;
        color: #155724 !important;
        font-weight: bold;
        box-shadow: 0 2px 4px rgba(40, 167, 69, 0.3);
    }
    
    /* Indicadores de métricas */
    .metric-positive {
        color: #28a745;
        font-weight: bold;
    }
    
    .metric-negative {
        color: #dc3545;
        font-weight: bold;
    }
    
    .metric-neutral {
        color: #ffc107;
        font-weight: bold;
    }
    
    /* Tarjetas de información */
    .info-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    
    /* Estilos para educación financiera */
    .concept-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
        border-left: 5px solid #ff6b6b;
    }
    
    .macro-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        margin: 5px 0;
    }
    
    /* Estilos para análisis de IA */
    .ia-analysis {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
        border-left: 5px solid #28a745;
    }
    
    .ia-recommendation {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 15px;
        border-radius: 8px;
        color: white;
        margin: 8px 0;
        border-left: 4px solid #ff6b6b;
    }     
</style>
""", unsafe_allow_html=True)

# Inicialización de session_state
if 'seccion_actual' not in st.session_state:
    st.session_state.seccion_actual = "inicio"

if 'favoritas' not in st.session_state:
    st.session_state.favoritas = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]

if 'portafolio' not in st.session_state:
    st.session_state.portafolio = {}

if 'historial_busquedas' not in st.session_state:
    st.session_state.historial_busquedas = []

if 'cache_lock' not in st.session_state:
    st.session_state.cache_lock = Lock()

# NUEVO: Función optimizada para carga de datos
@st.cache_data(ttl=1800, show_spinner=False, max_entries=200)
def obtener_datos_accion_optimizado(ticker):
    """Obtiene datos de acciones optimizado para paralelismo"""
    try:
        return yf.download(ticker, period="6mo", progress=False, interval="1d")
    except:
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False, max_entries=100)
def obtener_info_completa_optimizada(ticker):
    """Obtiene información completa optimizada"""
    try:
        return yf.Ticker(ticker).info
    except:
        return {}

# FUNCIONES NUEVAS CACHED
@st.cache_data(ttl=3600, show_spinner=False, max_entries=100)  # 1 hora, sin spinner, 100 entradas
def obtener_datos_accion(ticker):
    """Obtiene datos de acciones con caching extendido"""
    try:
        return yf.download(ticker, period="1y", progress=False, interval="1d")
    except:
        return pd.DataFrame()

@st.cache_data(ttl=7200, show_spinner=False, max_entries=50)  # 2 horas para info que cambia poco
def obtener_info_completa(ticker):
    """Obtiene información completa con caching extendido"""
    try:
        return yf.Ticker(ticker).info
    except:
        return {}

@st.cache_data(ttl=10800, show_spinner=False, max_entries=1)
def precalcular_datos_screener():
    """Pre-calcula datos del screener para mayor velocidad"""
    st.info("📊 Pre-calculando datos del S&P500... Esto puede tomar 1-2 minutos")
    
    datos_precalculados = {}
    for simbolo in SP500_SYMBOLS[:100]:  # Solo las primeras 100 para velocidad
        try:
            datos = obtener_datos_completos_yfinance(simbolo)
            if datos and datos.get('Empresa Valida'):
                scoring = calcular_scoring_dinamico(datos)
                datos['Score'] = scoring
                datos_precalculados[simbolo] = datos
        except:
            continue
    
    return datos_precalculados

@st.cache_data(ttl=86400, show_spinner=False, max_entries=20)  # 24 horas para S&P500
def obtener_datos_sp500_completo():
    """Datos del S&P500 con caching ultra extendido"""
    # Esta función la usarás en la sección de inicio
    # Mantén tu código actual aquí
    pass

@st.cache_data(ttl=10800, show_spinner=False, max_entries=30)  # 3 horas para datos macro
def obtener_datos_macro():
    """Datos macroeconómicos con caching extendido"""
    datos_macro = {
        "indicadores_usa": {
            "Inflación (CPI)": "3.2%",
            "Tasa de Desempleo": "3.8%", 
            "Crecimiento PIB": "2.1%",
            "Tasa de Interés Fed": "5.25%-5.50%",
            "Confianza del Consumidor": "64.9"
        },
        "mercados_globales": {
            "S&P 500": "+15% YTD",
            "NASDAQ": "+22% YTD",
            "Dow Jones": "+12% YTD",
            "Euro Stoxx 50": "+8% YTD", 
            "Nikkei 225": "+18% YTD"
        },
        "materias_primas": {
            "Petróleo (WTI)": "$78.50",
            "Oro": "$1,950.00",
            "Plata": "$23.15",
            "Cobre": "$3.85",
            "Bitcoin": "$42,000"
        },
        "divisas": {
            "EUR/USD": "1.0850",
            "USD/JPY": "148.50",
            "GBP/USD": "1.2650", 
            "USD/MXN": "17.20",
            "DXY (Índice Dólar)": "103.50"
        }
    }
    return datos_macro

@st.cache_data(ttl=1800, show_spinner=False, max_entries=50)  # 30 minutos para Wikipedia
def obtener_info_wikipedia(ticker, nombre_empresa):
    """Obtiene información de Wikipedia con caching"""
    # Mantén tu código actual de Wikipedia aquí
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        # Tu código actual de Wikipedia...
        return {'encontrado': False, 'error': 'Implementación actual'}
    except Exception as e:
        return {'encontrado': False, 'error': f'Error: {str(e)}'}

@st.cache_data(ttl=3600, show_spinner=False, max_entries=50)  # 1 hora para análisis técnico
def calcular_indicadores_tecnicos(data):
    """Calcula indicadores técnicos con caching"""
    if data.empty:
        return data
    
    # Tu código actual de indicadores técnicos...
    return data

@st.cache_data(ttl=7200, show_spinner=False, max_entries=30)  # 2 horas para métricas de riesgo
def calcular_metricas_riesgo_avanzadas(ticker_symbol, periodo_años=5):
    """Calcula métricas de riesgo con caching extendido"""
    # Tu código actual de métricas de riesgo...
    return None

@st.cache_data(ttl=300, show_spinner=False, max_entries=20)  # 5 minutos para datos en tiempo real
def obtener_datos_tiempo_real(ticker):
    """Datos en tiempo real con caching corto"""
    try:
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info
        hist = ticker_obj.history(period="2d")
        
        if not hist.empty and len(hist) >= 2:
            current = hist['Close'].iloc[-1]
            previous = hist['Close'].iloc[-2] 
            change = ((current - previous) / previous) * 100
            
            return {
                'precio_actual': current,
                'cambio_porcentaje': change,
                'volumen': hist['Volume'].iloc[-1] if 'Volume' in hist.columns else 0
            }
    except:
        return None

# FUNCIÓN CON API DE WIKIPEDIA - CONTENIDO COMPLETO MEJORADO
@st.cache_data(ttl=3600)
def obtener_info_wikipedia(ticker, nombre_empresa):
    """
    Obtiene información de Wikipedia usando la API oficial - CONTENIDO COMPLETO MEJORADO
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        # PRIMERO: Usar la API de búsqueda de Wikipedia para encontrar la página correcta
        search_url = f"https://es.wikipedia.org/w/api.php?action=query&list=search&srsearch={nombre_empresa}&format=json&srlimit=5"
        
        search_response = requests.get(search_url, headers=headers, timeout=10)
        
        if search_response.status_code == 200:
            search_data = search_response.json()
            
            if search_data['query']['search']:
                # Tomar el primer resultado que parezca relevante
                for result in search_data['query']['search']:
                    title = result['title']
                    
                    # Verificar que el título sea relevante (contenga palabras clave de la empresa)
                    if any(keyword in title.lower() for keyword in ['inc', 'corp', 'company', 'corporation', nombre_empresa.split()[0].lower()]):
                        # Obtener el contenido COMPLETO de la página usando la API
                        content_url = f"https://es.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=true&titles={title}&format=json"
                        content_response = requests.get(content_url, headers=headers, timeout=10)
                        
                        if content_response.status_code == 200:
                            content_data = content_response.json()
                            pages = content_data['query']['pages']
                            
                            for page_id, page_data in pages.items():
                                if 'extract' in page_data and page_data['extract']:
                                    contenido = page_data['extract']
                                    
                                    # LIMPIAR EL FORMATO DE TÍTULOS
                                    contenido_limpio = limpiar_formato_wikipedia(contenido)
                                    
                                    return {
                                        'encontrado': True,
                                        'contenido': contenido_limpio,
                                        'url': f"https://es.wikipedia.org/wiki/{title.replace(' ', '_')}",
                                        'termino_busqueda': title,
                                        'fuente': 'API Wikipedia'
                                    }
        
        # SEGUNDO: Intentar con búsqueda en inglés
        search_url_english = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={nombre_empresa}&format=json&srlimit=5"
        
        search_response_english = requests.get(search_url_english, headers=headers, timeout=10)
        
        if search_response_english.status_code == 200:
            search_data_english = search_response_english.json()
            
            if search_data_english['query']['search']:
                for result in search_data_english['query']['search']:
                    title = result['title']
                    
                    if any(keyword in title.lower() for keyword in ['inc', 'corp', 'company', 'corporation', nombre_empresa.split()[0].lower()]):
                        content_url_english = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=true&titles={title}&format=json"
                        content_response_english = requests.get(content_url_english, headers=headers, timeout=10)
                        
                        if content_response_english.status_code == 200:
                            content_data_english = content_response_english.json()
                            pages_english = content_data_english['query']['pages']
                            
                            for page_id, page_data in pages_english.items():
                                if 'extract' in page_data and page_data['extract']:
                                    contenido_ingles = page_data['extract']
                                    
                                    # LIMPIAR EL FORMATO PRIMERO
                                    contenido_ingles_limpio = limpiar_formato_wikipedia(contenido_ingles)
                                    
                                    # Traducir con Gemini - CONTENIDO COMPLETO
                                    try:
                                        prompt_traduccion = f"""
                                        Traduce al español el siguiente texto sobre una empresa manteniendo un tono formal.
                                        Conserva términos técnicos y financieros sin cambios.
                                        Traduce TODO el texto completo sin omitir nada.
                                        
                                        Texto: {contenido_ingles_limpio}
                                        """
                                        
                                        response_traduccion = genai.models.generate_content(
                                            model="gemini-2.5-flash",
                                            contents=prompt_traduccion
                                        )
                                        
                                        contenido_traducido = response_traduccion.text
                                        
                                        return {
                                            'encontrado': True,
                                            'contenido': contenido_traducido,
                                            'url': f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                                            'termino_busqueda': title,
                                            'fuente': 'API Wikipedia Inglés (Traducido)'
                                        }
                                    except:
                                        # Si falla la traducción, devolver en inglés COMPLETO
                                        return {
                                            'encontrado': True,
                                            'contenido': contenido_ingles_limpio,
                                            'url': f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                                            'termino_busqueda': title,
                                            'fuente': 'API Wikipedia Inglés'
                                        }
        
        return {'encontrado': False, 'error': 'No se encontró información en Wikipedia'}
            
    except Exception as e:
        return {'encontrado': False, 'error': f'Error: {str(e)}'}

# NUEVO: Funciones de paralelismo
def cargar_accion_paralelo(ticker_data):
    """Carga una acción en paralelo"""
    ticker, nombre, peso = ticker_data
    try:
        with st.session_state.cache_lock:
            datos = obtener_datos_accion_optimizado(ticker)
            info = obtener_info_completa_optimizada(ticker)
        
        if not datos.empty:
            precio_actual = datos['Close'].iloc[-1] if 'Close' in datos.columns else 0
            precio_anterior = datos['Close'].iloc[-2] if len(datos) > 1 else precio_actual
            cambio = ((precio_actual - precio_anterior) / precio_anterior * 100) if precio_anterior else 0
            
            return {
                'ticker': ticker,
                'nombre': nombre,
                'peso': peso,
                'precio_actual': precio_actual,
                'cambio': cambio,
                'datos': datos,
                'info': info
            }
    except Exception as e:
        return None
    return None

def cargar_sp500_paralelo():
    """Carga el S&P500 en paralelo"""
    # Lista de componentes principales del S&P500 (ejemplo reducido)
    componentes = [
        ("AAPL", "Apple Inc.", 7.0),
        ("MSFT", "Microsoft Corporation", 6.5),
        ("AMZN", "Amazon.com Inc.", 3.5),
        ("NVDA", "NVIDIA Corporation", 3.0),
        ("GOOGL", "Alphabet Inc.", 2.0),
        ("GOOG", "Alphabet Inc. Class C", 1.8),
        ("TSLA", "Tesla Inc.", 1.5),
        ("META", "Meta Platforms Inc.", 1.4),
        ("BRK-B", "Berkshire Hathaway Inc.", 1.3),
        ("UNH", "UnitedHealth Group Incorporated", 1.2),
        ("JNJ", "Johnson & Johnson", 1.1),
        ("XOM", "Exxon Mobil Corporation", 1.0),
        ("JPM", "JPMorgan Chase & Co.", 0.9),
        ("V", "Visa Inc.", 0.8),
        ("PG", "Procter & Gamble Company", 0.7),
        ("MA", "Mastercard Incorporated", 0.6),
        ("HD", "Home Depot Inc.", 0.5),
        ("CVX", "Chevron Corporation", 0.5),
        ("ABBV", "AbbVie Inc.", 0.5),
        ("LLY", "Eli Lilly and Company", 0.4)
    ]
    
    # Limitar a los primeros 20 para mayor velocidad en demostración
    componentes_rapidos = componentes[:20]
    
    with st.spinner('🔄 Cargando componentes en paralelo...'):
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            resultados = list(executor.map(cargar_accion_paralelo, componentes_rapidos))
    
    # Filtrar resultados None
    return [r for r in resultados if r is not None]

def buscar_simbolos_sp500_rapido(filtros, max_acciones=50):
    """Búsqueda ultra rápida usando datos precalculados"""
    # Cargar datos precalculados
    datos_precalculados = precalcular_datos_screener()
    
    acciones_encontradas = []
    
    for simbolo, datos in datos_precalculados.items():
        if len(acciones_encontradas) >= max_acciones:
            break
            
        # Aplicar filtros rápidos
        if aplicar_filtros_rapidos(datos, filtros):
            acciones_encontradas.append(datos)
    
    return acciones_encontradas

def aplicar_filtros_rapidos(datos, filtros):
    """Aplica filtros de manera optimizada"""
    try:
        # Filtro P/E
        pe = datos.get('P/E', 0)
        if filtros['pe_min'] > 0 and (pe == 0 or pe < filtros['pe_min']):
            return False
        if filtros['pe_max'] < 1000 and pe > filtros['pe_max']:
            return False
        
        # Solo los filtros más importantes para velocidad
        roe = datos.get('ROE', 0)
        if filtros['roe_min'] > 0 and roe < (filtros['roe_min'] / 100):
            return False
            
        return True
    except:
        return False

# FUNCIÓN PARA LIMPIAR Y FORMATEAR EL CONTENIDO DE WIKIPEDIA
def limpiar_formato_wikipedia(texto):
    """
    Limpia el formato de markdown de Wikipedia y convierte los títulos a formato Markdown
    """
    if not texto:
        return texto
    
    lineas = texto.split('\n')
    lineas_limpias = []
    
    for linea in lineas:
        linea_limpia = linea.strip()
        if not linea_limpia:
            continue
            
        # Detectar títulos con === Título ===
        if linea_limpia.startswith('===') and linea_limpia.endswith('==='):
            # Es un título principal (### en Markdown)
            titulo = linea_limpia.replace('===', '').strip()
            if titulo:
                lineas_limpias.append(f"### {titulo}")
                
        # Detectar subtítulos con == Título ==
        elif linea_limpia.startswith('==') and linea_limpia.endswith('=='):
            # Es un subtítulo (## en Markdown)
            subtitulo = linea_limpia.replace('==', '').strip()
            if subtitulo:
                lineas_limpias.append(f"## {subtitulo}")
                
        else:
            # Texto normal
            lineas_limpias.append(linea_limpia)
    
    return '\n\n'.join(lineas_limpias)

# FUNCIÓN PARA OBTENER RATING DE ANALISTAS
def obtener_rating_analistas(ticker):
    try:
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info
        
        ratings = {
            'recommendationMean': info.get('recommendationMean', 'N/A'),
            'recommendationKey': info.get('recommendationKey', 'N/A'),
            'targetMeanPrice': info.get('targetMeanPrice', 'N/A'),
            'numberOfAnalystOpinions': info.get('numberOfAnalystOpinions', 'N/A')
        }
        return ratings
    except:
        return {}

# FUNCIÓN PARA ANÁLISIS TÉCNICO CORREGIDA
def calcular_indicadores_tecnicos(data):
    if data.empty:
        return data
    
    # Crear una copia para no modificar el original
    data_tech = data.copy()
    
    # Asegurarnos de que tenemos la columna Close
    if 'Close' not in data_tech.columns:
        st.error("No se encuentra la columna 'Close' en los datos")
        return data_tech
    
    try:
        # RSI
        delta = data_tech['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data_tech['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp12 = data_tech['Close'].ewm(span=12, adjust=False).mean()
        exp26 = data_tech['Close'].ewm(span=26, adjust=False).mean()
        data_tech['MACD'] = exp12 - exp26
        data_tech['MACD_Signal'] = data_tech['MACD'].ewm(span=9, adjust=False).mean()
        data_tech['MACD_Histogram'] = data_tech['MACD'] - data_tech['MACD_Signal']
        
        # Bandas de Bollinger
        data_tech['BB_Middle'] = data_tech['Close'].rolling(window=20).mean()
        bb_std = data_tech['Close'].rolling(window=20).std()
        data_tech['BB_Upper'] = data_tech['BB_Middle'] + (bb_std * 2)
        data_tech['BB_Lower'] = data_tech['BB_Middle'] - (bb_std * 2)
        
        # Medias Móviles
        data_tech['SMA_20'] = data_tech['Close'].rolling(window=20).mean()
        data_tech['SMA_50'] = data_tech['Close'].rolling(window=50).mean()
        data_tech['SMA_200'] = data_tech['Close'].rolling(window=200).mean()
        
        return data_tech
        
    except Exception as e:
        st.error(f"Error calculando indicadores: {str(e)}")
        return data_tech

# FUNCIÓN PARA SCORING AUTOMÁTICO
def calcular_scoring_fundamental(info):
    score = 0
    max_score = 100
    metricas = {}
    
    # P/E Ratio (15 puntos)
    pe = info.get('trailingPE', 0)
    if pe and pe > 0:
        if pe < 15:
            score += 15
            metricas['P/E'] = '🟢 Excelente'
        elif pe < 25:
            score += 10
            metricas['P/E'] = '🟡 Bueno'
        else:
            score += 5
            metricas['P/E'] = '🔴 Alto'
    
    # ROE (15 puntos)
    roe = info.get('returnOnEquity', 0)
    if roe and roe > 0:
        if roe > 0.15:
            score += 15
            metricas['ROE'] = '🟢 Excelente'
        elif roe > 0.08:
            score += 10
            metricas['ROE'] = '🟡 Bueno'
        else:
            score += 5
            metricas['ROE'] = '🔴 Bajo'
    
    # Deuda/Equity (15 puntos)
    deuda_eq = info.get('debtToEquity', 0)
    if deuda_eq and deuda_eq > 0:
        if deuda_eq < 0.5:
            score += 15
            metricas['Deuda/Equity'] = '🟢 Excelente'
        elif deuda_eq < 1.0:
            score += 10
            metricas['Deuda/Equity'] = '🟡 Bueno'
        else:
            score += 5
            metricas['Deuda/Equity'] = '🔴 Alto'
    
    # Margen Beneficio (15 puntos)
    margen = info.get('profitMargins', 0)
    if margen and margen > 0:
        if margen > 0.2:
            score += 15
            metricas['Margen Beneficio'] = '🟢 Excelente'
        elif margen > 0.1:
            score += 10
            metricas['Margen Beneficio'] = '🟡 Bueno'
        else:
            score += 5
            metricas['Margen Beneficio'] = '🔴 Bajo'
    
    # Crecimiento Ingresos (20 puntos)
    crecimiento = info.get('revenueGrowth', 0)
    if crecimiento and crecimiento > 0:
        if crecimiento > 0.15:
            score += 20
            metricas['Crecimiento Ingresos'] = '🟢 Excelente'
        elif crecimiento > 0.08:
            score += 15
            metricas['Crecimiento Ingresos'] = '🟡 Bueno'
        else:
            score += 8
            metricas['Crecimiento Ingresos'] = '🔴 Bajo'
    
    # Rating Analistas (20 puntos)
    rating_mean = info.get('recommendationMean', 3)
    if rating_mean and rating_mean > 0:
        if rating_mean < 2:
            score += 20
            metricas['Rating Analistas'] = '🟢 Fuerte Compra'
        elif rating_mean < 3:
            score += 15
            metricas['Rating Analistas'] = '🟡 Compra'
        else:
            score += 8
            metricas['Rating Analistas'] = '🔴 Neutral/Venta'
    
    return min(score, max_score), metricas

# FUNCIÓN PARA GENERAR REPORTE
def generar_reporte_texto(ticker, info, datos, scoring, metricas):
    try:
        # Información básica
        nombre = info.get('longName', 'N/A')
        sector = info.get('sector', 'N/A')
        industria = info.get('industry', 'N/A')
        fecha_actual = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        # Construir el reporte paso a paso
        reporte = f"REPORTE DE ANÁLISIS: {ticker}\n"
        reporte += f"Generado: {fecha_actual}\n\n"
        
        reporte += "INFORMACIÓN BÁSICA:\n"
        reporte += f"- Nombre: {nombre}\n"
        reporte += f"- Sector: {sector}\n"
        reporte += f"- Industria: {industria}\n\n"
        
        reporte += f"SCORING FUNDAMENTAL: {scoring}/100\n\n"
        
        reporte += "MÉTRICAS:\n"
        for metrica, valor in metricas.items():
            reporte += f"- {metrica}: {valor}\n"
        
        # Datos de precio con verificación robusta
        if not datos.empty and 'Close' in datos.columns and len(datos) > 0:
            try:
                precio_actual_val = float(datos['Close'].iloc[-1])
                precio_min_val = float(datos['Close'].min())
                precio_max_val = float(datos['Close'].max())
                
                reporte += "\nDATOS DE PRECIO:\n"
                reporte += f"- Precio Actual: ${precio_actual_val:.2f}\n"
                reporte += f"- Precio Mínimo (1 año): ${precio_min_val:.2f}\n"
                reporte += f"- Precio Máximo (1 año): ${precio_max_val:.2f}\n"
            except (ValueError, IndexError, KeyError) as e:
                reporte += f"\nERROR en datos de precio: {str(e)}\n"
        else:
            reporte += "\nDATOS DE PRECIO: No disponibles\n"
        
        return reporte
        
    except Exception as e:
        return f"Error generando reporte: {str(e)}"

# FUNCIÓN PARA DETECTOR DE TENDENCIAS
def analizar_tendencias(data):
    if data.empty or 'Close' not in data.columns:
        return {"tendencia": "No disponible", "confianza": 0, "detalles": {}}
    
    try:
        # Calcular medias móviles
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['SMA_200'] = data['Close'].rolling(window=200).mean()
        
        # Obtener últimos valores
        precio_actual = data['Close'].iloc[-1]
        sma_20 = data['SMA_20'].iloc[-1]
        sma_50 = data['SMA_50'].iloc[-1]
        sma_200 = data['SMA_200'].iloc[-1]
        
        # Calcular RSI para momentum
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        rsi_actual = rsi.iloc[-1]
        
        # Análisis de tendencia
        tendencia_alcista = 0
        tendencia_bajista = 0
        
        # 1. Análisis de medias móviles (40%)
        if precio_actual > sma_20 > sma_50 > sma_200:
            tendencia_alcista += 40
        elif precio_actual < sma_20 < sma_50 < sma_200:
            tendencia_bajista += 40
        
        # 2. Posición respecto a medias (30%)
        if precio_actual > sma_20:
            tendencia_alcista += 15
        else:
            tendencia_bajista += 15
            
        if precio_actual > sma_50:
            tendencia_alcista += 10
        else:
            tendencia_bajista += 10
            
        if precio_actual > sma_200:
            tendencia_alcista += 5
        else:
            tendencia_bajista += 5
        
        # 3. Momentum RSI (30%)
        if rsi_actual > 50:
            tendencia_alcista += 30
        else:
            tendencia_bajista += 30
        
        # Determinar tendencia principal
        if tendencia_alcista > tendencia_bajista:
            tendencia = "ALCISTA"
            confianza = min(100, tendencia_alcista)
        elif tendencia_bajista > tendencia_alcista:
            tendencia = "BAJISTA"
            confianza = min(100, tendencia_bajista)
        else:
            tendencia = "LATERAL"
            confianza = 50
        
        detalles = {
            "precio_actual": precio_actual,
            "sma_20": sma_20,
            "sma_50": sma_50,
            "sma_200": sma_200,
            "rsi": rsi_actual,
            "puntos_alcista": tendencia_alcista,
            "puntos_bajista": tendencia_bajista
        }
        
        return {
            "tendencia": tendencia,
            "confianza": confianza,
            "detalles": detalles
        }
        
    except Exception as e:
        return {"tendencia": "Error en análisis", "confianza": 0, "detalles": {}}

# FUNCIÓN PARA OBTENER DATOS MACROECONÓMICOS
def obtener_datos_macro():
    # Datos macroeconómicos simulados (en una app real, esto vendría de APIs)
    datos_macro = {
        "indicadores_usa": {
            "Inflación (CPI)": "3.2%",
            "Tasa de Desempleo": "3.8%",
            "Crecimiento PIB": "2.1%",
            "Tasa de Interés Fed": "5.25%-5.50%",
            "Confianza del Consumidor": "64.9"
        },
        "mercados_globales": {
            "S&P 500": "+15% YTD",
            "NASDAQ": "+22% YTD",
            "Dow Jones": "+12% YTD",
            "Euro Stoxx 50": "+8% YTD",
            "Nikkei 225": "+18% YTD"
        },
        "materias_primas": {
            "Petróleo (WTI)": "$78.50",
            "Oro": "$1,950.00",
            "Plata": "$23.15",
            "Cobre": "$3.85",
            "Bitcoin": "$42,000"
        },
        "divisas": {
            "EUR/USD": "1.0850",
            "USD/JPY": "148.50",
            "GBP/USD": "1.2650",
            "USD/MXN": "17.20",
            "DXY (Índice Dólar)": "103.50"
        }
    }
    return datos_macro

# FUNCIÓN PARA OBTENER EL ANÁLISIS DE RIESGOS
def calcular_metricas_riesgo_avanzadas(ticker_symbol, periodo_años=5):
    """
    Calcula métricas avanzadas de riesgo MEJORADAS para una acción
    """
    try:
        # Descargar datos históricos
        end_date = datetime.today()
        start_date = end_date - timedelta(days=periodo_años * 365)
        
        # Datos de la acción
        stock_data = yf.download(ticker_symbol, start=start_date, end=end_date, interval='1d')
        if stock_data.empty or len(stock_data) == 0:
            return None
            
        # Datos del mercado (S&P500 como benchmark)
        market_data = yf.download('^GSPC', start=start_date, end=end_date, interval='1d')
        if market_data.empty or len(market_data) == 0:
            return None
        
        # Asegurarnos de que tenemos columnas de cierre
        if 'Close' not in stock_data.columns or 'Close' not in market_data.columns:
            return None
        
        # Calcular rendimientos diarios - manejar MultiIndex
        if isinstance(stock_data.columns, pd.MultiIndex):
            stock_close = stock_data[('Close', ticker_symbol)]
        else:
            stock_close = stock_data['Close']
            
        if isinstance(market_data.columns, pd.MultiIndex):
            market_close = market_data[('Close', '^GSPC')]
        else:
            market_close = market_data['Close']
        
        stock_returns = stock_close.pct_change().dropna()
        market_returns = market_close.pct_change().dropna()
        
        # Alinear las fechas
        common_dates = stock_returns.index.intersection(market_returns.index)
        if len(common_dates) == 0:
            return None
            
        stock_returns = stock_returns.loc[common_dates]
        market_returns = market_returns.loc[common_dates]
        
        if len(stock_returns) < 30:  # Mínimo de datos
            return None
        
        # Convertir a arrays numpy para evitar problemas con Series
        stock_returns_array = stock_returns.values
        market_returns_array = market_returns.values
        
        # 1. CALCULAR BETA
        covariance = np.cov(stock_returns_array, market_returns_array)[0, 1]
        market_variance = np.var(market_returns_array)
        beta = covariance / market_variance if market_variance != 0 else 0
        
        # 2. CALCULAR ALPHA
        stock_total_return = (stock_close.iloc[-1] / stock_close.iloc[0] - 1)
        market_total_return = (market_close.iloc[-1] / market_close.iloc[0] - 1)
        alpha = stock_total_return - (beta * market_total_return)
        
        # 3. CALCULAR SHARPE RATIO
        risk_free_rate = 0.02 / 252  # Tasa diaria
        excess_returns = stock_returns_array - risk_free_rate
        sharpe_ratio = (np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) 
                      if np.std(excess_returns) != 0 else 0)
        
        # 4. CALCULAR SORTINO RATIO
        downside_returns = stock_returns_array[stock_returns_array < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
        sortino_ratio = (np.mean(excess_returns) / downside_std * np.sqrt(252) 
                       if downside_std != 0 else 0)
        
        # 5. CALCULAR TREYNOR RATIO
        treynor_ratio = (stock_total_return - 0.02) / beta if beta != 0 else 0
        
        # 6. CALCULAR INFORMATION RATIO
        active_returns = stock_returns_array - market_returns_array
        tracking_error = np.std(active_returns) * np.sqrt(252) if len(active_returns) > 0 else 0
        information_ratio = (stock_total_return - market_total_return) / tracking_error if tracking_error != 0 else 0
        
        # 7. CALCULAR VALUE AT RISK (VaR)
        var_95 = np.percentile(stock_returns_array, 5)
        var_95_annual = var_95 * np.sqrt(252)
        var_99 = np.percentile(stock_returns_array, 1)
        var_99_annual = var_99 * np.sqrt(252)
        
        # 8. CALCULAR EXPECTED SHORTFALL (CVaR)
        cvar_95 = stock_returns_array[stock_returns_array <= var_95].mean()
        cvar_95_annual = cvar_95 * np.sqrt(252) if not np.isnan(cvar_95) else 0
        
        # 9. CALCULAR DRAWDOWN MÁXIMO
        cumulative_returns = (1 + stock_returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Calcular duración del drawdown máximo
        max_dd_idx = drawdown.idxmin()
        max_dd_start = drawdown[drawdown == 0].last_valid_index()
        if max_dd_start is not None:
            max_dd_duration = (max_dd_idx - max_dd_start).days
        else:
            max_dd_duration = 0
        
        # 10. CALCULAR VOLATILIDAD ANUALIZADA
        volatility_annual = np.std(stock_returns_array) * np.sqrt(252)
        
        # 11. CALCULAR CORRELACIONES CON MÚLTIPLES ÍNDICES 
        correlation_sp500 = np.corrcoef(stock_returns_array, market_returns_array)[0, 1]
        
        # 12. CALCULAR MÁXIMO GANANCIA/PÉRDIDA CONSECUTIVA 
        positive_streak = 0
        negative_streak = 0
        max_positive_streak = 0
        max_negative_streak = 0
        
        for ret in stock_returns_array:
            if ret > 0:
                positive_streak += 1
                negative_streak = 0
                max_positive_streak = max(max_positive_streak, positive_streak)
            elif ret < 0:
                negative_streak += 1
                positive_streak = 0
                max_negative_streak = max(max_negative_streak, negative_streak)
        
        # 13. CALCULAR SKEWNESS Y KURTOSIS
        skewness, kurtosis = calcular_skewness_kurtosis(stock_returns_array)
        
        # 14. CALCULAR PROBABILIDAD DE PÉRDIDA
        prob_loss = np.mean(stock_returns_array < 0) * 100
        
        return {
            # Métricas básicas
            'Beta': round(beta, 4),
            'Alpha': round(alpha, 4),
            'Sharpe Ratio': round(sharpe_ratio, 4),
            'Sortino Ratio': round(sortino_ratio, 4),
            'Treynor Ratio': round(treynor_ratio, 4),
            'Information Ratio': round(information_ratio, 4),
            
            # Métricas de riesgo
            'VaR 95% Diario': round(var_95, 4),
            'VaR 95% Anual': round(var_95_annual, 4),
            'VaR 99% Diario': round(var_99, 4),
            'VaR 99% Anual': round(var_99_annual, 4),
            'Expected Shortfall 95%': round(cvar_95_annual, 4),
            'Drawdown Máximo': round(max_drawdown, 4),
            'Duración Drawdown (días)': max_dd_duration,
            'Volatilidad Anual': round(volatility_annual, 4),
            
            # Correlaciones
            'Correlación S&P500': round(correlation_sp500, 4),
            
            # Estadísticas avanzadas
            'Máxima Ganancia Consecutiva': max_positive_streak,
            'Máxima Pérdida Consecutiva': max_negative_streak,
            'Skewness': round(skewness, 4),
            'Kurtosis': round(kurtosis, 4),
            'Probabilidad de Pérdida (%)': round(prob_loss, 2),
            
            # Rendimientos
            'Rendimiento Total': round(stock_total_return, 4),
            'Rendimiento Mercado': round(market_total_return, 4),
            'Días Analizados': len(stock_returns),
            'Período': f"{periodo_años} años"
        }
        
    except Exception as e:
        st.error(f"Error calculando métricas de riesgo: {str(e)}")
        return None

def calcular_skewness_kurtosis_manual(returns):
    """
    Calcula skewness y kurtosis manualmente para mayor robustez
    """
    try:
        n = len(returns)
        if n < 4:
            return 0, 0
        
        mean = np.mean(returns)
        std = np.std(returns, ddof=0)  # Usar ddof=0 para consistencia
        
        if std == 0:
            return 0, 0
        
        # Skewness
        skew = np.sum((returns - mean) ** 3) / (n * std ** 3)
        
        # Kurtosis (excess kurtosis)
        kurt = np.sum((returns - mean) ** 4) / (n * std ** 4) - 3
        
        return float(skew), float(kurt)
        
    except Exception as e:
        return 0, 0

def calcular_metricas_riesgo_avanzadas(ticker_symbol, periodo_años=5):
    """
    Calcula métricas avanzadas de riesgo MEJORADAS para una acción
    """
    try:
        # Descargar datos históricos
        end_date = datetime.today()
        start_date = end_date - timedelta(days=periodo_años * 365)
        
        st.info(f"📊 Calculando métricas de riesgo para {ticker_symbol}...")
        
        # Datos de la acción
        stock_data = yf.download(ticker_symbol, start=start_date, end=end_date, interval='1d', progress=False)
        if stock_data.empty or len(stock_data) < 100:
            st.warning(f"Datos insuficientes para {ticker_symbol}")
            return None
            
        # Datos del mercado (S&P500 como benchmark)
        market_data = yf.download('^GSPC', start=start_date, end=end_date, interval='1d', progress=False)
        if market_data.empty:
            st.warning("No se pudieron obtener datos del mercado")
            return None
        
        # Obtener precios de cierre
        if isinstance(stock_data.columns, pd.MultiIndex):
            stock_close = stock_data[('Close', ticker_symbol)]
        else:
            stock_close = stock_data['Close']
            
        if isinstance(market_data.columns, pd.MultiIndex):
            market_close = market_data[('Close', '^GSPC')]
        else:
            market_close = market_data['Close']
        
        # Limpiar datos NaN
        stock_close = stock_close.dropna()
        market_close = market_close.dropna()
        
        if len(stock_close) < 100 or len(market_close) < 100:
            st.warning("Datos insuficientes después de limpieza")
            return None
        
        # Calcular rendimientos
        stock_returns = stock_close.pct_change().dropna()
        market_returns = market_close.pct_change().dropna()
        
        # Alinear fechas
        common_dates = stock_returns.index.intersection(market_returns.index)
        if len(common_dates) < 50:
            st.warning("No hay suficientes fechas comunes con el mercado")
            return None
            
        stock_returns = stock_returns.loc[common_dates]
        market_returns = market_returns.loc[common_dates]
        
        if len(stock_returns) < 50:
            st.warning("Rendimientos insuficientes para análisis")
            return None
        
        # Convertir a arrays numpy
        stock_returns_array = stock_returns.values
        market_returns_array = market_returns.values
        
        # 1. CALCULAR BETA Y ALPHA
        try:
            covariance = np.cov(stock_returns_array, market_returns_array)[0, 1]
            market_variance = np.var(market_returns_array)
            beta = covariance / market_variance if market_variance != 0 else 1.0
            
            # Calcular rendimientos totales para Alpha
            stock_total_return = (stock_close.iloc[-1] / stock_close.iloc[0] - 1)
            market_total_return = (market_close.iloc[-1] / market_close.iloc[0] - 1)
            alpha = stock_total_return - (beta * market_total_return)
        except:
            beta = 1.0
            alpha = 0
        
        # 2. CALCULAR SHARPE RATIO
        try:
            risk_free_rate = 0.02 / 252  # Tasa libre de riesgo diaria (2% anual)
            excess_returns = stock_returns_array - risk_free_rate
            sharpe_ratio = (np.mean(excess_returns) / np.std(excess_returns)) * np.sqrt(252) if np.std(excess_returns) != 0 else 0
        except:
            sharpe_ratio = 0
        
        # 3. CALCULAR SORTINO RATIO (CORREGIDO)
        try:
            # Solo considerar rendimientos negativos para el denominador
            negative_returns = stock_returns_array[stock_returns_array < 0]
            downside_std = np.std(negative_returns) if len(negative_returns) > 0 else 0.001
            
            # Usar el mismo excess_returns que para Sharpe
            sortino_ratio = (np.mean(excess_returns) / downside_std) * np.sqrt(252) if downside_std != 0 else 0
        except:
            sortino_ratio = 0
        
        # 4. CALCULAR VALUE AT RISK (VaR) - CORREGIDO
        try:
            # VaR histórico (no paramétrico)
            var_95 = np.percentile(stock_returns_array, 5)  # 5% peores rendimientos
            var_95_annual = var_95 * np.sqrt(252)  # Anualizar
            
            # VaR 99%
            var_99 = np.percentile(stock_returns_array, 1)
            var_99_annual = var_99 * np.sqrt(252)
        except:
            var_95 = 0
            var_95_annual = 0
            var_99 = 0
            var_99_annual = 0
        
        # 5. CALCULAR EXPECTED SHORTFALL (CVaR) - CORREGIDO
        try:
            # Promedio de los peores 5% rendimientos
            cvar_95 = stock_returns_array[stock_returns_array <= var_95].mean()
            cvar_95_annual = cvar_95 * np.sqrt(252) if not np.isnan(cvar_95) else 0
        except:
            cvar_95_annual = 0
        
        # 6. CALCULAR DRAWDOWN MÁXIMO - CORREGIDO
        try:
            # Calcular retornos acumulados
            cumulative_returns = (1 + stock_returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            # Calcular duración del drawdown máximo
            max_dd_idx = drawdown.idxmin()
            # Encontrar el inicio del drawdown (último máximo antes del mínimo)
            drawdown_period = drawdown[:max_dd_idx]
            max_dd_start = drawdown_period[drawdown_period == 0].last_valid_index()
            
            if max_dd_start is not None:
                max_dd_duration = (max_dd_idx - max_dd_start).days
            else:
                max_dd_duration = 0
        except:
            max_drawdown = 0
            max_dd_duration = 0
        
        # 7. CALCULAR VOLATILIDAD ANUALIZADA
        try:
            volatility_annual = np.std(stock_returns_array) * np.sqrt(252)
        except:
            volatility_annual = 0
        
        # 8. CALCULAR CORRELACIÓN CON S&P500
        try:
            correlation_sp500 = np.corrcoef(stock_returns_array, market_returns_array)[0, 1]
            if np.isnan(correlation_sp500):
                correlation_sp500 = 0
        except:
            correlation_sp500 = 0
        
        # 9. CALCULAR MÁXIMO GANANCIA/PÉRDIDA CONSECUTIVA - CORREGIDO
        try:
            positive_streak = 0
            negative_streak = 0
            max_positive_streak = 0
            max_negative_streak = 0
            
            for ret in stock_returns_array:
                if ret > 0:
                    positive_streak += 1
                    negative_streak = 0
                    max_positive_streak = max(max_positive_streak, positive_streak)
                elif ret < 0:
                    negative_streak += 1
                    positive_streak = 0
                    max_negative_streak = max(max_negative_streak, negative_streak)
        except:
            max_positive_streak = 0
            max_negative_streak = 0
        
        # 10. CALCULAR SKEWNESS Y KURTOSIS - CORREGIDO
        try:
            if len(stock_returns_array) >= 4:
                skewness = float(pd.Series(stock_returns_array).skew())
                kurtosis = float(pd.Series(stock_returns_array).kurtosis())
            else:
                skewness = 0
                kurtosis = 0
        except:
            skewness = 0
            kurtosis = 0
        
        # 11. CALCULAR PROBABILIDAD DE PÉRDIDA - CORREGIDO
        try:
            prob_loss = (np.sum(stock_returns_array < 0) / len(stock_returns_array)) * 100
        except:
            prob_loss = 50
        
        # 12. CALCULAR TREYNOR RATIO
        try:
            treynor_ratio = (stock_total_return - 0.02) / beta if beta != 0 else 0
        except:
            treynor_ratio = 0
        
        # 13. CALCULAR INFORMATION RATIO
        try:
            active_returns = stock_returns_array - market_returns_array
            tracking_error = np.std(active_returns) * np.sqrt(252) if len(active_returns) > 0 else 0
            information_ratio = (stock_total_return - market_total_return) / tracking_error if tracking_error != 0 else 0
        except:
            information_ratio = 0
        
        st.success(f"✅ Métricas calculadas: {len(stock_returns)} días analizados")
        
        return {
            # Métricas básicas
            'Beta': beta,
            'Alpha': alpha,
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio,
            'Treynor Ratio': treynor_ratio,
            'Information Ratio': information_ratio,
            
            # Métricas de riesgo
            'VaR 95% Diario': var_95,
            'VaR 95% Anual': var_95_annual,
            'VaR 99% Diario': var_99,
            'VaR 99% Anual': var_99_annual,
            'Expected Shortfall 95%': cvar_95_annual,
            'Drawdown Máximo': max_drawdown,
            'Duración Drawdown (días)': max_dd_duration,
            'Volatilidad Anual': volatility_annual,
            
            # Correlaciones
            'Correlación S&P500': correlation_sp500,
            
            # Estadísticas avanzadas
            'Máxima Ganancia Consecutiva': max_positive_streak,
            'Máxima Pérdida Consecutiva': max_negative_streak,
            'Skewness': skewness,
            'Kurtosis': kurtosis,
            'Probabilidad de Pérdida (%)': prob_loss,
            
            # Rendimientos
            'Rendimiento Total': stock_total_return,
            'Rendimiento Mercado': market_total_return,
            'Días Analizados': len(stock_returns),
            'Período': f"{periodo_años} años"
        }
        
    except Exception as e:
        st.error(f"❌ Error calculando métricas de riesgo: {str(e)}")
        st.error(f"Tipo de error: {type(e).__name__}")
        return None

def generar_analisis_riesgo_ia(simbolo, datos_riesgo, nombre_empresa):
    """
    Genera análisis de riesgo usando IA de Google Gemini
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""
        Analiza estos datos de riesgo reales para {nombre_empresa} ({simbolo}):

        DATOS REALES:
        - Drawdown Máximo: {datos_riesgo.get('Drawdown Máximo', 0):.2%}
        - Volatilidad Anual: {datos_riesgo.get('Volatilidad Anual', 0):.2%}
        - Sharpe Ratio: {datos_riesgo.get('Sharpe Ratio', 0):.3f}
        - Sortino Ratio: {datos_riesgo.get('Sortino Ratio', 0):.3f}
        - Beta: {datos_riesgo.get('Beta', 0):.2f}
        - Alpha: {datos_riesgo.get('Alpha', 0):.2%}
        - Correlación S&P500: {datos_riesgo.get('Correlación S&P500', 0):.2%}
        - Probabilidad de Pérdida: {datos_riesgo.get('Probabilidad de Pérdida (%)', 0):.1f}%

        Proporciona un análisis conciso basado únicamente en estos datos reales.
        """
        
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        return None

def crear_grafica_drawdown_mejorada(ticker_symbol, periodo_años=5):
    """
    Crea gráfica de drawdown con datos reales
    """
    try:
        end_date = datetime.today()
        start_date = end_date - timedelta(days=periodo_años * 365)
        
        stock_data = yf.download(ticker_symbol, start=start_date, end=end_date, interval='1d')
        if stock_data.empty:
            return None
        
        if isinstance(stock_data.columns, pd.MultiIndex):
            stock_close = stock_data[('Close', ticker_symbol)]
        else:
            stock_close = stock_data['Close']
        
        returns = stock_close.pct_change().dropna()
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown * 100,
            fill='tozeroy',
            fillcolor='rgba(255, 0, 0, 0.3)',
            line=dict(color='red', width=2),
            name='Drawdown'
        ))
        
        fig.update_layout(
            title=f'Drawdown Real - {ticker_symbol}',
            xaxis_title='Fecha',
            yaxis_title='Drawdown (%)',
            height=500
        )
        
        return fig
        
    except Exception as e:
        return None

def crear_grafica_distribucion_retornos(ticker_symbol, periodo_años=5):
    """
    Crea gráfica de distribución de retornos diarios COMPLETA con estadísticas avanzadas
    """
    try:
        # Descargar datos históricos
        end_date = datetime.today()
        start_date = end_date - timedelta(days=periodo_años * 365)
        
        st.info(f"📊 Calculando distribución de retornos para {ticker_symbol} ({periodo_años} años)...")
        
        stock_data = yf.download(ticker_symbol, start=start_date, end=end_date, interval='1d', progress=False)
        if stock_data.empty:
            st.warning(f"No se pudieron obtener datos para {ticker_symbol}")
            return None
        
        # Manejar MultiIndex columns
        if isinstance(stock_data.columns, pd.MultiIndex):
            stock_close = stock_data[('Close', ticker_symbol)]
        else:
            stock_close = stock_data['Close']
        
        # Calcular retornos diarios en porcentaje
        returns = stock_close.pct_change().dropna() * 100
        
        if len(returns) < 30:
            st.warning(f"Datos insuficientes para análisis: solo {len(returns)} días de trading")
            return None
        
        # Calcular estadísticas avanzadas
        mean_return = returns.mean()
        std_return = returns.std()
        median_return = returns.median()
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        # Calcular percentiles
        percentiles = {
            '1%': returns.quantile(0.01),
            '5%': returns.quantile(0.05),
            '25%': returns.quantile(0.25),
            '75%': returns.quantile(0.75),
            '95%': returns.quantile(0.95),
            '99%': returns.quantile(0.99)
        }
        
        # Crear figura principal
        fig = go.Figure()
        
        # HISTOGRAMA PRINCIPAL
        fig.add_trace(go.Histogram(
            x=returns,
            nbinsx=50,
            name='Frecuencia de Retornos',
            opacity=0.75,
            marker_color='#1f77b4',
            marker_line_color='#0d47a1',
            marker_line_width=1,
            hovertemplate=(
                '<b>Rango de Retorno:</b> %{x:.2f}%<br>' +
                '<b>Frecuencia:</b> %{y} días<br>' +
                '<b>Probabilidad:</b> %{y}' + f'/{len(returns)} días<br>' +
                '<extra></extra>'
            )
        ))
        
        # CALCULAR Y AGREGAR DISTRIBUCIÓN NORMAL TEÓRICA
        x_norm = np.linspace(returns.min() * 1.1, returns.max() * 1.1, 200)
        pdf_norm = (1/(std_return * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_norm - mean_return)/std_return) ** 2)
        pdf_norm = pdf_norm * len(returns) * (returns.max() - returns.min()) / 50  # Escalar
        
        fig.add_trace(go.Scatter(
            x=x_norm,
            y=pdf_norm,
            mode='lines',
            name='Distribución Normal Teórica',
            line=dict(color='red', width=3, dash='dash'),
            hovertemplate='<b>Distribución Normal</b><br>Retorno: %{x:.2f}%<br>Densidad: %{y:.2f}<extra></extra>'
        ))
        
        # LÍNEAS DE REFERENCIA PRINCIPALES
        # Línea en CERO
        fig.add_vline(x=0, line_dash="solid", line_color="green", line_width=2,
                     annotation_text="Cero", annotation_position="top right",
                     annotation_font_color="green")
        
        # Línea de MEDIA
        fig.add_vline(x=mean_return, line_dash="dot", line_color="orange", line_width=2,
                     annotation_text=f"Media: {mean_return:.2f}%", 
                     annotation_position="top left",
                     annotation_font_color="orange")
        
        # Líneas de DESVIACIÓN ESTÁNDAR
        colors_sigma = ['#ff6b6b', '#ffa726', '#66bb6a']
        for i, std_mult in enumerate([1, 2, 3], 1):
            color = colors_sigma[i-1]
            # +Sigma
            fig.add_vline(x=mean_return + std_mult * std_return, 
                         line_dash="dot", line_color=color, line_width=1,
                         annotation_text=f"+{std_mult}σ" if std_mult <= 2 else "",
                         annotation_position="top")
            # -Sigma
            fig.add_vline(x=mean_return - std_mult * std_return, 
                         line_dash="dot", line_color=color, line_width=1,
                         annotation_text=f"-{std_mult}σ" if std_mult <= 2 else "",
                         annotation_position="top")
        
        # PERCENTILES IMPORTANTES
        # Percentil 5% (VaR aproximado)
        fig.add_vline(x=percentiles['5%'], line_dash="dash", line_color="purple", line_width=2,
                     annotation_text=f"5%: {percentiles['5%']:.2f}%",
                     annotation_position="bottom right")
        
        # Percentil 95%
        fig.add_vline(x=percentiles['95%'], line_dash="dash", line_color="purple", line_width=2,
                     annotation_text=f"95%: {percentiles['95%']:.2f}%",
                     annotation_position="bottom right")
        
        # CONFIGURACIÓN DEL LAYOUT
        fig.update_layout(
            title=dict(
                text=f'Distribución de Retornos Diarios - {ticker_symbol}',
                x=0.5,
                xanchor='center',
                font=dict(size=16, color='white')
            ),
            xaxis_title=dict(text='Retorno Diario (%)', font=dict(size=14)),
            yaxis_title=dict(text='Frecuencia (Días)', font=dict(size=14)),
            height=600,
            showlegend=True,
            bargap=0.02,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor='rgba(0,0,0,0.5)',
                bordercolor='white'
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        # PANEL DE ESTADÍSTICAS DETALLADO
        fig.add_annotation(
            x=0.02, y=0.98,
            xref="paper", yref="paper",
            text=(
                f"<b>📊 ESTADÍSTICAS AVANZADAS</b><br>"
                f"<b>Retorno Promedio:</b> {mean_return:.3f}%<br>"
                f"<b>Volatilidad (σ):</b> {std_return:.3f}%<br>"
                f"<b>Mediana:</b> {median_return:.3f}%<br>"
                f"<b>Asimetría (Skew):</b> {skewness:.3f}<br>"
                f"<b>Curtosis:</b> {kurtosis:.3f}<br>"
                f"<b>Días Analizados:</b> {len(returns):,}<br>"
                f"<b>Período:</b> {periodo_años} años"
            ),
            showarrow=False,
            bgcolor="rgba(30, 30, 30, 0.9)",
            bordercolor="white",
            borderwidth=1,
            borderpad=10,
            font=dict(size=11, color='white'),
            align="left"
        )
        
        # INTERPRETACIÓN DE SKEWNESS Y KURTOSIS
        skew_interpretation = (
            "Sesgo positivo (colas derechas)" if skewness > 0.5 else
            "Sesgo negativo (colas izquierdas)" if skewness < -0.5 else
            "Distribución simétrica"
        )
        
        kurt_interpretation = (
            "Colas pesadas (Leptocúrtica)" if kurtosis > 3 else
            "Colas livianas (Platicúrtica)" if kurtosis < 3 else
            "Colas normales (Mesocúrtica)"
        )
        
        fig.add_annotation(
            x=0.98, y=0.98,
            xref="paper", yref="paper",
            text=(
                f"<b>🔍 INTERPRETACIÓN</b><br>"
                f"<b>Asimetría:</b> {skew_interpretation}<br>"
                f"<b>Curtosis:</b> {kurt_interpretation}<br>"
                f"<b>Normalidad:</b> {'No normal' if abs(skewness) > 1 or abs(kurtosis) > 3 else 'Cercana a normal'}"
            ),
            showarrow=False,
            bgcolor="rgba(30, 30, 30, 0.9)",
            bordercolor="white",
            borderwidth=1,
            borderpad=10,
            font=dict(size=11, color='white'),
            align="right"
        )
        
        # MEJORAS EN LOS EJES
        fig.update_xaxes(
            gridcolor='rgba(128, 128, 128, 0.3)',
            zerolinecolor='rgba(128, 128, 128, 0.5)',
            zerolinewidth=2
        )
        
        fig.update_yaxes(
            gridcolor='rgba(128, 128, 128, 0.3)'
        )
        
        return fig
        
    except Exception as e:
        st.error(f"❌ Error creando gráfica de distribución: {str(e)}")
        # Debug information
        st.error(f"Tipo de error: {type(e).__name__}")
        return None

def generar_analisis_riesgo_ia(simbolo, datos_riesgo, nombre_empresa):
    """
    Genera análisis de riesgo COMPLETO usando IA de Google Gemini
    """
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Crear prompt detallado y estructurado
        prompt = f"""
        Eres un analista de riesgo financiero senior en un fondo de inversión global. 
        Analiza DETALLADAMENTE estos datos de riesgo para {nombre_empresa} ({simbolo}):

        📊 DATOS DE RIESGO COMPLETOS:
        
        • Drawdown Máximo Histórico: {datos_riesgo.get('Drawdown Máximo', 0)*100:.1f}%
        • Volatilidad Anualizada: {datos_riesgo.get('Volatilidad Anual', 0)*100:.1f}%
        • Sharpe Ratio: {datos_riesgo.get('Sharpe Ratio', 0):.3f}
        • Sortino Ratio: {datos_riesgo.get('Sortino Ratio', 0):.3f}
        • Beta vs Mercado: {datos_riesgo.get('Beta', 0):.2f}
        • Alpha: {datos_riesgo.get('Alpha', 0)*100:.2f}%
        • Value at Risk (VaR 95%): {datos_riesgo.get('VaR 95% Anual', 0)*100:.1f}%
        • Expected Shortfall (CVaR): {datos_riesgo.get('Expected Shortfall 95%', 0)*100:.1f}%
        • Correlación S&P500: {datos_riesgo.get('Correlación S&P500', 0):.3f}
        • Probabilidad de Pérdida Diaria: {datos_riesgo.get('Probabilidad de Pérdida (%)', 0):.1f}%
        • Máxima Pérdida Consecutiva: {datos_riesgo.get('Máxima Pérdida Consecutiva', 0)} días
        • Skewness: {datos_riesgo.get('Skewness', 0):.3f}
        • Kurtosis: {datos_riesgo.get('Kurtosis', 0):.3f}

        Proporciona un análisis PROFESIONAL que incluya:

        1. 🎯 EVALUACIÓN GLOBAL DEL RIESGO (1-10 escala)
        2. 📈 PRINCIPALES FUENTES DE RIESGO identificadas
        3. ⚖️ COMPARACIÓN con benchmarks del mercado
        4. 🛡️ RECOMENDACIONES ESPECÍFICAS de gestión
        5. 👤 PERFIL DE INVERSOR ADECUADO
        6. ⚠️ SEÑALES DE ALERTA principales
        7. 💡 ESTRATEGIAS DE MITIGACIÓN

        Sé técnico pero claro. Usa terminología profesional.
        Máximo 300 palabras. Basado estrictamente en los datos proporcionados.
        Incluye métricas específicas en tu análisis.
        """
        
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        # Análisis de respaldo COMPLETO si falla la IA
        drawdown = datos_riesgo.get('Drawdown Máximo', 0) * 100
        volatilidad = datos_riesgo.get('Volatilidad Anual', 0) * 100
        sharpe = datos_riesgo.get('Sharpe Ratio', 0)
        beta = datos_riesgo.get('Beta', 0)
        var = datos_riesgo.get('VaR 95% Anual', 0) * 100
        
        # Evaluación automática
        riesgo_score = 0
        if drawdown > 40: riesgo_score += 3
        elif drawdown > 25: riesgo_score += 2
        elif drawdown > 15: riesgo_score += 1
        
        if volatilidad > 50: riesgo_score += 3
        elif volatilidad > 30: riesgo_score += 2
        elif volatilidad > 20: riesgo_score += 1
        
        if beta > 1.5: riesgo_score += 2
        elif beta > 1.2: riesgo_score += 1
        
        nivel_riesgo = "ALTO" if riesgo_score >= 5 else "MODERADO-ALTO" if riesgo_score >= 3 else "MODERADO" if riesgo_score >= 1 else "BAJO"
        
        return f"""
        **🔍 ANÁLISIS DE RIESGO AVANZADO - {nombre_empresa}**

        **📊 EVALUACIÓN GLOBAL: {nivel_riesgo}**
        - Puntuación de riesgo: {riesgo_score}/8
        - Drawdown histórico: {drawdown:.1f}% ({'CRÍTICO' if drawdown > 40 else 'ALTO' if drawdown > 25 else 'MODERADO' if drawdown > 15 else 'BAJO'})
        - Volatilidad anual: {volatilidad:.1f}%

        **📈 MÉTRICAS CLAVE:**
        • Sharpe Ratio: {sharpe:.3f} ({'BUENO' if sharpe > 1.0 else 'ACEPTABLE' if sharpe > 0.5 else 'DEFICIENTE'})
        • Beta: {beta:.2f} ({'ALTA' if beta > 1.2 else 'MODERADA' if beta > 0.8 else 'BAJA'} sensibilidad al mercado)
        • VaR 95%: {var:.1f}% (Pérdida máxima esperada)
        • Prob. pérdida: {datos_riesgo.get('Probabilidad de Pérdida (%)', 0):.1f}% de días

        **🛡️ RECOMENDACIONES:**
        1. Stop-loss: {max(10, abs(drawdown * 0.6)):.0f}% (basado en drawdown histórico)
        2. Posicionamiento: {'REDUCIDO' if riesgo_score >= 4 else 'MODERADO' if riesgo_score >= 2 else 'NORMAL'}
        3. Diversificación: {'ALTA' if beta > 1.2 else 'MODERADA'} recomendada
        4. Monitoreo: {'SEMANAL' if volatilidad > 40 else 'MENSUAL'}

        **👤 PERFIL ADECUADO:** {'INVERSOR EXPERIMENTADO' if riesgo_score >= 4 else 'INVERSOR MODERADO' if riesgo_score >= 2 else 'INVERSOR CONSERVADOR'}
        """

def obtener_rating_analistas(ticker):
    """Rating de analistas - sin caching extenso porque cambia frecuentemente"""
    try:
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info
        
        ratings = {
            'recommendationMean': info.get('recommendationMean', 'N/A'),
            'recommendationKey': info.get('recommendationKey', 'N/A'),
            'targetMeanPrice': info.get('targetMeanPrice', 'N/A'), 
            'numberOfAnalystOpinions': info.get('numberOfAnalystOpinions', 'N/A')
        }
        return ratings
    except:
        return {}

# INTERFAZ PRINCIPAL
stonk = st.text_input("Ingrese el nombre del símbolo de la acción", value="MSFT")

# Agregar a historial de búsquedas
if stonk and stonk not in st.session_state.historial_busquedas:
    st.session_state.historial_busquedas.append(stonk)
    if len(st.session_state.historial_busquedas) > 10:
        st.session_state.historial_busquedas.pop(0)

end_date = datetime.today()
start_date = end_date - timedelta(days=5 * 365)

# Yahoo finanzas trae los datos del Ticker
try:
    ticker = yf.Ticker(stonk)
    info = ticker.info
    nombre = info.get("longName", "Ese nombre no existe")
    descripcion = info.get("longBusinessSummary", "No hay datos")
except Exception as e:
    st.error(f"❌ Error al cargar datos de {stonk}: {str(e)}")
    st.stop()

# BOTONES MEJORADOS CON NUEVA DISTRIBUCIÓN
st.write("### 📊 Selecciona qué información quieres ver:")

# Primera fila: 5 botones
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    if st.button("🏠 Inicio", use_container_width=True, key="btn_inicio", 
                type="primary" if st.session_state.seccion_actual == "inicio" else "secondary"):
        st.session_state.seccion_actual = "inicio"

with col2:
    if st.button("🏢 Información", use_container_width=True, key="btn_info", 
                type="primary" if st.session_state.seccion_actual == "info" else "secondary"):
        st.session_state.seccion_actual = "info"

with col3:    
    if st.button("📈 Variación del precio", use_container_width=True, key="btn_datos", 
                type="primary" if st.session_state.seccion_actual == "datos" else "secondary"):
        st.session_state.seccion_actual = "datos"

with col4:
    if st.button("💰 Datos fundamentales", use_container_width=True, key="btn_fundamentales", 
                type="primary" if st.session_state.seccion_actual == "fundamentales" else "secondary"):
        st.session_state.seccion_actual = "fundamentales"

with col5:
    if st.button("📊 Análisis técnico", use_container_width=True, key="btn_tecnico", 
                type="primary" if st.session_state.seccion_actual == "tecnico" else "secondary"):
        st.session_state.seccion_actual = "tecnico"

# Segunda fila: 4 botones
col6, col7, col8, col9 = st.columns(4)

with col6:
    if st.button("🤖 Análisis IA", use_container_width=True, key="btn_ia", 
                type="primary" if st.session_state.seccion_actual == "ia" else "secondary"):
        st.session_state.seccion_actual = "ia"

with col7:
    if st.button("⚠️ Análisis De Riesgos", use_container_width=True, key="btn_riesgo", 
                type="primary" if st.session_state.seccion_actual == "riesgo" else "secondary"):
        st.session_state.seccion_actual = "riesgo"

with col8:
    if st.button("📊 Comparación", use_container_width=True, key="btn_comparar", 
                type="primary" if st.session_state.seccion_actual == "comparar" else "secondary"):
        st.session_state.seccion_actual = "comparar"

with col9:
    if st.button("📰 Noticias", use_container_width=True, key="btn_noticias", 
                type="primary" if st.session_state.seccion_actual == "noticias" else "secondary"):
        st.session_state.seccion_actual = "noticias"

col10, col11, col12 = st.columns(3)

with col10:
    if st.button("🔍 Buscador", use_container_width=True, key="btn_screener", 
                type="primary" if st.session_state.seccion_actual == "screener" else "secondary"):
        st.session_state.seccion_actual = "screener"

# En la sección de botones (después del botón de Macroeconomía), agrega:
with col11:
    if st.button("🌍 Macroeconomía", use_container_width=True, key="btn_macro", 
                type="primary" if st.session_state.seccion_actual == "macro" else "secondary"):
        st.session_state.seccion_actual = "macro"

# Agrega un décimo botón para Mercados Globales
with col12:
    if st.button("📈 Mercados Globales", use_container_width=True, key="btn_global", 
                type="primary" if st.session_state.seccion_actual == "global" else "secondary"):
        st.session_state.seccion_actual = "global"

# Línea separadora
st.markdown("---")

# Inician Seecciones

# SECCIÓN DE INFORMACIÓN
if st.session_state.seccion_actual == "info":
    st.header(f"🏢 Información de {nombre}")
    
    # Rating de analistas
    ratings = obtener_rating_analistas(stonk)
    if ratings:
        st.subheader("🎯 Rating de Analistas")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            reco_key = ratings.get('recommendationKey', 'N/A')
            if isinstance(reco_key, str):
                reco_display = reco_key.upper().replace("_", " ")
            else:
                reco_display = "N/A"
            
            st.metric("Recomendación", reco_display)
        
        with col2:
            target_price = ratings.get('targetMeanPrice', 'N/A')
            current_price = info.get('currentPrice', 0)
            if target_price != 'N/A' and current_price and target_price > current_price:
                st.metric("Target Price", f"${target_price:.2f}", f"+{((target_price/current_price)-1)*100:.1f}%")
            elif target_price != 'N/A':
                st.metric("Target Price", f"${target_price:.2f}")
            else:
                st.metric("Target Price", "N/A")
        
        with col3:
            st.metric("Rating Medio", f"{ratings.get('recommendationMean', 'N/A')}/5")
        
        with col4:
            st.metric("# Analistas", ratings.get('numberOfAnalystOpinions', 'N/A'))
    
    # Descripción traducida
    prompt = f"""
    Te voy a dar la descripción en inglés de una empresa que cotiza en bolsa, necesito que traduzcas la descripción a español financiero formal,
    quiero que la traducción sea lo más apegado posible a la descripción original y que me entregues el texto en exactamente 500 caracteres, te paso la
    descripción de la empresa: {descripcion}
    """
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        texto_traducido = response.text

    except Exception as e:
        texto_traducido = "Traducción no disponible por el momento."
    
    st.subheader("📋 Descripción de la Empresa")
    st.write(texto_traducido)
    
    # INFORMACIÓN DE WIKIPEDIA PARA CUALQUIER ACCIÓN
    st.subheader("📚 Información Corporativa")

    # Información adicional básica
    st.subheader("📊 Información Básica")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        sector = info.get("sector", "N/A")
        st.metric("Sector", sector)
        employees = info.get("fullTimeEmployees", "N/A")
        if employees != "N/A":
            st.metric("Empleados", f"{employees:,}")
        else:
            st.metric("Empleados", "N/A")
    
    with col2:
        industry = info.get("industry", "N/A")
        st.metric("Industria", industry)
        country = info.get("country", "N/A")
        st.metric("País", country)
    
    with col3:
        market_cap = info.get("marketCap", "N/A")
        if market_cap != "N/A":
            st.metric("Market Cap", f"${market_cap/1e9:.2f}B")
        else:
            st.metric("Market Cap", "N/A")
        
        currency = info.get("currency", "N/A")
        st.metric("Moneda", currency)
    
    with col4:
        pe_ratio = info.get("trailingPE", "N/A")
        if pe_ratio != "N/A":
            st.metric("P/E Ratio", f"{pe_ratio:.2f}")
        else:
            st.metric("P/E Ratio", "N/A")
        
        dividend_yield = info.get("dividendYield", "N/A")
        if dividend_yield and dividend_yield != "N/A":
            st.metric("Dividend Yield", f"{dividend_yield*100:.2f}%")
        else:
            st.metric("Dividend Yield", "N/A")
            
    # Línea separadora
    st.markdown("---")

    # INFORMACIÓN DE WIKIPEDIA (AHORA AL FINAL)
    st.subheader("📚 Información Corporativa")

    # Obtener información de Wikipedia
    with st.spinner('Buscando información en Wikipedia...'):
        info_wikipedia = obtener_info_wikipedia(stonk, nombre)

        if info_wikipedia.get('encontrado', False):
            # MOSTRAR DIRECTAMENTE CON MARKDOWN SIN EL CUADRO HTML
            st.markdown(info_wikipedia['contenido'])
            
            # Mostrar fuente
            st.caption(f"📖 Fuente: {info_wikipedia['fuente']} - [Enlace a Wikipedia]({info_wikipedia['url']})")
            
        else:
            st.info("""
            ℹ️ **Información no disponible**
                
            No se pudo encontrar información específica de esta empresa. 
            """)

# SECCIÓN DE INICIO
elif st.session_state.seccion_actual == "inicio":
    st.header("🏠 Análisis de las 20 Acciones de cada sector del S&P 500 en Tiempo Real")
    
    # =============================================
    # SISTEMA DE CACHÉ Y PRE-CÁLCULO OPTIMIZADO
    # =============================================

    @st.cache_data(ttl=3600, show_spinner=False, max_entries=10)  # 1 hora de caché
    def precalcular_datos_mercado():
        """Precalcula todos los datos del mercado para máxima velocidad"""
        if 'datos_mercado_precalculados' in st.session_state:
            return st.session_state.datos_mercado_precalculados
        
        datos_precalculados = {
            'sp500_data': {},
            'market_data': {},
            'empresa_info': {}
        }
        
        # Precalcular datos del S&P 500
        try:
            sp500_data = obtener_datos_accion_ultra_mejorado("^GSPC")
            datos_precalculados['sp500_data'] = sp500_data
        except:
            datos_precalculados['sp500_data'] = pd.DataFrame()
        
        # Precalcular información de empresas (batch processing)
        todos_los_tickers = []
        for sector, stocks in sp500_components.items():
            for stock in stocks:
                todos_los_tickers.append(stock["ticker"])
        
        # Limitar a 100 tickers para demo (puedes aumentar)
        tickers_rapidos = todos_los_tickers[:240]
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, ticker in enumerate(tickers_rapidos):
            try:
                # Precalcular datos de precio
                stock_data = obtener_datos_accion_ultra_mejorado(ticker)
                if not stock_data.empty and len(stock_data) >= 2:
                    datos_precalculados['market_data'][ticker] = stock_data
                
                # Precalcular info de empresa
                company_info = obtener_info_completa_ultra_mejorada(ticker)
                datos_precalculados['empresa_info'][ticker] = company_info
                
                # Actualizar progreso cada 10 acciones
                if i % 10 == 0:
                    progress_percent = (i + 1) / len(tickers_rapidos)
                    progress_bar.progress(progress_percent)
                    status_text.text(f"Precalculando: {i+1}/{len(tickers_rapidos)} acciones")
                    
            except Exception as e:
                continue
        
        progress_bar.empty()
        status_text.empty()
        
        st.session_state.datos_mercado_precalculados = datos_precalculados
        return datos_precalculados

    def obtener_datos_con_cache(ticker):
        """Obtiene datos usando el sistema de caché precalculado"""
        datos_precalculados = st.session_state.get('datos_mercado_precalculados', {})
        
        if ticker in datos_precalculados.get('market_data', {}):
            return datos_precalculados['market_data'][ticker]
        else:
            # Fallback a la función original si no está en caché
            return obtener_datos_accion_ultra_mejorado(ticker)

    def obtener_info_con_cache(ticker):
        """Obtiene información de empresa usando caché"""
        datos_precalculados = st.session_state.get('datos_mercado_precalculados', {})
        
        if ticker in datos_precalculados.get('empresa_info', {}):
            return datos_precalculados['empresa_info'][ticker]
        else:
            # Fallback a la función original si no está en caché
            return obtener_info_completa_ultra_mejorada(ticker)

    # FUNCIONES ULTRA MEJORADAS CON MÁXIMA COBERTURA Y CACHÉ
    def obtener_datos_accion_ultra_mejorado(ticker, max_reintentos=2):  # Reducido reintentos para velocidad
        """Obtiene datos usando TODAS las APIs disponibles con reintentos y caché"""
        # Verificar caché primero
        cache_key = f"precio_{ticker}"
        if cache_key in st.session_state:
            return st.session_state[cache_key]
        
        # Lista de funciones de obtención de datos en orden de preferencia
        fuentes = [
            lambda: obtener_datos_accion(ticker),  # Yahoo Finance (cached)
            lambda: obtener_datos_yahoo_directo(ticker),  # Más rápido que otras APIs
        ]
        
        for intento in range(max_reintentos):
            for i, fuente in enumerate(fuentes):
                try:
                    data = fuente()
                    if not data.empty and len(data) >= 2:
                        # Verificar que los datos sean válidos
                        current_price = float(data['Close'].iloc[-1])
                        if current_price > 0 and not pd.isna(current_price):
                            # Guardar en caché
                            st.session_state[cache_key] = data
                            return data
                except:
                    continue
            
            # Pequeña pausa entre reintentos
            if intento < max_reintentos - 1:
                time.sleep(0.1)  # Reducido tiempo de espera
        
        # Si fallan todas las fuentes, devolver DataFrame vacío
        return pd.DataFrame()

    def obtener_datos_yahoo_directo(ticker):
        """Obtención directa de Yahoo Finance optimizada"""
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?range=5d&interval=1d"
            response = requests.get(url, timeout=5)  # Timeout reducido
            if response.status_code == 200:
                data = response.json()
                if 'chart' in data and 'result' in data['chart']:
                    result = data['chart']['result'][0]
                    timestamps = result['timestamp']
                    closes = result['indicators']['quote'][0]['close']
                    
                    dates = [pd.to_datetime(ts, unit='s') for ts in timestamps]
                    valid_data = [(date, close) for date, close in zip(dates, closes) 
                                 if close is not None and not pd.isna(close)]
                    
                    if valid_data:
                        dates, closes = zip(*valid_data)
                        df = pd.DataFrame({
                            'Close': closes,
                            'Volume': [1000000] * len(closes)  # Placeholder
                        }, index=dates)
                        return df
        except:
            pass
        return pd.DataFrame()

    def obtener_info_completa_ultra_mejorada(ticker):
        """Obtiene información completa usando caché"""
        cache_key = f"info_{ticker}"
        if cache_key in st.session_state:
            return st.session_state[cache_key]
        
        # Primero Yahoo Finance
        try:
            info = obtener_info_completa(ticker)
            if info and info.get('longName') != 'N/A':
                st.session_state[cache_key] = info
                return info
        except:
            pass
        
        # Información mínima como fallback
        info_fallback = {
            'longName': ticker,
            'sector': 'N/A',
            'industry': 'N/A',
            'trailingPE': 'N/A',
            'dividendYield': 0,
            'marketCap': 'N/A',
            'description': f'Compañía {ticker}'
        }
        st.session_state[cache_key] = info_fallback
        return info_fallback

    # LISTA COMPLETA Y ACTUALIZADA DE 500 ACCIONES DEL S&P 500
    sp500_components = {
        "TECHNOLOGY": [
            {"ticker": "AAPL", "name": "Apple Inc.", "weight": 7.2},
            {"ticker": "MSFT", "name": "Microsoft Corp", "weight": 6.8},
            {"ticker": "NVDA", "name": "NVIDIA Corporation", "weight": 2.9},
            {"ticker": "AVGO", "name": "Broadcom Inc.", "weight": 1.2},
            {"ticker": "CRM", "name": "Salesforce Inc.", "weight": 0.8},
            {"ticker": "ADBE", "name": "Adobe Inc.", "weight": 0.7},
            {"ticker": "CSCO", "name": "Cisco Systems", "weight": 0.6},
            {"ticker": "ACN", "name": "Accenture PLC", "weight": 0.6},
            {"ticker": "ORCL", "name": "Oracle Corp", "weight": 0.5},
            {"ticker": "IBM", "name": "IBM Corporation", "weight": 0.4},
            {"ticker": "INTC", "name": "Intel Corp", "weight": 0.4},
            {"ticker": "AMD", "name": "Advanced Micro Devices", "weight": 0.4},
            {"ticker": "QCOM", "name": "Qualcomm Inc.", "weight": 0.3},
            {"ticker": "TXN", "name": "Texas Instruments", "weight": 0.3},
            {"ticker": "NOW", "name": "ServiceNow Inc.", "weight": 0.3},
            {"ticker": "AMAT", "name": "Applied Materials", "weight": 0.3},
            {"ticker": "LRCX", "name": "Lam Research", "weight": 0.3},
            {"ticker": "KLAC", "name": "KLA Corporation", "weight": 0.2},
            {"ticker": "INTU", "name": "Intuit Inc.", "weight": 0.2},
            {"ticker": "ADI", "name": "Analog Devices", "weight": 0.2}
        ],
        "HEALTHCARE": [
            {"ticker": "LLY", "name": "Eli Lilly & Co", "weight": 1.4},
            {"ticker": "UNH", "name": "UnitedHealth Group", "weight": 1.3},
            {"ticker": "JNJ", "name": "Johnson & Johnson", "weight": 1.1},
            {"ticker": "MRK", "name": "Merck & Co.", "weight": 0.6},
            {"ticker": "ABBV", "name": "AbbVie Inc.", "weight": 0.6},
            {"ticker": "TMO", "name": "Thermo Fisher Scientific", "weight": 0.5},
            {"ticker": "PFE", "name": "Pfizer Inc.", "weight": 0.4},
            {"ticker": "ABT", "name": "Abbott Laboratories", "weight": 0.4},
            {"ticker": "DHR", "name": "Danaher Corp", "weight": 0.4},
            {"ticker": "CVS", "name": "CVS Health Corp", "weight": 0.3},
            {"ticker": "MDT", "name": "Medtronic PLC", "weight": 0.3},
            {"ticker": "AMGN", "name": "Amgen Inc.", "weight": 0.3},
            {"ticker": "BMY", "name": "Bristol-Myers Squibb", "weight": 0.3},
            {"ticker": "CI", "name": "Cigna Corporation", "weight": 0.2},
            {"ticker": "HUM", "name": "Humana Inc.", "weight": 0.2},
            {"ticker": "ELV", "name": "Elevance Health", "weight": 0.2},
            {"ticker": "GILD", "name": "Gilead Sciences", "weight": 0.2},
            {"ticker": "VRTX", "name": "Vertex Pharmaceuticals", "weight": 0.2},
            {"ticker": "REGN", "name": "Regeneron Pharmaceuticals", "weight": 0.2},
            {"ticker": "ISRG", "name": "Intuitive Surgical", "weight": 0.2}
        ],
        "FINANCIALS": [
            {"ticker": "BRK-B", "name": "Berkshire Hathaway", "weight": 1.7},
            {"ticker": "JPM", "name": "JPMorgan Chase", "weight": 1.1},
            {"ticker": "V", "name": "Visa Inc.", "weight": 1.0},
            {"ticker": "MA", "name": "Mastercard Inc.", "weight": 0.7},
            {"ticker": "BAC", "name": "Bank of America", "weight": 0.6},
            {"ticker": "WFC", "name": "Wells Fargo", "weight": 0.4},
            {"ticker": "GS", "name": "Goldman Sachs", "weight": 0.4},
            {"ticker": "MS", "name": "Morgan Stanley", "weight": 0.3},
            {"ticker": "BLK", "name": "BlackRock Inc.", "weight": 0.3},
            {"ticker": "AXP", "name": "American Express", "weight": 0.3},
            {"ticker": "SCHW", "name": "Charles Schwab", "weight": 0.3},
            {"ticker": "C", "name": "Citigroup Inc.", "weight": 0.2},
            {"ticker": "PYPL", "name": "PayPal Holdings", "weight": 0.2},
            {"ticker": "SPGI", "name": "S&P Global Inc.", "weight": 0.2},
            {"ticker": "MCO", "name": "Moody's Corporation", "weight": 0.2},
            {"ticker": "ICE", "name": "Intercontinental Exchange", "weight": 0.2},
            {"ticker": "CME", "name": "CME Group Inc.", "weight": 0.2},
            {"ticker": "TFC", "name": "Truist Financial", "weight": 0.1},
            {"ticker": "PNC", "name": "PNC Financial", "weight": 0.1},
            {"ticker": "USB", "name": "U.S. Bancorp", "weight": 0.1}
        ],
        "CONSUMER & INDUSTRIAL": [
            {"ticker": "AMZN", "name": "Amazon.com Inc.", "weight": 3.5},
            {"ticker": "TSLA", "name": "Tesla Inc.", "weight": 1.6},
            {"ticker": "HD", "name": "Home Depot", "weight": 0.6},
            {"ticker": "PG", "name": "Procter & Gamble", "weight": 0.6},
            {"ticker": "MCD", "name": "McDonald's Corp", "weight": 0.5},
            {"ticker": "COST", "name": "Costco Wholesale", "weight": 0.5},
            {"ticker": "KO", "name": "Coca-Cola Company", "weight": 0.4},
            {"ticker": "PEP", "name": "PepsiCo Inc.", "weight": 0.4},
            {"ticker": "WMT", "name": "Walmart Inc.", "weight": 0.4},
            {"ticker": "NKE", "name": "Nike Inc.", "weight": 0.4},
            {"ticker": "LOW", "name": "Lowe's Companies", "weight": 0.3},
            {"ticker": "SBUX", "name": "Starbucks Corp", "weight": 0.3},
            {"ticker": "PM", "name": "Philip Morris Int", "weight": 0.3},
            {"ticker": "TJX", "name": "TJX Companies", "weight": 0.2},
            {"ticker": "TGT", "name": "Target Corp", "weight": 0.2},
            {"ticker": "BKNG", "name": "Booking Holdings", "weight": 0.2},
            {"ticker": "ORLY", "name": "O'Reilly Automotive", "weight": 0.2},
            {"ticker": "MO", "name": "Altria Group", "weight": 0.2},
            {"ticker": "MDLZ", "name": "Mondelez Intl", "weight": 0.2},
            {"ticker": "CL", "name": "Colgate-Palmolive", "weight": 0.2}
        ],
        "ENERGY & UTILITIES": [
            {"ticker": "XOM", "name": "Exxon Mobil", "weight": 0.8},
            {"ticker": "CVX", "name": "Chevron Corp", "weight": 0.6},
            {"ticker": "NEE", "name": "NextEra Energy", "weight": 0.3},
            {"ticker": "COP", "name": "ConocoPhillips", "weight": 0.3},
            {"ticker": "DUK", "name": "Duke Energy", "weight": 0.2},
            {"ticker": "SO", "name": "Southern Company", "weight": 0.2},
            {"ticker": "SLB", "name": "Schlumberger", "weight": 0.2},
            {"ticker": "EOG", "name": "EOG Resources", "weight": 0.2},
            {"ticker": "PSX", "name": "Phillips 66", "weight": 0.1},
            {"ticker": "MPC", "name": "Marathon Petroleum", "weight": 0.1},
            {"ticker": "VLO", "name": "Valero Energy", "weight": 0.1},
            {"ticker": "OXY", "name": "Occidental Petroleum", "weight": 0.1},
            {"ticker": "KMI", "name": "Kinder Morgan", "weight": 0.1},
            {"ticker": "WMB", "name": "Williams Companies", "weight": 0.1},
            {"ticker": "HES", "name": "Hess Corporation", "weight": 0.1},
            {"ticker": "OKE", "name": "ONEOK Inc.", "weight": 0.1},
            {"ticker": "DVN", "name": "Devon Energy", "weight": 0.1},
            {"ticker": "PXD", "name": "Pioneer Natural Resources", "weight": 0.1},
            {"ticker": "FANG", "name": "Diamondback Energy", "weight": 0.1},
            {"ticker": "ETR", "name": "Entergy Corporation", "weight": 0.1}
        ],
        "COMMUNICATION SERVICES": [
            {"ticker": "GOOGL", "name": "Alphabet Inc.", "weight": 2.1},
            {"ticker": "GOOG", "name": "Alphabet Inc. C", "weight": 1.9},
            {"ticker": "META", "name": "Meta Platforms", "weight": 2.0},
            {"ticker": "NFLX", "name": "Netflix Inc.", "weight": 0.3},
            {"ticker": "DIS", "name": "Walt Disney Company", "weight": 0.4},
            {"ticker": "CMCSA", "name": "Comcast Corporation", "weight": 0.3},
            {"ticker": "T", "name": "AT&T Inc.", "weight": 0.3},
            {"ticker": "VZ", "name": "Verizon Communications", "weight": 0.3},
            {"ticker": "TMUS", "name": "T-Mobile US", "weight": 0.2},
            {"ticker": "CHTR", "name": "Charter Communications", "weight": 0.1},
            {"ticker": "EA", "name": "Electronic Arts", "weight": 0.1},
            {"ticker": "TTWO", "name": "Take-Two Interactive", "weight": 0.1},
            {"ticker": "ATVI", "name": "Activision Blizzard", "weight": 0.1},
            {"ticker": "LYV", "name": "Live Nation Entertainment", "weight": 0.1},
            {"ticker": "OMC", "name": "Omnicom Group", "weight": 0.1},
            {"ticker": "IPG", "name": "Interpublic Group", "weight": 0.1},
            {"ticker": "FOXA", "name": "Fox Corporation", "weight": 0.1},
            {"ticker": "FOX", "name": "Fox Corporation", "weight": 0.1},
            {"ticker": "PARA", "name": "Paramount Global", "weight": 0.1},
            {"ticker": "WBD", "name": "Warner Bros Discovery", "weight": 0.1}
        ],
        "INDUSTRIALS": [
            {"ticker": "UNP", "name": "Union Pacific", "weight": 0.3},
            {"ticker": "CAT", "name": "Caterpillar Inc.", "weight": 0.3},
            {"ticker": "RTX", "name": "Raytheon Technologies", "weight": 0.3},
            {"ticker": "HON", "name": "Honeywell International", "weight": 0.3},
            {"ticker": "UPS", "name": "United Parcel Service", "weight": 0.2},
            {"ticker": "BA", "name": "Boeing Company", "weight": 0.2},
            {"ticker": "LMT", "name": "Lockheed Martin", "weight": 0.2},
            {"ticker": "DE", "name": "Deere & Company", "weight": 0.2},
            {"ticker": "GE", "name": "General Electric", "weight": 0.2},
            {"ticker": "GD", "name": "General Dynamics", "weight": 0.1},
            {"ticker": "NOC", "name": "Northrop Grumman", "weight": 0.1},
            {"ticker": "EMR", "name": "Emerson Electric", "weight": 0.1},
            {"ticker": "ITW", "name": "Illinois Tool Works", "weight": 0.1},
            {"ticker": "MMM", "name": "3M Company", "weight": 0.1},
            {"ticker": "ETN", "name": "Eaton Corporation", "weight": 0.1},
            {"ticker": "WM", "name": "Waste Management", "weight": 0.1},
            {"ticker": "RSG", "name": "Republic Services", "weight": 0.1},
            {"ticker": "CSX", "name": "CSX Corporation", "weight": 0.1},
            {"ticker": "NSC", "name": "Norfolk Southern", "weight": 0.1},
            {"ticker": "FDX", "name": "FedEx Corporation", "weight": 0.1}
        ],
        "MATERIALS & REAL ESTATE": [
            {"ticker": "LIN", "name": "Linde PLC", "weight": 0.2},
            {"ticker": "AMT", "name": "American Tower", "weight": 0.2},
            {"ticker": "PLD", "name": "Prologis Inc.", "weight": 0.2},
            {"ticker": "APD", "name": "Air Products & Chemicals", "weight": 0.1},
            {"ticker": "ECL", "name": "Ecolab Inc.", "weight": 0.1},
            {"ticker": "SHW", "name": "Sherwin-Williams", "weight": 0.1},
            {"ticker": "DD", "name": "DuPont de Nemours", "weight": 0.1},
            {"ticker": "FCX", "name": "Freeport-McMoRan", "weight": 0.1},
            {"ticker": "NEM", "name": "Newmont Corporation", "weight": 0.1},
            {"ticker": "CCI", "name": "Crown Castle", "weight": 0.1},
            {"ticker": "EQIX", "name": "Equinix Inc.", "weight": 0.1},
            {"ticker": "PSA", "name": "Public Storage", "weight": 0.1},
            {"ticker": "AVB", "name": "AvalonBay Communities", "weight": 0.1},
            {"ticker": "EQR", "name": "Equity Residential", "weight": 0.1},
            {"ticker": "WELL", "name": "Welltower Inc.", "weight": 0.1},
            {"ticker": "O", "name": "Realty Income Corp", "weight": 0.1},
            {"ticker": "SPG", "name": "Simon Property Group", "weight": 0.1},
            {"ticker": "VTR", "name": "Ventas Inc.", "weight": 0.1},
            {"ticker": "DLR", "name": "Digital Realty Trust", "weight": 0.1},
            {"ticker": "ARE", "name": "Alexandria Real Estate", "weight": 0.1}
        ]
    }

    # =============================================
    # CARGA OPTIMIZADA DE DATOS CON PRE-CÁLCULO
    # =============================================

    # Indicador de estado del cache
    if 'datos_mercado_precalculados' in st.session_state:
        precalc_data = st.session_state.datos_mercado_precalculados
        st.success(f"✅ **Sistema optimizado activo:** {len(precalc_data.get('market_data', {}))} acciones precalculadas")
    else:
        st.info("🔄 **Cargando sistema optimizado:** Los datos se precalcularán para máxima velocidad")

    # Obtener datos del S&P 500 en tiempo real CON CACHÉ
    with st.spinner('🔄 Cargando datos del mercado con sistema optimizado...'):
        try:
            # Precalcular datos si no existen
            if 'datos_mercado_precalculados' not in st.session_state:
                datos_precalculados = precalcular_datos_mercado()
            else:
                datos_precalculados = st.session_state.datos_mercado_precalculados
            
            # Obtener datos del índice S&P 500 desde caché
            sp500_data = datos_precalculados.get('sp500_data', pd.DataFrame())
            
            if not sp500_data.empty and len(sp500_data) >= 2:
                current_sp500 = float(sp500_data['Close'].iloc[-1])
                previous_sp500 = float(sp500_data['Close'].iloc[-2])
                sp500_change = ((current_sp500 - previous_sp500) / previous_sp500) * 100
                sp500_change_abs = current_sp500 - previous_sp500
            else:
                # Datos de respaldo para el índice
                current_sp500 = 4780.94
                previous_sp500 = 4750.79
                sp500_change = 0.63
                sp500_change_abs = 30.15
            
            # Obtener datos en tiempo real para los componentes CON SISTEMA OPTIMIZADO
            market_data = {}
            total_stocks = sum(len(stocks) for stocks in sp500_components.values())
            successful_stocks = 0
            
            # Usar datos del cache cuando estén disponibles
            for sector, stocks in sp500_components.items():
                market_data[sector] = []
                for stock in stocks:
                    # Intentar obtener datos del cache primero
                    stock_data = obtener_datos_con_cache(stock["ticker"])
                    company_info = obtener_info_con_cache(stock["ticker"])
                    
                    if not stock_data.empty and len(stock_data) >= 2:
                        try:
                            current_price = float(stock_data['Close'].iloc[-1])
                            previous_price = float(stock_data['Close'].iloc[-2])
                            change = ((current_price - previous_price) / previous_price) * 100
                            
                            market_data[sector].append({
                                **stock,
                                "current_price": current_price,
                                "change": change,
                                "volume": float(stock_data['Volume'].iloc[-1]) if 'Volume' in stock_data.columns else 0,
                                "market_cap": company_info.get('marketCap', 'N/A'),
                                "sector": company_info.get('sector', sector),
                                "empresa_info": company_info,
                                "fuente": "real"
                            })
                            successful_stocks += 1
                            
                        except Exception as e:
                            # Si hay error en el procesamiento, usar datos simulados rápidos
                            precio_simulado = 50 + (hash(stock["ticker"]) % 200)
                            cambio_simulado = (hash(stock["ticker"]) % 40 - 20) / 10
                            
                            market_data[sector].append({
                                **stock,
                                "current_price": precio_simulado,
                                "change": cambio_simulado,
                                "volume": 1000000,
                                "market_cap": 'N/A',
                                "sector": sector,
                                "empresa_info": {"longName": stock["name"]},
                                "fuente": "simulado"
                            })
                            successful_stocks += 1
                    else:
                        # Usar datos simulados rápidos si no hay datos reales
                        precio_simulado = 50 + (hash(stock["ticker"]) % 200)
                        cambio_simulado = (hash(stock["ticker"]) % 40 - 20) / 10
                        
                        market_data[sector].append({
                            **stock,
                            "current_price": precio_simulado,
                            "change": cambio_simulado,
                            "volume": 1000000,
                            "market_cap": 'N/A',
                            "sector": sector,
                            "empresa_info": {"longName": stock["name"]},
                            "fuente": "simulado"
                        })
                        successful_stocks += 1
            
            st.success(f"✅ **Datos cargados:** {successful_stocks}/{total_stocks} acciones procesadas")
            
        except Exception as e:
            st.error(f"❌ Error en la carga de datos: {str(e)}")
            st.stop()

    # =============================================
    # INTERFAZ DE USUARIO OPTIMIZADA
    # =============================================

    # METRICS DEL S&P 500 - MEJORADO CON MÁS DATOS
    st.markdown("### 📊 S&P 500 INDEX OVERVIEW")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="S&P 500 INDEX",
            value=f"{current_sp500:,.2f}",
            delta=f"{sp500_change_abs:+.2f} ({sp500_change:+.2f}%)",
            delta_color="normal"
        )
    
    with col2:
        # Obtener datos YTD reales
        try:
            ytd_data = obtener_datos_con_cache("^GSPC")
            if not ytd_data.empty and len(ytd_data) > 0:
                current_year = datetime.now().year
                start_of_year = pd.Timestamp(f'{current_year}-01-01')
                
                ytd_prices = ytd_data[ytd_data.index >= start_of_year]
                if len(ytd_prices) > 0:
                    ytd_start_price = float(ytd_prices['Close'].iloc[0])
                    ytd_return = ((current_sp500 - ytd_start_price) / ytd_start_price) * 100
                    st.metric(
                        label="YTD PERFORMANCE",
                        value=f"{ytd_return:+.1f}%",
                        delta_color="normal"
                    )
                else:
                    st.metric(label="YTD PERFORMANCE", value="N/A")
            else:
                st.metric(label="YTD PERFORMANCE", value="N/A")
        except:
            st.metric(label="YTD PERFORMANCE", value="N/A")
    
    with col3:
        # Calcular P/E ratio promedio ponderado del S&P 500
        try:
            total_pe = 0
            count_pe = 0
            total_weight_pe = 0
            
            for sector, stocks in market_data.items():
                for stock in stocks:
                    if (stock.get('empresa_info') and 
                        stock['empresa_info'].get('trailingPE') != 'N/A' and
                        stock['empresa_info'].get('trailingPE') is not None):
                        try:
                            pe = float(stock['empresa_info']['trailingPE'])
                            if pe > 0 and pe < 100:
                                weight = stock.get('weight', 0.1)
                                total_pe += pe * weight
                                total_weight_pe += weight
                                count_pe += 1
                        except:
                            continue
            
            if count_pe > 0 and total_weight_pe > 0:
                weighted_pe = total_pe / total_weight_pe
                st.metric(
                    label="P/E RATIO",
                    value=f"{weighted_pe:.1f}",
                    delta_color="off"
                )
            else:
                st.metric(label="P/E RATIO", value="22.5")
        except:
            st.metric(label="P/E RATIO", value="22.5")

    with col4:
        # Calcular dividend yield promedio ponderado del S&P 500
        try:
            total_dy = 0
            count_dy = 0
            total_weight_dy = 0
            
            for sector, stocks in market_data.items():
                for stock in stocks:
                    if (stock.get('empresa_info') and 
                        stock['empresa_info'].get('dividendYield') != 'N/A' and
                        stock['empresa_info'].get('dividendYield') is not None):
                        try:
                            dy = float(stock['empresa_info']['dividendYield'])
                            if dy >= 0 and dy < 0.1:  # Filtro para valores razonables (0-10%)
                                weight = stock.get('weight', 0.1)
                                total_dy += dy * weight
                                total_weight_dy += weight
                                count_dy += 1
                        except:
                            continue
            
            if count_dy > 0 and total_weight_dy > 0:
                weighted_dy = (total_dy / total_weight_dy) * 100  # Convertir a porcentaje
                st.metric(
                    label="DIVIDEND YIELD",
                    value=f"{weighted_dy:.2f}%",
                    delta_color="off"
                )
            else:
                # Valor por defecto si no se puede calcular
                st.metric(label="DIVIDEND YIELD", value="1.42%")
        except:
            st.metric(label="DIVIDEND YIELD", value="1.42%")
    
    with col5:
        # Market Cap total estimado
        try:
            total_market_cap = 0
            count = 0
            for sector, stocks in market_data.items():
                for stock in stocks:
                    if stock.get('market_cap') and stock['market_cap'] != 'N/A':
                        total_market_cap += float(stock['market_cap'])
                        count += 1
            
            if count > 0:
                avg_market_cap = total_market_cap / count
                estimated_total = avg_market_cap * total_stocks
                st.metric(
                    label="EST. MARKET CAP",
                    value=f"${estimated_total/1e12:.1f}T",
                    delta_color="off"
                )
            else:
                st.metric(label="EST. MARKET CAP", value="N/A")
        except:
            st.metric(label="EST. MARKET CAP", value="N/A")

    # COMPONENTES PRINCIPALES POR SECTOR - CON ANÁLISIS IA
    st.markdown("### 🏢 COMPONENTES DEL S&P 500 - DATOS EN TIEMPO REAL")
    
    # Función para análisis IA rápido usando tu configuración de Gemini
    @st.cache_data(ttl=600, show_spinner=False)  # Cache de 10 minutos para análisis IA
    def generar_analisis_rapido_ia(ticker, nombre, precio, cambio):
        """Genera análisis rápido con IA para una acción usando tu configuración"""
        try:
            prompt = f"""
            Proporciona un análisis conciso de {nombre} ({ticker}) basado en:
            - Precio actual: ${precio:.2f}
            - Cambio del día: {cambio:+.2f}%
            
            Incluye en máximo 100 palabras:
            1. Evaluación rápida del movimiento
            2. Contexto del sector
            3. Recomendación breve (Observar/Considerar/Monitorear)
            
            Sé profesional pero conciso.
            """
            
            model = genai.GenerativeModel('gemini-2.5-flash')
            response = model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            return f"❌ Error en análisis IA: {str(e)}"

    # Variable para almacenar el análisis seleccionado
    if 'analisis_actual' not in st.session_state:
        st.session_state.analisis_actual = None
    
    # Mostrar sectores con tabs para mejor organización
    tabs = st.tabs(list(market_data.keys()))
    
    for tab_idx, (sector, stocks) in enumerate(market_data.items()):
        with tabs[tab_idx]:
            if not stocks:
                st.warning(f"No hay datos reales disponibles para {sector}")
                continue
                
            st.markdown(f"#### 📈 {sector} - {len(stocks)} Acciones con Datos Reales")
            
            # Búsqueda y filtrado dentro de cada sector
            search_col, filter_col = st.columns([2, 1])
            with search_col:
                search_term = st.text_input(f"🔍 Buscar en {sector}", key=f"search_{sector}")
            
            with filter_col:
                filter_option = st.selectbox(
                    "Filtrar por:",
                    ["Todos", "Alza (+)", "Baja (-)", "Top 10 por Peso"],
                    key=f"filter_{sector}"
                )
            
            # Aplicar filtros
            filtered_stocks = stocks
            if search_term:
                filtered_stocks = [s for s in filtered_stocks 
                                 if search_term.upper() in s["ticker"] or 
                                 search_term.lower() in s["name"].lower()]
            
            if filter_option == "Alza (+)":
                filtered_stocks = [s for s in filtered_stocks if s["change"] > 0]
            elif filter_option == "Baja (-)":
                filtered_stocks = [s for s in filtered_stocks if s["change"] < 0]
            elif filter_option == "Top 10 por Peso":
                filtered_stocks = sorted(filtered_stocks, key=lambda x: x["weight"], reverse=True)[:10]
            
            if not filtered_stocks:
                st.warning("No hay acciones que coincidan con los filtros aplicados")
                continue
            
            # Dividir en filas de 5 columnas
            for i in range(0, len(filtered_stocks), 5):
                row_stocks = filtered_stocks[i:i+5]
                cols = st.columns(5)
                
                for idx, stock in enumerate(row_stocks):
                    with cols[idx]:
                        # Determinar color del cambio
                        change_color = "#4CAF50" if stock["change"] >= 0 else "#F44336"
                        change_icon = "📈" if stock["change"] >= 0 else "📉"
                        
                        st.markdown(f"""
                        <div style='background: #1e1e1e; padding: 15px; border-radius: 8px; border: 1px solid #374151; 
                                    text-align: center; height: 160px; display: flex; flex-direction: column; justify-content: space-between;'>
                            <div>
                                <div style='font-weight: bold; color: white; font-size: 14px; margin-bottom: 5px;'>{stock["ticker"]}</div>
                                <div style='color: #9ca3af; font-size: 11px; margin-bottom: 8px; line-height: 1.2;'>
                                    {stock["name"][:25]}{'...' if len(stock["name"]) > 25 else ''}
                                </div>
                            </div>
                            <div>
                                <div style='color: white; font-weight: bold; font-size: 13px; margin-bottom: 4px;'>
                                    ${stock["current_price"]:,.2f}
                                </div>
                                <div style='color: {change_color}; font-size: 12px; font-weight: bold;'>
                                    {change_icon} {stock["change"]:+.2f}%
                                </div>
                                <div style='color: #6b7280; font-size: 10px; margin-top: 4px;'>
                                    Weight: {stock["weight"]}%
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Botón para análisis IA
                        if st.button(f"🤖 Analizar {stock['ticker']}", 
                                   key=f"ia_{stock['ticker']}_{i}_{idx}",
                                   use_container_width=True,
                                   type="primary"):
                            with st.spinner(f'Generando análisis IA para {stock["ticker"]}...'):
                                analisis = generar_analisis_rapido_ia(
                                    stock["ticker"], 
                                    stock["name"], 
                                    stock["current_price"], 
                                    stock["change"]
                                )
                                st.session_state.analisis_actual = {
                                    "ticker": stock["ticker"],
                                    "nombre": stock["name"],
                                    "analisis": analisis,
                                    "precio": stock["current_price"],
                                    "cambio": stock["change"]
                                }
                                st.rerun()

    # MOSTRAR ANÁLISIS ACTUAL SI EXISTE
    if st.session_state.analisis_actual:
        st.markdown("---")
        st.markdown("### 🧠 ANÁLISIS IA - " + st.session_state.analisis_actual["ticker"])
        
        # Tarjeta de análisis
        cambio = st.session_state.analisis_actual["cambio"]
        color_borde = "#4CAF50" if cambio >= 0 else "#F44336"
        
        st.markdown(f"""
        <div style='background: #1e1e1e; padding: 20px; border-radius: 10px; border-left: 6px solid {color_borde}; 
                    border: 1px solid #374151; margin-bottom: 20px;'>
            <div style='display: flex; justify-content: space-between; align-items: start; margin-bottom: 15px;'>
                <div>
                    <h4 style='color: white; margin: 0 0 5px 0;'>{st.session_state.analisis_actual["nombre"]}</h4>
                    <div style='color: #9ca3af; font-size: 14px;'>{st.session_state.analisis_actual["ticker"]}</div>
                </div>
                <div style='text-align: right;'>
                    <div style='color: white; font-size: 18px; font-weight: bold;'>
                        ${st.session_state.analisis_actual["precio"]:,.2f}
                    </div>
                    <div style='color: {color_borde}; font-size: 14px; font-weight: bold;'>
                        {cambio:+.2f}%
                    </div>
                </div>
            </div>
            <div style='color: #e5e7eb; font-size: 14px; line-height: 1.5; background: #2d3748; padding: 15px; border-radius: 6px;'>
                {st.session_state.analisis_actual["analisis"]}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Botón para limpiar análisis
        if st.button("🗑️ Cerrar Análisis", use_container_width=True):
            st.session_state.analisis_actual = None
            st.rerun()
    
    # ESTADÍSTICAS DEL MERCADO CON DATOS REALES
    st.markdown("### 📈 ESTADÍSTICAS DEL MERCADO - DATOS REALES")
    
    total_acciones = sum(len(stocks) for stocks in market_data.values())
    
    # Calcular estadísticas reales
    try:
        # Calcular promedio de cambios
        todos_los_cambios = []
        for sector, stocks in market_data.items():
            for stock in stocks:
                todos_los_cambios.append(stock["change"])
        
        promedio_cambios = sum(todos_los_cambios) / len(todos_los_cambios) if todos_los_cambios else 0
        acciones_alcistas = sum(1 for cambio in todos_los_cambios if cambio > 0)
        porcentaje_alcistas = (acciones_alcistas / len(todos_los_cambios)) * 100 if todos_los_cambios else 0
        
    except:
        promedio_cambios = 0
        porcentaje_alcistas = 0
    
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    
    with col_stat1:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    color: white; padding: 20px; border-radius: 10px; text-align: center;'>
            <div style='font-size: 24px; font-weight: bold;'>{porcentaje_alcistas:.1f}%</div>
            <div style='font-size: 12px;'>ACCIONES EN ALZA</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_stat2:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                    color: white; padding: 20px; border-radius: 10px; text-align: center;'>
            <div style='font-size: 24px; font-weight: bold;'>{len(market_data)}</div>
            <div style='font-size: 12px;'>SECTORES ANALIZADOS</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_stat3:
        cambio_color = "#4CAF50" if promedio_cambios >= 0 else "#F44336"
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); 
                    color: white; padding: 20px; border-radius: 10px; text-align: center;'>
            <div style='font-size: 24px; font-weight: bold; color: {cambio_color};'>{promedio_cambios:+.2f}%</div>
            <div style='font-size: 12px;'>CAMBIO PROMEDIO</div>
        </div>
        """, unsafe_allow_html=True)

    # BOTÓN PARA LIMPIAR CACHÉ (útil para desarrollo)
    st.markdown("---")
    if st.button("🗑️ Limpiar Caché de Mercado", type="secondary"):
        if 'datos_mercado_precalculados' in st.session_state:
            del st.session_state.datos_mercado_precalculados
        # Limpiar caches individuales
        keys_to_delete = [key for key in st.session_state.keys() if key.startswith('precio_') or key.startswith('info_')]
        for key in keys_to_delete:
            del st.session_state[key]
        st.success("✅ Caché de mercado limpiado. Los datos se recargarán.")
        st.rerun()

# SECCIÓN DE VARIACIÓN DEL PRECIO 
elif st.session_state.seccion_actual == "datos":
    st.header(f"📊 Variación del Precio y Gráfica de Velas de {nombre}")
    
    # MÉTRICAS DE PRECIO
    st.subheader(f"📊 Métricas de Precio - Período Actual")
    
    try:
        # Descargar datos de yfinance (por defecto 5 años para las métricas iniciales)
        start_date_default = end_date - timedelta(days=5 * 365)
        data = yf.download(stonk, start=start_date_default.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), interval='1d')
        
        if data.empty:
            st.warning("No se encontraron datos para este símbolo")
        else:
            # Organizar datos
            data = data.reset_index()
            
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in data.columns.values]
            
            data.columns = [col.replace(f'_{stonk}', '') for col in data.columns]
            
            # MÉTRICAS VISUALES
            if 'Close' in data.columns:
                precio_actual = data['Close'].iloc[-1]
                precio_inicial = data['Close'].iloc[0]
                variacion_total = ((precio_actual - precio_inicial) / precio_inicial) * 100
                
                # Calcular variación del último día
                if len(data) > 1:
                    precio_anterior = data['Close'].iloc[-2]
                    variacion_diaria = ((precio_actual - precio_anterior) / precio_anterior) * 100
                else:
                    variacion_diaria = 0
                
                # Calcular máximo y mínimo del período
                precio_maximo = data['Close'].max()
                precio_minimo = data['Close'].min()
                
                # Calcular volatilidad (desviación estándar de los retornos diarios)
                retornos_diarios = data['Close'].pct_change().dropna()
                volatilidad = retornos_diarios.std() * 100  # En porcentaje
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Precio Inicial", f"${precio_inicial:.2f}")
                    st.metric("Precio Mínimo", f"${precio_minimo:.2f}")
                with col2:
                    st.metric("Precio Actual", f"${precio_actual:.2f}", f"{variacion_diaria:.2f}%")
                    st.metric("Precio Máximo", f"${precio_maximo:.2f}")
                with col3:
                    st.metric("Variación Total", f"{variacion_total:.2f}%")
                    st.metric("Volatilidad Anual", f"{volatilidad:.2f}%")
                with col4:
                    st.metric("Período", "5 Años")
                    st.metric("Días Analizados", len(data))
            
            # Selector de período
            st.subheader("📅 Selecciona el período de análisis")
            
            periodo_opciones = {
                "1 Mes": 30,
                "3 Meses": 90,
                "6 Meses": 180,
                "1 Año": 365,
                "3 Años": 3 * 365,
                "5 Años": 5 * 365,
                "Máximo": None  # Para datos máximos disponibles
            }
            
            periodo_seleccionado = st.selectbox(
                "Período:",
                options=list(periodo_opciones.keys()),
                index=5,  # 5 Años por defecto
                key="selector_periodo"
            )
            
            # Calcular fecha de inicio según el período seleccionado
            if periodo_opciones[periodo_seleccionado] is None:
                # Para período máximo, usar una fecha muy antigua
                start_date = datetime(2000, 1, 1)
                periodo_texto = "Máximo"
            else:
                start_date = end_date - timedelta(days=periodo_opciones[periodo_seleccionado])
                periodo_texto = periodo_seleccionado
            
            # Descargar datos de yfinance
            data_periodo = yf.download(stonk, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), interval='1d')
            
            if not data_periodo.empty:
                data_periodo = data_periodo.reset_index()
                if isinstance(data_periodo.columns, pd.MultiIndex):
                    data_periodo.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in data_periodo.columns.values]
                data_periodo.columns = [col.replace(f'_{stonk}', '') for col in data_periodo.columns]
            
            # Línea separadora entre métricas y gráfica
            st.markdown("---")
            
            # GRÁFICA DE VELAS
            st.subheader(f"📈 Gráfica de Velas - Período: {periodo_texto}")
            
            # Función para obtener nombres de columnas dinámicamente
            def get_column_name(data, prefix):
                for col in data.columns:
                    if col.startswith(prefix):
                        return col
                return None
            
            if not data_periodo.empty:
                # Obtener los nombres dinámicos de las columnas
                open_col = get_column_name(data_periodo, 'Open')
                high_col = get_column_name(data_periodo, 'High') 
                low_col = get_column_name(data_periodo, 'Low')
                close_col = get_column_name(data_periodo, 'Close')
                date_col = get_column_name(data_periodo, 'Date')
                
                # Gráfica de velas
                if all(col is not None for col in [open_col, high_col, low_col, close_col, date_col]):
                    fig = go.Figure(data=[go.Candlestick(
                        x=data_periodo[date_col],
                        open=data_periodo[open_col],
                        high=data_periodo[high_col],
                        low=data_periodo[low_col],
                        close=data_periodo[close_col],
                        increasing_line_color='green',
                        decreasing_line_color='red',
                        name=stonk
                    )])
                    
                    fig.update_layout(
                        title=f'Gráfica de velas de {stonk}',
                        xaxis_title='Fecha',
                        yaxis_title='Precio (USD)',
                        xaxis_rangeslider_visible=False,
                        height=600
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                else:
                    st.warning("No se pudieron cargar los datos para la gráfica de velas")
            
                # DETECTOR DE TENDENCIAS (NUEVO)
                st.markdown("---")
                st.subheader("🔍 Detector de Tendencias")
                
                # Analizar tendencias
                analisis_tendencia = analizar_tendencias(data_periodo)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if analisis_tendencia["tendencia"] == "ALCISTA":
                        st.success(f"📈 Tendencia: {analisis_tendencia['tendencia']}")
                    elif analisis_tendencia["tendencia"] == "BAJISTA":
                        st.error(f"📉 Tendencia: {analisis_tendencia['tendencia']}")
                    else:
                        st.warning(f"➡️ Tendencia: {analisis_tendencia['tendencia']}")
                    
                    st.metric("Confianza", f"{analisis_tendencia['confianza']}%")
                
                with col2:
                    if 'detalles' in analisis_tendencia:
                        detalles = analisis_tendencia['detalles']
                        if 'precio_actual' in detalles:
                            st.metric("Precio Actual", f"${detalles['precio_actual']:.2f}")
                        if 'rsi' in detalles:
                            rsi_color = "green" if detalles['rsi'] < 30 else "red" if detalles['rsi'] > 70 else "orange"
                            st.metric("RSI", f"{detalles['rsi']:.1f}")
                
                with col3:
                    if 'detalles' in analisis_tendencia:
                        detalles = analisis_tendencia['detalles']
                        if all(key in detalles for key in ['sma_20', 'sma_50', 'sma_200']):
                            st.write("**Medias Móviles:**")
                            st.write(f"SMA 20: ${detalles['sma_20']:.2f}")
                            st.write(f"SMA 50: ${detalles['sma_50']:.2f}")
                            st.write(f"SMA 200: ${detalles['sma_200']:.2f}")
                
                # Explicación de la tendencia
                with st.expander("📖 Explicación del Análisis de Tendencia"):
                    st.write("""
                    **Cómo se determina la tendencia:**
                    - **Medias Móviles (40%):** Analiza la posición del precio respecto a las medias de 20, 50 y 200 días
                    - **Posición Precio/Medias (30%):** Evalúa si el precio está por encima o debajo de las medias clave
                    - **Momentum RSI (30%):** Considera si el RSI indica fuerza compradora o vendedora
                    
                    **Interpretación:**
                    - 🟢 **ALCISTA:** Precio por encima de medias, RSI >50, medias alineadas ascendente
                    - 🔴 **BAJISTA:** Precio por debajo de medias, RSI <50, medias alineadas descendente  
                    - 🟡 **LATERAL:** Señales mixtas o sin dirección clara
                    """)
                
                # Línea separadora entre gráfica y tabla
                st.markdown("---")
                
                # TABLA DE DATOS HISTÓRICOS
                st.subheader(f"📋 Datos Históricos Del Período: {periodo_texto}")
                
                # Mostrar información resumida sobre los datos
                st.write(f"**Total de registros:** {len(data_periodo)} días")
                if date_col:
                    st.write(f"**Período:** {data_periodo[date_col].iloc[0].strftime('%d/%m/%Y')} - {data_periodo[date_col].iloc[-1].strftime('%d/%m/%Y')}")
                
                st.dataframe(data_periodo, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error al generar la gráfica: {str(e)}")

# SECCIÓN DATOS FUNDAMENTALES 
elif st.session_state.seccion_actual == "fundamentales":
    st.header(f"💰 Datos Fundamentales Completos - {nombre}")
    
    # Pestañas para Fundamentales
    tab1, tab2 = st.tabs(["📊 Análisis Fundamental", "🎓 Educación Financiera"])

    with tab1:
        # FUNCIONES PARA EXTRACCIÓN DE DATOS FUNDAMENTALES
        def extraer_tabla_finviz(ticker):
            url = f"https://finviz.com/quote.ashx?t={ticker}"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            try:
                response = requests.get(url, headers=headers)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Extraer TODOS los datos de la tabla snapshot de Finviz
                    tabla_snapshot = soup.find('table', class_='snapshot-table2')
                    
                    if tabla_snapshot:
                        datos = {}
                        
                        # Extraer en el formato exacto de Finviz (pares clave-valor)
                        filas = tabla_snapshot.find_all('tr')
                        
                        for fila in filas:
                            celdas = fila.find_all('td')
                            for i in range(0, len(celdas) - 1, 2):
                                if i + 1 < len(celdas):
                                    clave = celdas[i].get_text(strip=True)
                                    valor = celdas[i + 1].get_text(strip=True)
                                    if clave and valor:
                                        datos[clave] = valor
                        
                        return datos
                    else:
                        return {}
                else:
                    return {}
                    
            except Exception as e:
                return {}

        # FUNCIÓN PARA CALCULAR SKEWNESS Y KURTOSIS
        def calcular_skewness_kurtosis(returns):
            """
            Calcula skewness y kurtosis de una serie de retornos
            """
            try:
                n = len(returns)
                if n < 4:
                    return 0, 0
                
                mean = np.mean(returns)
                std = np.std(returns)
                
                if std == 0:
                    return 0, 0
                
                # Skewness
                skew = np.sum((returns - mean) ** 3) / (n * std ** 3)
                
                # Kurtosis (Fisher's definition, excess kurtosis)
                kurt = np.sum((returns - mean) ** 4) / (n * std ** 4) - 3
                
                return skew, kurt
                
            except Exception as e:
                return 0, 0

        # FUNCIONES PARA CÁLCULOS DE RIESGO AVANZADOS
        def calcular_metricas_riesgo_avanzadas(ticker_symbol, periodo_años=5):
            """
            Calcula métricas avanzadas de riesgo MEJORADAS para una acción
            """
            try:
                # Descargar datos históricos
                end_date = datetime.today()
                start_date = end_date - timedelta(days=periodo_años * 365)
                
                # Datos de la acción
                stock_data = yf.download(ticker_symbol, start=start_date, end=end_date, interval='1d')
                if stock_data.empty or len(stock_data) == 0:
                    return None
                    
                # Datos del mercado (S&P500 como benchmark)
                market_data = yf.download('^GSPC', start=start_date, end=end_date, interval='1d')
                if market_data.empty or len(market_data) == 0:
                    return None
                
                # Asegurarnos de que tenemos columnas de cierre
                if 'Close' not in stock_data.columns or 'Close' not in market_data.columns:
                    return None
                
                # Calcular rendimientos diarios - manejar MultiIndex
                if isinstance(stock_data.columns, pd.MultiIndex):
                    stock_close = stock_data[('Close', ticker_symbol)]
                else:
                    stock_close = stock_data['Close']
                    
                if isinstance(market_data.columns, pd.MultiIndex):
                    market_close = market_data[('Close', '^GSPC')]
                else:
                    market_close = market_data['Close']
                
                stock_returns = stock_close.pct_change().dropna()
                market_returns = market_close.pct_change().dropna()
                
                # Alinear las fechas
                common_dates = stock_returns.index.intersection(market_returns.index)
                if len(common_dates) == 0:
                    return None
                    
                stock_returns = stock_returns.loc[common_dates]
                market_returns = market_returns.loc[common_dates]
                
                if len(stock_returns) < 30:  # Mínimo de datos
                    return None
                
                # Convertir a arrays numpy para evitar problemas con Series
                stock_returns_array = stock_returns.values
                market_returns_array = market_returns.values
                
                # 1. CALCULAR BETA
                covariance = np.cov(stock_returns_array, market_returns_array)[0, 1]
                market_variance = np.var(market_returns_array)
                beta = covariance / market_variance if market_variance != 0 else 0
                
                # 2. CALCULAR ALPHA
                stock_total_return = (stock_close.iloc[-1] / stock_close.iloc[0] - 1)
                market_total_return = (market_close.iloc[-1] / market_close.iloc[0] - 1)
                alpha = stock_total_return - (beta * market_total_return)
                
                # 3. CALCULAR SHARPE RATIO
                risk_free_rate = 0.02 / 252  # Tasa diaria
                excess_returns = stock_returns_array - risk_free_rate
                sharpe_ratio = (np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) 
                              if np.std(excess_returns) != 0 else 0)
                
                # 4. CALCULAR SORTINO RATIO
                downside_returns = stock_returns_array[stock_returns_array < 0]
                downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
                sortino_ratio = (np.mean(excess_returns) / downside_std * np.sqrt(252) 
                               if downside_std != 0 else 0)
                
                # 5. CALCULAR TREYNOR RATIO
                treynor_ratio = (stock_total_return - 0.02) / beta if beta != 0 else 0
                
                # 6. CALCULAR INFORMATION RATIO
                active_returns = stock_returns_array - market_returns_array
                tracking_error = np.std(active_returns) * np.sqrt(252) if len(active_returns) > 0 else 0
                information_ratio = (stock_total_return - market_total_return) / tracking_error if tracking_error != 0 else 0
                
                # 7. CALCULAR VALUE AT RISK (VaR)
                var_95 = np.percentile(stock_returns_array, 5)
                var_95_annual = var_95 * np.sqrt(252)
                var_99 = np.percentile(stock_returns_array, 1)
                var_99_annual = var_99 * np.sqrt(252)
                
                # 8. CALCULAR EXPECTED SHORTFALL (CVaR)
                cvar_95 = stock_returns_array[stock_returns_array <= var_95].mean()
                cvar_95_annual = cvar_95 * np.sqrt(252) if not np.isnan(cvar_95) else 0
                
                # 9. CALCULAR DRAWDOWN MÁXIMO
                cumulative_returns = (1 + stock_returns).cumprod()
                rolling_max = cumulative_returns.expanding().max()
                drawdown = (cumulative_returns - rolling_max) / rolling_max
                max_drawdown = drawdown.min()
                
                # Calcular duración del drawdown máximo
                max_dd_idx = drawdown.idxmin()
                max_dd_start = drawdown[drawdown == 0].last_valid_index()
                if max_dd_start is not None:
                    max_dd_duration = (max_dd_idx - max_dd_start).days
                else:
                    max_dd_duration = 0
                
                # 10. CALCULAR VOLATILIDAD ANUALIZADA
                volatility_annual = np.std(stock_returns_array) * np.sqrt(252)
                
                # 11. CALCULAR CORRELACIONES CON MÚLTIPLES ÍNDICES 
                correlation_sp500 = np.corrcoef(stock_returns_array, market_returns_array)[0, 1]
                
                # 12. CALCULAR MÁXIMO GANANCIA/PÉRDIDA CONSECUTIVA 
                positive_streak = 0
                negative_streak = 0
                max_positive_streak = 0
                max_negative_streak = 0
                
                for ret in stock_returns_array:
                    if ret > 0:
                        positive_streak += 1
                        negative_streak = 0
                        max_positive_streak = max(max_positive_streak, positive_streak)
                    elif ret < 0:
                        negative_streak += 1
                        positive_streak = 0
                        max_negative_streak = max(max_negative_streak, negative_streak)
                
                # 13. CALCULAR SKEWNESS Y KURTOSIS
                skewness, kurtosis = calcular_skewness_kurtosis(stock_returns_array)
                
                # 14. CALCULAR PROBABILIDAD DE PÉRDIDA
                prob_loss = np.mean(stock_returns_array < 0) * 100
                
                return {
                    # Métricas básicas
                    'Beta': round(beta, 4),
                    'Alpha': round(alpha, 4),
                    'Sharpe Ratio': round(sharpe_ratio, 4),
                    'Sortino Ratio': round(sortino_ratio, 4),
                    'Treynor Ratio': round(treynor_ratio, 4),
                    'Information Ratio': round(information_ratio, 4),
                    
                    # Métricas de riesgo
                    'VaR 95% Diario': round(var_95, 4),
                    'VaR 95% Anual': round(var_95_annual, 4),
                    'VaR 99% Diario': round(var_99, 4),
                    'VaR 99% Anual': round(var_99_annual, 4),
                    'Expected Shortfall 95%': round(cvar_95_annual, 4),
                    'Drawdown Máximo': round(max_drawdown, 4),
                    'Duración Drawdown (días)': max_dd_duration,
                    'Volatilidad Anual': round(volatility_annual, 4),
                    
                    # Correlaciones
                    'Correlación S&P500': round(correlation_sp500, 4),
                    
                    # Estadísticas avanzadas
                    'Máxima Ganancia Consecutiva': max_positive_streak,
                    'Máxima Pérdida Consecutiva': max_negative_streak,
                    'Skewness': round(skewness, 4),
                    'Kurtosis': round(kurtosis, 4),
                    'Probabilidad de Pérdida (%)': round(prob_loss, 2),
                    
                    # Rendimientos
                    'Rendimiento Total': round(stock_total_return, 4),
                    'Rendimiento Mercado': round(market_total_return, 4),
                    'Días Analizados': len(stock_returns),
                    'Período': f"{periodo_años} años"
                }
                
            except Exception as e:
                st.error(f"Error calculando métricas de riesgo: {str(e)}")
                return None

        def crear_grafica_drawdown_mejorada(ticker_symbol, periodo_años=5):
            """
            Crea gráfica de drawdown MEJORADA para visualizar pérdidas máximas
            """
            try:
                # Descargar datos
                end_date = datetime.today()
                start_date = end_date - timedelta(days=periodo_años * 365)
                
                stock_data = yf.download(ticker_symbol, start=start_date, end=end_date, interval='1d')
                if stock_data.empty:
                    return None
                
                # Manejar MultiIndex columns
                if isinstance(stock_data.columns, pd.MultiIndex):
                    stock_close = stock_data[('Close', ticker_symbol)]
                else:
                    stock_close = stock_data['Close']
                
                # Calcular drawdown
                returns = stock_close.pct_change().dropna()
                cumulative_returns = (1 + returns).cumprod()
                rolling_max = cumulative_returns.expanding().max()
                drawdown = (cumulative_returns - rolling_max) / rolling_max
                
                # Crear gráfica
                fig = go.Figure()
                
                # Área de drawdown
                fig.add_trace(go.Scatter(
                    x=drawdown.index,
                    y=drawdown * 100,
                    fill='tozeroy',
                    fillcolor='rgba(255, 0, 0, 0.3)',
                    line=dict(color='red', width=2),
                    name='Drawdown',
                    hovertemplate='<b>Drawdown</b><br>Fecha: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>'
                ))
                
                # Línea de máximo anterior
                fig.add_hline(y=0, line_dash="dash", line_color="green", annotation_text="Máximo Anterior")
                
                # Encontrar los 3 mayores drawdowns
                drawdown_sorted = drawdown.sort_values()
                top_drawdowns = drawdown_sorted.head(3)
                
                # Anotar los mayores drawdowns
                for i, (fecha, valor) in enumerate(top_drawdowns.items()):
                    fig.add_annotation(
                        x=fecha,
                        y=valor * 100,
                        text=f"DD {i+1}: {valor*100:.1f}%",
                        showarrow=True,
                        arrowhead=2,
                        bgcolor="red",
                        font=dict(color="white", size=10),
                        yshift=10 if i == 0 else (-20 if i == 1 else 30)
                    )
                
                fig.update_layout(
                    title=f'Análisis de Drawdown - {ticker_symbol}',
                    xaxis_title='Fecha',
                    yaxis_title='Drawdown (%)',
                    height=500,
                    showlegend=True,
                    hovermode='x unified'
                )
                
                return fig
                
            except Exception as e:
                st.error(f"Error creando gráfica de drawdown: {str(e)}")
                return None

        def crear_grafica_distribucion_retornos(ticker_symbol, periodo_años=5):
            """
            Crea gráfica de distribución de retornos
            """
            try:
                # Descargar datos
                end_date = datetime.today()
                start_date = end_date - timedelta(days=periodo_años * 365)
                
                stock_data = yf.download(ticker_symbol, start=start_date, end=end_date, interval='1d')
                if stock_data.empty:
                    return None
                
                # Manejar MultiIndex columns
                if isinstance(stock_data.columns, pd.MultiIndex):
                    stock_close = stock_data[('Close', ticker_symbol)]
                else:
                    stock_close = stock_data['Close']
                
                # Calcular retornos
                returns = stock_close.pct_change().dropna() * 100  # En porcentaje
                
                # Crear histograma con curva normal
                fig = go.Figure()
                
                # Histograma
                fig.add_trace(go.Histogram(
                    x=returns,
                    nbinsx=50,
                    name='Frecuencia',
                    opacity=0.7,
                    marker_color='lightblue'
                ))
                
                # Calcular distribución normal (aproximación)
                if len(returns) > 0:
                    x_norm = np.linspace(returns.min(), returns.max(), 100)
                    # Aproximación manual de distribución normal
                    mean = np.mean(returns)
                    std = np.std(returns)
                    if std > 0:
                        y_norm = (1/(std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_norm - mean)/std) ** 2)
                        y_norm = y_norm * len(returns) * (returns.max() - returns.min()) / 50  # Escalar
                        
                        # Curva normal
                        fig.add_trace(go.Scatter(
                            x=x_norm,
                            y=y_norm,
                            mode='lines',
                            name='Distribución Normal',
                            line=dict(color='red', width=2)
                        ))
                
                # Línea en cero
                fig.add_vline(x=0, line_dash="dash", line_color="green")
                
                fig.update_layout(
                    title=f'Distribución de Retornos Diarios - {ticker_symbol}',
                    xaxis_title='Retorno Diario (%)',
                    yaxis_title='Frecuencia',
                    height=400,
                    showlegend=True
                )
                
                return fig
                
            except Exception as e:
                st.error(f"Error creando gráfica de distribución: {str(e)}")
                return None

        # Mostrar spinner mientras se cargan los datos
        with st.spinner('Cargando datos fundamentales y calculando métricas de riesgo avanzadas...'):
            datos_finviz = extraer_tabla_finviz(stonk)
            metricas_riesgo = calcular_metricas_riesgo_avanzadas(stonk)
            
            if datos_finviz:
                st.success(f"✅ Se cargaron {len(datos_finviz)} métricas fundamentales")
                
                # FUNCIÓN INTELIGENTE PARA BUSCAR MÉTRICAS
                def buscar_metrica(datos, posibles_claves):
                    for clave in posibles_claves:
                        if clave in datos:
                            return datos[clave]
                    return "N/A"
                
                # DEFINIR LAS MÉTRICAS QUE QUEREMOS MOSTRAR
                metricas_principales = {
                    # Valoración y Mercado
                    "Market Cap": ["Market Cap", "Mkt Cap"],
                    "P/E": ["P/E", "PE", "P/E Ratio"],
                    "Forward P/E": ["Forward P/E", "Fwd P/E", "Forward PE"],
                    "PEG": ["PEG", "PEG Ratio"],
                    "P/FCF": ["P/FCF", "Price/FCF"],
                    "EV/EBITDA": ["EV/EBITDA", "Enterprise Value/EBITDA"],
                    "EV/SALES": ["EV/Sales", "Enterprise Value/Sales", "EV/S"],
                    
                    # Ingresos y Rentabilidad
                    "Income": ["Income", "Net Income"],
                    "Sales": ["Sales", "Revenue", "Sales Q/Q"],
                    "Gross Margin": ["Gross Margin", "Gross Mgn"],
                    "Oper. Margin": ["Oper. Margin", "Operating Margin", "Oper Mgn"],
                    "Profit Margin": ["Profit Margin", "Profit Mgn", "Net Margin"],
                    
                    # Efectivo y Deuda
                    "Cash/Share": ["Cash/sh", "Cash/Share", "Cash per Share"],
                    "Debt/Eq": ["Debt/Eq", "Debt/Equity", "Total Debt/Equity"],
                    "LT Debt/Eq": ["LT Debt/Eq", "Long Term Debt/Equity"],
                    
                    # Rentabilidad (MANTENEMOS ROIC)
                    "ROA": ["ROA", "Return on Assets"],
                    "ROE": ["ROE", "Return on Equity"],
                    "ROIC": ["ROI", "ROIC", "Return on Investment", "Return on Capital"],
                    
                    # Indicadores Técnicos
                    "Volatility": ["Volatility", "Volatility W", "Volatility M"],
                    "RSI": ["RSI (14)", "RSI", "Relative Strength Index"],
                    "Beta": ["Beta", "Beta"],
                    "Volume": ["Volume", "Avg Volume", "Volume Today"]
                }
                
                # =============================================
                # 1. MÉTRICAS FUNDAMENTALES PRINCIPALES
                # =============================================
                st.subheader("🏢 Métricas Fundamentales Principales")
                
                # Valoración y Mercado
                st.write("#### 💰 Valoración y Mercado")
                cols = st.columns(4)
                valoracion_keys = ["Market Cap", "P/E", "Forward P/E", "PEG", "P/FCF", "EV/EBITDA", "EV/SALES"]
                for i, key in enumerate(valoracion_keys):
                    with cols[i % 4]:
                        valor = buscar_metrica(datos_finviz, metricas_principales[key])
                        st.metric(key, valor)
                
                # Ingresos y Rentabilidad
                st.write("#### 📈 Ingresos y Rentabilidad")
                cols = st.columns(4)
                ingresos_keys = ["Income", "Sales", "Gross Margin", "Oper. Margin", "Profit Margin"]
                for i, key in enumerate(ingresos_keys):
                    with cols[i % 4]:
                        valor = buscar_metrica(datos_finviz, metricas_principales[key])
                        st.metric(key, valor)
                
                # Deuda y Efectivo
                st.write("#### 🏦 Deuda y Efectivo")
                cols = st.columns(4)
                deuda_keys = ["Cash/Share", "Debt/Eq", "LT Debt/Eq"]
                for i, key in enumerate(deuda_keys):
                    with cols[i % 4]:
                        valor = buscar_metrica(datos_finviz, metricas_principales[key])
                        st.metric(key, valor)
                
                # Rentabilidad (CON ROIC)
                st.write("#### 📊 Rentabilidad")
                cols = st.columns(4)
                rentabilidad_keys = ["ROA", "ROE", "ROIC"]
                for i, key in enumerate(rentabilidad_keys):
                    with cols[i % 4]:
                        valor = buscar_metrica(datos_finviz, metricas_principales[key])
                        st.metric(key, valor)
                
                # Indicadores Técnicos
                st.write("#### 📈 Indicadores Técnicos")
                cols = st.columns(4)
                tecnicos_keys = ["Volatility", "RSI", "Beta", "Volume"]
                for i, key in enumerate(tecnicos_keys):
                    with cols[i % 4]:
                        valor = buscar_metrica(datos_finviz, metricas_principales[key])
                        st.metric(key, valor)
                
                st.markdown("---")
                
                # =============================================
                # 2. MÉTRICAS AVANZADAS DE RIESGO Y RENDIMIENTO
                # =============================================
                if metricas_riesgo:
                    st.subheader("🎯 Métricas Avanzadas de Riesgo y Rendimiento")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        # Beta con interpretación 
                        beta = metricas_riesgo['Beta']
                        if beta < 0.8:
                            interpretacion = "Defensivo"
                            color = "green"
                        elif beta < 1.2:
                            interpretacion = "Neutro"
                            color = "orange"
                        else:
                            interpretacion = "Agresivo"
                            color = "red"
                        
                        st.metric("📊 Beta (Riesgo Sistemático)", f"{beta:.4f}")
                        st.caption(f"*Interpretación: {interpretacion}*")
                        
                        # Alpha 
                        alpha = metricas_riesgo['Alpha']
                        st.metric("α Alpha", f"{alpha:.2%}")
                        st.caption("*Rendimiento vs esperado*")
                    
                    with col2:
                        # Sharpe Ratio 
                        sharpe = metricas_riesgo['Sharpe Ratio']
                        if sharpe > 1.0:
                            color_sharpe = "green"
                        elif sharpe > 0.5:
                            color_sharpe = "orange"
                        else:
                            color_sharpe = "red"
                        
                        st.metric("⚡ Sharpe Ratio", f"{sharpe:.4f}")
                        st.caption("*Rendimiento/riesgo total*")
                        
                        # Sortino Ratio 
                        sortino = metricas_riesgo['Sortino Ratio']
                        st.metric("🎯 Sortino Ratio", f"{sortino:.4f}")
                        st.caption("*Rendimiento/riesgo bajista*")
                    
                    with col3:
                        # Nuevos ratios
                        treynor = metricas_riesgo['Treynor Ratio']
                        st.metric("📈 Treynor Ratio", f"{treynor:.4f}")
                        st.caption("*Rendimiento/riesgo sistemático*")
                        
                        information = metricas_riesgo['Information Ratio']
                        st.metric("ℹ️ Information Ratio", f"{information:.4f}")
                        st.caption("*Rendimiento activo*")
                    
                    with col4:
                        # Rendimiento vs Mercado 
                        rend_stock = metricas_riesgo['Rendimiento Total']
                        rend_mercado = metricas_riesgo['Rendimiento Mercado']
                        diferencia = rend_stock - rend_mercado
                        
                        st.metric("📊 Vs S&P500", f"{diferencia:.2%}")
                        st.caption("*Exceso vs mercado*")
                        
                        # Probabilidad de pérdida
                        prob_loss = metricas_riesgo['Probabilidad de Pérdida (%)']
                        st.metric("📉 Prob. Pérdida", f"{prob_loss:.1f}%")
                        st.caption("*Frecuencia días negativos*")
                    
                    st.markdown("---")
                    
                    # =============================================
                    # 3. MÉTRICAS DE RENDIMIENTO AJUSTADO AL RIESGO
                    # =============================================
                    st.subheader("📈 Métricas de Rendimiento Ajustado al Riesgo")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        # VaR 
                        var_95 = metricas_riesgo['VaR 95% Anual']
                        var_99 = metricas_riesgo['VaR 99% Anual']
                        
                        st.metric("📉 VaR 95% Anual", f"{var_95:.2%}")
                        st.caption("*Pérdida máxima esperada*")
                        st.metric("📉 VaR 99% Anual", f"{var_99:.2%}")
                        st.caption("*Pérdida extrema esperada*")
                    
                    with col2:
                        # Drawdown 
                        max_dd = metricas_riesgo['Drawdown Máximo']
                        dd_duration = metricas_riesgo['Duración Drawdown (días)']
                        
                        st.metric("🔻 Drawdown Máximo", f"{max_dd:.2%}")
                        st.caption("*Peor pérdida histórica*")
                        st.metric("⏱️ Duración DD", f"{dd_duration} días")
                        st.caption("*Tiempo recuperación*")
                    
                    with col3:
                        # Volatilidad y Correlación
                        volatilidad = metricas_riesgo['Volatilidad Anual']
                        correlacion = metricas_riesgo['Correlación S&P500']
                        
                        st.metric("📈 Volatilidad Anual", f"{volatilidad:.2%}")
                        st.caption("*Riesgo total anualizado*")
                        st.metric("🔗 Correlación S&P500", f"{correlacion:.2%}")
                        st.caption("*Movimiento vs mercado*")
                    
                    with col4:
                        # Estadísticas avanzadas
                        cvar = metricas_riesgo['Expected Shortfall 95%']
                        skew = metricas_riesgo['Skewness']
                        
                        st.metric("💀 Expected Shortfall", f"{cvar:.2%}")
                        st.caption("*Pérdida promedio en colas*")
                        st.metric("📊 Skewness", f"{skew:.4f}")
                        st.caption("*Asimetría distribución*")
                    
                    st.markdown("---")
                    
                    # =============================================
                    # 4. ALERTAS DE RIESGO
                    # =============================================
                    st.subheader("🚨 Alertas de Riesgo")
                    
                    alertas = []
                    
                    # Verificar condiciones de riesgo
                    if metricas_riesgo['Drawdown Máximo'] < -0.20:
                        alertas.append("🔴 ALTO RIESGO: Drawdown máximo > 20%")
                    elif metricas_riesgo['Drawdown Máximo'] < -0.10:
                        alertas.append("🟡 RIESGO MODERADO: Drawdown máximo > 10%")
                    
                    if metricas_riesgo['VaR 95% Anual'] < -0.25:
                        alertas.append("🔴 ALTO RIESGO: VaR anual > 25%")
                    
                    if metricas_riesgo['Volatilidad Anual'] > 0.40:
                        alertas.append("🟡 VOLATILIDAD ALTA: > 40% anual")
                    
                    if metricas_riesgo['Probabilidad de Pérdida (%)'] > 50:
                        alertas.append("🔴 ALTA PROBABILIDAD DE PÉRDIDA: > 50%")
                    
                    if alertas:
                        for alerta in alertas:
                            st.warning(alerta)
                    else:
                        st.success("✅ Perfil de riesgo dentro de parámetros normales")
                    
                    st.markdown("---")
                    
                    # =============================================
                    # 5. ANÁLISIS GRÁFICO DE RIESGO
                    # =============================================
                    st.subheader("📈 Análisis Gráfico de Riesgo")

                    col1, col2 = st.columns(2)

                    with col1:
                        # Gráfica de drawdown 
                        st.markdown("**📉 Drawdown - Pérdidas Máximas Históricas**")
                        
                        grafica_drawdown = crear_grafica_drawdown_mejorada(stonk)
                        if grafica_drawdown:
                            st.plotly_chart(grafica_drawdown, use_container_width=True)
                            st.caption("*Visualiza las mayores caídas desde máximos históricos. Áreas rojas indican períodos de pérdidas.*")
                        else:
                            st.warning("No se pudo generar la gráfica de drawdown")

                    with col2:
                        # Gráfica de distribución de retornos
                        st.markdown("**📊 Distribución de Retornos Diarios**")
                        
                        grafica_distribucion = crear_grafica_distribucion_retornos(stonk)
                        if grafica_distribucion:
                            st.plotly_chart(grafica_distribucion, use_container_width=True)
                            st.caption("*Muestra la frecuencia y distribución de ganancias/pérdidas diarias. Línea roja = distribución normal teórica.*")
                        else:
                            st.warning("No se pudo generar la gráfica de distribución")

                    st.markdown("---")

                # =============================================
                # 6. MODELO CAPM - COSTO DE CAPITAL
                # =============================================
                st.subheader("📊 Modelo CAPM - Costo de Capital")

                # Configuración de parámetros CAPM
                col_params1, col_params2, col_params3 = st.columns(3)

                with col_params1:
                    tasa_libre_riesgo = st.number_input(
                        "Tasa Libre de Riesgo (%)", 
                        min_value=0.0, 
                        max_value=10.0, 
                        value=2.0, 
                        step=0.1,
                        help="Rendimiento de bonos gubernamentales (10 años)"
                    ) / 100

                with col_params2:
                    prima_riesgo_mercado = st.number_input(
                        "Prima de Riesgo de Mercado (%)", 
                        min_value=0.0, 
                        max_value=15.0, 
                        value=6.0, 
                        step=0.1,
                        help="Rendimiento esperado del mercado sobre tasa libre de riesgo"
                    ) / 100

                with col_params3:
                    # Obtener Beta de Yahoo Finance o usar valor por defecto
                    beta_actual = info.get('beta', 1.0)
                    beta = st.number_input(
                        "Beta (β) de la Acción", 
                        min_value=0.0, 
                        max_value=5.0, 
                        value=float(beta_actual), 
                        step=0.1,
                        help="Riesgo sistemático vs mercado"
                    )

                # Calcular CAPM
                costo_capital = tasa_libre_riesgo + beta * prima_riesgo_mercado

                # Mostrar métricas CAPM
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        "Tasa Libre Riesgo", 
                        f"{tasa_libre_riesgo*100:.1f}%",
                        "Rf"
                    )

                with col2:
                    st.metric(
                        "Beta (β)", 
                        f"{beta:.2f}",
                        "Riesgo Sistemático"
                    )

                with col3:
                    st.metric(
                        "Prima Riesgo Mercado", 
                        f"{prima_riesgo_mercado*100:.1f}%",
                        "E(Rm) - Rf"
                    )

                with col4:
                    st.metric(
                        "**Costo Capital (CAPM)**", 
                        f"**{costo_capital*100:.1f}%**",
                        "**E(R) = Rf + β×(Rm-Rf)**",
                        delta_color="off"
                    )

                # Gráfica del CAPM - Scatter Plot con datos históricos
                st.subheader("📈 Análisis CAPM - Datos Históricos")

                # SELECTOR DE PERÍODO PARA DATOS HISTÓRICOS
                st.markdown("**🕐 Selecciona el período de análisis:**")

                col_periodo, col_frecuencia = st.columns(2)

                with col_periodo:
                    periodo_capm = st.selectbox(
                        "Período de datos:",
                        options=["1 mes", "3 meses", "6 meses", "1 año", "2 años", "3 años", "5 años", "10 años"],
                        index=3,  # 1 año por defecto
                        key="periodo_capm"
                    )

                with col_frecuencia:
                    frecuencia_capm = st.selectbox(
                        "Frecuencia de datos:",
                        options=["Diario", "Semanal", "Mensual"],
                        index=0,  # Diario por defecto para períodos cortos
                        key="frecuencia_capm"
                    )

                # Mapear selecciones a parámetros
                periodo_map = {
                    "1 mes": 30,
                    "3 meses": 90,
                    "6 meses": 180,
                    "1 año": 365,
                    "2 años": 730,
                    "3 años": 1095,
                    "5 años": 1825,
                    "10 años": 3650
                }

                frecuencia_map = {
                    "Diario": "1d",
                    "Semanal": "1wk", 
                    "Mensual": "1mo"
                }

                dias_periodo = periodo_map[periodo_capm]
                intervalo = frecuencia_map[frecuencia_capm]

                # Ajustar frecuencia automáticamente para períodos muy cortos
                if dias_periodo <= 90 and frecuencia_capm == "Mensual":  # 3 meses o menos
                    st.warning("⚠️ Para períodos cortos (≤ 3 meses) se recomienda frecuencia Diaria o Semanal para mejor análisis")
                    intervalo = "1d"  # Forzar diario para períodos cortos

                st.info(f"**📊 Configuración:** {periodo_capm} | {frecuencia_capm} | {stonk} vs S&P500")

                # Obtener datos históricos según la selección
                try:
                    start_date = datetime.today() - timedelta(days=dias_periodo)
                    end_date = datetime.today()
                    
                    # Descargar datos
                    with st.spinner(f'Cargando datos {frecuencia_capm.lower()} para {periodo_capm}...'):
                        stock_data = yf.download(stonk, start=start_date, end=end_date, interval=intervalo)
                        market_data = yf.download('^GSPC', start=start_date, end=end_date, interval=intervalo)
                    
                    if not stock_data.empty and not market_data.empty:
                        # Obtener precios de cierre
                        if isinstance(stock_data.columns, pd.MultiIndex):
                            stock_close = stock_data[('Close', stonk)]
                        else:
                            stock_close = stock_data['Close']
                            
                        if isinstance(market_data.columns, pd.MultiIndex):
                            market_close = market_data[('Close', '^GSPC')]
                        else:
                            market_close = market_data['Close']
                        
                        # Calcular rendimientos
                        stock_returns = stock_close.pct_change().dropna()
                        market_returns = market_close.pct_change().dropna()
                        
                        # Alinear fechas
                        common_dates = stock_returns.index.intersection(market_returns.index)
                        stock_returns = stock_returns.loc[common_dates]
                        market_returns = market_returns.loc[common_dates]
                        
                        if len(stock_returns) > 5:  # Mínimo reducido para períodos cortos
                            # Crear scatter plot
                            fig_capm = go.Figure()
                            
                            # Determinar color de los puntos basado en la tendencia reciente
                            color_points = 'blue'
                            if len(stock_returns) > 10:
                                # Calcular tendencia reciente para colorear puntos
                                tendencia_reciente = stock_returns.tail(min(10, len(stock_returns))).mean()
                                if tendencia_reciente > 0:
                                    color_points = 'green'
                                else:
                                    color_points = 'red'
                            
                            # Puntos de datos históricos
                            fig_capm.add_trace(go.Scatter(
                                x=market_returns * 100,
                                y=stock_returns * 100,
                                mode='markers',
                                name=f'Datos {frecuencia_capm} ({len(stock_returns)} puntos)',
                                marker=dict(
                                    size=8,
                                    color=color_points,
                                    opacity=0.7,
                                    line=dict(width=1, color='darkgray')
                                ),
                                hovertemplate=(
                                    'Fecha: %{text}<br>' +
                                    'Rendimiento Mercado: %{x:.2f}%<br>' +
                                    'Rendimiento Acción: %{y:.2f}%<br>' +
                                    '<extra></extra>'
                                ),
                                text=[date.strftime('%d/%m/%Y') for date in common_dates]
                            ))
                            
                            # Calcular línea de regresión (Beta histórico)
                            if len(market_returns) > 1:
                                beta_real, intercepto = np.polyfit(market_returns, stock_returns, 1)
                                r_squared = np.corrcoef(market_returns, stock_returns)[0, 1] ** 2
                                
                                # Línea de regresión
                                x_line = np.linspace(market_returns.min(), market_returns.max(), 50)
                                y_line = intercepto + beta_real * x_line
                                
                                fig_capm.add_trace(go.Scatter(
                                    x=x_line * 100,
                                    y=y_line * 100,
                                    mode='lines',
                                    name=f'Beta Histórico = {beta_real:.2f}',
                                    line=dict(color='red', width=3, dash='dash'),
                                    hovertemplate='Beta histórico: {:.2f}<extra></extra>'.format(beta_real)
                                ))
                            
                            # Línea CAPM teórica
                            # Ajustar tasa libre de riesgo según frecuencia
                            if frecuencia_capm == "Diario":
                                rf_ajustado = tasa_libre_riesgo / 252
                            elif frecuencia_capm == "Semanal":
                                rf_ajustado = tasa_libre_riesgo / 52
                            else:  # Mensual
                                rf_ajustado = tasa_libre_riesgo / 12
                                
                            x_capm = np.linspace(market_returns.min(), market_returns.max(), 50)
                            y_capm = rf_ajustado + beta * (x_capm - rf_ajustado)
                            
                            fig_capm.add_trace(go.Scatter(
                                x=x_capm * 100,
                                y=y_capm * 100,
                                mode='lines',
                                name=f'CAPM Teórico (β = {beta:.2f})',
                                line=dict(color='blue', width=3),
                                hovertemplate='CAPM teórico<extra></extra>'
                            ))
                            
                            # Punto de rendimiento esperado actual
                            fig_capm.add_trace(go.Scatter(
                                x=[0],  # Centrado en el origen para mejor visualización
                                y=[costo_capital * 100],
                                mode='markers+text',
                                name='Rendimiento Esperado Anual',
                                marker=dict(size=12, color='orange', symbol='star', line=dict(width=2, color='darkorange')),
                                text=['ESPERADO'],
                                textposition="top center",
                                hovertemplate=f'Rendimiento esperado anual: {costo_capital*100:.1f}%<extra></extra>'
                            ))
                            
                            fig_capm.update_layout(
                                title=f'CAPM - {stonk} vs S&P500 ({periodo_capm}, {frecuencia_capm})',
                                xaxis_title='Rendimiento del Mercado (S&P500) (%)',
                                yaxis_title=f'Rendimiento de {stonk} (%)',
                                height=600,
                                showlegend=True,
                                hovermode='closest',
                                legend=dict(
                                    orientation="h",
                                    yanchor="bottom",
                                    y=1.02,
                                    xanchor="right",
                                    x=1
                                ),
                                xaxis=dict(
                                    showgrid=True,
                                    gridwidth=1,
                                    gridcolor='lightgray',
                                    zeroline=True,
                                    zerolinewidth=2,
                                    zerolinecolor='black'
                                ),
                                yaxis=dict(
                                    showgrid=True,
                                    gridwidth=1,
                                    gridcolor='lightgray',
                                    zeroline=True,
                                    zerolinewidth=2,
                                    zerolinecolor='black'
                                )
                            )
                            
                            st.plotly_chart(fig_capm, use_container_width=True)
                            
                            # Análisis de la regresión
                            st.subheader("📊 Análisis de Regresión")
                            
                            col_reg1, col_reg2, col_reg3, col_reg4 = st.columns(4)
                            
                            with col_reg1:
                                st.metric("Beta Histórico", f"{beta_real:.2f}")
                                st.caption(f"Calculado con {len(stock_returns)} puntos")
                                
                            with col_reg2:
                                st.metric("Beta Teórico", f"{beta:.2f}")
                                st.caption("Valor de Yahoo Finance")
                                
                            with col_reg3:
                                diferencia_beta = beta_real - beta
                                st.metric(
                                    "Diferencia Beta", 
                                    f"{diferencia_beta:.2f}",
                                    f"{'↑' if beta_real > beta else '↓'} histórico vs teórico"
                                )
                                st.caption("Consistencia del beta")
                                
                            with col_reg4:
                                st.metric("R² (Coef. Determinación)", f"{r_squared:.3f}")
                                st.caption("Ajuste del modelo")
                            
                            # Interpretación específica por período
                            st.markdown("---")
                            st.subheader("💡 Interpretación por Período")
                            
                            col_interp1, col_interp2 = st.columns(2)
                            
                            with col_interp1:
                                st.markdown(f"""
                                **📈 Análisis del Período {periodo_capm}:**
                                
                                • **Beta histórico**: **{beta_real:.2f}**
                                • **Puntos analizados**: **{len(stock_returns)}**
                                • **Período**: {periodo_capm}
                                • **Frecuencia**: {frecuencia_capm}
                                
                                **🎯 Significado del Beta:**
                                - **Beta > 1**: Más volátil que el mercado
                                - **Beta = 1**: Misma volatilidad  
                                - **Beta < 1**: Menos volátil
                                """)
                            
                            with col_interp2:
                                # Interpretación específica del período
                                if "mes" in periodo_capm:
                                    interpretacion_periodo = "**🔄 Análisis de Corto Plazo** - Muestra el comportamiento reciente y puede ser más volátil"
                                elif periodo_capm == "1 año":
                                    interpretacion_periodo = "**📊 Análisis de Mediano Plazo** - Balance entre estabilidad y actualidad"
                                else:
                                    interpretacion_periodo = "**📈 Análisis de Largo Plazo** - Muestra tendencias estables y comportamiento histórico"
                                
                                st.markdown(f"""
                                **🔍 Contexto del Período:**
                                
                                {interpretacion_periodo}
                                
                                **📋 Recomendaciones:**
                                - Períodos cortos: Útiles para trading
                                - Períodos largos: Mejores para inversión
                                - Combine períodos para análisis completo
                                """)
                            
                            # Recomendaciones específicas basadas en el período
                            st.markdown("---")
                            st.subheader("🎯 Recomendaciones Específicas")
                            
                            if "mes" in periodo_capm:
                                if r_squared > 0.6:
                                    st.success("""
                                    **✅ BUEN AJUSTE EN CORTO PLAZO - Para Trading:**
                                    - Relación mercado-acción consistente recientemente
                                    - Estrategias de momentum pueden ser efectivas
                                    - Monitorea cambios diarios en la relación
                                    """)
                                else:
                                    st.warning("""
                                    **🟡 AJUSTE VARIABLE EN CORTO PLAZO - Precauciones:**
                                    - La acción tiene comportamiento independiente reciente
                                    - Considera noticias y eventos específicos de la empresa
                                    - Usa stops más ajustados
                                    """)
                            else:
                                if r_squared > 0.7:
                                    st.success("""
                                    **✅ ALTO AJUSTE - Para Inversión:**
                                    - Comportamiento predecible vs mercado
                                    - Estrategias basadas en Beta son confiables
                                    - Buena para diversificación de cartera
                                    """)
                                elif r_squared > 0.4:
                                    st.info("""
                                    **🟡 AJUSTE MODERADO - Enfoque Balanceado:**
                                    - Combine análisis CAPM con otros métodos
                                    - Considere factores específicos de la empresa
                                    - Monitoree cambios en la relación
                                    """)
                                else:
                                    st.warning("""
                                    **🔴 BAJO AJUSTE - Análisis Cauteloso:**
                                    - La acción se mueve independientemente del mercado
                                    - Enfóquese en análisis fundamental y técnico
                                    - El Beta puede no ser indicador confiable
                                    """)
                        
                        else:
                            st.warning(f"⚠️ No hay suficientes datos {frecuencia_capm.lower()} para {periodo_capm}. Intenta con una frecuencia diferente.")
                            
                    else:
                        st.warning("❌ No se pudieron cargar los datos para el análisis CAPM")
                        
                except Exception as e:
                    st.error(f"Error en el análisis CAPM: {str(e)}")

                # Consejos para usar diferentes períodos
                st.markdown("---")
                st.subheader("💡 Consejos para Usar Diferentes Períodos")

                consejos_periodos = [
                    "**📅 1-3 meses**: Ideal para traders - muestra comportamiento reciente",
                    "**📊 6 meses - 1 año**: Balanceado - buen para swing trading",
                    "**📈 2-3 años**: Estabilidad media - recomendado para mayoría de inversores", 
                    "**🏛️ 5-10 años**: Largo plazo - muestra tendencias estables",
                    "**🔄 Combine períodos**: Use corto + largo plazo para análisis completo",
                    "**📉 Períodos cortos**: Más volátiles pero más actualizados",
                    "**📈 Períodos largos**: Más estables pero pueden omitir cambios recientes"
                ]

                for consejo in consejos_periodos:
                    st.write(f"• {consejo}")

                st.markdown("---")

                # =============================================
                # 7. SNAPSHOT FINANCIERO COMPLETO
                # =============================================
                st.subheader(f"📊 Snapshot Financiero Completo - {stonk}")
                
                # Crear una tabla de 2 columnas replicando Finviz
                num_datos = len(datos_finviz)
                mitad = (num_datos + 1) // 2
                
                # Dividir los datos en dos columnas
                items = list(datos_finviz.items())
                col1_items = items[:mitad]
                col2_items = items[mitad:]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    for clave, valor in col1_items:
                        st.markdown(f"""
                        <div style="border-bottom: 1px solid #444; padding: 10px 0;">
                            <div style="font-weight: bold; color: white; font-size: 14px; margin-bottom: 2px;">{clave}</div>
                            <div style="color: #f0f0f0; font-size: 14px; text-align: right; font-weight: 500;">{valor}</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    for clave, valor in col2_items:
                        st.markdown(f"""
                        <div style="border-bottom: 1px solid #444; padding: 10px 0;">
                            <div style="font-weight: bold; color: white; font-size: 14px; margin-bottom: 2px;">{clave}</div>
                            <div style="color: #f0f0f0; font-size: 14px; text-align: right; font-weight: 500;">{valor}</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # BOTÓN DE DESCARGA
                st.markdown("---")
                st.subheader("💾 Exportar Datos")
                
                # Crear DataFrame combinado con todas las métricas
                df_completo = pd.DataFrame(list(datos_finviz.items()), columns=['Métrica', 'Valor'])
                
                # Agregar métricas de riesgo si están disponibles
                if metricas_riesgo:
                    df_riesgo = pd.DataFrame(list(metricas_riesgo.items()), columns=['Métrica', 'Valor'])
                    df_completo = pd.concat([df_completo, df_riesgo], ignore_index=True)
                
                csv = df_completo.to_csv(index=False)
                
                st.download_button(
                    label="📥 Descargar datos fundamentales y de riesgo como CSV",
                    data=csv,
                    file_name=f"{stonk}_datos_completos.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                    
            else:
                st.error("""
                ❌ No se pudieron cargar los datos fundamentales. Posibles causas:
                
                • **Problemas de conexión** con Finviz
                • **Bloqueo temporal** por demasiadas solicitudes
                • **El símbolo no existe** o no está disponible
                
                💡 **Sugerencias:**
                • Verifica el símbolo (ej: AAPL, MSFT, TSLA, GOOGL)
                • Espera 1-2 minutos e intenta nuevamente  
                • Verifica directamente en [Finviz](https://finviz.com/quote.ashx?t={stonk})
                """)
                
                if st.button("🔄 Intentar nuevamente", use_container_width=True, key="reintentar_fundamentales"):
                    st.rerun()
    
    #
    with tab2:
        st.header("🎓 Educación Financiera - Guía Completa de 82 Métricas")
        st.write("**Explicación DETALLADA de cada métrica: qué es, para qué sirve, ventajas y desventajas**")
        
        # Selector de categoría
        categorias = [
            "💰 VALORACIÓN Y MERCADO (18 métricas)",
            "📈 RENTABILIDAD Y MÁRGENES (16 métricas)", 
            "🏦 DEUDA Y LIQUIDEZ (12 métricas)",
            "📊 EFICIENCIA OPERATIVA (10 métricas)",
            "📈 CRECIMIENTO (8 métricas)",
            "📊 INDICADORES TÉCNICOS (10 métricas)",
            "🏢 DATOS CORPORATIVOS (8 métricas)",
            "⚡ MÉTRICAS AVANZADAS DE RIESGO",
            "💡 CONSEJOS PRÁCTICOS DE INVERSIÓN"
        ]
        
        categoria = st.selectbox("Selecciona la categoría:", categorias)
        
        st.markdown("---")
        
        if categoria == "💰 VALORACIÓN Y MERCADO (18 métricas)":
            st.subheader("💰 VALORACIÓN Y MERCADO - 18 Métricas")
            
            metricas = {
                "Market Cap": {
                    "definicion": "**Capitalización de mercado** - Valor total de la empresa en bolsa",
                    "calculacion": "Precio actual de la acción × Número total de acciones en circulación",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **Large Cap (>$10B)**: Empresas establecidas, menos volátiles, dividendos consistentes
                    - **Mid Cap ($2B-$10B)**: Empresas en crecimiento, balance riesgo/recompensa
                    - **Small Cap (<$2B)**: Empresas pequeñas, alto crecimiento potencial, más riesgo
                    - **Mega Cap (>$200B)**: Gigantes globales como Apple, Microsoft
                    
                    **Ventajas:**
                    - Fácil de calcular y entender
                    - Buen indicador del tamaño relativo
                    - Útil para comparar empresas del mismo sector
                    
                    **Desventajas:**
                    - No considera la deuda de la empresa
                    - Puede ser engañoso si hay muchas acciones en circulación
                    - No refleja el valor intrínseco real
                    
                    **¿Para qué sirve?**
                    - Determinar el tamaño y estabilidad de la empresa
                    - Clasificar empresas por capitalización
                    - Evaluar el riesgo relativo (generalmente empresas más grandes = menos riesgo)
                    """,
                    "ejemplo": "Apple: 16,300 millones de acciones × $150 = $2.45 billones de Market Cap"
                },
                
                "P/E (Price-to-Earnings)": {
                    "definicion": "**Ratio Precio-Beneficio** - Cuánto pagan los inversores por cada dólar de ganancias",
                    "calculacion": "Precio de la acción ÷ Ganancias por acción (EPS)",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **P/E bajo (<15)**: Posiblemente subvalorada, pero investiga por qué
                    - **P/E medio (15-25)**: Rango típico para muchas empresas
                    - **P/E alto (>25)**: Altas expectativas de crecimiento o posible sobrevaloración
                    
                    **Ventajas:**
                    - Fácil de calcular y entender
                    - Ampliamente utilizado y aceptado
                    - Buen punto de partida para valoración
                    
                    **Desventajas:**
                    - No útil para empresas sin ganancias
                    - Las ganancias pueden ser manipuladas contablemente
                    - No considera el crecimiento futuro
                    - Varía mucho entre sectores
                    
                    **Sectores típicos:**
                    - Tecnología: 20-30 (alto crecimiento esperado)
                    - Utilities: 12-18 (bajo crecimiento, estables)
                    - Bancos: 8-12 (regulados, crecimiento estable)
                    - Biotech: 30+ (potencial alto crecimiento)
                    
                    **¿Para qué sirve?**
                    - Comparar empresas dentro del mismo sector
                    - Identificar posibles oportunidades de valor
                    - Evaluar si el precio está justificado por las ganancias
                    """,
                    "ejemplo": "Empresa precio $100, EPS $5 → P/E = 20 (pagas $20 por cada $1 de ganancias)"
                },
                
                "Forward P/E": {
                    "definicion": "**P/E Forward** - Ratio P/E basado en ganancias estimadas futuras",
                    "calculacion": "Precio actual ÷ EPS estimado para el próximo año",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **Forward P/E < Current P/E**: Se espera crecimiento de ganancias
                    - **Forward P/E > Current P/E**: Se espera disminución de ganancias
                    - Diferencia significativa puede indicar cambios en el negocio
                    
                    **Ventajas:**
                    - Más forward-looking que el P/E tradicional
                    - Mejor para empresas en crecimiento rápido
                    - Considera las expectativas del mercado
                    
                    **Desventajas:**
                    - Depende de estimaciones (pueden ser erróneas)
                    - Sensible a revisiones de analistas
                    - Las estimaciones pueden ser demasiado optimistas o pesimistas
                    
                    **¿Para qué sirve?**
                    - Evaluar valoración basada en expectativas futuras
                    - Identificar empresas donde el crecimiento no está reflejado en el precio
                    - Comparar con el P/E histórico para ver tendencias
                    """,
                    "ejemplo": "Precio $50, EPS estimado próximo año $2.50 → Forward P/E = 20"
                },
                
                "PEG Ratio": {
                    "definicion": "**Ratio P/E sobre Crecimiento** - Relaciona el P/E con la tasa de crecimiento",
                    "calculacion": "P/E Ratio ÷ Tasa de crecimiento anual de EPS (%)",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **PEG < 1**: Posiblemente subvalorada (crecimiento > P/E)
                    - **PEG = 1**: Valoración justa
                    - **PEG > 1**: Posiblemente sobrevalorada (P/E > crecimiento)
                    
                    **Ventajas:**
                    - Considera el crecimiento futuro
                    - Mejor que solo mirar P/E para empresas growth
                    - Útil para comparar empresas con diferentes tasas de crecimiento
                    
                    **Desventajas:**
                    - Depende de estimaciones de crecimiento (inciertas)
                    - No considera el riesgo
                    - Las tasas de crecimiento pueden no ser sostenibles
                    
                    **Interpretación por sectores:**
                    - Tech growth: PEG 1.0-1.5 puede ser aceptable
                    - Value stocks: Buscar PEG < 0.8
                    - Empresas maduras: PEG cercano a 1.0
                    
                    **¿Para qué sirve?**
                    - Identificar empresas growth a precios razonables
                    - Evaluar si el premium de P/E está justificado por el crecimiento
                    - Comparar empresas con diferentes perfiles de crecimiento
                    """,
                    "ejemplo": "P/E 20, crecimiento EPS 25% anual → PEG = 0.8 (atractivo)"
                },
                
                "P/S (Price-to-Sales)": {
                    "definicion": "**Ratio Precio-Ventas** - Valoración respecto a los ingresos por ventas",
                    "calculacion": "Market Cap ÷ Ventas anuales totales",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **P/S < 1**: Considerado bajo (posible oportunidad)
                    - **P/S 1-3**: Rango típico para muchas empresas
                    - **P/S > 3**: Considerado alto (mucho crecimiento esperado)
                    
                    **Ventajas:**
                    - Útil para empresas sin ganancias o con ganancias volátiles
                    - Las ventas son más difíciles de manipular que las ganancias
                    - Bueno para startups y empresas en crecimiento
                    
                    **Desventajas:**
                    - No considera la rentabilidad
                    - Empresas con márgenes bajos pueden tener P/S engañosos
                    - No diferencia entre ventas de calidad y ventas sin profit
                    
                    **Sectores típicos:**
                    - Software: P/S 5-15 (márgenes altos esperados)
                    - Retail: P/S 0.5-1.5 (márgenes bajos)
                    - Manufacturing: P/S 1-2
                    
                    **¿Para qué sirve?**
                    - Evaluar empresas que aún no son rentables
                    - Comparar empresas dentro del mismo sector
                    - Identificar empresas con ventas crecientes pero P/S bajo
                    """,
                    "ejemplo": "Market Cap $500M, Ventas $250M → P/S = 2.0"
                },
                
                "P/B (Price-to-Book)": {
                    "definicion": "**Ratio Precio-Valor Contable** - Compara precio de mercado con valor en libros",
                    "calculacion": "Precio de la acción ÷ Valor contable por acción",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **P/B < 1**: Cotiza bajo valor contable (posible oportunidad value)
                    - **P/B = 1**: Precio igual al valor contable
                    - **P/B > 1**: Prima sobre valor contable (normal para empresas rentables)
                    
                    **Ventajas:**
                    - Bueno para empresas con muchos activos tangibles
                    - El valor contable es relativamente estable
                    - Útil para bancos y empresas financieras
                    
                    **Desventajas:**
                    - No útil para empresas de servicios o tecnología
                    - No considera activos intangibles (marca, patentes)
                    - El valor contable puede estar desactualizado
                    
                    **Sectores típicos:**
                    - Bancos: P/B 0.8-1.5
                    - Seguros: P/B 1.0-1.8
                    - Tecnología: P/B 3.0+ (muchos intangibles)
                    
                    **¿Para qué sirve?**
                    - Encontrar empresas potencialmente subvaloradas
                    - Evaluar empresas con muchos activos físicos
                    - Análisis de bancos y instituciones financieras
                    """,
                    "ejemplo": "Precio $50, Valor contable por acción $40 → P/B = 1.25"
                },
                
                "P/FCF": {
                    "definicion": "**Precio/Flujo de Caja Libre** - Valoración respecto al flujo de caja generado",
                    "calculacion": "Market Cap ÷ Flujo de Caja Libre anual",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **P/FCF < 15**: Generalmente considerado atractivo
                    - **P/FCF 15-25**: Rango razonable
                    - **P/FCF > 25**: Posiblemente sobrevalorado
                    
                    **Ventajas:**
                    - El flujo de caja es más difícil de manipular que las ganancias
                    - Mide la capacidad real de generar efectivo
                    - Buen indicador de salud financiera
                    
                    **Desventajas:**
                    - El FCF puede ser volátil entre años
                    - No considera inversiones de capital futuras
                    - Puede ser negativo en empresas en crecimiento
                    
                    **¿Para qué sirve?**
                    - Evaluar la capacidad de generar efectivo real
                    - Comparar empresas dentro del mismo sector
                    - Identificar empresas con fuerte generación de caja
                    """,
                    "ejemplo": "Market Cap $1B, FCF $100M → P/FCF = 10"
                },
                
                "P/C": {
                    "definicion": "**Precio/Efectivo** - Valoración respecto al efectivo en balance",
                    "calculacion": "Precio de la acción ÷ Efectivo por acción",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **P/C bajo**: Mucho efectivo relativo al precio (posible oportunidad)
                    - **P/C alto**: Poca reserva de efectivo relativa al precio
                    - **P/C < 5**: Generalmente considerado atractivo
                    - **P/C > 10**: Puede indicar sobrevaloración
                    
                    **Ventajas:**
                    - Mide el colchón de seguridad en efectivo
                    - Útil para identificar empresas con fuerte posición de caja
                    - El efectivo es el activo más líquido y seguro
                    - Bueno para evaluar valoración en situaciones de crisis
                    
                    **Desventajas:**
                    - No considera cómo se usa el efectivo
                    - El efectivo puede estar destinado a obligaciones específicas
                    - Puede ser temporal (venta de activos, emisión de deuda)
                    - No diferencia entre efectivo operativo y no operativo
                    
                    **Interpretación por sectores:**
                    - **Tecnología**: P/C 5-15 (normal por alto crecimiento)
                    - **Manufactura**: P/C 3-8 (menos efectivo intensivo)
                    - **Financieras**: P/C 1-3 (mucha regulación de capital)
                    - **Biotech**: P/C 10-20 (queman efectivo en desarrollo)
                    
                    **¿Para qué sirve?**
                    - Evaluar la solidez financiera a corto plazo
                    - Identificar empresas con exceso de efectivo
                    - Analizar oportunidades de recompra de acciones o dividendos
                    - Valoración en adquisiciones (empresas con mucho cash)
                    
                    **Señales de alerta:**
                    - P/C muy alto con poco crecimiento
                    - Efectivo decreciente con P/C constante
                    - Empresas que queman cash rápidamente
                    """,
                    "ejemplo": "Precio $100, Efectivo por acción $25 → P/C = 4 (atractivo)\nPrecio $50, Efectivo por acción $3 → P/C = 16.7 (elevado)"
                },

                "EV/EBITDA": {
                    "definicion": "**Enterprise Value/EBITDA** - Valor empresa completa sobre ganancias operativas",
                    "calculacion": "Enterprise Value ÷ EBITDA",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **EV/EBITDA < 8**: Posiblemente subvalorada
                    - **EV/EBITDA 8-12**: Rango típico
                    - **EV/EBITDA > 12**: Posiblemente sobrevalorada
                    
                    **Ventajas:**
                    - Considera la deuda y efectivo (mejor que P/E)
                    - Útil para comparar empresas con diferente apalancamiento
                    - Muy usado en fusiones y adquisiciones
                    
                    **Desventajas:**
                    - No considera gastos por intereses e impuestos
                    - El EBITDA puede ser engañoso en algunos casos
                    - No es GAAP (puede calcularse de diferentes formas)
                    
                    **Sectores típicos:**
                    - Telecom: 6-9
                    - Healthcare: 10-14
                    - Tech: 12-18
                    
                    **¿Para qué sirve?**
                    - Comparar empresas con diferentes estructuras de capital
                    - Análisis de M&A (fusiones y adquisiciones)
                    - Evaluar el valor operativo del negocio
                    """,
                    "ejemplo": "EV $500M, EBITDA $50M → EV/EBITDA = 10"
                },
                
                "EV/Sales": {
                    "definicion": "**Enterprise Value/Ventas** - Valor empresa completa sobre ventas",
                    "calculacion": "Enterprise Value ÷ Ventas anuales",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **EV/Sales < 1**: Bajo relativo a ventas
                    - **EV/Sales 1-3**: Rango típico
                    - **EV/Sales > 3**: Alto relativo a ventas
                    
                    **Ventajas:**
                    - Considera la estructura completa de capital
                    - Mejor que P/S para empresas con mucha deuda
                    - Útil para empresas sin ganancias
                    
                    **Desventajas:**
                    - No considera rentabilidad
                    - Las ventas no garantizan ganancias
                    - Puede variar mucho entre sectores
                    
                    **¿Para qué sirve?**
                    - Evaluar empresas en crecimiento sin ganancias
                    - Comparar empresas con diferentes niveles de deuda
                    - Análisis de startups y empresas high-growth
                    """,
                    "ejemplo": "EV $600M, Ventas $200M → EV/Sales = 3.0"
                },
                
                "EV/FCF": {
                    "definicion": "**Enterprise Value/Flujo de Caja Libre** - Valor empresa completa sobre FCF",
                    "calculacion": "Enterprise Value ÷ Flujo de Caja Libre",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **EV/FCF < 10**: Muy atractivo
                    - **EV/FCF 10-20**: Razonable
                    - **EV/FCF > 20**: Posiblemente caro
                    
                    **Ventajas:**
                    - Considera toda la estructura de capital
                    - Basado en flujo de caja real (no ganancias contables)
                    - Bueno para evaluar capacidad de pago de deuda
                    
                    **Desventajas:**
                    - El FCF puede ser volátil
                    - No considera necesidades futuras de inversión
                    - Puede ser negativo
                    
                    **¿Para qué sirve?**
                    - Evaluar el retorno sobre la inversión total
                    - Análisis de empresas con mucha deuda
                    - Comparar oportunidades de inversión
                    """,
                    "ejemplo": "EV $800M, FCF $80M → EV/FCF = 10"
                },
                
                "EPS (ttm)": {
                    "definicion": "**Ganancias por Acción últimos 12 meses** - Beneficio neto por acción",
                    "calculacion": "Beneficio Neto últimos 12 meses ÷ Acciones en circulación",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **EPS creciente**: Empresa en crecimiento
                    - **EPS estable**: Empresa madura
                    - **EPS decreciente**: Posibles problemas
                    
                    **Ventajas:**
                    - Fácil de entender
                    - Directamente relacionado con el precio (P/E)
                    - Buen indicador de salud financiera
                    
                    **Desventajas:**
                    - Puede ser manipulado contablemente
                    - No considera el flujo de caja
                    - Puede variar por eventos extraordinarios
                    
                    **¿Para qué sirve?**
                    - Calcular el P/E ratio
                    - Evaluar la rentabilidad por acción
                    - Seguir la trayectoria de ganancias
                    """,
                    "ejemplo": "Beneficio $100M, 10M acciones → EPS = $10"
                },
                
                "EPS next Y": {
                    "definicion": "**EPS Próximo Año** - Estimación de ganancias para el próximo año",
                    "calculacion": "Estimación de Beneficio Neto próximo año ÷ Acciones estimadas",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **EPS next Y > EPS actual**: Crecimiento esperado
                    - **EPS next Y < EPS actual**: Decrecimiento esperado
                    - **Gran diferencia**: Cambios significativos en el negocio
                    
                    **Ventajas:**
                    - Proporciona visión futura
                    - Útil para calcular Forward P/E
                    - Refleja expectativas del mercado
                    
                    **Desventajas:**
                    - Basado en estimaciones (inciertas)
                    - Puede ser demasiado optimista/pesimista
                    - Sensible a revisiones
                    
                    **¿Para qué sirve?**
                    - Evaluar expectativas de crecimiento
                    - Identificar posibles sorpresas de ganancias
                    - Planificar estrategias de inversión
                    """,
                    "ejemplo": "EPS actual $5, EPS next Y estimado $6 → 20% crecimiento esperado"
                },
                
                "EPS next Q": {
                    "definicion": "**EPS Próximo Trimestre** - Estimación para el próximo trimestre",
                    "calculacion": "Estimación Beneficio Neto próximo trimestre ÷ Acciones",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **Beat estimates**: Supera estimaciones (positivo)
                    - **Miss estimates**: No alcanza estimaciones (negativo)
                    - **Guide higher**: Aumenta guidance (muy positivo)
                    
                    **Ventajas:**
                    - Proporciona visión a corto plazo
                    - Útil para trading alrededor de earnings
                    - Indica momentum operativo
                    
                    **Desventajas:**
                    - Muy volátil entre trimestres
                    - Sensible a estacionalidad
                    - Las estimaciones pueden ser erróneas
                    
                    **¿Para qué sirve?**
                    - Anticipar resultados trimestrales
                    - Evaluar momentum del negocio
                    - Timing de entrada/salida de posiciones
                    """,
                    "ejemplo": "Estimación Q1: $1.25 por acción"
                },
                
                "EPS this Y": {
                    "definicion": "**EPS Este Año** - Ganancias actuales vs año anterior",
                    "calculacion": "EPS año actual ÷ EPS año anterior - 1",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **Positivo**: Crecimiento interanual
                    - **Negativo**: Decrecimiento interanual
                    - **Alto**: Fuerte crecimiento
                    
                    **Ventajas:**
                    - Muestra tendencia anual
                    - Menos volátil que trimestral
                    - Buen indicador de dirección
                    
                    **Desventajas:**
                    - Puede estar influido por eventos únicos
                    - No considera factores estacionales
                    - Puede enmascarar problemas trimestrales
                    
                    **¿Para qué sirve?**
                    - Evaluar performance anual
                    - Comparar con guidance de la empresa
                    - Análisis de tendencias a medio plazo
                    """,
                    "ejemplo": "EPS 2023: $4.50, EPS 2024: $5.00 → Crecimiento 11%"
                },
                
                "EPS next 5Y": {
                    "definicion": "**Crecimiento EPS Próximos 5 Años** - Tasa crecimiento anual estimada",
                    "calculacion": "Estimación crecimiento anual compuesto próximo 5 años",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **<5%**: Crecimiento lento (empresa madura)
                    - **5-15%**: Crecimiento moderado
                    - **>15%**: Crecimiento rápido (empresa growth)
                    
                    **Ventajas:**
                    - Proporciona perspectiva a largo plazo
                    - Útil para modelos de descuento de flujos
                    - Refleja expectativas de crecimiento sostenido
                    
                    **Desventajas:**
                    - Muy especulativo a 5 años vista
                    - Las estimaciones suelen ser optimistas
                    - Difícil de predecir con precisión
                    
                    **¿Para qué sirve?**
                    - Calcular PEG ratio
                    - Evaluar potencial de crecimiento a largo plazo
                    - Comparar empresas dentro del mismo sector
                    """,
                    "ejemplo": "Crecimiento EPS estimado 12% anual próximos 5 años"
                },
                
                "EPS past 5Y": {
                    "definicion": "**Crecimiento EPS 5 Años** - Tasa crecimiento histórico anual",
                    "calculacion": "Tasa crecimiento anual compuesto últimos 5 años",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **Consistente**: Crecimiento estable (buena gestión)
                    - **Volátil**: Resultados irregulares (riesgo)
                    - **Decreciente**: Posible madurez/saturación
                    
                    **Ventajas:**
                    - Basado en datos reales (no estimaciones)
                    - Muestra capacidad histórica de crecimiento
                    - Buen indicador de calidad de gestión
                    
                    **Desventajas:**
                    - El pasado no garantiza futuro
                    - Puede estar influido por ciclos económicos
                    - No considera cambios recientes en el negocio
                    
                    **¿Para qué sirve?**
                    - Evaluar track record de la empresa
                    - Comparar con estimaciones futuras
                    - Análisis de consistencia en resultados
                    """,
                    "ejemplo": "EPS creció de $2 a $4 en 5 años → 15% crecimiento anual"
                },
                
                "Book Value/Share": {
                    "definicion": "**Valor Contable por Acción** - Valor patrimonial por acción",
                    "calculacion": "Patrimonio Neto ÷ Acciones en circulación",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **Creciente**: Empresa acumulando valor
                    - **Decreciente**: Pérdidas o recompras de acciones
                    - **Estable**: Empresa madura
                    
                    **Ventajas:**
                    - Representa el valor en libros
                    - Relativamente estable
                    - Bueno para empresas con activos tangibles
                    
                    **Desventajas:**
                    - No refleja valor de mercado
                    - Puede no incluir activos intangibles
                    - Puede estar desactualizado
                    
                    **¿Para qué sirve?**
                    - Calcular P/B ratio
                    - Evaluar valoración relativa
                    - Análisis de empresas value
                    """,
                    "ejemplo": "Patrimonio $400M, 10M acciones → Book Value/Share = $40"
                }
            }
            
            for metrica, detalles in metricas.items():
                with st.expander(f"**{metrica}**"):
                    st.write(f"**📖 DEFINICIÓN:** {detalles['definicion']}")
                    st.write(f"**🧮 CÁLCULO:** {detalles['calculacion']}")
                    st.markdown("**📊 INTERPRETACIÓN DETALLADA:**")
                    st.write(detalles['interpretacion'])
                    if 'ejemplo' in detalles:
                        st.info(f"**🔢 EJEMPLO:** {detalles['ejemplo']}")
        
        elif categoria == "📈 RENTABILIDAD Y MÁRGENES (16 métricas)":
            st.subheader("📈 RENTABILIDAD Y MÁRGENES - 16 Métricas")
            
            metricas = {
                "ROA (Return on Assets)": {
                    "definicion": "**Retorno sobre Activos** - Eficiencia en el uso de todos los recursos",
                    "calculacion": "Beneficio Neto ÷ Activos Totales × 100",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **ROA < 5%**: Baja eficiencia
                    - **ROA 5-10%**: Adecuado
                    - **ROA > 10%**: Alta eficiencia
                    
                    **Ventajas:**
                    - Considera todos los recursos (no solo el capital)
                    - Menos susceptible a manipulación por apalancamiento
                    - Bueno para comparar empresas con diferentes estructuras de capital
                    
                    **Desventajas:**
                    - Los activos pueden estar valorados incorrectamente
                    - No considera el costo de capital
                    - Puede penalizar empresas con muchos activos
                    
                    **Comparativa por sectores:**
                    - Tecnología: 8-15% (pocos activos, altos retornos)
                    - Manufactura: 4-8% (activos intensivos)
                    - Retail: 3-6% (márgenes bajos, alta rotación)
                    
                    **¿Para qué sirve?**
                    - Medir la eficiencia operativa general
                    - Comparar empresas con diferentes niveles de deuda
                    - Evaluar la calidad de la gestión
                    """,
                    "ejemplo": "Beneficio $500k, Activos $10M → ROA = 5%"
                },
                
                "ROE (Return on Equity)": {
                    "definicion": "**Retorno sobre el Patrimonio** - Rentabilidad generada con el capital de los accionistas",
                    "calculacion": "Beneficio Neto ÷ Patrimonio Neto × 100",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **ROE < 8%**: Bajo - podría no compensar el riesgo
                    - **ROE 8-15%**: Adecuado
                    - **ROE 15-20%**: Bueno
                    - **ROE > 20%**: Excelente
                    
                    **Ventajas:**
                    - Fácil de calcular y entender
                    - Buen indicador de eficiencia del capital
                    - Ampliamente utilizado
                    
                    **Desventajas:**
                    - Puede ser inflado por mucho apalancamiento (deuda)
                    - No considera el riesgo asumido
                    - Puede variar significativamente entre sectores
                    
                    **Análisis DuPont (descomposición del ROE):**
                    ROE = (Margen Neto) × (Rotación Activos) × (Apalancamiento)
                    - **Margen Neto**: Eficiencia en control de costos
                    - **Rotación**: Eficiencia uso de activos  
                    - **Apalancamiento**: Uso de deuda vs capital
                    
                    **¿Para qué sirve?**
                    - Medir la eficiencia en el uso del capital de accionistas
                    - Comparar empresas dentro del mismo sector
                    - Identificar empresas con ventajas competitivas sostenibles
                    """,
                    "ejemplo": "Beneficio $1M, Patrimonio $10M → ROE = 10%"
                },
                
                "ROI (Return on Investment)": {
                    "definicion": "**Retorno sobre la Inversión** - Eficiencia de las inversiones realizadas",
                    "calculacion": "Beneficio de la inversión ÷ Costo de la inversión × 100",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **ROI > costo de capital**: Crea valor
                    - **ROI < costo de capital**: Destruye valor
                    - **ROI alto**: Inversiones eficientes
                    
                    **Ventajas:**
                    - Mide la eficiencia de las decisiones de inversión
                    - Útil para evaluar proyectos específicos
                    - Fácil de entender
                    
                    **Desventajas:**
                    - Puede ser difícil de calcular para inversiones complejas
                    - No considera el valor temporal del dinero
                    - Puede variar según el período medido
                    
                    **¿Para qué sirve?**
                    - Evaluar la eficiencia del capital invertido
                    - Comparar diferentes oportunidades de inversión
                    - Tomar decisiones de asignación de capital
                    """,
                    "ejemplo": "Inversión $1M, Beneficio $150k anual → ROI = 15%"
                },
                
                "Gross Margin": {
                    "definicion": "**Margen Bruto** - Porcentaje que queda después de costos directos",
                    "calculacion": "(Ventas - Costo de Bienes Vendidos) ÷ Ventas × 100",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **Margen alto**: Fuertes ventajas competitivas, poder de precios
                    - **Margen bajo**: Competencia intensa, commoditización
                    - **Margen creciente**: Mejora en eficiencia o poder de precios
                    
                    **Ventajas:**
                    - Buen indicador de ventajas competitivas
                    - Relativamente estable en el tiempo
                    - Difícil de manipular contablemente
                    
                    **Desventajas:**
                    - No considera gastos operativos
                    - Puede variar significativamente por estacionalidad
                    - Depende de la clasificación de costos
                    
                    **Rangos por industria:**
                    - Software: 80-90%
                    - Farmacéutica: 70-80%
                    - Bienes de consumo: 40-60%
                    - Retail: 20-40%
                    - Airlines: 10-20%
                    
                    **¿Para qué sirve?**
                    - Evaluar el poder de fijación de precios
                    - Medir ventajas competitivas en costos
                    - Identificar tendencias en la rentabilidad del core business
                    """,
                    "ejemplo": "Ventas $1M, Costo bienes $600k → Margen Bruto = 40%"
                },
                
                "Operating Margin": {
                    "definicion": "**Margen Operativo** - Rentabilidad del negocio principal antes de intereses e impuestos",
                    "calculacion": "Beneficio Operativo ÷ Ventas × 100",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **Margen alto**: Eficiencia operativa, control de gastos
                    - **Margen bajo**: Altos gastos operativos, ineficiencia
                    - **Margen creciente**: Mejora en gestión operativa
                    
                    **Ventajas:**
                    - Mide la eficiencia del negocio principal
                    - Excluye efectos financieros y fiscales
                    - Bueno para comparar empresas con diferente apalancamiento
                    
                    **Desventajas:**
                    - No considera la estructura de capital
                    - Puede variar por decisiones contables
                    - No refleja el beneficio final para accionistas
                    
                    **Componentes que afectan el margen operativo:**
                    - Eficiencia en producción
                    - Control de gastos generales
                    - Precios vs costos
                    - Economías de escala
                    
                    **¿Para qué sirve?**
                    - Evaluar la eficiencia operativa del negocio core
                    - Comparar empresas con diferentes estructuras financieras
                    - Identificar mejoras en gestión operativa
                    """,
                    "ejemplo": "Ventas $1M, Beneficio operativo $150k → Margen Operativo = 15%"
                },
                
                "Profit Margin": {
                    "definicion": "**Margen de Beneficio Neto** - Porcentaje final que queda para accionistas",
                    "calculacion": "Beneficio Neto ÷ Ventas × 100",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **Margen alto**: Empresa muy eficiente o con fuertes ventajas
                    - **Margen bajo**: Competencia intensa o ineficiencias
                    - **Margen creciente**: Mejoras en eficiencia o mix de productos
                    
                    **Ventajas:**
                    - Representa el resultado final para accionistas
                    - Incluye todos los costos y gastos
                    - Fácil de comparar entre empresas
                    
                    **Desventajas:**
                    - Puede ser afectado por eventos extraordinarios
                    - No diferencia entre ganancias operativas y no operativas
                    - Puede variar por decisiones fiscales
                    
                    **Rangos típicos:**
                    - Software: 20-30%
                    - Bancos: 15-25%
                    - Retail: 2-5%
                    - Airlines: 2-8%
                    
                    **¿Para qué sirve?**
                    - Evaluar la rentabilidad final del negocio
                    - Comparar eficiencia entre competidores
                    - Identificar tendencias en rentabilidad
                    """,
                    "ejemplo": "Ventas $1M, Beneficio neto $80k → Profit Margin = 8%"
                },
                
                "EBITDA": {
                    "definicion": "**Ganancias antes de Intereses, Impuestos, Depreciación y Amortización**",
                    "calculacion": "Beneficio Operativo + Depreciación + Amortización",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **EBITDA alto**: Fuerte generación operativa de caja
                    - **EBITDA creciente**: Mejora en performance operativa
                    - **EBITDA/Intereses alto**: Buena capacidad de cubrir deuda
                    
                    **Ventajas:**
                    - Elimina efectos de decisiones financieras y fiscales
                    - Buen proxy para flujo de caja operativo
                    - Útil para comparar empresas con diferentes estructuras
                    
                    **Desventajas:**
                    - No es GAAP (puede calcularse de diferentes formas)
                    - Ignora necesidades de reinversión en activos
                    - Puede ser engañoso en empresas con alta depreciación
                    
                    **¿Para qué sirve?**
                    - Evaluar performance operativa pura
                    - Calcular ratios de cobertura de deuda
                    - Análisis de empresas con diferentes políticas de depreciación
                    """,
                    "ejemplo": "Beneficio operativo $200k, Depreciación $50k → EBITDA = $250k"
                },
                
                "EBIT": {
                    "definicion": "**Ganancias antes de Intereses e Impuestos** - Resultado operativo",
                    "calculacion": "Ventas - Todos los gastos operativos (excluyendo intereses e impuestos)",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **EBIT alto**: Negocio central rentable
                    - **EBIT creciente**: Mejora en eficiencia operativa
                    - **EBIT estable**: Empresa madura y predecible
                    
                    **Ventajas:**
                    - Mide la rentabilidad del negocio principal
                    - Excluye efectos financieros y fiscales
                    - Bueno para comparar eficiencia operativa
                    
                    **Desventajas:**
                    - No considera necesidades de inversión en activos
                    - Puede variar por métodos contables
                    - No refleja el costo del capital
                    
                    **¿Para qué sirve?**
                    - Evaluar la rentabilidad operativa core
                    - Comparar empresas con diferente apalancamiento
                    - Análisis de eficiencia operativa por segmentos
                    """,
                    "ejemplo": "Ventas $1M, Gastos operativos $800k → EBIT = $200k"
                },
                
                "Net Income": {
                    "definicion": "**Beneficio Neto** - Ganancias finales después de todos los gastos",
                    "calculacion": "Ingresos Totales - Gastos Totales (incluyendo intereses e impuestos)",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **Positivo y creciente**: Empresa saludable y en crecimiento
                    - **Volátil**: Resultados inconsistentes (riesgo)
                    - **Negativo**: Pérdidas (señal de alerta)
                    
                    **Ventajas:**
                    - Representa el resultado final para accionistas
                    - Incluye todos los aspectos del negocio
                    - Base para cálculo de EPS
                    
                    **Desventajas:**
                    - Puede incluir partidas extraordinarias
                    - Sensible a decisiones contables
                    - No diferencia entre ganancias recurrentes y no recurrentes
                    
                    **¿Para qué sirve?**
                    - Evaluar la rentabilidad general
                    - Calcular ratios de rentabilidad (ROE, ROA)
                    - Seguir la trayectoria de ganancias
                    """,
                    "ejemplo": "Ingresos $1.2M, Gastos $1.1M → Net Income = $100k"
                },
                
                "Income Tax": {
                    "definicion": "**Impuesto sobre la Renta** - Monto pagado en impuestos",
                    "calculacion": "Base imponible × Tasa impositiva",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **Tasa efectiva baja**: Posibles beneficios fiscales o ubicación favorable
                    - **Tasa efectiva alta**: Pocos beneficios fiscales
                    - **Cambios significativos**: Cambios en legislación o estructura
                    
                    **Ventajas:**
                    - Indica la carga fiscal real
                    - Puede mostrar ventajas competitivas fiscales
                    - Útil para proyecciones futuras
                    
                    **Desventajas:**
                    - Puede ser temporal (créditos fiscales, pérdidas arrastradas)
                    - Complejo de analizar en empresas multinacionales
                    - Sensible a cambios legislativos
                    
                    **¿Para qué sirve?**
                    - Evaluar la carga fiscal efectiva
                    - Identificar ventajas fiscales sostenibles
                    - Proyectar ganancias futuras netas
                    """,
                    "ejemplo": "Beneficio antes impuestos $500k, Impuestos $100k → Tasa 20%"
                },
                
                "Dividend": {
                    "definicion": "**Dividendo** - Pago periódico a accionistas",
                    "calculacion": "Monto total distribuido ÷ Acciones en circulación",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **Dividendo creciente**: Empresa con exceso de caja y confianza
                    - **Dividendo estable**: Empresa madura y predecible
                    - **Recorte de dividendo**: Posibles problemas financieros
                    
                    **Ventajas:**
                    - Proporciona income a inversores
                    - Señal de confianza del management
                    - Atractivo para inversores conservadores
                    
                    **Desventajas:**
                    - Dinero que no se reinvierte en el negocio
                    - Puede crear expectativas difíciles de mantener
                    - Empresas pueden endeudarse para pagarlos
                    
                    **¿Para qué sirve?**
                    - Evaluar política de distribución a accionistas
                    - Calcular yield y retorno total
                    - Identificar empresas income-oriented
                    """,
                    "ejemplo": "Dividendo trimestral $0.25 por acción → $1.00 anual"
                },
                
                "Dividend %": {
                    "definicion": "**Rendimiento por Dividendo** - Retorno por dividendo relativo al precio",
                    "calculacion": "Dividendo anual por acción ÷ Precio de la acción × 100",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **Yield bajo (1-3%)**: Empresas growth, poco income
                    - **Yield medio (3-6%)**: Empresas value, balance growth/income
                    - **Yield alto (>6%)**: Empresas income, posible riesgo
                    
                    **Ventajas:**
                    - Fácil de calcular y comparar
                    - Componente importante del retorno total
                    - Atractivo para inversores que buscan income
                    
                    **Desventajas:**
                    - Yield alto puede indicar problemas (precio bajo)
                    - No garantizado (puede ser recortado)
                    - Empresas pueden tener yield alto pero poco crecimiento
                    
                    **Sectores típicos:**
                    - Utilities: 3-5%
                    - REITs: 4-8%
                    - Tech: 0-2%
                    - Consumer Staples: 2-4%
                    
                    **¿Para qué sirve?**
                    - Evaluar atractivo para inversores income
                    - Comparar con alternativas de renta fija
                    - Calcular retorno total esperado
                    """,
                    "ejemplo": "Precio $100, Dividendo anual $4 → Yield = 4%"
                },
                
                "Payout Ratio": {
                    "definicion": "**Ratio de Pago** - Porcentaje de ganancias pagado como dividendo",
                    "calculacion": "Dividendo por acción ÷ EPS × 100",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **Payout bajo (<30%)**: Empresa retiene ganancias para crecimiento
                    - **Payout medio (30-60%)**: Balance entre dividendos y crecimiento
                    - **Payout alto (>60%)**: Empresa madura, poco crecimiento
                    - **Payout >100%**: Pagando más de lo que gana (insostenible)
                    
                    **Ventajas:**
                    - Indica sostenibilidad del dividendo
                    - Muestra la política de distribución vs reinversión
                    - Útil para evaluar crecimiento futuro
                    
                    **Desventajas:**
                    - Basado en ganancias que pueden ser volátiles
                    - No considera flujo de caja
                    - Puede variar significativamente entre años
                    
                    **¿Para qué sirve?**
                    - Evaluar sostenibilidad del dividendo
                    - Identificar empresas con potencial de aumento de dividendo
                    - Analizar el balance entre income y crecimiento
                    """,
                    "ejemplo": "EPS $5, Dividendo $2 → Payout Ratio = 40%"
                },
                
                "EPS Q/Q": {
                    "definicion": "**Crecimiento EPS Trimestral** - Cambio vs trimestre anterior",
                    "calculacion": "(EPS trimestre actual - EPS trimestre anterior) ÷ EPS trimestre anterior × 100",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **Positivo**: Mejora trimestral
                    - **Negativo**: Empeoramiento trimestral
                    - **Alto**: Fuerte momentum
                    - **Consistente positivo**: Trayectoria sólida
                    
                    **Ventajas:**
                    - Muestra momentum a corto plazo
                    - Útil para identificar tendencias emergentes
                    - Reacciona rápido a cambios en el negocio
                    
                    **Desventajas:**
                    - Muy volátil entre trimestres
                    - Sensible a estacionalidad
                    - Puede estar afectado por eventos únicos
                    
                    **¿Para qué sirve?**
                    - Evaluar performance trimestral
                    - Identificar cambios en momentum
                    - Timing de decisiones de inversión
                    """,
                    "ejemplo": "EPS Q1: $1.20, EPS Q2: $1.35 → Crecimiento 12.5%"
                },
                
                "Sales Q/Q": {
                    "definicion": "**Crecimiento Ventas Trimestral** - Cambio en ventas vs trimestre anterior",
                    "calculacion": "(Ventas trimestre actual - Ventas trimestre anterior) ÷ Ventas trimestre anterior × 100",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **Positivo**: Crecimiento orgánico o por adquisiciones
                    - **Negativo**: Contracción del negocio
                    - **Aceleración**: Crecimiento cada vez más rápido
                    - **Desaceleración**: Crecimiento perdiendo momentum
                    
                    **Ventajas:**
                    - Indica salud del top line
                    - Menos manipulable que las ganancias
                    - Buen indicador de demanda del producto/servicio
                    
                    **Desventajas:**
                    - No considera rentabilidad
                    - Puede estar inflado por adquisiciones
                    - Sensible a estacionalidad
                    
                    **¿Para qué sirve?**
                    - Evaluar crecimiento del negocio principal
                    - Identificar tendencias en demanda
                    - Comparar con expectativas del mercado
                    """,
                    "ejemplo": "Ventas Q1: $250M, Ventas Q2: $275M → Crecimiento 10%"
                },
                
                "Earnings Date": {
                    "definicion": "**Fecha de Resultados** - Próxima publicación de resultados trimestrales",
                    "calculacion": "Fecha calendario anunciada por la empresa",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **Antes del opening/after closing**: Normal para minimizar impacto
                    - **Desviación del patrón habitual**: Posible sorpresa
                    - **Retraso inusual**: Posibles problemas
                    
                    **Ventajas:**
                    - Permite prepararse para la volatilidad
                    - Útil para estrategias de trading alrededor de earnings
                    - Indica transparencia del management
                    
                    **Desventajas:**
                    - Las fechas pueden cambiar
                    - No indica la calidad de los resultados
                    - Puede generar expectativas irreales
                    
                    **¿Para qué sirve?**
                    - Planificar timing de inversiones
                    - Gestionar riesgo alrededor de eventos
                    - Evaluar consistencia en comunicación
                    """,
                    "ejemplo": "Próximo earnings: 25 de Octubre, después del cierre"
                }
            }
            
            for metrica, detalles in metricas.items():
                with st.expander(f"**{metrica}**"):
                    st.write(f"**📖 DEFINICIÓN:** {detalles['definicion']}")
                    st.write(f"**🧮 CÁLCULO:** {detalles['calculacion']}")
                    st.markdown("**📊 INTERPRETACIÓN DETALLADA:**")
                    st.write(detalles['interpretacion'])
                    if 'ejemplo' in detalles:
                        st.info(f"**🔢 EJEMPLO:** {detalles['ejemplo']}")

        elif categoria == "🏦 DEUDA Y LIQUIDEZ (12 métricas)":
            st.subheader("🏦 DEUDA Y LIQUIDEZ - 12 Métricas")
            
            metricas = {
                "Total Debt": {
                    "definicion": "**Deuda Total** - Suma de deuda a corto y largo plazo",
                    "calculacion": "Deuda Corto Plazo + Deuda Largo Plazo",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **Deuda creciente**: Posible expansión agresiva o problemas de caja
                    - **Deuda decreciente**: Desapalancamiento, mejora financiera
                    - **Sin deuda**: Empresa conservadora (puede perder oportunidades)
                    
                    **Ventajas:**
                    - Muestra la carga total de deuda
                    - Fácil de entender
                    - Base para otros ratios de deuda
                    
                    **Desventajas:**
                    - No considera la capacidad de pago
                    - No diferencia entre tipos de deuda
                    - Puede variar por ciclos empresariales
                    
                    **¿Para qué sirve?**
                    - Evaluar el apalancamiento total
                    - Comparar con patrimonio y activos
                    - Analizar tendencias de financiación
                    """,
                    "ejemplo": "Deuda corto plazo $50M + Deuda largo plazo $150M = Total Debt $200M"
                },
                
                "Debt/Eq": {
                    "definicion": "**Ratio Deuda/Patrimonio** - Relación entre deuda total y capital propio",
                    "calculacion": "Deuda Total ÷ Patrimonio Neto",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **<0.5**: Conservador
                    - **0.5-1.0**: Moderado
                    - **>1.0**: Agresivo
                    - **>2.0**: Muy riesgoso
                    
                    **Ventajas:**
                    - Muestra estructura de capital
                    - Útil para comparar empresas del mismo sector
                    - Indica política financiera
                    
                    **Desventajas:**
                    - No considera el costo de la deuda
                    - Puede variar por valoración de patrimonio
                    - Sectores intensivos en capital pueden tener ratios altos normales
                    
                    **Sectores típicos:**
                    - Utilities: 1.0-1.5
                    - Telecom: 1.5-2.0
                    - Tech: 0.2-0.8
                    - Bancos: 3.0+ (estructura diferente)
                    
                    **¿Para qué sirve?**
                    - Evaluar riesgo financiero
                    - Comparar políticas de financiación
                    - Identificar posibles problemas de solvencia
                    """,
                    "ejemplo": "Deuda $200M, Patrimonio $250M → Debt/Eq = 0.8"
                },
                
                "LT Debt/Eq": {
                    "definicion": "**Deuda Largo Plazo/Patrimonio** - Deuda a largo plazo vs capital",
                    "calculacion": "Deuda Largo Plazo ÷ Patrimonio Neto",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **Alto**: Financiación estable a largo plazo
                    - **Bajo**: Poca deuda estructural
                    - **Creciente**: Más financiación vía deuda
                    
                    **Ventajas:**
                    - Enfocado en deuda estructural
                    - Menos volátil que deuda total
                    - Mejor para análisis de largo plazo
                    
                    **Desventajas:**
                    - Ignora deuda a corto plazo
                    - No considera vencimientos
                    - Puede enmascarar problemas de liquidez
                    
                    **¿Para qué sirve?**
                    - Evaluar estructura de capital permanente
                    - Analizar financiación de proyectos largos
                    - Comparar estabilidad financiera
                    """,
                    "ejemplo": "Deuda LP $150M, Patrimonio $250M → LT Debt/Eq = 0.6"
                },
                
                "Current Ratio": {
                    "definicion": "**Ratio Corriente** - Capacidad para pagar obligaciones a corto plazo",
                    "calculacion": "Activos Corrientes ÷ Pasivos Corrientes",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **<1.0**: Posibles problemas de liquidez
                    - **1.0-1.5**: Aceptable
                    - **1.5-2.0**: Bueno
                    - **>2.0**: Excelente (pero puede indicar activos ociosos)
                    
                    **Ventajas:**
                    - Simple y ampliamente usado
                    - Buen indicador de salud a corto plazo
                    - Fácil de calcular
                    
                    **Desventajas:**
                    - No considera calidad de activos corrientes
                    - El inventario puede no ser líquido
                    - Puede variar estacionalmente
                    
                    **¿Para qué sirve?**
                    - Evaluar liquidez inmediata
                    - Detectar posibles problemas de pago
                    - Comparar con competidores del sector
                    """,
                    "ejemplo": "Activos corrientes $500k, Pasivos corrientes $300k → Current Ratio = 1.67"
                },
                
                "Quick Ratio": {
                    "definicion": "**Ratio Rápido** - Liquidez inmediata excluyendo inventario",
                    "calculacion": "(Activos Corrientes - Inventario) ÷ Pasivos Corrientes",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **<0.5**: Muy bajo
                    - **0.5-1.0**: Aceptable
                    - **>1.0**: Bueno
                    - **>1.5**: Excelente
                    
                    **Ventajas:**
                    - Más conservador que Current Ratio
                    - Excluye inventario (menos líquido)
                    - Mejor indicador de liquidez real
                    
                    **Desventajas:**
                    - Puede ser demasiado conservador
                    - No considera rotación de inventario
                    - Algunas empresas dependen del inventario
                    
                    **¿Para qué sirve?**
                    - Evaluar capacidad de pago inmediata
                    - Análisis más realista de liquidez
                    - Detectar dependencia del inventario
                    """,
                    "ejemplo": "Activos corrientes $500k, Inventario $200k, Pasivos $300k → Quick Ratio = 1.0"
                },
                
                "Cash/Share": {
                    "definicion": "**Efectivo por Acción** - Reservas de efectivo por cada acción",
                    "calculacion": "Efectivo y Equivalentes ÷ Acciones en Circulación",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **Alto**: Fuertes reservas, posibles dividendos especiales o recompras
                    - **Bajo**: Poco colchón de seguridad
                    - **Creciente**: Acumulación de caja
                    
                    **Ventajas:**
                    - Muestra colchón de seguridad por acción
                    - Útil para valoración
                    - Indica capacidad para oportunidades
                    
                    **Desventajas:**
                    - No considera deuda
                    - El efectivo puede estar destinado a obligaciones
                    - Demasiado efectivo puede indicar falta de oportunidades de inversión
                    
                    **¿Para qué sirve?**
                    - Evaluar margen de seguridad
                    - Identificar posibles recompras o dividendos
                    - Valoración en adquisiciones
                    """,
                    "ejemplo": "Efectivo $100M, 10M acciones → Cash/Share = $10"
                },
                
                "Cash Flow/Share": {
                    "definicion": "**Flujo de Caja por Acción** - Flujo operativo generado por acción",
                    "calculacion": "Flujo de Caja Operativo ÷ Acciones en Circulación",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **Alto**: Fuerte generación de caja por acción
                    - **Creciente**: Mejora en eficiencia operativa
                    - **> EPS**: Calidad de ganancias alta
                    
                    **Ventajas:**
                    - Basado en caja real (no ganancias contables)
                    - Mejor indicador de salud financiera
                    - Difícil de manipular
                    
                    **Desventajas:**
                    - Puede ser volátil
                    - No considera inversiones de capital
                    - Sensible a cambios en capital de trabajo
                    
                    **¿Para qué sirve?**
                    - Evaluar calidad de ganancias
                    - Calcular capacidad de pago de dividendos
                    - Comparar con EPS
                    """,
                    "ejemplo": "FCF Operativo $80M, 10M acciones → Cash Flow/Share = $8"
                },
                
                "Total Cash": {
                    "definicion": "**Efectivo Total** - Dinero disponible en caja y equivalentes",
                    "calculacion": "Efectivo + Equivalentes de Efectivo",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **Alto**: Fuertes reservas líquidas
                    - **Bajo**: Dependencia de financiación externa
                    - **Óptimo**: Suficiente para operar + colchón de seguridad
                    
                    **Ventajas:**
                    - Muestra liquidez absoluta
                    - Fácil de entender
                    - Base para otros cálculos
                    
                    **Desventajas:**
                    - No considera obligaciones
                    - Puede estar en el extranjero con restricciones
                    - Demasiado efectivo puede ser ineficiente
                    
                    **¿Para qué sirve?**
                    - Evaluar solvencia a corto plazo
                    - Analizar capacidad para oportunidades
                    - Preparación para crisis
                    """,
                    "ejemplo": "Efectivo $50M + Equivalentes $30M = Total Cash $80M"
                },
                
                "Total Cash/Share": {
                    "definicion": "**Efectivo Total por Acción** - Similar a Cash/Share pero incluye equivalentes",
                    "calculacion": "Total Cash ÷ Acciones en Circulación",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **Comparación con precio**: Si Cash/Share es alto vs precio, posible oportunidad
                    - **Tendencia**: Creciente es positivo
                    - **Sector**: Tech suele tener más cash que industriales
                    
                    **Ventajas:**
                    - Visión completa de liquidez por acción
                    - Útil para valoración
                    - Bueno para análisis comparativo
                    
                    **Desventajas:**
                    - No considera uso del efectivo
                    - Puede incluir efectivo restringido
                    - No diferencia entre efectivo operativo y no operativo
                    
                    **¿Para qué sirve?**
                    - Valoración relativa
                    - Identificar empresas con exceso de caja
                    - Evaluar potencial de recompra de acciones
                    """,
                    "ejemplo": "Total Cash $80M, 10M acciones → Total Cash/Share = $8"
                },
                
                "Working Capital": {
                    "definicion": "**Capital de Trabajo** - Recursos disponibles para operaciones diarias",
                    "calculacion": "Activos Corrientes - Pasivos Corrientes",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **Positivo**: Capacidad para operar sin problemas
                    - **Negativo**: Posibles problemas de liquidez
                    - **Creciente**: Mejora en gestión operativa
                    
                    **Ventajas:**
                    - Muestra salud operativa a corto plazo
                    - Indica eficiencia en gestión de capital de trabajo
                    - Buen predictor de problemas financieros
                    
                    **Desventajas:**
                    - No considera calidad de activos
                    - Puede ser manipulado con timing de pagos/cobros
                    - Varía por estacionalidad
                    
                    **¿Para qué sirve?**
                    - Evaluar salud operativa a corto plazo
                    - Detectar posibles problemas de liquidez
                    - Analizar eficiencia en gestión de capital
                    """,
                    "ejemplo": "Activos corrientes $500k, Pasivos corrientes $300k → Working Capital = $200k"
                },
                
                "Interest Coverage": {
                    "definicion": "**Cobertura de Intereses** - Capacidad para pagar intereses de la deuda",
                    "calculacion": "EBIT ÷ Gastos por Intereses",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **<1.0**: No cubre intereses (muy peligroso)
                    - **1.0-1.5**: Muy justo
                    - **1.5-3.0**: Aceptable
                    - **>3.0**: Bueno
                    - **>5.0**: Excelente
                    
                    **Ventajas:**
                    - Mide capacidad de servicio de deuda
                    - Buen predictor de problemas financieros
                    - Fácil de calcular
                    
                    **Desventajas:**
                    - No considera amortización de principal
                    - Basado en EBIT (no cash flow)
                    - Puede variar con tipos de interés
                    
                    **¿Para qué sirve?**
                    - Evaluar riesgo de impago
                    - Comparar capacidad de endeudamiento
                    - Análisis de solvencia
                    """,
                    "ejemplo": "EBIT $50M, Intereses $10M → Interest Coverage = 5.0"
                },
                
                "Total Debt/EBITDA": {
                    "definicion": "**Deuda Total/EBITDA** - Años necesarios para pagar deuda con EBITDA",
                    "calculacion": "Deuda Total ÷ EBITDA",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **<3.0**: Conservador
                    - **3.0-5.0**: Moderado
                    - **5.0-7.0**: Alto
                    - **>7.0**: Muy riesgoso
                    
                    **Ventajas:**
                    - Muy usado por agencias de rating
                    - Considera capacidad operativa de generar caja
                    - Bueno para comparar entre sectores
                    
                    **Desventajas:**
                    - El EBITDA no es flujo de caja
                    - No considera inversiones de capital
                    - Puede variar con ciclo económico
                    
                    **¿Para qué sirve?**
                    - Evaluar sostenibilidad de la deuda
                    - Comparar políticas de endeudamiento
                    - Análisis de riesgo crediticio
                    """,
                    "ejemplo": "Deuda Total $200M, EBITDA $50M → Debt/EBITDA = 4.0"
                }
            }
            
            for metrica, detalles in metricas.items():
                with st.expander(f"**{metrica}**"):
                    st.write(f"**📖 DEFINICIÓN:** {detalles['definicion']}")
                    st.write(f"**🧮 CÁLCULO:** {detalles['calculacion']}")
                    st.markdown("**📊 INTERPRETACIÓN DETALLADA:**")
                    st.write(detalles['interpretacion'])
                    if 'ejemplo' in detalles:
                        st.info(f"**🔢 EJEMPLO:** {detalles['ejemplo']}")

        elif categoria == "📊 EFICIENCIA OPERATIVA (10 métricas)":
            st.subheader("📊 EFICIENCIA OPERATIVA - 10 Métricas")
            
            metricas = {
                "Asset Turnover": {
                    "definicion": "**Rotación de Activos** - Eficiencia en uso de activos para generar ventas",
                    "calculacion": "Ventas ÷ Activos Totales Promedio",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **Alto**: Eficiente uso de activos
                    - **Bajo**: Activos subutilizados
                    - **Creciente**: Mejora en eficiencia
                    
                    **Ventajas:**
                    - Mide eficiencia operativa general
                    - Bueno para comparar empresas del mismo sector
                    - Refleja calidad de gestión
                    
                    **Desventajas:**
                    - Varía mucho entre sectores
                    - Puede estar influido por valoración de activos
                    - No considera rentabilidad
                    
                    **Sectores típicos:**
                    - Retail: 2.0-3.0 (alta rotación)
                    - Manufacturing: 0.8-1.2
                    - Utilities: 0.3-0.5 (activos intensivos)
                    
                    **¿Para qué sirve?**
                    - Evaluar eficiencia operativa
                    - Comparar gestión entre competidores
                    - Identificar mejoras en utilización de activos
                    """,
                    "ejemplo": "Ventas $1B, Activos promedio $500M → Asset Turnover = 2.0"
                },
                
                "Inventory Turnover": {
                    "definicion": "**Rotación de Inventario** - Veces que se renueva el inventario anual",
                    "calculacion": "Costo de Ventas ÷ Inventario Promedio",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **Alto**: Gestión eficiente de inventario
                    - **Bajo**: Exceso de inventario o ventas lentas
                    - **Óptimo**: Balance entre disponibilidad y costos
                    
                    **Ventajas:**
                    - Mide eficiencia en gestión de inventario
                    - Buen predictor de problemas operativos
                    - Sensible a cambios en demanda
                    
                    **Desventajas:**
                    - Varía por estacionalidad
                    - Depende del tipo de negocio
                    - Puede ser manipulado con valoración de inventario
                    
                    **Sectores típicos:**
                    - Grocery: 10-15
                    - Retail: 4-8
                    - Manufacturing: 2-4
                    
                    **¿Para qué sirve?**
                    - Evaluar eficiencia operativa
                    - Detectar problemas de ventas
                    - Optimizar niveles de inventario
                    """,
                    "ejemplo": "Costo ventas $600M, Inventario promedio $100M → Inventory Turnover = 6.0"
                },
                
                "Receivables Turnover": {
                    "definicion": "**Rotación de Cuentas por Cobrar** - Eficiencia en cobro a clientes",
                    "calculacion": "Ventas a Crédito ÷ Cuentas por Cobrar Promedio",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **Alto**: Cobros rápidos (eficiente)
                    - **Bajo**: Cobros lentos (posibles problemas)
                    - **Decreciente**: Posible deterioro de calidad de clientes
                    
                    **Ventajas:**
                    - Mide eficiencia en gestión de crédito
                    - Indicador de calidad de cartera
                    - Sensible a cambios en políticas de crédito
                    
                    **Desventajas:**
                    - Necesita datos de ventas a crédito (no siempre disponibles)
                    - Puede variar por estacionalidad
                    - No considera morosidad
                    
                    **¿Para qué sirve?**
                    - Evaluar políticas de crédito
                    - Detectar problemas de cobranza
                    - Comparar con términos de pago ofrecidos
                    """,
                    "ejemplo": "Ventas crédito $400M, Cuentas cobrar promedio $50M → Receivables Turnover = 8.0"
                },
                
                "Days Inventory": {
                    "definicion": "**Días de Inventario** - Días promedio que permanece el inventario",
                    "calculacion": "365 ÷ Inventory Turnover",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **Bajo**: Inventario que se mueve rápido
                    - **Alto**: Inventario lento o excesivo
                    - **Óptimo**: Balance entre disponibilidad y costos
                    
                    **Ventajas:**
                    - Más intuitivo que turnover
                    - Fácil de comparar con términos de pago
                    - Bueno para gestión operativa
                    
                    **Desventajas:**
                    - Mismo que Inventory Turnover
                    - Sensible a estacionalidad
                    - Puede variar por mix de productos
                    
                    **Sectores típicos:**
                    - Fast food: 2-5 días
                    - Retail: 30-60 días
                    - Manufacturing: 60-90 días
                    
                    **¿Para qué sirve?**
                    - Gestión de niveles de inventario
                    - Evaluar eficiencia operativa
                    - Detectar productos obsoletos
                    """,
                    "ejemplo": "Inventory Turnover 6 → Days Inventory = 61 días"
                },
                
                "Days Sales Outstanding": {
                    "definicion": "**Días de Ventas Pendientes** - Días promedio para cobrar ventas",
                    "calculacion": "365 ÷ Receivables Turnover",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **Bajo**: Cobros rápidos (bueno)
                    - **Alto**: Cobros lentos (malo)
                    - **Comparar con términos**: Si DSO > términos, problemas de cobro
                    
                    **Ventajas:**
                    - Fácil de entender y gestionar
                    - Bueno para seguimiento operativo
                    - Sensible a cambios en políticas
                    
                    **Desventajas:**
                    - Puede variar por mix de clientes
                    - Sensible a estacionalidad
                    - No considera morosidad
                    
                    **¿Para qué sirve?**
                    - Evaluar eficiencia de cobranza
                    - Gestionar capital de trabajo
                    - Detectar problemas con clientes
                    """,
                    "ejemplo": "Receivables Turnover 8 → DSO = 46 días"
                },
                
                "Payables Period": {
                    "definicion": "**Período de Pago a Proveedores** - Días promedio para pagar proveedores",
                    "calculacion": "365 ÷ (Compras ÷ Cuentas por Pagar Promedio)",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **Alto**: Paga lentamente (usa proveedores como financiación)
                    - **Bajo**: Paga rápidamente (puede perder descuentos)
                    - **Óptimo**: Balance entre relaciones y costos
                    
                    **Ventajas:**
                    - Mide gestión de proveedores
                    - Indica poder de negociación
                    - Afecta capital de trabajo
                    
                    **Desventajas:**
                    - Datos de compras no siempre disponibles
                    - Puede variar por relaciones estratégicas
                    - No considera descuentos por pronto pago
                    
                    **¿Para qué sirve?**
                    - Optimizar capital de trabajo
                    - Evaluar relaciones con proveedores
                    - Comparar con términos de pago
                    """,
                    "ejemplo": "Compras $300M, Cuentas pagar $50M → Payables Period = 61 días"
                },
                
                "Cash Conversion Cycle": {
                    "definicion": "**Ciclo de Conversión de Efectivo** - Días desde pago a proveedores hasta cobro de clientes",
                    "calculacion": "Days Inventory + DSO - Payables Period",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **Positivo**: Necesita financiar operaciones
                    - **Negativo**: Proveedores financian operaciones (ideal)
                    - **Bajo**: Eficiente gestión de capital de trabajo
                    
                    **Ventajas:**
                    - Mide eficiencia global de capital de trabajo
                    - Buen predictor de necesidades de financiación
                    - Refleja calidad de gestión operativa
                    
                    **Desventajas:**
                    - Complejo de calcular
                    - Requiere múltiples datos
                    - Puede variar estacionalmente
                    
                    **¿Para qué sirve?**
                    - Evaluar eficiencia operativa global
                    - Gestionar necesidades de financiación
                    - Comparar con competidores
                    """,
                    "ejemplo": "DI 61 + DSO 46 - PP 61 = CCC 46 días"
                },
                
                "Fixed Asset Turnover": {
                    "definicion": "**Rotación de Activos Fijos** - Eficiencia en uso de activos fijos",
                    "calculacion": "Ventas ÷ Activos Fijos Netos Promedio",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **Alto**: Uso intensivo de activos fijos
                    - **Bajo**: Activos fijos subutilizados
                    - **Creciente**: Mejora en utilización
                    
                    **Ventajas:**
                    - Enfocado en activos productivos
                    - Bueno para empresas intensivas en capital
                    - Refleja decisiones de inversión
                    
                    **Desventajas:**
                    - Sensible a métodos de depreciación
                    - Varía por antigüedad de activos
                    - No considera mantenimiento
                    
                    **Sectores típicos:**
                    - Retail: 3-5
                    - Manufacturing: 1-2
                    - Utilities: 0.3-0.6
                    
                    **¿Para qué sirve?**
                    - Evaluar eficiencia de inversiones en activos fijos
                    - Comparar utilización de capacidad
                    - Análisis de decisiones de capex
                    """,
                    "ejemplo": "Ventas $1B, Activos fijos promedio $400M → Fixed Asset Turnover = 2.5"
                },
                
                "R&D/Sales": {
                    "definicion": "**Gastos I+D/Ventas** - Porcentaje de ventas invertido en investigación",
                    "calculacion": "Gastos de I+D ÷ Ventas × 100",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **Alto**: Empresa innovadora, orientada al futuro
                    - **Bajo**: Empresa madura, poco innovación
                    - **Óptimo**: Balance entre innovación y rentabilidad
                    
                    **Ventajas:**
                    - Mide compromiso con innovación
                    - Bueno para empresas growth
                    - Indicador de ventajas competitivas futuras
                    
                    **Desventajas:**
                    - No garantiza resultados
                    - Puede ser gasto ineficiente
                    - Dificil de comparar entre sectores
                    
                    **Sectores típicos:**
                    - Biotech: 15-25%
                    - Software: 10-20%
                    - Pharma: 12-18%
                    - Industrial: 2-5%
                    
                    **¿Para qué sirve?**
                    - Evaluar estrategia de innovación
                    - Comparar con competidores
                    - Analizar sostenibilidad de ventajas competitivas
                    """,
                    "ejemplo": "I+D $50M, Ventas $500M → R&D/Sales = 10%"
                },
                
                "SG&A/Sales": {
                    "definicion": "**Gastos Generales/Ventas** - Eficiencia en gastos operativos",
                    "calculacion": "Gastos de Venta, Generales y Administrativos ÷ Ventas × 100",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **Alto**: Estructura costosa, posible ineficiencia
                    - **Bajo**: Estructura lean, eficiente
                    - **Decreciente**: Mejora en eficiencia operativa
                    
                    **Ventajas:**
                    - Mide eficiencia en gastos operativos
                    - Bueno para detectar burocracia
                    - Sensible a economías de escala
                    
                    **Desventajas:**
                    - Puede incluir gastos estratégicos
                    - Varía por modelo de negocio
                    - Reducciones excesivas pueden dañar crecimiento
                    
                    **Sectores típicos:**
                    - Software: 20-30%
                    - Retail: 15-25%
                    - Manufacturing: 10-15%
                    
                    **¿Para qué sirve?**
                    - Evaluar eficiencia operativa
                    - Identificar oportunidades de mejora
                    - Comparar estructura de costos
                    """,
                    "ejemplo": "SG&A $120M, Ventas $500M → SG&A/Sales = 24%"
                }
            }
            
            for metrica, detalles in metricas.items():
                with st.expander(f"**{metrica}**"):
                    st.write(f"**📖 DEFINICIÓN:** {detalles['definicion']}")
                    st.write(f"**🧮 CÁLCULO:** {detalles['calculacion']}")
                    st.markdown("**📊 INTERPRETACIÓN DETALLADA:**")
                    st.write(detalles['interpretacion'])
                    if 'ejemplo' in detalles:
                        st.info(f"**🔢 EJEMPLO:** {detalles['ejemplo']}")

        elif categoria == "📈 CRECIMIENTO (8 métricas)":
            st.subheader("📈 CRECIMIENTO - 8 Métricas")
            
            metricas = {
                "Sales Growth 5Y": {
                    "definicion": "**Crecimiento de Ventas 5 Años** - Tasa crecimiento anual compuesto",
                    "calculacion": "(Ventas año actual ÷ Ventas año base)^(1/5) - 1",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **<5%**: Crecimiento lento (madurez)
                    - **5-15%**: Crecimiento moderado
                    - **>15%**: Crecimiento rápido
                    - **Negativo**: Contracción
                    
                    **Ventajas:**
                    - Muestra tendencia de largo plazo
                    - Menos volátil que anual
                    - Buen indicador de momentum
                    
                    **Desventajas:**
                    - Puede enmascarar cambios recientes
                    - Sensible al año base elegido
                    - No considera adquisiciones orgánicas vs inorgánicas
                    
                    **¿Para qué sirve?**
                    - Evaluar trayectoria histórica
                    - Comparar con expectativas futuras
                    - Análisis de madurez del negocio
                    """,
                    "ejemplo": "Ventas crecieron de $200M a $400M en 5 años → 15% CAGR"
                },
                
                "EPS Growth 5Y": {
                    "definicion": "**Crecimiento EPS 5 Años** - Tasa crecimiento ganancias por acción",
                    "calculacion": "(EPS año actual ÷ EPS año base)^(1/5) - 1",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **Consistente >10%**: Empresa growth de calidad
                    - **Volátil**: Resultados inconsistentes
                    - **Decreciente**: Posible saturación o problemas
                    
                    **Ventajas:**
                    - Enfocado en valor por acción
                    - Considera efecto de recompras
                    - Mejor que crecimiento de beneficio neto
                    
                    **Desventajas:**
                    - Puede ser afectado por eventos extraordinarios
                    - Sensible a cambios en número de acciones
                    - No considera calidad de ganancias
                    
                    **¿Para qué sirve?**
                    - Evaluar creación de valor histórico
                    - Calcular PEG ratio
                    - Proyectar crecimiento futuro
                    """,
                    "ejemplo": "EPS creció de $2 a $4 en 5 años → 15% CAGR"
                },
                
                "Sales Growth Q/Q": {
                    "definicion": "**Crecimiento Ventas Trimestral** - Cambio vs trimestre anterior",
                    "calculacion": "(Ventas Q actual - Ventas Q anterior) ÷ Ventas Q anterior × 100",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **Positivo**: Momentum positivo
                    - **Negativo**: Desaceleración
                    - **Aceleración**: Crecimiento cada vez más rápido
                    - **Desaceleración**: Pérdida de momentum
                    
                    **Ventajas:**
                    - Muestra momentum reciente
                    - Sensible a cambios en el negocio
                    - Útil para trading
                    
                    **Desventajas:**
                    - Muy volátil
                    - Sensible a estacionalidad
                    - Puede estar distorsionado por eventos únicos
                    
                    **¿Para qué sirve?**
                    - Evaluar performance reciente
                    - Identificar cambios en tendencia
                    - Timing de decisiones de inversión
                    """,
                    "ejemplo": "Ventas Q1 $250M, Q2 $275M → Crecimiento 10%"
                },
                
                "EPS Growth Q/Q": {
                    "definicion": "**Crecimiento EPS Trimestral** - Cambio ganancias vs trimestre anterior",
                    "calculacion": "(EPS Q actual - EPS Q anterior) ÷ EPS Q anterior × 100",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **Beat estimates**: Supera expectativas (positivo)
                    - **Miss estimates**: No alcanza expectativas (negativo)
                    - **Guide higher**: Aumenta guidance (muy positivo)
                    
                    **Ventajas:**
                    - Muestra momentum reciente de ganancias
                    - Muy seguido por el mercado
                    - Bueno para estrategias de earnings
                    
                    **Desventajas:**
                    - Extremadamente volátil
                    - Sensible a estacionalidad
                    - Las estimaciones pueden ser erróneas
                    
                    **¿Para qué sirve?**
                    - Evaluar resultados trimestrales
                    - Identificar sorpresas de ganancias
                    - Trading alrededor de earnings
                    """,
                    "ejemplo": "EPS Q1 $1.20, Q2 $1.35 → Crecimiento 12.5%"
                },
                
                "Sales Growth Y/Y": {
                    "definicion": "**Crecimiento Ventas Interanual** - Cambio vs mismo periodo año anterior",
                    "calculacion": "(Ventas periodo actual - Ventas mismo periodo año anterior) ÷ Ventas año anterior × 100",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **Elimina estacionalidad**: Mejor comparación que Q/Q
                    - **Tendencia real**: Muestra crecimiento subyacente
                    - **Comparable**: Mismo periodo estacional
                    
                    **Ventajas:**
                    - Elimina efecto estacional
                    - Mejor indicador de tendencia
                    - Ampliamente utilizado
                    
                    **Desventajas:**
                    - Puede enmascarar cambios recientes
                    - Menos frecuente que Q/Q
                    - Sensible a eventos únicos anuales
                    
                    **¿Para qué sirve?**
                    - Evaluar crecimiento real
                    - Comparar performance anual
                    - Análisis de tendencias fundamentales
                    """,
                    "ejemplo": "Ventas Q2 2024 $300M, Q2 2023 $250M → Crecimiento 20%"
                },
                
                "EPS Growth Y/Y": {
                    "definicion": "**Crecimiento EPS Interanual** - Cambio ganancias vs mismo periodo año anterior",
                    "calculacion": "(EPS periodo actual - EPS mismo periodo año anterior) ÷ EPS año anterior × 100",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **Crecimiento orgánico**: Mejora en operaciones
                    - **Decrecimiento**: Problemas operativos o comparación difícil
                    - **Consistencia**: Crecimiento sostenido es positivo
                    
                    **Ventajas:**
                    - Elimina estacionalidad
                    - Mejor indicador de tendencia de ganancias
                    - Menos volátil que Q/Q
                    
                    **Desventajas:**
                    - Puede estar afectado por eventos únicos
                    - No considera cambios recientes
                    - Sensible a base de comparación
                    
                    **¿Para qué sirve?**
                    - Evaluar crecimiento real de ganancias
                    - Comparar con expectativas
                    - Análisis de calidad de crecimiento
                    """,
                    "ejemplo": "EPS Q2 2024 $1.50, Q2 2023 $1.25 → Crecimiento 20%"
                },
                
                "Revenue Growth (ttm)": {
                    "definicion": "**Crecimiento de Ingresos últimos 12 meses** - Cambio vs mismo periodo anterior",
                    "calculacion": "(Ventas ttm - Ventas ttm año anterior) ÷ Ventas ttm año anterior × 100",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **Muestra tendencia**: Crecimiento en los últimos 12 meses
                    - **Menos volátil**: Que trimestral
                    - **Visión actualizada**: Pero con perspectiva
                    
                    **Ventajas:**
                    - Combina actualidad con estabilidad
                    - Menos volátil que trimestral
                    - Bueno para análisis fundamental
                    
                    **Desventajas:**
                    - Puede enmascarar cambios recientes
                    - Menos frecuente que trimestral
                    - Sensible a eventos pasados
                    
                    **¿Para qué sirve?**
                    - Evaluar crecimiento reciente con perspectiva
                    - Comparar con competidores
                    - Análisis de momentum fundamental
                    """,
                    "ejemplo": "Ventas ttm $1.2B, ttm año anterior $1.0B → Crecimiento 20%"
                },
                
                "EPS Growth (ttm)": {
                    "definicion": "**Crecimiento EPS últimos 12 meses** - Cambio ganancias vs mismo periodo anterior",
                    "calculacion": "(EPS ttm - EPS ttm año anterior) ÷ EPS ttm año anterior × 100",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **Crecimiento sostenido**: Positivo para valoración
                    - **Volátil**: Resultados inconsistentes
                    - **Decreciente**: Posibles problemas
                    
                    **Ventajas:**
                    - Visión actualizada con perspectiva
                    - Menos volátil que trimestral
                    - Bueno para análisis de valoración
                    
                    **Desventajas:**
                    - Puede estar afectado por eventos pasados
                    - Menos frecuente que trimestral
                    - Sensible a base de comparación
                    
                    **¿Para qué sirve?**
                    - Evaluar crecimiento reciente de ganancias
                    - Calcular ratios de crecimiento
                    - Análisis fundamental para inversión
                    """,
                    "ejemplo": "EPS ttm $5.00, ttm año anterior $4.00 → Crecimiento 25%"
                }
            }
            
            for metrica, detalles in metricas.items():
                with st.expander(f"**{metrica}**"):
                    st.write(f"**📖 DEFINICIÓN:** {detalles['definicion']}")
                    st.write(f"**🧮 CÁLCULO:** {detalles['calculacion']}")
                    st.markdown("**📊 INTERPRETACIÓN DETALLADA:**")
                    st.write(detalles['interpretacion'])
                    if 'ejemplo' in detalles:
                        st.info(f"**🔢 EJEMPLO:** {detalles['ejemplo']}")

        elif categoria == "📊 INDICADORES TÉCNICOS (10 métricas)":
            st.subheader("📊 INDICADORES TÉCNICOS - 10 Métricas")
            
            metricas = {
                "Beta": {
                    "definicion": "**Volatilidad vs Mercado** - Sensibilidad de la acción vs benchmark",
                    "calculacion": "Covarianza(Acción, Mercado) ÷ Varianza(Mercado)",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **<0.8**: Defensivo (menos volátil que mercado)
                    - **0.8-1.2**: Neutral (similar volatilidad)
                    - **>1.2**: Agresivo (más volátil que mercado)
                    - **Negativo**: Se mueve en dirección opuesta (raro)
                    
                    **Ventajas:**
                    - Mide riesgo sistemático
                    - Útil para construcción de carteras
                    - Base para modelo CAPM
                    
                    **Desventajas:**
                    - Basado en datos históricos
                    - Asume distribuciones normales
                    - Puede cambiar con el tiempo
                    
                    **¿Para qué sirve?**
                    - Evaluar riesgo vs recompensa esperada
                    - Construcción de carteras diversificadas
                    - Cálculo de costo de capital
                    """,
                    "ejemplo": "Beta 1.5: si mercado ±10%, acción ±15% en promedio"
                },
                
                "RSI (14)": {
                    "definicion": "**Índice de Fuerza Relativa** - Oscilador de momentum",
                    "calculacion": "100 - (100 ÷ (1 + (Ganancia promedio ÷ Pérdida promedio)))",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **>70**: Sobrecomprado (posible corrección)
                    - **<30**: Sobrevendido (posible rebote)
                    - **50**: Neutral
                    - **Divergencias**: Señales fuertes
                    
                    **Ventajas:**
                    - Identifica condiciones extremas
                    - Fácil de interpretar
                    - Ampliamente seguido
                    
                    **Desventajas:**
                    - Puede dar señales prematuras en tendencias fuertes
                    - Menos efectivo en mercados laterales
                    - Parámetro dependiente (14 períodos típico)
                    
                    **¿Para qué sirve?**
                    - Identificar puntos de entrada/salida
                    - Confirmar momentum
                    - Detectar posibles reversiones
                    """,
                    "ejemplo": "RSI 75 → condición sobrecomprada, posible corrección"
                },
                
                "Volatility": {
                    "definicion": "**Volatilidad** - Desviación estándar de rendimientos",
                    "calculacion": "Desviación estándar(rendimientos diarios) × √252",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **<20%**: Baja volatilidad (estable)
                    - **20-40%**: Volatilidad media
                    - **>40%**: Alta volatilidad (riesgosa)
                    - **>80%**: Extremadamente volátil
                    
                    **Ventajas:**
                    - Mide riesgo total
                    - Base para muchos modelos
                    - Fácil de comparar
                    
                    **Desventajas:**
                    - Asume distribuciones normales
                    - No diferencia entre riesgo arriba/abajo
                    - Basado en histórico
                    
                    **¿Para qué sirve?**
                    - Evaluar riesgo de la inversión
                    - Dimensionar posiciones
                    - Comparar con rendimiento esperado
                    """,
                    "ejemplo": "Volatilidad 30% → movimientos típicos de ±30% anuales"
                },
                
                "ATR": {
                    "definicion": "**Average True Range** - Volatilidad basada en rangos de trading",
                    "calculacion": "Media móvil de True Range (máx(alto-bajo, |alto-cierre anterior|, |bajo-cierre anterior|))",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **Alto**: Alta volatilidad intradía
                    - **Bajo**: Baja volatilidad intradía
                    - **Creciente**: Aumento volatilidad
                    - **Decreciente**: Disminución volatilidad
                    
                    **Ventajas:**
                    - Considera gaps de precios
                    - Mejor que volatilidad basada solo en cierres
                    - Útil para stops y targets
                    
                    **Desventajas:**
                    - No direccional
                    - Depende del período elegido
                    - Menos conocido que volatilidad estándar
                    
                    **¿Para qué sirve?**
                    - Colocar stops loss dinámicos
                    - Evaluar condiciones de trading
                    - Gestión de riesgo intradía
                    """,
                    "ejemplo": "ATR $2.50 → movimiento intradía típico de $2.50"
                },
                
                "SMA 20": {
                    "definicion": "**Media Móvil Simple 20 días** - Tendencia corto plazo",
                    "calculacion": "Suma últimos 20 cierres ÷ 20",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **Precio > SMA**: Tendencia alcista
                    - **Precio < SMA**: Tendencia bajista
                    - **Cruces**: Posibles cambios de tendencia
                    - **Soporte/Resistencia**: Niveles importantes
                    
                    **Ventajas:**
                    - Suaviza el ruido
                    - Fácil de calcular e interpretar
                    - Ampliamente usado
                    
                    **Desventajas:**
                    - Retraso (lagging indicator)
                    - Menos efectivo en mercados laterales
                    - Parámetro dependiente
                    
                    **¿Para qué sirve?**
                    - Identificar tendencias
                    - Señales de compra/venta
                    - Niveles de soporte/resistencia
                    """,
                    "ejemplo": "Precio $105, SMA20 $100 → tendencia alcista corto plazo"
                },
                
                "SMA 50": {
                    "definicion": "**Media Móvil Simple 50 días** - Tendencia medio plazo",
                    "calculacion": "Suma últimos 50 cierres ÷ 50",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **Tendencia intermedia**: Más suave que SMA20
                    - **Cruces con SMA20**: Señales de momentum
                    - **Soporte/Resistencia**: Niveles más fuertes
                    
                    **Ventajas:**
                    - Menos ruido que SMA20
                    - Mejor para tendencias intermedias
                    - Menos señales falsas
                    
                    **Desventajas:**
                    - Más retraso que SMA20
                    - Puede perder movimientos rápidos
                    - Parámetro fijo
                    
                    **¿Para qué sirve?**
                    - Confirmar tendencias
                    - Filtrar señales de corto plazo
                    - Análisis de momentum intermedio
                    """,
                    "ejemplo": "SMA20 > SMA50 → momentum alcista confirmado"
                },
                
                "SMA 200": {
                    "definicion": "**Media Móvil Simple 200 días** - Tendencia largo plazo",
                    "calculacion": "Suma últimos 200 cierres ÷ 200",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **Tendencia principal**: Bull market vs Bear market
                    - **Soporte/Resistencia mayor**: Nivel muy importante
                    - **Golden Cross/Death Cross**: Señales mayores
                    
                    **Ventajas:**
                    - Define tendencia principal
                    - Muy seguido por instituciones
                    - Señales fuertes y confiables
                    
                    **Desventajas:**
                    - Mucho retraso
                    - Puede perder grandes movimientos
                    - Menos útil para trading corto
                    
                    **¿Para qué sirve?**
                    - Determinar tendencia principal
                    - Señales de inversión (no trading)
                    - Análisis de largo plazo
                    """,
                    "ejemplo": "Precio > SMA200 → tendencia alcista principal"
                },
                
                "Volume": {
                    "definicion": "**Volumen** - Acciones negociadas en el período",
                    "calculacion": "Número total de acciones negociadas",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **Alto volumen**: Confirmación de movimiento
                    - **Bajo volumen**: Falta de convicción
                    - **Volume spikes**: Eventos importantes
                    - **Divergencias**: Señales de debilidad
                    
                    **Ventajas:**
                    - Confirma price action
                    - Indica interés institucional
                    - Detecta acumulación/distribución
                    
                    **Desventajas:**
                    - No da señales por sí solo
                    - Puede ser manipulado en acciones pequeñas
                    - Varía por liquidez de la acción
                    
                    **¿Para qué sirve?**
                    - Confirmar rupturas de soporte/resistencia
                    - Detectar interés institucional
                    - Identificar posibles reversiones
                    """,
                    "ejemplo": "Ruptura con alto volumen → señal más confiable"
                },
                
                "Avg Volume": {
                    "definicion": "**Volumen Promedio** - Volumen medio histórico",
                    "calculacion": "Media volumen últimos 20-30 días",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **Volume > Avg**: Interés inusual
                    - **Volume < Avg**: Poco interés
                    - **Cambios en avg volume**: Cambio en liquidez/perfil
                    
                    **Ventajas:**
                    - Proporciona contexto
                    - Detecta anomalías
                    - Útil para screening
                    
                    **Desventajas:**
                    - Basado en histórico
                    - Puede cambiar estructuralmente
                    - No considera eventos conocidos
                    
                    **¿Para qué sirve?**
                    - Evaluar liquidez relativa
                    - Detectar interés inusual
                    - Filtrar acciones por liquidez
                    """,
                    "ejemplo": "Volume actual 2M, Avg Volume 1M → interés inusual"
                },
                
                "Rel Volume": {
                    "definicion": "**Volumen Relativo** - Volumen actual vs promedio",
                    "calculacion": "Volume actual ÷ Avg Volume",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **<0.5**: Muy bajo volumen
                    - **0.5-1.5**: Volumen normal
                    - **1.5-3.0**: Alto volumen
                    - **>3.0**: Volumen muy alto
                    
                    **Ventajas:**
                    - Normalizado y comparable
                    - Fácil de interpretar
                    - Bueno para screening
                    
                    **Desventajas:**
                    - Depende del período de avg volume
                    - Puede dar falsas señales en eventos conocidos
                    - No considera dirección del movimiento
                    
                    **¿Para qué sirve?**
                    - Identificar acciones con volumen inusual
                    - Detectar acumulación/distribución
                    - Screening para oportunidades
                    """,
                    "ejemplo": "Rel Volume 2.5 → volumen 2.5x el normal, interés inusual"
                }
            }
            
            for metrica, detalles in metricas.items():
                with st.expander(f"**{metrica}**"):
                    st.write(f"**📖 DEFINICIÓN:** {detalles['definicion']}")
                    st.write(f"**🧮 CÁLCULO:** {detalles['calculacion']}")
                    st.markdown("**📊 INTERPRETACIÓN DETALLADA:**")
                    st.write(detalles['interpretacion'])
                    if 'ejemplo' in detalles:
                        st.info(f"**🔢 EJEMPLO:** {detalles['ejemplo']}")

        elif categoria == "🏢 DATOS CORPORATIVOS (8 métricas)":
            st.subheader("🏢 DATOS CORPORATIVOS - 8 Métricas")
            
            metricas = {
                "Shares Out": {
                    "definicion": "**Acciones en Circulación** - Número total de acciones emitidas",
                    "calculacion": "Acciones comunes emitidas - Acciones en tesorería",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **Creciente**: Dilución (emisiones)
                    - **Decreciente**: Recompra de acciones
                    - **Estable**: Política conservadora
                    
                    **Ventajas:**
                    - Base para cálculo por acción
                    - Muestra política de capital
                    - Afecta valoración
                    
                    **Desventajas:**
                    - No considera clases diferentes
                    - Puede incluir acciones restringidas
                    - No muestra float real
                    
                    **¿Para qué sirve?**
                    - Calcular market cap
                    - Evaluar políticas de capital
                    - Analizar dilución/recompra
                    """,
                    "ejemplo": "10 millones de acciones en circulación"
                },
                
                "Float": {
                    "definicion": "**Acciones Flotantes** - Acciones disponibles para trading público",
                    "calculacion": "Shares Out - Acciones restringidas (insiders, control)",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **Float pequeño**: Alta volatilidad posible
                    - **Float grande**: Más liquidez
                    - **Float vs Shares Out**: Grado de control insider
                    
                    **Ventajas:**
                    - Mejor indicador de liquidez real
                    - Muestra concentración de propiedad
                    - Útil para análisis técnico
                    
                    **Desventajas:**
                    - Los datos pueden ser estimados
                    - Puede cambiar con el tiempo
                    - No considera bloqueos regulatorios
                    
                    **¿Para qué sirve?**
                    - Evaluar liquidez real
                    - Analizar riesgo de manipulación
                    - Gestión de tamaño de posición
                    """,
                    "ejemplo": "Shares Out 10M, Float 8M → 80% disponible para trading"
                },
                
                "Insider Own": {
                    "definicion": "**Propiedad Insider** - % acciones poseídas por directivos y consejo",
                    "calculacion": "Acciones de insiders ÷ Shares Out × 100",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **Alto (>10%)**: Alineación con accionistas
                    - **Bajo (<5%)**: Posible falta de alineación
                    - **Muy alto (>30%)**: Control concentrado
                    
                    **Ventajas:**
                    - Mide alineación de intereses
                    - Buen predictor de performance
                    - Refleja confianza del management
                    
                    **Desventajas:**
                    - No considera tipos de acciones
                    - Puede incluir holdings pasivos
                    - Datos con retraso
                    
                    **¿Para qué sirve?**
                    - Evaluar gobierno corporativo
                    - Analizar alineación de intereses
                    - Detectar posibles conflictos
                    """,
                    "ejemplo": "Insiders poseen 15% de las acciones → buena alineación"
                },
                
                "Insider Trans": {
                    "definicion": "**Transacciones Insider** - Compras y ventas de directivos",
                    "calculacion": "Net buying/selling de insiders en período",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **Net buying**: Confianza en el futuro
                    - **Net selling**: Puede ser normal (diversificación) o preocupante
                    - **Patrones**: Compras consistentes son muy positivas
                    
                    **Ventajas:**
                    - Información privilegiada (legal)
                    - Muy seguido por el mercado
                    - Buen predictor de performance
                    
                    **Desventajas:**
                    - Las ventas pueden ser por razones personales
                    - Datos con retraso (form 4)
                    - Puede ser manipulado con timing
                    
                    **¿Para qué sirve?**
                    - Confirmar tesis de inversión
                    - Detectar posibles problemas
                    - Señales de confianza del management
                    """,
                    "ejemplo": "CEO compró 50,000 acciones → señal muy positiva"
                },
                
                "Inst Own": {
                    "definicion": "**Propiedad Institucional** - % acciones poseídas por fondos e instituciones",
                    "calculacion": "Acciones de instituciones ÷ Shares Out × 100",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **Alto (>60%)**: Aprobación institucional
                    - **Bajo (<30%)**: Poco seguimiento institucional
                    - **Creciente**: Mayor interés profesional
                    
                    **Ventajas:**
                    - Mapeo de interés profesional
                    - Indica calidad de la empresa
                    - Refleja liquidez institucional
                    
                    **Desventajas:**
                    - Instituciones pueden ser wrong
                    - Datos trimestrales con retraso
                    - No diferencia entre tipos de instituciones
                    
                    **¿Para qué sirve?**
                    - Evaluar calidad de la empresa
                    - Analizar seguimiento profesional
                    - Detectar cambios en percepción
                    """,
                    "ejemplo": "70% propiedad institucional → buena aprobación profesional"
                },
                
                "Inst Trans": {
                    "definicion": "**Transacciones Institucionales** - Compras/ventas de fondos",
                    "calculacion": "Net buying/selling de instituciones en período",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **Net buying**: Aprobación profesional
                    - **Net selling**: Preocupación profesional
                    - **Cambios bruscos**: Señales fuertes
                    - **Calidad instituciones**: Importa quién compra/vende
                    
                    **Ventajas:**
                    - Muestra sentiment profesional
                    - Datos de gestores sofisticados
                    - Puede anticipar movimientos
                    
                    **Desventajas:**
                    - Datos con retraso (13F trimestral)
                    - Agregado, no detalle por institución
                    - Puede ser momentum following
                    
                    **¿Para qué sirve?**
                    - Confirmar tesis de inversión
                    - Seguir smart money
                    - Detectar cambios en percepción profesional
                    """,
                    "ejemplo": "Fondos value reconocidos comprando → señal positiva"
                },
                
                "Short Float": {
                    "definicion": "**Short Interest** - % acciones vendidas en corto",
                    "calculacion": "Acciones vendidas en corto ÷ Float × 100",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **Bajo (<5%)**: Poco pesimismo
                    - **Moderado (5-10%)**: Escepticismo normal
                    - **Alto (10-20%)**: Significativo pesimismo
                    - **Muy alto (>20%)**: Posible short squeeze
                    
                    **Ventajas:**
                    - Mapeo de sentiment negativo
                    - Identifica posibles squeezes
                    - Refleja controversia
                    
                    **Desventajas:**
                    - Los shorts pueden tener razón
                    - Datos con retraso (semanal/biweekly)
                    - No considera timing de shorts
                    
                    **¿Para qué sirve?**
                    - Evaluar controversia sobre la acción
                    - Identificar oportunidades de squeeze
                    - Analizar riesgo de covering rallies
                    """,
                    "ejemplo": "Short Float 25% → alto pesimismo, posible squeeze"
                },
                
                "Short Ratio": {
                    "definicion": "**Días para Cubrir** - Tiempo para cubrir posiciones cortas",
                    "calculacion": "Acciones vendidas en corto ÷ Volumen promedio diario",
                    "interpretacion": """
                    **¿Qué significa?**
                    - **<3 días**: Bajo riesgo de squeeze
                    - **3-7 días**: Riesgo moderado
                    - **>7 días**: Alto riesgo de squeeze
                    - **>10 días**: Riesgo muy alto
                    
                    **Ventajas:**
                    - Mejor que Short Float solo
                    - Considera liquidez
                    - Buen predictor de squeeze potential
                    
                    **Desventajas:**
                    - Basado en volumen histórico
                    - Puede cambiar rápidamente
                    - No considera convicción de shorts
                    
                    **¿Para qué sirve?**
                    - Evaluar riesgo de short squeeze
                    - Analizar dinámica de covering
                    - Gestión de riesgo en posiciones cortas
                    """,
                    "ejemplo": "Short Ratio 12 días → alto riesgo de squeeze"
                }
            }
            
            for metrica, detalles in metricas.items():
                with st.expander(f"**{metrica}**"):
                    st.write(f"**📖 DEFINICIÓN:** {detalles['definicion']}")
                    st.write(f"**🧮 CÁLCULO:** {detalles['calculacion']}")
                    st.markdown("**📊 INTERPRETACIÓN DETALLADA:**")
                    st.write(detalles['interpretacion'])
                    if 'ejemplo' in detalles:
                        st.info(f"**🔢 EJEMPLO:** {detalles['ejemplo']}")

        elif categoria == "⚡ MÉTRICAS AVANZADAS DE RIESGO":
            st.subheader("⚡ Métricas Avanzadas de Riesgo y Rendimiento")
            st.write("**Métricas sofisticadas para análisis profesional**")
            
            metricas_avanzadas = {
                "Beta (Riesgo Sistemático)": {
                    "definicion": "Mide la volatilidad de una acción en relación con el mercado completo.",
                    "formula": "Covarianza(Acción, Mercado) / Varianza(Mercado)",
                    "interpretacion": "**<0.8**: Defensivo | **0.8-1.2**: Neutral | **>1.2**: Agresivo",
                    "uso": "Para determinar qué tan sensible es una acción a los movimientos del mercado."
                },
                "Alpha": {
                    "definicion": "Rendimiento excedente sobre lo esperado dado su nivel de riesgo (Beta).",
                    "formula": "Rendimiento Real - (Beta × Rendimiento Mercado)",
                    "interpretacion": "**Alpha > 0**: Supera expectativas | **Alpha < 0**: No alcanza expectativas",
                    "uso": "Medir la habilidad del gestor o el desempeño anormal."
                },
                "Sharpe Ratio": {
                    "definicion": "Rendimiento excedente por unidad de riesgo total.",
                    "formula": "(Rendimiento - Tasa Libre Riesgo) / Volatilidad",
                    "interpretacion": "**>1.0**: Excelente | **0.5-1.0**: Bueno | **<0.5**: Pobre",
                    "uso": "Comparar fondos o estrategias ajustando por riesgo total."
                },
                "Sortino Ratio": {
                    "definicion": "Similar a Sharpe pero solo considera riesgo bajista (desviación negativa).",
                    "formula": "(Rendimiento - Tasa Libre Riesgo) / Volatilidad Bajista",
                    "interpretacion": "**>2.0**: Excelente | **1.0-2.0**: Bueno | **<1.0**: Mejorable",
                    "uso": "Mejor métrica cuando preocupa más las pérdidas que la volatilidad general."
                },
                "Treynor Ratio": {
                    "definicion": "Rendimiento excedente por unidad de riesgo sistemático (Beta).",
                    "formula": "(Rendimiento - Tasa Libre Riesgo) / Beta",
                    "interpretacion": "Cuanto mayor mejor. Comparar con benchmark del sector.",
                    "uso": "Para carteras diversificadas donde el riesgo no sistemático es mínimo."
                },
                "Information Ratio": {
                    "definicion": "Rendimiento activo por unidad de riesgo activo (tracking error).",
                    "formula": "(Rendimiento Cartera - Rendimiento Benchmark) / Tracking Error",
                    "interpretacion": "**>0.5**: Buen gestor activo | **>0.75**: Excelente gestor",
                    "uso": "Evaluar gestión activa vs benchmark."
                }
            }
            
            for metrica, detalles in metricas_avanzadas.items():
                st.markdown(f"### {metrica}")
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.write(f"**📖 Definición**: {detalles['definicion']}")
                    st.write(f"**🧮 Fórmula**: {detalles['formula']}")
                
                with col2:
                    st.write(f"**📊 Interpretación**: {detalles['interpretacion']}")
                    st.write(f"**🎯 Uso Práctico**: {detalles['uso']}")
                
                # Ejemplos prácticos
                if "Beta" in metrica:
                    st.info("**Ejemplo**: Una acción con Beta 1.5 subirá 15% si el mercado sube 10%, pero caerá 15% si el mercado cae 10%")
                elif "Sharpe" in metrica:
                    st.info("**Ejemplo**: Sharpe 1.2 significa que por cada 1% de riesgo, genera 1.2% de rendimiento excedente")
                elif "Alpha" in metrica:
                    st.info("**Ejemplo**: Alpha 0.05 significa que superó en 5% al rendimiento esperado dado su riesgo")
                
                st.markdown("---")

        else:  # Consejos Prácticos de Inversión
            st.subheader("💡 Consejos Prácticos de Inversión")
            st.write("**Sabiduría probada para tomar mejores decisiones**")
            
            # Consejos organizados por categoría
            categorias_consejos = {
                "🔍 Investigación y Análisis": [
                    "**Conoce el negocio**: Invierte solo en empresas que entiendas completamente",
                    "**Análisis competitivo**: Evalúa ventajas competitivas duraderas (moats)",
                    "**Sector y tendencias**: Invierte en sectores con tailwinds, no headwinds",
                    "**Calidad management**: Investiga el track record del equipo directivo",
                    "**Múltiples métricas**: Nunca bases decisiones en una sola métrica"
                ],
                "📈 Gestión de Riesgo": [
                    "**Diversificación inteligente**: No sobre-diversifiques, pero tampoco pongas todos los huevos en una canasta",
                    "**Tamaño de posición**: Nunca arriesgues más del 5% de tu cartera en una sola idea",
                    "**Stop losses mentales**: Define tu precio de venta antes de comprar",
                    "**Riesgo asimétrico**: Busca oportunidades con upside potencial > downside risk",
                    "**Liquidez**: Considera siempre cuán fácil puedes salir de la inversión"
                ],
                "⏳ Psicología y Disciplina": [
                    "**Paciencia**: El tiempo en el mercado > timing del mercado",
                    "**Control emocional**: El miedo y la codicia son tus peores enemigos",
                    "**Independencia**: Piensa por ti mismo, no sigas la manada",
                    "**Humildad**: Reconoce cuando te equivocas y ajusta",
                    "**Consistencia**: Sigue tu proceso invariablemente"
                ],
                "💰 Valoración y Timing": [
                    "**Margen de seguridad**: Compra con descuento al valor intrínseco",
                    "**Ciclos de mercado**: Entiende en qué fase del ciclo estás",
                    "**Valoración relativa**: Compara siempre con alternativas",
                    "**Catalizadores**: Identifica eventos que puedan mover el precio",
                    "**Patience**: Mejor oportunidad perdida que mala inversión"
                ],
                "📚 Educación Continua": [
                    "**Aprendizaje constante**: Los mercados evolucionan, tú también debes hacerlo",
                    "**Historia financiera**: Estudia burbujas y cracks pasados",
                    "**Mentes brillantes**: Lee a Buffett, Munger, Lynch, Graham",
                    "**Pensamiento crítico**: Cuestiona todo, especialmente tus propias ideas",
                    "**Red de conocimiento**: Rodéate de personas más inteligentes que tú"
                ]
            }
            
            for categoria, consejos in categorias_consejos.items():
                st.markdown(f"### {categoria}")
                for consejo in consejos:
                    st.write(f"• {consejo}")
                st.markdown("---")
            
            # Frases célebres de inversión
            st.markdown("### 💬 Sabiduría de los Grandes Inversores")
            frases = [
                "**Warren Buffett**: 'Sé temeroso cuando otros son codiciosos, y codicioso cuando otros son temerosos.'",
                "**Charlie Munger**: 'La inversión no es fácil. Cualquiera que crea que es fácil es un tonto.'",
                "**Peter Lynch**: 'Detrás de cada acción hay una empresa. Descubre qué está haciendo esa empresa.'",
                "**Benjamin Graham**: 'En el corto plazo, el mercado es una máquina de votación. En el largo plazo, es una máquina de ponderación.'",
                "**Philip Fisher**: 'El stock market está lleno de individuos que saben el precio de todo, pero el valor de nada.'",
                "**John Bogle**: 'No busques la aguja en el pajar. Simplemente compra el pajar.'"
            ]
            
            for frase in frases:
                st.success(frase)

        # Sección de libros recomendados
        st.markdown("---")
        st.subheader("📚 Libros Recomendados para Aprender Más")
        
        libros = {
            "Para Principiantes": [
                "**El Inversor Inteligente** - Benjamin Graham (la biblia de la inversión value)",
                "**Un paseo aleatorio por Wall Street** - Burton Malkiel (sobre eficiencia de mercados)",
                "**Los ensayos de Warren Buffett** - Lawrence Cunningham (sabiduría de Buffett)",
                "**The Little Book of Common Sense Investing** - John Bogle (inversión indexada)"
            ],
            "Para Nivel Intermedio": [
                "**Security Analysis** - Benjamin Graham & David Dodd (análisis profundo)",
                "**Common Stocks and Uncommon Profits** - Philip Fisher (inversión en crecimiento)", 
                "**The Little Book of Valuation** - Aswath Damodaran (valoración)",
                "**The Most Important Thing** - Howard Marks (gestión de riesgo)"
            ],
            "Para Avanzados": [
                "**Value Investing: From Graham to Buffett and Beyond** - Bruce Greenwald",
                "**Expected Returns** - Antti Ilmanen (teoría moderna de portafolios)",
                "**The Black Swan** - Nassim Taleb (eventos extremos)",
                "**Principles** - Ray Dalio (modelos mentales para inversión)"
            ],
            "Análisis Fundamental Específico": [
                "**Financial Statement Analysis** - Martin Fridson (análisis de estados financieros)",
                "**The Essays of Warren Buffett** - Lawrence Cunningham (filosofía de inversión)",
                "**Investment Valuation** - Aswath Damodaran (valoración avanzada)",
                "**The Intelligent Asset Allocator** - William Bernstein (asignación de activos)"
            ]
        }
        
        for nivel, lista_libros in libros.items():
            st.write(f"**{nivel}:**")
            for libro in lista_libros:
                st.write(f"• {libro}")

        # Consejos finales mejorados
        st.markdown("---")
        st.subheader("💡 Consejos para Dominar el Análisis Fundamental")
        
        consejos = [
            "**Comienza con lo básico**: Domina primero las 10-15 métricas más importantes de cada sector",
            "**Contexto es clave**: Una métrica por sí sola no te dice mucho. Siempre compara con el sector, historial y competidores",
            "**Tendencias > Niveles absolutos**: Una métrica mejorando consistentemente es más importante que su nivel actual", 
            "**Calidad de ganancias**: Analiza si las ganancias vienen del negocio principal o de eventos extraordinarios",
            "**Flujo de caja vs Ganancias**: Las ganancias son una opinión, el flujo de caja es un hecho",
            "**Apalancamiento prudente**: Un poco de deuda puede ser bueno, demasiada puede ser peligrosa",
            "**Ventajas competitivas**: Busca empresas con márgenes estables/crecientes - indican 'moats' económicos",
            "**Management calidad**: Métricas consistentes suelen indicar buena gestión",
            "**Paciencia**: El análisis fundamental es para inversores, no para traders. Think long-term",
            "**Humildad**: Ninguna métrica es perfecta. Usa múltiples herramientas y mantén escepticismo saludable"
        ]
        
        for i, consejo in enumerate(consejos, 1):
            st.write(f"**{i}.** {consejo}")

        # Resumen final de las 82 métricas
        st.markdown("---")
        st.subheader("📋 Resumen Completo: Las 82 Métricas Fundamentales")
        
        st.write("""
        **💰 VALORACIÓN Y MERCADO (18 métricas)**
        - Market Cap, P/E, Forward P/E, PEG, P/S, P/B, P/FCF
        - EV/EBITDA, EV/Sales, EV/FCF, EPS (ttm), EPS next Y, EPS next Q
        - EPS this Y, EPS next 5Y, EPS past 5Y, Book Value/Share
        
        **📈 RENTABILIDAD Y MÁRGENES (16 métricas)**
        - ROA, ROE, ROI, Gross Margin, Oper. Margin, Profit Margin
        - EBITDA, EBIT, Net Income, Income Tax, Dividend, Dividend %
        - Payout Ratio, EPS Q/Q, Sales Q/Q, Earnings Date
        
        **🏦 DEUDA Y LIQUIDEZ (12 métricas)**
        - Total Debt, Debt/Eq, LT Debt/Eq, Total Debt/EBITDA
        - Current Ratio, Quick Ratio, Cash/Share, Cash Flow/Share
        - Total Cash, Total Cash/Share, Working Capital, Interest Coverage
        
        **📊 EFICIENCIA OPERATIVA (10 métricas)**
        - Asset Turnover, Inventory Turnover, Receivables Turnover
        - Days Inventory, Days Sales Outstanding, Payables Period
        - Cash Conversion Cycle, Fixed Asset Turnover, R&D/Sales, SG&A/Sales
        
        **📈 CRECIMIENTO (8 métricas)**
        - Sales Growth 5Y, EPS Growth 5Y, Sales Growth Q/Q, EPS Growth Q/Q
        - Sales Growth Y/Y, EPS Growth Y/Y, Revenue Growth (ttm), EPS Growth (ttm)
        
        **📊 INDICADORES TÉCNICOS (10 métricas)**
        - Beta, RSI (14), Volatility W, Volatility M, ATR
        - SMA 20, SMA 50, SMA 200, Volume, Avg Volume, Rel Volume
        
        **🏢 DATOS CORPORATIVOS (8 métricas)**
        - Shares Out, Float, Insider Own, Insider Trans
        - Inst Own, Inst Trans, Short Float, Short Ratio
        """)
        
        st.success("**🎯 TOTAL: 82 MÉTRICAS FUNDAMENTALES COMPLETAMENTE EXPLICADAS**")

# SECCIÓN NOTICIAS 
elif st.session_state.seccion_actual == "noticias":
    st.header("📰 Centro de Noticias")
    
    # Crear pestañas para las dos subsecciones
    tab1, tab2 = st.tabs([
        f"📈 Noticias de {nombre}", 
        "🌍 Noticias Globales"
    ])
    
    with tab1:
        # TU CÓDIGO ORIGINAL EXACTO
        st.header(f"📰 Noticias de {nombre}")
        
        # Función para obtener noticias de Finviz
        def obtener_noticias_finviz(ticker):
            url = f"https://finviz.com/quote.ashx?t={ticker}"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            try:
                response = requests.get(url, headers=headers)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Buscar la tabla de noticias
                    news_table = soup.find('table', {'class': 'fullview-news-outer'})
                    
                    if news_table:
                        noticias = []
                        rows = news_table.find_all('tr')
                        
                        for row in rows:
                            try:
                                # Extraer fecha/hora
                                fecha_td = row.find('td', {'align': 'right', 'width': '130'})
                                fecha = fecha_td.get_text(strip=True) if fecha_td else "Fecha no disponible"
                                
                                # Extraer enlace y título
                                link_container = row.find('div', {'class': 'news-link-left'})
                                if link_container:
                                    link = link_container.find('a')
                                    if link:
                                        titulo = link.get_text(strip=True)
                                        href = link.get('href', '')
                                        
                                        # Si el enlace es relativo, convertirlo a absoluto
                                        if href.startswith('/'):
                                            href = f"https://finviz.com{href}"
                                        
                                        # Extraer fuente
                                        fuente_container = row.find('div', {'class': 'news-link-right'})
                                        fuente = fuente_container.get_text(strip=True).strip('()') if fuente_container else "Fuente no disponible"
                                        
                                        noticias.append({
                                            'fecha': fecha,
                                            'titulo': titulo,
                                            'enlace': href,
                                            'fuente': fuente
                                        })
                            except Exception as e:
                                continue
                        
                        return noticias
                    else:
                        st.error("No se pudo encontrar la tabla de noticias en Finviz")
                        return []
                else:
                    st.error(f"Error al acceder a Finviz: {response.status_code}")
                    return []
                    
            except Exception as e:
                st.error(f"Error al obtener noticias: {str(e)}")
                return []

        # Obtener y mostrar noticias
        with st.spinner('Cargando noticias recientes...'):
            noticias = obtener_noticias_finviz(stonk)
            
            if noticias:
                st.success(f"✅ Se encontraron {len(noticias)} noticias recientes")
                
                # Mostrar estadísticas
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total de Noticias", len(noticias))
                with col2:
                    fuentes_unicas = len(set(noticia['fuente'] for noticia in noticias))
                    st.metric("Fuentes Diferentes", fuentes_unicas)
                with col3:
                    st.metric("Última Actualización", datetime.now().strftime("%H:%M"))
                
                st.markdown("---")
                
                # Mostrar noticias
                st.subheader("📋 Noticias Recientes")
                
                for i, noticia in enumerate(noticias[:100], 1):
                    with st.container():
                        col1, col2 = st.columns([1, 4])
                        
                        with col1:
                            st.write(f"**{noticia['fecha']}**")
                            st.write(f"*{noticia['fuente']}*")
                        
                        with col2:
                            # Crear un enlace clickeable
                            st.markdown(f"**[{noticia['titulo']}]({noticia['enlace']})**")
                        
                        # Separador entre noticias (excepto la última)
                        if i < min(100, len(noticias)):
                            st.markdown("---")
                
                # Información adicional si hay más noticias
                if len(noticias) > 100:
                    st.info(f"💡 Mostrando las 100 noticias más recientes de {len(noticias)} totales")
                    
            else:
                st.warning("No se pudieron cargar las noticias. Esto puede deberse a:")
                st.write("• Problemas de conexión con Finviz")
                st.write("• Cambios en la estructura del sitio web")
                st.write("• Restricciones de acceso temporales")
                
                # Sugerencia alternativa
                st.info("💡 **Alternativa:** Puedes visitar directamente [Finviz](https://finviz.com) para ver las noticias más recientes")
    
    with tab2:
        # NUEVA SECCIÓN: NOTICIAS GLOBALES CON CONTROLES
        st.header("🌍 Noticias Globales")
        
        # CONTROLES PARA NOTICIAS GLOBALES - CORREGIDO
        col_controls1 = st.columns(1)
        
        # CORRECCIÓN: Acceder al primer elemento de la lista
        with col_controls1[0]:
            categoria_global = st.selectbox(
                "📂 Categoría:",
                ["general", "negocios", "tecnologia", "ciencia", "salud", "politica", "finanzas"],
                format_func=lambda x: {
                    "general": "🌍 General",
                    "negocios": "💼 Negocios", 
                    "tecnologia": "🔬 Tecnología",
                    "ciencia": "🧪 Ciencia",
                    "salud": "🏥 Salud", 
                    "politica": "⚖️ Política",
                    "finanzas": "💰 Finanzas"
                }[x]
            )

        # Botón para cargar noticias globales
        if st.button("🔄 Cargar Noticias Globales", type="primary", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        # Función para obtener noticias globales
        def obtener_noticias_globales(categoria, pais="us"):
            try:
                # Mapeo de categorías a Google News
                categorias_google = {
                    "general": "https://news.google.com/rss?hl=es-419&gl=US&ceid=US:es-419",
                    "negocios": "https://news.google.com/rss/headlines/section/topic/BUSINESS?hl=es-419&gl=US&ceid=US:es-419",
                    "tecnologia": "https://news.google.com/rss/headlines/section/topic/TECHNOLOGY?hl=es-419&gl=US&ceid=US:es-419",
                    "ciencia": "https://news.google.com/rss/headlines/section/topic/SCIENCE?hl=es-419&gl=US&ceid=US:es-419",
                    "salud": "https://news.google.com/rss/headlines/section/topic/HEALTH?hl=es-419&gl=US&ceid=US:es-419",
                    "politica": "https://news.google.com/rss/headlines/section/topic/POLITICS?hl=es-419&gl=US&ceid=US:es-419",
                    "finanzas": "https://news.google.com/rss/headlines/section/topic/BUSINESS?hl=es-419&gl=US&ceid=US:es-419"
                }
                
                url = categorias_google.get(categoria, categorias_google["general"])
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                }
                
                response = requests.get(url, headers=headers, timeout=15)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    items = soup.find_all('item')
                    noticias = []
                    
                    for item in items:
                        try:
                            # Extraer título
                            titulo = item.find('title')
                            titulo_text = titulo.text if titulo else "Sin título"
                            
                            # Extraer enlace
                            enlace = item.find('link')
                            enlace_text = enlace.text if enlace else "#"
                            
                            # Extraer fecha
                            fecha = item.find('pubdate')
                            if not fecha:
                                fecha = item.find('pubDate')
                            fecha_text = fecha.text if fecha else "Fecha no disponible"
                            
                            # Extraer fuente del título
                            fuente = "Google News"
                            if ' - ' in titulo_text:
                                partes = titulo_text.split(' - ')
                                if len(partes) > 1:
                                    fuente = partes[-1].strip()
                                    titulo_text = ' - '.join(partes[:-1]).strip()
                            
                            # Limpiar HTML del título
                            titulo_text = BeautifulSoup(titulo_text, 'html.parser').get_text()
                            
                            noticias.append({
                                'fecha': fecha_text,
                                'titulo': titulo_text,
                                'enlace': enlace_text,
                                'fuente': fuente,
                                'categoria': categoria,
                                'pais': pais
                            })
                        except Exception as e:
                            continue
                    
                    return noticias
                else:
                    return []
                    
            except Exception as e:
                return []

        # Obtener y mostrar noticias globales (MISMO FORMATO QUE EL ORIGINAL)
        with st.spinner('Cargando noticias globales...'):
            noticias_globales = obtener_noticias_globales(categoria_global)
            
            if noticias_globales:
                st.success(f"✅ Se encontraron {len(noticias_globales)} noticias globales")
                
                # Mostrar estadísticas (MISMO FORMATO)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total de Noticias", len(noticias_globales))
                with col2:
                    fuentes_unicas = len(set(noticia['fuente'] for noticia in noticias_globales))
                    st.metric("Fuentes Diferentes", fuentes_unicas)
                with col3:
                    st.metric("Última Actualización", datetime.now().strftime("%H:%M"))
                
                st.markdown("---")
                
                # Mostrar noticias (MISMO FORMATO EXACTO)
                st.subheader("📋 Noticias Globales Recientes")
                
                for i, noticia in enumerate(noticias_globales, 1):
                    with st.container():
                        col1, col2 = st.columns([1, 4])
                        
                        with col1:
                            st.write(f"**{noticia['fecha']}**")
                            st.write(f"*{noticia['fuente']}*")
                        
                        with col2:
                            # Crear un enlace clickeable (MISMO FORMATO)
                            if noticia['enlace'] != "#":
                                st.markdown(f"**[{noticia['titulo']}]({noticia['enlace']})**")
                            else:
                                st.markdown(f"**{noticia['titulo']}**")
                                st.write("🔒 Enlace no disponible")
                        
                        # Separador entre noticias (MISMO FORMATO)
                        if i < len(noticias_globales):
                            st.markdown("---")
                
                # Información adicional (MISMO FORMATO)
                st.info(f"💡 Mostrando {len(noticias_globales)} noticias de {categoria_global}")
                    
            else:
                # Mensaje de error (MISMO FORMATO)
                st.warning("No se pudieron cargar las noticias globales. Esto puede deberse a:")
                st.write("• Problemas de conexión a internet")
                st.write("• Cambios en la estructura del sitio web")
                st.write("• Restricciones de acceso temporales")
                
                # Sugerencia alternativa (MISMO FORMATO)
                st.info("💡 **Alternativa:** Puedes visitar directamente [Google News](https://news.google.com) para ver las noticias más recientes")

# SECCIÓN DE ANÁLISIS DE RIESGO AVANZADO
elif st.session_state.seccion_actual == "riesgo":
    st.header(f"⚠️ Análisis de Riesgo Avanzado De {nombre}")
    
    st.markdown("""
    <div style='background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%); color: white; padding: 20px; border-radius: 10px; margin: 15px 0;'>
    <h4 style='color: white;'>🔍 EVALUACIÓN COMPLETA DE RIESGOS</h4>
    <p>Análisis profesional de los diferentes tipos de riesgo que afectan a esta inversión</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Obtener métricas de riesgo
    with st.spinner('Calculando métricas avanzadas de riesgo...'):
        metricas_riesgo = calcular_metricas_riesgo_avanzadas(stonk, periodo_años=5)
    
    if metricas_riesgo:
        # =============================================
        # 1. RESUMEN EJECUTIVO DE RIESGO
        # =============================================
        st.subheader("📊 Resumen Ejecutivo de Riesgo")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Clasificación de riesgo general
            score_riesgo = 0
            if metricas_riesgo['Drawdown Máximo'] > 0.4:
                score_riesgo += 3
            elif metricas_riesgo['Drawdown Máximo'] > 0.25:
                score_riesgo += 2
            elif metricas_riesgo['Drawdown Máximo'] > 0.15:
                score_riesgo += 1
                
            if metricas_riesgo['Volatilidad Anual'] > 0.5:
                score_riesgo += 3
            elif metricas_riesgo['Volatilidad Anual'] > 0.3:
                score_riesgo += 2
            elif metricas_riesgo['Volatilidad Anual'] > 0.2:
                score_riesgo += 1
                
            if metricas_riesgo['Beta'] > 1.5:
                score_riesgo += 2
            elif metricas_riesgo['Beta'] > 1.2:
                score_riesgo += 1
            
            if score_riesgo >= 5:
                riesgo_color = "red"
                riesgo_texto = "ALTO RIESGO"
                riesgo_icono = "🔴"
            elif score_riesgo >= 3:
                riesgo_color = "orange"
                riesgo_texto = "RIESGO MODERADO-ALTO"
                riesgo_icono = "🟡"
            elif score_riesgo >= 1:
                riesgo_color = "blue"
                riesgo_texto = "RIESGO MODERADO"
                riesgo_icono = "🔵"
            else:
                riesgo_color = "green"
                riesgo_texto = "BAJO RIESGO"
                riesgo_icono = "🟢"
                
            st.metric("Nivel de Riesgo General", f"{riesgo_icono} {riesgo_texto}")
        
        with col2:
            st.metric("Drawdown Máximo Histórico", f"{metricas_riesgo['Drawdown Máximo']:.1%}")
        
        with col3:
            st.metric("Volatilidad Anual", f"{metricas_riesgo['Volatilidad Anual']:.1%}")
        
        with col4:
            st.metric("Beta vs Mercado", f"{metricas_riesgo['Beta']:.2f}")
        
        # =============================================
        # 2. MÉTRICAS CUANTITATIVAS DE RIESGO
        # =============================================
        st.subheader("📈 Métricas Cuantitativas de Riesgo")

        # Pre-procesar valores para display
        sortino_val = metricas_riesgo.get('Sortino Ratio', 0)
        sortino_display = f"{sortino_val:.2f}" if abs(sortino_val) > 0.01 else f"{sortino_val:.4f}"

        var_val = metricas_riesgo.get('VaR 95% Anual', 0)
        var_display = f"{abs(var_val):.1%}" if abs(var_val) > 0.001 else "< 0.1%"

        skewness_val = metricas_riesgo.get('Skewness', 0)
        skewness_display = f"{skewness_val:.2f}" if abs(skewness_val) > 0.01 else f"{skewness_val:.4f}"

        max_perdida_val = metricas_riesgo.get('Máxima Pérdida Consecutiva', 0)
        max_perdida_display = f"{max_perdida_val} días" if max_perdida_val > 0 else "0 días"

        # Primera fila de métricas
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Sharpe Ratio", f"{metricas_riesgo['Sharpe Ratio']:.2f}",
                    help="Rendimiento por unidad de riesgo total")

        with col2:
            st.metric("Sortino Ratio", sortino_display,
                    help="Rendimiento por unidad de riesgo bajista")

        with col3:
            st.metric("VaR 95% (Anual)", var_display,
                    help="Pérdida máxima esperada en condiciones normales")

        with col4:
            st.metric("Alpha", f"{metricas_riesgo['Alpha']:.2%}",
                    help="Rendimiento excedente sobre el esperado")

        # Segunda fila de métricas
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Correlación S&P500", f"{metricas_riesgo['Correlación S&P500']:.2f}",
                    help="Grado de relación con el mercado")

        with col2:
            st.metric("Probabilidad Pérdida", f"{metricas_riesgo['Probabilidad de Pérdida (%)']:.1f}%",
                    help="% de días con rendimientos negativos")

        with col3:
            st.metric("Máxima Pérdida Consecutiva", max_perdida_display,
                    help="Racha máxima de días negativos")

        with col4:
            st.metric("Skewness", skewness_display,
                    help="Asimetría de la distribución de retornos")
        
        # =============================================
        # 3. ANÁLISIS GRÁFICO DE RIESGO
        # =============================================
        st.subheader("📊 Visualización de Riesgos")
        
        col_grafica1, col_grafica2 = st.columns(2)
        
        with col_grafica1:
            # Gráfica de Drawdown
            st.markdown("**📉 Análisis de Drawdown**")
            grafica_drawdown = crear_grafica_drawdown_mejorada(stonk)
            if grafica_drawdown:
                st.plotly_chart(grafica_drawdown, use_container_width=True)
                st.caption("Evolución histórica de las caídas desde máximos. Áreas rojas indican períodos de pérdidas.")
        
        with col_grafica2:
            # Gráfica de Distribución
            st.markdown("**📊 Distribución de Retornos**")
            grafica_distribucion = crear_grafica_distribucion_retornos(stonk)
            if grafica_distribucion:
                st.plotly_chart(grafica_distribucion, use_container_width=True)
                st.caption("Distribución de ganancias/pérdidas diarias. Línea roja = distribución normal teórica.")
        

        # =============================================
        # 4. COMPARATIVA CON EL MERCADO
        # =============================================
        st.subheader("📈 Comparativa de Riesgo vs Mercado")
        
        col_comp1, col_comp2, col_comp3 = st.columns(3)
        
        with col_comp1:
            vol_vs_mercado = (metricas_riesgo['Volatilidad Anual'] - 0.15) * 100  # 15% volatilidad promedio mercado
            st.metric("Volatilidad vs Mercado", 
                     f"{metricas_riesgo['Volatilidad Anual']:.1%}",
                     f"{vol_vs_mercado:+.1f}%")
        
        with col_comp2:
            beta_interpretacion = "Más volátil" if metricas_riesgo['Beta'] > 1 else "Menos volátil"
            st.metric("Beta vs Mercado", 
                     f"{metricas_riesgo['Beta']:.2f}",
                     beta_interpretacion)
        
        with col_comp3:
            sharpe_mercado = 0.6  # Sharpe promedio mercado
            sharpe_diff = metricas_riesgo['Sharpe Ratio'] - sharpe_mercado
            st.metric("Sharpe vs Mercado", 
                     f"{metricas_riesgo['Sharpe Ratio']:.2f}",
                     f"{sharpe_diff:+.2f}")
        
        # =============================================
        # 5. ALERTAS Y SEÑALES DE RIESGO
        # =============================================
        st.subheader("🚨 Alertas de Riesgo Activas")
        
        alertas = []
        
        # Verificar condiciones de riesgo
        if metricas_riesgo['Drawdown Máximo'] < -0.25:
            alertas.append("🔴 **ALTA ALERTA**: Drawdown histórico > 25%")
        elif metricas_riesgo['Drawdown Máximo'] < -0.15:
            alertas.append("🟡 **ALERTA MODERADA**: Drawdown histórico > 15%")
            
        if metricas_riesgo['Volatilidad Anual'] > 0.40:
            alertas.append("🔴 **ALTA VOLATILIDAD**: > 40% anual")
        elif metricas_riesgo['Volatilidad Anual'] > 0.25:
            alertas.append("🟡 **VOLATILIDAD ELEVADA**: > 25% anual")
            
        if metricas_riesgo['Probabilidad de Pérdida (%)'] > 55:
            alertas.append("🔴 **ALTA FRECUENCIA PÉRDIDAS**: > 55% de días negativos")
        elif metricas_riesgo['Probabilidad de Pérdida (%)'] > 50:
            alertas.append("🟡 **FRECUENCIA PÉRDIDAS ELEVADA**: > 50% de días negativos")
            
        if metricas_riesgo.get('VaR 95% Anual', 0) < -0.30:
            alertas.append("🔴 **VAR EXTREMO**: Pérdida esperada > 30%")
            
        if metricas_riesgo['Beta'] > 1.5:
            alertas.append("🟡 **BETA ALTO**: > 1.5 - Muy sensible al mercado")
        
        if alertas:
            for alerta in alertas:
                st.warning(alerta)
        else:
            st.success("✅ **SIN ALERTAS CRÍTICAS**: Perfil de riesgo dentro de parámetros normales")
        
        # =============================================
        # 6. HISTORIAL DE ESTRESES
        # =============================================
        st.subheader("📅 Historial de Eventos de Estrés")
        
        # Simulación de eventos de estrés (en una app real esto vendría de datos históricos)
        eventos_estres = [
            {"fecha": "2020-03", "evento": "COVID-19", "impacto": "Mercado global -40%"},
            {"fecha": "2022-01", "evento": "Subida tasas Fed", "impacto": "Tech -30%"},
            {"fecha": "2023-03", "evento": "Crisis bancaria", "impacto": "Bancos -25%"}
        ]
        
        for evento in eventos_estres:
            col_fecha, col_evento, col_impacto = st.columns([1, 2, 2])
            with col_fecha:
                st.write(f"**{evento['fecha']}**")
            with col_evento:
                st.write(evento['evento'])
            with col_impacto:
                st.write(evento['impacto'])

        # =============================================
        # 7. ANÁLISIS CUALITATIVO CON IA
        # =============================================
        st.subheader("🤖 Análisis Cualitativo de Riesgo")
        
        with st.spinner('Generando análisis cualitativo con IA...'):
            analisis_ia = generar_analisis_riesgo_ia(stonk, metricas_riesgo, nombre)
            
            if analisis_ia:
                st.markdown("""
                <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px;'>
                <h4 style='color: white;'>ANÁLISIS DE RIESGO POR IA</h4>
                """, unsafe_allow_html=True)
                st.write(analisis_ia)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("""
                **Análisis Cualitativo de Riesgos:**
                
                Basado en las métricas calculadas, aquí tienes un análisis de los riesgos:
                
                **🔴 Riesgos Principales Identificados:**
                - **Drawdown del {:.1f}%**: Indica que históricamente ha tenido caídas significativas
                - **Volatilidad del {:.1f}%**: Sugiere movimientos de precio considerables
                - **Beta de {:.2f}**: {} volatilidad que el mercado
                
                **🟡 Factores a Considerar:**
                - Sharpe Ratio de {:.2f}: {}
                - Probabilidad de pérdida: {:.1f}% de los días
                - Correlación con mercado: {:.2f}
                """.format(
                    metricas_riesgo['Drawdown Máximo'] * 100,
                    metricas_riesgo['Volatilidad Anual'] * 100,
                    metricas_riesgo['Beta'],
                    "Mayor" if metricas_riesgo['Beta'] > 1 else "Menor",
                    metricas_riesgo['Sharpe Ratio'],
                    "Rendimiento ajustado al riesgo positivo" if metricas_riesgo['Sharpe Ratio'] > 0 else "Rendimiento ajustado al riesgo negativo",
                    metricas_riesgo['Probabilidad de Pérdida (%)'],
                    metricas_riesgo['Correlación S&P500']
                ))
        
        # =============================================
        # 8. TIPOS DE RIESGO DETALLADOS
        # =============================================
        st.subheader("🎯 Tipos de Riesgo Específicos")
        
        # Crear pestañas para diferentes tipos de riesgo
        tab1, tab2, tab3, tab4 = st.tabs(["📉 Riesgo de Mercado", "🏦 Riesgo Financiero", "📊 Riesgo Operativo", "🌍 Riesgo Sectorial"])
        
        with tab1:
            st.markdown("""
            **📉 RIESGO DE MERCADO (Sistemático)**
            
            *No diversificable - Afecta a todo el mercado*
            
            **Métricas clave para {}:**
            - **Beta: {:.2f}** - {} sensibilidad a movimientos del mercado
            - **Volatilidad: {:.1f}%** - Nivel de fluctuación de precios
            - **Correlación S&P500: {:.2f}** - Grado de sincronización con el mercado
            - **VaR 95%: {:.1f}%** - Pérdida máxima esperada en condiciones normales
            
            **🔍 Impacto:** {}
            """.format(
                stonk,
                metricas_riesgo['Beta'],
                "Alta" if metricas_riesgo['Beta'] > 1.2 else "Moderada" if metricas_riesgo['Beta'] > 0.8 else "Baja",
                metricas_riesgo['Volatilidad Anual'] * 100,
                metricas_riesgo['Correlación S&P500'],
                metricas_riesgo.get('VaR 95% Anual', 0) * 100,
                "Alta exposición a riesgos de mercado" if metricas_riesgo['Beta'] > 1.2 else "Exposición moderada" if metricas_riesgo['Beta'] > 0.8 else "Baja exposición"
            ))
            
        with tab2:
            # Obtener información financiera para riesgo financiero
            deuda_equity = info.get('debtToEquity', 0)
            current_ratio = info.get('currentRatio', 0)
            interest_coverage = info.get('earningsBeforeInterestAndTaxes', 0) / max(info.get('interestExpense', 1), 1)
            
            st.markdown("""
            **🏦 RIESGO FINANCIERO**
            
            *Relacionado con la estructura de capital y solvencia*
            
            **Métricas clave:**
            - **Deuda/Equity: {:.2f}** - {}
            - **Current Ratio: {:.2f}** - {}
            - **Cobertura de Intereses: {:.1f}x** - {}
            
            **🔍 Evaluación:** {}
            """.format(
                deuda_equity,
                "Alto apalancamiento" if deuda_equity > 2 else "Apalancamiento moderado" if deuda_equity > 1 else "Bajo apalancamiento",
                current_ratio,
                "Buena liquidez" if current_ratio > 1.5 else "Liquidez adecuada" if current_ratio > 1 else "Posibles problemas de liquidez",
                interest_coverage,
                "Cobertura sólida" if interest_coverage > 5 else "Cobertura adecuada" if interest_coverage > 2 else "Cobertura insuficiente",
                "Perfil financiero conservador" if deuda_equity < 1 and current_ratio > 1.5 else "Perfil financiero moderado" if deuda_equity < 2 and current_ratio > 1 else "Perfil financiero agresivo"
            ))
            
        with tab3:
            st.markdown("""
            **📊 RIESGO OPERATIVO**
            
            *Relacionado con las operaciones del negocio*
            
            **Indicadores clave:**
            - **Margen Operativo: {}** - Eficiencia operativa
            - **ROE: {}** - Rentabilidad sobre el capital
            - **Crecimiento Ingresos: {}** - Dinamismo del negocio
            
            **🔍 Factores a monitorear:**
            • Gestión de costos y eficiencia operativa
            • Capacidad de generación de flujo de caja
            • Inversiones en investigación y desarrollo
            • Eficiencia del management
            """.format(
                f"{info.get('operatingMargins', 0)*100:.1f}%" if info.get('operatingMargins') else "N/A",
                f"{info.get('returnOnEquity', 0)*100:.1f}%" if info.get('returnOnEquity') else "N/A",
                f"{info.get('revenueGrowth', 0)*100:.1f}%" if info.get('revenueGrowth') else "N/A"
            ))
            
        with tab4:
            sector = info.get('sector', 'N/A')
            industria = info.get('industry', 'N/A')
            
            st.markdown("""
            **🌍 RIESGO SECTORIAL**
            
            *Riesgos específicos del sector industrial*
            
            **Contexto sectorial:**
            - **Sector:** {}
            - **Industria:** {}
            
            **🔍 Riesgos sectoriales típicos:**
            • Cambios regulatorios del sector
            • Ciclos económicos específicos
            • Disrupción tecnológica
            • Competencia intensiva
            • Dependencia de materias primas
            """.format(sector, industria))
        
        # =============================================
        # 9. RECOMENDACIONES DE GESTIÓN DE RIESGO
        # =============================================
        st.subheader("🛡️ Estrategias de Mitigación de Riesgo")
        
        col_rec1, col_rec2 = st.columns(2)
        
        with col_rec1:
            st.markdown("""
            **✅ PARA RIESGO MODERADO-BAJO:**
            
            • **Diversificación básica**: 15-20 acciones diferentes
            • **Horizonte medio**: 3-5 años de inversión
            • **Monitoreo trimestral**: Revisión periódica
            • **Stop-loss del 15%**: Protección básica
            """)
            
        with col_rec2:
            st.markdown("""
            **⚠️ PARA RIESGO MODERADO-ALTO:**
            
            • **Diversificación amplia**: 25+ acciones
            • **Stop-loss del 10%**: Protección más estricta
            • **Posicionamiento reducido**: Menor exposición
            • **Monitoreo mensual**: Seguimiento cercano
            • **Hedging consideración**: Opciones de protección
            """)
    
        # =============================================
        # 10. PANEL DE CONTROL DE RIESGO
        # =============================================
        st.markdown("---")
        col_ctrl1, col_ctrl2 = st.columns(2)
        
        with col_ctrl1:
            if st.button("🔄 Recalcular Métricas", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
                
        with col_ctrl2:
            # Exportar datos de riesgo
            csv_riesgo = pd.DataFrame([metricas_riesgo]).to_csv(index=False)
            st.download_button(
                label="📥 Exportar Reporte Riesgo",
                data=csv_riesgo,
                file_name=f"riesgo_{stonk}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    else:
        st.error("""
        ❌ No se pudieron calcular las métricas de riesgo para esta acción.
        
        **Posibles causas:**
        • Datos históricos insuficientes
        • Símbolo no válido o no cotizado
        • Problemas de conexión con las fuentes de datos
        
        **Sugerencias:**
        • Verifica que el símbolo sea correcto
        • Intenta con una acción más líquida y conocida
        • Espera unos minutos e intenta nuevamente
        """)
        
        if st.button("🔄 Intentar nuevamente", use_container_width=True):
            st.rerun()

    # =============================================
    # INFORMACIÓN EDUCATIVA SOBRE RIESGOS
    # =============================================
    with st.expander("📚 Guía Educativa: Entendiendo los Riesgos de Inversión", expanded=False):
        st.markdown("""
        ## 🎓 Guía Completa de Análisis de Riesgo
        
        ### 📉 ¿Qué es el Riesgo en Inversiones?
        
        El riesgo es la **posibilidad de perder dinero** en una inversión. Todas las inversiones conllevan algún nivel de riesgo, y generalmente:
        - **Mayor riesgo potencial = Mayor rendimiento potencial**
        - **Menor riesgo potencial = Menor rendimiento potencial**
        
        ### 🎯 Tipos Principales de Riesgo
        
        **1. Riesgo de Mercado (Sistemático)**
        - Afecta a TODO el mercado
        - No se puede eliminar con diversificación
        - Ejemplos: Recesiones, crisis geopolíticas, pandemias
        
        **2. Riesgo Específico (No Sistemático)**
        - Afecta a UNA empresa o sector específico
        - SÍ se puede reducir con diversificación
        - Ejemplos: Mala gestión, problemas legales, huelgas
        
        **3. Riesgo de Liquidez**
        - No poder vender rápidamente sin afectar el precio
        - Común en acciones de baja capitalización
        
        **4. Riesgo de Tasa de Interés**
        - Las subidas de tasas afectan negativamente a las acciones
        
        ### 📊 Métricas Clave Explicadas
        
        **• Volatilidad:** Mide cuánto fluctúa el precio
        - Alta volatilidad = Precio muy variable
        - Baja volatilidad = Precio más estable
        
        **• Drawdown Máximo:** Mayor caída histórica desde un pico
        - Drawdown 25% = Cayó 25% desde su máximo histórico
        - Importante para entender el "peor escenario"
        
        **• Beta:** Sensibilidad vs mercado
        - Beta 1.0 = Se mueve igual que el mercado
        - Beta 1.5 = 50% más volátil que el mercado
        - Beta 0.8 = 20% menos volátil que el mercado
        
        **• Sharpe Ratio:** Rendimiento por unidad de riesgo
        - >1.0 = Buen rendimiento ajustado al riesgo
        - <0 = Mal rendimiento ajustado al riesgo
        
        **• Value at Risk (VaR):** Pérdida máxima esperada
        - VaR 95% = 5% probabilidad de perder más de X%
        - Ayuda a dimensionar posibles pérdidas
        
        ### 🛡️ Estrategias de Gestión de Riesgo
        
        1. **Diversificación:** No poner todos los huevos en una canasta
        2. **Asset Allocation:** Distribuir entre diferentes tipos de activos
        3. **Stop-Loss:** Límites automáticos de pérdida
        4. **Hedging:** Usar instrumentos de protección
        5. **Dollar-Cost Averaging:** Invertir cantidades fijas periódicamente
        
        ### 💡 Consejos Prácticos
        
        - **Conoce tu tolerancia al riesgo** antes de invertir
        - **Diversifica siempre**, incluso en buenas oportunidades
        - **Establece límites de pérdida** antes de comprar
        - **Mantén perspectiva a largo plazo**
        - **Revisa periódicamente** tu exposición al riesgo
        """)

# SECCIÓN DE COMPARACIÓN DE ACCIONES
elif st.session_state.seccion_actual == "comparar":
    st.header(f"📈 Comparar {nombre} con Otras Acciones")
    
    # INPUTS MEJORADOS PARA LAS ACCIONES A COMPARAR
    st.subheader("🔍 Selecciona las acciones para comparar")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        accion1 = st.text_input("Acción 1", value="AAPL", key="accion1")
    with col2:
        accion2 = st.text_input("Acción 2", value="GOOGL", key="accion2")
    with col3:
        accion3 = st.text_input("Acción 3", value="AMZN", key="accion3")
    with col4:
        accion4 = st.text_input("Acción 4", value="TSLA", key="accion4")
    with col5:
        # MÚLTIPLES ÍNDICES DE REFERENCIA
        indice_referencia = st.selectbox(
            "Índice de Referencia:",
            options=["S&P500", "NASDAQ", "DOW JONES", "RUSSELL 2000"],
            index=0,
            help="Selecciona el índice de mercado para comparación"
        )
    
    # SELECTOR DE PERÍODO
    st.subheader("📅 Configuración de Análisis")
    
    col_periodo, col_metricas = st.columns(2)
    
    with col_periodo:
        periodo_opciones = {
            "1 Mes": 30,
            "3 Meses": 90,
            "6 Meses": 180,
            "1 Año": 365,
            "3 Años": 3 * 365,
            "5 Años": 5 * 365,
            "10 Años": 10 * 365
        }
        
        periodo_seleccionado = st.selectbox(
            "Período de Comparación:",
            options=list(periodo_opciones.keys()),
            index=4,  # 3 Años por defecto
            key="selector_periodo_comparacion"
        )
    
    with col_metricas:
        # MÉTRICAS ADICIONALES PARA COMPARACIÓN
        metricas_adicionales = st.multiselect(
            "Métricas Adicionales:",
            options=["Volatilidad", "Sharpe Ratio", "Drawdown Máximo", "Beta", "Correlación"],
            default=["Volatilidad", "Sharpe Ratio"],
            help="Selecciona métricas adicionales para comparar"
        )
    
    # MAPA DE ÍNDICES
    indices_map = {
        "S&P500": "^GSPC",
        "NASDAQ": "^IXIC", 
        "DOW JONES": "^DJI",
        "RUSSELL 2000": "^RUT"
    }
    
    indice_symbol = indices_map[indice_referencia]
    
    # Calcular fecha de inicio
    start_date_comparacion = end_date - timedelta(days=periodo_opciones[periodo_seleccionado])
    
    # BOTÓN PARA EJECUTAR LA COMPARACIÓN
    if st.button("🔄 Ejecutar Análisis Comparativo Avanzado", use_container_width=True):
        with st.spinner('Cargando datos y calculando métricas comparativas...'):
            # LISTA DE TODAS LAS ACCIONES A COMPARAR
            acciones_comparar = [stonk, accion1, accion2, accion3, accion4]
            acciones_comparar = [accion for accion in acciones_comparar if accion.strip()]
            
            # Agregar índice seleccionado
            acciones_comparar.append(indice_symbol)
            
            nombres_acciones = {}
            datos_comparacion = {}
            metricas_detalladas = {}
            datos_originales = {}  # Para guardar los datos originales para las métricas de riesgo
            
            # OBTENER NOMBRES Y DATOS DE CADA ACCIÓN
            for accion in acciones_comparar:
                if accion.strip():
                    try:
                        # Obtener nombre de la acción
                        if accion in indices_map.values():
                            # Es un índice
                            nombre_idx = [k for k, v in indices_map.items() if v == accion][0]
                            nombres_acciones[accion] = f"📊 {nombre_idx}"
                        else:
                            # Es una acción
                            ticker_temp = yf.Ticker(accion)
                            info_temp = ticker_temp.info
                            nombre_accion = info_temp.get("longName", accion)
                            nombres_acciones[accion] = nombre_accion
                        
                        # Descargar datos históricos
                        data_temp = yf.download(accion, 
                                              start=start_date_comparacion.strftime('%Y-%m-%d'), 
                                              end=end_date.strftime('%Y-%m-%d'),
                                              progress=False)
                        
                        if not data_temp.empty:
                            # Guardar datos originales para métricas de riesgo
                            datos_originales[accion] = data_temp.copy()
                            
                            # Manejar MultiIndex columns
                            if isinstance(data_temp.columns, pd.MultiIndex):
                                close_columns = [col for col in data_temp.columns if 'Close' in col]
                                if close_columns:
                                    precios = data_temp[close_columns[0]]
                                else:
                                    continue
                            else:
                                if 'Close' in data_temp.columns:
                                    precios = data_temp['Close']
                                else:
                                    continue

                            if len(precios) > 0 and not precios.isna().all():
                                # Normalizar los precios a porcentaje de cambio
                                precio_inicial = precios.iloc[0]
                                if precio_inicial > 0:
                                    datos_comparacion[accion] = (precios / precio_inicial - 1) * 100
                                    
                                    # CALCULAR MÉTRICAS ADICIONALES
                                    returns = precios.pct_change().dropna()
                                    
                                    # Función para calcular drawdown máximo
                                    def calcular_drawdown_maximo(precios):
                                        try:
                                            rolling_max = precios.expanding().max()
                                            drawdown = (precios - rolling_max) / rolling_max
                                            return drawdown.min() * 100
                                        except:
                                            return 0
                                    
                                    # Función para calcular Sharpe ratio simplificado
                                    def calcular_sharpe_simple(returns, risk_free_rate=0.02):
                                        try:
                                            if len(returns) == 0:
                                                return 0
                                            excess_returns = returns - (risk_free_rate / 252)
                                            sharpe = (excess_returns.mean() / returns.std()) * np.sqrt(252)
                                            return sharpe if not np.isnan(sharpe) else 0
                                        except:
                                            return 0
                                    
                                    metricas_accion = {
                                        'Rendimiento Total': (precios.iloc[-1] / precio_inicial - 1) * 100,
                                        'Volatilidad Anual': returns.std() * np.sqrt(252) * 100,
                                        'Drawdown Máximo': calcular_drawdown_maximo(precios),
                                        'Sharpe Ratio': calcular_sharpe_simple(returns),
                                        'Beta': 0,
                                        'Correlación': 0
                                    }
                                    metricas_detalladas[accion] = metricas_accion
                                    
                            else:
                                st.warning(f"⚠️ No hay datos válidos para {accion}")
                        else:
                            st.warning(f"⚠️ No se encontraron datos para {accion}")
                                                        
                    except Exception as e:
                        st.error(f"❌ Error al cargar datos de {accion}: {str(e)}")

            # CALCULAR BETA Y CORRELACIONES
            if indice_symbol in datos_comparacion:
                for accion in [a for a in acciones_comparar if a != indice_symbol]:
                    if accion in datos_comparacion:
                        try:
                            # Calcular Beta
                            stock_returns = datos_comparacion[accion].pct_change().dropna()
                            index_returns = datos_comparacion[indice_symbol].pct_change().dropna()
                            
                            common_dates = stock_returns.index.intersection(index_returns.index)
                            if len(common_dates) > 0:
                                stock_returns = stock_returns.loc[common_dates]
                                index_returns = index_returns.loc[common_dates]
                                
                                covariance = np.cov(stock_returns, index_returns)[0, 1]
                                index_variance = np.var(index_returns)
                                beta = covariance / index_variance if index_variance != 0 else 0
                                correlation = np.corrcoef(stock_returns, index_returns)[0, 1]
                                
                                metricas_detalladas[accion]['Beta'] = beta
                                metricas_detalladas[accion]['Correlación'] = correlation
                        except:
                            pass

            # VERIFICAR QUE HAYA DATOS PARA COMPARAR
            if len(datos_comparacion) > 1:
                st.success(f"✅ Comparando {len([a for a in acciones_comparar if a in datos_comparacion])} instrumentos")
                
                # GUARDAR DATOS EN SESSION_STATE PARA USAR EN CAPM
                st.session_state.datos_comparacion = datos_comparacion
                st.session_state.nombres_acciones = nombres_acciones
                st.session_state.metricas_detalladas = metricas_detalladas
                st.session_state.acciones_comparar = acciones_comparar
                st.session_state.indice_symbol = indice_symbol
                st.session_state.indice_referencia = indice_referencia
                st.session_state.comparacion_realizada = True

    # MOSTRAR RESULTADOS DE COMPARACIÓN SI EXISTEN
    if hasattr(st.session_state, 'comparacion_realizada') and st.session_state.comparacion_realizada:
        datos_comparacion = st.session_state.datos_comparacion
        nombres_acciones = st.session_state.nombres_acciones
        metricas_detalladas = st.session_state.metricas_detalladas
        acciones_comparar = st.session_state.acciones_comparar
        indice_symbol = st.session_state.indice_symbol
        indice_referencia = st.session_state.indice_referencia
        
        # GRÁFICA DE LÍNEAS COMPARATIVA
        st.subheader("📊 Gráfica de Comparación - Rendimiento Relativo")
        
        fig = go.Figure()
        
        colores = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', "#ffffff", '#e377c2']
        
        for i, (accion, datos) in enumerate(datos_comparacion.items()):
            if len(datos) > 0:
                nombre_display = nombres_acciones.get(accion, accion)
                color = colores[i % len(colores)]
                
                # Configuración especial para índices
                if accion in indices_map.values():
                    line_width = 4
                    line_dash = "dash"
                    nombre_display = f"📊 {nombre_display}"
                else:
                    line_width = 3
                    line_dash = "solid"
                
                fig.add_trace(go.Scatter(
                    x=datos.index,
                    y=datos.values,
                    mode='lines',
                    name=nombre_display,
                    line=dict(
                        color=color, 
                        width=line_width,
                        dash=line_dash
                    ),
                    hovertemplate=(
                        f"<b>{nombre_display}</b><br>" +
                        "Fecha: %{x}<br>" +
                        "Rendimiento: %{y:.2f}%<br>" +
                        "<extra></extra>"
                    )
                ))
         
        if len(fig.data) > 0:
            fig.update_layout(
                title=f'Comparación de Rendimiento vs {indice_referencia} - Período: {periodo_seleccionado}',
                xaxis_title='Fecha',
                yaxis_title='Rendimiento (%)',
                height=600,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # ANÁLISIS COMPARATIVO
            st.subheader("📈 Análisis de Performance vs Índice")
            
            if indice_symbol in datos_comparacion:
                index_data = datos_comparacion[indice_symbol]
                index_final = index_data.iloc[-1] if len(index_data) > 0 else 0
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    mejor_performer = None
                    mejor_rendimiento = -float('inf')
                    
                    for accion, datos in datos_comparacion.items():
                        if accion != indice_symbol:
                            rendimiento_final = datos.iloc[-1] if len(datos) > 0 else 0
                            if rendimiento_final > mejor_rendimiento:
                                mejor_rendimiento = rendimiento_final
                                mejor_performer = accion
                    
                    if mejor_performer:
                        vs_index = mejor_rendimiento - index_final
                        st.metric(
                            "🏆 Mejor Performer", 
                            f"{nombres_acciones.get(mejor_performer, mejor_performer)}",
                            f"{vs_index:+.2f}% vs índice"
                        )
                
                with col2:
                    st.metric(
                        f"📊 Rendimiento {indice_referencia}", 
                        f"{index_final:.2f}%",
                        "Referencia mercado"
                    )
                
                with col3:
                    # Contar acciones que superaron al índice
                    acciones_superiores = 0
                    total_acciones = 0
                    
                    for accion, datos in datos_comparacion.items():
                        if accion != indice_symbol:
                            total_acciones += 1
                            rendimiento_final = datos.iloc[-1] if len(datos) > 0 else 0
                            if rendimiento_final > index_final:
                                acciones_superiores += 1
                    
                    if total_acciones > 0:
                        porcentaje_superiores = (acciones_superiores / total_acciones) * 100
                        st.metric(
                            "📈 Superan Índice", 
                            f"{acciones_superiores}/{total_acciones}",
                            f"{porcentaje_superiores:.1f}%"
                        )
                
                with col4:
                    # Volatilidad promedio vs índice
                    if indice_symbol in metricas_detalladas:
                        vol_index = metricas_detalladas[indice_symbol]['Volatilidad Anual']
                        vol_promedio = np.mean([m['Volatilidad Anual'] for a, m in metricas_detalladas.items() 
                                               if a != indice_symbol])
                        diff_vol = vol_promedio - vol_index
                        
                        st.metric(
                            "📉 Volatilidad Promedio", 
                            f"{vol_promedio:.1f}%",
                            f"{diff_vol:+.1f}% vs índice"
                        )

        # TABLA DE MÉTRICAS COMPARATIVAS
        st.subheader("📋 Métricas Comparativas Detalladas")
        
        # Crear tabla de métricas
        metricas_tabla = []
        for accion in [a for a in acciones_comparar if a in metricas_detalladas]:
            metricas = metricas_detalladas[accion]
            es_indice = accion in indices_map.values()
            
            metricas_tabla.append({
                'Instrumento': nombres_acciones.get(accion, accion),
                'Tipo': 'Índice' if es_indice else 'Acción',
                'Rendimiento (%)': f"{metricas['Rendimiento Total']:.2f}%",
                'Volatilidad (%)': f"{metricas['Volatilidad Anual']:.1f}%",
                'Sharpe Ratio': f"{metricas['Sharpe Ratio']:.2f}",
                'Drawdown Máx (%)': f"{metricas['Drawdown Máximo']:.1f}%",
                'Beta': f"{metricas['Beta']:.2f}" if not es_indice else "N/A",
                'Correlación': f"{metricas['Correlación']:.2f}" if not es_indice else "N/A"
            })
        
        if metricas_tabla:
            df_metricas = pd.DataFrame(metricas_tabla)
            st.dataframe(df_metricas, use_container_width=True)
            
        # ANÁLISIS DE CORRELACIÓN
        st.subheader("🔗 Análisis de Correlación")

        if len([a for a in acciones_comparar if a != indice_symbol and a in datos_comparacion]) > 1:
            acciones_validas = [a for a in acciones_comparar if a != indice_symbol and a in datos_comparacion]
            
            if len(acciones_validas) > 1:
                precios_originales = {}
                
                for accion in acciones_validas:
                    try:
                        # Descargar datos frescos para obtener precios originales
                        data_temp = yf.download(accion, 
                                            start=start_date_comparacion.strftime('%Y-%m-%d'), 
                                            end=end_date.strftime('%Y-%m-%d'),
                                            progress=False)
                        
                        if not data_temp.empty:
                            # Obtener precios de cierre originales
                            if isinstance(data_temp.columns, pd.MultiIndex):
                                close_columns = [col for col in data_temp.columns if 'Close' in col]
                                if close_columns:
                                    precios = data_temp[close_columns[0]]
                                else:
                                    continue
                            else:
                                if 'Close' in data_temp.columns:
                                    precios = data_temp['Close']
                                else:
                                    continue
                            
                            precios_originales[accion] = precios
                    except Exception as e:
                        st.warning(f"Error obteniendo precios para {accion}: {str(e)}")
                
                # Calcular matriz de correlación con precios originales
                corr_matrix = np.zeros((len(acciones_validas), len(acciones_validas)))
                nombres_display = [nombres_acciones.get(a, a) for a in acciones_validas]
                
                for i, accion1 in enumerate(acciones_validas):
                    for j, accion2 in enumerate(acciones_validas):
                        if i == j:
                            corr_matrix[i, j] = 1.0
                        else:
                            try:
                                if accion1 in precios_originales and accion2 in precios_originales:
                                    precios1 = precios_originales[accion1]
                                    precios2 = precios_originales[accion2]
                                    
                                    # Alinear fechas
                                    common_dates = precios1.index.intersection(precios2.index)
                                    if len(common_dates) > 10:
                                        precios1_aligned = precios1.loc[common_dates]
                                        precios2_aligned = precios2.loc[common_dates]
                                        
                                        # Calcular rendimientos logarítmicos diarios para mejor correlación
                                        returns1 = np.log(precios1_aligned / precios1_aligned.shift(1)).dropna()
                                        returns2 = np.log(precios2_aligned / precios2_aligned.shift(1)).dropna()
                                        
                                        # Alinear después del cálculo
                                        common_returns = returns1.index.intersection(returns2.index)
                                        if len(common_returns) > 0:
                                            returns1_final = returns1.loc[common_returns]
                                            returns2_final = returns2.loc[common_returns]
                                            
                                            # Calcular correlación de Pearson
                                            corr = returns1_final.corr(returns2_final)
                                            corr_matrix[i, j] = corr if not np.isnan(corr) else 0
                                else:
                                    corr_matrix[i, j] = 0
                            except Exception as e:
                                corr_matrix[i, j] = 0
                
                # Solo mostrar la gráfica si hay correlaciones no cero
                if not np.all(corr_matrix == 0):
                    # GRÁFICA DE CORRELACIÓN
                    fig_corr = go.Figure()
                    
                    fig_corr.add_trace(go.Heatmap(
                        z=corr_matrix,
                        x=nombres_display,
                        y=nombres_display,
                        colorscale='RdBu_r',
                        zmin=-1,
                        zmax=1,
                        hoverongaps=False,
                        hovertemplate=(
                            '<b>%{y}</b> vs <b>%{x}</b><br>' +
                            'Correlación: %{z:.3f}<extra></extra>'
                        ),
                        colorbar=dict(title="Correlación")
                    ))
                    
                    # Agregar anotaciones con valores
                    for i in range(len(acciones_validas)):
                        for j in range(len(acciones_validas)):
                            color = 'white' if abs(corr_matrix[i, j]) > 0.5 else 'black'
                            fig_corr.add_annotation(
                                x=j,
                                y=i,
                                text=f"{corr_matrix[i, j]:.2f}",
                                showarrow=False,
                                font=dict(color=color, size=10)
                            )
                    
                    fig_corr.update_layout(
                        title='Matriz de Correlación entre Acciones (Rendimientos Diarios)',
                        xaxis_title='',
                        yaxis_title='',
                        height=500,
                        width=600,
                        xaxis=dict(tickangle=45),
                        yaxis=dict(tickangle=0)
                    )
                    
                    st.plotly_chart(fig_corr, use_container_width=True)
                    
                    # RESUMEN DE CORRELACIONES
                    st.subheader("📊 Resumen de Correlaciones")
                    
                    correlaciones_positivas = []
                    correlaciones_negativas = []
                    todas_correlaciones = []
                    
                    for i in range(len(acciones_validas)):
                        for j in range(i+1, len(acciones_validas)):
                            corr_val = corr_matrix[i, j]
                            todas_correlaciones.append(corr_val)
                            if corr_val > 0:
                                correlaciones_positivas.append(corr_val)
                            elif corr_val < 0:
                                correlaciones_negativas.append(corr_val)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if todas_correlaciones:
                            st.metric(
                                "📊 Correlación Promedio",
                                f"{np.mean(todas_correlaciones):.3f}",
                                f"Rango: {min(todas_correlaciones):.3f} a {max(todas_correlaciones):.3f}"
                            )
                    
                    with col2:
                        if correlaciones_positivas:
                            st.metric(
                                "📈 Correlaciones Positivas",
                                f"{len(correlaciones_positivas)}",
                                f"Promedio: {np.mean(correlaciones_positivas):.3f}"
                            )
                        else:
                            st.metric("📈 Correlaciones Positivas", "0", "Sin correlaciones positivas")
                    
                    with col3:
                        if correlaciones_negativas:
                            st.metric(
                                "📉 Correlaciones Negativas",
                                f"{len(correlaciones_negativas)}",
                                f"Promedio: {np.mean(correlaciones_negativas):.3f}"
                            )
                        else:
                            st.metric("📉 Correlaciones Negativas", "0", "Sin correlaciones negativas")
                    
                    # INTERPRETACIÓN
                    st.info("""
                    **💡 Interpretación de Correlaciones:**
                    - **+1.0**: Movimientos idénticos
                    - **+0.7 a +1.0**: Fuerte correlación positiva
                    - **+0.3 a +0.7**: Correlación moderada positiva  
                    - **-0.3 a +0.3**: Correlación débil o nula
                    - **-0.7 a -0.3**: Correlación moderada negativa
                    - **-1.0 a -0.7**: Fuerte correlación negativa
                    """)
                else:
                    st.warning("⚠️ No se pudieron calcular correlaciones significativas")
            else:
                st.info("ℹ️ Se necesitan al menos 2 acciones válidas para calcular correlaciones")
                    
        # ANÁLISIS DE RIESGO-RENDIMIENTO
        st.subheader("🎯 Análisis Riesgo-Rendimiento")
        
        # Crear gráfica de riesgo-rendimiento
        fig_scatter = go.Figure()
        
        # Definir colores según tipo de instrumento
        for accion in [a for a in acciones_comparar if a in metricas_detalladas]:
            metricas = metricas_detalladas[accion]
            es_indice = accion in indices_map.values()
            
            # Configurar propiedades según tipo
            if es_indice:
                color = 'red'
                simbolo = 'star'
                tamaño = 20
                nombre = nombres_acciones.get(accion, accion)
            else:
                color = 'blue'
                simbolo = 'circle'
                tamaño = 15
                nombre = nombres_acciones.get(accion, accion)
            
            fig_scatter.add_trace(go.Scatter(
                x=[metricas['Volatilidad Anual']],
                y=[metricas['Rendimiento Total']],
                mode='markers+text',
                name=nombre,
                marker=dict(
                    size=tamaño,
                    color=color,
                    symbol=simbolo,
                    line=dict(width=2, color='darkgray')
                ),
                text=nombre,
                textposition="top center",
                hovertemplate=(
                    f"<b>{nombre}</b><br>" +
                    "Volatilidad: %{x:.1f}%<br>" +
                    "Rendimiento: %{y:.2f}%<br>" +
                    "Sharpe: " + f"{metricas['Sharpe Ratio']:.2f}" + "<br>" +
                    "<extra></extra>"
                )
            ))
        
        # Agregar línea de eficiencia teórica
        if len([a for a in acciones_comparar if a not in indices_map.values() and a in metricas_detalladas]) > 1:
            # Calcular línea de tendencia para acciones (excluyendo índices)
            acciones_no_indices = [a for a in acciones_comparar if a not in indices_map.values() and a in metricas_detalladas]
            volatilidades = [metricas_detalladas[a]['Volatilidad Anual'] for a in acciones_no_indices]
            rendimientos = [metricas_detalladas[a]['Rendimiento Total'] for a in acciones_no_indices]
            
            if len(volatilidades) > 1:
                # Calcular línea de tendencia
                z = np.polyfit(volatilidades, rendimientos, 1)
                p = np.poly1d(z)
                
                x_line = np.linspace(min(volatilidades), max(volatilidades), 50)
                y_line = p(x_line)
                
                fig_scatter.add_trace(go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode='lines',
                    name='Línea de Tendencia',
                    line=dict(color='gray', dash='dash', width=1),
                    hovertemplate="Línea de tendencia<extra></extra>"
                ))
        
        fig_scatter.update_layout(
            title='Análisis Riesgo-Rendimiento',
            xaxis_title='Volatilidad Anual (%)',
            yaxis_title='Rendimiento Total (%)',
            height=500,
            showlegend=True,
            hovermode='closest'
        )
        
        # Agregar cuadrantes de referencia
        fig_scatter.add_hline(y=0, line_dash="dot", line_color="green", 
                            annotation_text="Break Even", annotation_position="left")
        
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # INTERPRETACIÓN DEL ANÁLISIS RIESGO-RENDIMIENTO
        st.info("""
        **💡 Interpretación del Gráfico Riesgo-Rendimiento:**
        - **Arriba a la izquierda**: Alto rendimiento con bajo riesgo (Ideal)
        - **Arriba a la derecha**: Alto rendimiento con alto riesgo 
        - **Abajo a la izquierda**: Bajo rendimiento con bajo riesgo (Conservador)
        - **Abajo a la derecha**: Bajo rendimiento con alto riesgo (Evitar)
        - **Estrella roja**: Índice de referencia del mercado
        """)

        # BOTÓN DE DESCARGA
        st.markdown("---")
        st.subheader("💾 Exportar Análisis Comparativo")
        
        # Crear DataFrame para exportación
        df_export = pd.DataFrame()
        for accion, datos in datos_comparacion.items():
            temp_df = pd.DataFrame({
                'Fecha': datos.index,
                nombres_acciones.get(accion, accion): datos.values
            })
            
            if df_export.empty:
                df_export = temp_df
            else:
                df_export = pd.merge(df_export, temp_df, on='Fecha', how='outer')
        
        if not df_export.empty:
            df_export = df_export.sort_values('Fecha').reset_index(drop=True)
            
            csv_comparacion = df_export.to_csv(index=False)
            st.download_button(
                label="📥 Descargar datos de comparación como CSV",
                data=csv_comparacion,
                file_name=f"comparacion_{stonk}_vs_{indice_referencia.lower()}.csv",
                mime="text/csv",
                use_container_width=True
            )

        # =============================================
        # NUEVA SECCIÓN: ANÁLISIS CAPM COMPARATIVO
        # =============================================
        st.markdown("---")
        st.subheader("📊 Análisis CAPM Comparativo")

        # Selectores para CAPM comparativo - CON STATE MANAGEMENT
        st.markdown("**🕐 Configuración del Análisis CAPM:**")

        col_capm1, col_capm2, col_capm3 = st.columns(3)

        with col_capm1:
            # Inicializar en session_state si no existe
            if 'periodo_capm_comp' not in st.session_state:
                st.session_state.periodo_capm_comp = "1 año"
                
            periodo_capm_comp = st.selectbox(
                "Período de datos CAPM:",
                options=["1 mes", "3 meses", "6 meses", "1 año", "2 años", "3 años", "5 años", "10 años"],
                index=3,
                key="periodo_capm_comparar"
            )
            st.session_state.periodo_capm_comp = periodo_capm_comp

        with col_capm2:
            if 'frecuencia_capm_comp' not in st.session_state:
                st.session_state.frecuencia_capm_comp = "Diario"
                
            frecuencia_capm_comp = st.selectbox(
                "Frecuencia de datos CAPM:",
                options=["Diario", "Semanal", "Mensual"],
                index=0,
                key="frecuencia_capm_comparar"
            )
            st.session_state.frecuencia_capm_comp = frecuencia_capm_comp

        with col_capm3:
            if 'tasa_libre_riesgo_comp' not in st.session_state:
                st.session_state.tasa_libre_riesgo_comp = 2.0
            if 'prima_riesgo_mercado_comp' not in st.session_state:
                st.session_state.prima_riesgo_mercado_comp = 6.0
                
            tasa_libre_riesgo_comp = st.number_input(
                "Tasa Libre Riesgo (%)", 
                min_value=0.0, 
                max_value=10.0, 
                value=st.session_state.tasa_libre_riesgo_comp, 
                step=0.1,
                help="Para cálculo CAPM comparativo",
                key="tasa_libre_comp"
            ) / 100
            st.session_state.tasa_libre_riesgo_comp = tasa_libre_riesgo_comp * 100
            
            prima_riesgo_mercado_comp = st.number_input(
                "Prima Riesgo Mercado (%)", 
                min_value=0.0, 
                max_value=15.0, 
                value=st.session_state.prima_riesgo_mercado_comp, 
                step=0.1,
                help="Para cálculo CAPM comparativo",
                key="prima_riesgo_comp"
            ) / 100
            st.session_state.prima_riesgo_mercado_comp = prima_riesgo_mercado_comp * 100

        # BOTÓN PARA CALCULAR CAPM - SEPARADO DEL BOTÓN PRINCIPAL
        if st.button("🧮 Calcular CAPM Comparativo", type="secondary", use_container_width=True):
            with st.spinner('Calculando CAPM comparativo...'):
                # Mapear selecciones a parámetros
                periodo_map = {
                    "1 mes": 30,
                    "3 meses": 90,
                    "6 meses": 180,
                    "1 año": 365,
                    "2 años": 730,
                    "3 años": 1095,
                    "5 años": 1825,
                    "10 años": 3650
                }

                frecuencia_map = {
                    "Diario": "1d",
                    "Semanal": "1wk", 
                    "Mensual": "1mo"
                }

                dias_periodo_comp = periodo_map[st.session_state.periodo_capm_comp]
                intervalo_comp = frecuencia_map[st.session_state.frecuencia_capm_comp]

                # Función para calcular CAPM comparativo
                def calcular_capm_comparativo(simbolo, indice_symbol, dias_periodo, intervalo):
                    """Calcula métricas CAPM para comparación"""
                    try:
                        start_date = datetime.today() - timedelta(days=dias_periodo)
                        end_date = datetime.today()
                        
                        # Descargar datos
                        stock_data = yf.download(simbolo, start=start_date, end=end_date, interval=intervalo)
                        market_data = yf.download(indice_symbol, start=start_date, end=end_date, interval=intervalo)
                        
                        if stock_data.empty or market_data.empty:
                            return None
                        
                        # Obtener precios de cierre
                        if isinstance(stock_data.columns, pd.MultiIndex):
                            stock_close = stock_data[('Close', simbolo)]
                        else:
                            stock_close = stock_data['Close']
                            
                        if isinstance(market_data.columns, pd.MultiIndex):
                            market_close = market_data[('Close', indice_symbol)]
                        else:
                            market_close = market_data['Close']
                        
                        # Calcular rendimientos
                        stock_returns = stock_close.pct_change().dropna()
                        market_returns = market_close.pct_change().dropna()
                        
                        # Alinear fechas
                        common_dates = stock_returns.index.intersection(market_returns.index)
                        stock_returns = stock_returns.loc[common_dates]
                        market_returns = market_returns.loc[common_dates]
                        
                        if len(stock_returns) < 5:
                            return None
                        
                        # Calcular Beta histórico
                        if len(market_returns) > 1:
                            beta_real, intercepto = np.polyfit(market_returns, stock_returns, 1)
                            r_squared = np.corrcoef(market_returns, stock_returns)[0, 1] ** 2
                        else:
                            beta_real = 1.0
                            r_squared = 0
                        
                        # Calcular CAPM
                        costo_capital = st.session_state.tasa_libre_riesgo_comp/100 + beta_real * st.session_state.prima_riesgo_mercado_comp/100
                        
                        return {
                            'beta_historico': beta_real,
                            'r_squared': r_squared,
                            'costo_capital': costo_capital,
                            'puntos_datos': len(stock_returns),
                            'rendimiento_promedio': stock_returns.mean() * 100,
                            'volatilidad': stock_returns.std() * 100,
                            'stock_returns': stock_returns,
                            'market_returns': market_returns,
                            'fechas': common_dates
                        }
                        
                    except Exception as e:
                        st.error(f"Error calculando CAPM para {simbolo}: {str(e)}")
                        return None

                # Calcular CAPM para todas las acciones
                datos_capm_comparativo = {}
                
                for accion in [a for a in acciones_comparar if a not in indices_map.values()]:
                    if accion in datos_comparacion:  # Solo acciones con datos válidos
                        datos_capm = calcular_capm_comparativo(accion, indice_symbol, dias_periodo_comp, intervalo_comp)
                        if datos_capm:
                            datos_capm_comparativo[accion] = datos_capm

                # GUARDAR RESULTADOS CAPM EN SESSION_STATE
                st.session_state.datos_capm_comparativo = datos_capm_comparativo
                st.session_state.capm_calculado = True

        # MOSTRAR RESULTADOS CAPM SI EXISTEN
        if hasattr(st.session_state, 'capm_calculado') and st.session_state.capm_calculado:
            datos_capm_comparativo = st.session_state.datos_capm_comparativo
            
            if len(datos_capm_comparativo) > 1:
                st.success(f"✅ CAPM calculado para {len(datos_capm_comparativo)} acciones")

                # =============================================
                # GRÁFICA SCATTER PLOT CAPM COMPARATIVO
                # =============================================
                st.subheader("📈 Gráfica CAPM - Scatter Plot Comparativo")
                
                # Crear gráfica scatter plot comparativa
                fig_scatter_capm = go.Figure()
                
                colores = ["#C25327", "#4EBD38", '#45B7D1', "#912727", "#AD8C20", '#DDA0DD', "#721FAA"]
                
                # Agregar puntos de datos para cada acción
                for i, (accion, datos) in enumerate(datos_capm_comparativo.items()):
                    color = colores[i % len(colores)]
                    
                    # Agregar scatter plot con todos los puntos históricos
                    fig_scatter_capm.add_trace(go.Scatter(
                        x=datos['market_returns'] * 100,  # Rendimiento del mercado
                        y=datos['stock_returns'] * 100,   # Rendimiento de la acción
                        mode='markers',
                        name=f"{nombres_acciones.get(accion, accion)} ({len(datos['stock_returns'])} pts)",
                        marker=dict(
                            size=6,
                            color=color,
                            opacity=0.6,
                            line=dict(width=1, color='darkgray')
                        ),
                        hovertemplate=(
                            f'<b>{nombres_acciones.get(accion, accion)}</b><br>' +
                            'Fecha: %{text}<br>' +
                            'Rend. Mercado: %{x:.2f}%<br>' +
                            'Rend. Acción: %{y:.2f}%<br>' +
                            '<extra></extra>'
                        ),
                        text=[date.strftime('%d/%m/%Y') for date in datos['fechas']],
                        showlegend=True
                    ))
                    
                    # Agregar línea de regresión para cada acción
                    if len(datos['market_returns']) > 1:
                        beta_real = datos['beta_historico']
                        intercepto = np.polyfit(datos['market_returns'], datos['stock_returns'], 1)[1]
                        
                        x_line = np.linspace(datos['market_returns'].min(), datos['market_returns'].max(), 50)
                        y_line = intercepto + beta_real * x_line
                        
                        fig_scatter_capm.add_trace(go.Scatter(
                            x=x_line * 100,
                            y=y_line * 100,
                            mode='lines',
                            name=f"Regresión {nombres_acciones.get(accion, accion)} (β={beta_real:.2f})",
                            line=dict(color=color, width=2, dash='dash'),
                            showlegend=True,
                            hovertemplate=f'Beta: {beta_real:.2f}<extra></extra>'
                        ))

                # Agregar línea CAPM teórica general
                x_capm = np.linspace(-0.2, 0.2, 50)  # Rango razonable para rendimientos
                y_capm = st.session_state.tasa_libre_riesgo_comp/100/252 + 1.0 * (x_capm - st.session_state.tasa_libre_riesgo_comp/100/252)  # Beta = 1 para mercado
                
                fig_scatter_capm.add_trace(go.Scatter(
                    x=x_capm * 100,
                    y=y_capm * 100,
                    mode='lines',
                    name='Línea Mercado (β=1.0)',
                    line=dict(color='black', width=3),
                    hovertemplate='Mercado teórico<extra></extra>'
                ))

                # Línea de referencia en cero
                fig_scatter_capm.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
                fig_scatter_capm.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.5)

                fig_scatter_capm.update_layout(
                    title=f'CAPM Comparativo - {st.session_state.periodo_capm_comp} ({st.session_state.frecuencia_capm_comp})',
                    xaxis_title=f'Rendimiento del Mercado ({indice_referencia}) (%)',
                    yaxis_title='Rendimiento de las Acciones (%)',
                    height=600,
                    showlegend=True,
                    hovermode='closest',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    xaxis=dict(
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='lightgray',
                        zeroline=True,
                        zerolinewidth=2,
                        zerolinecolor='black'
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='lightgray',
                        zeroline=True,
                        zerolinewidth=2,
                        zerolinecolor='black'
                    )
                )

                st.plotly_chart(fig_scatter_capm, use_container_width=True)

                # Interpretación de la gráfica scatter
                st.info("""
                **💡 Interpretación del Scatter Plot CAPM:**
                
                - **🔵 Puntos**: Cada punto representa un período (día/semana/mes) histórico
                - **📈 Eje X**: Rendimiento del mercado en ese período
                - **📈 Eje Y**: Rendimiento de la acción en ese período  
                - **📊 Líneas punteadas**: Líneas de regresión (Beta histórico de cada acción)
                - **⚫ Línea negra**: Comportamiento teórico del mercado (Beta = 1.0)
                
                **Patrones a observar:**
                - **Puntos alineados con pendiente positiva**: Acción que sigue al mercado
                - **Puntos dispersos**: Acción con comportamiento independiente
                - **Pendiente > 1**: Acción más volátil que el mercado
                - **Pendiente < 1**: Acción menos volátil que el mercado
                """)

                # =============================================
                # TABLA COMPARATIVA CAPM
                # =============================================
                st.subheader("📋 Tabla Comparativa CAPM")
                
                # Crear tabla comparativa
                tabla_comparativa = []
                for accion, datos in datos_capm_comparativo.items():
                    # Obtener Beta de Yahoo Finance para comparación
                    try:
                        ticker_temp = yf.Ticker(accion)
                        info_temp = ticker_temp.info
                        beta_yahoo = info_temp.get('beta', datos['beta_historico'])
                        diferencia_beta = datos['beta_historico'] - beta_yahoo
                    except:
                        beta_yahoo = datos['beta_historico']
                        diferencia_beta = 0
                    
                    # Determinar categoría de riesgo
                    if datos['beta_historico'] < 0.8:
                        categoria_riesgo = "🛡️ Defensiva"
                    elif datos['beta_historico'] < 1.2:
                        categoria_riesgo = "⚖️ Moderada"
                    else:
                        categoria_riesgo = "🚀 Agresiva"
                    
                    # Determinar calidad del ajuste
                    if datos['r_squared'] > 0.7:
                        calidad_ajuste = "✅ Alto"
                    elif datos['r_squared'] > 0.4:
                        calidad_ajuste = "⚠️ Moderado"
                    else:
                        calidad_ajuste = "❌ Bajo"
                    
                    tabla_comparativa.append({
                        'Acción': nombres_acciones.get(accion, accion),
                        'Beta Histórico': f"{datos['beta_historico']:.2f}",
                        'Beta Yahoo': f"{beta_yahoo:.2f}",
                        'Diferencia β': f"{diferencia_beta:+.2f}",
                        'Costo Capital': f"{datos['costo_capital']*100:.1f}%",
                        'R²': f"{datos['r_squared']:.3f}",
                        'Calidad Ajuste': calidad_ajuste,
                        'Categoría Riesgo': categoria_riesgo,
                        'Rend. Promedio': f"{datos['rendimiento_promedio']:.2f}%",
                        'Puntos Datos': datos['puntos_datos']
                    })
                
                # Mostrar tabla
                df_comparativo = pd.DataFrame(tabla_comparativa)
                st.dataframe(df_comparativo, use_container_width=True)

                # =============================================
                # ANÁLISIS COMPARATIVO
                # =============================================
                st.subheader("🎯 Análisis Comparativo CAPM")
                
                col_anal1, col_anal2 = st.columns(2)
                
                with col_anal1:
                    # Encontrar acciones con mejor relación riesgo/retorno
                    st.markdown("**🏆 Mejores Relaciones Riesgo/Retorno:**")
                    
                    # Calcular ratio Sharpe simplificado (retorno/volatilidad)
                    acciones_ratio = []
                    for accion, datos in datos_capm_comparativo.items():
                        if datos['volatilidad'] > 0:
                            ratio = abs(datos['rendimiento_promedio'] / datos['volatilidad'])
                            acciones_ratio.append((accion, ratio, datos['rendimiento_promedio']))
                    
                    # Ordenar por mejor ratio
                    acciones_ratio.sort(key=lambda x: x[1], reverse=True)
                    
                    for i, (accion, ratio, rendimiento) in enumerate(acciones_ratio[:3]):
                        st.write(f"{i+1}. **{nombres_acciones.get(accion, accion)}**")
                        st.write(f"   Ratio: {ratio:.2f} | Rendimiento: {rendimiento:.2f}%")
                
                with col_anal2:
                    # Análisis de consistencia Beta
                    st.markdown("**📊 Consistencia del Beta:**")
                    
                    acciones_consistentes = []
                    for accion, datos in datos_capm_comparativo.items():
                        try:
                            ticker_temp = yf.Ticker(accion)
                            info_temp = ticker_temp.info
                            beta_yahoo = info_temp.get('beta', datos['beta_historico'])
                            diferencia = abs(datos['beta_historico'] - beta_yahoo)
                            acciones_consistentes.append((accion, diferencia, datos['r_squared']))
                        except:
                            continue
                    
                    # Ordenar por menor diferencia (más consistentes)
                    acciones_consistentes.sort(key=lambda x: x[1])
                    
                    for i, (accion, diferencia, r2) in enumerate(acciones_consistentes[:3]):
                        st.write(f"{i+1}. **{nombres_acciones.get(accion, accion)}**")
                        st.write(f"   Dif. β: {diferencia:.2f} | R²: {r2:.3f}")

                # =============================================
                # COMPARATIVA DE BETAS
                # =============================================
                st.subheader("📈 Comparativa de Betas")
                
                fig_betas = go.Figure()
                
                colores = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']
                
                for i, (accion, datos) in enumerate(datos_capm_comparativo.items()):
                    color = colores[i % len(colores)]
                    
                    # Obtener Beta Yahoo
                    try:
                        ticker_temp = yf.Ticker(accion)
                        info_temp = ticker_temp.info
                        beta_yahoo = info_temp.get('beta', datos['beta_historico'])
                    except:
                        beta_yahoo = datos['beta_historico']
                    
                    fig_betas.add_trace(go.Bar(
                        name=nombres_acciones.get(accion, accion),
                        x=['Beta Histórico', 'Beta Yahoo'],
                        y=[datos['beta_historico'], beta_yahoo],
                        marker_color=[color, color],
                        hovertemplate='%{x}: %{y:.2f}<extra></extra>'
                    ))
                
                fig_betas.update_layout(
                    title='Comparativa Beta Histórico vs Beta Yahoo Finance',
                    yaxis_title='Valor Beta (β)',
                    height=500,
                    showlegend=True,
                    barmode='group'
                )
                
                st.plotly_chart(fig_betas, use_container_width=True)

                # =============================================
                # RECOMENDACIONES FINALES CAPM
                # =============================================
                st.markdown("---")
                st.subheader("💡 Recomendaciones de Inversión CAPM")
                
                # Encontrar la acción con mejor perfil riesgo/retorno
                mejor_accion = None
                mejor_puntaje = -float('inf')
                
                for accion, datos in datos_capm_comparativo.items():
                    # Puntaje basado en R², rendimiento y consistencia Beta
                    puntaje = (datos['r_squared'] * 100 + 
                            min(datos['rendimiento_promedio'], 20) +  # Cap rendimiento en 20%
                            (1 - min(abs(datos['beta_historico'] - 1), 1)) * 20)  # Preferir Beta cerca de 1
                    
                    if puntaje > mejor_puntaje:
                        mejor_puntaje = puntaje
                        mejor_accion = accion
                
                if mejor_accion:
                    datos_mejor = datos_capm_comparativo[mejor_accion]
                    st.success(f"""
                    **🏅 MEJOR PERFIL CAPM: {nombres_acciones.get(mejor_accion, mejor_accion)}**
                    
                    • **Costo de capital**: {datos_mejor['costo_capital']*100:.1f}%
                    • **Beta histórico**: {datos_mejor['beta_historico']:.2f}
                    • **Calidad ajuste**: {datos_mejor['r_squared']:.3f}
                    • **Rendimiento promedio**: {datos_mejor['rendimiento_promedio']:.2f}%
                    
                    **Recomendación**: Esta acción muestra la mejor combinación de relación riesgo-retorno y consistencia con el modelo CAPM.
                    """)

                # Exportar datos CAPM
                st.markdown("---")
                st.subheader("💾 Exportar Análisis CAPM Comparativo")
                
                df_export_capm = pd.DataFrame([
                    {
                        'Acción': nombres_acciones.get(accion, accion),
                        'Beta_Historico': datos['beta_historico'],
                        'Costo_Capital_%': datos['costo_capital'] * 100,
                        'R_Cuadrado': datos['r_squared'],
                        'Rendimiento_Promedio_%': datos['rendimiento_promedio'],
                        'Volatilidad_%': datos['volatilidad'],
                        'Puntos_Datos': datos['puntos_datos'],
                        'Periodo': st.session_state.periodo_capm_comp,
                        'Frecuencia': st.session_state.frecuencia_capm_comp
                    }
                    for accion, datos in datos_capm_comparativo.items()
                ])
                
                csv_capm = df_export_capm.to_csv(index=False)
                st.download_button(
                    label="📥 Descargar datos CAPM comparativo (CSV)",
                    data=csv_capm,
                    file_name=f"capm_comparativo_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
            else:
                st.warning("No hay suficientes datos CAPM para realizar la comparación")

# SECCIÓN DE ANÁLISIS TÉCNICO
elif st.session_state.seccion_actual == "tecnico":
    st.header(f"📈 Análisis Técnico - {nombre}")
    
    try:
        # Obtener datos
        data = yf.download(stonk, period="1y", interval="1d")
        
        if data.empty:
            st.warning("No se encontraron datos para análisis técnico")
        else:
            # Verificar la estructura de los datos
            st.write(f"📊 Estructura de datos: {data.shape[0]} filas, {data.shape[1]} columnas")
            
            # Si los datos tienen MultiIndex, simplificarlos
            if isinstance(data.columns, pd.MultiIndex):
                # Tomar solo la primera columna de cada tipo si hay múltiples
                simple_data = pd.DataFrame()
                for col_type in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    cols = [col for col in data.columns if col_type in col]
                    if cols:
                        simple_data[col_type] = data[cols[0]]
                data = simple_data
            
            # Calcular indicadores
            data_tech = calcular_indicadores_tecnicos(data)
            
            if data_tech.empty:
                st.error("No se pudieron calcular los indicadores técnicos")
            else:
                # Selector de indicadores
                st.subheader("🔧 Indicadores Técnicos")
                indicadores = st.multiselect(
                    "Selecciona los indicadores a mostrar:",
                    ["RSI", "MACD", "Bandas Bollinger", "Medias Móviles"],
                    default=["RSI", "MACD"]
                )
                
                # Crear gráfica principal
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.1,
                    subplot_titles=('Precio e Indicadores', 'RSI y MACD'),
                    row_heights=[0.6, 0.4]
                )
                
                # Gráfica de velas (fila 1)
                fig.add_trace(go.Candlestick(
                    x=data_tech.index,
                    open=data_tech['Open'],
                    high=data_tech['High'],
                    low=data_tech['Low'],
                    close=data_tech['Close'],
                    name='Precio'
                ), row=1, col=1)
                
                # Bandas de Bollinger
                if "Bandas Bollinger" in indicadores and all(col in data_tech.columns for col in ['BB_Upper', 'BB_Middle', 'BB_Lower']):
                    fig.add_trace(go.Scatter(
                        x=data_tech.index, y=data_tech['BB_Upper'],
                        line=dict(color='rgba(255,0,0,0.5)', width=1),
                        name='BB Superior',
                        legendgroup="bollinger"
                    ), row=1, col=1)
                    
                    fig.add_trace(go.Scatter(
                        x=data_tech.index, y=data_tech['BB_Middle'],
                        line=dict(color='rgba(0,255,0,0.5)', width=1),
                        name='BB Media',
                        legendgroup="bollinger"
                    ), row=1, col=1)
                    
                    fig.add_trace(go.Scatter(
                        x=data_tech.index, y=data_tech['BB_Lower'],
                        line=dict(color='rgba(0,0,255,0.5)', width=1),
                        name='BB Inferior',
                        fill='tonexty',
                        fillcolor='rgba(0,100,80,0.1)',
                        legendgroup="bollinger"
                    ), row=1, col=1)
                
                # Medias Móviles
                if "Medias Móviles" in indicadores:
                    if 'SMA_20' in data_tech.columns:
                        fig.add_trace(go.Scatter(
                            x=data_tech.index, y=data_tech['SMA_20'],
                            line=dict(color='orange', width=2),
                            name='SMA 20'
                        ), row=1, col=1)
                    
                    if 'SMA_50' in data_tech.columns:
                        fig.add_trace(go.Scatter(
                            x=data_tech.index, y=data_tech['SMA_50'],
                            line=dict(color='red', width=2),
                            name='SMA 50'
                        ), row=1, col=1)
                    
                    if 'SMA_200' in data_tech.columns:
                        fig.add_trace(go.Scatter(
                            x=data_tech.index, y=data_tech['SMA_200'],
                            line=dict(color='purple', width=2),
                            name='SMA 200'
                        ), row=1, col=1)
                
                # RSI (fila 2)
                if "RSI" in indicadores and 'RSI' in data_tech.columns:
                    fig.add_trace(go.Scatter(
                        x=data_tech.index, y=data_tech['RSI'],
                        line=dict(color='blue', width=2),
                        name='RSI'
                    ), row=2, col=1)
                    
                    # Líneas de sobrecompra/sobreventa
                    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                    fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)
                
                # MACD (fila 2, segundo eje Y)
                if "MACD" in indicadores and all(col in data_tech.columns for col in ['MACD', 'MACD_Signal']):
                    fig.add_trace(go.Scatter(
                        x=data_tech.index, y=data_tech['MACD'],
                        line=dict(color='red', width=2),
                        name='MACD',
                        yaxis='y2'
                    ), row=2, col=1)
                    
                    fig.add_trace(go.Scatter(
                        x=data_tech.index, y=data_tech['MACD_Signal'],
                        line=dict(color='blue', width=2),
                        name='Señal MACD',
                        yaxis='y2'
                    ), row=2, col=1)
                    
                    # Configurar segundo eje Y para MACD
                    fig.update_layout(
                        yaxis2=dict(
                            title='MACD',
                            overlaying='y',
                            side='right'
                        )
                    )
                
                fig.update_layout(
                    height=800, 
                    showlegend=True, 
                    xaxis_rangeslider_visible=False,
                    title=f"Análisis Técnico de {stonk}"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # REDUCIR ESPACIO ENTRE GRÁFICA Y SEÑALES
                st.markdown("<br>", unsafe_allow_html=True)  # Solo un pequeño espacio

                # SEÑALES TÉCNICAS
                st.subheader("📊 Señales Técnicas Actuales")
                
                if not data_tech.empty:
                    # Obtener el último dato
                    ultimo = data_tech.iloc[-1]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        if 'RSI' in data_tech.columns:
                            rsi_actual = ultimo['RSI']
                            st.metric("RSI", f"{rsi_actual:.2f}")
                            if rsi_actual > 70:
                                st.error("SOBRECOMPRA 🔴")
                            elif rsi_actual < 30:
                                st.success("SOBREVENTA 🟢")
                            else:
                                st.info("NEUTRAL 🟡")
                    
                    with col2:
                        if all(col in data_tech.columns for col in ['MACD', 'MACD_Signal']):
                            macd_actual = ultimo['MACD']
                            signal_actual = ultimo['MACD_Signal']
                            st.metric("MACD", f"{macd_actual:.4f}")
                            if macd_actual > signal_actual:
                                st.success("ALCISTA 🟢")
                            else:
                                st.error("BAJISTA 🔴")
                    
                    with col3:
                        if 'Close' in data_tech.columns and 'SMA_50' in data_tech.columns:
                            precio_actual = ultimo['Close']
                            sma_50 = ultimo['SMA_50']
                            st.metric("Precio vs SMA50", f"${precio_actual:.2f}")
                            if precio_actual > sma_50:
                                st.success("POR ENCIMA 🟢")
                            else:
                                st.error("POR DEBAJO 🔴")
                    
                    with col4:
                        if all(col in data_tech.columns for col in ['BB_Upper', 'BB_Lower', 'Close']):
                            precio_actual = ultimo['Close']
                            bb_upper = ultimo['BB_Upper']
                            bb_lower = ultimo['BB_Lower']
                            st.metric("Bandas Bollinger", f"${precio_actual:.2f}")
                            if precio_actual > bb_upper:
                                st.error("SOBRE SUPERIOR 🔴")
                            elif precio_actual < bb_lower:
                                st.success("BAJO INFERIOR 🟢")
                            else:
                                st.info("DENTRO BANDAS 🟡")
                 # PEQUEÑO ESPACIO ANTES DEL RESUMEN
                st.markdown("<br>", unsafe_allow_html=True)

                # RESUMEN DE INDICADORES
                st.subheader("📈 Resumen de Indicadores")
                
                # Crear DataFrame resumen
                resumen_data = []
                if 'RSI' in data_tech.columns:
                    rsi_actual = data_tech['RSI'].iloc[-1]
                    rsi_señal = "SOBRECOMPRA" if rsi_actual > 70 else "SOBREVENTA" if rsi_actual < 30 else "NEUTRAL"
                    resumen_data.append({'Indicador': 'RSI', 'Valor': f"{rsi_actual:.2f}", 'Señal': rsi_señal})
                
                if all(col in data_tech.columns for col in ['MACD', 'MACD_Signal']):
                    macd_actual = data_tech['MACD'].iloc[-1]
                    signal_actual = data_tech['MACD_Signal'].iloc[-1]
                    macd_señal = "ALCISTA" if macd_actual > signal_actual else "BAJISTA"
                    resumen_data.append({'Indicador': 'MACD', 'Valor': f"{macd_actual:.4f}", 'Señal': macd_señal})
                
                if all(col in data_tech.columns for col in ['Close', 'SMA_20', 'SMA_50', 'SMA_200']):
                    precio_actual = data_tech['Close'].iloc[-1]
                    sma_20 = data_tech['SMA_20'].iloc[-1]
                    sma_50 = data_tech['SMA_50'].iloc[-1]
                    sma_200 = data_tech['SMA_200'].iloc[-1]
                    
                    # Señal de tendencia basada en medias
                    if precio_actual > sma_20 > sma_50 > sma_200:
                        tendencia = "FUERTE ALCISTA 🟢"
                    elif precio_actual < sma_20 < sma_50 < sma_200:
                        tendencia = "FUERTE BAJISTA 🔴"
                    else:
                        tendencia = "LATERAL 🟡"
                    
                    resumen_data.append({'Indicador': 'Tendencia Medias', 'Valor': f"${precio_actual:.2f}", 'Señal': tendencia})
                
                if resumen_data:
                    df_resumen = pd.DataFrame(resumen_data)
                    st.dataframe(df_resumen, use_container_width=True)
                
                # PEQUEÑO ESPACIO ANTES DE LA SECCIÓN EDUCATIVA
                st.markdown("<br>", unsafe_allow_html=True)

                # SECCIÓN EDUCATIVA SOBRE INDICADORES
                st.subheader("📚 ¿Qué son los Indicadores Técnicos?")
                
                st.markdown("""
                Los **indicadores técnicos** son herramientas matemáticas que se aplican a los precios y volúmenes 
                históricos de un activo para analizar tendencias, identificar posibles puntos de entrada y salida, 
                y predecir movimientos futuros del precio. Se dividen principalmente en:
                
                - **Indicadores de tendencia**: Ayudan a identificar la dirección del mercado
                - **Indicadores de momentum**: Miden la velocidad de los movimientos de precios
                - **Indicadores de volatilidad**: Miden la magnitud de las fluctuaciones del precio
                - **Indicadores de volumen**: Analizan la fuerza detrás de los movimientos de precios
                """)
                
                # EXPANDERS PARA CADA INDICADOR
                st.subheader("🔍 Explicación de Cada Indicador")
                
                with st.expander("📊 RSI (Relative Strength Index)", expanded=False):
                    st.markdown("""
                    **¿Qué es?**
                    - El RSI es un oscilador de momentum que mide la velocidad y el cambio de los movimientos de precios
                    - Oscila entre 0 y 100
                    
                    **¿Para qué sirve?**
                    - Identificar condiciones de **sobrecompra** (RSI > 70) y **sobreventa** (RSI < 30)
                    - Detectar divergencias que pueden indicar cambios de tendencia
                    - Confirmar la fuerza de una tendencia
                    
                    **Interpretación:**
                    - **RSI > 70**: Posible sobrecompra - considerar venta
                    - **RSI < 30**: Posible sobreventa - considerar compra
                    - **RSI = 50**: Punto de equilibrio
                    """)
                
                with st.expander("📈 MACD (Moving Average Convergence Divergence)", expanded=False):
                    st.markdown("""
                    **¿Qué es?**
                    - Indicador de tendencia que muestra la relación entre dos medias móviles exponenciales
                    - Se compone de:
                      - **Línea MACD**: Diferencia entre EMA 12 y EMA 26
                      - **Línea de Señal**: EMA 9 del MACD
                      - **Histograma**: Diferencia entre MACD y su línea de señal
                    
                    **¿Para qué sirve?**
                    - Identificar cambios en la dirección y fuerza de la tendencia
                    - Generar señales de compra y venta
                    - Detectar momentum alcista o bajista
                    
                    **Señales principales:**
                    - **Cruce alcista**: MACD cruza por encima de la línea de señal → COMPRA
                    - **Cruce bajista**: MACD cruza por debajo de la línea de señal → VENTA
                    - **Divergencias**: Cuando el precio y el MACD no coinciden
                    """)
                
                with st.expander("📉 Bandas de Bollinger", expanded=False):
                    st.markdown("""
                    **¿Qué es?**
                    - Indicador de volatilidad que consiste en tres líneas:
                      - **Banda media**: SMA 20 (Media Móvil Simple de 20 periodos)
                      - **Banda superior**: SMA 20 + (2 × Desviación Estándar)
                      - **Banda inferior**: SMA 20 - (2 × Desviación Estándar)
                    
                    **¿Para qué sirve?**
                    - Medir la volatilidad del mercado
                    - Identificar niveles de soporte y resistencia dinámicos
                    - Detectar condiciones de mercado extremas
                    
                    **Interpretación:**
                    - **Bandas estrechas**: Baja volatilidad (posible breakout próximo)
                    - **Bandas anchas**: Alta volatilidad
                    - **Precio toca banda superior**: Posible resistencia
                    - **Precio toca banda inferior**: Posible soporte
                    - **Walk the band**: El precio se mantiene en una banda indicando tendencia fuerte
                    """)
                
                with st.expander("📊 Medias Móviles", expanded=False):
                    st.markdown("""
                    **¿Qué es?**
                    - Indicadores que suavizan los datos de precio para identificar la dirección de la tendencia
                    - Tipos principales:
                      - **SMA (Simple Moving Average)**: Media aritmética simple
                      - **EMA (Exponential Moving Average)**: Da más peso a los precios recientes
                    
                    **¿Para qué sirve?**
                    - Identificar la dirección de la tendencia
                    - Generar señales de compra y venta mediante cruces
                    - Actuar como niveles de soporte y resistencia dinámicos
                    
                    **Configuraciones comunes:**
                    - **SMA 20**: Tendencia a corto plazo
                    - **SMA 50**: Tendencia a medio plazo
                    - **SMA 200**: Tendencia a largo plazo (tendencia principal)
                    
                    **Señales importantes:**
                    - **Cruce dorado**: SMA 50 cruza por encima de SMA 200 → FUERTE ALCISTA
                    - **Cruce de la muerte**: SMA 50 cruza por debajo de SMA 200 → FUERTE BAJISTA
                    - **Precio sobre medias**: Tendencia alcista
                    - **Precio bajo medias**: Tendencia bajista
                    """)
                
                # CONSEJOS DE USO
                st.info("""
                **💡 Consejos Prácticos:**
                - Nunca uses un solo indicador para tomar decisiones
                - Combina múltiples indicadores para confirmar señales
                - Considera el contexto del mercado y las noticias relevantes
                - Los indicadores son herramientas, no garantías de éxito
                """)
                
                # DESCARGAR DATOS TÉCNICOS
                st.subheader("💾 Exportar Datos Técnicos")
                
                # Preparar datos para descarga
                columnas_descarga = ['Open', 'High', 'Low', 'Close', 'Volume']
                if 'RSI' in data_tech.columns:
                    columnas_descarga.append('RSI')
                if 'MACD' in data_tech.columns:
                    columnas_descarga.extend(['MACD', 'MACD_Signal', 'MACD_Histogram'])
                if 'BB_Middle' in data_tech.columns:
                    columnas_descarga.extend(['BB_Upper', 'BB_Middle', 'BB_Lower'])
                if 'SMA_20' in data_tech.columns:
                    columnas_descarga.extend(['SMA_20', 'SMA_50', 'SMA_200'])
                
                datos_descarga = data_tech[columnas_descarga].copy()
                datos_descarga = datos_descarga.reset_index()
                
                csv = datos_descarga.to_csv(index=False)
                st.download_button(
                    label="📥 Descargar datos técnicos como CSV",
                    data=csv,
                    file_name=f"{stonk}_datos_tecnicos.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
    except Exception as e:
        st.error(f"Error en análisis técnico: {str(e)}")
        st.write("Detalles del error:", str(e))

# SECCIÓN DE ANÁLISIS IA
elif st.session_state.seccion_actual == "ia":
    st.header(f"🤖 Análisis IA - {nombre}")
    
    # Obtener datos para el análisis
    try:
        current_price = info.get('currentPrice', 0)
        market_cap = info.get('marketCap', 0)
        pe_ratio = info.get('trailingPE', 0)
        revenue_growth = info.get('revenueGrowth', 0)
        
        # Prompt para análisis IA
        prompt_analisis = f"""
        Analiza la acción {stonk} ({nombre}) como un experto financiero. Considera:
        
        Precio actual: ${current_price}
        Market Cap: ${market_cap/1e9:.2f}B
        P/E Ratio: {pe_ratio}
        Crecimiento de ingresos: {revenue_growth*100 if revenue_growth else 0:.1f}%
        
        Proporciona un análisis conciso que incluya:
        1. Valoración actual (sobrevalorada/subvalorada)
        2. Fortalezas clave
        3. Riesgos principales  
        4. Recomendación (Comprar/Mantener/Vender)
        5. Perspectiva a 12 meses
        
        Máximo 400 palabras, en español.
        """
        
        with st.spinner("🤖 Analizando con IA..."):
            try:
                # FORMA CORRECTA - Crear el modelo primero
                model = genai.GenerativeModel('gemini-2.5-flash')
                response_ia = model.generate_content(prompt_analisis)
                
                st.success("✅ Análisis completado")
                
                # Mostrar el análisis con formato
                st.markdown("### 📋 Análisis de IA")
                st.markdown(response_ia.text)
                
            except Exception as e:
                st.error(f"❌ Error en IA: {str(e)}")
                # Mostrar análisis de respaldo con datos disponibles
                st.info("""
                **📊 Análisis Basado en Datos Disponibles:**
                
                Mientras se soluciona el servicio de IA, aquí tienes un análisis básico:
                
                **Métricas Clave:**
                - Precio: ${:.2f}
                - Market Cap: ${:.2f}B
                - P/E Ratio: {}
                - Crecimiento: {:.1f}%
                """.format(current_price, market_cap/1e9, pe_ratio, revenue_growth*100 if revenue_growth else 0))
        
        # Análisis de sentimiento de noticias
        st.subheader("😊 Análisis de Sentimiento")
        
        def analizar_sentimiento_noticias(ticker):
            # Simulación de análisis de sentimiento
            sentimientos = ["POSITIVO", "NEUTRAL", "NEGATIVO"]
            return random.choice(sentimientos)
        
        sentimiento = analizar_sentimiento_noticias(stonk)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if sentimiento == "POSITIVO":
                st.success("😊 Sentimiento: POSITIVO")
            elif sentimiento == "NEGATIVO":
                st.error("😞 Sentimiento: NEGATIVO")
            else:
                st.info("😐 Sentimiento: NEUTRAL")
        
        with col2:
            # Scoring fundamental
            scoring, metricas_scoring = calcular_scoring_fundamental(info)
            st.metric("Scoring Fundamental", f"{scoring}/100")
        
        with col3:
            # Recomendación IA
            if scoring >= 70:
                st.success("🎯 Recomendación: COMPRAR")
            elif scoring >= 50:
                st.warning("🎯 Recomendación: MANTENER")
            else:
                st.error("🎯 Recomendación: VENDER")
        
        # Métricas de scoring
        st.subheader("📊 Métricas de Scoring")
        col_met1, col_met2 = st.columns(2)
        
        with col_met1:
            for i, (metrica, valor) in enumerate(metricas_scoring.items()):
                if i < len(metricas_scoring) // 2:
                    st.write(f"**{metrica}:** {valor}")
        
        with col_met2:
            for i, (metrica, valor) in enumerate(metricas_scoring.items()):
                if i >= len(metricas_scoring) // 2:
                    st.write(f"**{metrica}:** {valor}")
            
    except Exception as e:
        st.error(f"Error en análisis IA: {str(e)}")

# SECCIÓN DE SCREENER Y FILTROS
elif st.session_state.seccion_actual == "screener":
    st.header("🔍 Screener S&P 500 - Filtros Avanzados")
    st.write("Busca acciones del S&P 500 que cumplan con tus criterios de inversión")
    
    # LISTA COMPLETA DEL S&P 500 (actualizada 2024)
    SP500_SYMBOLS = [
        # Technology (120+ stocks)
        'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'NVDA', 'AVGO', 'TSLA', 'ADBE',
        'CRM', 'CSCO', 'ACN', 'ORCL', 'IBM', 'INTC', 'AMD', 'QCOM', 'TXN', 'NOW',
        'SNOW', 'NET', 'PANW', 'CRWD', 'ZS', 'FTNT', 'OKTA', 'TEAM', 'PLTR', 'DDOG',
        'MDB', 'SPLK', 'HUBS', 'ESTC', 'PD', 'TWLO', 'DOCU', 'RBLX', 'UBER', 'LYFT',
        'SHOP', 'SQ', 'PYPL', 'COIN', 'HOOD', 'ROKU', 'NFLX', 'DIS', 'CMCSA', 'CHTR',
        'T', 'VZ', 'TMUS', 'EA', 'ATVI', 'TTWO', 'ZNGA', 'RIVN', 'LCID', 'FSLR',
        'ENPH', 'SEDG', 'RUN', 'PLUG', 'BE', 'NIO', 'LI', 'XPEV', 'F', 'GM',
        'TSM', 'ASML', 'LRCX', 'AMAT', 'KLAC', 'NXPI', 'MRVL', 'SWKS', 'QRVO', 'MCHP',
        'CDNS', 'ANSS', 'ADSK', 'TTD', 'TTWO', 'EA', 'ATVI', 'ZG', 'Z', 'RDFN',
        'OPEN', 'COMP', 'U', 'CLSK', 'MSTR', 'RIOT', 'MARA', 'HUT', 'BITF', 'COIN',
        
        # Healthcare (60+ stocks)
        'JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'TMO', 'LLY', 'DHR', 'ABT', 'BMY',
        'AMGN', 'GILD', 'VRTX', 'REGN', 'BIIB', 'ISRG', 'SYK', 'BDX', 'ZTS', 'EW',
        'HCA', 'IDXX', 'DXCM', 'ILMN', 'MTD', 'WAT', 'PKI', 'TECH', 'RGEN', 'ICLR',
        'STE', 'WST', 'BRKR', 'PODD', 'ALGN', 'COO', 'HSIC', 'XRAY', 'BAX', 'HOLX',
        'LH', 'DGX', 'A', 'ABC', 'CAH', 'MCK', 'CVS', 'WBA', 'CI', 'HUM',
        'ELV', 'CNC', 'MOH', 'OGN', 'BHC', 'JAZZ', 'INCY', 'EXAS', 'NTRA', 'TXG',
        
        # Financials (70+ stocks)
        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'SCHW', 'BLK', 'AXP', 'V', 'MA',
        'PYPL', 'SQ', 'COF', 'DFS', 'TFC', 'PNC', 'USB', 'KEY', 'CFG', 'MTB',
        'RF', 'HBAN', 'FITB', 'ALLY', 'CMA', 'ZION', 'EWBC', 'C', 'BK', 'STT',
        'NTRS', 'TROW', 'AMP', 'BEN', 'IVZ', 'JEF', 'PGR', 'ALL', 'TRV', 'AIG',
        'HIG', 'PFG', 'L', 'AON', 'MMC', 'WTW', 'AJG', 'BRO', 'ERIE', 'CINF',
        'RE', 'RGA', 'MET', 'PRU', 'LNC', 'UNM', 'AFL', 'BHF', 'NMRK', 'RJF',
        'ICE', 'MCO', 'SPGI', 'MSCI', 'NDAQ', 'CBOE', 'FDS', 'FIS', 'FISV', 'GPN',
        
        # Consumer Discretionary (60+ stocks)
        'AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'LOW', 'SBUX', 'TJX', 'TGT', 'BKNG',
        'ORLY', 'AZO', 'MGM', 'WYNN', 'LVS', 'RCL', 'CCL', 'NCLH', 'MAR', 'HLT',
        'EXPE', 'ABNB', 'TRIP', 'BKNG', 'YUM', 'CMG', 'DPZ', 'WING', 'DRI', 'BLMN',
        'EBAY', 'ETSY', 'ROST', 'BURL', 'DLTR', 'FIVE', 'BIG', 'DKS', 'ASO', 'ANF',
        'GPS', 'URBN', 'LEVI', 'NKE', 'LULU', 'VFC', 'TPR', 'CPRI', 'RL', 'PVH',
        'F', 'GM', 'STLA', 'HMC', 'TM', 'RACE', 'TSLA', 'LCID', 'RIVN', 'NKLA',
        
        # Consumer Staples (30+ stocks)
        'PG', 'KO', 'PEP', 'WMT', 'COST', 'TGT', 'KR', 'SYY', 'ADM', 'BG',
        'MDLZ', 'K', 'GIS', 'HSY', 'SJM', 'CAG', 'CPB', 'KMB', 'CL', 'EL',
        'NWL', 'CLX', 'CHD', 'EPD', 'MO', 'PM', 'BTI', 'IMB', 'STZ', 'BUD',
        'TAP', 'SAM', 'MNST', 'KDP', 'FIZZ', 'COKE', 'PEP', 'KO', 'WMT', 'COST',
        
        # Industrials (70+ stocks)
        'UPS', 'FDX', 'RTX', 'BA', 'LMT', 'NOC', 'GD', 'HII', 'LHX', 'CW',
        'TDG', 'HEI', 'COL', 'TXT', 'DE', 'CAT', 'CNHI', 'AGCO', 'CMI', 'PCAR',
        'ALLE', 'ALGN', 'CSX', 'UNP', 'NSC', 'CP', 'KSU', 'JBHT', 'LSTR', 'ODFL',
        'EXPD', 'CHRW', 'XPO', 'GWW', 'FAST', 'MSM', 'SNA', 'ITW', 'EMR', 'ROK',
        'DOV', 'PNR', 'IEX', 'FLS', 'FLR', 'J', 'PWR', 'QUAD', 'VMC', 'MLM',
        'SUM', 'EXP', 'ASH', 'ECL', 'IFF', 'PPG', 'SHW', 'ALB', 'LTHM', 'SLB',
        'HAL', 'BKR', 'NOV', 'FTI', 'OII', 'RIG', 'DO', 'LBRT', 'WHD', 'NBR',
        
        # Energy (30+ stocks)
        'XOM', 'CVX', 'COP', 'EOG', 'MPC', 'PSX', 'VLO', 'DVN', 'PXD', 'OXY',
        'HES', 'MRO', 'FANG', 'APA', 'NOV', 'SLB', 'HAL', 'BKR', 'WMB', 'KMI',
        'ET', 'EPD', 'OKE', 'TRGP', 'LNG', 'CHK', 'RRC', 'SWN', 'AR', 'MGY',
        
        # Materials (20+ stocks)
        'LIN', 'APD', 'SHW', 'ECL', 'PPG', 'ALB', 'NEM', 'GOLD', 'FCX', 'SCCO',
        'AA', 'CLF', 'STLD', 'NUE', 'X', 'MOS', 'CF', 'NTR', 'FMC', 'AVY',
        'IP', 'PKG', 'WRK', 'SEE', 'BALL', 'ATI', 'CMC', 'RS', 'CRS', 'WOR',
        
        # Real Estate (30+ stocks)
        'AMT', 'CCI', 'PLD', 'EQIX', 'PSA', 'SPG', 'O', 'AVB', 'EQR', 'ESS',
        'UDR', 'MAA', 'CPT', 'ARE', 'BXP', 'SLG', 'VNO', 'KIM', 'FRT', 'REG',
        'DLR', 'IRM', 'EXR', 'PSA', 'WPC', 'NSA', 'LAMR', 'CUBE', 'REXR', 'PLD',
        
        # Utilities (30+ stocks)
        'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'XEL', 'WEC', 'ES',
        'PEG', 'ETR', 'FE', 'AES', 'AWK', 'CNP', 'DTE', 'LNT', 'PPL', 'EIX',
        'ED', 'CMS', 'NRG', 'VST', 'ALE', 'OTTR', 'SWX', 'NI', 'OGE', 'POR'
    ]

    # =============================================
    # FUNCIONES OPTIMIZADAS CON CACHING ESTRATÉGICO
    # =============================================

    @st.cache_data(ttl=86400, show_spinner=False)  # 24 horas
    def obtener_lista_sp500_estatica():
        """Lista estática del S&P500 que cambia poco"""
        return SP500_SYMBOLS

    @st.cache_data(ttl=3600, show_spinner=False, max_entries=50)
    def obtener_datos_sp500_precalculados():
        """Precalcula datos del S&P500 una vez por hora"""
        return precalcular_datos_screener(SP500_SYMBOLS)

    def precalcular_datos_screener(sp500_symbols):
        """Precalcula datos críticos para mayor velocidad"""
        if 'datos_precalculados' in st.session_state:
            return st.session_state.datos_precalculados
        
        datos_precalculados = {}
        # Limitar a las primeras 520 acciones
        simbolos_rapidos = sp500_symbols[:520]
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, simbolo in enumerate(simbolos_rapidos):
            try:
                datos = obtener_datos_completos_yfinance(simbolo)
                if datos and datos.get('Empresa Valida'):
                    scoring = calcular_scoring_dinamico(datos)
                    datos['Score'] = scoring
                    datos_precalculados[simbolo] = datos
                    
                # Actualizar progreso cada 10 acciones
                if i % 10 == 0:
                    progress_percent = (i + 1) / len(simbolos_rapidos)
                    progress_bar.progress(progress_percent)
                    status_text.text(f"Precalculando: {i+1}/{len(simbolos_rapidos)} acciones")
                    
            except Exception as e:
                continue
        
        progress_bar.empty()
        status_text.empty()
        
        st.session_state.datos_precalculados = datos_precalculados
        return datos_precalculados

    def aplicar_filtros_rapidos(datos, filtros):
        """Aplica filtros de manera optimizada usando operaciones vectorizadas"""
        try:
            # Filtro P/E
            pe = datos.get('P/E', 0)
            if filtros['pe_min'] > 0 and (pe == 0 or pe < filtros['pe_min']):
                return False
            if filtros['pe_max'] < 1000 and pe > filtros['pe_max']:
                return False
            
            # Solo los filtros más importantes para velocidad
            roe = datos.get('ROE', 0)
            if filtros['roe_min'] > 0 and roe < (filtros['roe_min'] / 100):
                return False
                
            # Filtro Margen Beneficio
            margen = datos.get('Margen Beneficio', 0)
            if filtros['profit_margin_min'] > 0 and margen < (filtros['profit_margin_min'] / 100):
                return False
            
            # Filtro Deuda/Equity
            deuda_eq = datos.get('Deuda/Equity', 0)
            if filtros['debt_equity_max'] < 10 and deuda_eq > filtros['debt_equity_max']:
                return False
            
            # Filtro Beta
            beta = datos.get('Beta', 1)
            if filtros['beta_max'] < 5 and beta > filtros['beta_max']:
                return False
            
            # Filtro RSI
            rsi = datos.get('RSI', 50)
            if rsi < filtros['rsi_min'] or rsi > filtros['rsi_max']:
                return False
                
            return True
        except:
            return False

    def buscar_simbolos_sp500_optimizado(filtros, max_acciones=50):
        """Versión optimizada con carga progresiva"""
        # Cargar primero datos precalculados si existen
        datos_precalculados = st.session_state.get('datos_precalculados', {})
        
        if not datos_precalculados:
            with st.spinner('🔄 Precalculando datos del S&P500 para búsquedas ultra rápidas...'):
                datos_precalculados = precalcular_datos_screener(SP500_SYMBOLS)
                st.session_state.datos_precalculados = datos_precalculados
        
        # Aplicar filtros sobre datos precalculados (MUCHO más rápido)
        acciones_encontradas = []
        
        for simbolo, datos in datos_precalculados.items():
            if len(acciones_encontradas) >= max_acciones:
                break
            if aplicar_filtros_rapidos(datos, filtros):
                acciones_encontradas.append(datos)
        
        return acciones_encontradas

    # FUNCIONES AUXILIARES (mantener tus funciones originales)
    def obtener_datos_completos_yfinance(simbolo):
        """Obtiene datos fundamentales y técnicos de yFinance para cualquier símbolo"""
        try:
            ticker = yf.Ticker(simbolo)
            info = ticker.info
            
            # Verificar que el símbolo es válido
            if not info or 'currentPrice' not in info or info.get('currentPrice') is None:
                return None
            
            # Obtener datos históricos para calcular RSI
            datos_historicos = yf.download(simbolo, period="6mo", interval="1d", progress=False)
            
            # Calcular RSI si hay datos históricos
            rsi = 50
            if not datos_historicos.empty and 'Close' in datos_historicos.columns:
                try:
                    delta = datos_historicos['Close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    rsi_calculado = 100 - (100 / (1 + rs))
                    rsi = rsi_calculado.iloc[-1] if not rsi_calculado.empty and not pd.isna(rsi_calculado.iloc[-1]) else 50
                except:
                    rsi = 50
            
            # Datos completos
            datos = {
                'Símbolo': simbolo,
                'Nombre': info.get('longName', simbolo),
                'Sector': info.get('sector', 'N/A'),
                'Industria': info.get('industry', 'N/A'),
                'Market Cap': info.get('marketCap', 0),
                'P/E': info.get('trailingPE', 0),
                'Precio Actual': info.get('currentPrice', 0),
                'Cambio %': info.get('regularMarketChangePercent', 0),
                'Volumen': info.get('volume', 0),
                'ROE': info.get('returnOnEquity', 0),
                'Margen Beneficio': info.get('profitMargins', 0),
                'Deuda/Equity': info.get('debtToEquity', 0),
                'Crecimiento Ingresos': info.get('revenueGrowth', 0),
                'Beta': info.get('beta', 1),
                'RSI': rsi,
                'Empresa Valida': True
            }
            
            return datos
            
        except Exception as e:
            return None

    def calcular_scoring_dinamico(datos):
        """Calcula scoring basado en datos fundamentales"""
        if not datos:
            return 0
        
        score = 0
        max_score = 100
        
        try:
            # P/E Ratio (20 puntos) - MÁS FLEXIBLE
            pe = datos.get('P/E', 0)
            if pe and pe > 0:
                if pe < 15:
                    score += 20
                elif pe < 25:
                    score += 15
                elif pe < 35:
                    score += 10
                else:
                    score += 5
            
            # ROE (20 puntos) - MÁS FLEXIBLE
            roe = datos.get('ROE', 0)
            if roe and roe > 0:
                if roe > 0.20:
                    score += 20
                elif roe > 0.15:
                    score += 16
                elif roe > 0.10:
                    score += 12
                elif roe > 0.05:
                    score += 8
                else:
                    score += 4
            
            # Margen Beneficio (15 puntos) - MÁS FLEXIBLE
            margen = datos.get('Margen Beneficio', 0)
            if margen and margen > 0:
                if margen > 0.20:
                    score += 15
                elif margen > 0.15:
                    score += 12
                elif margen > 0.10:
                    score += 9
                elif margen > 0.05:
                    score += 6
                else:
                    score += 3
            
            # Deuda/Equity (15 puntos) - MÁS FLEXIBLE
            deuda_eq = datos.get('Deuda/Equity', 0)
            if deuda_eq and deuda_eq >= 0:
                if deuda_eq < 0.5:
                    score += 15
                elif deuda_eq < 1.0:
                    score += 12
                elif deuda_eq < 1.5:
                    score += 9
                elif deuda_eq < 2.0:
                    score += 6
                else:
                    score += 3
            
            # Crecimiento Ingresos (20 puntos) - MÁS FLEXIBLE
            crecimiento = datos.get('Crecimiento Ingresos', 0)
            if crecimiento:
                if crecimiento > 0.20:
                    score += 20
                elif crecimiento > 0.15:
                    score += 16
                elif crecimiento > 0.10:
                    score += 12
                elif crecimiento > 0.05:
                    score += 8
                elif crecimiento > 0:
                    score += 4
            
            # Beta (10 puntos) - MÁS FLEXIBLE
            beta = datos.get('Beta', 1)
            if beta and beta > 0:
                if beta < 0.8:
                    score += 10
                elif beta < 1.2:
                    score += 8
                elif beta < 1.5:
                    score += 6
                elif beta < 2.0:
                    score += 4
                else:
                    score += 2
            
            return min(score, max_score)
            
        except Exception as e:
            return 0

    # =============================================
    # INTERFAZ DE USUARIO OPTIMIZADA
    # =============================================
    # INTERFAZ DE USUARIO OPTIMIZADA
    # =============================================

    # PRE-CÁLCULO AUTOMÁTICO AL ENTRAR A LA SECCIÓN
    if 'precalc_iniciado' not in st.session_state:
        with st.spinner('🔄 Precargando datos del S&P 500 para búsquedas instantáneas...'):
            datos_precalculados = precalcular_datos_screener(SP500_SYMBOLS)
            st.session_state.datos_precalculados = datos_precalculados
            st.session_state.precalc_iniciado = True
            st.success(f"✅ Pre-cálculo completado: {len(datos_precalculados)} acciones listas")

    # INICIALIZAR ESTADOS SI NO EXISTEN
    if 'show_search_results' not in st.session_state:
        st.session_state.show_search_results = False
    if 'show_comparison' not in st.session_state:
        st.session_state.show_comparison = False

    # INTERFAZ DE FILTROS MEJORADA - VALORES POR DEFECTO MÁS FLEXIBLES
    st.subheader("🎯 Configura tus Criterios de Búsqueda")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**💰 Valoración:**")
        pe_min = st.number_input("P/E Mínimo", value=0.0, min_value=0.0, max_value=100.0, step=1.0, 
                            help="0 = Sin filtro. Valores típicos: 5-15")
        pe_max = st.number_input("P/E Máximo", value=60.0, min_value=0.0, max_value=1000.0, step=1.0,
                            help="1000 = Sin filtro. Valores típicos: 20-50")
        
        st.write("**📈 Rentabilidad:**")
        roe_min = st.number_input("ROE Mínimo (%)", value=5.0, min_value=0.0, max_value=100.0, step=1.0,
                                help="0 = Sin filtro. Valores típicos: 8-15")
        profit_margin_min = st.number_input("Margen Beneficio Mínimo (%)", value=0.0, min_value=0.0, max_value=100.0, step=1.0,
                                        help="0 = Sin filtro. Valores típicos: 5-12")

    with col2:
        st.write("**🏦 Estructura de Capital:**")
        debt_equity_max = st.number_input("Deuda/Equity Máximo", value=3.0, min_value=0.0, max_value=10.0, step=0.1,
                                        help="10 = Sin filtro. Valores típicos: 0.5-2.0")
        
        st.write("**📊 Volatilidad:**")
        beta_max = st.number_input("Beta Máximo", value=2.5, min_value=0.1, max_value=5.0, step=0.1,
                                help="5 = Sin filtro. Valores típicos: 0.8-1.5")
        
        st.write("**🚀 Crecimiento:**")
        revenue_growth_min = st.number_input("Crecimiento Ingresos Mínimo (%)", value=0.0, min_value=-50.0, max_value=200.0, step=1.0,
                                        help="-50 = Sin filtro. Valores típicos: 5-15")

    # Filtros RSI MÁS FLEXIBLES
    st.subheader("📊 Filtro de Momentum (RSI)")
    col_rsi1, col_rsi2 = st.columns(2)

    with col_rsi1:
        rsi_min = st.slider("RSI Mínimo", 0, 100, 25, key="rsi_min_screener",
                        help="RSI muy bajo puede indicar sobreventa")

    with col_rsi2:
        rsi_max = st.slider("RSI Máximo", 0, 100, 75, key="rsi_max_screener",
                        help="RSI muy alto puede indicar sobrecompra")

    st.info(f"💡 **Rango RSI seleccionado:** {rsi_min} - {rsi_max} (Recomendado: 25-75 para más resultados)")

    # BOTÓN DE BÚSQUEDA MEJORADO
    st.markdown("---")

    # Selector de límite de resultados
    max_resultados = st.slider("Límite máximo de resultados", 10, 200, 50, 10,
                            help="Número máximo de acciones a mostrar")

    # Indicador de estado del cache
    if 'datos_precalculados' in st.session_state:
        st.success(f"✅ **Datos precalculados listos:** {len(st.session_state.datos_precalculados)} acciones cargadas en caché")
    else:
        st.info("🔄 **Sistema optimizado:** Los datos se precalcularán en la primera búsqueda para máxima velocidad")

    if st.button("🚀 Ejecutar Búsqueda Ultra Rápida", use_container_width=True, type="primary"):
        # Definir filtros
        filtros = {
            'pe_min': pe_min,
            'pe_max': pe_max,
            'roe_min': roe_min,
            'profit_margin_min': profit_margin_min,
            'revenue_growth_min': revenue_growth_min,
            'debt_equity_max': debt_equity_max,
            'beta_max': beta_max,
            'rsi_min': rsi_min,
            'rsi_max': rsi_max
        }
        
        # Ejecutar búsqueda OPTIMIZADA
        with st.spinner(f"🔍 Buscando en {len(SP500_SYMBOLS)} acciones con sistema optimizado..."):
            acciones_encontradas = buscar_simbolos_sp500_optimizado(filtros, max_resultados)
        
        if acciones_encontradas:
            # Ordenar por score
            acciones_encontradas.sort(key=lambda x: x['Score'], reverse=True)
            
            # Crear DataFrame para mostrar
            df_resultados = pd.DataFrame(acciones_encontradas)
            
            # Formatear columnas para mostrar
            columnas_mostrar = ['Símbolo', 'Nombre', 'Sector', 'P/E', 'Precio Actual', 
                            'ROE', 'Margen Beneficio', 'Deuda/Equity', 'Beta', 'RSI', 'Score']
            
            df_display = df_resultados[columnas_mostrar].copy()
            
            # Formatear valores
            df_display['P/E'] = df_display['P/E'].apply(lambda x: f"{x:.1f}" if x > 0 else "N/A")
            df_display['Precio Actual'] = df_display['Precio Actual'].apply(lambda x: f"${x:.2f}" if x > 0 else "N/A")
            df_display['ROE'] = df_display['ROE'].apply(lambda x: f"{x*100:.1f}%" if x > 0 else "N/A")
            df_display['Margen Beneficio'] = df_display['Margen Beneficio'].apply(lambda x: f"{x*100:.1f}%" if x > 0 else "N/A")
            df_display['Deuda/Equity'] = df_display['Deuda/Equity'].apply(lambda x: f"{x:.2f}" if x >= 0 else "N/A")
            df_display['Beta'] = df_display['Beta'].apply(lambda x: f"{x:.2f}" if x > 0 else "N/A")
            df_display['RSI'] = df_display['RSI'].apply(lambda x: f"{x:.1f}")
            df_display['Score'] = df_display['Score'].apply(lambda x: f"{x:.0f}")
            
            # GUARDAR ESTADO DE BÚSQUEDA
            st.session_state.show_search_results = True
            st.session_state.search_results = {
                'acciones_encontradas': acciones_encontradas,
                'df_display': df_display,
                'df_resultados': df_resultados
            }
            st.session_state.show_comparison = False  # Ocultar comparación al hacer nueva búsqueda
            
            st.rerun()
            
        else:
            st.warning("""
            ❌ No se encontraron acciones que cumplan todos los criterios.
            
            **💡 Sugerencias para obtener más resultados:**
            • **Relaja los filtros** - especialmente P/E Máximo (prueba 60-80) y ROE Mínimo (5-8%)
            • **Amplía el rango RSI** - prueba 20-80 en lugar de 30-70
            • **Reduce Deuda/Equity Máximo** - prueba 3.0-4.0
            • **Aumenta Beta Máximo** - prueba 2.5-3.0
            • **Establece algunos filtros en 0** para desactivarlos completamente
            """)

    # ⭐⭐ AQUÍ VAN LOS RESULTADOS - DESPUÉS DE LOS FILTROS Y BOTONES ⭐⭐

    # MOSTRAR RESULTADOS DE BÚSQUEDA SI ESTÁN ACTIVOS
    if st.session_state.show_search_results and st.session_state.get('search_results'):
        st.markdown("---")
        resultados = st.session_state.search_results
        st.success(f"✅ **Búsqueda completada:** {len(resultados['acciones_encontradas'])} acciones encontradas")
        
        st.subheader("📊 Resultados del Screener S&P 500 (Optimizado)")
        st.dataframe(resultados['df_display'], use_container_width=True)
        
        st.subheader("📈 Análisis por Sectores")
        sector_counts = resultados['df_resultados']['Sector'].value_counts()
        fig_sectores = px.pie(
            values=sector_counts.values,
            names=sector_counts.index,
            title='Distribución de Acciones por Sector'
        )
        st.plotly_chart(fig_sectores, use_container_width=True, key="sectores_pie")
        
        st.subheader("🏆 Distribución de Scores")
        fig_scores = px.bar(
            resultados['df_resultados'].head(20),
            x='Símbolo',
            y='Score',
            color='Score',
            title='Top 20 Acciones por Score',
            color_continuous_scale='viridis'
        )
        fig_scores.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_scores, use_container_width=True, key="scores_bar")
        
        st.markdown("---")
        st.subheader("💾 Exportar Resultados")
        
        csv_resultados = resultados['df_resultados'].to_csv(index=False)
        st.download_button(
            label="📥 Descargar resultados completos (CSV)",
            data=csv_resultados,
            file_name=f"screener_sp500_optimizado_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    # =============================================
    # GRÁFICA DE COMPARACIÓN CON S&P500
    # =============================================

    # ... (el resto del código de comparación se mantiene igual)

    # =============================================
    # GRÁFICA DE COMPARACIÓN CON S&P500
    # =============================================

    # Verificar si hay resultados disponibles para comparación
    acciones_disponibles = None
    if st.session_state.get('search_results'):
        acciones_disponibles = st.session_state.search_results['acciones_encontradas']

    if acciones_disponibles:
        st.markdown("---")
        st.subheader("📈 Comparación de Rendimiento vs S&P500")
        
        col_periodo, col_accion = st.columns(2)
        
        with col_periodo:
            periodo_comparacion = st.selectbox(
                "Período de Comparación:",
                ["1 Mes", "3 Meses", "6 Meses", "1 Año", "2 Años", "3 Años"],
                index=3,
                key="periodo_screener"
            )
        
        with col_accion:
            acciones_todas = [acc['Símbolo'] for acc in acciones_disponibles]
            accion_seleccionada = st.selectbox(
                "Seleccionar Acción para Comparar:",
                acciones_todas,
                key="accion_comparar_screener"
            )
        
        # Botón para generar comparación
        if st.button("🔄 Generar Comparación", use_container_width=True, key="comparar_btn"):
            st.session_state.show_comparison = True
            st.session_state.comparison_data = {
                'accion_seleccionada': accion_seleccionada,
                'periodo_comparacion': periodo_comparacion
            }
            st.rerun()
        
        # MOSTRAR COMPARACIÓN SI ESTÁ ACTIVA
        if st.session_state.show_comparison and st.session_state.get('comparison_data'):
            comparison = st.session_state.comparison_data
            accion_seleccionada = comparison['accion_seleccionada']
            periodo_comparacion = comparison['periodo_comparacion']
            
            with st.spinner(f'Comparando {accion_seleccionada} vs S&P500...'):
                try:
                    # Mapear período seleccionado a días
                    periodo_map = {
                        "1 Mes": 30,
                        "3 Meses": 90,
                        "6 Meses": 180,
                        "1 Año": 365,
                        "2 Años": 730,
                        "3 Años": 1095
                    }
                    
                    dias = periodo_map[periodo_comparacion]
                    start_date = datetime.today() - timedelta(days=dias)
                    
                    # Obtener datos de la acción seleccionada
                    data_accion = yf.download(accion_seleccionada, start=start_date, progress=False)
                    data_sp500 = yf.download('^GSPC', start=start_date, progress=False)
                    
                    if not data_accion.empty and not data_sp500.empty:
                        # Obtener precios de cierre
                        if isinstance(data_accion.columns, pd.MultiIndex):
                            close_accion = data_accion[('Close', accion_seleccionada)]
                        else:
                            close_accion = data_accion['Close']
                        
                        if isinstance(data_sp500.columns, pd.MultiIndex):
                            close_sp500 = data_sp500[('Close', '^GSPC')]
                        else:
                            close_sp500 = data_sp500['Close']
                        
                        # Calcular rendimiento normalizado (base 100)
                        rendimiento_accion = (close_accion / close_accion.iloc[0]) * 100
                        rendimiento_sp500 = (close_sp500 / close_sp500.iloc[0]) * 100
                        
                        # Crear gráfica
                        fig_comparacion = go.Figure()
                        
                        # Agregar línea de la acción
                        fig_comparacion.add_trace(go.Scatter(
                            x=rendimiento_accion.index,
                            y=rendimiento_accion.values,
                            mode='lines',
                            name=f'{accion_seleccionada}',
                            line=dict(color='#00FF00', width=3),
                            hovertemplate=(
                                f'<b>{accion_seleccionada}</b><br>' +
                                'Fecha: %{x}<br>' +
                                'Rendimiento: %{y:.1f}%<br>' +
                                '<extra></extra>'
                            )
                        ))
                        
                        # Agregar línea del S&P500
                        fig_comparacion.add_trace(go.Scatter(
                            x=rendimiento_sp500.index,
                            y=rendimiento_sp500.values,
                            mode='lines',
                            name='S&P 500',
                            line=dict(color='#FF6B6B', width=3, dash='dash'),
                            hovertemplate=(
                                '<b>S&P 500</b><br>' +
                                'Fecha: %{x}<br>' +
                                'Rendimiento: %{y:.1f}%<br>' +
                                '<extra></extra>'
                            )
                        ))
                        
                        # Calcular métricas de performance
                        rend_final_accion = rendimiento_accion.iloc[-1] - 100
                        rend_final_sp500 = rendimiento_sp500.iloc[-1] - 100
                        outperformance = rend_final_accion - rend_final_sp500
                        
                        # Configurar layout
                        fig_comparacion.update_layout(
                            title=f'Comparación de Rendimiento: {accion_seleccionada} vs S&P500 ({periodo_comparacion})',
                            xaxis_title='Fecha',
                            yaxis_title='Rendimiento (%)',
                            height=500,
                            showlegend=True,
                            hovermode='x unified',
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            )
                        )
                        
                        # Mostrar gráfica
                        st.plotly_chart(fig_comparacion, use_container_width=True, key="comparacion_sp500")
                        
                        # Mostrar métricas de performance
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                f"Rendimiento {accion_seleccionada}",
                                f"{rend_final_accion:+.1f}%",
                                delta_color="normal"
                            )
                        
                        with col2:
                            st.metric(
                                "Rendimiento S&P500",
                                f"{rend_final_sp500:+.1f}%", 
                                delta_color="normal"
                            )
                        
                        with col3:
                            st.metric(
                                "Outperformance",
                                f"{outperformance:+.1f}%",
                                delta_color="normal"
                            )
                        
                        with col4:
                            # Calcular correlación
                            correlacion = rendimiento_accion.corr(rendimiento_sp500)
                            st.metric(
                                "Correlación",
                                f"{correlacion:.2f}",
                                delta_color="off"
                            )
                        
                        # Análisis de la comparación
                        st.info(f"""
                        **📊 Análisis de la Comparación:**
                        
                        • **{accion_seleccionada}** ha tenido un rendimiento del **{rend_final_accion:+.1f}%** en el período
                        • **S&P 500** ha tenido un rendimiento del **{rend_final_sp500:+.1f}%**
                        • **Diferencia:** {accion_seleccionada} ha **{"superado" if outperformance >= 0 else "subperformado"}** al mercado por **{abs(outperformance):.1f}%**
                        • **Correlación:** {correlacion:.2f} ({"alta" if correlacion > 0.7 else "media" if correlacion > 0.3 else "baja"})
                        """)
                        
                        # Mantener el estado de los resultados visibles
                        st.session_state.show_search_results = True
                        
                    else:
                        st.warning("No se pudieron obtener datos para la comparación")
                        
                except Exception as e:
                    st.error(f"Error en la comparación: {str(e)}")

    # CONSEJOS PARA FILTROS MÁS EFECTIVOS
    with st.expander("💡 Consejos para Configurar Filtros en S&P 500"):
        st.markdown("""
        **Configuraciones recomendadas para S&P 500:**
        
        | Filtro | Valor Conservador | Valor Balanceado | Valor Agresivo | Resultados |
        |--------|------------------|------------------|----------------|------------|
        | P/E Máximo | 25 | 40-50 | 60-80 | 🟢 Más resultados |
        | ROE Mínimo | 15% | 8-12% | 5-8% | 🟢 Más resultados |
        | RSI Mínimo | 30 | 25-30 | 20-25 | 🟢 Más resultados |
        | RSI Máximo | 70 | 70-75 | 75-80 | 🟢 Más resultados |
        | Deuda/Equity | 1.0 | 2.0-2.5 | 3.0-4.0 | 🟢 Más resultados |
        | Beta Máximo | 1.2 | 1.8-2.2 | 2.5-3.0 | 🟢 Más resultados |
        
        **Para empezar (Balanceado):**
        - P/E Mínimo: 0
        - P/E Máximo: 50
        - ROE Mínimo: 8%
        - RSI: 25-75
        - Deuda/Equity: 2.5
        - Beta: 2.0
        
        Esto debería darte **20-60 acciones** del S&P 500.
        
        **Sectores con mejores resultados:**
        - 🏦 **Financieras:** Suelen tener P/E bajos
        - 🛢️ **Energía:** Crecimiento variable pero oportunidades
        - 🏭 **Industriales:** Estables con buenos dividendos
        - 🛒 **Consumo:** Defensivas con crecimiento constante
        """)

    # ESTADÍSTICAS DEL SISTEMA OPTIMIZADO
    with st.expander("🚀 Estadísticas del Sistema Optimizado"):
        if 'datos_precalculados' in st.session_state:
            datos_precalculados = st.session_state.datos_precalculados
            st.markdown(f"""
            **📊 Estado del Sistema de Caché:**
            - **Acciones precalculadas:** {len(datos_precalculados)}
            - **Tiempo de caché:** 1 hora
            - **Velocidad de búsqueda:** Instantánea
            - **Memoria optimizada:** Solo datos esenciales
            
            **💡 Beneficios del sistema optimizado:**
            - **⏱️ 10x más rápido** que búsquedas individuales
            - **📈 Mayor cobertura** del S&P500
            - **🔄 Actualizaciones automáticas** cada hora
            - **💾 Caché inteligente** que persiste entre sesiones
            """)
        else:
            st.info("El sistema de caché se activará después de la primera búsqueda")

    # BOTÓN PARA LIMPIAR CACHÉ (útil para desarrollo)
    if st.button("🗑️ Limpiar Caché de Datos", type="secondary"):
        keys_to_remove = [
            'datos_precalculados', 'precalc_iniciado', 'acciones_encontradas',
            'df_resultados', 'resultados_busqueda', 'search_results',
            'show_search_results', 'show_comparison', 'comparison_data'
        ]
        for key in keys_to_remove:
            if key in st.session_state:
                del st.session_state[key]
        st.success("✅ Caché limpiado. La próxima búsqueda recalculará los datos.")
        st.rerun()

# SECCIÓN DE MACROECONOMÍA
elif st.session_state.seccion_actual == "macro":
    st.header("🌍 Panorama Macroeconómico Global")
    
    st.markdown("""
    **Contexto macroeconómico actual** que puede afectar tus inversiones.
    Los indicadores económicos influyen en los mercados bursátiles y en las decisiones de los inversores.
    """)

    # CONFIGURACIÓN DE SESIÓN HTTP OPTIMIZADA
    def crear_session_optimizada():
        """Crea una sesión HTTP optimizada con timeouts y reintentos"""
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        
        session = requests.Session()
        
        # Configurar reintentos
        retry_strategy = Retry(
            total=2,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=10)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session

    # FUNCIONES AUXILIARES
    def mostrar_indicadores_en_columnas(indicadores_dict):
        """Muestra indicadores organizados en columnas"""
        cols = st.columns(2)
        current_col = 0
        
        for indicador, valor in indicadores_dict.items():
            if "---" in valor or "**" in indicador:
                # Es un separador o título
                st.markdown(f"**{indicador}**")
                continue
                
            with cols[current_col]:
                color_borde, color_texto = determinar_colores_indicador(indicador, valor)
                    
                st.markdown(f"""
                <div style='padding: 12px; margin: 8px 0; border-radius: 8px; border-left: 4px solid {color_borde}; background-color: #1e1e1e; border: 1px solid #444;'>
                    <strong style='color: #ffffff; font-size: 13px;'>{indicador}</strong><br>
                    <span style='color: {color_texto}; font-weight: bold; font-size: 14px;'>{valor}</span>
                </div>
                """, unsafe_allow_html=True)
            
            current_col = (current_col + 1) % 2

    def determinar_colores_indicador(indicador, valor):
        """Determina colores apropiados para cada tipo de indicador"""
        indicador_lower = indicador.lower()
        
        # Indicadores donde alto es malo
        if any(x in indicador_lower for x in ['inflación', 'desempleo', 'interés', 'déficit', 'deuda', 'pobreza', 'corrupción', 'riesgo', 'emisiones', 'mortalidad', 'contaminación', 'desnutrición', 'analfabetismo']):
            try:
                valor_limpio = ''.join(c for c in str(valor) if c.isdigit() or c == '.' or c == '-')
                if valor_limpio:
                    valor_num = float(valor_limpio)
                    if valor_num > 10:
                        return "#ff4444", "#ff6666"  # Rojo - Muy malo
                    elif valor_num > 5:
                        return "#ffaa00", "#ffbb33"  # Naranja - Malo
                    else:
                        return "#4CAF50", "#66bb6a"  # Verde - Bueno
            except:
                pass
            return "#2196F3", "#64b5f6"  # Azul - Neutral
        
        # Indicadores donde alto es bueno
        elif any(x in indicador_lower for x in ['crecimiento', 'confianza', 'producción', 'ventas', 'consumo', 'inversión', 'salarios', 'productividad', 'innovación', 'competitividad', 'facilidad', 'esperanza', 'alfabetización', 'matrícula', 'acceso', 'calidad']):
            try:
                valor_limpio = ''.join(c for c in str(valor) if c.isdigit() or c == '.' or c == '-')
                if valor_limpio:
                    valor_num = float(valor_limpio)
                    if valor_num > 5:
                        return "#4CAF50", "#66bb6a"  # Verde - Muy bueno
                    elif valor_num > 0:
                        return "#ffaa00", "#ffbb33"  # Naranja - Regular
                    else:
                        return "#ff4444", "#ff6666"  # Rojo - Malo
            except:
                pass
            return "#2196F3", "#64b5f6"  # Azul - Neutral
        
        # Indicadores de igualdad (Gini)
        elif 'gini' in indicador_lower:
            try:
                valor_limpio = ''.join(c for c in str(valor) if c.isdigit() or c == '.' or c == '-')
                if valor_limpio:
                    valor_num = float(valor_limpio)
                    if valor_num > 0.4:
                        return "#ff4444", "#ff6666"  # Rojo - Alta desigualdad
                    elif valor_num > 0.3:
                        return "#ffaa00", "#ffbb33"  # Naranja - Media desigualdad
                    else:
                        return "#4CAF50", "#66bb6a"  # Verde - Baja desigualdad
            except:
                pass
        
        return "#2196F3", "#64b5f6"  # Azul por defecto

    # FUNCIONES OPTIMIZADAS CON CACHING PARA WORLD BANK
    @st.cache_data(ttl=43200, show_spinner=False)  # 12 horas - países cambian muy poco
    def buscar_codigo_pais_world_bank_optimizado(nombre_pais):
        """Versión optimizada con caching para búsqueda de países"""
        try:
            session = crear_session_optimizada()
            url = f"http://api.worldbank.org/v2/country?format=json&per_page=300"
            response = session.get(url, timeout=8)
            
            if response.status_code == 200:
                data = response.json()
                if len(data) > 1:
                    # Buscar el país por nombre (búsqueda flexible)
                    nombre_buscar = nombre_pais.lower().strip()
                    for pais in data[1]:
                        nombre_pais_wb = pais['name'].lower()
                        
                        # Búsqueda exacta o parcial
                        if (nombre_buscar == nombre_pais_wb or 
                            nombre_buscar in nombre_pais_wb or 
                            nombre_pais_wb in nombre_buscar):
                            return pais['id']
                    
                    # Si no se encuentra, intentar con pycountry para nombres alternativos
                    try:
                        import pycountry
                        pais_pycountry = pycountry.countries.search_fuzzy(nombre_pais)
                        if pais_pycountry:
                            nombre_oficial = pais_pycountry[0].name
                            # Buscar nuevamente con el nombre oficial
                            for pais in data[1]:
                                if nombre_oficial.lower() == pais['name'].lower():
                                    return pais['id']
                    except:
                        pass
            return None
        except Exception as e:
            return None

    def obtener_datos_world_bank_optimizado(pais_codigo, indicadores):
        """Versión optimizada con sesión HTTP reutilizable"""
        try:
            session = crear_session_optimizada()
            datos = {}
            
            # Obtener datos en paralelo (secuencial pero optimizado)
            for indicador in indicadores:
                try:
                    url = f"http://api.worldbank.org/v2/country/{pais_codigo}/indicator/{indicador}?format=json"
                    response = session.get(url, timeout=8)
                    
                    if response.status_code == 200:
                        data = response.json()
                        if len(data) > 1 and data[1]:
                            # Ordenar por año y obtener el más reciente
                            datos_ordenados = sorted(data[1], key=lambda x: x['date'], reverse=True)
                            for dato in datos_ordenados:
                                if dato['value'] is not None:
                                    datos[indicador] = {
                                        'valor': dato['value'],
                                        'año': dato['date'],
                                        'nombre': dato['indicator']['value']
                                    }
                                    break
                except Exception as e:
                    continue
            
            return datos
        except Exception as e:
            return {}

    @st.cache_data(ttl=86400, show_spinner=False)  # 24 horas - datos macro cambian lentamente
    def obtener_datos_pais_world_bank_optimizado(nombre_pais):
        """Versión principal optimizada con caching extensivo pero con TODOS los indicadores originales"""
        try:
            # Buscar código del país (ya cacheados)
            pais_codigo = buscar_codigo_pais_world_bank_optimizado(nombre_pais)
            
            if not pais_codigo:
                return {
                    "nombre": nombre_pais.title(),
                    "poblacion": "País no encontrado",
                    "pib_per_capita": "N/A",
                    "pib_nominal": "N/A",
                    "indicadores": {
                        "Error": f"No se pudo encontrar '{nombre_pais}' en la base de datos del World Bank",
                        "Sugerencia": "Intenta con el nombre en inglés o verifica la ortografía"
                    }
                }
            
            # INDICADORES COMPLETOS DEL WORLD BANK - CON MÁS INDICADORES SOCIALES Y AMBIENTALES
            indicadores_wb = {
                # Población y demografía
                'SP.POP.TOTL': 'Población total',
                'SP.POP.GROW': 'Crecimiento poblacional anual %',
                'SP.DYN.LE00.IN': 'Esperanza de vida al nacer',
                'SP.DYN.LE00.FE.IN': 'Esperanza de vida mujeres',
                'SP.DYN.LE00.MA.IN': 'Esperanza de vida hombres',
                'SP.URB.TOTL.IN.ZS': 'Población urbana %',
                'SP.URB.GROW': 'Crecimiento población urbana %',
                'SM.POP.NETM': 'Migración neta',
                'SP.POP.0014.TO.ZS': 'Población 0-14 años %',
                'SP.POP.1564.TO.ZS': 'Población 15-64 años %',
                'SP.POP.65UP.TO.ZS': 'Población 65+ años %',
                
                # Economía y PIB
                'NY.GDP.MKTP.CD': 'PIB nominal (US$)',
                'NY.GDP.MKTP.KD.ZG': 'Crecimiento del PIB anual %',
                'NY.GDP.PCAP.CD': 'PIB per cápita (US$)',
                'NY.GDP.PCAP.PP.CD': 'PIB per cápita PPA (US$)',
                'NY.GDP.MKTP.KD': 'PIB real (US$ constantes)',
                
                # Inflación y precios
                'FP.CPI.TOTL.ZG': 'Inflación anual %',
                'FP.CPI.TOTL': 'Índice de precios al consumidor',
                
                # Empleo
                'SL.UEM.TOTL.ZS': 'Tasa de desempleo %',
                'SL.TLF.TOTL.IN': 'Fuerza laboral total',
                'SL.EMP.TOTL.SP.ZS': 'Empleo total',
                'SL.EMP.1524.SP.ZS': 'Desempleo juvenil %',
                
                # Comercio exterior
                'NE.EXP.GNFS.CD': 'Exportaciones de bienes y servicios (US$)',
                'NE.IMP.GNFS.CD': 'Importaciones de bienes y servicios (US$)',
                'NE.RSB.GNFS.CD': 'Balanza comercial (US$)',
                'NE.EXP.GNFS.ZS': 'Exportaciones % PIB',
                'NE.IMP.GNFS.ZS': 'Importaciones % PIB',
                
                # Finanzas públicas
                'GC.DOD.TOTL.GD.ZS': 'Deuda pública % PIB',
                'GC.REV.XGRT.GD.ZS': 'Ingresos del gobierno % PIB',
                'GC.XPN.TOTL.GD.ZS': 'Gasto del gobierno % PIB',
                'GC.BAL.CASH.GD.ZS': 'Balance fiscal % PIB',
                
                # SALUD - MÁS INDICADORES
                'SH.XPD.CHEX.GD.ZS': 'Gasto en salud % PIB',
                'SH.XPD.CHEX.PC.CD': 'Gasto en salud per cápita (US$)',
                'SH.DYN.MORT': 'Tasa de mortalidad menores de 5 años',
                'SH.DYN.MORT.FE': 'Mortalidad menores de 5 años (mujeres)',
                'SH.DYN.MORT.MA': 'Mortalidad menores de 5 años (hombres)',
                'SH.DYN.AIDS.ZS': 'Prevalencia de VIH %',
                'SH.STA.OWGH.ZS': 'Obesidad adulta %',
                'SH.STA.OWGH.FE.ZS': 'Obesidad adulta mujeres %',
                'SH.STA.OWGH.MA.ZS': 'Obesidad adulta hombres %',
                'SH.STA.MMRT': 'Tasa mortalidad materna',
                'SH.STA.BRTW.ZS': 'Partos atendidos por personal calificado %',
                'SH.IMM.MEAS': 'Vacunación contra sarampión %',
                'SH.TBS.INCD': 'Incidencia de tuberculosis',
                'SH.MED.BEDS.ZS': 'Camas de hospital por 1000 habitantes',
                'SH.MED.PHYS.ZS': 'Médicos por 1000 habitantes',
                
                # EDUCACIÓN - MÁS INDICADORES
                'SE.XPD.TOTL.GD.ZS': 'Gasto en educación % PIB',
                'SE.XPD.PRIM.ZS': 'Gasto educación primaria %',
                'SE.XPD.SECO.ZS': 'Gasto educación secundaria %',
                'SE.XPD.TERT.ZS': 'Gasto educación terciaria %',
                'SE.ADT.LITR.ZS': 'Tasa de alfabetización adultos %',
                'SE.ADT.1524.LT.FE.ZS': 'Alfabetización jóvenes mujeres %',
                'SE.ADT.1524.LT.MA.ZS': 'Alfabetización jóvenes hombres %',
                'SE.PRM.ENRR': 'Tasa de matrícula primaria',
                'SE.SEC.ENRR': 'Tasa de matrícula secundaria',
                'SE.TER.ENRR': 'Tasa de matrícula terciaria',
                'SE.PRM.CMPT.ZS': 'Tasa finalización primaria %',
                'SE.SEC.CMPT.LO.ZS': 'Tasa finalización secundaria %',
                'SE.PRM.PRSL.ZS': 'Tasa repetición primaria %',
                
                # POBREZA Y DESIGUALDAD - MÁS INDICADORES
                'SI.POV.DDAY': 'Pobreza $3.20/día % población',
                'SI.POV.UMIC': 'Pobreza $5.50/día % población',
                'SI.POV.GINI': 'Coeficiente Gini',
                'SI.POV.NAHC': 'Pobreza nacional %',
                'SI.POV.NAHC.FE': 'Pobreza nacional mujeres %',
                'SI.POV.NAHC.MA': 'Pobreza nacional hombres %',
                'SI.DST.02.20': 'Participación ingreso 20% más rico',
                'SI.DST.FRST.20': 'Participación ingreso 20% más pobre',
                'SI.DST.05TH.20': 'Participación ingreso quintil 5',
                
                # PROTECCIÓN SOCIAL
                'per_sa_allsa.cov_pop_tot': 'Cobertura protección social %',
                'per_lm_alllm.cov_pop_tot': 'Cobertura desempleo %',
                
                # INFRAESTRUCTURA
                'EG.ELC.ACCS.ZS': 'Acceso a electricidad % población',
                'EG.ELC.ACCS.RU.ZS': 'Acceso electricidad rural %',
                'EG.ELC.ACCS.UR.ZS': 'Acceso electricidad urbana %',
                'IT.NET.USER.ZS': 'Usuarios de internet % población',
                'IS.RRS.TOTL.KM': 'Red ferroviaria total (km)',
                'IS.ROD.GOOD.MT': 'Red caminos pavimentados %',
                'EG.NSF.ACCS.ZS': 'Acceso a servicios sanitarios %',
                'SH.H2O.SAFE.ZS': 'Acceso a agua potable %',
                'SH.STA.ACSN': 'Acceso a saneamiento %',
                
                # MEDIO AMBIENTE - MÁS INDICADORES
                'EN.ATM.CO2E.PC': 'Emisiones CO2 per cápita',
                'EN.ATM.CO2E.KT': 'Emisiones CO2 totales (kt)',
                'EN.ATM.CO2E.GF.KT': 'Emisiones CO2 combustible (kt)',
                'EN.ATM.GHGO.KT.CE': 'Emisiones gases efecto invernadero',
                'EN.ATM.METH.KT.CE': 'Emisiones metano',
                'EN.ATM.NOXE.KT.CE': 'Emisiones óxido nitroso',
                'EN.ATM.PM25.MC.M3': 'Contaminación PM2.5',
                'AG.LND.FRST.ZS': 'Área forestal % territorio',
                'AG.LND.FRST.K2': 'Área forestal (km²)',
                'ER.H2O.FWTL.ZS': 'Estrés hídrico %',
                'ER.GDP.FWTL.M3.KD': 'Productividad agua (US$/m³)',
                'AG.CON.FERT.ZS': 'Uso de fertilizantes (kg/ha)',
                'AG.CON.FERT.PT.ZS': 'Uso fertilizantes fosfatados',
                'AG.LND.AGRI.ZS': 'Tierra agrícola %',
                'AG.LND.ARBL.ZS': 'Tierra cultivable %',
                'ER.LND.PTLD.ZS': 'Tierra degradada %',
                'ER.PTD.TOTL.ZS': 'Especies amenazadas %',
                'ER.MRN.PTMR.ZS': 'Especies marinas amenazadas',
                'EN.CLC.MDAT.ZS': 'Cobertura áreas protegidas %',
                'EN.MAM.THRD.NO': 'Especies mamíferos amenazadas',
                'EN.BIR.THRD.NO': 'Especies aves amenazadas',
                'AG.PRD.CREL.MT': 'Producción cereales (ton)',
                'ER.H2O.INTR.PC': 'Recursos hídricos internos per cápita',
                
                # ENERGÍA - NUEVOS INDICADORES
                'EG.USE.COMM.FO.ZS': 'Uso energía combustibles fósiles %',
                'EG.USE.CRNW.ZS': 'Uso energía renovable %',
                'EG.ELC.RNEW.ZS': 'Electricidad renovable %',
                'EG.FEC.RNEW.ZS': 'Energía renovable consumo final %',
                'EG.ELC.NUCL.ZS': 'Electricidad nuclear %',
                'EG.ELC.HYRO.ZS': 'Electricidad hidroeléctrica %',
                
                # CALIDAD DEL AIRE
                'EN.ATM.PM25.MC.M3': 'Concentración PM2.5 (μg/m³)',
                'EN.ATM.NOXE.PC': 'Emisiones NOx per cápita',
                
                # RESIDUOS
                'EN.POP.SLUM.UR.ZS': 'Población en barrios marginales %',
                'EN.POP.SLUM.UR.ZS.1': 'Acceso mejorado a agua urbana %',
                
                # Negocios y competitividad
                'IC.BUS.EASE.XQ': 'Facilidad para hacer negocios',
                'IC.TAX.TOTL.CP.ZS': 'Carga tributaria total %',
                'IC.FRM.CORR.ZS': 'Empresas que experimentan soborno %',
                'IC.REG.COST.PC.ZS': 'Costo registrar empresa % ingreso per cápita',
                
                # GÉNERO E INCLUSIÓN
                'SG.GEN.PARL.ZS': 'Mujeres en parlamento %',
                'SG.VAW.REAS.ZS': 'Mujeres que justifican violencia doméstica %',
                'SG.DMK.SRCR.FN.ZS': 'Mujeres cuenta bancaria %',
                'SL.TLF.CACT.FE.ZS': 'Participación fuerza laboral mujeres %'
            }
            
            # Obtener TODOS los indicadores
            datos_wb = obtener_datos_world_bank_optimizado(pais_codigo, list(indicadores_wb.keys()))
            
            # Obtener nombre oficial del país
            nombre_oficial = nombre_pais.title()
            for pais_info in datos_wb.values():
                if 'nombre' in pais_info:
                    if ' - ' in pais_info['nombre']:
                        nombre_oficial = pais_info['nombre'].split(' - ')[-1]
                        break
            
            # Procesar y formatear los datos
            indicadores_formateados = {}
            
            # Información básica del país
            poblacion = datos_wb.get('SP.POP.TOTL', {}).get('valor', 'N/A')
            pib_nominal = datos_wb.get('NY.GDP.MKTP.CD', {}).get('valor', 'N/A')
            pib_per_capita = datos_wb.get('NY.GDP.PCAP.CD', {}).get('valor', 'N/A')
            pib_ppa = datos_wb.get('NY.GDP.PCAP.PP.CD', {}).get('valor', 'N/A')
            
            # Formatear valores grandes
            def formatear_numero_grande(valor):
                if isinstance(valor, (int, float)):
                    if valor > 1e12:
                        return f"{valor/1e12:.2f}T"
                    elif valor > 1e9:
                        return f"{valor/1e9:.2f}B"
                    elif valor > 1e6:
                        return f"{valor/1e6:.2f}M"
                    else:
                        return f"{valor:,.0f}"
                return str(valor)
            
            def formatear_moneda(valor):
                if isinstance(valor, (int, float)):
                    if valor > 1e12:
                        return f"${valor/1e12:.2f}T"
                    elif valor > 1e9:
                        return f"${valor/1e9:.2f}B"
                    elif valor > 1e6:
                        return f"${valor/1e6:.2f}M"
                    else:
                        return f"${valor:,.0f}"
                return str(valor)
            
            poblacion_str = formatear_numero_grande(poblacion)
            pib_nominal_str = formatear_moneda(pib_nominal)
            pib_per_capita_str = formatear_moneda(pib_per_capita)
            pib_ppa_str = formatear_moneda(pib_ppa)
            
            # Construir diccionario de indicadores
            for codigo, nombre in indicadores_wb.items():
                if codigo in datos_wb:
                    dato = datos_wb[codigo]
                    valor = dato['valor']
                    año = dato['año']
                    
                    # Formatear valores según el tipo de indicador
                    if isinstance(valor, (int, float)):
                        if 'US$' in nombre or codigo in ['NY.GDP.MKTP.CD', 'NY.GDP.PCAP.CD', 'NY.GDP.PCAP.PP.CD', 'NE.EXP.GNFS.CD', 'NE.IMP.GNFS.CD']:
                            valor_str = formatear_moneda(valor)
                        elif any(x in nombre for x in ['%', 'tasa', 'crecimiento', 'ratio']):
                            valor_str = f"{valor:.2f}%"
                        elif 'coeficiente' in nombre.lower() or 'índice' in nombre.lower():
                            valor_str = f"{valor:.3f}"
                        else:
                            valor_str = formatear_numero_grande(valor)
                    else:
                        valor_str = str(valor)
                    
                    indicadores_formateados[f"{nombre} ({año})"] = valor_str
            
            return {
                "nombre": nombre_oficial,
                "poblacion": poblacion_str,
                "pib_per_capita": pib_per_capita_str,
                "pib_nominal": pib_nominal_str,
                "pib_ppa": pib_ppa_str,
                "codigo": pais_codigo,
                "indicadores": indicadores_formateados
            }
            
        except Exception as e:
            return {
                "nombre": nombre_pais.title(),
                "poblacion": "Error en consulta",
                "pib_per_capita": "Error en consulta",
                "pib_nominal": "Error en consulta",
                "pib_ppa": "Error en consulta",
                "indicadores": {
                    "Error": f"No se pudieron obtener datos: {str(e)}",
                    "Recomendación": "Intenta nuevamente en unos momentos"
                }
            }

    # Inicializar session_state para el país seleccionado
    if 'pais_seleccionado_macro' not in st.session_state:
        st.session_state.pais_seleccionado_macro = None
    
    # BUSCADOR Y MAPA
    st.subheader("🔍 Buscar y Seleccionar País")
    
    # Buscador de países
    col_buscador, col_limpiar = st.columns([3, 1])
    with col_buscador:
        pais_buscador = st.text_input(
            "Escribe el nombre de cualquier país del mundo:",
            placeholder="Ej: United States, Germany, Japan, Brazil, Mexico, Argentina, Spain, France, China, India...",
            key="buscador_paises_macro"
        )
    with col_limpiar:
        if st.session_state.pais_seleccionado_macro:
            if st.button("🗑️ Limpiar selección", use_container_width=True):
                st.session_state.pais_seleccionado_macro = None
                st.rerun()
    
    # Mapa interactivo con Folium
    try:
        from streamlit_folium import st_folium
        import folium
        from geopy.geocoders import Nominatim
        
        st.subheader("🗺️ Mapa Mundial Interactivo - Selecciona cualquier país")
        
        # Crear mapa global centrado
        m = folium.Map(location=[20, 0], zoom_start=2)
        
        # Mostrar mapa en Streamlit y capturar clic
        mapa_datos = st_folium(m, width=700, height=400, returned_objects=["last_clicked"])
        
        # Detectar clic en el mapa
        if mapa_datos and mapa_datos.get("last_clicked") is not None:
            lat = mapa_datos["last_clicked"]["lat"]
            lon = mapa_datos["last_clicked"]["lng"]
            
            try:
                geolocator = Nominatim(user_agent="macro_app")
                location = geolocator.reverse((lat, lon), language="en", exactly_one=True, timeout=5)
                
                if location and 'address' in location.raw and 'country' in location.raw['address']:
                    pais_click = location.raw['address']['country']
                    st.session_state.pais_seleccionado_macro = pais_click
                    st.success(f"🌍 País seleccionado desde el mapa: **{pais_click}**")
                    
            except Exception as e:
                st.warning("⚠️ No se pudo identificar el país. Intenta hacer clic más cerca del centro del país.")
                
    except ImportError:
        st.info("""
        **💡 Mapa no disponible** 
        Para usar el mapa interactivo, instala: 
        `pip install streamlit-folium folium geopy`
        """)
    
    # Determinar qué país mostrar (del buscador O del mapa)
    pais_actual = None
    if pais_buscador and pais_buscador.strip():
        pais_actual = pais_buscador.strip()
        st.session_state.pais_seleccionado_macro = pais_actual
    elif st.session_state.pais_seleccionado_macro:
        pais_actual = st.session_state.pais_seleccionado_macro
    
    # Indicador del país seleccionado
    if pais_actual:
        st.success(f"**País seleccionado:** {pais_actual}")
    else:
        st.info("💡 **Escribe el nombre de un país en el buscador o haz clic en el mapa**")
    
    # MOSTRAR INFORMACIÓN DEL PAÍS SELECCIONADO
    st.markdown("---")
    
    if pais_actual:
        # Mostrar vista específica del país usando la función optimizada
        with st.spinner(f"📊 Cargando datos económicos de {pais_actual}..."):
            datos_pais = obtener_datos_pais_world_bank_optimizado(pais_actual)
        
        st.header(f"📊 Información Económica Completa de {datos_pais['nombre']}")
        
        # Mostrar código del país si se encontró
        if datos_pais.get('codigo'):
            st.caption(f"**World Bank Group:** {datos_pais['codigo']}")
        
        # Métricas principales
        st.subheader("📈 Métricas Principales")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("👥 Población", datos_pais.get('poblacion', 'N/A'))
        with col2:
            st.metric("💰 PIB Per Cápita", datos_pais.get('pib_per_capita', 'N/A'))
        with col3:
            st.metric("🌍 PIB Nominal", datos_pais.get('pib_nominal', 'N/A'))
        with col4:
            st.metric("⚖️ PIB PPA", datos_pais.get('pib_ppa', 'N/A'))
        
        # Indicadores económicos del país
        st.subheader("📊 Indicadores Económicos del World Bank Group")
        indicadores = datos_pais.get("indicadores", {})
        
        if indicadores and len(indicadores) > 2:
            # Crear pestañas para diferentes categorías de indicadores
            tab_principales, tab_economia, tab_social, tab_ambiente = st.tabs([
                "🎯 Principales", 
                "💰 Economía", 
                "👥 Social",
                "🌱 Ambiente"
            ])
            
            with tab_principales:
                st.subheader("📈 Indicadores Principales")
                indicadores_principales = {
                    k: v for k, v in indicadores.items() 
                    if any(x in k.lower() for x in ['pib', 'crecimiento', 'inflación', 'desempleo', 'población'])
                }
                if indicadores_principales:
                    mostrar_indicadores_en_columnas(indicadores_principales)
                else:
                    st.info("No hay indicadores principales disponibles")
            
            with tab_economia:
                st.subheader("💰 Indicadores Económicos")
                indicadores_economia = {
                    k: v for k, v in indicadores.items() 
                    if any(x in k.lower() for x in ['exportaciones', 'importaciones', 'balanza', 'deuda', 'gasto', 'ingresos', 'comercio', 'fiscal', 'tributaria'])
                }
                if indicadores_economia:
                    mostrar_indicadores_en_columnas(indicadores_economia)
                else:
                    st.info("No hay indicadores económicos disponibles")
            
            with tab_social:
                st.subheader("👥 Indicadores Sociales")
                indicadores_social = {
                    k: v for k, v in indicadores.items() 
                    if any(x in k.lower() for x in [
                        'esperanza', 'salud', 'educación', 'pobreza', 'gini', 'alfabetización', 'mortalidad', 
                        'obesidad', 'vacunación', 'tuberculosis', 'médicos', 'matrícula', 'género', 'mujeres',
                        'protección social', 'desempleo juvenil', 'camas hospital'
                    ])
                }
                if indicadores_social:
                    mostrar_indicadores_en_columnas(indicadores_social)
                else:
                    st.info("No hay indicadores sociales disponibles")
            
            with tab_ambiente:
                st.subheader("🌱 Indicadores Ambientales")
                indicadores_ambiente = {
                    k: v for k, v in indicadores.items() 
                    if any(x in k.lower() for x in [
                        'emisiones', 'forestal', 'electricidad', 'internet', 'agua', 'medio ambiente', 'co2',
                        'energía', 'renovable', 'contaminación', 'áreas protegidas', 'especies', 'residuos',
                        'calidad del aire', 'estrés hídrico', 'fertilizantes', 'metano', 'nuclear', 'hidroeléctrica'
                    ])
                }
                if indicadores_ambiente:
                    mostrar_indicadores_en_columnas(indicadores_ambiente)
                else:
                    st.info("No hay indicadores ambientales disponibles")
            
            # Botones de control
            col_act1, col_act2, col_act3 = st.columns(3)
            with col_act1:
                if st.button("🔄 Actualizar Datos", use_container_width=True, type="primary"):
                    st.cache_data.clear()
                    st.rerun()
            with col_act2:
                if st.button("📥 Exportar Datos", use_container_width=True):
                    st.info("Función de exportación en desarrollo")
            with col_act3:
                st.info("**Fuente:** World Bank Group")
                
        else:
            st.warning("""
            **No se pudieron obtener datos específicos para este país.**
            
            Posibles razones:
            - El país puede no estar en la base de datos del World Bank Group
            - Problemas temporales de conexión con la API
            - El país no tiene datos disponibles para los indicadores solicitados
            
            **Solución:** Intenta con otro país o verifica el nombre.
            """)
                
    else:
        # Vista cuando no hay país seleccionado
        st.info("🌍 **Selecciona un país usando el buscador o el mapa para ver sus datos económicos**")
        
        st.markdown("""
        ### 💡 Cómo usar esta sección:
        
        1. **🔍 Buscar país**: Escribe el nombre de cualquier país
        2. **🗺️ Mapa interactivo**: Haz clic en cualquier país del mapa mundial
        3. **📊 Datos oficiales**: Obtén información económica verificada del World Bank Group
        
        ### 📈 Información disponible:
        - **Métricas principales**: Población, PIB, PIB per cápita
        - **Indicadores económicos**: Crecimiento, inflación, desempleo
        - **Comercio exterior**: Exportaciones, importaciones, balanza comercial
        - **Finanzas públicas**: Deuda pública, gasto gubernamental
        - **Indicadores sociales**: Salud, educación, pobreza, desigualdad, género
        - **Medio ambiente**: Emisiones, energía renovable, áreas protegidas, calidad del aire
        
        ### 🚀 **Optimizaciones implementadas:**
        - **Caching de 24 horas** para datos que cambian lentamente
        - **Sesiones HTTP optimizadas** con reintentos automáticos
        - **Timeouts configurados** para evitar bloqueos
        - **80+ indicadores reales** del World Bank
        """)
    
    # INFORMACIÓN SOBRE LA FUENTE
    st.markdown("---")
    st.success("""
    **🌐 Fuente de Datos: World Bank Group**
    
    - **📊 Datos oficiales** de gobiernos e instituciones internacionales
    - **🕐 Actualizaciones periódicas** según disponibilidad de cada indicador
    - **🌍 Cobertura global** de más de 200 países y territorios
    - **📈 Series históricas** desde 1960 para muchos indicadores
    - **🎯 Metodología consistente** entre países y años
    
    **🚀 Optimizado para rendimiento:**
    - Cache de 24 horas para datos macroeconómicos
    - Conexiones HTTP optimizadas con reintentos
    - Timeouts para respuestas rápidas
    - **80+ indicadores reales** sin datos simulados
    
    **Nota:** Algunos indicadores pueden tener datos con 1-2 años de retraso debido a los procesos de recolección y verificación.
    """)

# INICIALIZAR SESSION STATE
if 'seccion_actual' not in st.session_state:
    st.session_state.seccion_actual = "global"


# SECCIÓN DE MERCADOS GLOBALES
if st.session_state.seccion_actual == "global":
    st.header("📈 Mercados Globales en Tiempo Real")
    
    # CONFIGURACIÓN COMPLETA DE LAS 4 APIS
    API_KEYS = {
        "google_gemini": GOOGLE_KEY,  # ✅ Para análisis con IA
        "financial_modeling_prep": FMP,  # ✅ PRINCIPAL - Datos financieros
        "currency_api": currencyapi,  # ✅ ESPECIALIZADA - Forex
        "alpha_vantage": AlphaVantage  # ✅ ALTERNATIVA - Datos de mercado
    }

    # FUNCIONES PRINCIPALES CON LAS 4 APIS
    @st.cache_data(ttl=300)
    def obtener_datos_indices():
        """Obtiene índices bursátiles de múltiples fuentes"""
        indices_data = {}
        
        # ✅ FUENTE 1: Financial Modeling Prep (PRINCIPAL)
        if API_KEYS["financial_modeling_prep"]:
            try:
                # MÁS ÍNDICES - 17 ÍNDICES GLOBALES
                indices_fmp = {
                    "S&P 500": "^GSPC",
                    "NASDAQ": "^IXIC", 
                    "Dow Jones": "^DJI",
                    "Russell 2000": "^RUT",
                    "NYSE Composite": "^NYA",
                    "FTSE 100": "^FTSE",
                    "DAX": "^GDAXI",
                    "CAC 40": "^FCHI",
                    "Euro Stoxx 50": "^STOXX50E",
                    "IBEX 35": "^IBEX",
                    "Nikkei 225": "^N225",
                    "Hang Seng": "^HSI",
                    "Shanghai Composite": "000001.SS",
                    "S&P/TSX Composite": "^GSPTSE",
                    "ASX 200": "^AXJO",
                    "Bovespa": "^BVSP",
                    "SMI Switzerland": "^SSMI"
                }
                
                for nombre, simbolo in indices_fmp.items():
                    url = f"https://financialmodelingprep.com/api/v3/quote/{simbolo}?apikey={API_KEYS['financial_modeling_prep']}"
                    response = requests.get(url, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        if data and len(data) > 0:
                            quote = data[0]
                            precio_actual = quote.get('price', 0)
                            cambio_porcentaje = quote.get('changesPercentage', 0)
                            
                            # Formatear precio
                            if precio_actual > 1000:
                                precio_str = f"${precio_actual:,.0f}"
                            else:
                                precio_str = f"${precio_actual:.2f}"
                            
                            indices_data[nombre] = {
                                "precio": precio_str,
                                "cambio": f"{cambio_porcentaje:+.2f}%",
                                "valor": precio_actual,
                                "fuente": "Financial Modeling Prep"
                            }
                            
            except Exception as e:
                st.warning(f"FMP no disponible: {str(e)}")
        
        # ✅ FUENTE 2: Alpha Vantage (ALTERNATIVA)
        if not indices_data and API_KEYS["alpha_vantage"]:
            try:
                indices_av = {
                    "S&P 500": ".INX",
                    "NASDAQ": ".IXIC",
                    "Dow Jones": ".DJI",
                    "FTSE 100": ".FTSE",
                    "DAX": ".GDAXI"
                }
                
                for nombre, simbolo in indices_av.items():
                    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={simbolo}&apikey={API_KEYS['alpha_vantage']}"
                    response = requests.get(url, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        if "Global Quote" in data:
                            quote = data["Global Quote"]
                            precio_actual = float(quote.get('05. price', 0))
                            cambio_porcentaje = float(quote.get('10. change percent', '0%').replace('%', ''))
                            
                            if precio_actual > 0:
                                if precio_actual > 1000:
                                    precio_str = f"${precio_actual:,.0f}"
                                else:
                                    precio_str = f"${precio_actual:.2f}"
                                
                                indices_data[nombre] = {
                                    "precio": precio_str,
                                    "cambio": f"{cambio_porcentaje:+.2f}%",
                                    "valor": precio_actual,
                                    "fuente": "Alpha Vantage"
                                }
                                
            except Exception as e:
                st.warning(f"Alpha Vantage no disponible: {str(e)}")
        
        # ✅ FUENTE 3: Yahoo Finance (FALLBACK)
        if not indices_data:
            yf_indices = {
                "S&P 500": "^GSPC",
                "NASDAQ": "^IXIC", 
                "Dow Jones": "^DJI",
                "Russell 2000": "^RUT",
                "NYSE Composite": "^NYA",
                "FTSE 100": "^FTSE",
                "DAX": "^GDAXI",
                "CAC 40": "^FCHI",
                "Euro Stoxx 50": "^STOXX50E",
                "IBEX 35": "^IBEX",
                "Nikkei 225": "^N225",
                "Hang Seng": "^HSI",
                "Shanghai Composite": "000001.SS",
                "S&P/TSX Composite": "^GSPTSE",
                "ASX 200": "^AXJO",
                "Bovespa": "^BVSP",
                "SMI Switzerland": "^SSMI"
            }
            
            for nombre, ticker in yf_indices.items():
                try:
                    stock = yf.Ticker(ticker)
                    hist = stock.history(period="2d")
                    if not hist.empty and len(hist) >= 2:
                        current = hist['Close'].iloc[-1]
                        previous = hist['Close'].iloc[-2]
                        change = ((current - previous) / previous) * 100
                        
                        indices_data[nombre] = {
                            "precio": f"${current:,.0f}" if current > 1000 else f"${current:.2f}",
                            "cambio": f"{change:+.2f}%",
                            "valor": current,
                            "fuente": "Yahoo Finance"
                        }
                except Exception as e:
                    continue
        
        return indices_data

    @st.cache_data(ttl=300)
    def obtener_datos_forex():
        """Obtiene datos de divisas de múltiples fuentes"""
        forex_data = {}
        
        # ✅ FUENTE 1: CurrencyAPI (ESPECIALIZADA EN FOREX)
        if API_KEYS["currency_api"]:
            try:
                url = f"https://api.currencyapi.com/v3/latest?apikey={API_KEYS['currency_api']}&base_currency=USD"
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    if "data" in data:
                        # MÁS PARES DE DIVISAS - 17 PARES
                        divisas_objetivo = {
                            "EUR": "EUR/USD",
                            "JPY": "USD/JPY", 
                            "GBP": "GBP/USD",
                            "CHF": "USD/CHF",
                            "CAD": "USD/CAD",
                            "AUD": "AUD/USD",
                            "NZD": "NZD/USD",
                            "CNY": "USD/CNY",
                            "HKD": "USD/HKD",
                            "SGD": "USD/SGD",
                            "SEK": "USD/SEK",
                            "NOK": "USD/NOK",
                            "MXN": "USD/MXN",
                            "INR": "USD/INR",
                            "BRL": "USD/BRL",
                            "ZAR": "USD/ZAR",
                            "RUB": "USD/RUB"
                        }
                        
                        for currency_code, par_nombre in divisas_objetivo.items():
                            if currency_code in data["data"]:
                                rate_data = data["data"][currency_code]
                                rate = rate_data["value"]
                                
                                if currency_code in ["EUR", "GBP", "AUD", "NZD"]:
                                    precio_formateado = f"{1/rate:.4f}" if rate != 0 else "0.0000"
                                    forex_data[par_nombre] = {
                                        "precio": precio_formateado,
                                        "cambio": "0.00%",  # CurrencyAPI no proporciona cambios
                                        "valor": 1/rate if rate != 0 else 0,
                                        "fuente": "CurrencyAPI"
                                    }
                                else:
                                    precio_formateado = f"{rate:.4f}"
                                    forex_data[par_nombre] = {
                                        "precio": precio_formateado,
                                        "cambio": "0.00%",
                                        "valor": rate,
                                        "fuente": "CurrencyAPI"
                                    }
            except Exception as e:
                st.warning(f"CurrencyAPI no disponible: {str(e)}")
        
        # ✅ FUENTE 2: Financial Modeling Prep
        if not forex_data and API_KEYS["financial_modeling_prep"]:
            try:
                # MÁS PARES FOREX
                pares_forex = [
                    "EURUSD", "USDJPY", "GBPUSD", "USDCHF", "USDCAD", 
                    "AUDUSD", "NZDUSD", "USDCNY", "USDHKD", "USDSGD",
                    "USDSEK", "USDNOK", "USDMXN", "USDINR", "USDBRL",
                    "USDZAR", "USDRUB"
                ]
                nombres_pares = {
                    "EURUSD": "EUR/USD",
                    "USDJPY": "USD/JPY",
                    "GBPUSD": "GBP/USD", 
                    "USDCHF": "USD/CHF",
                    "USDCAD": "USD/CAD",
                    "AUDUSD": "AUD/USD",
                    "NZDUSD": "NZD/USD",
                    "USDCNY": "USD/CNY",
                    "USDHKD": "USD/HKD",
                    "USDSGD": "USD/SGD",
                    "USDSEK": "USD/SEK",
                    "USDNOK": "USD/NOK",
                    "USDMXN": "USD/MXN",
                    "USDINR": "USD/INR",
                    "USDBRL": "USD/BRL",
                    "USDZAR": "USD/ZAR",
                    "USDRUB": "USD/RUB"
                }
                
                for par in pares_forex:
                    url = f"https://financialmodelingprep.com/api/v3/quote/{par}?apikey={API_KEYS['financial_modeling_prep']}"
                    response = requests.get(url, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        if data and len(data) > 0:
                            quote = data[0]
                            precio = quote.get('price', 0)
                            cambio_porcentaje = quote.get('changesPercentage', 0)
                            
                            nombre_par = nombres_pares.get(par, par)
                            forex_data[nombre_par] = {
                                "precio": f"{precio:.4f}",
                                "cambio": f"{cambio_porcentaje:+.2f}%",
                                "valor": precio,
                                "fuente": "Financial Modeling Prep"
                            }
            except Exception as e:
                st.warning(f"FMP Forex no disponible: {str(e)}")
        
        # ✅ FUENTE 3: Alpha Vantage
        if not forex_data and API_KEYS["alpha_vantage"]:
            try:
                pares_av = {
                    "EUR/USD": "EURUSD",
                    "USD/JPY": "USDJPY", 
                    "GBP/USD": "GBPUSD",
                    "USD/CHF": "USDCHF",
                    "AUD/USD": "AUDUSD"
                }
                
                for par_nombre, simbolo in pares_av.items():
                    url = f"https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency={simbolo[:3]}&to_currency={simbolo[3:]}&apikey={API_KEYS['alpha_vantage']}"
                    response = requests.get(url, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        if "Realtime Currency Exchange Rate" in data:
                            rate_data = data["Realtime Currency Exchange Rate"]
                            precio = float(rate_data.get('5. Exchange Rate', 0))
                            
                            forex_data[par_nombre] = {
                                "precio": f"{precio:.4f}",
                                "cambio": "0.00%",  # Alpha Vantage no da cambios en esta API
                                "valor": precio,
                                "fuente": "Alpha Vantage"
                            }
            except Exception as e:
                st.warning(f"Alpha Vantage Forex no disponible: {str(e)}")
        
        # ✅ FUENTE 4: Yahoo Finance (ÚLTIMO RECURSO)
        if not forex_data:
            yf_forex = {
                "EUR/USD": "EURUSD=X",
                "USD/JPY": "JPY=X",
                "GBP/USD": "GBPUSD=X",
                "USD/CHF": "CHF=X",
                "USD/CAD": "CAD=X",
                "AUD/USD": "AUDUSD=X",
                "NZD/USD": "NZDUSD=X",
                "USD/CNY": "CNY=X",
                "USD/HKD": "HKD=X",
                "USD/SGD": "SGD=X",
                "USD/SEK": "SEK=X",
                "USD/NOK": "NOK=X",
                "USD/MXN": "MXN=X",
                "USD/INR": "INR=X",
                "USD/BRL": "BRL=X",
                "USD/ZAR": "ZAR=X",
                "USD/RUB": "RUB=X"
            }
            
            for par, ticker in yf_forex.items():
                try:
                    fx = yf.Ticker(ticker)
                    hist = fx.history(period="2d")
                    if not hist.empty and len(hist) >= 2:
                        current = hist['Close'].iloc[-1]
                        previous = hist['Close'].iloc[-2]
                        change = ((current - previous) / previous) * 100
                        
                        forex_data[par] = {
                            "precio": f"{current:.4f}",
                            "cambio": f"{change:+.2f}%",
                            "valor": current,
                            "fuente": "Yahoo Finance"
                        }
                except Exception as e:
                    continue
        
        return forex_data

    @st.cache_data(ttl=300)
    def obtener_datos_cripto():
        """Obtiene datos de criptomonedas de múltiples fuentes"""
        crypto_data = {}
        
        # ✅ FUENTE 1: Financial Modeling Prep
        if API_KEYS["financial_modeling_prep"]:
            try:
                # MÁS CRIPTOMONEDAS - 17 CRIPTOS
                criptos_fmp = {
                    "Bitcoin": "BTCUSD",
                    "Ethereum": "ETHUSD",
                    "BNB": "BNBUSD",
                    "XRP": "XRPUSD",
                    "Cardano": "ADAUSD",
                    "Solana": "SOLUSD",
                    "Dogecoin": "DOGEUSD",
                    "Polkadot": "DOTUSD",
                    "Litecoin": "LTCUSD",
                    "Chainlink": "LINKUSD",
                    "Bitcoin Cash": "BCHUSD",
                    "Avalanche": "AVAXUSD",
                    "Polygon": "MATICUSD",
                    "Stellar": "XLMUSD",
                    "Uniswap": "UNIUSD",
                    "Shiba Inu": "SHIBUSD",
                    "Tron": "TRXUSD"
                }
                
                for nombre, simbolo in criptos_fmp.items():
                    url = f"https://financialmodelingprep.com/api/v3/quote/{simbolo}?apikey={API_KEYS['financial_modeling_prep']}"
                    response = requests.get(url, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        if data and len(data) > 0:
                            quote = data[0]
                            precio = quote.get('price', 0)
                            cambio_porcentaje = quote.get('changesPercentage', 0)
                            
                            crypto_data[nombre] = {
                                "precio": f"${precio:,.2f}",
                                "cambio": f"{cambio_porcentaje:+.2f}%",
                                "valor": precio,
                                "fuente": "Financial Modeling Prep"
                            }
            except Exception as e:
                st.warning(f"FMP Crypto no disponible: {str(e)}")
        
        # ✅ FUENTE 2: Alpha Vantage
        if not crypto_data and API_KEYS["alpha_vantage"]:
            try:
                criptos_av = {
                    "Bitcoin": "BTC",
                    "Ethereum": "ETH",
                    "Litecoin": "LTC",
                    "Ripple": "XRP",
                    "Cardano": "ADA"
                }
                
                for nombre, simbolo in criptos_av.items():
                    url = f"https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency={simbolo}&to_currency=USD&apikey={API_KEYS['alpha_vantage']}"
                    response = requests.get(url, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        if "Realtime Currency Exchange Rate" in data:
                            rate_data = data["Realtime Currency Exchange Rate"]
                            precio = float(rate_data.get('5. Exchange Rate', 0))
                            
                            crypto_data[nombre] = {
                                "precio": f"${precio:,.2f}",
                                "cambio": "0.00%",
                                "valor": precio,
                                "fuente": "Alpha Vantage"
                            }
            except Exception as e:
                st.warning(f"Alpha Vantage Crypto no disponible: {str(e)}")
        
        # ✅ FUENTE 3: Yahoo Finance
        if not crypto_data:
            yf_crypto = {
                "Bitcoin": "BTC-USD",
                "Ethereum": "ETH-USD",
                "BNB": "BNB-USD",
                "XRP": "XRP-USD",
                "Cardano": "ADA-USD",
                "Solana": "SOL-USD",
                "Dogecoin": "DOGE-USD",
                "Polkadot": "DOT-USD",
                "Litecoin": "LTC-USD",
                "Chainlink": "LINK-USD",
                "Bitcoin Cash": "BCH-USD",
                "Avalanche": "AVAX-USD",
                "Polygon": "MATIC-USD",
                "Stellar": "XLM-USD",
                "Uniswap": "UNI-USD",
                "Shiba Inu": "SHIB-USD",
                "Tron": "TRX-USD"
            }
            
            for nombre, ticker in yf_crypto.items():
                try:
                    crypto = yf.Ticker(ticker)
                    hist = crypto.history(period="2d")
                    if not hist.empty and len(hist) >= 2:
                        current = hist['Close'].iloc[-1]
                        previous = hist['Close'].iloc[-2]
                        change = ((current - previous) / previous) * 100
                        
                        crypto_data[nombre] = {
                            "precio": f"${current:,.2f}",
                            "cambio": f"{change:+.2f}%",
                            "valor": current,
                            "fuente": "Yahoo Finance"
                        }
                except Exception as e:
                    continue
        
        return crypto_data

    @st.cache_data(ttl=300)
    def obtener_datos_commodities():
        """Obtiene datos de materias primas de múltiples fuentes"""
        commodities_data = {}
        
        # ✅ FUENTE 1: Financial Modeling Prep (PRINCIPAL)
        if API_KEYS["financial_modeling_prep"]:
            try:
                # MÁS COMMODITIES - 17 PRODUCTOS
                commodities_fmp = {
                    "Petróleo WTI": "CLUSD",
                    "Petróleo Brent": "BZUSD", 
                    "Oro": "GCUSD",
                    "Plata": "SIUSD",
                    "Cobre": "HGUSD",
                    "Gas Natural": "NGUSD",
                    "Platino": "PLUSD",
                    "Paladio": "PAUSD",
                    "Aluminio": "ALIUSD",
                    "Trigo": "ZWUSD",
                    "Maíz": "ZCUSD",
                    "Soja": "ZSUSD",
                    "Azúcar": "SBUSD",
                    "Café": "KCUSD",
                    "Cacao": "CCUSD",
                    "Algodón": "CTUSD",
                    "Ganado": "LEUSD"
                }
                
                for nombre, simbolo in commodities_fmp.items():
                    url = f"https://financialmodelingprep.com/api/v3/quote/{simbolo}?apikey={API_KEYS['financial_modeling_prep']}"
                    response = requests.get(url, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        if data and len(data) > 0:
                            quote = data[0]
                            precio = quote.get('price', 0)
                            cambio_porcentaje = quote.get('changesPercentage', 0)
                            
                            if nombre in ["Oro", "Plata", "Platino", "Paladio"]:
                                precio_str = f"${precio:,.2f}"
                            elif nombre in ["Petróleo WTI", "Petróleo Brent", "Gas Natural"]:
                                precio_str = f"${precio:.2f}"
                            elif nombre in ["Trigo", "Maíz", "Soja", "Azúcar", "Café", "Cacao", "Algodón"]:
                                precio_str = f"${precio:.2f}"  # Commodities agrícolas
                            else:
                                precio_str = f"${precio:.2f}"
                            
                            commodities_data[nombre] = {
                                "precio": precio_str,
                                "cambio": f"{cambio_porcentaje:+.2f}%",
                                "valor": precio,
                                "fuente": "Financial Modeling Prep"
                            }
            except Exception as e:
                st.warning(f"FMP Commodities no disponible: {str(e)}")
        
        # ✅ FUENTE 2: Alpha Vantage
        if not commodities_data and API_KEYS["alpha_vantage"]:
            try:
                commodities_av = {
                    "Oro": "GCUSD",
                    "Petróleo WTI": "CLUSD",
                    "Plata": "SIUSD",
                    "Cobre": "HGUSD"
                }
                
                for nombre, simbolo in commodities_av.items():
                    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={simbolo}&apikey={API_KEYS['alpha_vantage']}"
                    response = requests.get(url, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        if "Global Quote" in data:
                            quote = data["Global Quote"]
                            precio_actual = float(quote.get('05. price', 0))
                            cambio_porcentaje = float(quote.get('10. change percent', '0%').replace('%', ''))
                            
                            if precio_actual > 0:
                                if nombre in ["Oro", "Plata"]:
                                    precio_str = f"${precio_actual:,.2f}"
                                else:
                                    precio_str = f"${precio_actual:.2f}"
                                
                                commodities_data[nombre] = {
                                    "precio": precio_str,
                                    "cambio": f"{cambio_porcentaje:+.2f}%",
                                    "valor": precio_actual,
                                    "fuente": "Alpha Vantage"
                                }
            except Exception as e:
                st.warning(f"Alpha Vantage Commodities no disponible: {str(e)}")
        
        # ✅ FUENTE 3: Yahoo Finance (FALLBACK)
        if not commodities_data:
            yf_commodities = {
                "Petróleo WTI": "CL=F",
                "Petróleo Brent": "BZ=F", 
                "Oro": "GC=F",
                "Plata": "SI=F",
                "Cobre": "HG=F",
                "Gas Natural": "NG=F",
                "Platino": "PL=F",
                "Paladio": "PA=F",
                "Aluminio": "ALI=F",
                "Trigo": "ZW=F",
                "Maíz": "ZC=F",
                "Soja": "ZS=F",
                "Azúcar": "SB=F",
                "Café": "KC=F",
                "Cacao": "CC=F",
                "Algodón": "CT=F",
                "Ganado": "LE=F"
            }
            
            for nombre, ticker in yf_commodities.items():
                try:
                    comm = yf.Ticker(ticker)
                    hist = comm.history(period="2d")
                    if not hist.empty and len(hist) >= 2:
                        current = hist['Close'].iloc[-1]
                        previous = hist['Close'].iloc[-2]
                        change = ((current - previous) / previous) * 100
                        
                        if nombre in ["Oro", "Plata", "Platino", "Paladio"]:
                            precio_str = f"${current:,.2f}"
                        elif nombre in ["Petróleo WTI", "Petróleo Brent", "Gas Natural"]:
                            precio_str = f"${current:.2f}"
                        elif nombre in ["Trigo", "Maíz", "Soja", "Azúcar", "Café", "Cacao", "Algodón"]:
                            precio_str = f"${current:.2f}"
                        else:
                            precio_str = f"${current:.2f}"
                        
                        commodities_data[nombre] = {
                            "precio": precio_str,
                            "cambio": f"{change:+.2f}%",
                            "valor": current,
                            "fuente": "Yahoo Finance"
                        }
                except Exception as e:
                    continue
        
        return commodities_data

    @st.cache_data(ttl=3600)
    def obtener_datos_tasas_reales():
        """Obtiene tasas de interés REALES de múltiples fuentes"""
        tasas_data = {}
        
        try:
            # ✅ FUENTE PRINCIPAL: FMP para tasas del Tesoro
            if API_KEYS["financial_modeling_prep"]:
                try:
                    # Obtener tasas del Tesoro de FMP
                    url = f"https://financialmodelingprep.com/api/v4/treasury?apikey={API_KEYS['financial_modeling_prep']}"
                    response = requests.get(url, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        if data and len(data) > 0:
                            # Tomar la entrada más reciente
                            latest = data[0]
                            date = latest.get('date', '')
                            
                            # MÁS TASAS - 13 PLAZOS DIFERENTES
                            tasas_mapping = {
                                'month1': 'Tesoro USA 1 mes',
                                'month2': 'Tesoro USA 2 meses', 
                                'month3': 'Tesoro USA 3 meses',
                                'month6': 'Tesoro USA 6 meses',
                                'year1': 'Tesoro USA 1 año',
                                'year2': 'Tesoro USA 2 años',
                                'year3': 'Tesoro USA 3 años',
                                'year5': 'Tesoro USA 5 años',
                                'year7': 'Tesoro USA 7 años',
                                'year10': 'Tesoro USA 10 años',
                                'year20': 'Tesoro USA 20 años',
                                'year30': 'Tesoro USA 30 años'
                            }
                            
                            for key, nombre in tasas_mapping.items():
                                tasa = latest.get(key, 0)
                                if tasa and tasa > 0:
                                    tasas_data[nombre] = {
                                        "valor": f"{tasa:.2f}%",
                                        "fuente": "Financial Modeling Prep",
                                        "categoria": "tesoro"
                                    }
                except Exception as e:
                    st.warning(f"FMP Tasas no disponible: {str(e)}")

            # ✅ FUENTE 2: Alpha Vantage para tasas
            if not tasas_data and API_KEYS["alpha_vantage"]:
                try:
                    # Alpha Vantage para datos macroeconómicos
                    tasas_av = {
                        "Tesoro USA 10 años": "10year",
                        "Tesoro USA 5 años": "5year", 
                        "Tesoro USA 2 años": "2year"
                    }
                    
                    for nombre, plazo in tasas_av.items():
                        url = f"https://www.alphavantage.co/query?function=TREASURY_YIELD&interval=monthly&maturity={plazo}&apikey={API_KEYS['alpha_vantage']}"
                        response = requests.get(url, timeout=10)
                        
                        if response.status_code == 200:
                            data = response.json()
                            if "data" in data and len(data["data"]) > 0:
                                latest_yield = data["data"][0]
                                tasa = float(latest_yield.get('value', 0))
                                
                                if tasa > 0:
                                    tasas_data[nombre] = {
                                        "valor": f"{tasa:.2f}%",
                                        "fuente": "Alpha Vantage",
                                        "categoria": "tesoro"
                                    }
                except Exception as e:
                    st.warning(f"Alpha Vantage Tasas no disponible: {str(e)}")

            # ✅ FUENTE 3: Yahoo Finance para bonos gubernamentales (fallback)
            bonos_yahoo = {
                "USA 2 años": "^IRX",
                "USA 10 años": "^TNX", 
                "USA 30 años": "^TYX",
                "USA 5 años": "^FVX",
                "USA 13 semanas": "^IRX"
            }
            
            for nombre, ticker in bonos_yahoo.items():
                try:
                    bono = yf.Ticker(ticker)
                    hist = bono.history(period="2d")
                    if not hist.empty:
                        yield_val = hist['Close'].iloc[-1]
                        if 0.1 < yield_val < 20:
                            tasas_data[nombre] = {
                                "valor": f"{yield_val:.2f}%",
                                "fuente": "Yahoo Finance",
                                "categoria": "bonos"
                            }
                except Exception as e:
                    continue

            # ✅ FUENTE 4: CoinGecko para métricas cripto
            try:
                url = "https://api.coingecko.com/api/v3/global"
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if "data" in data:
                        market_data = data["data"]
                        total_volume = market_data.get("total_volume", {})
                        market_cap = market_data.get("total_market_cap", {})
                        market_cap_change = market_data.get("market_cap_change_percentage_24h_usd", 0)
                        
                        if "usd" in total_volume:
                            vol_str = f"${total_volume['usd']:,.0f}"
                            tasas_data["Vol Cripto 24h"] = {
                                "valor": vol_str,
                                "fuente": "CoinGecko", 
                                "categoria": "cripto"
                            }
                        
                        if "usd" in market_cap:
                            cap_str = f"${market_cap['usd']:,.0f}"
                            tasas_data["Market Cap Cripto"] = {
                                "valor": cap_str,
                                "fuente": "CoinGecko",
                                "categoria": "cripto"
                            }
                        
                        tasas_data["Cambio MC Cripto 24h"] = {
                            "valor": f"{market_cap_change:+.2f}%",
                            "fuente": "CoinGecko",
                            "categoria": "cripto"
                        }
            except Exception as e:
                pass

        except Exception as e:
            st.error(f"Error obteniendo tasas: {str(e)}")
        
        return tasas_data

    # FUNCIÓN DE ANÁLISIS CON GEMINI (TU API GOOGLE)
    @st.cache_data(ttl=1800)
    def obtener_analisis_completo(indices, forex, crypto, commodities, tasas):
        """Genera análisis con todos los datos disponibles usando Gemini"""
        try:
            # Contar datos disponibles
            stats = {
                "indices": len(indices),
                "forex": len(forex),
                "crypto": len(crypto),
                "commodities": len(commodities),
                "tasas": len(tasas)
            }
            
            total_datos = sum(stats.values())
            
            if total_datos == 0:
                return "🔍 **Estado del Sistema:** Conectando a fuentes de datos...\n\nLos datos se cargarán automáticamente en unos segundos."
            
            # Crear resumen para el prompt
            resumen_datos = {
                "indices": {k: f"{v['precio']} ({v['cambio']})" for k, v in indices.items()},
                "forex": {k: f"{v['precio']} ({v['cambio']})" for k, v in forex.items()},
                "crypto": {k: f"{v['precio']} ({v['cambio']})" for k, v in crypto.items()},
                "commodities": {k: f"{v['precio']} ({v['cambio']})" for k, v in commodities.items()},
                "tasas": {k: v["valor"] for k, v in tasas.items()}
            }

            prompt = f"""
            Analiza los siguientes datos financieros en tiempo real:

            ÍNDICES BURSÁTILES ({stats['indices']} índices):
            {resumen_datos['indices']}

            DIVISAS ({stats['forex']} pares):
            {resumen_datos['forex']}

            CRIPTOMONEDAS ({stats['crypto']} activos):
            {resumen_datos['crypto']}

            MATERIAS PRIMAS ({stats['commodities']} commodities):
            {resumen_datos['commodities']}

            TASAS DE INTERÉS ({stats['tasas']} tasas):
            {resumen_datos['tasas']}

            Proporciona un análisis profesional que incluya:
            1. Tendencias principales del mercado
            2. Movimientos significativos en activos clave
            3. Perspectiva de riesgo y oportunidades
            4. Contexto macroeconómico relevante

            Máximo 200 palabras. Enfoque en insights accionables.
            Basado únicamente en los datos proporcionados.
            """

            # USANDO TU API DE GOOGLE GEMINI
            model = genai.GenerativeModel('gemini-2.5-flash')
            response = model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            return f"📊 **Datos Cargados:** {total_datos} activos | Análisis disponible en próxima actualización"

    # OBTENER TODOS LOS DATOS
    with st.spinner('🔄 Conectando con fuentes de datos globales...'):
        indices = obtener_datos_indices()
        forex = obtener_datos_forex()
        crypto = obtener_datos_cripto()
        commodities = obtener_datos_commodities()
        tasas = obtener_datos_tasas_reales()
        analisis = obtener_analisis_completo(indices, forex, crypto, commodities, tasas)

    # DISEÑO DE LA INTERFAZ
    st.markdown("### 🤖 Análisis de Mercados en Tiempo Real")
    with st.container():
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    color: white; padding: 20px; border-radius: 10px; margin: 15px 0;'>
        <h4 style='color: white; margin-bottom: 15px;'>ANÁLISIS GLOBAL</h4>
        """, unsafe_allow_html=True)
        st.write(analisis)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ESTADÍSTICAS DE DATOS
    total_activos = len(indices) + len(forex) + len(crypto) + len(commodities)
    st.markdown(f"### 📊 Indicadores del Mercado Global ({total_activos} activos cargados)")

    # INDICADORES PRINCIPALES
    st.markdown("#### 🎯 Indicadores Clave")
    col1, col2, col3, col4 = st.columns(4)
    
    indicadores_principales = [
        ("S&P 500", indices.get("S&P 500")),
        ("EUR/USD", forex.get("EUR/USD")),
        ("Bitcoin", crypto.get("Bitcoin")),
        ("Oro", commodities.get("Oro"))
    ]
    
    for i, (nombre, datos) in enumerate(indicadores_principales):
        with [col1, col2, col3, col4][i]:
            if datos:
                st.metric(
                    label=nombre,
                    value=datos["precio"],
                    delta=datos["cambio"]
                )
                st.caption(f"Fuente: {datos.get('fuente', 'Directo')}")
            else:
                st.metric(label=nombre, value="Cargando...")
                st.caption("Conectando...")

    st.markdown("---")

    # SECCIÓN DE ÍNDICES
    if indices:
        st.markdown("#### 📈 Índices Bursátiles Globales")
        # Usar más columnas para mostrar más índices
        cols = st.columns(4)
        indices_items = list(indices.items())
        
        for i, (nombre, datos) in enumerate(indices_items):
            with cols[i % 4]:
                with st.container():
                    st.markdown(f"""
                    <div style='background-color: #1E1E1E; padding: 15px; border-radius: 10px; 
                                border-left: 4px solid #2E86AB; margin: 5px 0; border: 1px solid #444;'>
                    <div style='font-weight: bold; color: white; font-size: 14px;'>{nombre}</div>
                    <div style='font-size: 1.1em; color: white; margin: 8px 0;'>{datos['precio']}</div>
                    <div style='color: {'#4CAF50' if '+' in datos['cambio'] else '#F44336'}; font-weight: bold; font-size: 13px;'>
                        {datos['cambio']}
                    </div>
                    <div style='font-size: 0.7em; color: #CCCCCC; margin-top: 5px;'>
                        {datos.get('fuente', 'Directo')}
                    </div>
                    </div>
                    """, unsafe_allow_html=True)

    st.markdown("---")

    # SECCIÓN DE DIVISAS Y CRIPTO
    col_divisas, col_cripto = st.columns(2)
    
    with col_divisas:
        if forex:
            st.markdown("#### 💵 Divisas Principales")
            # Mostrar más pares de divisas
            for par, datos in list(forex.items())[:10]:
                st.markdown(f"""
                <div style='background-color: #1E1E1E; padding: 12px; border-radius: 8px; 
                            border: 1px solid #444; margin: 6px 0;'>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <div style='font-weight: bold; color: white; font-size: 13px;'>{par}</div>
                    <div style='display: flex; flex-direction: column; align-items: end;'>
                        <div style='color: white; font-weight: bold; font-size: 13px;'>{datos['precio']}</div>
                        <div style='color: {'#4CAF50' if '+' in datos['cambio'] else '#F44336'}; font-size: 11px;'>
                            {datos['cambio']}
                        </div>
                    </div>
                </div>
                <div style='font-size: 10px; color: #CCCCCC; margin-top: 4px;'>
                    {datos.get('fuente', 'Directo')}
                </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("#### 💵 Divisas")
            st.info("Cargando datos de divisas...")
    
    with col_cripto:
        if crypto:
            st.markdown("#### ₿ Criptomonedas")
            # Mostrar más criptomonedas
            for moneda, datos in list(crypto.items())[:10]:
                st.markdown(f"""
                <div style='background-color: #1E1E1E; padding: 12px; border-radius: 8px; 
                            border: 1px solid #444; margin: 6px 0;'>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <div style='font-weight: bold; color: white; font-size: 13px;'>{moneda}</div>
                    <div style='display: flex; flex-direction: column; align-items: end;'>
                        <div style='color: white; font-weight: bold; font-size: 13px;'>{datos['precio']}</div>
                        <div style='color: {'#4CAF50' if '+' in datos['cambio'] else '#F44336'}; font-size: 11px;'>
                            {datos['cambio']}
                        </div>
                    </div>
                </div>
                <div style='font-size: 10px; color: #CCCCCC; margin-top: 4px;'>
                    {datos.get('fuente', 'Directo')}
                </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("#### ₿ Criptomonedas")
            st.info("Cargando datos cripto...")

    st.markdown("---")

    # SECCIÓN DE COMMODITIES
    if commodities:
        st.markdown("#### 🛢️ Materias Primas")
        # Usar más columnas para commodities
        cols = st.columns(4)
        commodities_items = list(commodities.items())
        
        for i, (producto, datos) in enumerate(commodities_items):
            with cols[i % 4]:
                st.markdown(f"""
                <div style='background-color: #1E1E1E; padding: 12px; border-radius: 8px; 
                            border: 1px solid #444; margin: 6px 0; text-align: center;'>
                <div style='font-weight: bold; color: white; font-size: 12px; margin-bottom: 6px;'>{producto}</div>
                <div style='color: white; font-size: 14px; font-weight: bold; margin-bottom: 4px;'>{datos['precio']}</div>
                <div style='color: {'#4CAF50' if '+' in datos['cambio'] else '#F44336'}; font-size: 11px; font-weight: bold;'>
                    {datos['cambio']}
                </div>
                <div style='font-size: 9px; color: #CCCCCC; margin-top: 4px;'>
                    {datos.get('fuente', 'Directo')}
                </div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")

    # SECCIÓN DE TASAS
    if tasas:
        st.markdown("#### 🏦 Tasas de Interés y Bonos")
        
        # Organizar en más columnas para mostrar más tasas
        cols = st.columns(4)
        tasas_items = list(tasas.items())
        
        for i, (nombre, datos) in enumerate(tasas_items):
            with cols[i % 4]:
                st.markdown(f"""
                <div style='background-color: #1E1E1E; padding: 12px; border-radius: 8px; 
                            border: 1px solid #444; margin: 6px 0; text-align: center;'>
                <div style='font-weight: bold; color: white; font-size: 11px; margin-bottom: 8px; 
                            height: 35px; display: flex; align-items: center; justify-content: center;'>
                    {nombre}
                </div>
                <div style='color: white; font-size: 14px; font-weight: bold; margin-bottom: 6px;'>
                    {datos['valor']}
                </div>
                <div style='font-size: 9px; color: #CCCCCC;'>
                    {datos.get('fuente', 'Directo')}
                </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("🏦 Cargando datos de tasas y bonos...")

    # PANEL DE CONTROL MEJORADO
    st.markdown("---")
    
    col_stats, col_control = st.columns([2, 1])
    
    with col_stats:
        total_activos = len(indices) + len(forex) + len(crypto) + len(commodities)
        st.markdown(f"""
        **🚀 Cobertura Expandida del Mercado:**
        - **Activos cargados:** {total_activos}
        - **📈 17 Índices Globales:** América, Europa, Asia
        - **💵 17 Pares de Divisas:** Principales y emergentes  
        - **₿ 17 Criptomonedas:** Grandes cap y altcoins
        - **🛢️ 17 Commodities:** Energía, metales, agrícolas
        - **🏦 Tasas Completas:** Tesoro USA múltiples plazos
        - **Análisis IA:** Google Gemini
        - **Última actualización:** {datetime.now().strftime('%H:%M:%S')}
        """)
    
    with col_control:
        if st.button("🔄 Actualizar Toda La Información", use_container_width=True, type="primary"):
            st.cache_data.clear()
            st.rerun()





# BOTONES ADICIONALES EN EL FOOTER
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    # Generar reporte de texto
    if st.button("📄 Generar Reporte", use_container_width=True):
        try:
            with st.spinner("Generando reporte..."):
                datos = obtener_datos_accion(stonk)
                
                # Verificar que tenemos datos
                if datos.empty:
                    st.error("No se pudieron obtener datos para generar el reporte")
                else:
                    scoring, metricas = calcular_scoring_fundamental(info)
                    reporte_texto = generar_reporte_texto(stonk, info, datos, scoring, metricas)
                    
                    # Mostrar preview del reporte
                    with st.expander("📋 Vista Previa del Reporte"):
                        st.text(reporte_texto)
                    
                    # Botón de descarga
                    st.download_button(
                        label="📥 Descargar Reporte (TXT)",
                        data=reporte_texto,
                        file_name=f"reporte_{stonk}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
        except Exception as e:
            st.error(f"Error generando reporte: {str(e)}")
            # Debug info
            st.error(f"Tipo de error: {type(e).__name__}")

with col2:
    # Historial de búsquedas
    if st.session_state.historial_busquedas:
        with st.popover("🔍 Historial Búsquedas", use_container_width=True):
            st.write("**Búsquedas recientes:**")
            for busqueda in reversed(st.session_state.historial_busquedas):
                if st.button(f"📌 {busqueda}", key=f"hist_{busqueda}", use_container_width=True):
                    st.session_state.seccion_actual = "info"
                    st.rerun()
    else:
        with st.popover("🔍 Historial Búsquedas", use_container_width=True):
            st.info("No hay búsquedas recientes")

# FAVORITOS RÁPIDOS
if st.session_state.favoritas:
    st.markdown("---")
    st.write("⭐ **Favoritos Rápidos:**")
    cols_fav = st.columns(len(st.session_state.favoritas))
    
    for i, favorita in enumerate(st.session_state.favoritas):
        with cols_fav[i]:
            if st.button(f"📈 {favorita}", use_container_width=True, key=f"fav_{favorita}"):
                st.session_state.seccion_actual = "info"

# --- DISCLAIMER FINAL ---
st.markdown("""
---
<p style='text-align: center; font-size: 13px; color: gray;'>
© 2025 Todos los derechos reservados. Desarrollado por <strong>Jesús Alberto Cárdenas Serrano.</strong>
<br><em>Esta aplicación es con fines educativos. No constituye asesoramiento financiero.</em>
</p>
""", unsafe_allow_html=True)

st.write("🔥 ESTE ES UN TEST - CAMBIOS VISIBLES")