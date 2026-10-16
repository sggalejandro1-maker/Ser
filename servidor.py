import os
import requests
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

app = FastAPI(title="Sergio BTC Scalper Pro", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

HISTORIAL_PREDICCIONES = {
    "aciertos": 0,
    "fallos": 0,
    "total": 0,
    "winrate": 0.0,
    "ultima_senal": None,
}

def obtener_velas_binance(symbol="BTCUSDT", interval="1m", limit=300):
    """Consulta la API de Binance con reintentos y timeouts optimizados."""
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    for _ in range(2):
        try:
            response = requests.get(url, headers=HEADERS, timeout=2.5)
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data, columns=[
                    "timestamp", "open", "high", "low", "close", "volume",
                    "close_time", "quote_av", "trades", "tb_base", "tb_quote", "ignore"
                ])
                df["open"] = df["open"].astype(float)
                df["close"] = df["close"].astype(float)
                df["high"] = df["high"].astype(float)
                df["low"] = df["low"].astype(float)
                df["volume"] = df["volume"].astype(float)
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
                df["close_time"] = df["close_time"].astype(int)
                return df
        except Exception as e:
            print(f"Reintento klines {interval}: {e}")
    return None

def resamplear_velas(df_1m, regla="5min"):
    """Genera velas de temporalidades mayores a partir de 1m localmente."""
    if df_1m is None or len(df_1m) == 0:
        return None
    try:
        df = df_1m.copy()
        df.set_index("timestamp", inplace=True)
        resampled = df.resample(regla).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "close_time": "last"
        }).dropna().reset_index()
        return resampled
    except Exception as e:
        print(f"Error resampleando {regla}: {e}")
        return None

def obtener_libro_ordenes(symbol="BTCUSDT", limit=20):
    try:
        url = f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit={limit}"
        response = requests.get(url, headers=HEADERS, timeout=2.0)
        if response.status_code == 200:
            data = response.json()
            bids = data.get("bids", [])
            asks = data.get("asks", [])

            compras_usd = sum(float(b[0]) * float(b[1]) for b in bids)
            ventas_usd = sum(float(a[0]) * float(a[1]) for a in asks)
            total_usd = compras_usd + ventas_usd

            if total_usd > 0:
                porcentaje_compras = round((compras_usd / total_usd) * 100, 1)
                porcentaje_ventas = round((ventas_usd / total_usd) * 100, 1)
                desequilibrio = round(porcentaje_compras - porcentaje_ventas, 1)
            else:
                porcentaje_compras, porcentaje_ventas, desequilibrio = 50.0, 50.0, 0.0

            top_bids = [{"precio": round(float(b[0]), 2), "monto_usd": round(float(b[0]) * float(b[1]), 2)} for b in bids[:5]]
            top_asks = [{"precio": round(float(a[0]), 2), "monto_usd": round(float(a[0]) * float(a[1]), 2)} for a in asks[:5]]

            return {
                "compras_usd": round(compras_usd),
                "ventas_usd": round(ventas_usd),
                "porcentaje_compras": porcentaje_compras,
                "porcentaje_ventas": porcentaje_ventas,
                "desequilibrio": f"{'+' if desequilibrio > 0 else ''}{desequilibrio}%",
                "top_bids": top_bids,
                "top_asks": top_asks,
            }
    except Exception as e:
        print(f"Error libro ordenes: {e}")

    return {
        "compras_usd": 0, "ventas_usd": 0, "porcentaje_compras": 50.0,
        "porcentaje_ventas": 50.0, "desequilibrio": "0%", "top_bids": [], "top_asks": []
    }

def calcular_rsi(serie, periodo=14):
    delta = serie.diff()
    ganancia = (delta.where(delta > 0, 0)).rolling(window=periodo).mean()
    perdida = (-delta.where(delta < 0, 0)).rolling(window=periodo).mean()
    rs = ganancia / perdida
    return 100 - (100 / (1 + rs))

def calcular_atr(df, periodo=14):
    alta_baja = df["high"] - df["low"]
    alta_cierre = (df["high"] - df["close"].shift()).abs()
    baja_cierre = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([alta_baja, alta_cierre, baja_cierre], axis=1).max(axis=1)
    return tr.rolling(window=periodo).mean()

def analizar_tf(df, tendencia_macro_alcista=True):
    if df is None or len(df) < 20:
        return {
            "estado": "ESPERAR", "sesgo": "NEUTRAL", "presion_cierre": 50,
            "explicacion": "Cargando datos...", "indice_alcista": 0, "indice_bajista": 0,
            "precio_actual": 0.0, "rsi": 0.0, "ema9": 0.0, "ema21": 0.0, "ema50": 0.0, "ema200": 0.0,
            "volatilidad_valida": True, "close_time": 0
        }

    df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()
    df["rsi"] = calcular_rsi(df["close"], periodo=14)
    df["atr"] = calcular_atr(df, periodo=14)
    df["vol_sma"] = df["volume"].rolling(window=20).mean()

    v = df.iloc[-1]
    puntos_alcistas = 0
    puntos_bajistas = 0

    if v["close"] > v["ema200"] and v["ema9"] > v["ema50"]:
        sesgo = "ALCISTA (BULLISH)"
    elif v["close"] < v["ema200"] and v["ema9"] < v["ema50"]:
        sesgo = "BAJISTA (BEARISH)"
    else:
        sesgo = "NEUTRAL / EN RANGO"

    rango_total = v["high"] - v["low"]
    presion_cierre = round(((v["close"] - v["low"]) / rango_total) * 100) if rango_total > 0 else 50

    if v["close"] > v["ema50"]: puntos_alcistas += 35
    else: puntos_bajistas += 35

    if v["ema9"] > v["ema21"]: puntos_alcistas += 35
    else: puntos_bajistas += 35

    if not pd.isna(v["rsi"]):
        if v["rsi"] > 53: puntos_alcistas += 30
        elif v["rsi"] < 47: puntos_bajistas += 30

    volatilidad_ok = not pd.isna(v["atr"]) and v["atr"] >= 5.0
    volumen_ok = pd.isna(v["vol_sma"]) or v["volume"] >= (v["vol_sma"] * 0.8)

    razones = []
    if v["close"] > v["ema50"]: razones.append("precio sobre EMA50")
    else: razones.append("precio bajo EMA50")

    if v["ema9"] > v["ema21"]: razones.append("cruce alcista EMA 9/21")
    else: razones.append("cruce bajista EMA 9/21")

    if not pd.isna(v["rsi"]):
        if v["rsi"] > 53: razones.append("RSI alcista (>53)")
        elif v["rsi"] < 47: razones.append("RSI bajista (<47)")

    if not volumen_ok: razones.append("volumen bajo (filtrado)")

    explicacion = "Conclusión basada en: " + ", ".join(razones) + "."

    if volatilidad_ok and volumen_ok and puntos_alcistas >= 75:
        estado = "POSIBLE LONG" if tendencia_macro_alcista else "ESPERAR (Contra Tendencia 15M)"
    elif volatilidad_ok and volumen_ok and puntos_bajistas >= 75:
        estado = "POSIBLE SHORT" if not tendencia_macro_alcista else "ESPERAR (Contra Tendencia 15M)"
    else:
        estado = "ESPERAR"

    return {
        "estado": estado,
        "sesgo": sesgo,
        "presion_cierre": presion_cierre,
        "explicacion": explicacion,
        "indice_alcista": puntos_alcistas,
        "indice_bajista": puntos_bajistas,
        "precio_actual": round(v["close"], 2),
        "rsi": round(v["rsi"], 2) if not pd.isna(v["rsi"]) else 0,
        "ema9": round(v["ema9"], 2),
        "ema21": round(v["ema21"], 2),
        "ema50": round(v["ema50"], 2),
        "ema200": round(v["ema200"], 2),
        "volatilidad_valida": volatilidad_ok,
        "close_time": int(v["close_time"])
    }

@app.get("/api/senales")
def obtener_senales():
    try:
        # Una sola consulta principal a Binance para aligerar la conexión en Render
        df_1m = obtener_velas_binance("BTCUSDT", interval="1m", limit=300)
        
        if df_1m is not None:
            df_3m = resamplear_velas(df_1m, "3min")
            df_5m = resamplear_velas(df_1m, "5min")
            df_15m = resamplear_velas(df_1m, "15min")
        else:
            df_3m, df_5m, df_15m = None, None, None

        tendencia_macro_alcista = True
        if df_15m is not None and len(df_15m) > 0:
            ema9_15 = df_15m["close"].ewm(span=9, adjust=False).mean().iloc[-1]
            ema21_15 = df_15m["close"].ewm(span=21, adjust=False).mean().iloc[-1]
            tendencia_macro_alcista = ema9_15 > ema21_15

        resultado = {
            "1m": analizar_tf(df_1m, tendencia_macro_alcista),
            "3m": analizar_tf(df_3m, tendencia_macro_alcista),
            "5m": analizar_tf(df_5m, tendencia_macro_alcista),
        }

        presion_5m = resultado["5m"]["presion_cierre"]
        sesgo_5m = resultado["5m"]["sesgo"]
        estados = [resultado[tf]["estado"] for tf in ["1m", "3m", "5m"]]

        if estados.count("POSIBLE LONG") >= 2:
            decision_global = "ALTA CONFLUENCIA - POSIBLE LONG"
            reaccion_txt = "Confluencia alcista validada con volumen y macro 15m."
        elif estados.count("POSIBLE SHORT") >= 2:
            decision_global = "ALTA CONFLUENCIA - POSIBLE SHORT"
            reaccion_txt = "Dominio bajista validado con volumen institucional."
        else:
            decision_global = "FILTRO INSTITUCIONAL - ESPERAR"
            reaccion_txt = f"Presión de cierre en 5m: {presion_5m}%. Mercado bajo revisión."

        resultado["sistema"] = {
            "decision": decision_global,
            "reaccion": reaccion_txt,
            "sesgo_global": sesgo_5m,
            "fibo_titulo": "TENDENCIA MACRO 15M FILTRADA",
            "posible_alza": presion_5m,
            "posible_baja": 100 - presion_5m,
        }

        resultado["precision"] = {
            "aciertos": HISTORIAL_PREDICCIONES["aciertos"],
            "fallos": HISTORIAL_PREDICCIONES["fallos"],
            "total": HISTORIAL_PREDICCIONES["total"],
            "winrate": HISTORIAL_PREDICCIONES["winrate"],
        }

        resultado["libro"] = obtener_libro_ordenes("BTCUSDT")
        return resultado

    except Exception as e:
        print(f"Error general en api/senales: {e}")
        return {
            "1m": {"precio_actual": 0.0, "estado": "ESPERAR", "sesgo": "-", "indice_alcista": 0, "indice_bajista": 0, "presion_cierre": 0, "explicacion": "Reintentando...", "rsi": 0, "ema9": 0, "ema21": 0, "ema50": 0},
            "3m": {"precio_actual": 0.0, "estado": "ESPERAR", "sesgo": "-", "indice_alcista": 0, "indice_bajista": 0, "presion_cierre": 0, "explicacion": "Reintentando...", "rsi": 0, "ema9": 0, "ema21": 0, "ema50": 0},
            "5m": {"precio_actual": 0.0, "estado": "ESPERAR", "sesgo": "-", "indice_alcista": 0, "indice_bajista": 0, "presion_cierre": 0, "explicacion": "Reintentando...", "rsi": 0, "ema9": 0, "ema21": 0, "ema50": 0},
            "sistema": {"decision": "RECONECTANDO...", "reaccion": "Estableciendo conexión rápida con Binance.", "sesgo_global": "-", "fibo_titulo": "Cargando...", "posible_alza": 50, "posible_baja": 50},
            "precision": {"aciertos": 0, "fallos": 0, "total": 0, "winrate": 0.0},
            "libro": {"compras_usd": 0, "ventas_usd": 0, "porcentaje_compras": 50, "porcentaje_ventas": 50, "desequilibrio": "0%", "top_bids": [], "top_asks": []}
        }

@app.get("/", response_class=HTMLResponse)
def leer_interfaz():
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    return "<h3>Error: No se encontró el archivo index.html</h3>"
