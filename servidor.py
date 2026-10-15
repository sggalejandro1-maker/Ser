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
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    )
}

HISTORIAL_PREDICCIONES = {
    "aciertos": 0,
    "fallos": 0,
    "total": 0,
    "winrate": 0.0,
    "ultima_senal": None,
}


def obtener_velas_binance(symbol="BTCUSDT", interval="5m", limit=200):
  try:
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    response = requests.get(url, headers=HEADERS, timeout=4)
    if response.status_code != 200:
      return None

    data = response.json()
    df = pd.DataFrame(
        data,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_av",
            "trades",
            "tb_base",
            "tb_quote",
            "ignore",
        ],
    )

    df["open"] = df["open"].astype(float)
    df["close"] = df["close"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["volume"] = df["volume"].astype(float)
    df["close_time"] = df["close_time"].astype(int)
    return df
  except Exception as e:
    print(f"Error klines {interval}: {e}")
    return None


def obtener_libro_ordenes(symbol="BTCUSDT", limit=20):
  try:
    url = f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit={limit}"
    response = requests.get(url, headers=HEADERS, timeout=3)
    if response.status_code != 200:
      return None

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
      porcentaje_compras = 50.0
      porcentaje_ventas = 50.0
      desequilibrio = 0.0

    top_bids = [
        {
            "precio": round(float(b[0]), 2),
            "monto_usd": round(float(b[0]) * float(b[1]), 2),
        }
        for b in bids[:5]
    ]
    top_asks = [
        {
            "precio": round(float(a[0]), 2),
            "monto_usd": round(float(a[0]) * float(a[1]), 2),
        }
        for a in asks[:5]
    ]

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
        "compras_usd": 0,
        "ventas_usd": 0,
        "porcentaje_compras": 50,
        "porcentaje_ventas": 50,
        "desequilibrio": "0%",
        "top_bids": [],
        "top_asks": [],
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


def evaluar_precision(df_5m, estado_global):
  global HISTORIAL_PREDICCIONES
  if df_5m is None or len(df_5m) < 2:
    return

  vela_actual = df_5m.iloc[-1]
  vela_anterior = df_5m.iloc[-2]
  close_time_anterior = int(vela_anterior["close_time"])

  if (
      HISTORIAL_PREDICCIONES["ultima_senal"] is not None
      and HISTORIAL_PREDICCIONES["ultima_senal"]["close_time"]
      == close_time_anterior
      and not HISTORIAL_PREDICCIONES["ultima_senal"].get("procesado", False)
  ):

    precio_entrada = HISTORIAL_PREDICCIONES["ultima_senal"]["precio_entrada"]
    precio_cierre = vela_anterior["close"]
    tipo_senal = HISTORIAL_PREDICCIONES["ultima_senal"]["estado"]

    if tipo_senal == "POSIBLE LONG":
      if precio_cierre > precio_entrada:
        HISTORIAL_PREDICCIONES["aciertos"] += 1
      else:
        HISTORIAL_PREDICCIONES["fallos"] += 1
      HISTORIAL_PREDICCIONES["total"] += 1

    elif tipo_senal == "POSIBLE SHORT":
      if precio_cierre < precio_entrada:
        HISTORIAL_PREDICCIONES["aciertos"] += 1
      else:
        HISTORIAL_PREDICCIONES["fallos"] += 1
      HISTORIAL_PREDICCIONES["total"] += 1

    HISTORIAL_PREDICCIONES["ultima_senal"]["procesado"] = True

    if HISTORIAL_PREDICCIONES["total"] > 0:
      HISTORIAL_PREDICCIONES["winrate"] = round(
          (
              HISTORIAL_PREDICCIONES["aciertos"]
              / HISTORIAL_PREDICCIONES["total"]
          )
          * 100,
          1,
      )

  if "POSIBLE" in estado_global:
    HISTORIAL_PREDICCIONES["ultima_senal"] = {
        "estado": "POSIBLE LONG"
        if "LONG" in estado_global
        else "POSIBLE SHORT",
        "precio_entrada": vela_actual["close"],
        "close_time": int(vela_actual["close_time"]),
        "procesado": False,
    }


def analizar_tf(df, tendencia_macro_alcista=True):
  df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
  df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()
  df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
  df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()
  df["rsi"] = calcular_rsi(df["close"], periodo=14)
  df["atr"] = calcular_atr(df, periodo=14)

  # NUEVO FILTRO: Volumen Institucional (volumen mayor al promedio de 20 velas)
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
  if rango_total > 0:
    presion_cierre = round(((v["close"] - v["low"]) / rango_total) * 100)
  else:
    presion_cierre = 50

  if v["close"] > v["ema50"]:
    puntos_alcistas += 35
  else:
    puntos_bajistas += 35

  if v["ema9"] > v["ema21"]:
    puntos_alcistas += 35
  else:
    puntos_bajistas += 35

  if not pd.isna(v["rsi"]):
    if v["rsi"] > 53:
      puntos_alcistas += 30
    elif v["rsi"] < 47:
      puntos_bajistas += 30

  volatilidad_ok = True
  if not pd.isna(v["atr"]) and v["atr"] < 5.0:
    volatilidad_ok = False

  # NUEVO: Validar que el volumen esté respaldado
  volumen_ok = True
  if not pd.isna(v["vol_sma"]) and v["volume"] < (v["vol_sma"] * 0.8):
    volumen_ok = False  # Volumen bajo, se descarta fuerza institucional

  razones = []
  if v["close"] > v["ema50"]:
    razones.append("precio sobre EMA50")
  else:
    razones.append("precio bajo EMA50")

  if v["ema9"] > v["ema21"]:
    razones.append("cruce alcista EMA 9/21")
  else:
    razones.append("cruce bajista EMA 9/21")

  if not pd.isna(v["rsi"]):
    if v["rsi"] > 53:
      razones.append("RSI alcista (>53)")
    elif v["rsi"] < 47:
      razones.append("RSI bajista (<47)")

  if not volumen_ok:
    razones.append("volumen bajo (filtrado)")

  explicacion = "Conclusión basada en: " + ", ".join(razones) + "."

  # Aplicar filtros estrictos de calidad
  if volatilidad_ok and volumen_ok and puntos_alcistas >= 75:
    if tendencia_macro_alcista:
      estado = "POSIBLE LONG"
    else:
      estado = "ESPERAR (Contra Tendencia 15M)"
  elif volatilidad_ok and volumen_ok and puntos_bajistas >= 75:
    if not tendencia_macro_alcista:
      estado = "POSIBLE SHORT"
    else:
      estado = "ESPERAR (Contra Tendencia 15M)"
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
      "close_time": int(v["close_time"]),
  }


@app.get("/api/senales")
def obtener_senales():
  resultado = {}
  df_5m = None

  # Consultar temporalidad macro de 15M para validar tendencia
  df_15m = obtener_velas_binance("BTCUSDT", interval="15m", limit=50)
  tendencia_macro_alcista = True
  if df_15m is not None and len(df_15m) > 0:
    v15 = df_15m.iloc[-1]
    ema50_15m = v15["close"].ewm(span=50, adjust=False).mean() if False else None
    # Simplificación macro: Precio sobre EMA 50 de 15m o EMA9 > EMA21
    ema9_15 = df_15m["close"].ewm(span=9, adjust=False).mean().iloc[-1]
    ema21_15 = df_15m["close"].ewm(span=21, adjust=False).mean().iloc[-1]
    tendencia_macro_alcista = ema9_15 > ema21_15

  for tf in ["1m", "3m", "5m"]:
    df = obtener_velas_binance("BTCUSDT", interval=tf)
    if df is not None:
      resultado[tf] = analizar_tf(df, tendencia_macro_alcista)
      if tf == "5m":
        df_5m = df

  if "5m" in resultado:
    presion_5m = resultado["5m"]["presion_cierre"]
    sesgo_5m = resultado["5m"]["sesgo"]
    estados = [
        resultado.get(tf, {}).get("estado", "ESPERAR")
        for tf in ["1m", "3m", "5m"]
    ]

    if estados.count("POSIBLE LONG") >= 2:
      decision_global = "ALTA CONFLUENCIA - POSIBLE LONG"
      reaccion_txt = (
          f"Confluencia alcista validada con volumen y macro 15m."
      )
    elif estados.count("POSIBLE SHORT") >= 2:
      decision_global = "ALTA CONFLUENCIA - POSIBLE SHORT"
      reaccion_txt = f"Dominio bajista validado con volumen institucional."
    else:
      decision_global = "FILTRO INSTITUCIONAL - ESPERAR"
      reaccion_txt = (
          f"Mercado filtrado por volumen o contra tendencia macro de 15m."
      )

    evaluar_precision(df_5m, decision_global)

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


@app.get("/", response_class=HTMLResponse)
def leer_interfaz():
  if os.path.exists("index.html"):
    with open("index.html", "r", encoding="utf-8") as f:
      return f.read()
  return "<h3>Error: No se encontró el archivo index.html</h3>"