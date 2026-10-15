import os
import time
import pandas as pd
import requests


def obtener_velas_binance(symbol="BTCUSDT", interval="5m", limit=100):
  url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
  response = requests.get(url)
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
  df["close"] = df["close"].astype(float)
  df["high"] = df["high"].astype(float)
  df["low"] = df["low"].astype(float)
  return df


def calcular_rsi(serie, periodo=14):
  delta = serie.diff()
  ganancia = (delta.where(delta > 0, 0)).rolling(window=periodo).mean()
  perdida = (-delta.where(delta < 0, 0)).rolling(window=periodo).mean()
  rs = ganancia / perdida
  return 100 - (100 / (1 + rs))


def analizar_temporalidad(df):
  df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
  df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()
  df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
  df["rsi"] = calcular_rsi(df["close"], periodo=14)

  v = df.iloc[-1]
  puntos_alcistas = 0
  puntos_bajistas = 0

  if v["close"] > v["ema50"]:
    puntos_alcistas += 40
  else:
    puntos_bajistas += 40

  if v["ema9"] > v["ema21"]:
    puntos_alcistas += 40
  else:
    puntos_bajistas += 40

  if not pd.isna(v["rsi"]):
    if v["rsi"] > 52:
      puntos_alcistas += 20
    elif v["rsi"] < 48:
      puntos_bajistas += 20

  if puntos_alcistas >= 70:
    estado = "POSIBLE LONG"
  elif puntos_bajistas >= 70:
    estado = "POSIBLE SHORT"
  else:
    estado = "ESPERAR"

  return {
      "estado": estado,
      "alcista": puntos_alcistas,
      "bajista": puntos_bajistas,
      "precio": v["close"],
  }


# --- BUCLE PRINCIPAL EN TIEMPO REAL ---
if __name__ == "__main__":
  print("Iniciando monitor en vivo de Bitcoin...")
  time.sleep(1)

  while True:
    # Limpiar pantalla de la terminal (funciona en Windows)
    os.system("cls")

    print("=== BITCOIN LIVE PREDICTION (BINANCE) ===")
    print("Actualizando automáticamente cada 5 segundos...\n")

    for tf in ["1m", "3m", "5m"]:
      df_velas = obtener_velas_binance("BTCUSDT", interval=tf)
      if df_velas is not None:
        resultado = analizar_temporalidad(df_velas)
        print(f"--- Temporalidad: {tf.upper()} ---")
        print(f"Precio actual: {resultado['precio']}")
        print(f"Estado: {resultado['estado']}")
        print(
            f"Índice Alcista: {resultado['alcista']}% | Índice Bajista:"
            f" {resultado['bajista']}%\n"
        )

    # Esperar 5 segundos antes de la siguiente consulta
    time.sleep(5)