import time
import requests
import pandas as pd
import schedule
from datetime import datetime

# ==========================================
# BASE URL API CONFIGURATION
# ==========================================
COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"
PIONEX_BASE_URL = "https://api.pionex.com"

# ==========================================
# 1. API HELPER FUNCTIONS
# ==========================================

def get_pionex_pairs():
    """Mengambil daftar pair USDT yang aktif menggunakan Pionex API."""
    url = f"{PIONEX_BASE_URL}/api/v1/market/symbols"
    try:
        res = requests.get(url, timeout=10).json()
        if res.get("result"):
            return {data['symbol']: data for data in res['data']['symbols'] if data['symbol'].endswith('_USDT') and data['enable']}
    except Exception as e:
        print(f"[!] Gagal mengambil data Pionex: {e}")
    return {}

def get_top_100_coingecko():
    """Mengambil 100 koin teratas berdasarkan marketcap menggunakan CoinGecko API."""
    url = f"{COINGECKO_BASE_URL}/coins/markets"
    params = {
        'vs_currency': 'usd',
        'order': 'market_cap_desc',
        'per_page': 100,
        'page': 1,
        'sparkline': False
    }
    try:
        res = requests.get(url, params=params, timeout=10)
        if res.status_code == 200:
            return res.json()
    except Exception as e:
        print(f"[!] Gagal mengambil data CoinGecko: {e}")
    return []

def get_ohlcv(coin_id, days='7'):
    """Mengambil data OHLCV historis menggunakan CoinGecko API."""
    url = f"{COINGECKO_BASE_URL}/coins/{coin_id}/ohlc"
    params = {'vs_currency': 'usd', 'days': days}
    try:
        res = requests.get(url, params=params, timeout=10)
        if res.status_code == 200:
            df = pd.DataFrame(res.json(), columns=['time', 'open', 'high', 'low', 'close'])
            # Proxy volume volatilitas untuk perhitungan indikator volume
            df['volume'] = df['close'] * (df['high'] - df['low'])
            return df
    except Exception:
        pass
    return None

# ==========================================
# 2. TECHNICAL ANALYSIS (GRADE A+ COMBO)
# ==========================================

def calculate_indicators(df):
    """Menghitung EMA 20, 50, 200, Stochastic RSI (14, 14, 3, 3), & Volume MA 20."""
    if df is None or len(df) < 50:
        return None

    # EMA Calculations
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema200'] = df['close'].ewm(span=200, adjust=False).mean() if len(df) >= 200 else df['close'].ewm(span=len(df), adjust=False).mean()

    # RSI (14)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-9)
    df['rsi'] = 100 - (100 / (1 + rs))

    # Stochastic RSI (14, 14, 3, 3)
    stoch_rsi = (df['rsi'] - df['rsi'].rolling(14).min()) / ((df['rsi'].rolling(14).max() - df['rsi'].rolling(14).min()).replace(0, 1e-9))
    df['stoch_k'] = stoch_rsi.rolling(3).mean() * 100
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()

    # Volume MA 20
    df['vol_ma20'] = df['volume'].rolling(20).mean()

    return df

def analyze_signal(df):
    """Menganalisis setup aturan Long & Short Scalping."""
    if df is None or len(df) < 3:
        return None

    curr = df.iloc[-1]
    prev = df.iloc[-2]

    close = curr['close']
    ema20, ema50, ema200 = curr['ema20'], curr['ema50'], curr['ema200']
    k_curr, d_curr = curr['stoch_k'], curr['stoch_d']
    k_prev, d_prev = prev['stoch_k'], prev['stoch_d']
    vol_curr, vol_ma = curr['volume'], curr['vol_ma20']

    # Konfirmasi Volume
    vol_valid = vol_curr > vol_ma

    # --- CONDITIONAL CHECK FOR LONG ---
    bullish_trend = (close > ema200) and (ema20 > ema50)
    stoch_cross_up = (k_prev <= d_prev) and (k_curr > d_curr) and (k_curr < 20 or d_curr < 20)

    if bullish_trend and stoch_cross_up and vol_valid:
        return {
            'type': 'LONG',
            'grade': 'Grade A+ Combo (Bullish)',
            'op': close,
            'tp': close * 1.02,  # Fixed TP 2%
            'sl': close * 0.99,  # Fixed SL 1%
            'score': (k_curr - d_curr) + (vol_curr / (vol_ma + 1e-9))
        }

    # --- CONDITIONAL CHECK FOR SHORT ---
    bearish_trend = (close < ema200) and (ema20 < ema50)
    stoch_cross_down = (k_prev >= d_prev) and (k_curr < d_curr) and (k_curr > 80 or d_curr > 80)

    if bearish_trend and stoch_cross_down and vol_valid:
        return {
            'type': 'SHORT',
            'grade': 'Grade A+ Combo (Bearish)',
            'op': close,
            'tp': close * 0.98,  # Fixed TP 2%
            'sl': close * 1.01,  # Fixed SL 1%
            'score': (d_curr - k_curr) + (vol_curr / (vol_ma + 1e-9))
        }

    return None

# ==========================================
# 3. MAIN SCOUTBOT EXECUTION ENGINE
# ==========================================

def run_scoutbot():
    print(f"\n==================================================")
    print(f"   SCOUTBOT RUNNING - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"==================================================")

    print("[1/4] Mengambil daftar pasar Pionex...")
    pionex_pairs = get_pionex_pairs()
    if not pionex_pairs:
        print("[!] Pionex API tidak dapat dijangkau. Membatalkan analisis.")
        return

    print("[2/4] Mengambil Top 100 Koin dari CoinGecko...")
    top_100 = get_top_100_coingecko()
    if not top_100:
        print("[!] CoinGecko API tidak merespons.")
        return

    candidates = []

    print("[3/4] Melakukan analisis indikator & konfirmasi Pionex...")
    for coin in top_100:
        symbol_usdt = f"{coin['symbol'].upper()}_USDT"

        # Validasi ketersediaan pair di Pionex
        if symbol_usdt not in pionex_pairs:
            continue

        # Fetch Data Candle & Analisis
        df = get_ohlcv(coin['id'])
        df = calculate_indicators(df)
        signal = analyze_signal(df)

        if signal:
            signal['symbol'] = symbol_usdt
            signal['name'] = coin['name']
            candidates.append(signal)

        # Rate limit delay ringan
        time.sleep(0.2)

    # Sortir kandidat terbaik (4 hingga 10 koin)
    candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)[:10]

    print("[4/4] Output Sinyal Eksekusi Scalping:\n")
    if not candidates:
        print(" [i] Tidak ada koin yang memenuhi kriteria Grade A+ pada siklus 4 jam ini.")
    else:
        for idx, item in enumerate(candidates, 1):
            print(f"--------------------------------------------------")
            print(f"#{idx} Nama Koin             : {item['name']} ({item['symbol']})")
            print(f"   Tipe Posisi           : {item['type']}")
            print(f"   Grade Eksekusi Koin   : {item['grade']}")
            print(f"   Open Posisi (OP)      : ${item['op']:.4f}")
            print(f"   Take Profit (TP 2%)   : ${item['tp']:.4f}")
            print(f"   Stop Loss (SL 1%)     : ${item['sl']:.4f}")

    print(f"==================================================\n")

# ==========================================
# 4. SCHEDULER (EKSEKUSI SETIAP 4 JAM SEKALI)
# ==========================================

if __name__ == "__main__":
    run_scoutbot()

    schedule.every(4).hours.do(run_scoutbot)

    print("[*] Scoutbot aktif. Menunggu jadwal eksekusi berikutnya (Setiap 4 jam sekali)...")
    while True:
        schedule.run_pending()
        time.sleep(60)
