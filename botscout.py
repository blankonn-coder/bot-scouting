import logging
import os
import sys
import time
from datetime import datetime, timezone

import pandas as pd
import requests

# ==========================================
# KONFIGURASI (dibaca dari GitHub Secrets / environment)
# ==========================================
COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"
PIONEX_BASE_URL = "https://api.pionex.com"

COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY", "")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")

TOP_N = 100
MAX_SIGNALS = 10
KLINE_INTERVAL = "4H"
KLINE_LIMIT = 300        # cukup untuk EMA200 + warm-up
MIN_CANDLES = 210
REQUEST_DELAY = 0.3      # jeda antar request ke Pionex (detik)
HTTP_TIMEOUT = 15
MAX_RETRIES = 4

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("scoutbot")


# ==========================================
# 1. HTTP HELPER (retry + backoff)
# ==========================================
def http_get(url, params=None, headers=None):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=HTTP_TIMEOUT)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                wait = 5 * attempt
                log.warning("HTTP %s dari %s, retry %s/%s dalam %ss",
                            r.status_code, url, attempt, MAX_RETRIES, wait)
                time.sleep(wait)
                continue
            log.warning("HTTP %s dari %s, dilewati", r.status_code, url)
            return None
        except (requests.RequestException, ValueError) as e:
            log.warning("Request gagal (%s), retry %s/%s", e, attempt, MAX_RETRIES)
            time.sleep(5 * attempt)
    return None


# ==========================================
# 2. API HELPER
# ==========================================
def get_pionex_pairs():
    """Daftar simbol USDT yang aktif di Pionex."""
    data = http_get(f"{PIONEX_BASE_URL}/api/v1/common/symbols")
    if not data or not data.get("result"):
        return set()
    return {
        s["symbol"]
        for s in data["data"]["symbols"]
        if s["symbol"].endswith("_USDT") and s.get("enable")
    }


def get_top_coins():
    """Top 100 koin berdasarkan market cap dari CoinGecko (1 request)."""
    headers = {"x-cg-demo-api-key": COINGECKO_API_KEY} if COINGECKO_API_KEY else None
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": TOP_N,
        "page": 1,
        "sparkline": False,
    }
    data = http_get(f"{COINGECKO_BASE_URL}/coins/markets", params=params, headers=headers)
    return data or []


def get_klines(symbol):
    """OHLCV 4H dari Pionex (volume asli). Candle terakhir dibuang karena belum tutup."""
    data = http_get(
        f"{PIONEX_BASE_URL}/api/v1/market/klines",
        params={"symbol": symbol, "interval": KLINE_INTERVAL, "limit": KLINE_LIMIT},
    )
    if not data or not data.get("result"):
        return None

    df = pd.DataFrame(data["data"]["klines"])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("time").reset_index(drop=True)
    return df.iloc[:-1].reset_index(drop=True)


# ==========================================
# 3. TECHNICAL ANALYSIS (Grade A+ Combo)
# ==========================================
def calculate_indicators(df):
    """EMA 20/50/200, Stochastic RSI (14,14,3,3), Volume MA 20."""
    if df is None or len(df) < MIN_CANDLES:
        return None
    df = df.copy()

    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()

    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))

    rsi_min = df["rsi"].rolling(14).min()
    rsi_max = df["rsi"].rolling(14).max()
    stoch_rsi = (df["rsi"] - rsi_min) / (rsi_max - rsi_min).replace(0, 1e-9)
    df["stoch_k"] = stoch_rsi.rolling(3).mean() * 100
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    df["vol_ma20"] = df["volume"].rolling(20).mean()
    return df


def analyze_signal(df):
    """Long/Short scalping berdasarkan candle yang sudah tutup."""
    if df is None:
        return None

    curr = df.iloc[-1]
    prev = df.iloc[-2]
    close = curr["close"]
    ema20, ema50, ema200 = curr["ema20"], curr["ema50"], curr["ema200"]
    k_curr, d_curr = curr["stoch_k"], curr["stoch_d"]
    k_prev, d_prev = prev["stoch_k"], prev["stoch_d"]
    vol_curr, vol_ma = curr["volume"], curr["vol_ma20"]

    if pd.isna(k_curr) or pd.isna(d_curr) or pd.isna(vol_ma):
        return None

    vol_valid = vol_curr > vol_ma

    # LONG
    bullish_trend = (close > ema200) and (ema20 > ema50)
    stoch_cross_up = (k_prev <= d_prev) and (k_curr > d_curr) and (k_curr < 20 or d_curr < 20)
    if bullish_trend and stoch_cross_up and vol_valid:
        return {
            "type": "LONG",
            "grade": "Grade A+ Combo (Bullish)",
            "op": close,
            "tp": close * 1.02,
            "sl": close * 0.99,
            "score": (k_curr - d_curr) + (vol_curr / (vol_ma + 1e-9)),
        }

    # SHORT
    bearish_trend = (close < ema200) and (ema20 < ema50)
    stoch_cross_down = (k_prev >= d_prev) and (k_curr < d_curr) and (k_curr > 80 or d_curr > 80)
    if bearish_trend and stoch_cross_down and vol_valid:
        return {
            "type": "SHORT",
            "grade": "Grade A+ Combo (Bearish)",
            "op": close,
            "tp": close * 0.98,
            "sl": close * 1.01,
            "score": (d_curr - k_curr) + (vol_curr / (vol_ma + 1e-9)),
        }
    return None


# ==========================================
# 4. OUTPUT (log + Discord)
# ==========================================
def fmt_price(p):
    return f"{p:.6g}"


def format_signal(idx, item):
    return (
        f"**#{idx} {item['name']} ({item['symbol']})**\n"
        f"Posisi: {item['type']} | {item['grade']}\n"
        f"OP: ${fmt_price(item['op'])} | TP 2%: ${fmt_price(item['tp'])} | SL 1%: ${fmt_price(item['sl'])}\n\n"
    )


def send_discord(text):
    if not DISCORD_WEBHOOK_URL:
        log.warning("DISCORD_WEBHOOK_URL belum di-set, output hanya di log.")
        return

    # Discord membatasi 2000 karakter per pesan
    chunks, buf = [], ""
    for line in text.splitlines(keepends=True):
        if len(buf) + len(line) > 1900:
            chunks.append(buf)
            buf = ""
        buf += line
    if buf:
        chunks.append(buf)

    for chunk in chunks:
        r = requests.post(DISCORD_WEBHOOK_URL, json={"content": chunk}, timeout=HTTP_TIMEOUT)
        if r.status_code == 429:
            time.sleep(float(r.json().get("retry_after", 1)) + 0.5)
            r = requests.post(DISCORD_WEBHOOK_URL, json={"content": chunk}, timeout=HTTP_TIMEOUT)
        if r.status_code >= 300:
            log.warning("Discord gagal kirim pesan: HTTP %s", r.status_code)
        time.sleep(1)


# ==========================================
# 5. MAIN (satu siklus, lalu exit)
# ==========================================
def run_scoutbot():
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    log.info("SCOUTBOT RUNNING - %s", now)

    log.info("[1/4] Mengambil daftar pair Pionex...")
    pairs = get_pionex_pairs()
    if not pairs:
        log.error("Pionex tidak dapat dijangkau atau daftar pair kosong.")
        return 1

    log.info("[2/4] Mengambil Top %s koin dari CoinGecko...", TOP_N)
    top = get_top_coins()
    if not top:
        log.error("CoinGecko tidak merespons.")
        return 1

    log.info("[3/4] Analisis indikator...")
    candidates, seen = [], set()
    for coin in top:
        symbol = f"{coin['symbol'].upper()}_USDT"
        if symbol not in pairs or symbol in seen:
            continue
        seen.add(symbol)

        df = calculate_indicators(get_klines(symbol))
        signal = analyze_signal(df)
        if signal:
            signal["symbol"] = symbol
            signal["name"] = coin["name"]
            candidates.append(signal)

        time.sleep(REQUEST_DELAY)

    candidates.sort(key=lambda x: x["score"], reverse=True)
    candidates = candidates[:MAX_SIGNALS]

    log.info("[4/4] Mengirim hasil...")
    header = f"**SCOUTBOT {now}** | pair dianalisis: {len(seen)}\n\n"
    if candidates:
        msg = header + "".join(format_signal(i, c) for i, c in enumerate(candidates, 1))
    else:
        msg = header + "Tidak ada koin yang memenuhi kriteria Grade A+ pada siklus ini."

    print(msg)
    send_discord(msg)
    return 0


if __name__ == "__main__":
    sys.exit(run_scoutbot())
