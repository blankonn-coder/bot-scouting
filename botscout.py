"""
ScoutBot - pemindai sinyal Grade A+ Combo (4H) untuk pair USDT.

Sumber data:
  - Daftar pair & venue eksekusi : Pionex  (https://api.pionex.com)
  - OHLCV (candle)               : Pionex, cadangan Binance
                                   (https://api.binance.com lalu https://data-api.binance.vision)
  - Ranking top koin             : CoinGecko

Dijalankan satu kali per job (GitHub Actions cron). Hasil dikirim ke Discord webhook.
"""
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
PIONEX_BASE_URL = "https://api.pionex.com"
BINANCE_BASE_URLS = ["https://api.binance.com", "https://data-api.binance.vision"]
COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"

COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY", "").strip()
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()

TOP_N = 100              # jumlah koin dari CoinGecko
MAX_SIGNALS = 10         # maksimal sinyal per siklus
KLINE_LIMIT = 300        # cukup untuk warm-up EMA200
MIN_CANDLES = 210        # minimal candle tertutup agar indikator valid
INTERVAL_PIONEX = "4H"   # format Pionex: 1M,5M,15M,30M,60M,4H,8H,12H,1D
INTERVAL_BINANCE = "4h"  # format Binance: 1m,5m,15m,30m,1h,2h,4h,...
INTERVAL_MS = 4 * 60 * 60 * 1000
REQUEST_DELAY = 0.25     # jeda antar koin (detik)
HTTP_TIMEOUT = 15
MAX_RETRIES = 3

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("scoutbot")


# ==========================================
# 1. HTTP HELPER
# ==========================================
def http_get(url, params=None, headers=None):
    """
    GET dengan retry untuk error jaringan, 429, dan 5xx.
    Return JSON, atau None jika gagal. Error 4xx lain tidak diulang.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=HTTP_TIMEOUT)
        except requests.RequestException as e:
            log.warning("Koneksi ke %s gagal (%s), percobaan %d/%d", url, e, attempt, MAX_RETRIES)
            time.sleep(3 * attempt)
            continue

        if r.status_code == 200:
            try:
                return r.json()
            except ValueError:
                log.warning("Response dari %s bukan JSON: %s", url, r.text[:150])
                return None

        if r.status_code == 429 or r.status_code >= 500:
            log.warning("HTTP %s dari %s, percobaan %d/%d", r.status_code, url, attempt, MAX_RETRIES)
            time.sleep(5 * attempt)
            continue

        # 400 (simbol tidak ada), 403/451 (wilayah dibatasi), dll: tidak perlu diulang
        log.warning("HTTP %s dari %s: %s", r.status_code, url, r.text[:150])
        return None
    return None


# ==========================================
# 2. DATA SOURCE: PIONEX
# ==========================================
def get_pionex_pairs():
    """Set pair USDT aktif di Pionex, format BASE_QUOTE (contoh: BTC_USDT)."""
    data = http_get(f"{PIONEX_BASE_URL}/api/v1/common/symbols")
    if not isinstance(data, dict) or not data.get("result"):
        return set()

    symbols = (data.get("data") or {}).get("symbols") or []
    return {
        s["symbol"]
        for s in symbols
        if str(s.get("symbol", "")).endswith("_USDT") and s.get("enable", True)
    }


# ==========================================
# 3. DATA SOURCE: BINANCE
# ==========================================
def klines_binance(base):
    """Candle 4H dari Binance untuk BASE/USDT. Mencoba host utama lalu host cadangan."""
    symbol = f"{base}USDT"
    for host in BINANCE_BASE_URLS:
        data = http_get(
            f"{host}/api/v3/klines",
            params={"symbol": symbol, "interval": INTERVAL_BINANCE, "limit": KLINE_LIMIT},
        )
        # Binance: list of [open_time, open, high, low, close, volume, close_time, ...]
        if isinstance(data, list) and data:
            df = pd.DataFrame([row[:6] for row in data],
                              columns=["time", "open", "high", "low", "close", "volume"])
            return finalize(df)
    return None


# ==========================================
# 4. DATA SOURCE: PIONEX KLINES
# ==========================================
def klines_pionex(base):
    """Candle 4H dari Pionex (volume asli)."""
    data = http_get(
        f"{PIONEX_BASE_URL}/api/v1/market/klines",
        params={"symbol": f"{base}_USDT", "interval": INTERVAL_PIONEX, "limit": KLINE_LIMIT},
    )
    if not isinstance(data, dict) or not data.get("result"):
        return None

    rows = (data.get("data") or {}).get("klines") or []
    if not rows:
        return None

    df = pd.DataFrame(rows)
    need = ["time", "open", "high", "low", "close", "volume"]
    if not all(c in df.columns for c in need):
        log.warning("Format klines Pionex %s tidak sesuai, kolom yang ada: %s",
                    base, list(df.columns))
        return None
    return finalize(df[need])


def finalize(df):
    """Konversi tipe, urutkan, dan buang candle yang belum tutup."""
    for col in ["time", "open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna().sort_values("time").reset_index(drop=True)
    if df.empty:
        return None

    # Pastikan timestamp dalam milidetik
    if df["time"].iloc[-1] < 1e12:
        df["time"] = df["time"] * 1000

    # Candle terakhir masih berjalan jika open_time + interval > sekarang
    now_ms = int(time.time() * 1000)
    if df["time"].iloc[-1] + INTERVAL_MS > now_ms:
        df = df.iloc[:-1]

    return df.reset_index(drop=True)


def get_candles(base, use_pionex):
    """Coba Pionex dulu (jika tersedia), lalu Binance. Return (df, sumber) atau (None, None)."""
    if use_pionex:
        df = klines_pionex(base)
        if df is not None and len(df) >= MIN_CANDLES:
            return df, "Pionex"

    df = klines_binance(base)
    if df is not None and len(df) >= MIN_CANDLES:
        return df, "Binance"
    return None, None


def get_top_coins():
    """Top N koin berdasarkan market cap dari CoinGecko."""
    headers = {"x-cg-demo-api-key": COINGECKO_API_KEY} if COINGECKO_API_KEY else None
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": TOP_N,
        "page": 1,
        "sparkline": "false",
    }
    data = http_get(f"{COINGECKO_BASE_URL}/coins/markets", params=params, headers=headers)
    return data if isinstance(data, list) else []


# ==========================================
# 5. TECHNICAL ANALYSIS (Grade A+ Combo)
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
    if df is None or len(df) < 2:
        return None

    curr = df.iloc[-1]
    prev = df.iloc[-2]

    values = [curr["stoch_k"], curr["stoch_d"], prev["stoch_k"], prev["stoch_d"], curr["vol_ma20"]]
    if any(pd.isna(v) for v in values):
        return None

    close = curr["close"]
    ema20, ema50, ema200 = curr["ema20"], curr["ema50"], curr["ema200"]
    k_curr, d_curr = curr["stoch_k"], curr["stoch_d"]
    k_prev, d_prev = prev["stoch_k"], prev["stoch_d"]
    vol_curr, vol_ma = curr["volume"], curr["vol_ma20"]

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
# 6. OUTPUT (log + Discord)
# ==========================================
def fmt_price(p):
    return f"{p:.6g}"


def format_signal(idx, item):
    return (
        f"**#{idx} {item['name']} ({item['symbol']})** | sumber data: {item['source']}\n"
        f"Posisi: {item['type']} | {item['grade']}\n"
        f"OP: ${fmt_price(item['op'])} | TP 2%: ${fmt_price(item['tp'])} | SL 1%: ${fmt_price(item['sl'])}\n\n"
    )


def split_message(text, limit=1900):
    """Discord membatasi 2000 karakter per pesan."""
    chunks, buf = [], ""
    for line in text.splitlines(keepends=True):
        if len(buf) + len(line) > limit:
            chunks.append(buf)
            buf = ""
        buf += line
    if buf:
        chunks.append(buf)
    return chunks


def send_discord(text):
    """Kirim ke webhook. Return True jika semua bagian terkirim."""
    if not DISCORD_WEBHOOK_URL:
        log.error("DISCORD_WEBHOOK_URL kosong.")
        return False

    ok = True
    for chunk in split_message(text):
        try:
            r = requests.post(DISCORD_WEBHOOK_URL, json={"content": chunk}, timeout=HTTP_TIMEOUT)
            if r.status_code == 429:
                time.sleep(float(r.json().get("retry_after", 1)) + 0.5)
                r = requests.post(DISCORD_WEBHOOK_URL, json={"content": chunk}, timeout=HTTP_TIMEOUT)
            if r.status_code >= 300:
                log.error("Discord menolak pesan: HTTP %s %s", r.status_code, r.text[:150])
                ok = False
        except requests.RequestException as e:
            log.error("Gagal menghubungi Discord: %s", e)
            ok = False
        time.sleep(1)
    return ok


def notify_error(msg):
    log.error(msg)
    send_discord(f"**SCOUTBOT ERROR** {msg}")


# ==========================================
# 7. MAIN (satu siklus, lalu exit)
# ==========================================
def run_scoutbot():
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    log.info("SCOUTBOT RUNNING - %s", now)

    if not DISCORD_WEBHOOK_URL:
        log.error("Secret DISCORD_WEBHOOK_URL kosong. Cek nama dan isi secret di GitHub.")
        return 1

    log.info("[1/4] Mengambil daftar pair Pionex...")
    pionex_pairs = get_pionex_pairs()
    use_pionex = bool(pionex_pairs)
    if use_pionex:
        log.info("Pair Pionex: %s", len(pionex_pairs))
    else:
        log.warning("Pionex tidak dapat dijangkau. Analisis memakai Binance untuk semua data.")

    log.info("[2/4] Mengambil Top %s koin dari CoinGecko...", TOP_N)
    top = get_top_coins()
    if not top:
        notify_error("CoinGecko tidak merespons. Cek rate limit atau COINGECKO_API_KEY.")
        return 1

    log.info("[3/4] Analisis indikator...")
    candidates, seen = [], set()
    stats = {"no_pair": 0, "no_data": 0, "Pionex": 0, "Binance": 0}

    for coin in top:
        base = coin["symbol"].upper()
        if base in seen:
            continue
        if use_pionex and f"{base}_USDT" not in pionex_pairs:
            stats["no_pair"] += 1
            continue
        seen.add(base)

        df, source = get_candles(base, use_pionex)
        if df is None:
            stats["no_data"] += 1
            time.sleep(REQUEST_DELAY)
            continue

        stats[source] += 1
        signal = analyze_signal(calculate_indicators(df))
        if signal:
            signal["symbol"] = f"{base}_USDT"
            signal["name"] = coin["name"]
            signal["source"] = source
            candidates.append(signal)

        time.sleep(REQUEST_DELAY)

    log.info("Hasil: dianalisis=%s | tanpa pair=%s | data kosong=%s | Pionex=%s | Binance=%s | sinyal=%s",
             len(seen), stats["no_pair"], stats["no_data"],
             stats["Pionex"], stats["Binance"], len(candidates))

    if not seen:
        notify_error("Tidak ada koin dari CoinGecko yang punya pair USDT di Pionex.")
        return 1

    if stats["Pionex"] == 0 and stats["Binance"] == 0:
        notify_error("Data candle gagal diambil dari Pionex maupun Binance. "
                     "Cek log untuk detail. Kemungkinan jaringan runner diblokir atau endpoint berubah.")
        return 1

    candidates.sort(key=lambda x: x["score"], reverse=True)
    candidates = candidates[:MAX_SIGNALS]

    log.info("[4/4] Mengirim hasil...")
    header = (f"**SCOUTBOT {now}** | dianalisis: {len(seen)} koin | "
              f"Pionex: {stats['Pionex']} | Binance: {stats['Binance']}\n\n")
    if candidates:
        msg = header + "".join(format_signal(i, c) for i, c in enumerate(candidates, 1))
    else:
        msg = header + "Tidak ada koin yang memenuhi kriteria Grade A+ pada siklus ini."

    print(msg)
    return 0 if send_discord(msg) else 1


if __name__ == "__main__":
    sys.exit(run_scoutbot())
