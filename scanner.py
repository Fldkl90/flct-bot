#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FLCT v5 Tarayici - Binance USDT Perpetual Futures (Pine v5 "FLCT v5 with Divergence" portu)
============================================================================================
Tek seferlik calisir (surekli bot DEGIL): tarar, sinyal varsa Telegram'a atar, cikar.
GitHub Actions cron ile tetiklenmek uzere tasarlandi.

CEKIRDEK SINYAL (Pine longcon/shortcon ile birebir):
  LONG  = (lmBull & TDDn==9) | (lmBull[1] & TDDn==9) | (lmBull & TDDn[1]==9)
  SHORT = (lmBear & TDUp==9) | (lmBear[1] & TDUp==9) | (lmBear & TDUp[1]==9)
  yani LeLedc dip/tepe + TD9, ayni mumda veya +-1 bar arayla.

DIVERGENCE TEYIDI (cekirdek saglaninca eklenen filtre):
  LONG  -> son DIV_LOOKBACK mumda pozitif regular divergence sayisi esigi gecmeli
  SHORT -> son DIV_LOOKBACK mumda negatif regular divergence sayisi esigi gecmeli
  DIV_MODE=perbar  -> mumlardan BIRINDE sayi >= MIN_DIV  (default; grafik sayi etiketi gibi)
  DIV_MODE=cumulative -> 3 muma yayilmis TOPLAM >= MIN_DIV
  Divergence 10 indikatorde aranir: MACD, Hist, RSI, Stoch, CCI, MOM, OBV, VWMACD, CMF, MFI
"""

import os
import time
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import numpy as np
import pandas as pd

# ============================ AYARLAR (env ile degistirilebilir) ============================
TELEGRAM_TOKEN     = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID", "")
TIMEFRAME          = os.environ.get("TIMEFRAME", "1h")
MIN_VOLUME_USDT    = float(os.environ.get("MIN_VOLUME_USDT", "50000000"))
KLINE_LIMIT        = int(os.environ.get("KLINE_LIMIT", "300"))
MAX_WORKERS        = int(os.environ.get("MAX_WORKERS", "5"))
MAX_CANDLE_AGE_MIN = float(os.environ.get("MAX_CANDLE_AGE_MIN", "20"))
FORCE_RUN          = os.environ.get("FORCE_RUN", "false").lower() == "true"
SEND_HEARTBEAT     = os.environ.get("SEND_HEARTBEAT", "false").lower() == "true"

# Divergence teyit ayarlari
DIV_LOOKBACK = int(os.environ.get("DIV_LOOKBACK", "3"))     # son kac mum
MIN_DIV      = int(os.environ.get("MIN_DIV", "3"))          # minimum divergence sayisi
DIV_MODE     = os.environ.get("DIV_MODE", "perbar").lower() # "perbar" | "cumulative"

# Cekirdek strateji secimi
CORE_MODE = os.environ.get("CORE_MODE", "wavetrend").lower()  # "wavetrend" | "td9"
WT_N1 = int(os.environ.get("WT_N1", "10"))
WT_N2 = int(os.environ.get("WT_N2", "21"))
OB_LEVEL = float(os.environ.get("OB_LEVEL", "50"))
OS_LEVEL = float(os.environ.get("OS_LEVEL", "-50"))

# Strateji parametreleri (Pine defaultlari)
PRD       = 5
MAXPP     = 10
MAXBARS   = 100
# LeLedc
MAJ_QUAL      = 6
MAJ_LEN       = 30
CLOSE_VAL     = 4

BINANCE = "https://fapi.binance.com"

NAME_MAP = {"macd": "MACD", "hist": "Hist", "rsi": "RSI", "stoch": "Stoch",
            "cci": "CCI", "mom": "MOM", "obv": "OBV", "vwmacd": "VWMACD",
            "cmf": "CMF", "mfi": "MFI"}

# ============================ INDIKATORLER (Pine ta.* ile esitlenmis) ============================
def _ema(s, n): return s.ewm(span=n, adjust=False).mean()
def _rma(s, n): return s.ewm(alpha=1.0 / n, adjust=False).mean()
def _sma(s, n): return s.rolling(n).mean()

def macd_calc(c):
    line = _ema(c, 12) - _ema(c, 26)
    hist = line - _ema(line, 9)
    return line.values, hist.values

def rsi_calc(c):
    d = c.diff()
    rs = _rma(d.clip(lower=0), 14) / _rma((-d).clip(lower=0), 14)
    return (100 - 100 / (1 + rs)).values

def mom_calc(c):  return (c - c.shift(10)).values

def cci_calc(c):
    ma = _sma(c, 10)
    md = (c - ma).abs().rolling(10).mean()
    return ((c - ma) / (0.015 * md)).values

def obv_calc(c, v):
    return (np.sign(c.diff().fillna(0)) * v).cumsum().values

def stoch_calc(c, h, l):
    ll = l.rolling(14).min(); hh = h.rolling(14).max()
    return _sma(100 * (c - ll) / (hh - ll), 3).values

def vwmacd_calc(c, v):
    fast = (c * v).rolling(12).sum() / v.rolling(12).sum()
    slow = (c * v).rolling(26).sum() / v.rolling(26).sum()
    return (fast - slow).values

def cmf_calc(c, h, l, v):
    rng = (h - l).replace(0, np.nan)
    mfv = (((c - l) - (h - c)) / rng) * v
    return (_sma(mfv, 21) / _sma(v, 21)).values

def mfi_calc(c, v):
    ch = c.diff(); raw = v * c
    us = pd.Series(np.where(ch <= 0, 0.0, raw), index=c.index).rolling(14).sum()
    ls = pd.Series(np.where(ch >= 0, 0.0, raw), index=c.index).rolling(14).sum()
    return (100 - 100 / (1 + us / ls)).values

# ============================ PIVOT & DIVERGENCE ============================
def find_pivots(arr, left, right, kind):
    n = len(arr); out = []
    for i in range(left, n - right):
        c = arr[i]; ls = arr[i - left:i]; rs = arr[i + 1:i + right + 1]
        if kind == "low":
            if c < ls.min() and c < rs.min():
                out.append((i + right, c))
        else:
            if c > ls.max() and c > rs.max():
                out.append((i + right, c))
    return out

def pos_reg_div_len(ind, close, idx, pls):
    if np.isnan(ind[idx]) or np.isnan(ind[idx - 1]):
        return 0
    if not (ind[idx] > ind[idx - 1] or close[idx] > close[idx - 1]):
        return 0
    sp = 1; checked = 0
    for (conf_idx, pv) in pls:
        if checked >= MAXPP: break
        checked += 1
        length = idx - conf_idx + PRD
        if length > MAXBARS: break
        if length <= 5: continue
        j = idx - length
        if j < 0 or np.isnan(ind[j]): continue
        if not (ind[idx - 1] > ind[j] and close[idx - 1] < pv): continue
        denom = length - sp
        s1 = (ind[idx - sp] - ind[j]) / denom; s2 = (close[idx - sp] - close[j]) / denom
        v1 = ind[idx - sp] - s1; v2 = close[idx - sp] - s2
        ok = True
        for y in range(1 + sp, length):
            if ind[idx - y] < v1 or close[idx - y] < v2:
                ok = False; break
            v1 -= s1; v2 -= s2
        if ok: return length
    return 0

def neg_reg_div_len(ind, close, idx, phs):
    if np.isnan(ind[idx]) or np.isnan(ind[idx - 1]):
        return 0
    if not (ind[idx] < ind[idx - 1] or close[idx] < close[idx - 1]):
        return 0
    sp = 1; checked = 0
    for (conf_idx, pv) in phs:
        if checked >= MAXPP: break
        checked += 1
        length = idx - conf_idx + PRD
        if length > MAXBARS: break
        if length <= 5: continue
        j = idx - length
        if j < 0 or np.isnan(ind[j]): continue
        if not (ind[idx - 1] < ind[j] and close[idx - 1] > pv): continue
        denom = length - sp
        s1 = (ind[idx - sp] - ind[j]) / denom; s2 = (close[idx - sp] - close[j]) / denom
        v1 = ind[idx - sp] - s1; v2 = close[idx - sp] - s2
        ok = True
        for y in range(1 + sp, length):
            if ind[idx - y] > v1 or close[idx - y] > v2:
                ok = False; break
            v1 -= s1; v2 -= s2
        if ok: return length
    return 0

def pos_div_names(i, pls_all, inds, close):
    pls = sorted([p for p in pls_all if p[0] <= i], key=lambda p: -p[0])
    return [name for name, arr in inds.items() if pos_reg_div_len(arr, close, i, pls) > 0]

def neg_div_names(i, phs_all, inds, close):
    phs = sorted([p for p in phs_all if p[0] <= i], key=lambda p: -p[0])
    return [name for name, arr in inds.items() if neg_reg_div_len(arr, close, i, phs) > 0]

# ============================ LeLedc & TD9 ============================
def lele_calc(o, h, l, c):
    n = len(c); major = np.zeros(n); b_arr = np.zeros(n); s_arr = np.zeros(n)
    for i in range(n):
        b = b_arr[i - 1] if i >= 1 else 0
        s = s_arr[i - 1] if i >= 1 else 0
        ret = 0
        if i >= CLOSE_VAL and c[i] > c[i - CLOSE_VAL]: b += 1
        if i >= CLOSE_VAL and c[i] < c[i - CLOSE_VAL]: s += 1
        if i >= MAJ_LEN - 1:
            hh = h[i - MAJ_LEN + 1:i + 1].max(); ll = l[i - MAJ_LEN + 1:i + 1].min()
            if b > MAJ_QUAL and c[i] < o[i] and h[i] >= hh: b = 0; ret = -1
            if s > MAJ_QUAL and c[i] > o[i] and l[i] <= ll: s = 0; ret = 1
        b_arr[i] = b; s_arr[i] = s; major[i] = ret
    return major

def wavetrend_calc(h, l, c):
    # Joy_Bangla FLCT WaveTrend: ap=hlc3, esa=ema(ap,n1), d=ema(|ap-esa|,n1),
    # ci=(ap-esa)/(0.015*d), tci=ema(ci,n2); wt1=tci
    ap = (h + l + c) / 3.0
    esa = _ema(ap, WT_N1)
    d = _ema((ap - esa).abs(), WT_N1)
    ci = (ap - esa) / (0.015 * d)
    return _ema(ci, WT_N2).values

def td9_calc(c):
    n = len(c); TD = np.zeros(n, dtype=int); TS = np.zeros(n, dtype=int)
    for i in range(n):
        TD[i] = (TD[i - 1] + 1) if (i >= 4 and c[i] > c[i - 4]) else 0
        TS[i] = (TS[i - 1] + 1) if (i >= 4 and c[i] < c[i - 4]) else 0
    return TD, TS

# ============================ SINYAL DEGERLENDIRME ============================
def _div_pass(counts):
    return (sum(counts) >= MIN_DIV) if DIV_MODE == "cumulative" else (max(counts) >= MIN_DIV)

def eval_signals(df):
    c = df["close"].astype(float); h = df["high"].astype(float); l = df["low"].astype(float)
    v = df["volume"].astype(float); o = df["open"].astype(float).values
    close = c.values; high = h.values; low = l.values
    n = len(close)
    if n < 150:
        return (False, False, [], [], 0)

    # --- Cekirdek (ucuz): CORE_MODE'a gore ---
    major = lele_calc(o, high, low, close)
    idx = n - 1
    lmBull0, lmBull1 = major[idx] == 1, major[idx - 1] == 1
    lmBear0, lmBear1 = major[idx] == -1, major[idx - 1] == -1

    if CORE_MODE == "td9":
        # FLCT v5: lmBull + TDDn==9 (+-1 bar) / lmBear + TDUp==9 (+-1 bar)
        TD, TS = td9_calc(close)
        TDDn0, TDDn1 = TS[idx] == 9, TS[idx - 1] == 9
        TDUp0, TDUp1 = TD[idx] == 9, TD[idx - 1] == 9
        longcore  = (lmBull0 and TDDn0) or (lmBull1 and TDDn0) or (lmBull0 and TDDn1)
        shortcore = (lmBear0 and TDUp0) or (lmBear1 and TDUp0) or (lmBear0 and TDUp1)
    else:
        # Joy_Bangla FLCT: lmBull + wt1<OS (long), wt1>OB (short)
        wt = wavetrend_calc(h, l, c)
        longcore  = bool(lmBull0 and wt[idx] < OS_LEVEL)
        shortcore = bool(wt[idx] > OB_LEVEL)

    if not (longcore or shortcore):
        return (False, False, [], [], int(major[idx]))

    # --- Divergence teyidi (sadece cekirdek varsa hesapla) ---
    inds = {}
    inds["macd"], inds["hist"] = macd_calc(c)
    inds["rsi"]   = rsi_calc(c)
    inds["stoch"] = stoch_calc(c, h, l)
    inds["cci"]   = cci_calc(c)
    inds["mom"]   = mom_calc(c)
    inds["obv"]   = obv_calc(c, v)
    inds["vwmacd"]= vwmacd_calc(c, v)
    inds["cmf"]   = cmf_calc(c, h, l, v)
    inds["mfi"]   = mfi_calc(c, v)

    pls_all = find_pivots(close, PRD, PRD, "low")
    phs_all = find_pivots(close, PRD, PRD, "high")

    longsig = shortsig = False
    li = si = []
    if longcore:
        bars = [pos_div_names(idx - k, pls_all, inds, close) for k in range(DIV_LOOKBACK)]
        counts = [len(b) for b in bars]
        if _div_pass(counts):
            longsig = True
            li = bars[int(np.argmax(counts))]
    if shortcore:
        bars = [neg_div_names(idx - k, phs_all, inds, close) for k in range(DIV_LOOKBACK)]
        counts = [len(b) for b in bars]
        if _div_pass(counts):
            shortsig = True
            si = bars[int(np.argmax(counts))]

    return (longsig, shortsig, li, si, int(major[idx]))

# ============================ BINANCE & TELEGRAM ============================
def _get(url, params=None, tries=4):
    for k in range(tries):
        r = requests.get(url, params=params, timeout=25)
        if r.status_code in (429, 418):
            time.sleep(2 * (k + 1)); continue
        r.raise_for_status(); return r
    r.raise_for_status()

def get_symbols():
    info = _get(BINANCE + "/fapi/v1/exchangeInfo").json()
    return [s["symbol"] for s in info["symbols"]
            if s.get("contractType") == "PERPETUAL" and s.get("status") == "TRADING"
            and s.get("quoteAsset") == "USDT"]

def get_volumes():
    return {d["symbol"]: float(d["quoteVolume"]) for d in _get(BINANCE + "/fapi/v1/ticker/24hr").json()}

def get_klines(sym):
    raw = _get(BINANCE + "/fapi/v1/klines",
               params={"symbol": sym, "interval": TIMEFRAME, "limit": KLINE_LIMIT}).json()
    if not isinstance(raw, list) or len(raw) < 150:
        return None
    df = pd.DataFrame(raw, columns=["t", "open", "high", "low", "close", "volume",
                                    "ct", "qv", "n", "tb", "tq", "ig"])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df["ct"] = df["ct"].astype("int64")
    return df.iloc[:-1].reset_index(drop=True)  # olusan (kapanmamis) mumu at

def send_telegram(text):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("[UYARI] Telegram bilgileri yok. Mesaj:\n" + text); return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    while text:
        chunk, text = text[:3800], text[3800:]
        try:
            requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": chunk,
                                     "parse_mode": "HTML", "disable_web_page_preview": True}, timeout=25)
        except Exception as e:
            print("[Telegram hata]", e)

# ============================ ANA AKIS ============================
def scan_symbol(sym):
    try:
        df = get_klines(sym)
        if df is None or len(df) < 150:
            return None
        longc, shortc, li, si, _ = eval_signals(df)
        if longc:  return ("LONG", sym, [NAME_MAP[x] for x in li])
        if shortc: return ("SHORT", sym, [NAME_MAP[x] for x in si])
    except Exception as e:
        print(f"[hata] {sym}: {e}")
    return None

def main():
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    if not FORCE_RUN:
        probe = get_klines("BTCUSDT")
        if probe is not None:
            age = (now_ms - int(probe["ct"].iloc[-1])) / 60000.0
            if age > MAX_CANDLE_AGE_MIN:
                print(f"Yeni kapanan mum yok (yas {age:.1f} dk). Cikiliyor."); return

    syms = get_symbols()
    vols = get_volumes()
    syms = [s for s in syms if vols.get(s, 0) >= MIN_VOLUME_USDT]
    print(f"{len(syms)} sembol taraniyor | TF={TIMEFRAME} | CORE={CORE_MODE} | DIV_MODE={DIV_MODE} MIN_DIV={MIN_DIV} LOOKBACK={DIV_LOOKBACK}")

    longs, shorts = [], []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for f in as_completed({ex.submit(scan_symbol, s): s for s in syms}):
            res = f.result()
            if res:
                side, sym, names = res
                (longs if side == "LONG" else shorts).append((sym, names))

    longs.sort(); shorts.sort()
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    if not longs and not shorts:
        print("Sinyal yok.")
        if SEND_HEARTBEAT:
            send_telegram(f"🟦 FLCT v5 | TF {TIMEFRAME} | {ts}\nTarama tamam, sinyal yok.")
        return

    core_tag = "TD9" if CORE_MODE == "td9" else "WT"
    lines = [f"🔔 <b>FLCT Sinyal</b> ({core_tag}) | TF {TIMEFRAME} | {ts}", ""]
    if longs:
        lines.append(f"🟢 <b>LONG</b> ({len(longs)})")
        for sym, names in longs:
            lines.append(f"• <b>{sym}</b> — FLCT/{core_tag} | div({len(names)}): {', '.join(names)}")
        lines.append("")
    if shorts:
        lines.append(f"🔴 <b>SHORT</b> ({len(shorts)})")
        for sym, names in shorts:
            lines.append(f"• <b>{sym}</b> — FLCT/{core_tag} | div({len(names)}): {', '.join(names)}")

    msg = "\n".join(lines)
    print(msg)
    send_telegram(msg)

if __name__ == "__main__":
    main()
