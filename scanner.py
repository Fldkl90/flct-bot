#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FLCT v5 Multi-TF Telegram Bot
==============================
crontab ile her 5 dakikada bir tetiklenir.
5m / 15m / 30m / 45m / 1h / 2h / 3h / 4h / 1d / 1w / 1M
Her TF için yeni kapanan mum varsa tarar, sinyal varsa Telegram'a atar.
Aynı mum için tekrar sinyal gönderilmez (state dosyası ile dedup).
"""

import os, json, time
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import numpy as np
import pandas as pd

# ── Config (.env veya ortam değişkenleri) ─────────────────────────────────────
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
MIN_VOLUME_USDT  = float(os.environ.get("MIN_VOLUME_USDT", "50000000"))
MAX_WORKERS      = int(os.environ.get("MAX_WORKERS", "5"))
DIV_LOOKBACK     = int(os.environ.get("DIV_LOOKBACK", "3"))
MIN_DIV          = int(os.environ.get("MIN_DIV", "3"))
DIV_MODE         = os.environ.get("DIV_MODE", "perbar").lower()
STATE_FILE       = Path(os.environ.get("STATE_FILE", "/home/ubuntu/flct_state.json"))

# Aktif TF listesi — istediğini çıkar
ENABLED_TFS = ["5m", "15m", "30m", "45m", "1h", "2h", "3h", "4h", "1d", "1w", "1M"]

# TF → (binance_interval, limit, merge_n)
# 45m = 15m * 3, 2h = 1h * 2, 3h = 1h * 3
TF_PARAMS = {
    "5m":  ("5m",  300, 1),
    "15m": ("15m", 300, 1),
    "30m": ("30m", 300, 1),
    "45m": ("15m", 900, 3),
    "1h":  ("1h",  300, 1),
    "2h":  ("1h",  600, 2),
    "3h":  ("1h",  900, 3),
    "4h":  ("4h",  300, 1),
    "1d":  ("1d",  300, 1),
    "1w":  ("1w",  200, 1),
    "1M":  ("1M",  100, 1),
}

BINANCE  = "https://fapi.binance.com"
PRD      = 5; MAXPP = 10; MAXBARS = 100
MAJ_QUAL = 6; MAJ_LEN = 30; CLOSE_VAL = 4

NAME_MAP = {
    "macd":"MACD","hist":"Hist","rsi":"RSI","stoch":"Stoch","cci":"CCI",
    "mom":"MOM","obv":"OBV","vwmacd":"VWMACD","cmf":"CMF","mfi":"MFI"
}

TF_EMOJI = {
    "5m":"⚡","15m":"🕐","30m":"🕑","45m":"🕒",
    "1h":"🕓","2h":"🕔","3h":"🕕","4h":"🕖",
    "1d":"📅","1w":"📆","1M":"🗓"
}

# ── İndikatörler ──────────────────────────────────────────────────────────────
def _ema(s, n): return s.ewm(span=n, adjust=False).mean()
def _rma(s, n): return s.ewm(alpha=1.0/n, adjust=False).mean()
def _sma(s, n): return s.rolling(n).mean()

def macd_calc(c):
    line = _ema(c,12) - _ema(c,26)
    return line.values, (line - _ema(line,9)).values

def rsi_calc(c):
    d = c.diff()
    rs = _rma(d.clip(lower=0),14) / _rma((-d).clip(lower=0),14)
    return (100 - 100/(1+rs)).values

def mom_calc(c): return (c - c.shift(10)).values

def cci_calc(c):
    ma = _sma(c,10); md = (c-ma).abs().rolling(10).mean()
    return ((c-ma)/(0.015*md)).values

def obv_calc(c,v):
    return (np.sign(c.diff().fillna(0))*v).cumsum().values

def stoch_calc(c,h,l):
    ll = l.rolling(14).min(); hh = h.rolling(14).max()
    return _sma(100*(c-ll)/(hh-ll),3).values

def vwmacd_calc(c,v):
    fast = (c*v).rolling(12).sum()/v.rolling(12).sum()
    slow = (c*v).rolling(26).sum()/v.rolling(26).sum()
    return (fast-slow).values

def cmf_calc(c,h,l,v):
    rng = (h-l).replace(0,np.nan)
    mfv = (((c-l)-(h-c))/rng)*v
    return (_sma(mfv,21)/_sma(v,21)).values

def mfi_calc(c,v):
    ch = c.diff(); raw = v*c
    us = pd.Series(np.where(ch<=0,0.0,raw),index=c.index).rolling(14).sum()
    ls = pd.Series(np.where(ch>=0,0.0,raw),index=c.index).rolling(14).sum()
    return (100 - 100/(1+us/ls)).values

# ── Pivot & Divergence ────────────────────────────────────────────────────────
def find_pivots(arr, left, right, kind):
    n = len(arr); out = []
    for i in range(left, n-right):
        c = arr[i]; ls = arr[i-left:i]; rs = arr[i+1:i+right+1]
        if kind=="low":
            if c < ls.min() and c < rs.min(): out.append((i+right, c))
        else:
            if c > ls.max() and c > rs.max(): out.append((i+right, c))
    return out

def pos_reg_div_len(ind, close, idx, pls):
    if np.isnan(ind[idx]) or np.isnan(ind[idx-1]): return 0
    if not (ind[idx]>ind[idx-1] or close[idx]>close[idx-1]): return 0
    sp=1; checked=0
    for (conf_idx,pv) in pls:
        if checked>=MAXPP: break
        checked+=1
        length = idx-conf_idx+PRD
        if length>MAXBARS: break
        if length<=5: continue
        j = idx-length
        if j<0 or np.isnan(ind[j]): continue
        if not (ind[idx-1]>ind[j] and close[idx-1]<pv): continue
        denom=length-sp
        s1=(ind[idx-sp]-ind[j])/denom; s2=(close[idx-sp]-close[j])/denom
        v1=ind[idx-sp]-s1; v2=close[idx-sp]-s2; ok=True
        for y in range(1+sp,length):
            if ind[idx-y]<v1 or close[idx-y]<v2: ok=False; break
            v1-=s1; v2-=s2
        if ok: return length
    return 0

def neg_reg_div_len(ind, close, idx, phs):
    if np.isnan(ind[idx]) or np.isnan(ind[idx-1]): return 0
    if not (ind[idx]<ind[idx-1] or close[idx]<close[idx-1]): return 0
    sp=1; checked=0
    for (conf_idx,pv) in phs:
        if checked>=MAXPP: break
        checked+=1
        length = idx-conf_idx+PRD
        if length>MAXBARS: break
        if length<=5: continue
        j = idx-length
        if j<0 or np.isnan(ind[j]): continue
        if not (ind[idx-1]<ind[j] and close[idx-1]>pv): continue
        denom=length-sp
        s1=(ind[idx-sp]-ind[j])/denom; s2=(close[idx-sp]-close[j])/denom
        v1=ind[idx-sp]-s1; v2=close[idx-sp]-s2; ok=True
        for y in range(1+sp,length):
            if ind[idx-y]>v1 or close[idx-y]>v2: ok=False; break
            v1-=s1; v2-=s2
        if ok: return length
    return 0

def pos_div_names(i, pls_all, inds, close):
    pls = sorted([p for p in pls_all if p[0]<=i], key=lambda p:-p[0])
    return [k for k,arr in inds.items() if pos_reg_div_len(arr,close,i,pls)>0]

def neg_div_names(i, phs_all, inds, close):
    phs = sorted([p for p in phs_all if p[0]<=i], key=lambda p:-p[0])
    return [k for k,arr in inds.items() if neg_reg_div_len(arr,close,i,phs)>0]

# ── LeLedc & TD9 ──────────────────────────────────────────────────────────────
def lele_calc(o,h,l,c):
    n=len(c); major=np.zeros(n); b_arr=np.zeros(n); s_arr=np.zeros(n)
    for i in range(n):
        b = b_arr[i-1] if i>=1 else 0
        s = s_arr[i-1] if i>=1 else 0
        ret=0
        if i>=CLOSE_VAL and c[i]>c[i-CLOSE_VAL]: b+=1
        if i>=CLOSE_VAL and c[i]<c[i-CLOSE_VAL]: s+=1
        if i>=MAJ_LEN-1:
            hh=h[i-MAJ_LEN+1:i+1].max(); ll=l[i-MAJ_LEN+1:i+1].min()
            if b>MAJ_QUAL and c[i]<o[i] and h[i]>=hh: b=0; ret=-1
            if s>MAJ_QUAL and c[i]>o[i] and l[i]<=ll: s=0; ret=1
        b_arr[i]=b; s_arr[i]=s; major[i]=ret
    return major

def td9_calc(c):
    n=len(c); TD=np.zeros(n,dtype=int); TS=np.zeros(n,dtype=int)
    for i in range(n):
        TD[i] = (TD[i-1]+1) if (i>=4 and c[i]>c[i-4]) else 0
        TS[i] = (TS[i-1]+1) if (i>=4 and c[i]<c[i-4]) else 0
    return TD,TS

# ── Sinyal değerlendirme ──────────────────────────────────────────────────────
def _div_pass(counts):
    return (sum(counts)>=MIN_DIV) if DIV_MODE=="cumulative" else (max(counts)>=MIN_DIV)

def eval_signals(df):
    c=df["close"].astype(float); h=df["high"].astype(float)
    l=df["low"].astype(float); v=df["volume"].astype(float)
    o=df["open"].astype(float).values
    close=c.values; high=h.values; low=l.values
    n=len(close)
    if n<150: return (False,False,[],[])

    major=lele_calc(o,high,low,close)
    idx=n-1
    lmBull0,lmBull1 = major[idx]==1, major[idx-1]==1
    lmBear0,lmBear1 = major[idx]==-1, major[idx-1]==-1

    TD,TS=td9_calc(close)
    TDDn0,TDDn1 = TS[idx]==9, TS[idx-1]==9
    TDUp0,TDUp1 = TD[idx]==9, TD[idx-1]==9

    longcore  = (lmBull0 and TDDn0) or (lmBull1 and TDDn0) or (lmBull0 and TDDn1)
    shortcore = (lmBear0 and TDUp0) or (lmBear1 and TDUp0) or (lmBear0 and TDUp1)

    if not (longcore or shortcore): return (False,False,[],[])

    inds={}
    inds["macd"],inds["hist"] = macd_calc(c)
    inds["rsi"]    = rsi_calc(c)
    inds["stoch"]  = stoch_calc(c,h,l)
    inds["cci"]    = cci_calc(c)
    inds["mom"]    = mom_calc(c)
    inds["obv"]    = obv_calc(c,v)
    inds["vwmacd"] = vwmacd_calc(c,v)
    inds["cmf"]    = cmf_calc(c,h,l,v)
    inds["mfi"]    = mfi_calc(c,v)

    pls_all=find_pivots(close,PRD,PRD,"low")
    phs_all=find_pivots(close,PRD,PRD,"high")

    longsig=shortsig=False; li=si=[]
    if longcore:
        bars=[pos_div_names(idx-k,pls_all,inds,close) for k in range(DIV_LOOKBACK)]
        counts=[len(b) for b in bars]
        if _div_pass(counts):
            longsig=True; li=bars[int(np.argmax(counts))]
    if shortcore:
        bars=[neg_div_names(idx-k,phs_all,inds,close) for k in range(DIV_LOOKBACK)]
        counts=[len(b) for b in bars]
        if _div_pass(counts):
            shortsig=True; si=bars[int(np.argmax(counts))]

    return (longsig,shortsig,li,si)

# ── Binance ───────────────────────────────────────────────────────────────────
def _get(url, params=None, tries=4):
    for k in range(tries):
        r=requests.get(url,params=params,timeout=25)
        if r.status_code in (429,418): time.sleep(2*(k+1)); continue
        r.raise_for_status(); return r
    r.raise_for_status()

def get_symbols():
    info=_get(BINANCE+"/fapi/v1/exchangeInfo").json()
    return [s["symbol"] for s in info["symbols"]
            if s.get("contractType")=="PERPETUAL" and s.get("status")=="TRADING"
            and s.get("quoteAsset")=="USDT"]

def get_volumes():
    return {d["symbol"]:float(d["quoteVolume"])
            for d in _get(BINANCE+"/fapi/v1/ticker/24hr").json()}

def merge_nx(klines, n):
    """n mumu birleştir (45m, 2h, 3h için)"""
    out=[]; rem=len(klines)%n
    for i in range(rem, len(klines)-n+1, n):
        g=klines[i:i+n]; f=g[0]; last=g[-1]
        mxH=max(float(x[2]) for x in g)
        mnL=min(float(x[3]) for x in g)
        sv5=sum(float(x[5]) for x in g)
        sv7=sum(float(x[7]) for x in g)
        out.append([f[0],f[1],str(mxH),str(mnL),last[4],str(sv5),last[6],str(sv7)])
    return out

def get_klines(sym, tf):
    api_tf, limit, merge = TF_PARAMS[tf]
    raw=_get(BINANCE+"/fapi/v1/klines",
             params={"symbol":sym,"interval":api_tf,"limit":limit}).json()
    if not isinstance(raw,list) or len(raw)<20: return None
    if merge>1: raw=merge_nx(raw,merge)
    if len(raw)<20: return None
    # Son mumu at (henüz kapanmamış)
    raw=raw[:-1]
    cols=["t","open","high","low","close","volume","ct","qv","n","tb","tq","ig"]
    df=pd.DataFrame(raw,columns=cols[:len(raw[0])])
    for col in ["open","high","low","close","volume"]:
        df[col]=df[col].astype(float)
    # ct (close time) son kolondan al
    if len(raw[0])>=7:
        df["ct"]=pd.Series([int(x[6]) for x in raw])
    return df

def get_last_close_time(sym, tf):
    """Bir sembolün son kapanan mum zamanını döner (ms)."""
    try:
        df=get_klines(sym,tf)
        if df is None or "ct" not in df.columns: return None
        return int(df["ct"].iloc[-1])
    except: return None

# ── State dosyası ─────────────────────────────────────────────────────────────
def load_state():
    try:
        if STATE_FILE.exists():
            return json.loads(STATE_FILE.read_text())
    except: pass
    return {}

def save_state(state):
    try: STATE_FILE.write_text(json.dumps(state))
    except Exception as e: print(f"[state hata] {e}")

# ── Telegram ──────────────────────────────────────────────────────────────────
def send_telegram(text):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("[UYARI] Telegram bilgileri yok.\n"+text); return
    url=f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    while text:
        chunk,text=text[:3800],text[3800:]
        try:
            requests.post(url,data={"chat_id":TELEGRAM_CHAT_ID,"text":chunk,
                                    "parse_mode":"HTML","disable_web_page_preview":True},timeout=25)
        except Exception as e: print(f"[Telegram hata] {e}")

# ── Tek sembol tarama ─────────────────────────────────────────────────────────
def scan_symbol(sym, tf):
    try:
        df=get_klines(sym,tf)
        if df is None or len(df)<150: return None
        longc,shortc,li,si=eval_signals(df)
        if longc: return ("LONG",sym,[NAME_MAP.get(x,x) for x in li])
        if shortc: return ("SHORT",sym,[NAME_MAP.get(x,x) for x in si])
    except Exception as e:
        print(f"[hata] {sym}/{tf}: {e}")
    return None

# ── TF tarama ─────────────────────────────────────────────────────────────────
def scan_tf(tf, syms):
    longs=[]; shorts=[]
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs={ex.submit(scan_symbol,s,tf):s for s in syms}
        for f in as_completed(futs):
            res=f.result()
            if res:
                side,sym,names=res
                (longs if side=="LONG" else shorts).append((sym,names))
    longs.sort(); shorts.sort()
    return longs,shorts

# ── Mesaj formatla ────────────────────────────────────────────────────────────
def build_message(tf, longs, shorts):
    emoji=TF_EMOJI.get(tf,"🔔")
    ts=datetime.now(timezone.utc).strftime("%H:%M UTC")
    lines=[f"{emoji} <b>FLCT v5 | {tf.upper()}</b> | {ts}",""]
    if longs:
        lines.append(f"🟢 <b>LONG ({len(longs)})</b>")
        for sym,names in longs:
            sym_clean=sym.replace("USDT","")
            div_str=", ".join(names) if names else "—"
            lines.append(f"• <b>{sym_clean}</b> | div: {div_str}")
        lines.append("")
    if shorts:
        lines.append(f"🔴 <b>SHORT ({len(shorts)})</b>")
        for sym,names in shorts:
            sym_clean=sym.replace("USDT","")
            div_str=", ".join(names) if names else "—"
            lines.append(f"• <b>{sym_clean}</b> | div: {div_str}")
    return "\n".join(lines)

# ── Ana akış ──────────────────────────────────────────────────────────────────
def main():
    state=load_state()

    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] FLCT Multi-TF başladı")

    # Sembol ve hacim listesi (bir kez çek)
    try:
        syms=get_symbols()
        vols=get_volumes()
        syms=[s for s in syms if vols.get(s,0)>=MIN_VOLUME_USDT]
        print(f"  {len(syms)} sembol (hacim filtresi geçti)")
    except Exception as e:
        print(f"[kritik hata] Sembol/hacim çekilemedi: {e}"); return

    changed=False

    for tf in ENABLED_TFS:
        # BTC üzerinden son kapanan mum zamanını al
        last_ct=get_last_close_time("BTCUSDT",tf)
        if last_ct is None:
            print(f"  [{tf}] kline alınamadı, atlandı"); continue

        prev_ct=state.get(tf,0)
        if last_ct<=prev_ct:
            print(f"  [{tf}] yeni mum yok (ct={last_ct}), atlandı"); continue

        # Yeni mum kapandı — tara
        print(f"  [{tf}] yeni mum! taranıyor...")
        longs,shorts=scan_tf(tf,syms)

        state[tf]=last_ct; changed=True

        if not longs and not shorts:
            print(f"  [{tf}] sinyal yok"); continue

        msg=build_message(tf,longs,shorts)
        print(msg)
        send_telegram(msg)

    if changed:
        save_state(state)

if __name__=="__main__":
    main()
