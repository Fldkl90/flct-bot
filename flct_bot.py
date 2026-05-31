import asyncio
import aiohttp
import logging
from datetime import datetime, timezone
from telegram import Bot
from telegram.ext import Application, CommandHandler, ContextTypes
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ── Ayarlar ──────────────────────────────────────────────────
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN', 'BURAYA_TOKEN')
CHAT_ID        = os.environ.get('CHAT_ID', 'BURAYA_CHAT_ID')

# Hangi TF'ler otomatik taransın
AUTO_TFS = ['15m', '1h', '4h', '1d']

# ── LeLedc — Pine birebir ─────────────────────────────────────
def leledc(opens, closes, highs, lows, qual=6, length=30, cv=4):
    bi, si = 0, 0
    signals = [0] * len(closes)
    for i in range(cv, len(closes)):
        if closes[i] > closes[i - cv]:
            bi += 1
        elif closes[i] < closes[i - cv]:
            si += 1
        hl = min(length, i + 1)
        max_h = max(highs[i - j] for j in range(hl))
        min_l = min(lows[i - j]  for j in range(hl))
        if bi > qual and closes[i] < opens[i] and highs[i] >= max_h:
            signals[i] = -1
            bi = 0
        elif si > qual and closes[i] > opens[i] and lows[i] <= min_l:
            signals[i] = 1
            si = 0
    return signals

# ── TD Sequential — Pine birebir ─────────────────────────────
def td_seq(closes):
    TD, TS = 0, 0
    tda = [0] * len(closes)
    tsa = [0] * len(closes)
    for i in range(4, len(closes)):
        TD = min(TD + 1, 200) if closes[i] > closes[i - 4] else 0
        TS = min(TS + 1, 200) if closes[i] < closes[i - 4] else 0
        tda[i] = TD
        tsa[i] = TS
    return tda, tsa

# ── Sinyal — Pine ±1 bar ─────────────────────────────────────
def get_signal(lele, tda, tsa, n):
    if n < 1:
        return False, False
    l0, l1 = lele[n] == 1,  lele[n-1] == 1
    b0, b1 = lele[n] == -1, lele[n-1] == -1
    ts0, ts1 = tsa[n] == 9, tsa[n-1] == 9
    td0, td1 = tda[n] == 9, tda[n-1] == 9
    long_sig  = (l0 and ts0) or (l1 and ts0) or (l0 and ts1)
    short_sig = (b0 and td0) or (b1 and td0) or (b0 and td1)
    if long_sig and short_sig:
        lb = n if l0 else (n-1 if l1 else -1)
        br = n if b0 else (n-1 if b1 else -1)
        if lb > br: short_sig = False
        else:       long_sig  = False
    return long_sig, short_sig

# ── Kline birleştirme ─────────────────────────────────────────
def merge_nx(klines, n):
    out, rem = [], len(klines) % n
    for i in range(rem, len(klines) - n + 1, n):
        g = klines[i:i+n]
        out.append({
            'open':  float(g[0]['open']),
            'high':  max(float(k['high'])  for k in g),
            'low':   min(float(k['low'])   for k in g),
            'close': float(g[-1]['close']),
        })
    return out

# ── API params ────────────────────────────────────────────────
def get_api_params(tf):
    if tf == '45m': return '15m', 900, 3
    if tf == '2h':  return '1h',  600, 2
    if tf == '3h':  return '1h',  900, 3
    if tf == '1M':  return '1M',  200, 0
    return tf, 300, 0

# ── Futures sembollerini çek ──────────────────────────────────
async def fetch_symbols(session):
    url = 'https://fapi.binance.com/fapi/v1/exchangeInfo'
    async with session.get(url) as r:
        d = await r.json()
    return sorted([
        s['symbol'] for s in d['symbols']
        if s['quoteAsset'] == 'USDT'
        and s['contractType'] == 'PERPETUAL'
        and s['status'] == 'TRADING'
    ])

# ── Kline çek ve analiz et ────────────────────────────────────
async def fetch_and_analyze(session, sym, tf):
    api_tf, limit, merge = get_api_params(tf)
    url = f'https://fapi.binance.com/fapi/v1/klines?symbol={sym}&interval={api_tf}&limit={limit}'
    async with session.get(url) as r:
        if r.status != 200:
            return None
        raw = await r.json()
    if not isinstance(raw, list) or len(raw) < 10:
        return None

    klines = [{'open': k[1], 'high': k[2], 'low': k[3], 'close': k[4]} for k in raw]
    if merge > 0:
        klines = merge_nx(klines, merge)
    if len(klines) < 10:
        return None

    opens  = [float(k['open'])  for k in klines]
    highs  = [float(k['high'])  for k in klines]
    lows   = [float(k['low'])   for k in klines]
    closes = [float(k['close']) for k in klines]

    lele = leledc(opens, closes, highs, lows)
    tda, tsa = td_seq(closes)
    n = len(closes) - 2  # son kapanmış mum

    long_sig, short_sig = get_signal(lele, tda, tsa, n)
    return {'long': long_sig, 'short': short_sig, 'price': closes[n]}

# ── Tarama ───────────────────────────────────────────────────
async def scan(tf):
    results = []
    async with aiohttp.ClientSession() as session:
        symbols = await fetch_symbols(session)
        logger.info(f'{tf} taraması başladı — {len(symbols)} sembol')

        # 10'lu batch
        for i in range(0, len(symbols), 10):
            batch = symbols[i:i+10]
            tasks = [fetch_and_analyze(session, sym, tf) for sym in batch]
            answers = await asyncio.gather(*tasks, return_exceptions=True)
            for sym, res in zip(batch, answers):
                if isinstance(res, dict) and (res['long'] or res['short']):
                    results.append((sym, res))
            await asyncio.sleep(0.1)  # rate limit

    return results

# ── Telegram mesajı gönder ────────────────────────────────────
async def send_signals(bot, tf, results):
    if not results:
        return
    now = datetime.now(timezone.utc).strftime('%H:%M UTC')
    lines = [f'📡 *FLCT v5 — {tf.upper()} Tarama* ({now})\n']
    for sym, res in results:
        base = sym.replace('USDT', '')
        if res['long']:
            lines.append(f'🟢 *LONG* — {base} — ${res["price"]:,.4g}')
        else:
            lines.append(f'🔴 *SHORT* — {base} — ${res["price"]:,.4g}')
    msg = '\n'.join(lines)
    await bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode='Markdown')
    logger.info(f'{tf}: {len(results)} sinyal gönderildi')

# ── Bar kapanış zamanı hesabı (UTC takvim bazlı) ──────────────
def next_bar_close_utc(tf):
    now = datetime.now(timezone.utc)
    if tf in ('3m', '5m', '15m', '30m', '45m', '1h', '2h', '3h', '4h'):
        mins = {'3m':3,'5m':5,'15m':15,'30m':30,'45m':45,'1h':60,'2h':120,'3h':180,'4h':240}[tf]
        total_min = now.hour * 60 + now.minute
        next_min  = ((total_min // mins) + 1) * mins
        delta_sec = (next_min - total_min) * 60 - now.second
        return delta_sec
    if tf == '1d':
        # Bir sonraki UTC gece yarısı
        from datetime import timedelta
        tomorrow = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        return int((tomorrow - now).total_seconds())
    if tf == '1w':
        from datetime import timedelta
        days_to_monday = (7 - now.weekday()) % 7 or 7
        next_monday = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=days_to_monday)
        return int((next_monday - now).total_seconds())
    if tf == '1M':
        from datetime import timedelta
        if now.month == 12:
            next_month = now.replace(year=now.year+1, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        else:
            next_month = now.replace(month=now.month+1, day=1, hour=0, minute=0, second=0, microsecond=0)
        return int((next_month - now).total_seconds())
    return 3600

# ── Otomatik zamanlayıcı ──────────────────────────────────────
async def auto_scanner(bot):
    """Her TF'nin bar kapanışında tarama yapar."""
    # İlk çalıştırmada her TF için ne kadar beklemek gerektiğini hesapla
    tasks = {}
    for tf in AUTO_TFS:
        wait = next_bar_close_utc(tf)
        tasks[tf] = asyncio.create_task(run_tf_loop(bot, tf, wait))
        logger.info(f'{tf} ilk tarama {wait//60}dk {wait%60}sn sonra')
    await asyncio.gather(*tasks.values())

async def run_tf_loop(bot, tf, initial_wait):
    """Bir TF için sonsuz döngü: bekle → tara → sinyal gönder → tekrarla."""
    await asyncio.sleep(initial_wait + 5)  # +5sn bar kapanış toleransı
    while True:
        try:
            results = await scan(tf)
            await send_signals(bot, tf, results)
        except Exception as e:
            logger.error(f'{tf} tarama hatası: {e}')
        # Bir sonraki bar kapanışına kadar bekle
        wait = next_bar_close_utc(tf)
        logger.info(f'{tf} sonraki tarama {wait//60}dk sonra')
        await asyncio.sleep(wait + 5)

# ── /tara komutu ──────────────────────────────────────────────
async def cmd_tara(update, context: ContextTypes.DEFAULT_TYPE):
    tf = context.args[0].lower() if context.args else '1h'
    valid = ['3m','5m','15m','30m','45m','1h','2h','3h','4h','1d','1w','1M']
    if tf not in valid:
        await update.message.reply_text(f'Geçersiz TF. Seçenekler: {", ".join(valid)}')
        return
    await update.message.reply_text(f'⏳ {tf.upper()} taraması başladı...')
    try:
        results = await scan(tf)
        await send_signals(context.bot, tf, results)
        if not results:
            await update.message.reply_text(f'✅ {tf.upper()} taraması bitti — sinyal yok.')
    except Exception as e:
        await update.message.reply_text(f'❌ Hata: {e}')

# ── /durum komutu ─────────────────────────────────────────────
async def cmd_durum(update, context: ContextTypes.DEFAULT_TYPE):
    now = datetime.now(timezone.utc)
    lines = ['📊 *Bot Durumu*\n']
    for tf in AUTO_TFS:
        wait = next_bar_close_utc(tf)
        h, m = divmod(wait // 60, 60)
        lines.append(f'*{tf.upper()}* → {h}sa {m}dk sonra kapanır')
    await update.message.reply_text('\n'.join(lines), parse_mode='Markdown')

# ── /yardim komutu ────────────────────────────────────────────
async def cmd_yardim(update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        '🤖 *FLCT v5 Bot Komutları*\n\n'
        '/tara `[tf]` — Manuel tarama (örn: /tara 4h)\n'
        '/durum — Sonraki bar kapanışları\n'
        '/yardim — Bu mesaj\n\n'
        f'⚙️ Otomatik tarama: {", ".join(AUTO_TFS)}'
    )
    await update.message.reply_text(msg, parse_mode='Markdown')

# ── Ana fonksiyon ─────────────────────────────────────────────
async def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler('tara',   cmd_tara))
    app.add_handler(CommandHandler('durum',  cmd_durum))
    app.add_handler(CommandHandler('yardim', cmd_yardim))

    bot = app.bot
    logger.info('FLCT Bot başlatıldı')

    # Otomatik taramayı arka planda başlat
    asyncio.create_task(auto_scanner(bot))

    # Bot'u çalıştır
    await app.initialize()
    await app.start()
    await app.updater.start_polling()
    await asyncio.Event().wait()  # sonsuza kadar çalış

if __name__ == '__main__':
    asyncio.run(main())
