"""
Multi-Timeframe Crypto Setup Bot
---------------------------------
Analyzes BTC, ETH, SOL, SUI, and XRP across five timeframes (15m, 30m,
1h, 4h, 8h) every 2 hours, looks for multi-timeframe trend agreement
("confluence"), factors in BTC's trend as a correlation check for the
altcoins, and sends ONE combined Telegram message per cycle listing
any qualifying setups with entry / stop-loss / take-profit / risk and
a plain-English explanation of what was actually seen on each chart.

READ THIS FIRST
================
No script, and no human trader, is right 100% of the time — markets
are genuinely uncertain, not just "hard to compute." What this bot
gives you is a disciplined, multi-factor READ of the charts (the kind
of checklist a discretionary trader runs through), not a guarantee.
Position-size around the stop-loss, not around confidence in any one
signal.

The "macro headlines" section is a raw headline pull (no key needed)
for awareness — it is NOT a substitute for actually reading the news
or an economic model. Treat it as a nudge to go check, not an answer.

SETUP
=====
1. pip install requests pandas numpy
2. Fill in TELEGRAM_TOKEN / TELEGRAM_CHAT_ID below or set them as
   environment variables of the same name.
3. Run once locally to test:      python crypto_alert_bot.py --once
4. For free 24/7 scheduling, see the accompanying GitHub Actions
   workflow (.github/workflows/crypto-alerts.yml) — it triggers this
   script with --once every 2 hours.

CUSTOMIZE
=========
- COINS: which pairs to track (Binance symbol format)
- TIMEFRAMES / TF_WEIGHTS: which candles to read and how much each
  timeframe counts toward the overall trend bias
- CONFLUENCE_THRESHOLD: how much timeframe agreement is required
  before a setup is flagged at all (raise this = fewer, stronger
  setups; lower it = more, weaker setups)
"""

import os
import sys
import time
import traceback
from datetime import datetime, timezone

import requests
import pandas as pd
import numpy as np

# ------------------------- CONFIG -------------------------

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "PUT_YOUR_BOT_TOKEN_HERE")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "PUT_YOUR_CHAT_ID_HERE")

COINS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "SUIUSDT", "XRPUSDT"]
TIMEFRAMES = ["15m", "30m", "1h", "4h", "8h"]
# Higher timeframes carry more weight toward the overall trend bias.
TF_WEIGHTS = {"15m": 1, "30m": 1, "1h": 2, "4h": 3, "8h": 3}
MAX_SCORE = sum(TF_WEIGHTS.values())  # used to normalize confidence %

RSI_PERIOD = 14
EMA_FAST = 9
EMA_SLOW = 21
ATR_PERIOD = 14

# A setup only fires if the weighted timeframe score is at least this
# large in magnitude (out of MAX_SCORE). Raise for stricter filtering.
CONFLUENCE_THRESHOLD = 6

# Entry/exit sizing — based on 15m volatility since setups are meant
# to play out on a roughly 2-hour horizon (~8 x 15m candles).
ATR_STOP_MULTIPLIER = 1.5
RISK_REWARD_RATIO = 2.0
RISK_LOW_PCT = 1.0
RISK_MEDIUM_PCT = 2.5

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
NEWS_RSS_URL = "https://news.google.com/rss/search?q=crypto%20OR%20bitcoin%20OR%20fed%20interest%20rate&hl=en-US&gl=US&ceid=US:en"
NEWS_HEADLINE_COUNT = 4

TELEGRAM_MAX_LEN = 3800  # stay under Telegram's ~4096 char limit per message

# ------------------------- DATA -------------------------

def fetch_klines(symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
    """Fetch OHLCV candles from Binance's public REST API (no key needed)."""
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    resp = requests.get(BINANCE_KLINES_URL, params=params, timeout=10)
    resp.raise_for_status()
    raw = resp.json()

    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ]
    df = pd.DataFrame(raw, columns=cols)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    return df

def fetch_macro_headlines(limit: int = NEWS_HEADLINE_COUNT):
    """Pull recent headline titles only (no article content) as a quick
    macro-awareness nudge. Best-effort — failures are silently skipped
    so a news outage never blocks the technical analysis."""
    try:
        resp = requests.get(NEWS_RSS_URL, timeout=8)
        resp.raise_for_status()
        import xml.etree.ElementTree as ET
        root = ET.fromstring(resp.content)
        titles = [item.findtext("title") for item in root.findall(".//item")]
        return [t for t in titles if t][:limit]
    except Exception:
        return []

# ------------------------- INDICATORS -------------------------

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def compute_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period).mean()

def detect_candle_pattern(df: pd.DataFrame) -> str:
    """Very simple last-candle pattern check: engulfing or doji."""
    if len(df) < 2:
        return "no pattern"
    prev, last = df.iloc[-2], df.iloc[-1]
    body = abs(last["close"] - last["open"])
    rng = last["high"] - last["low"]
    prev_bullish = prev["close"] > prev["open"]
    last_bullish = last["close"] > last["open"]

    if rng > 0 and body / rng < 0.1:
        return "doji (indecision)"
    if (not prev_bullish and last_bullish
            and last["open"] <= prev["close"] and last["close"] >= prev["open"]):
        return "bullish engulfing"
    if (prev_bullish and not last_bullish
            and last["open"] >= prev["close"] and last["close"] <= prev["open"]):
        return "bearish engulfing"
    return "no strong pattern"

# ------------------------- PER-TIMEFRAME ANALYSIS -------------------------

def analyze_timeframe(symbol: str, tf: str) -> dict | None:
    df = fetch_klines(symbol, tf, limit=100)
    if len(df) < max(EMA_SLOW, RSI_PERIOD, ATR_PERIOD) + 2:
        return None

    df["rsi"] = compute_rsi(df["close"], RSI_PERIOD)
    df["ema_fast"] = compute_ema(df["close"], EMA_FAST)
    df["ema_slow"] = compute_ema(df["close"], EMA_SLOW)
    df["atr"] = compute_atr(df, ATR_PERIOD)
    last = df.iloc[-1]

    if last["ema_fast"] > last["ema_slow"] and last["rsi"] > 50:
        trend = 1
    elif last["ema_fast"] < last["ema_slow"] and last["rsi"] < 50:
        trend = -1
    else:
        trend = 0

    return {
        "tf": tf,
        "trend": trend,
        "close": last["close"],
        "rsi": last["rsi"],
        "ema_fast": last["ema_fast"],
        "ema_slow": last["ema_slow"],
        "atr": last["atr"],
        "pattern": detect_candle_pattern(df),
    }

def analyze_coin(symbol: str, btc_score: float | None) -> dict:
    tf_results = {}
    for tf in TIMEFRAMES:
        try:
            tf_results[tf] = analyze_timeframe(symbol, tf)
        except Exception:
            tf_results[tf] = None

    weighted_score = sum(
        TF_WEIGHTS[tf] * res["trend"]
        for tf, res in tf_results.items() if res is not None
    )
    confidence_pct = round(abs(weighted_score) / MAX_SCORE * 100)
    direction = "BUY" if weighted_score > 0 else "SELL" if weighted_score < 0 else "NEUTRAL"

    btc_alignment = None
    if btc_score is not None and symbol != "BTCUSDT":
        same_sign = (weighted_score > 0 and btc_score > 0) or (weighted_score < 0 and btc_score < 0)
        btc_alignment = "aligned with BTC" if same_sign else "conflicts with BTC"

    has_setup = abs(weighted_score) >= CONFLUENCE_THRESHOLD and direction != "NEUTRAL"
    # Require BTC agreement for altcoin setups to avoid fighting the market leader.
    if has_setup and btc_alignment == "conflicts with BTC":
        has_setup = False

    result = {
        "symbol": symbol,
        "tf_results": tf_results,
        "weighted_score": weighted_score,
        "confidence_pct": confidence_pct,
        "direction": direction,
        "btc_alignment": btc_alignment,
        "has_setup": has_setup,
    }

    if has_setup:
        tf15 = tf_results.get("15m")
        entry = tf15["close"] if tf15 else tf_results[TIMEFRAMES[0]]["close"]
        atr15 = tf15["atr"] if tf15 else None
        if atr15 is None or pd.isna(atr15):
            has_setup = False
            result["has_setup"] = False
        else:
            stop_distance = atr15 * ATR_STOP_MULTIPLIER
            if direction == "BUY":
                stop_loss = entry - stop_distance
                take_profit = entry + stop_distance * RISK_REWARD_RATIO
            else:
                stop_loss = entry + stop_distance
                take_profit = entry - stop_distance * RISK_REWARD_RATIO
            risk_pct = (stop_distance / entry) * 100
            risk_level = "LOW" if risk_pct <= RISK_LOW_PCT else "MEDIUM" if risk_pct <= RISK_MEDIUM_PCT else "HIGH"
            result.update({
                "entry": entry,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "risk_pct": risk_pct,
                "risk_level": risk_level,
            })

    return result

# ------------------------- EXPLANATION TEXT -------------------------

def build_reasoning(result: dict) -> str:
    lines = []
    for tf in TIMEFRAMES:
        r = result["tf_results"].get(tf)
        if r is None:
            continue
        trend_word = "bullish" if r["trend"] == 1 else "bearish" if r["trend"] == -1 else "mixed/flat"
        lines.append(
            f"  {tf}: {trend_word} (EMA{EMA_FAST} {'>' if r['ema_fast']>r['ema_slow'] else '<'} "
            f"EMA{EMA_SLOW}, RSI {r['rsi']:.0f}, last candle: {r['pattern']})"
        )
    return "\n".join(lines)

def format_coin_section(result: dict) -> str:
    symbol = result["symbol"]
    if not result["has_setup"]:
        return (
            f"— {symbol}: no qualifying setup (confidence {result['confidence_pct']}%, "
            f"timeframes not in strong agreement)"
        )

    align_note = f"\nBTC check: {result['btc_alignment']}" if result["btc_alignment"] else ""
    return (
        f"✅ {symbol} — {result['direction']} setup (confidence {result['confidence_pct']}%)\n"
        f"Entry: {result['entry']:.4f}\n"
        f"Stop-loss: {result['stop_loss']:.4f}\n"
        f"Take-profit: {result['take_profit']:.4f}\n"
        f"Risk: {result['risk_level']} (stop is {result['risk_pct']:.2f}% away)\n"
        f"Setup window: ~2 hours, re-checked next cycle{align_note}\n"
        f"Why:\n{build_reasoning(result)}"
    )

# ------------------------- TELEGRAM -------------------------

def send_telegram_message(text: str) -> None:
    if "PUT_YOUR" in TELEGRAM_TOKEN or "PUT_YOUR" in TELEGRAM_CHAT_ID:
        print("[WARN] Telegram not configured — printing instead:\n", text)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        resp = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        print(f"[ERROR] Failed to send Telegram message: {e}")

def send_long_message(text: str) -> None:
    """Split into multiple Telegram messages if needed."""
    if len(text) <= TELEGRAM_MAX_LEN:
        send_telegram_message(text)
        return
    parts = text.split("\n\n")
    chunk = ""
    for part in parts:
        if len(chunk) + len(part) + 2 > TELEGRAM_MAX_LEN:
            send_telegram_message(chunk)
            chunk = part
        else:
            chunk = f"{chunk}\n\n{part}" if chunk else part
    if chunk:
        send_telegram_message(chunk)

# ------------------------- MAIN CYCLE -------------------------

def run_cycle():
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    sections = [f"📊 Market scan — {ts}"]

    headlines = fetch_macro_headlines()
    if headlines:
        sections.append("Macro headlines (context only, not analyzed in depth):\n" +
                         "\n".join(f"  • {h}" for h in headlines))

    btc_result = analyze_coin("BTCUSDT", btc_score=None)
    sections.append(format_coin_section(btc_result))

    for symbol in COINS:
        if symbol == "BTCUSDT":
            continue
        result = analyze_coin(symbol, btc_score=btc_result["weighted_score"])
        sections.append(format_coin_section(result))

    sections.append("⚠️ Technical + headline scan only — not a guarantee. Size around the stop-loss.")

    send_long_message("\n\n".join(sections))

def main():
    if "--once" in sys.argv:
        run_cycle()
        return
    print("Starting multi-timeframe crypto bot. Checking every 2 hours. Ctrl+C to stop.\n")
    while True:
        try:
            run_cycle()
        except Exception:
            print("[ERROR] Problem during cycle:")
            traceback.print_exc()
        time.sleep(2 * 60 * 60)

if __name__ == "__main__":
    main()
