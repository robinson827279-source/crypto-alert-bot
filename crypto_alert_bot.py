"""
Multi-Factor Crypto Market Analysis & Trade Setup Bot
======================================================

This version keeps the existing Binance + Telegram plumbing, but replaces
simple EMA9/21 + RSI timeframe voting with a transparent, multi-factor setup
engine.

Main flow:
    MARKET UNIVERSE
    -> LIQUIDITY FILTER
    -> OPPORTUNITY FILTER
    -> OHLCV / INDICATORS
    -> MARKET STRUCTURE
    -> S/R + SUPPLY/DEMAND
    -> PRICE ACTION
    -> VOLUME + MOMENTUM + VOLATILITY
    -> LIQUIDITY
    -> MULTI-TIMEFRAME CONTEXT
    -> MARKET REGIME / OVEREXTENSION
    -> BTC / ALT CONTEXT
    -> ENTRY CONFIRMATION
    -> STRUCTURAL SL / LOGICAL TP
    -> SETUP QUALITY SCORE
    -> RANKING
    -> TELEGRAM

IMPORTANT:
- This is a rule-based market scanner, not a guaranteed predictor.
- The score is NOT a probability. Do not call it "80% confidence" unless
  it is later statistically calibrated with out-of-sample backtesting.
- Every factor is returned explicitly so it can be logged and backtested.
- The bot does not force a daily trade quota. Zero confirmed setups is valid.

Dependencies:
    pip install requests pandas numpy

Environment variables:
    TELEGRAM_TOKEN
    TELEGRAM_CHAT_ID

Optional environment variables:
    MAX_MARKET_SCAN=100
    MAX_DEEP_ANALYSIS=25
    MIN_24H_QUOTE_VOLUME=10000000
    MIN_TRADES_24H=5000
    SCAN_INTERVAL_MINUTES=10
    SEND_WATCHLIST=true
    SEND_NO_TRADE_SUMMARY=true
    MIN_SETUP_SCORE=8
    MIN_RR=2.0
    MAX_SPREAD_PCT=0.25
"""

import os
import sys
import time
import traceback
import math
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Optional

import requests
import pandas as pd
import numpy as np

# ------------------------- CONFIG -------------------------

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "PUT_YOUR_BOT_TOKEN_HERE")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "PUT_YOUR_CHAT_ID_HERE")

# Binance provides several public market-data endpoints. The data-api endpoint
# is documented for public /api/v3 market-data routes, including exchangeInfo,
# klines and ticker endpoints. We try several endpoints so one restricted route
# does not take down the whole scanner.
BINANCE_BASE_URLS = [
    "https://data-api.binance.vision",
    "https://api.binance.com",
    "https://api-gcp.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://api4.binance.com",
]
BINANCE_BASE_URL = BINANCE_BASE_URLS[0]
BINANCE_KLINES_URL = f"{BINANCE_BASE_URL}/api/v3/klines"
BINANCE_EXCHANGE_INFO_URL = f"{BINANCE_BASE_URL}/api/v3/exchangeInfo"
BINANCE_TICKER_24H_URL = f"{BINANCE_BASE_URL}/api/v3/ticker/24hr"
BINANCE_BOOK_TICKER_URL = f"{BINANCE_BASE_URL}/api/v3/ticker/bookTicker"

# BTC and ETH are always part of the deep-analysis set.
CORE_COINS = ["BTCUSDT", "ETHUSDT"]

# These are preferred large/liquid assets, but the market scanner can add
# other liquid USDT pairs dynamically.
PREFERRED_COINS = [
    "SOLUSDT", "XRPUSDT", "BNBUSDT", "ADAUSDT", "DOGEUSDT", "TRXUSDT",
    "AVAXUSDT", "LINKUSDT", "DOTUSDT", "LTCUSDT", "BCHUSDT", "TONUSDT",
    "SUIUSDT",
]

# Optional hard allow-list. Leave empty to allow dynamic selection.
# Example: ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
COIN_ALLOWLIST = []

TIMEFRAMES = ["15m", "30m", "1h", "4h", "8h"]
HTF_TIMEFRAMES = ["8h", "4h"]
SETUP_TIMEFRAME = "1h"
CONFIRMATION_TIMEFRAMES = ["30m", "15m"]

TF_WEIGHTS = {"15m": 1, "30m": 1, "1h": 2, "4h": 3, "8h": 3}

RSI_PERIOD = 14
EMA_FAST = 9
EMA_SLOW = 21
MA_MID = 50
MA_LONG = 200
ATR_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
VOLUME_LOOKBACK = 20
SWING_LEFT = 3
SWING_RIGHT = 3

MAX_MARKET_SCAN = int(os.environ.get("MAX_MARKET_SCAN", "100"))
MAX_DEEP_ANALYSIS = int(os.environ.get("MAX_DEEP_ANALYSIS", "25"))
MIN_24H_QUOTE_VOLUME = float(os.environ.get("MIN_24H_QUOTE_VOLUME", "10000000"))
MIN_TRADES_24H = int(os.environ.get("MIN_TRADES_24H", "5000"))
MAX_SPREAD_PCT = float(os.environ.get("MAX_SPREAD_PCT", "0.25"))

MIN_SETUP_SCORE = float(os.environ.get("MIN_SETUP_SCORE", "8"))
MIN_RR = float(os.environ.get("MIN_RR", "2.0"))
MAX_SETUP_AGE_CANDLES = 8

# For local/long-running execution. Existing GitHub Actions can still invoke
# --once; if the workflow remains every 2h, it will remain every 2h.
SCAN_INTERVAL_MINUTES = int(os.environ.get("SCAN_INTERVAL_MINUTES", "10"))
SEND_WATCHLIST = os.environ.get("SEND_WATCHLIST", "true").lower() == "true"
SEND_NO_TRADE_SUMMARY = os.environ.get("SEND_NO_TRADE_SUMMARY", "true").lower() == "true"

NEWS_RSS_URL = (
    "https://news.google.com/rss/search?q=crypto%20OR%20bitcoin%20OR%20fed%20"
    "interest%20rate&hl=en-US&gl=US&ceid=US:en"
)
NEWS_HEADLINE_COUNT = 6

TELEGRAM_MAX_LEN = 3800
REQUEST_TIMEOUT = 12

# ------------------------- HTTP HELPERS -------------------------

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "MultiFactorCryptoBot/2.0"})


def get_json(url: str, params: Optional[dict] = None):
    """GET JSON, failing over across Binance public market-data hosts.

    GitHub Actions can receive HTTP 451 from one Binance host depending on the
    runner/IP location. Public market-data endpoints are also available through
    data-api.binance.vision, so do not make one host a single point of failure.
    Non-Binance URLs (for example Google News RSS) are requested normally.
    """
    if "binance.com" not in url and "binance.vision" not in url:
        resp = SESSION.get(url, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json()

    if "/api/" in url:
        parsed_path = "/api/" + url.split("/api/", 1)[1]
    elif "/sapi/" in url:
        parsed_path = "/sapi/" + url.split("/sapi/", 1)[1]
    else:
        parsed_path = "/" + url.split("/", 3)[-1]
    last_error = None
    for base in BINANCE_BASE_URLS:
        candidate = base + parsed_path
        try:
            resp = SESSION.get(candidate, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            last_error = exc
            continue

    if last_error:
        raise last_error
    raise RuntimeError("No Binance market-data endpoint available")


# ------------------------- MARKET UNIVERSE -------------------------

STABLE_BASES = {
    "USDT", "USDC", "BUSD", "FDUSD", "TUSD", "USDP", "DAI", "EUR",
    "TRY", "BRL", "GBP", "JPY", "AUD", "BIDR", "UAH", "RUB",
}


def fetch_exchange_symbols() -> list[str]:
    """Return active Binance spot USDT symbols, or [] if discovery is unavailable."""
    try:
        data = get_json(BINANCE_EXCHANGE_INFO_URL)
    except Exception as exc:
        print(f"WARNING: Binance exchangeInfo unavailable: {exc}")
        print("WARNING: Falling back to the configured liquid-coin universe.")
        return []

    symbols = []
    for item in data.get("symbols", []):
        if item.get("status") != "TRADING":
            continue
        if item.get("quoteAsset") != "USDT":
            continue
        if item.get("isSpotTradingAllowed") is False:
            continue
        base = item.get("baseAsset", "")
        if base in STABLE_BASES:
            continue
        symbols.append(item["symbol"])
    return symbols


def fetch_24h_tickers() -> dict:
    """Fetch 24h statistics; return empty data instead of crashing the scan."""
    try:
        rows = get_json(BINANCE_TICKER_24H_URL)
        return {row["symbol"]: row for row in rows}
    except Exception as exc:
        print(f"WARNING: Binance 24h ticker unavailable: {exc}")
        return {}


def fetch_book_tickers() -> dict:
    """Fetch best bid/ask; return empty data if unavailable."""
    try:
        rows = get_json(BINANCE_BOOK_TICKER_URL)
        return {row["symbol"]: row for row in rows}
    except Exception as exc:
        print(f"WARNING: Binance book ticker unavailable: {exc}")
        return {}


def build_market_universe() -> tuple[list[str], dict]:
    """
    Select a dynamic liquid USDT universe while preserving BTC/ETH.

    If Binance market discovery or 24h ticker discovery is unavailable from the
    runner, fall back safely to the configured core + preferred universe rather
    than crashing the entire Telegram bot.
    """
    exchange_symbols = set(fetch_exchange_symbols())
    tickers = fetch_24h_tickers()
    books = fetch_book_tickers()

    # Fallback is intentionally explicit and configurable. It keeps the bot
    # operational if the runner receives HTTP 451 or another access failure.
    fallback_symbols = list(dict.fromkeys(CORE_COINS + PREFERRED_COINS))

    if not exchange_symbols or not tickers:
        selected = fallback_symbols[:MAX_MARKET_SCAN]
        if COIN_ALLOWLIST:
            selected = [s for s in selected if s in COIN_ALLOWLIST]
        return selected, {
            "exchange_usdt_pairs": len(exchange_symbols),
            "liquid_candidates": 0,
            "selected": len(selected),
            "tickers": tickers,
            "fallback": True,
        }

    candidates = []
    for symbol in exchange_symbols:
        if COIN_ALLOWLIST and symbol not in COIN_ALLOWLIST:
            continue
        ticker = tickers.get(symbol)
        book = books.get(symbol)
        if not ticker:
            continue

        quote_volume = float(ticker.get("quoteVolume", 0) or 0)
        trades = int(float(ticker.get("count", 0) or 0))
        last_price = float(ticker.get("lastPrice", 0) or 0)
        bid = float(book.get("bidPrice", 0) or 0) if book else 0
        ask = float(book.get("askPrice", 0) or 0) if book else 0
        spread_pct = ((ask - bid) / last_price * 100) if last_price and ask >= bid else 0

        if quote_volume < MIN_24H_QUOTE_VOLUME:
            continue
        if trades < MIN_TRADES_24H:
            continue
        # If book data is unavailable, don't reject the symbol solely because
        # the spread could not be measured. If book data exists, enforce it.
        if book and spread_pct > MAX_SPREAD_PCT:
            continue

        liquidity_score = (
            math.log10(max(quote_volume, 1)) * 2
            + math.log10(max(trades, 1))
            - spread_pct * 2
        )
        candidates.append({
            "symbol": symbol,
            "quote_volume": quote_volume,
            "trades": trades,
            "spread_pct": spread_pct,
            "price_change_pct": float(ticker.get("priceChangePercent", 0) or 0),
            "liquidity_score": liquidity_score,
        })

    candidates.sort(key=lambda x: x["liquidity_score"], reverse=True)

    preferred = [c for c in candidates if c["symbol"] in PREFERRED_COINS]
    preferred.sort(key=lambda x: PREFERRED_COINS.index(x["symbol"]))

    selected = []
    for symbol in CORE_COINS:
        if symbol in exchange_symbols and symbol in tickers:
            selected.append(symbol)

    for item in preferred + candidates:
        if item["symbol"] not in selected:
            selected.append(item["symbol"])
        if len(selected) >= MAX_MARKET_SCAN:
            break

    # If filters are unusually restrictive, preserve the configured preferred
    # universe rather than ending up with an empty scan.
    if not selected:
        selected = [s for s in fallback_symbols if s in exchange_symbols or not exchange_symbols]

    selected = selected[:MAX_MARKET_SCAN]
    stats = {
        "exchange_usdt_pairs": len(exchange_symbols),
        "liquid_candidates": len(candidates),
        "selected": len(selected),
        "tickers": tickers,
        "fallback": False,
    }
    return selected, stats


# ------------------------- DATA -------------------------


def fetch_klines(symbol: str, interval: str, limit: int = 260) -> pd.DataFrame:
    """Fetch OHLCV candles and exclude the currently open candle."""
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    raw = get_json(BINANCE_KLINES_URL, params=params)
    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ]
    df = pd.DataFrame(raw, columns=cols)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    now = pd.Timestamp.now(tz="UTC")
    df = df[df["close_time"] <= now].copy()
    return df.reset_index(drop=True)


# ------------------------- NEWS / CONTEXT -------------------------

POSITIVE_WORDS = {
    "approval", "approved", "adoption", "inflow", "bullish", "surge",
    "rally", "growth", "partnership", "launch", "record", "positive",
}
NEGATIVE_WORDS = {
    "hack", "exploit", "lawsuit", "ban", "banned", "outflow", "bearish",
    "crash", "liquidation", "fraud", "investigation", "negative", "war",
}
HIGH_IMPACT_WORDS = {
    "fed", "fomc", "rate decision", "interest rate", "cpi", "inflation",
    "jobs report", "payroll", "employment", "sec", "etf", "tariff",
    "central bank",
}


def classify_headline(title: str) -> dict:
    text = title.lower()
    pos = sum(word in text for word in POSITIVE_WORDS)
    neg = sum(word in text for word in NEGATIVE_WORDS)
    high = any(word in text for word in HIGH_IMPACT_WORDS)
    sentiment = "Positive" if pos > neg else "Negative" if neg > pos else "Neutral"
    asset_relevant = any(x in text for x in [
        "bitcoin", "btc", "ethereum", "eth", "solana", "sol", "xrp",
        "binance", "crypto", "cryptocurrency", "digital asset",
    ])
    market_wide = any(x in text for x in [
        "fed", "fomc", "cpi", "inflation", "interest rate", "tariff",
        "recession", "jobs report", "employment", "central bank",
    ])
    return {
        "title": title,
        "sentiment": sentiment,
        "high_impact": high,
        "asset_relevant": asset_relevant,
        "market_wide": market_wide,
    }


def fetch_macro_headlines(limit: int = NEWS_HEADLINE_COUNT) -> list[dict]:
    try:
        resp = SESSION.get(NEWS_RSS_URL, timeout=8)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        titles = [item.findtext("title") for item in root.findall(".//item")]
        return [classify_headline(t) for t in titles if t][:limit]
    except Exception:
        return []


def news_context(headlines: list[dict]) -> dict:
    if not headlines:
        return {
            "risk": "UNKNOWN",
            "sentiment": "Neutral",
            "high_impact": False,
            "reason": "No headline data available",
        }
    high = sum(1 for x in headlines if x["high_impact"])
    pos = sum(1 for x in headlines if x["sentiment"] == "Positive")
    neg = sum(1 for x in headlines if x["sentiment"] == "Negative")
    sentiment = "Positive" if pos > neg else "Negative" if neg > pos else "Neutral"
    risk = "HIGH" if high >= 2 else "ELEVATED" if high == 1 else "NORMAL"
    return {
        "risk": risk,
        "sentiment": sentiment,
        "high_impact": high > 0,
        "reason": f"{high} high-impact headline(s); overall sentiment {sentiment.lower()}",
    }


# ------------------------- INDICATORS -------------------------


def compute_rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)


def compute_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema9"] = compute_ema(df["close"], 9)
    df["ema21"] = compute_ema(df["close"], 21)
    df["ema50"] = compute_ema(df["close"], 50)
    df["ema200"] = compute_ema(df["close"], 200)
    df["rsi"] = compute_rsi(df["close"])
    df["atr"] = compute_atr(df)

    ema_fast = compute_ema(df["close"], MACD_FAST)
    ema_slow = compute_ema(df["close"], MACD_SLOW)
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = compute_ema(df["macd"], MACD_SIGNAL)
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    df["vol_avg"] = df["volume"].rolling(VOLUME_LOOKBACK).mean()
    df["relative_volume"] = df["volume"] / df["vol_avg"].replace(0, np.nan)
    df["atr_pct"] = df["atr"] / df["close"] * 100
    df["atr_avg"] = df["atr"].rolling(20).mean()
    df["atr_ratio"] = df["atr"] / df["atr_avg"].replace(0, np.nan)
    return df


# ------------------------- SWINGS / MARKET STRUCTURE -------------------------


def find_swings(df: pd.DataFrame, left: int = SWING_LEFT, right: int = SWING_RIGHT) -> tuple[list[dict], list[dict]]:
    highs, lows = [], []
    if len(df) < left + right + 1:
        return highs, lows
    for i in range(left, len(df) - right):
        h = df["high"].iloc[i]
        l = df["low"].iloc[i]
        left_highs = df["high"].iloc[i-left:i]
        right_highs = df["high"].iloc[i+1:i+right+1]
        left_lows = df["low"].iloc[i-left:i]
        right_lows = df["low"].iloc[i+1:i+right+1]
        if h > left_highs.max() and h >= right_highs.max():
            highs.append({"index": i, "price": float(h)})
        if l < left_lows.min() and l <= right_lows.min():
            lows.append({"index": i, "price": float(l)})
    return highs, lows


def classify_swings(highs: list[dict], lows: list[dict]) -> dict:
    high_labels = []
    low_labels = []
    for prev, cur in zip(highs[:-1], highs[1:]):
        label = "HH" if cur["price"] > prev["price"] else "LH"
        high_labels.append({**cur, "label": label})
    for prev, cur in zip(lows[:-1], lows[1:]):
        label = "HL" if cur["price"] > prev["price"] else "LL"
        low_labels.append({**cur, "label": label})

    return {
        "highs": high_labels,
        "lows": low_labels,
        "last_high": high_labels[-1] if high_labels else None,
        "last_low": low_labels[-1] if low_labels else None,
    }


def analyze_structure(df: pd.DataFrame) -> dict:
    highs, lows = find_swings(df)
    classified = classify_swings(highs, lows)
    close = float(df["close"].iloc[-1])

    last_high = classified["last_high"]
    last_low = classified["last_low"]
    prior_high = classified["highs"][-2] if len(classified["highs"]) >= 2 else None
    prior_low = classified["lows"][-2] if len(classified["lows"]) >= 2 else None

    bullish = bool(last_high and last_low and last_high["label"] == "HH" and last_low["label"] == "HL")
    bearish = bool(last_high and last_low and last_high["label"] == "LH" and last_low["label"] == "LL")

    # BOS: current close breaks the most recent confirmed swing.
    bos = None
    if last_high and close > last_high["price"]:
        bos = "bullish BOS"
    elif last_low and close < last_low["price"]:
        bos = "bearish BOS"

    # CHoCH: break against the previously established swing sequence.
    choch = None
    if prior_high and prior_low:
        prior_bearish = prior_high["label"] == "LH" and prior_low["label"] == "LL"
        prior_bullish = prior_high["label"] == "HH" and prior_low["label"] == "HL"
        if prior_bearish and close > last_high["price"]:
            choch = "bullish CHoCH"
        elif prior_bullish and close < last_low["price"]:
            choch = "bearish CHoCH"

    # Range detection: compact recent range with no decisive structure break.
    recent = df.tail(40)
    range_high = float(recent["high"].max())
    range_low = float(recent["low"].min())
    range_pct = (range_high - range_low) / close * 100 if close else 0
    recent_atr_pct = float(recent["atr"].iloc[-1] / close * 100) if "atr" in recent and close else 0
    is_range = range_pct <= max(6.0, recent_atr_pct * 12) and bos is None

    if bullish:
        trend = "strong uptrend" if bos == "bullish BOS" else "weak uptrend"
    elif bearish:
        trend = "strong downtrend" if bos == "bearish BOS" else "weak downtrend"
    elif is_range:
        trend = "range"
    else:
        trend = "transition"

    return {
        "trend": trend,
        "bias": 1 if "uptrend" in trend else -1 if "downtrend" in trend else 0,
        "bos": bos,
        "choch": choch,
        "swings": classified,
        "range_high": range_high,
        "range_low": range_low,
        "range_pct": range_pct,
        "is_range": is_range,
    }


# ------------------------- SUPPORT / RESISTANCE -------------------------


def cluster_levels(levels: list[float], tolerance_pct: float = 0.35) -> list[float]:
    if not levels:
        return []
    levels = sorted(levels)
    clusters = [[levels[0]]]
    for price in levels[1:]:
        anchor = float(np.mean(clusters[-1]))
        if anchor and abs(price - anchor) / anchor * 100 <= tolerance_pct:
            clusters[-1].append(price)
        else:
            clusters.append([price])
    return [float(np.mean(c)) for c in clusters]


def detect_sr_zones(df: pd.DataFrame, structure: dict) -> dict:
    support = []
    resistance = []

    if structure["swings"]["lows"]:
        support.extend(x["price"] for x in structure["swings"]["lows"][-12:])
    if structure["swings"]["highs"]:
        resistance.extend(x["price"] for x in structure["swings"]["highs"][-12:])

    # Previous day/week levels from candle timestamps.
    d = df.copy().set_index("open_time")
    daily = d.resample("1D").agg({"high": "max", "low": "min"}).dropna()
    weekly = d.resample("1W").agg({"high": "max", "low": "min"}).dropna()
    if len(daily) >= 2:
        support.append(float(daily["low"].iloc[-2]))
        resistance.append(float(daily["high"].iloc[-2]))
    if len(weekly) >= 2:
        support.append(float(weekly["low"].iloc[-2]))
        resistance.append(float(weekly["high"].iloc[-2]))

    support = cluster_levels(support)
    resistance = cluster_levels(resistance)
    price = float(df["close"].iloc[-1])

    def nearest(levels, side):
        if side == "below":
            valid = [x for x in levels if x < price]
            return max(valid) if valid else None
        valid = [x for x in levels if x > price]
        return min(valid) if valid else None

    nearest_support = nearest(support, "below")
    nearest_resistance = nearest(resistance, "above")

    def distance(level):
        return abs(price - level) / price * 100 if level else None

    location = "middle of nowhere"
    near_level = None
    if nearest_support and distance(nearest_support) <= 0.8:
        location = "near support"
        near_level = nearest_support
    elif nearest_resistance and distance(nearest_resistance) <= 0.8:
        location = "near resistance"
        near_level = nearest_resistance
    elif structure["is_range"]:
        if price <= structure["range_low"] * 1.01:
            location = "near range low"
            near_level = structure["range_low"]
        elif price >= structure["range_high"] * 0.99:
            location = "near range high"
            near_level = structure["range_high"]

    return {
        "supports": support,
        "resistances": resistance,
        "nearest_support": nearest_support,
        "nearest_resistance": nearest_resistance,
        "support_distance_pct": distance(nearest_support),
        "resistance_distance_pct": distance(nearest_resistance),
        "location": location,
        "near_level": near_level,
    }


# ------------------------- SUPPLY / DEMAND -------------------------


def detect_supply_demand(df: pd.DataFrame, structure: dict) -> dict:
    """Heuristic zones: strong displacement away from a recent candle/base."""
    zones = []
    lookback = min(100, len(df) - 2)
    if lookback < 10:
        return {"zones": [], "nearest_demand": None, "nearest_supply": None}

    work = df.tail(lookback).reset_index(drop=True)
    median_range = (work["high"] - work["low"]).rolling(20).median()

    for i in range(3, len(work) - 2):
        candle_range = work.loc[i, "high"] - work.loc[i, "low"]
        if candle_range <= 0:
            continue
        future_move = work.loc[i+1:i+2, "close"].iloc[-1] - work.loc[i, "close"]
        threshold = max(float(median_range.iloc[i] or 0) * 1.4, float(work["atr"].iloc[i] or 0) * 1.2)
        if threshold <= 0:
            continue
        if future_move >= threshold:
            zones.append({
                "type": "demand",
                "low": float(work.loc[i, "low"]),
                "high": float(max(work.loc[i, "open"], work.loc[i, "close"])),
                "index": i,
                "strength": min(3, 1 + int(abs(future_move) / threshold)),
            })
        elif future_move <= -threshold:
            zones.append({
                "type": "supply",
                "low": float(min(work.loc[i, "open"], work.loc[i, "close"])),
                "high": float(work.loc[i, "high"]),
                "index": i,
                "strength": min(3, 1 + int(abs(future_move) / threshold)),
            })

    price = float(df["close"].iloc[-1])
    active = []
    for zone in zones:
        if zone["type"] == "demand" and price < zone["low"]:
            continue
        if zone["type"] == "supply" and price > zone["high"]:
            continue
        tests = 0
        for _, row in df.tail(80).iterrows():
            if zone["low"] <= row["low"] <= zone["high"] or zone["low"] <= row["high"] <= zone["high"]:
                tests += 1
        zone["tests"] = tests
        zone["fresh"] = tests <= 1
        zone["strength"] = max(1, zone["strength"] - max(0, tests - 2))
        active.append(zone)

    demand = [z for z in active if z["type"] == "demand" and z["high"] < price * 1.01]
    supply = [z for z in active if z["type"] == "supply" and z["low"] > price * 0.99]
    nearest_demand = max(demand, key=lambda z: z["high"]) if demand else None
    nearest_supply = min(supply, key=lambda z: z["low"]) if supply else None

    return {
        "zones": active[-20:],
        "nearest_demand": nearest_demand,
        "nearest_supply": nearest_supply,
    }


# ------------------------- PRICE ACTION / LIQUIDITY -------------------------


def candle_features(df: pd.DataFrame) -> dict:
    if len(df) < 5:
        return {}
    last = df.iloc[-1]
    prev = df.iloc[-2]
    body = abs(last["close"] - last["open"])
    rng = max(last["high"] - last["low"], 1e-12)
    upper = last["high"] - max(last["open"], last["close"])
    lower = min(last["open"], last["close"]) - last["low"]
    bullish = last["close"] > last["open"]
    bearish = last["close"] < last["open"]

    rejection = lower / rng >= 0.55 if bullish else upper / rng >= 0.55 if bearish else max(lower, upper) / rng >= 0.6
    pin_bar = max(upper, lower) / rng >= 0.65 and body / rng <= 0.35
    bullish_engulf = (
        prev["close"] < prev["open"] and bullish and
        last["open"] <= prev["close"] and last["close"] >= prev["open"]
    )
    bearish_engulf = (
        prev["close"] > prev["open"] and bearish and
        last["open"] >= prev["close"] and last["close"] <= prev["open"]
    )

    median_range = (df["high"] - df["low"]).tail(20).median()
    displacement = bool(median_range and rng >= 1.5 * median_range and body / rng >= 0.65)

    recent = df.tail(8)
    compression = bool(
        len(recent) >= 5 and
        (recent["high"].max() - recent["low"].min()) <
        1.8 * float(df["atr"].iloc[-1]) * 3
    )

    return {
        "bullish": bullish,
        "bearish": bearish,
        "rejection": rejection,
        "pin_bar": pin_bar,
        "bullish_engulfing": bullish_engulf,
        "bearish_engulfing": bearish_engulf,
        "displacement": displacement,
        "compression": compression,
        "body_pct_range": body / rng,
    }


def detect_breakout_retest(df: pd.DataFrame, sr: dict) -> dict:
    price = float(df["close"].iloc[-1])
    prev = float(df["close"].iloc[-2])
    resistance = sr.get("nearest_resistance")
    support = sr.get("nearest_support")

    breakout = None
    retest = None
    failed = None
    if resistance:
        breakout = price > resistance and prev <= resistance
        retest = price > resistance and float(df["low"].iloc[-1]) <= resistance * 1.003
        failed = prev > resistance and price < resistance
    if support:
        breakout = breakout or (price < support and prev >= support)
        retest = retest or (price < support and float(df["high"].iloc[-1]) >= support * 0.997)
        failed = failed or (prev < support and price > support)

    return {
        "breakout": bool(breakout),
        "retest": bool(retest),
        "failed_breakout": bool(failed),
    }


def detect_liquidity(df: pd.DataFrame, sr: dict, structure: dict) -> dict:
    highs, lows = structure["swings"]["highs"], structure["swings"]["lows"]
    recent = df.tail(60)
    price = float(df["close"].iloc[-1])

    equal_highs = False
    equal_lows = False
    if len(highs) >= 2:
        equal_highs = abs(highs[-1]["price"] - highs[-2]["price"]) / price * 100 <= 0.35
    if len(lows) >= 2:
        equal_lows = abs(lows[-1]["price"] - lows[-2]["price"]) / price * 100 <= 0.35

    sweep = None
    last = df.iloc[-1]
    if sr.get("nearest_support"):
        level = sr["nearest_support"]
        if last["low"] < level and last["close"] > level:
            sweep = "support liquidity sweep + reclaim"
    if sr.get("nearest_resistance"):
        level = sr["nearest_resistance"]
        if last["high"] > level and last["close"] < level:
            sweep = "resistance liquidity sweep + rejection"

    prev_day_high = sr["resistances"][-2] if len(sr["resistances"]) >= 2 else None
    prev_day_low = sr["supports"][-2] if len(sr["supports"]) >= 2 else None

    return {
        "equal_highs": equal_highs,
        "equal_lows": equal_lows,
        "sweep": sweep,
        "major_swing_high": highs[-1]["price"] if highs else None,
        "major_swing_low": lows[-1]["price"] if lows else None,
        "previous_reference_high": prev_day_high,
        "previous_reference_low": prev_day_low,
        "recent_range_high": float(recent["high"].max()),
        "recent_range_low": float(recent["low"].min()),
    }


# ------------------------- MOMENTUM / VOLATILITY / REGIME -------------------------


def detect_divergence(df: pd.DataFrame, indicator_col: str) -> Optional[str]:
    if len(df) < 40:
        return None
    segment = df.tail(50).copy()
    highs, lows = find_swings(segment, 3, 3)
    if len(highs) >= 2:
        a, b = highs[-2], highs[-1]
        if segment[indicator_col].iloc[b["index"]] < segment[indicator_col].iloc[a["index"]] and b["price"] > a["price"]:
            return "bearish divergence"
    if len(lows) >= 2:
        a, b = lows[-2], lows[-1]
        if segment[indicator_col].iloc[b["index"]] > segment[indicator_col].iloc[a["index"]] and b["price"] < a["price"]:
            return "bullish divergence"
    return None


def momentum_features(df: pd.DataFrame) -> dict:
    last = df.iloc[-1]
    prev = df.iloc[-2]
    rsi = float(last["rsi"])
    macd_hist = float(last["macd_hist"])
    prev_hist = float(prev["macd_hist"])
    return {
        "rsi": rsi,
        "rsi_state": "overbought" if rsi >= 70 else "oversold" if rsi <= 30 else "bullish" if rsi > 50 else "bearish",
        "rsi_50_reclaim": bool(prev["rsi"] <= 50 < rsi),
        "rsi_50_rejection": bool(prev["rsi"] >= 50 > rsi),
        "rsi_direction": "rising" if rsi > float(prev["rsi"]) else "falling" if rsi < float(prev["rsi"]) else "flat",
        "rsi_divergence": detect_divergence(df, "rsi"),
        "macd": float(last["macd"]),
        "macd_signal": float(last["macd_signal"]),
        "macd_hist": macd_hist,
        "macd_cross": "bullish crossover" if prev["macd"] <= prev["macd_signal"] < last["macd"] else "bearish crossover" if prev["macd"] >= prev["macd_signal"] > last["macd"] else None,
        "macd_hist_direction": "expanding" if abs(macd_hist) > abs(prev_hist) else "contracting",
        "macd_divergence": detect_divergence(df, "macd"),
    }


def volatility_features(df: pd.DataFrame) -> dict:
    last = df.iloc[-1]
    atr_pct = float(last["atr_pct"])
    atr_ratio = float(last["atr_ratio"]) if pd.notna(last["atr_ratio"]) else 1.0
    return {
        "atr": float(last["atr"]),
        "atr_pct": atr_pct,
        "atr_ratio": atr_ratio,
        "state": "high volatility" if atr_ratio >= 1.35 else "low volatility" if atr_ratio <= 0.75 else "normal volatility",
        "expanding": atr_ratio > 1.05,
        "contracting": atr_ratio < 0.95,
    }


def regime_features(df: pd.DataFrame, structure: dict, momentum: dict, volatility: dict) -> dict:
    price = float(df["close"].iloc[-1])
    ema9 = float(df["ema9"].iloc[-1])
    ema21 = float(df["ema21"].iloc[-1])
    ema50 = float(df["ema50"].iloc[-1])
    ema200 = float(df["ema200"].iloc[-1])
    ma_alignment = "bullish" if ema9 > ema21 > ema50 > ema200 else "bearish" if ema9 < ema21 < ema50 < ema200 else "mixed"
    trend = structure["trend"]
    if trend == "range" and volatility["state"] == "high volatility":
        regime = "range + high volatility"
    elif trend == "range":
        regime = "range"
    elif volatility["state"] == "high volatility":
        regime = f"{trend} + high volatility"
    else:
        regime = trend

    return {
        "regime": regime,
        "ma_alignment": ma_alignment,
        "price_vs_ema200": "above" if price > ema200 else "below",
    }


def overextension_features(df: pd.DataFrame, sr: dict) -> dict:
    last = df.iloc[-1]
    price = float(last["close"])
    atr = float(last["atr"])
    move_10 = abs(price / float(df["close"].iloc[-11]) - 1) * 100 if len(df) >= 11 else 0
    dist_ema21_atr = abs(price - float(last["ema21"])) / atr if atr else 0
    consecutive = 0
    direction = 1 if last["close"] > last["open"] else -1 if last["close"] < last["open"] else 0
    for i in range(len(df) - 1, max(-1, len(df) - 8), -1):
        cdir = 1 if df["close"].iloc[i] > df["open"].iloc[i] else -1 if df["close"].iloc[i] < df["open"].iloc[i] else 0
        if cdir == direction and direction != 0:
            consecutive += 1
        else:
            break
    resistance_close = sr.get("resistance_distance_pct") is not None and sr["resistance_distance_pct"] < 1.0
    support_close = sr.get("support_distance_pct") is not None and sr["support_distance_pct"] < 1.0
    overextended = dist_ema21_atr >= 3.0 or move_10 >= 10.0 or consecutive >= 6
    return {
        "overextended": bool(overextended),
        "distance_ema21_atr": dist_ema21_atr,
        "recent_move_10_pct": move_10,
        "consecutive_same_direction": consecutive,
        "near_resistance": resistance_close,
        "near_support": support_close,
    }


# ------------------------- TIMEFRAME ANALYSIS -------------------------


def analyze_timeframe(symbol: str, tf: str, cache: Optional[dict] = None) -> Optional[dict]:
    df = fetch_klines(symbol, tf, limit=260)
    if len(df) < 80:
        return None
    df = add_indicators(df)
    structure = analyze_structure(df)
    sr = detect_sr_zones(df, structure)
    sd = detect_supply_demand(df, structure)
    candle = candle_features(df)
    breakout = detect_breakout_retest(df, sr)
    liquidity = detect_liquidity(df, sr, structure)
    momentum = momentum_features(df)
    volatility = volatility_features(df)
    regime = regime_features(df, structure, momentum, volatility)
    overextension = overextension_features(df, sr)
    last = df.iloc[-1]

    # EMA9/21 is retained as context, not as the primary directional rule.
    context_bias = 1 if last["ema9"] > last["ema21"] else -1 if last["ema9"] < last["ema21"] else 0

    return {
        "tf": tf,
        "df": df,
        "close": float(last["close"]),
        "structure": structure,
        "sr": sr,
        "supply_demand": sd,
        "candle": candle,
        "breakout": breakout,
        "liquidity": liquidity,
        "momentum": momentum,
        "volatility": volatility,
        "regime": regime,
        "overextension": overextension,
        "context_bias": context_bias,
        "volume": {
            "relative_volume": float(last["relative_volume"]) if pd.notna(last["relative_volume"]) else 1.0,
            "average": float(last["vol_avg"]) if pd.notna(last["vol_avg"]) else 0,
            "spike": bool(pd.notna(last["relative_volume"]) and last["relative_volume"] >= 1.8),
            "expansion": bool(len(df) >= 3 and df["volume"].iloc[-1] > df["volume"].iloc[-2] > df["volume"].iloc[-3]),
            "contraction": bool(len(df) >= 3 and df["volume"].iloc[-1] < df["volume"].iloc[-2] < df["volume"].iloc[-3]),
        },
    }


# ------------------------- BTC / ALT CONTEXT -------------------------


def btc_context(btc: dict, alt: dict) -> dict:
    b = btc["tf_results"]
    a = alt["tf_results"]
    btc_4h = b.get("4h")
    btc_1h = b.get("1h")
    alt_4h = a.get("4h")
    if not btc_4h or not alt_4h:
        return {"status": "unknown", "score": 0, "relative_strength": 0}

    btc_bias = btc_4h["structure"]["bias"]
    alt_bias = alt_4h["structure"]["bias"]
    alt_return = (a["1h"]["close"] / a["1h"]["df"]["close"].iloc[-5] - 1) if btc_1h and len(a["1h"]["df"]) >= 5 else 0
    btc_return = (btc_1h["close"] / btc_1h["df"]["close"].iloc[-5] - 1) if btc_1h and len(btc_1h["df"]) >= 5 else 0
    relative_strength = alt_return - btc_return

    score = 0
    if btc_bias == alt_bias and btc_bias != 0:
        score += 1
    elif btc_bias != 0 and alt_bias != 0 and btc_bias != alt_bias:
        score -= 1
    if relative_strength > 0.01:
        score += 1
    elif relative_strength < -0.01:
        score -= 1

    status = "aligned" if score > 0 else "conflict" if score < 0 else "neutral"
    return {
        "status": status,
        "score": score,
        "relative_strength": relative_strength,
        "btc_bias": btc_bias,
        "alt_bias": alt_bias,
    }


# ------------------------- OPPORTUNITY FILTER -------------------------


def opportunity_score(symbol: str, ticker: dict) -> float:
    """Cheap pre-filter before expensive five-timeframe analysis."""
    change = abs(float(ticker.get("priceChangePercent", 0) or 0))
    volume = float(ticker.get("quoteVolume", 0) or 0)
    return math.log10(max(volume, 1)) + min(change, 20) * 0.1


def select_deep_analysis_candidates(symbols: list[str], stats: dict) -> list[str]:
    tickers = stats["tickers"]
    ranked = sorted(
        symbols,
        key=lambda s: opportunity_score(s, tickers.get(s, {})),
        reverse=True,
    )
    # BTC/ETH are always retained, then the most active candidates.
    output = []
    for s in CORE_COINS + ranked:
        if s in symbols and s not in output:
            output.append(s)
        if len(output) >= MAX_DEEP_ANALYSIS:
            break
    return output


# ------------------------- ENTRY / RISK ENGINE -------------------------


def directional_score(result: dict, direction: int) -> tuple[float, dict]:
    """Transparent 12-point setup score."""
    tf = result["tf_results"]
    htf8 = tf.get("8h")
    htf4 = tf.get("4h")
    one = tf.get("1h")
    m30 = tf.get("30m")
    m15 = tf.get("15m")
    if not all([htf8, htf4, one, m30, m15]):
        return 0, {}

    factors = {
        "market_structure": 0,
        "location": 0,
        "price_action": 0,
        "volume": 0,
        "momentum": 0,
        "trend": 0,
        "mtf_alignment": 0,
        "market_context": 0,
    }

    # 2 points: HTF structure + lower timeframe structure confirmation.
    if htf8["structure"]["bias"] == direction and htf4["structure"]["bias"] == direction:
        factors["market_structure"] += 1
    if (m15["structure"]["choch"] == ("bullish CHoCH" if direction == 1 else "bearish CHoCH") or
            m30["structure"]["bos"] == ("bullish BOS" if direction == 1 else "bearish BOS")):
        factors["market_structure"] += 1

    # Location: meaningful S/R or supply/demand area.
    loc = one["sr"]["location"]
    sd = one["supply_demand"]
    good_loc = (direction == 1 and ("support" in loc or sd["nearest_demand"])) or \
               (direction == -1 and ("resistance" in loc or sd["nearest_supply"]))
    if good_loc:
        factors["location"] += 2
    elif loc not in ("middle of nowhere", ""):
        factors["location"] += 1

    # Price action confirmation in 15m/30m.
    for r in [m15, m30]:
        c = r["candle"]
        bullish_pa = c.get("bullish_engulfing") or c.get("displacement") or c.get("rejection") or r["breakout"]["retest"]
        bearish_pa = c.get("bearish_engulfing") or c.get("displacement") or c.get("rejection") or r["breakout"]["retest"]
        if (direction == 1 and bullish_pa) or (direction == -1 and bearish_pa):
            factors["price_action"] += 1
            break
    if m15["liquidity"]["sweep"]:
        factors["price_action"] = min(2, factors["price_action"] + 1)

    # Volume confirmation.
    if m15["volume"]["relative_volume"] >= 1.2 or m30["volume"]["relative_volume"] >= 1.2:
        factors["volume"] = 1

    # Momentum.
    mom = m15["momentum"]
    if direction == 1:
        momentum_ok = mom["rsi"] > 50 and mom["rsi_direction"] == "rising" and mom["macd_hist"] > 0
        if mom["rsi_divergence"] == "bullish divergence" or mom["macd_divergence"] == "bullish divergence":
            momentum_ok = True
    else:
        momentum_ok = mom["rsi"] < 50 and mom["rsi_direction"] == "falling" and mom["macd_hist"] < 0
        if mom["rsi_divergence"] == "bearish divergence" or mom["macd_divergence"] == "bearish divergence":
            momentum_ok = True
    factors["momentum"] = int(momentum_ok)

    # Trend context, including MA50/200 without using them as a standalone buy/sell trigger.
    trend_ok = htf4["structure"]["bias"] == direction and htf4["regime"]["ma_alignment"] in (
        "bullish" if direction == 1 else "bearish", "mixed"
    )
    factors["trend"] = int(trend_ok)

    # MTF alignment: lower timeframe can confirm but not override HTF bias.
    if htf8["structure"]["bias"] == direction and htf4["structure"]["bias"] == direction:
        factors["mtf_alignment"] += 1
    if one["structure"]["bias"] == direction or m30["structure"]["bias"] == direction or m15["structure"]["bias"] == direction:
        factors["mtf_alignment"] += 1

    # Market context: BTC alignment is supplied separately.
    btc_score = result.get("btc_context", {}).get("score", 0)
    if btc_score > 0 and direction == 1:
        factors["market_context"] = 1
    elif btc_score < 0 and direction == -1:
        factors["market_context"] = 1
    elif symbol_is_btc(result["symbol"]):
        factors["market_context"] = 1

    total = sum(factors.values())
    return total, factors


def symbol_is_btc(symbol: str) -> bool:
    return symbol == "BTCUSDT"


def find_structural_stop(result: dict, direction: int, entry: float) -> Optional[float]:
    tf15 = result["tf_results"]["15m"]
    tf30 = result["tf_results"]["30m"]
    candidates = []
    for tf in [tf15, tf30, result["tf_results"]["1h"]]:
        swings = tf["structure"]["swings"]
        if direction == 1:
            for x in swings["lows"][-6:]:
                if x["price"] < entry:
                    candidates.append(x["price"])
        else:
            for x in swings["highs"][-6:]:
                if x["price"] > entry:
                    candidates.append(x["price"])

    if direction == 1:
        sd = tf15["supply_demand"]["nearest_demand"]
        if sd:
            candidates.append(sd["low"])
        return min(candidates) if candidates else None
    sd = tf15["supply_demand"]["nearest_supply"]
    if sd:
        candidates.append(sd["high"])
    return max(candidates) if candidates else None


def find_targets(result: dict, direction: int, entry: float, stop: float) -> list[float]:
    candidates = []
    for tf in [result["tf_results"]["15m"], result["tf_results"]["1h"], result["tf_results"]["4h"], result["tf_results"]["8h"]]:
        sr = tf["sr"]
        levels = sr["resistances"] if direction == 1 else sr["supports"]
        for level in levels:
            if (direction == 1 and level > entry) or (direction == -1 and level < entry):
                candidates.append(level)
        liq = tf["liquidity"]
        level = liq["major_swing_high"] if direction == 1 else liq["major_swing_low"]
        if level and ((direction == 1 and level > entry) or (direction == -1 and level < entry)):
            candidates.append(level)

    risk = abs(entry - stop)
    if risk <= 0:
        return []
    candidates = sorted(set(round(x, 10) for x in candidates), reverse=direction == -1)
    targets = [x for x in candidates if abs(x - entry) / risk >= MIN_RR]
    return targets[:3]


def evaluate_entry_sequence(result: dict, direction: int) -> tuple[bool, list[str]]:
    tf = result["tf_results"]
    htf = tf["4h"]["structure"]["bias"] == direction and tf["8h"]["structure"]["bias"] == direction
    one = tf["1h"]
    m15 = tf["15m"]
    m30 = tf["30m"]
    reasons = []

    if not htf:
        reasons.append("higher timeframe conflict")

    expected_choch = "bullish CHoCH" if direction == 1 else "bearish CHoCH"
    expected_bos = "bullish BOS" if direction == 1 else "bearish BOS"
    confirmation = m15["structure"]["choch"] == expected_choch or m15["structure"]["bos"] == expected_bos or m30["structure"]["choch"] == expected_choch or m30["structure"]["bos"] == expected_bos
    if not confirmation:
        reasons.append("no lower timeframe CHoCH/BOS confirmation")

    loc = one["sr"]["location"]
    sd = one["supply_demand"]
    location_ok = (direction == 1 and ("support" in loc or sd["nearest_demand"])) or (direction == -1 and ("resistance" in loc or sd["nearest_supply"]))
    if not location_ok:
        reasons.append("no favorable support/resistance or supply/demand location")

    sweep = m15["liquidity"]["sweep"] or m30["liquidity"]["sweep"]
    pa = m15["candle"]
    displacement = pa["displacement"] or m30["candle"]["displacement"]
    if not (sweep or displacement or pa["rejection"] or pa["bullish_engulfing"] or pa["bearish_engulfing"]):
        reasons.append("no price-action confirmation")

    volume_ok = m15["volume"]["relative_volume"] >= 1.0 or m30["volume"]["relative_volume"] >= 1.0
    if not volume_ok:
        reasons.append("weak volume")

    return len(reasons) == 0, reasons


def build_trade(result: dict, direction: int) -> dict:
    entry = result["tf_results"]["15m"]["close"]
    stop = find_structural_stop(result, direction, entry)
    if stop is None:
        return {"valid": False, "reasons": ["no structural invalidation level"]}

    # ATR sanity check: reject absurdly tight or huge structural stops.
    atr = result["tf_results"]["15m"]["volatility"]["atr"]
    risk = abs(entry - stop)
    if risk < atr * 0.25:
        stop = entry - atr * 0.5 if direction == 1 else entry + atr * 0.5
        risk = abs(entry - stop)
    if risk > atr * 5:
        return {"valid": False, "reasons": ["structural stop is too wide for current volatility"]}

    targets = find_targets(result, direction, entry, stop)
    if not targets:
        return {"valid": False, "reasons": ["no logical target provides minimum R:R"]}

    rr_values = [abs(tp - entry) / risk for tp in targets]
    return {
        "valid": rr_values[0] >= MIN_RR,
        "entry": entry,
        "stop_loss": stop,
        "targets": targets,
        "risk_distance": risk,
        "risk_pct": risk / entry * 100,
        "rr": rr_values[0],
        "reasons": [] if rr_values[0] >= MIN_RR else ["poor R:R"],
    }


# ------------------------- COIN ANALYSIS -------------------------


def analyze_coin(symbol: str, btc_result: Optional[dict] = None, news: Optional[dict] = None) -> dict:
    tf_results = {}
    errors = []
    for tf in TIMEFRAMES:
        try:
            tf_results[tf] = analyze_timeframe(symbol, tf)
        except Exception as exc:
            tf_results[tf] = None
            errors.append(f"{tf}: {exc}")

    result = {
        "symbol": symbol,
        "tf_results": tf_results,
        "errors": errors,
        "status": "NO TRADE",
        "has_setup": False,
        "direction": "NEUTRAL",
        "setup_score": 0,
        "factor_scores": {},
        "rejection_reasons": [],
    }

    if any(tf_results.get(tf) is None for tf in TIMEFRAMES):
        result["rejection_reasons"] = ["insufficient or unavailable timeframe data"]
        return result

    if btc_result and symbol != "BTCUSDT":
        result["btc_context"] = btc_context(btc_result, result)
    else:
        result["btc_context"] = {"status": "self", "score": 0}

    # Determine HTF bias first. Lower timeframe signals cannot override it.
    htf8 = tf_results["8h"]["structure"]["bias"]
    htf4 = tf_results["4h"]["structure"]["bias"]
    if htf8 == htf4 == 1:
        direction = 1
    elif htf8 == htf4 == -1:
        direction = -1
    else:
        direction = 0

    if direction == 0:
        result["rejection_reasons"].append("higher timeframe conflict or unclear bias")
        result["status"] = "WATCH" if tf_results["1h"]["structure"]["trend"] != "range" else "NO TRADE"
        return result

    result["direction"] = "BUY" if direction == 1 else "SELL"

    score, factors = directional_score(result, direction)
    result["setup_score"] = score
    result["factor_scores"] = factors

    sequence_ok, sequence_reasons = evaluate_entry_sequence(result, direction)
    result["rejection_reasons"].extend(sequence_reasons)

    # Hard filters.
    for tf_name in ["1h", "15m"]:
        if tf_results[tf_name]["overextension"]["overextended"]:
            result["rejection_reasons"].append("overextended")
            break

    # Avoid buying directly into major resistance / selling directly into major support.
    if direction == 1 and tf_results["1h"]["overextension"]["near_resistance"]:
        result["rejection_reasons"].append("resistance too close")
    if direction == -1 and tf_results["1h"]["overextension"]["near_support"]:
        result["rejection_reasons"].append("support too close")

    btc_ctx = result["btc_context"]
    if symbol != "BTCUSDT" and btc_ctx.get("status") == "conflict":
        result["rejection_reasons"].append("BTC conflict")

    if news and news.get("risk") == "HIGH":
        result["rejection_reasons"].append("high-impact news risk")

    trade = build_trade(result, direction)
    result["trade"] = trade
    if not trade.get("valid"):
        result["rejection_reasons"].extend(trade.get("reasons", []))

    # De-duplicate while preserving order.
    result["rejection_reasons"] = list(dict.fromkeys(result["rejection_reasons"]))

    hard_fail = any(x in result["rejection_reasons"] for x in [
        "higher timeframe conflict or unclear bias",
        "poor R:R",
        "no structural invalidation level",
        "structural stop is too wide for current volatility",
        "overextended",
        "resistance too close",
        "support too close",
        "BTC conflict",
        "high-impact news risk",
    ])

    if score >= MIN_SETUP_SCORE and sequence_ok and trade.get("valid") and not hard_fail:
        result["status"] = "CONFIRMED"
        result["has_setup"] = True
    elif score >= max(5, MIN_SETUP_SCORE - 2):
        result["status"] = "WAITING FOR CONFIRMATION"
    elif tf_results["1h"]["sr"]["location"] != "middle of nowhere":
        result["status"] = "SETUP FORMING"
    else:
        result["status"] = "WATCH"

    return result


# ------------------------- OUTPUT / TELEGRAM -------------------------


def fmt_price(value: float) -> str:
    if value >= 1000:
        return f"{value:,.2f}"
    if value >= 1:
        return f"{value:,.4f}"
    return f"{value:.8f}".rstrip("0").rstrip(".")


def build_reasoning(result: dict) -> str:
    tf = result["tf_results"]
    direction = result["direction"]
    lines = []
    for name in ["8h", "4h", "1h", "30m", "15m"]:
        r = tf[name]
        lines.append(
            f"{name}: {r['structure']['trend']}, "
            f"BOS={r['structure']['bos'] or 'none'}, "
            f"CHoCH={r['structure']['choch'] or 'none'}, "
            f"RSI={r['momentum']['rsi']:.0f}, "
            f"RVOL={r['volume']['relative_volume']:.2f}"
        )

    pa = tf["15m"]["candle"]
    liq = tf["15m"]["liquidity"]["sweep"]
    location = tf["1h"]["sr"]["location"]
    if result["has_setup"]:
        action = "bullish" if direction == "BUY" else "bearish"
        lines.append(
            f"Context: {action} higher timeframe structure, {location}, "
            f"15m price action confirmed."
        )
        if liq:
            lines.append(f"Liquidity: {liq}.")
        if pa["displacement"]:
            lines.append("Price action: displacement detected.")
    return "\n".join(lines)


def format_confirmed(result: dict, rank: int) -> str:
    tf = result["tf_results"]
    trade = result["trade"]
    targets = trade["targets"]
    ctx = result.get("btc_context", {})
    score = result["factor_scores"]
    return (
        f"#{rank} {result['symbol']}\n"
        f"Bias: {result['direction']}\n"
        f"Market regime: {tf['4h']['regime']['regime']}\n"
        f"8H: {tf['8h']['structure']['trend']}\n"
        f"4H: {tf['4h']['structure']['trend']} | "
        f"{tf['4h']['structure']['last_high']['label'] if tf['4h']['structure']['last_high'] else 'n/a'} + "
        f"{tf['4h']['structure']['last_low']['label'] if tf['4h']['structure']['last_low'] else 'n/a'}\n"
        f"1H location: {tf['1h']['sr']['location']}\n"
        f"15M confirmation: {tf['15m']['structure']['choch'] or tf['15m']['structure']['bos'] or 'confirmed price action'}\n"
        f"Support: {fmt_price(tf['1h']['sr']['nearest_support']) if tf['1h']['sr']['nearest_support'] else 'n/a'}\n"
        f"Resistance: {fmt_price(tf['1h']['sr']['nearest_resistance']) if tf['1h']['sr']['nearest_resistance'] else 'n/a'}\n"
        f"Liquidity: {tf['15m']['liquidity']['sweep'] or 'no recent sweep'}\n"
        f"Volume: RVOL {tf['15m']['volume']['relative_volume']:.2f}x\n"
        f"RSI: {tf['15m']['momentum']['rsi']:.0f} ({tf['15m']['momentum']['rsi_state']})\n"
        f"MACD: {'bullish' if tf['15m']['momentum']['macd_hist'] > 0 else 'bearish'} histogram\n"
        f"MA context: {tf['4h']['regime']['ma_alignment']}\n"
        f"Entry: {fmt_price(trade['entry'])}\n"
        f"SL: {fmt_price(trade['stop_loss'])}\n"
        f"TP1: {fmt_price(targets[0])}\n"
        f"TP2: {fmt_price(targets[1]) if len(targets) > 1 else 'n/a'}\n"
        f"R:R: 1:{trade['rr']:.1f}\n"
        f"Setup score: {result['setup_score']}/12\n"
        f"Factors: Structure {score['market_structure']}/2, Location {score['location']}/2, "
        f"PA {score['price_action']}/2, Volume {score['volume']}/1, Momentum {score['momentum']}/1, "
        f"Trend {score['trend']}/1, MTF {score['mtf_alignment']}/2, Context {score['market_context']}/1\n"
        f"BTC/ALT: {ctx.get('status', 'self')}\n"
        f"Status: CONFIRMED\n\n"
        f"Why:\n{build_reasoning(result)}"
    )


def format_watch(result: dict) -> str:
    reasons = ", ".join(result["rejection_reasons"][:3]) or "confirmation still developing"
    return f"• {result['symbol']}: {result['status']} | score {result['setup_score']}/12 | {reasons}"


def send_telegram_message(text: str) -> None:
    if "PUT_YOUR" in TELEGRAM_TOKEN or "PUT_YOUR" in TELEGRAM_CHAT_ID:
        print("[WARN] Telegram not configured — printing instead:\n", text)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        resp = SESSION.post(
            url,
            data={"chat_id": TELEGRAM_CHAT_ID, "text": text},
            timeout=10,
        )
        resp.raise_for_status()
    except Exception as exc:
        print(f"[ERROR] Failed to send Telegram message: {exc}")


def send_long_message(text: str) -> None:
    if len(text) <= TELEGRAM_MAX_LEN:
        send_telegram_message(text)
        return
    chunks = []
    current = ""
    for block in text.split("\n\n"):
        if current and len(current) + len(block) + 2 > TELEGRAM_MAX_LEN:
            chunks.append(current)
            current = block
        else:
            current = f"{current}\n\n{block}" if current else block
    if current:
        chunks.append(current)
    for chunk in chunks:
        send_telegram_message(chunk)


# ------------------------- MAIN CYCLE -------------------------


def run_cycle():
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    sections = [f"📊 Multi-Factor Crypto Market Scan — {ts}"]

    headlines = fetch_macro_headlines()
    news = news_context(headlines)
    if headlines:
        sections.append(
            "News context: " + news["reason"] + "\n" +
            "\n".join(f"• {h['title']} [{h['sentiment']}{', HIGH IMPACT' if h['high_impact'] else ''}]" for h in headlines[:4])
        )

    symbols, market_stats = build_market_universe()
    deep_symbols = select_deep_analysis_candidates(symbols, market_stats)

    sections.append(
        f"Universe: {market_stats['exchange_usdt_pairs']} USDT pairs | "
        f"liquid candidates: {market_stats['liquid_candidates']} | "
        f"selected: {market_stats['selected']} | deep analysis: {len(deep_symbols)}"
    )

    # BTC first, because it is the market reference for altcoin context.
    btc_result = analyze_coin("BTCUSDT", btc_result=None, news=news) if "BTCUSDT" in deep_symbols else None
    results = []
    if btc_result:
        results.append(btc_result)

    for symbol in deep_symbols:
        if symbol == "BTCUSDT":
            continue
        try:
            results.append(analyze_coin(symbol, btc_result=btc_result, news=news))
        except Exception as exc:
            print(f"[ERROR] {symbol}: {exc}")

    confirmed = sorted(
        [r for r in results if r["has_setup"]],
        key=lambda r: (r["setup_score"], r.get("trade", {}).get("rr", 0)),
        reverse=True,
    )
    watch = [r for r in results if not r["has_setup"] and r["status"] in {"WATCH", "SETUP FORMING", "WAITING FOR CONFIRMATION"}]

    sections.append(
        f"Market scan complete.\n"
        f"Coins selected for deep analysis: {len(results)}\n"
        f"Potential/watch setups: {len(watch)}\n"
        f"Confirmed setups: {len(confirmed)}"
    )

    if confirmed:
        sections.append("🔥 CONFIRMED SETUPS")
        for rank, result in enumerate(confirmed[:10], start=1):
            sections.append(format_confirmed(result, rank))
    elif SEND_NO_TRADE_SUMMARY:
        sections.append(
            "NO HIGH QUALITY TRADE SETUP\n"
            "No market currently meets the minimum conditions for a confirmed entry.\n"
            "The bot does not force a trade quota."
        )

    if SEND_WATCHLIST and watch:
        watch_sorted = sorted(watch, key=lambda r: r["setup_score"], reverse=True)
        sections.append("👀 WATCHLIST / DEVELOPING")
        sections.append("\n".join(format_watch(r) for r in watch_sorted[:8]))

    sections.append(
        "⚠️ Rule-based technical/news scan only. Setup score is not a probability. "
        "Risk should be sized around the structural stop-loss."
    )

    send_long_message("\n\n".join(sections))


def main():
    if "--once" in sys.argv:
        run_cycle()
        return

    print(
        f"Starting multi-factor crypto bot. Checking every {SCAN_INTERVAL_MINUTES} minutes. Ctrl+C to stop.\n"
    )
    while True:
        try:
            run_cycle()
        except Exception:
            print("[ERROR] Problem during cycle:")
            traceback.print_exc()
        time.sleep(SCAN_INTERVAL_MINUTES * 60)


if __name__ == "__main__":
    main()
