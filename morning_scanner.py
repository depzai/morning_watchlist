"""
Morning Pre-Market Scanner
==========================
Runs at 9:00 AM ET and produces a ranked watchlist of high-conviction
intraday setups combining:
  1. Gap % from prior close  (yfinance -- wider universe, no Alpaca gaps)
  2. Volume ratio vs 20-day average  (yfinance)
  3. News sentiment (Finnhub)
  4. Daily Supertrend trend direction (yfinance)
  5. Relative strength vs SPY

Universe: S&P 500 + Nasdaq 100 + a broad mid-cap extension (~1,500 tickers)
Output  : ranked watchlist -> Google Sheets ("ranging" / morning_watchlist tab)

GitHub Actions secrets required:
    FINNHUB_API_KEY      finnhub.io free key
    GSPREAD_SA_KEY_JSON  Google service account JSON

Optional (not required any more for gap detection):
    ALPACA_API_KEY / ALPACA_SECRET_KEY  (kept for SPY baseline only)

Setup:
    pip install yfinance pandas numpy requests gspread google-auth
"""

import os
import time
import tempfile
import logging
import re
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo
from io import StringIO

import numpy as np
import pandas as pd
import requests
import yfinance as yf

try:
    import gspread
    from google.oauth2.service_account import Credentials
except ImportError:
    raise ImportError("pip install gspread google-auth")


# ===========================================================================
# CONFIG
# ===========================================================================

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")

# Scanner filters
MIN_PRICE      = 5.0
MIN_GAP_PCT    = 2.0
MAX_GAP_PCT    = 30.0
MIN_AVG_VOL    = 200_000   # 20-day avg volume floor
TOP_N          = 20

# Supertrend params
ATR_PERIOD = 10
MULTIPLIER = 3.0

# Scoring weights (must sum to 100)
W_GAP          = 25
W_VOLUME       = 25
W_SENTIMENT    = 20
W_SUPERTREND   = 20
W_REL_STRENGTH = 10

# Google Sheets
GSHEET_NAME   = "ranging"
WATCHLIST_TAB = "morning_watchlist"
ET            = ZoneInfo("America/New_York")

WATCHLIST_HEADERS = [
    "run_timestamp", "date", "ticker", "score",
    "gap_pct", "premarket_volume", "volume_ratio",
    "sentiment_score", "sentiment_label", "news_headline",
    "supertrend_signal", "rel_strength_vs_spy",
    "prior_close", "premarket_price",
    "stop_loss", "target",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ===========================================================================
# GOOGLE SHEETS
# ===========================================================================

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


def _gsheet_client():
    raw = os.environ.get("GSPREAD_SA_KEY_JSON")
    if not raw:
        raise EnvironmentError("GSPREAD_SA_KEY_JSON not set")
    tf = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    tf.write(raw)
    tf.flush()
    creds = Credentials.from_service_account_file(tf.name, scopes=SCOPES)
    return gspread.authorize(creds)


def _ensure_tab(spreadsheet, tab_name, headers):
    try:
        ws = spreadsheet.worksheet(tab_name)
    except gspread.WorksheetNotFound:
        ws = spreadsheet.add_worksheet(title=tab_name, rows=10000, cols=len(headers))
        log.info(f"  Created tab: {tab_name}")

    existing = ws.row_values(1)
    if existing != headers:
        if existing:
            ws.delete_rows(1)
        ws.insert_row(headers, index=1, value_input_option="RAW")
        log.info(f"  Wrote headers to tab: {tab_name}")

    return ws


def log_watchlist_to_sheet(watchlist: list, run_ts: str, today: str):
    if not watchlist:
        log.info("  No watchlist to log.")
        return
    try:
        gc  = _gsheet_client()
        ss  = gc.open(GSHEET_NAME)
        ws  = _ensure_tab(ss, WATCHLIST_TAB, WATCHLIST_HEADERS)
        rows = [[
            run_ts, today,
            w["ticker"],
            w["score"],
            round(w["gap_pct"], 2),
            w["premarket_volume"],
            round(w["volume_ratio"], 2),
            round(w["sentiment_score"], 3),
            w["sentiment_label"],
            w["news_headline"][:200] if w["news_headline"] else "",
            w["supertrend_signal"],
            round(w["rel_strength_vs_spy"], 3),
            round(w["prior_close"], 2),
            round(w["premarket_price"], 2),
            round(w["stop_loss"], 2),
            round(w["target"], 2),
        ] for w in watchlist]
        ws.append_rows(rows, value_input_option="RAW")
        log.info(f"  Logged {len(rows)} tickers to '{GSHEET_NAME}' / {WATCHLIST_TAB}")
    except Exception as e:
        log.error(f"  Failed to log watchlist: {e}")


# ===========================================================================
# 1. UNIVERSE
# ===========================================================================

def _get_sp500() -> list:
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; scanner/1.0)"}
        r = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers=headers, timeout=15,
        )
        r.raise_for_status()
        tickers = pd.read_html(r.text)[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
        log.info(f"  S&P 500: {len(tickers)} tickers")
        return tickers
    except Exception as e:
        log.warning(f"  S&P 500 scrape failed: {e}")
    return []


def _get_nasdaq100() -> list:
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; scanner/1.0)"}
        r = requests.get("https://en.wikipedia.org/wiki/Nasdaq-100", headers=headers, timeout=15)
        r.raise_for_status()
        for table in pd.read_html(r.text):
            for col in ("Ticker", "Symbol", "Ticker symbol"):
                if col in table.columns:
                    tickers = (table[col].astype(str).str.strip()
                               .str.replace(r"\[.*?\]", "", regex=True).tolist())
                    tickers = [t for t in tickers if t and t.lower() != "nan"]
                    if len(tickers) > 50:
                        log.info(f"  Nasdaq-100: {len(tickers)} tickers")
                        return tickers
    except Exception as e:
        log.warning(f"  Nasdaq-100 scrape failed: {e}")
    return []


def _get_russell1000() -> list:
    """
    Pull Russell 1000 tickers from iShares IWB ETF holdings CSV.
    This gives ~1000 mid/large cap US stocks -- much wider than S&P+NDX alone.
    """
    try:
        url = 
"https://www.ishares.com/us/products/239707/ishares-russell-1000-etf/1467271812596.ajax?fileType=csv&fileName=IWB_holdings&dataType=fund"
        r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        # iShares CSV has a few header rows before the actual data
        df = pd.read_csv(StringIO(r.text), skiprows=9)
        tickers = df["Ticker"].dropna().astype(str).str.strip().tolist()
        tickers = [t for t in tickers if t and t != "-" and t.lower() != "nan" and len(t) <= 5]
        log.info(f"  Russell 1000: {len(tickers)} tickers from iShares")
        return tickers
    except Exception as e:
        log.warning(f"  Russell 1000 fetch failed: {e}")
    return []


# Extended mid-cap supplement in case Russell fetch fails
_MIDCAP_SUPPLEMENT = [
    "PLTR","COIN","HOOD","SOFI","RIVN","LCID","NIO","XPEV","LI","DKNG",
    "RBLX","U","SNAP","PINS","SPOT","ABNB","DASH","UBER","LYFT","AFRM",
    "UPST","SQ","PYPL","SHOP","SE","MELI","GRAB","BIDU","JD","PDD",
    "BABA","TCOM","NTES","TME","BILI","IQ","VIPS","ZTO","YMM","QFIN",
    "CRWD","S","PANW","ZS","OKTA","TENB","CYBR","VRNS","QLYS","RDWR",
    "NET","DDOG","MDB","SNOW","GTLB","HCP","CFLT","MNDY","BRZE","ZI",
    "PATH","AI","C3AI","BBAI","SOUN","IREN","MARA","RIOT","CLSK","CORZ",
    "WULF","BTBT","HUT","BITF","CIFR","APLD","TSLA","NVDA","AMD","INTC",
    "QCOM","AVGO","MRVL","SWKS","QRVO","MPWR","MTSI","NXPI","ON","WOLF",
    "SMCI","DELL","HPE","NTAP","PSTG","STX","WDC","LOGI","ZBRA","TER",
    "AMAT","LRCX","KLAC","ASML","KLIC","UCTT","ONTO","ACLS","FORM","CAMT",
    "SPY","QQQ","IWM","DIA","XLF","XLK","XLE","XLV","XLI","XLP",
]


def get_universe() -> list:
    sp500   = _get_sp500()
    nasdaq  = _get_nasdaq100()
    russell = _get_russell1000()
    combined = list(dict.fromkeys(
        sp500 + nasdaq + russell + _MIDCAP_SUPPLEMENT
    ))
    # Remove ETFs and non-standard tickers for gap scanning
    combined = [t for t in combined if t and len(t) <= 5 and "." not in t]
    log.info(f"  Total universe: {len(combined)} unique tickers")
    return combined


# ===========================================================================
# 2. GAP DETECTION via yfinance (fixes missing gaps vs Alpaca)
# ===========================================================================

def get_gap_data(tickers: list) -> list:
    """
    Download 2 days of 1-minute pre-market data via yfinance to compute:
      - gap %  = (today open or latest price - yesterday close) / yesterday close
      - premarket volume  = sum of volume in pre-market session today
      - volume_ratio      = today premarket vol / 20-day avg daily vol

    Falls back to daily bars if pre-market data unavailable.
    """
    log.info(f"  Fetching gap data for {len(tickers)} tickers via yfinance ...")

    # Step 1: get 22 days of daily closes to compute 20-day avg volume + prev close
    # Download in bulk for speed
    chunk_size = 200
    daily_data = {}

    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        try:
            raw = yf.download(
                chunk,
                period="25d",
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            if isinstance(raw.columns, pd.MultiIndex):
                for t in chunk:
                    try:
                        df = raw.xs(t, axis=1, level=1).dropna()
                        if not df.empty:
                            daily_data[t] = df
                    except Exception:
                        pass
            else:
                # Single ticker returned
                if len(chunk) == 1 and not raw.empty:
                    daily_data[chunk[0]] = raw.dropna()
        except Exception as e:
            log.warning(f"  Daily bulk download chunk {i//chunk_size+1} error: {e}")
        time.sleep(0.3)

    log.info(f"  Daily data fetched for {len(daily_data)} tickers")

    # Step 2: get today's pre-market price using 1d/1m with prepost=True
    # Do this per-ticker only for those with meaningful daily data
    gaps = []
    no_premarket = 0

    for ticker, df_daily in daily_data.items():
        try:
            if len(df_daily) < 2:
                continue

            prev_close  = float(df_daily["Close"].iloc[-2])
            avg_vol_20d = float(df_daily["Volume"].iloc[-21:-1].mean()) if len(df_daily) >= 21 else float(df_daily["Volume"].mean())

            if prev_close <= 0 or avg_vol_20d < MIN_AVG_VOL:
                continue

            # Try to get today's latest price (pre-market or intraday)
            try:
                tick   = yf.Ticker(ticker)
                info   = tick.fast_info
                current_price = float(info.last_price) if hasattr(info, "last_price") and info.last_price else 0.0
                # If fast_info doesn't work, use today's daily bar open
                if current_price <= 0:
                    today_df = yf.download(ticker, period="1d", interval="1m",
                                           prepost=True, progress=False)
                    if not today_df.empty:
                        current_price = float(today_df["Close"].iloc[-1])
            except Exception:
                current_price = 0.0

            # Final fallback: use today's daily bar if available
            if current_price <= 0:
                current_price = float(df_daily["Close"].iloc[-1])

            if current_price < MIN_PRICE or current_price <= 0:
                continue

            gap_pct = ((current_price - prev_close) / prev_close) * 100

            if gap_pct < MIN_GAP_PCT or gap_pct > MAX_GAP_PCT:
                continue

            # Get today's volume so far
            try:
                today_df_v = yf.download(ticker, period="1d", interval="1m",
                                         prepost=True, progress=False)
                premarket_vol = int(today_df_v["Volume"].sum()) if not today_df_v.empty else 0
            except Exception:
                premarket_vol = 0

            volume_ratio = premarket_vol / avg_vol_20d if avg_vol_20d > 0 and premarket_vol > 0 else 0.0

            gaps.append({
                "ticker"          : ticker,
                "prior_close"     : round(prev_close, 2),
                "premarket_price" : round(current_price, 2),
                "premarket_volume": premarket_vol,
                "avg_vol_20d"     : int(avg_vol_20d),
                "gap_pct"         : round(gap_pct, 2),
                "volume_ratio"    : round(volume_ratio, 2),
            })

        except Exception as ex:
            log.debug(f"  Gap parse error {ticker}: {ex}")
            continue

    gaps.sort(key=lambda x: x["gap_pct"], reverse=True)
    log.info(f"  Gaps found: {len(gaps)} tickers with gap between {MIN_GAP_PCT}% and {MAX_GAP_PCT}%")
    return gaps


# ===========================================================================
# 3. NEWS SENTIMENT via Finnhub
# ===========================================================================

POSITIVE_WORDS = {
    "beat", "beats", "topped", "exceeded", "exceeds", "surpassed", "record",
    "raised", "raises", "boosted", "boosts", "lifted", "lifts", "increased",
    "increases", "above", "upbeat", "upgrade", "upgraded", "outperform",
    "overweight", "buy", "bullish", "breakout", "rallies", "rally", "surge",
    "surges", "soars", "soar", "jumps", "jump", "spikes", "spike", "climbs",
    "climb", "rises", "rise", "profit", "profits", "growth", "gains", "gain",
    "win", "wins", "deal", "merger", "acquisition", "partnership", "approval",
    "approved", "launch", "launches", "expands", "expansion", "positive",
    "strong", "stronger", "strength", "robust", "solid", "better",
}

NEGATIVE_WORDS = {
    "missed", "misses", "miss", "below", "disappointed", "disappoints",
    "disappointing", "cut", "cuts", "lowered", "lowers", "reduced", "reduces",
    "slashed", "slashes", "warned", "warns", "warning", "downgrade",
    "downgraded", "underperform", "underweight", "sell", "bearish", "falls",
    "fall", "drops", "drop", "slides", "slide", "slumps", "slump", "plunges",
    "plunge", "tumbles", "tumble", "declines", "decline", "sinks", "sink",
    "loss", "losses", "weak", "weaker", "weakness", "concern", "concerns",
    "risk", "risks", "lawsuit", "sued", "investigation", "probe", "recall",
    "halt", "halted", "suspended", "layoffs", "restructuring", "bankruptcy",
    "default", "negative", "pressure",
}


def score_headline(text: str) -> float:
    if not text:
        return 0.0
    text_lower = text.lower()
    pos   = sum(1 for w in POSITIVE_WORDS if w in text_lower)
    neg   = sum(1 for w in NEGATIVE_WORDS if w in text_lower)
    total = pos + neg
    if total == 0:
        return 0.0
    return round((pos - neg) / total, 3)


def fetch_news_sentiment(tickers: list) -> dict:
    if not FINNHUB_API_KEY:
        log.warning("  FINNHUB_API_KEY not set -- skipping sentiment")
        return {}

    log.info(f"  Fetching news sentiment for {len(tickers)} tickers via Finnhub ...")

    today_str     = datetime.now().strftime("%Y-%m-%d")
    yesterday_str = (datetime.now() - timedelta(hours=24)).strftime("%Y-%m-%d")
    results       = {}

    for ticker in tickers:
        # Finnhub uses plain ticker symbols -- strip any dots/dashes
        finnhub_ticker = ticker.replace("-", ".").replace(".", "-")
        try:
            r = requests.get(
                "https://finnhub.io/api/v1/company-news",
                params={
                    "symbol": ticker,
                    "from"  : yesterday_str,
                    "to"    : today_str,
                    "token" : FINNHUB_API_KEY,
                },
                timeout=10,
            )
            r.raise_for_status()
            articles = r.json()

            if not isinstance(articles, list) or not articles:
                results[ticker] = {"score": 0.0, "label": "neutral", "headline": ""}
                time.sleep(0.5)
                continue

            recent = articles[:5]
            scores = [
                score_headline(a.get("headline", "") + " " + a.get("summary", ""))
                for a in recent
            ]
            avg_score    = round(sum(scores) / len(scores), 3)
            top_headline = recent[0].get("headline", "")
            label        = "positive" if avg_score > 0.05 else "negative" if avg_score < -0.05 else "neutral"

            results[ticker] = {"score": avg_score, "label": label, "headline": top_headline}
            time.sleep(1.1)  # stay under 60 calls/min free tier

        except Exception as e:
            log.warning(f"  News fetch failed for {ticker}: {e}")
            results[ticker] = {"score": 0.0, "label": "neutral", "headline": ""}

    pos_ct  = sum(1 for v in results.values() if v["label"] == "positive")
    neg_ct  = sum(1 for v in results.values() if v["label"] == "negative")
    neut_ct = sum(1 for v in results.values() if v["label"] == "neutral")
    log.info(f"  Sentiment: {pos_ct} positive, {neg_ct} negative, {neut_ct} neutral")
    return results


# ===========================================================================
# 4. SUPERTREND SIGNAL (daily)
# ===========================================================================

def compute_supertrend(df: pd.DataFrame) -> int:
    try:
        high, low, close = df["High"], df["Low"], df["Close"]
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr     = tr.ewm(alpha=1 / ATR_PERIOD, adjust=False).mean()
        hl_avg  = (high + low) / 2
        upper   = (hl_avg + MULTIPLIER * atr).values.copy()
        lower   = (hl_avg - MULTIPLIER * atr).values.copy()
        close_v = close.values.copy()
        trend   = np.zeros(len(df), dtype=int)
        trend[0] = -1
        for i in range(1, len(df)):
            upper[i] = upper[i] if upper[i] < upper[i-1] or close_v[i-1] > upper[i-1] else upper[i-1]
            lower[i] = lower[i] if lower[i] > lower[i-1] or close_v[i-1] < lower[i-1] else lower[i-1]
            if trend[i-1] == -1:
                trend[i] = 1 if close_v[i] > upper[i-1] else -1
            else:
                trend[i] = -1 if close_v[i] < lower[i-1] else 1
        return int(trend[-1])
    except Exception:
        return 0


def get_supertrend_signals(tickers: list) -> dict:
    log.info(f"  Computing daily Supertrend for {len(tickers)} tickers ...")
    end   = datetime.today()
    start = end - timedelta(days=90)

    try:
        raw = yf.download(
            tickers,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception as e:
        log.warning(f"  yfinance Supertrend download failed: {e}")
        return {}

    signals = {}
    for ticker in tickers:
        try:
            df = raw.xs(ticker, axis=1, level=1).copy() if isinstance(raw.columns, pd.MultiIndex) else raw.copy()
            df = df.dropna()
            if len(df) >= ATR_PERIOD * 3:
                signals[ticker] = compute_supertrend(df)
        except Exception:
            continue

    return signals


# ===========================================================================
# 5. RELATIVE STRENGTH vs SPY
# ===========================================================================

def get_spy_change() -> float:
    try:
        spy = yf.Ticker("SPY")
        df  = yf.download("SPY", period="2d", interval="1d", auto_adjust=True, progress=False)
        if len(df) >= 2:
            prev  = float(df["Close"].iloc[-2])
            # Get latest price from fast_info
            curr = float(spy.fast_info.last_price)
            if prev > 0 and curr > 0:
                return ((curr - prev) / prev) * 100
    except Exception:
        pass
    return 0.0


# ===========================================================================
# 6. SCORING
# ===========================================================================

def score_ticker(gap_data: dict, sentiment: dict, supertrend: int,
                 spy_change: float) -> dict:
    gap_score    = min(100, max(0, (gap_data["gap_pct"] - MIN_GAP_PCT) / (10 - MIN_GAP_PCT) * 100))
    vol_score    = min(100, max(0, (gap_data["volume_ratio"] - 0.5) / 2.5 * 100))
    sent_raw     = sentiment.get("score", 0.0)
    sent_score   = min(100, max(0, (sent_raw + 1) / 2 * 100))
    st_score     = 100 if supertrend == 1 else 0 if supertrend == -1 else 50
    rel_strength = gap_data["gap_pct"] - spy_change
    rs_score     = min(100, max(0, (rel_strength + 5) / 10 * 100))

    composite = round(
        (gap_score * W_GAP          / 100) +
        (vol_score * W_VOLUME       / 100) +
        (sent_score * W_SENTIMENT   / 100) +
        (st_score  * W_SUPERTREND   / 100) +
        (rs_score  * W_REL_STRENGTH / 100),
        1
    )

    entry     = gap_data["premarket_price"]
    stop_loss = round(entry * 0.98, 2)
    target    = round(entry * 1.04, 2)

    return {
        "ticker"             : gap_data["ticker"],
        "score"              : composite,
        "gap_pct"            : gap_data["gap_pct"],
        "premarket_volume"   : gap_data["premarket_volume"],
        "volume_ratio"       : gap_data["volume_ratio"],
        "sentiment_score"    : sentiment.get("score", 0.0),
        "sentiment_label"    : sentiment.get("label", "neutral"),
        "news_headline"      : sentiment.get("headline", ""),
        "supertrend_signal"  : "bullish" if supertrend == 1 else "bearish" if supertrend == -1 else "neutral",
        "rel_strength_vs_spy": round(rel_strength, 3),
        "prior_close"        : gap_data["prior_close"],
        "premarket_price"    : entry,
        "stop_loss"          : stop_loss,
        "target"             : target,
    }


# ===========================================================================
# 7. MAIN
# ===========================================================================

def run_morning_scanner():
    run_ts = datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S ET")
    today  = date.today().isoformat()

    log.info("=" * 60)
    log.info("  MORNING PRE-MARKET SCANNER")
    log.info(f"  Run: {run_ts}")
    log.info("=" * 60)

    # Build universe
    tickers = get_universe()

    # Find gaps
    log.info("\n[1/5] Scanning for gaps ...")
    gap_list = get_gap_data(tickers)

    if not gap_list:
        log.warning("No tickers found gapping > %.1f%%. Market may be flat or data issue.", MIN_GAP_PCT)
        return

    gap_tickers = [g["ticker"] for g in gap_list]
    log.info(f"  {len(gap_tickers)} tickers gapping up, proceeding ...")

    # SPY baseline
    log.info("\n[2/5] Getting SPY baseline ...")
    spy_change = get_spy_change()
    log.info(f"  SPY vs prior close: {spy_change:+.2f}%")

    # News sentiment
    log.info("\n[3/5] Fetching news sentiment ...")
    sentiment_map = fetch_news_sentiment(gap_tickers)

    # Supertrend
    log.info("\n[4/5] Computing Supertrend signals ...")
    supertrend_map = get_supertrend_signals(gap_tickers)

    # Score and rank
    log.info("\n[5/5] Scoring and ranking ...")
    watchlist = []
    for gap_data in gap_list:
        ticker    = gap_data["ticker"]
        sentiment = sentiment_map.get(ticker, {"score": 0.0, "label": "neutral", "headline": ""})
        st_signal = supertrend_map.get(ticker, 0)
        scored    = score_ticker(gap_data, sentiment, st_signal, spy_change)
        watchlist.append(scored)

    watchlist.sort(key=lambda x: x["score"], reverse=True)
    watchlist = watchlist[:TOP_N]

    print(f"\n{'=' * 80}")
    print(f"  TOP {len(watchlist)} SETUPS -- {today}")
    print(f"{'=' * 80}")
    print(f"  {'#':<3} {'TICKER':<7} {'SCORE':<7} {'GAP%':<7} {'VOL_RATIO':<11} "
          f"{'SENTIMENT':<11} {'TREND':<10} {'ENTRY':<8} {'STOP':<8} {'TARGET'}")
    print(f"  {'-' * 75}")
    for i, w in enumerate(watchlist, 1):
        print(f"  {i:<3} {w['ticker']:<7} {w['score']:<7} "
              f"{w['gap_pct']:>+5.1f}%  "
              f"{w['volume_ratio']:>6.2f}x    "
              f"{w['sentiment_label']:<11} "
              f"{w['supertrend_signal']:<10} "
              f"${w['premarket_price']:<7.2f} "
              f"${w['stop_loss']:<7.2f} "
              f"${w['target']:.2f}")
    print()

    log.info("Logging to Google Sheets ...")
    log_watchlist_to_sheet(watchlist, run_ts, today)

    log.info("=" * 60)
    log.info(f"  Universe scanned        : {len(tickers)}")
    log.info(f"  Gapping tickers found   : {len(gap_list)}")
    log.info(f"  Top setups logged       : {len(watchlist)}")
    log.info(f"  SPY vs prior close      : {spy_change:+.2f}%")
    log.info("=" * 60)

    return watchlist


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    run_morning_scanner()

