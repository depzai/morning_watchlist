"""
Morning Pre-Market Scanner
==========================
Runs at 9:00 AM ET (before open) and produces a ranked watchlist of
high-conviction intraday setups combining:
  1. Gap % from prior close
  2. Pre-market volume vs 20-day average
  3. News sentiment (Finnhub)
  4. Daily Supertrend trend direction
  5. Relative strength vs SPY

Output: ranked watchlist -> Google Sheets ("ranging" / morning_watchlist tab)

GitHub Actions secrets required:
    ALPACA_API_KEY       Alpaca paper/live key
    ALPACA_SECRET_KEY    Alpaca secret
    FINNHUB_API_KEY      Finnhub API key (get at finnhub.io -- free tier works)
    GSPREAD_SA_KEY_JSON  Google service account JSON

Setup:
    pip install alpaca-trade-api yfinance pandas numpy requests gspread google-auth
"""

import os
import time
import tempfile
import logging
import re
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf

try:
    import alpaca_trade_api as tradeapi
except ImportError:
    raise ImportError("pip install alpaca-trade-api")

try:
    import gspread
    from google.oauth2.service_account import Credentials
except ImportError:
    raise ImportError("pip install gspread google-auth")


# ===========================================================================
# CONFIG
# ===========================================================================

ALPACA_API_KEY    = os.getenv("ALPACA_API_KEY",    "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL   = os.getenv("ALPACA_BASE_URL",   "https://paper-api.alpaca.markets")
FINNHUB_API_KEY   = os.getenv("FINNHUB_API_KEY",   "")

# Scanner filters
MIN_PRICE      = 5.0
MIN_AVG_VOLUME = 500_000
MIN_GAP_PCT    = 2.0
MAX_GAP_PCT    = 30.0
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
        ws.append_row(headers, value_input_option="RAW")
        log.info(f"  Created tab: {tab_name}")
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
# 1. UNIVERSE -- S&P 500 + Nasdaq 100
# ===========================================================================

def _get_sp500() -> list:
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; scanner/1.0)"}
        r = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers=headers,
            timeout=15,
        )
        r.raise_for_status()
        tables = pd.read_html(r.text)
        tickers = tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
        log.info(f"  S&P 500: {len(tickers)} tickers from Wikipedia")
        return tickers
    except Exception as e:
        log.warning(f"  Wikipedia S&P 500 failed: {e}")

    try:
        r = requests.get(
            "https://pkgstore.datahub.io/core/s-and-p-500-companies/constituents_csv/data/constituents_csv.csv",
            timeout=15,
        )
        r.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(r.text))
        tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
        log.info(f"  S&P 500: {len(tickers)} tickers from datahub.io")
        return tickers
    except Exception as e:
        log.warning(f"  datahub.io S&P 500 failed: {e}")

    log.warning("  Using hardcoded S&P 500 fallback (503 tickers)")
    return [
        "MMM","AOS","ABT","ABBV","ACN","ADBE","AMD","AES","AFL","A",
        "APD","ABNB","AKAM","ALB","ARE","ALGN","ALLE","LNT","ALL","GOOGL",
        "GOOG","MO","AMZN","AMCR","AEE","AAL","AEP","AXP","AIG","AMT",
        "AWK","AMP","AME","AMGN","APH","ADI","ANSS","AON","APA","AAPL",
        "AMAT","APTV","ACGL","ADM","ANET","AJG","AIZ","T","ATO","ADSK",
        "AZO","AVB","AVY","AXON","BKR","BALL","BAC","BK","BBWI","BAX",
        "BDX","BRK.B","BBY","BIO","TECH","BIIB","BLK","BX","BA","BCR",
        "BMY","AVGO","BR","BRO","BF.B","BLDR","BG","CDNS","CZR","CPT",
        "CPB","COF","CAH","KMX","CCL","CARR","CTLT","CAT","CBOE","CBRE",
        "CDW","CE","COR","CNC","CNX","CDAY","CF","CRL","SCHW","CHTR",
        "CVX","CMG","CB","CHD","CI","CINF","CTAS","CSCO","C","CFG",
        "CLX","CME","CMS","KO","CTSH","CL","CMCSA","CAG","COP","ED",
        "STZ","CEG","COO","CPRT","GLW","CTVA","CSGP","COST","CTRA","CRWD",
        "CCI","CSX","CMI","CVS","DHI","DHR","DRI","DVA","DAY","DECK",
        "DE","DAL","DVN","DXCM","FANG","DLR","DFS","DG","DLTR","D",
        "DPZ","DOV","DOW","DHI","DTE","DUK","DD","EMN","ETN","EBAY",
        "ECL","EIX","EW","EA","ELV","LLY","EMR","ENPH","ETR","EOG",
        "EPAM","EQT","EFX","EQIX","EQR","ESS","EL","ETSY","EG","EVRG",
        "ES","EXC","EXPE","EXPD","EXR","XOM","FFIV","FDS","FICO","FAST",
        "FRT","FDX","FIS","FITB","FSLR","FE","FI","FLT","FMC","F",
        "FTNT","FTV","FOXA","FOX","BEN","FCX","GRMN","IT","GE","GEHC",
        "GEN","GNRC","GD","GIS","GM","GPC","GILD","GPN","GL","GS",
        "HAL","HIG","HAS","HCA","DOC","HSIC","HSY","HES","HPE","HLT",
        "HOLX","HD","HON","HRL","HST","HWM","HPQ","HUBB","HUM","HBAN",
        "HII","IBM","IEX","IDXX","ITW","INCY","IR","PODD","INTC","ICE",
        "IFF","IP","IPG","INTU","ISRG","IVZ","INVH","IQV","IRM","JBHT",
        "JBL","JKHY","J","JNJ","JCI","JPM","JNPR","K","KVUE","KDP",
        "KEY","KEYS","KMB","KIM","KMI","KLAC","KHC","KR","LHX","LH",
        "LRCX","LW","LVS","LDOS","LEN","LNC","LIN","LYV","LKQ","LMT",
        "L","LOW","LULU","LYB","MTB","MRO","MPC","MKTX","MAR","MMC",
        "MLM","MAS","MA","MTCH","MKC","MCD","MCK","MDT","MRK","META",
        "MET","MTD","MGM","MCHP","MU","MSFT","MAA","MRNA","MHK","MOH",
        "TAP","MDLZ","MPWR","MNST","MCO","MS","MOS","MSI","MSCI","NDAQ",
        "NTAP","NFLX","NEM","NWSA","NWS","NEE","NKE","NI","NDSN","NSC",
        "NTRS","NOC","NCLH","NRG","NUE","NVR","NVDA","NWA","ORLY","OXY",
        "ODFL","OMC","ON","OKE","ORCL","OTIS","PCAR","PKG","PLTR","PH",
        "PAYX","PAYC","PYPL","PNR","PEP","PFE","PCG","PM","PSX","PNW",
        "PNC","POOL","PPG","PPL","PFG","PG","PGR","PRU","PEG","PTC",
        "PSA","PHM","QRVO","PWR","QCOM","DGX","RL","RJF","RTX","O",
        "REG","REGN","RF","RSG","RMD","RVTY","ROK","ROL","ROP","ROST",
        "RCL","SPGI","CRM","SBAC","SLB","STX","SRE","NOW","SHW","SPG",
        "SWKS","SJM","SW","SNA","SOLV","SO","LUV","SWK","SBUX","STT",
        "STLD","STE","SYK","SMCI","SYF","SNPS","SYY","TMUS","TROW","TTWO",
        "TPR","TRGP","TGT","TEL","TDY","TFX","TER","TSLA","TXN","TXT",
        "TMO","TJX","TSCO","TT","TDG","TRV","TRMB","TFC","TYL","TSN",
        "USB","UBER","UDR","ULTA","UNP","UAL","UPS","URI","UNH","UHS",
        "VLO","VTR","VLTO","VRSN","VRSK","VZ","VRTX","VTRS","VICI","V",
        "VST","VMC","WRB","GWW","WAB","WBA","WMT","DIS","WBD","WM",
        "WAT","WEC","WFC","WELL","WST","WDC","WHR","WRK","WY","WHR",
        "WYNN","XEL","XYL","YUM","ZBRA","ZBH","ZTS",
    ]


def _get_nasdaq100() -> list:
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; scanner/1.0)"}
        r = requests.get(
            "https://en.wikipedia.org/wiki/Nasdaq-100",
            headers=headers,
            timeout=15,
        )
        r.raise_for_status()
        tables = pd.read_html(r.text)
        for table in tables:
            # Wikipedia uses "Ticker" or "Symbol" depending on page version
            for col in ("Ticker", "Symbol", "Ticker symbol"):
                if col in table.columns:
                    tickers = (table[col]
                                .astype(str)
                                .str.strip()
                                .str.replace(r"\[.*?\]", "", regex=True)
                                .tolist())
                    tickers = [t for t in tickers if t and t.lower() != "nan"]
                    if len(tickers) > 50:   # sanity check -- real index has 100+
                        log.info(f"  Nasdaq-100: {len(tickers)} tickers from Wikipedia")
                        return tickers
    except Exception as e:
        log.warning(f"  Wikipedia Nasdaq-100 failed: {e}")

    log.warning("  Using hardcoded Nasdaq-100 fallback")
    return [
        "MSFT","AAPL","NVDA","AMZN","META","TSLA","GOOGL","GOOG","AVGO","COST",
        "NFLX","TMUS","ASML","CSCO","ADBE","AMD","PEP","INTU","AMAT","TXN",
        "QCOM","ISRG","AMGN","BKNG","CMCSA","ARM","MU","PANW","ADI","LRCX",
        "SBUX","INTC","KLAC","MELI","MDLZ","REGN","GILD","SNPS","CDNS","CEG",
        "CRWD","CTAS","ABNB","MAR","ORLY","MRVL","MNST","PYPL","PCAR","FTNT",
        "ADSK","DASH","WDAY","KDP","AZN","CHTR","ROP","NXPI","MCHP","PAYX",
        "TEAM","AEP","ROST","DXCM","IDXX","CPRT","ODFL","EA","VRSK","FAST",
        "FANG","XEL","CTSH","BIIB","ON","BKR","KHC","CSGP","DDOG","GEHC",
        "EXC","TTD","CCEP","LULU","ANSS","ILMN","WBA","MDB","ZS","SIRI",
        "MTCH","LCID","RIVN","DLTR","ZM","PDD","MRNA","GFS","ALGN","CDW",
    ]


def get_universe() -> list:
    sp500   = _get_sp500()
    nasdaq  = _get_nasdaq100()
    combined = list(dict.fromkeys(sp500 + nasdaq))
    log.info(f"  Combined universe: {len(combined)} unique tickers (S&P 500 + Nasdaq-100)")
    return combined


# ===========================================================================
# 2. PRE-MARKET / INTRADAY DATA via Alpaca
# ===========================================================================

def get_alpaca_client():
    return tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, api_version="v2")


def _get_current_price(snap) -> float:
    """
    Extract the best available current price from a snapshot.
    Priority: latest_trade -> minute_bar -> daily_bar open
    This works both pre-market and during market hours.
    """
    # Latest trade is the most reliable source at any time of day
    try:
        p = float(snap.latest_trade.p)
        if p > 0:
            return p
    except Exception:
        pass

    # Fall back to latest minute bar close
    try:
        p = float(snap.minute_bar.c)
        if p > 0:
            return p
    except Exception:
        pass

    # Last resort: today's daily bar open
    try:
        p = float(snap.daily_bar.o)
        if p > 0:
            return p
    except Exception:
        pass

    return 0.0


def _get_current_volume(snap) -> int:
    """
    Get best available volume. During market hours daily_bar.v is the
    accumulated intraday volume, which is more meaningful than minute_bar.v.
    """
    try:
        v = int(snap.daily_bar.v)
        if v > 0:
            return v
    except Exception:
        pass

    try:
        return int(snap.minute_bar.v)
    except Exception:
        return 0


def _get_prev_close(snap) -> float:
    # Try previous_daily_bar first (most accurate)
    try:
        v = float(snap.previous_daily_bar.c)
        if v > 0:
            return v
    except Exception:
        pass
    # Fall back to daily_bar open (today's open ~ yesterday's close)
    try:
        v = float(snap.daily_bar.o)
        if v > 0:
            return v
    except Exception:
        pass
    return 0.0


def get_premarket_snapshots(tickers: list) -> dict:
    api = get_alpaca_client()
    log.info(f"  Fetching Alpaca snapshots for {len(tickers)} tickers ...")

    snapshots  = {}
    chunk_size = 100

    for i in range(0, len(tickers), chunk_size):
        # Alpaca uses dots not dashes (e.g. BRK.B not BRK-B)
        chunk = [t.replace("-", ".") for t in tickers[i:i + chunk_size]]
        try:
            snaps = api.get_snapshots(chunk)
            for ticker, snap in snaps.items():
                snapshots[ticker] = snap
        except Exception as e:
            log.warning(f"  Snapshot chunk {i//chunk_size + 1} error: {e}")
        time.sleep(0.2)

    log.info(f"  Got snapshots for {len(snapshots)} tickers")

    # Debug: show a sample snapshot to verify data quality
    if snapshots:
        sample_ticker = next(iter(snapshots))
        sample_snap   = snapshots[sample_ticker]
        sample_price  = _get_current_price(sample_snap)
        sample_prev   = _get_prev_close(sample_snap)
        sample_vol    = _get_current_volume(sample_snap)
        log.info(f"  Sample [{sample_ticker}]: price={sample_price:.2f}, prev_close={sample_prev:.2f}, volume={sample_vol:,}")

    return snapshots


def parse_gap_data(snapshots: dict) -> list:
    gaps          = []
    skipped_price = 0
    skipped_gap   = 0
    skipped_low   = 0

    for ticker, snap in snapshots.items():
        try:
            prev_close    = _get_prev_close(snap)
            current_price = _get_current_price(snap)
            current_vol   = _get_current_volume(snap)

            # Get 20-day avg volume from daily bar if available
            # We use previous_daily_bar volume as a rough proxy
            try:
                prev_vol = float(snap.previous_daily_bar.v)
            except Exception:
                prev_vol = 0

            if prev_close <= 0 or current_price <= 0:
                skipped_price += 1
                continue

            if current_price < MIN_PRICE:
                skipped_low += 1
                continue

            gap_pct = ((current_price - prev_close) / prev_close) * 100

            if gap_pct < MIN_GAP_PCT or gap_pct > MAX_GAP_PCT:
                skipped_gap += 1
                continue

            # Volume ratio: today's vol vs yesterday's vol as a proxy
            vol_ratio = current_vol / prev_vol if prev_vol > 0 else 0.0

            gaps.append({
                "ticker"          : ticker,
                "prior_close"     : prev_close,
                "premarket_price" : current_price,
                "premarket_volume": current_vol,
                "daily_vol_avg"   : prev_vol,
                "gap_pct"         : gap_pct,
                "volume_ratio"    : vol_ratio,
            })

        except Exception as ex:
            log.debug(f"  parse error for {ticker}: {ex}")
            continue

    log.info(f"  Snapshot parse: {len(gaps)} gaps found | "
             f"skipped price={skipped_price}, low_price={skipped_low}, outside_gap_range={skipped_gap}")

    gaps.sort(key=lambda x: x["gap_pct"], reverse=True)
    return gaps


# ===========================================================================
# 3. NEWS SENTIMENT via Finnhub
# ===========================================================================

POSITIVE_WORDS = {
    "beat", "beats", "record", "surge", "surges", "jumps", "jump", "rally",
    "rallies", "upgrade", "upgraded", "raises", "raised", "strong", "stronger",
    "growth", "profit", "profits", "win", "wins", "deal", "acquisition",
    "partnership", "approval", "approved", "launch", "launches", "positive",
    "outperform", "buy", "bullish", "gains", "gain", "soars", "soar",
}

NEGATIVE_WORDS = {
    "miss", "misses", "missed", "falls", "fall", "drops", "drop", "cut",
    "cuts", "downgrade", "downgraded", "loss", "losses", "weak", "weaker",
    "decline", "declines", "warning", "warn", "lawsuit", "investigation",
    "recall", "halt", "halted", "sell", "bearish", "negative", "concern",
    "concerns", "disappoints", "disappointing", "slump", "slumps",
}


def score_headline(text: str) -> float:
    if not text:
        return 0.0
    words = set(re.sub(r"[^a-z\s]", "", text.lower()).split())
    pos   = len(words & POSITIVE_WORDS)
    neg   = len(words & NEGATIVE_WORDS)
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
        try:
            url    = "https://finnhub.io/api/v1/company-news"
            params = {
                "symbol": ticker,
                "from"  : yesterday_str,
                "to"    : today_str,
                "token" : FINNHUB_API_KEY,
            }
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            articles = r.json()

            if not articles:
                results[ticker] = {"score": 0.0, "label": "neutral", "headline": ""}
                continue

            recent = articles[:5]
            scores = [
                score_headline(a.get("headline", "") + " " + a.get("summary", ""))
                for a in recent
            ]
            avg_score    = round(sum(scores) / len(scores), 3)
            top_headline = recent[0].get("headline", "")
            label        = "positive" if avg_score > 0.1 else "negative" if avg_score < -0.1 else "neutral"

            results[ticker] = {
                "score"   : avg_score,
                "label"   : label,
                "headline": top_headline,
            }

            time.sleep(1.1)  # Finnhub free tier: 60 calls/min

        except Exception as e:
            log.warning(f"  News fetch failed for {ticker}: {e}")
            results[ticker] = {"score": 0.0, "label": "neutral", "headline": ""}

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
        log.warning(f"  yfinance download failed: {e}")
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
        snap  = get_alpaca_client().get_snapshot("SPY")
        prev  = _get_prev_close(snap)
        curr  = _get_current_price(snap)
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
    ticker = gap_data["ticker"]

    gap_score    = min(100, max(0, (gap_data["gap_pct"] - MIN_GAP_PCT) / (10 - MIN_GAP_PCT) * 100))
    vol_score    = min(100, max(0, (gap_data["volume_ratio"] - 0.5) / 2.5 * 100))
    sent_raw     = sentiment.get("score", 0.0)
    sent_score   = min(100, max(0, (sent_raw + 1) / 2 * 100))
    st_score     = 100 if supertrend == 1 else 0 if supertrend == -1 else 50
    rel_strength = gap_data["gap_pct"] - spy_change
    rs_score     = min(100, max(0, (rel_strength + 5) / 10 * 100))

    composite = round(
        (gap_score   * W_GAP          / 100) +
        (vol_score   * W_VOLUME       / 100) +
        (sent_score  * W_SENTIMENT    / 100) +
        (st_score    * W_SUPERTREND   / 100) +
        (rs_score    * W_REL_STRENGTH / 100),
        1
    )

    entry     = gap_data["premarket_price"]
    stop_loss = round(entry * 0.98, 2)
    target    = round(entry * 1.04, 2)

    return {
        "ticker"             : ticker,
        "score"              : composite,
        "gap_pct"            : gap_data["gap_pct"],
        "premarket_volume"   : gap_data["premarket_volume"],
        "volume_ratio"       : gap_data["volume_ratio"],
        "sentiment_score"    : sentiment.get("score", 0.0),
        "sentiment_label"    : sentiment.get("label", "neutral"),
        "news_headline"      : sentiment.get("headline", ""),
        "supertrend_signal"  : "bullish" if supertrend == 1 else "bearish" if supertrend == -1 else "neutral",
        "rel_strength_vs_spy": rel_strength,
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

    tickers = get_universe()

    log.info("\n[1/5] Fetching snapshots ...")
    snapshots = get_premarket_snapshots(tickers)
    gap_list  = parse_gap_data(snapshots)

    if not gap_list:
        log.warning("No tickers found with gap > %.1f%%. Check Alpaca credentials and data feed.", MIN_GAP_PCT)
        log.warning("Try running: python -c \"import alpaca_trade_api as t; api=t.REST('%s','%s','%s'); s=api.get_snapshot('AAPL'); print(s.latest_trade.p, 
s.previous_daily_bar.c)\"",
                    ALPACA_API_KEY[:4] + "...", "***", ALPACA_BASE_URL)
        return

    gap_tickers = [g["ticker"] for g in gap_list]

    log.info("\n[2/5] Getting SPY baseline ...")
    spy_change = get_spy_change()
    log.info(f"  SPY change vs prior close: {spy_change:+.2f}%")

    log.info("\n[3/5] Fetching news sentiment ...")
    sentiment_map = fetch_news_sentiment(gap_tickers)

    log.info("\n[4/5] Computing Supertrend signals ...")
    supertrend_map = get_supertrend_signals(gap_tickers)

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
              f"{w['volume_ratio']:>6.1f}x    "
              f"{w['sentiment_label']:<11} "
              f"{w['supertrend_signal']:<10} "
              f"${w['premarket_price']:<7.2f} "
              f"${w['stop_loss']:<7.2f} "
              f"${w['target']:.2f}")
    print()

    log.info("Logging to Google Sheets ...")
    log_watchlist_to_sheet(watchlist, run_ts, today)

    log.info("=" * 60)
    log.info(f"  Tickers scanned         : {len(snapshots)}")
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

