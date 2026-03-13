import os
import io
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

GPR_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"
CACHE_PATH = "data/cache/gpr_daily.parquet"

GEOPOLITICAL_KEYWORDS = [
    "war", "conflict", "missile", "military", "invasion", "attack",
    "sanctions", "nuclear", "terror", "crisis", "troops", "nato",
    "geopolitical", "coup", "ceasefire", "escalation", "airstrike",
    "blockade", "hostage", "insurgency", "drone strike"
]

def load_gpr_index() -> pd.DataFrame:
    """
    Load the Caldara-Iacoviello daily GPR index.
    Falls back to synthetic GPR if download fails.
    Returns DataFrame with columns: date, gpr
    """
    os.makedirs("data/cache", exist_ok=True)

    # Try cache first (refresh if older than 3 days)
    if os.path.exists(CACHE_PATH):
        age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(CACHE_PATH))
        if age < timedelta(days=3):
            return pd.read_parquet(CACHE_PATH)

    try:
        headers = {"User-Agent": "Mozilla/5.0 (research project; contact: researcher@university.edu)"}
        resp = requests.get(GPR_URL, headers=headers, timeout=15)
        resp.raise_for_status()
        df = pd.read_excel(io.BytesIO(resp.content))

        # Normalise column names — file has varied layouts across versions
        df.columns = [str(c).strip().lower() for c in df.columns]
        date_col = next((c for c in df.columns if "date" in c or "year" in c or c == "mon"), None)
        gpr_col  = next((c for c in df.columns if c in ("gpr", "gprd", "gpr_daily", "gpr daily")), None)

        if date_col is None or gpr_col is None:
            raise ValueError(f"Unexpected columns: {list(df.columns)}")

        df = df[[date_col, gpr_col]].copy()
        df.columns = ["date", "gpr"]
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "gpr"])
        df = df.sort_values("date").reset_index(drop=True)
        df.to_parquet(CACHE_PATH)
        return df

    except Exception as e:
        print(f"[GPR] Download failed ({e}), using synthetic fallback")
        return _synthetic_gpr()


def _synthetic_gpr() -> pd.DataFrame:
    """
    Synthetic GPR that mimics the real index's known spikes.
    Used only when the live download fails.
    """
    dates = pd.date_range("2015-01-01", datetime.today(), freq="B")
    np.random.seed(42)
    base = 100 + np.cumsum(np.random.normal(0, 2, len(dates)))
    base = np.clip(base, 50, 400)

    # Known historical spikes
    spikes = {
        "2020-03-01": 280, "2020-03-15": 320,   # COVID crash
        "2022-02-24": 350, "2022-03-10": 380,   # Ukraine invasion
        "2023-10-08": 260, "2023-10-20": 290,   # Middle East
        "2021-08-15": 200,                        # Afghanistan
    }
    idx = pd.Series(base, index=dates)
    for date_str, val in spikes.items():
        try:
            spike_date = pd.Timestamp(date_str)
            mask = (idx.index >= spike_date) & (idx.index <= spike_date + timedelta(days=30))
            decay = np.linspace(val, 100, mask.sum())
            idx[mask] = np.maximum(idx[mask], decay)
        except Exception:
            pass

    df = pd.DataFrame({"date": idx.index, "gpr": idx.values})
    return df.reset_index(drop=True)


def classify_gpr_regime(gpr_series: pd.Series, high_pct: float = 0.75) -> pd.Series:
    """
    Classify each GPR value as 'calm' or 'crisis'.
    crisis = top (1-high_pct) percentile of historical values.
    """
    threshold = gpr_series.quantile(high_pct)
    return pd.Series(
        np.where(gpr_series >= threshold, "crisis", "calm"),
        index=gpr_series.index
    )


def get_current_gpr_level(gpr_df: pd.DataFrame) -> dict:
    """
    Return the most recent GPR reading and its regime classification.
    """
    latest = gpr_df.dropna(subset=["gpr"]).iloc[-1]
    historical = gpr_df["gpr"].dropna()
    pct_rank = float((historical < latest["gpr"]).mean() * 100)
    threshold_75 = historical.quantile(0.75)
    threshold_90 = historical.quantile(0.90)

    if latest["gpr"] >= threshold_90:
        regime = "extreme"
        color  = "red"
    elif latest["gpr"] >= threshold_75:
        regime = "elevated"
        color  = "orange"
    else:
        regime = "calm"
        color  = "green"

    return {
        "value":     round(float(latest["gpr"]), 1),
        "date":      latest["date"],
        "regime":    regime,
        "color":     color,
        "pct_rank":  round(pct_rank, 1),
        "threshold_crisis": round(float(threshold_75), 1),
    }


def fetch_geopolitical_news(api_key: str, max_articles: int = 8) -> list[dict]:
    """
    Fetch live geopolitical headlines from NewsAPI.org.
    Returns list of {title, source, url, publishedAt, sentiment_score}
    Falls back to curated mock headlines if key is missing or quota exceeded.
    """
    if not api_key or api_key == "YOUR_NEWSAPI_KEY":
        return _mock_headlines()

    try:
        query = " OR ".join(GEOPOLITICAL_KEYWORDS[:8])
        url = (
            f"https://newsapi.org/v2/everything"
            f"?q={query}"
            f"&language=en"
            f"&sortBy=publishedAt"
            f"&pageSize={max_articles}"
            f"&apiKey={api_key}"
        )
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        articles = []
        for a in data.get("articles", [])[:max_articles]:
            title = a.get("title", "") or ""
            score = _simple_sentiment(title)
            articles.append({
                "title":       title,
                "source":      (a.get("source") or {}).get("name", "Unknown"),
                "url":         a.get("url", "#"),
                "publishedAt": a.get("publishedAt", ""),
                "sentiment":   score,
            })
        return articles if articles else _mock_headlines()

    except Exception as e:
        print(f"[News] API call failed ({e}), using mock headlines")
        return _mock_headlines()


def _simple_sentiment(text: str) -> float:
    """
    Naive keyword-based sentiment for geopolitical headlines.
    Returns -1.0 (very negative) to +1.0 (positive).
    """
    negative = ["war", "attack", "killed", "missile", "crisis", "invasion",
                 "conflict", "escalation", "nuclear", "terror", "sanctions",
                 "airstrike", "explosion", "casualties", "ceasefire broken"]
    positive = ["ceasefire", "peace", "deal", "agreement", "diplomacy",
                 "withdrawal", "talks", "accord", "resolved", "de-escalation"]
    t = text.lower()
    neg = sum(1 for w in negative if w in t)
    pos = sum(1 for w in positive if w in t)
    total = neg + pos
    if total == 0:
        return 0.0
    return round((pos - neg) / total, 2)


def _mock_headlines() -> list[dict]:
    return [
        {
            "title":       "NATO allies increase defence spending amid regional tensions",
            "source":      "Reuters (demo)",
            "url":         "#",
            "publishedAt": datetime.today().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "sentiment":   -0.3,
        },
        {
            "title":       "UN Security Council convenes emergency session on conflict escalation",
            "source":      "BBC News (demo)",
            "url":         "#",
            "publishedAt": datetime.today().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "sentiment":   -0.6,
        },
        {
            "title":       "G7 finance ministers discuss new sanctions package",
            "source":      "FT (demo)",
            "url":         "#",
            "publishedAt": datetime.today().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "sentiment":   -0.2,
        },
        {
            "title":       "Central banks signal readiness to act on geopolitical shock",
            "source":      "Bloomberg (demo)",
            "url":         "#",
            "publishedAt": datetime.today().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "sentiment":   0.1,
        },
        {
            "title":       "Peace talks resume after weeks of diplomatic deadlock",
            "source":      "AP (demo)",
            "url":         "#",
            "publishedAt": datetime.today().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "sentiment":   0.5,
        },
    ]