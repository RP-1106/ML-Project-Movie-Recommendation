"""
feature_engineering.py
======================
Feature engineering pipeline for the VOD clickstream ML project.

Inputs:
    netflix_uk_data.csv

Outputs:
    data.csv
        - user_idx, item_idx, user_id, item_id, datetime

    item_features.csv
        - one row per item with genre one-hots + metadata

    user_features.csv
        - one row per user with behavioural aggregates + preferences

This version keeps the dataset creation clean and modeling-ready:
- consistent user/item integer mappings across all files
- interaction log includes integer indices
- item table contains content + popularity features
- user table contains behavioural and preference features
- TMDB enrichment fills missing metadata when available

Run locally or in Colab.
Set TMDB_API_KEY as an environment variable if you want enrichment.
"""

from __future__ import annotations

import os
from dotenv import load_dotenv
import time
import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import requests


# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
load_dotenv()
INPUT_CSV = "netflix_uk_data_filtered.csv"
OUT_DATA = "data2.csv"
OUT_ITEMS = "item_features2.csv"
OUT_USERS = "user_features2.csv"

# TMDB config
# Do not hardcode your API key here. Use an environment variable instead.
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_ENABLED = bool(TMDB_API_KEY)
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_DELAY = 0.25
TMDB_TIMEOUT = 8

# Genre vocabulary seen in the raw data / useful for one-hot encoding
GENRE_VOCAB = [
    "Action", "Adventure", "Animation", "Biography", "Comedy", "Crime",
    "Documentary", "Drama", "Family", "Fantasy", "Film-Noir", "History",
    "Horror", "Music", "Musical", "Mystery", "News", "Reality-TV",
    "Romance", "Sci-Fi", "Short", "Sport", "Talk-Show", "Thriller",
    "War", "Western",
]
GENRES_EXCLUDE = {"NOT AVAILABLE"}
GENRE_COLS = [f"genre_{g.lower().replace('-', '_')}" for g in GENRE_VOCAB]


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def build_id_maps(df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Build stable, shared integer mappings for users and items."""
    user_ids = sorted(df["user_id"].dropna().astype(str).unique())
    item_ids = sorted(df["item_id"].dropna().astype(str).unique())

    user2idx = {u: i for i, u in enumerate(user_ids)}
    item2idx = {it: i for i, it in enumerate(item_ids)}
    return user2idx, item2idx


def ensure_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def safe_to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def normalize_text(x) -> str:
    if pd.isna(x):
        return "NOT AVAILABLE"
    s = str(x).strip()
    return s if s else "NOT AVAILABLE"


# ----------------------------------------------------------------------------
# Load raw data
# ----------------------------------------------------------------------------
def load_raw(path: str) -> pd.DataFrame:
    log.info(f"Loading raw data from {path}")
    df = pd.read_csv(path)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Standardize column names
    rename_map = {}
    if "movie_id" in df.columns and "item_id" not in df.columns:
        rename_map["movie_id"] = "item_id"
    df = df.rename(columns=rename_map)

    # Validate core columns
    ensure_columns(df, ["user_id", "item_id", "datetime"], "raw dataset")

    # Parse types
    df = df.copy()
    df["user_id"] = df["user_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)
    df["datetime"] = safe_to_datetime(df["datetime"])
    df = df.dropna(subset=["datetime", "user_id", "item_id"])

    # Optional raw columns: create placeholders if absent so downstream code is stable
    if "title" not in df.columns:
        df["title"] = df["item_id"]
    if "genres" not in df.columns:
        df["genres"] = "NOT AVAILABLE"
    if "release_date" not in df.columns:
        df["release_date"] = "NOT AVAILABLE"
    if "duration" not in df.columns:
        df["duration"] = np.nan

    log.info(
        f"  Raw shape: {df.shape}  |  users: {df['user_id'].nunique():,}  |  items: {df['item_id'].nunique():,}"
    )
    return df


# ----------------------------------------------------------------------------
# Interaction log
# ----------------------------------------------------------------------------
def build_interaction_log(df: pd.DataFrame, user2idx: Dict[str, int], item2idx: Dict[str, int]) -> pd.DataFrame:
    """Build the clean interaction log for SASRec / LightGCN / baselines."""
    log.info("Building interaction log …")

    interactions = df[["user_id", "item_id", "datetime"]].copy()
    interactions["user_idx"] = interactions["user_id"].map(user2idx)
    interactions["item_idx"] = interactions["item_id"].map(item2idx)

    interactions = interactions.dropna(subset=["user_idx", "item_idx"])
    interactions["user_idx"] = interactions["user_idx"].astype(int)
    interactions["item_idx"] = interactions["item_idx"].astype(int)

    interactions = interactions.sort_values(["user_idx", "datetime"]).reset_index(drop=True)
    interactions = interactions[["user_idx", "item_idx", "user_id", "item_id", "datetime"]]

    log.info(f"  Interaction log shape: {interactions.shape}")
    return interactions


# ----------------------------------------------------------------------------
# TMDB enrichment
# ----------------------------------------------------------------------------
def _tmdb_search(title: str, year: Optional[int] = None) -> Optional[dict]:
    params = {"api_key": TMDB_API_KEY, "query": title, "language": "en-US"}
    if year:
        params["year"] = year
    try:
        resp = requests.get(f"{TMDB_BASE_URL}/search/movie", params=params, timeout=TMDB_TIMEOUT)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        return results[0] if results else None
    except Exception as e:
        log.warning(f"    TMDB search failed for '{title}': {e}")
        return None


def _tmdb_details(tmdb_id: int) -> Optional[dict]:
    params = {"api_key": TMDB_API_KEY, "language": "en-US"}
    try:
        resp = requests.get(f"{TMDB_BASE_URL}/movie/{tmdb_id}", params=params, timeout=TMDB_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        log.warning(f"    TMDB details failed for id {tmdb_id}: {e}")
        return None


def enrich_with_tmdb(item_df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing item metadata from TMDB where possible."""
    if not TMDB_ENABLED:
        log.info("  TMDB enrichment skipped (no API key).")
        return item_df

    needs_genres = item_df["genres_raw"].apply(normalize_text).eq("NOT AVAILABLE")
    needs_date = item_df["release_date_raw"].apply(normalize_text).eq("NOT AVAILABLE")
    needs_title = item_df["title"].apply(normalize_text).eq("NOT AVAILABLE")
    needs_enrichment = needs_genres | needs_date | needs_title
    to_enrich = item_df[needs_enrichment].copy()

    log.info(f"  TMDB enrichment: {len(to_enrich):,} items need enriching …")

    tmdb_updates: Dict[str, Dict[str, object]] = {}
    for _, row in to_enrich.iterrows():
        item_id = row["item_id"]
        title = normalize_text(row["title"])

        year = None
        try:
            year_str = str(row.get("release_date_raw", ""))[:4]
            year = int(year_str)
        except Exception:
            year = None

        result = _tmdb_search(title, year)
        time.sleep(TMDB_DELAY)
        if result is None:
            continue

        tmdb_id = result.get("id")
        details = _tmdb_details(tmdb_id) if tmdb_id else None
        time.sleep(TMDB_DELAY)

        update: Dict[str, object] = {}
        if details:
            genres_list = [g.get("name") for g in details.get("genres", []) if g.get("name")]
            if genres_list:
                update["genres_raw"] = ", ".join(genres_list)

            rd = details.get("release_date", "")
            if rd:
                update["release_date_raw"] = rd

            update["tmdb_popularity"] = details.get("popularity", np.nan)
            update["tmdb_vote_avg"] = details.get("vote_average", np.nan)
            update["tmdb_runtime_min"] = details.get("runtime", np.nan)
            update["original_language"] = details.get("original_language", "")

        if update:
            tmdb_updates[item_id] = update

    log.info(f"  TMDB enriched {len(tmdb_updates):,} items.")

    # Make sure string columns can accept string values like "en"
    for col in ["genres_raw", "release_date_raw", "original_language", "title"]:
        if col in item_df.columns:
            item_df[col] = item_df[col].astype("object")

    # Make sure numeric TMDB columns exist
    for col in ["tmdb_runtime_min", "tmdb_popularity", "tmdb_vote_avg"]:
        if col not in item_df.columns:
            item_df[col] = np.nan

    item_df = item_df.set_index("item_id")
    for item_id, update in tmdb_updates.items():
        if item_id not in item_df.index:
            continue
        for col, val in update.items():
            if col not in item_df.columns:
                if col in ["genres_raw", "release_date_raw", "original_language", "title"]:
                    item_df[col] = pd.Series(index=item_df.index, dtype="object")
                else:
                    item_df[col] = np.nan

            if col in ["genres_raw", "release_date_raw", "original_language", "title"]:
                item_df[col] = item_df[col].astype("object")
            item_df.loc[item_id, col] = val
    item_df = item_df.reset_index()
    return item_df


# ----------------------------------------------------------------------------
# Item features
# ----------------------------------------------------------------------------
def _parse_release_year(date_str) -> float:
    try:
        s = str(date_str).strip()
        if s in ("NOT AVAILABLE", "", "nan", "NaT"):
            return np.nan
        return float(s[:4])
    except Exception:
        return np.nan


def _year_to_decade_bucket(year: float) -> int:
    """
    Map year to ordinal decade bucket:
        unknown     -> -1
        before 1960 -> 0
        1960s       -> 1
        1970s       -> 2
        1980s       -> 3
        1990s       -> 4
        2000s       -> 5
        2010s       -> 6
        2020s       -> 7
    """
    if pd.isna(year):
        return -1
    y = int(year)
    if y < 1960:
        return 0
    return min(7, (y - 1960) // 10 + 1)


def _click_duration_bucket(seconds: float) -> int:
    """Bucket per-click watch duration into coarse categories."""
    if pd.isna(seconds) or seconds <= 0:
        return 0
    if seconds < 900:
        return 1
    if seconds < 2700:
        return 2
    if seconds < 5400:
        return 3
    if seconds < 9000:
        return 4
    return 5


def _runtime_bucket_from_minutes(runtime_min: float) -> int:
    """Bucket TMDB runtime into coarse categories."""
    if pd.isna(runtime_min) or runtime_min <= 0:
        return -1
    if runtime_min < 60:
        return 0
    if runtime_min < 100:
        return 1
    if runtime_min < 140:
        return 2
    return 3


def build_item_features(df: pd.DataFrame, item2idx: Dict[str, int]) -> pd.DataFrame:
    """Build one row per item with content, metadata, and popularity features."""
    log.info("Building item feature table …")

    base_cols = ["item_id", "title", "genres", "release_date", "duration"]
    item_meta = df.drop_duplicates("item_id")[base_cols].copy()
    item_meta = item_meta.rename(
        columns={
            "genres": "genres_raw",
            "release_date": "release_date_raw",
        }
    ).reset_index(drop=True)

    item_meta["item_id"] = item_meta["item_id"].astype(str)
    item_meta["title"] = item_meta["title"].apply(normalize_text)
    item_meta["genres_raw"] = item_meta["genres_raw"].apply(normalize_text)
    item_meta["release_date_raw"] = item_meta["release_date_raw"].apply(normalize_text)

    # Keep both behavioral duration and runtime metadata separate
    item_meta["median_click_duration_sec"] = (
        df.groupby("item_id")["duration"].median().reindex(item_meta["item_id"]).values
    )

    # Popularity stats
    click_counts = df.groupby("item_id").size().reset_index(name="click_count")
    unique_users = df.groupby("item_id")["user_id"].nunique().reset_index(name="n_unique_users")
    first_seen = df.groupby("item_id")["datetime"].min().reset_index(name="first_seen")
    last_seen = df.groupby("item_id")["datetime"].max().reset_index(name="last_seen")

    item_meta = (
        item_meta.merge(click_counts, on="item_id", how="left")
        .merge(unique_users, on="item_id", how="left")
        .merge(first_seen, on="item_id", how="left")
        .merge(last_seen, on="item_id", how="left")
    )

    item_meta["log_popularity"] = np.log1p(item_meta["click_count"].fillna(0))
    item_meta["item_repeat_ratio"] = item_meta["click_count"] / item_meta["n_unique_users"].replace(0, np.nan)
    item_meta["item_age_days"] = (item_meta["last_seen"] - item_meta["first_seen"]).dt.days

    # TMDB enrichment adds/updates release_date / genres / runtime / language / vote stats
    item_meta = enrich_with_tmdb(item_meta)

    # Fill missing TMDB columns if enrichment was skipped
    if "tmdb_runtime_min" not in item_meta.columns:
        item_meta["tmdb_runtime_min"] = np.nan
    if "tmdb_popularity" not in item_meta.columns:
        item_meta["tmdb_popularity"] = np.nan
    if "tmdb_vote_avg" not in item_meta.columns:
        item_meta["tmdb_vote_avg"] = np.nan
    if "original_language" not in item_meta.columns:
        item_meta["original_language"] = pd.Series(index=item_meta.index, dtype="object")
    else:
        item_meta["original_language"] = item_meta["original_language"].astype("object")

    # Genre one-hot encoding
    for col in GENRE_COLS:
        item_meta[col] = 0

    for idx, row in item_meta.iterrows():
        genres_str = normalize_text(row["genres_raw"])
        if genres_str == "NOT AVAILABLE":
            continue

        # Many raw datasets use comma-separated genres after TMDB enrichment.
        # Be robust to comma, pipe, or mixed separators.
        raw_parts = genres_str.replace("|", ",").split(",")
        genres = [g.strip() for g in raw_parts if g.strip()]

        for genre in genres:
            if genre in GENRES_EXCLUDE:
                continue
            col_name = f"genre_{genre.lower().replace('-', '_').replace(' ', '_')}"
            if col_name in item_meta.columns:
                item_meta.at[idx, col_name] = 1
            else:
                # If the genre is not in the fixed vocab, create a new one-hot column.
                # This keeps the table complete even if TMDB returns something unexpected.
                item_meta[col_name] = 0
                item_meta.at[idx, col_name] = 1

    # Temporal metadata
    item_meta["release_year"] = item_meta["release_date_raw"].apply(_parse_release_year)
    item_meta["decade_bucket"] = item_meta["release_year"].apply(_year_to_decade_bucket)

    # Separate content runtime and behavioral click duration
    item_meta["runtime_bucket"] = item_meta["tmdb_runtime_min"].apply(_runtime_bucket_from_minutes)
    item_meta["click_duration_bucket"] = item_meta["median_click_duration_sec"].apply(_click_duration_bucket)

    # Integer index via shared mapping
    item_meta["item_idx"] = item_meta["item_id"].map(item2idx)
    item_meta["item_idx"] = item_meta["item_idx"].astype(int)

    # Standardize language column naming
    item_meta["original_language"] = item_meta["original_language"].fillna("").astype(str)

    # Final column order
    keep_cols = [
        "item_idx",
        "item_id",
        "title",
        "genres_raw",
        "release_date_raw",
        "release_year",
        "decade_bucket",
        "tmdb_runtime_min",
        "runtime_bucket",
        "click_duration_bucket",
        "median_click_duration_sec",
        "click_count",
        "n_unique_users",
        "log_popularity",
        "item_repeat_ratio",
        "item_age_days",
        "tmdb_popularity",
        "tmdb_vote_avg",
        "original_language",
    ]

    # Keep all genre columns, including any unexpected ones created during enrichment
    genre_cols_present = [c for c in item_meta.columns if c.startswith("genre_")]
    keep_cols.extend(sorted(set(genre_cols_present)))

    item_features = item_meta[keep_cols].copy()
    log.info(f"  Item feature table shape: {item_features.shape}")
    return item_features


# ----------------------------------------------------------------------------
# Session helpers for user features
# ----------------------------------------------------------------------------
def add_sessions(df: pd.DataFrame, gap_minutes: int = 30) -> pd.DataFrame:
    """Create session ids from time gaps if explicit sessions are absent."""
    df = df.sort_values(["user_id", "datetime"]).copy()
    prev_dt = df.groupby("user_id")["datetime"].shift()
    new_session = prev_dt.isna() | ((df["datetime"] - prev_dt) > pd.Timedelta(minutes=gap_minutes))
    df["session_id"] = new_session.groupby(df["user_id"]).cumsum()
    return df


def hour_to_bucket(h: float) -> int:
    if pd.isna(h):
        return -1
    h = int(h)
    if 5 <= h < 12:
        return 0  # morning
    if 12 <= h < 17:
        return 1  # afternoon
    if 17 <= h < 22:
        return 2  # evening
    return 3      # night


# ----------------------------------------------------------------------------
# User features
# ----------------------------------------------------------------------------
def build_user_features(df: pd.DataFrame, item_features: pd.DataFrame, user2idx: Dict[str, int]) -> pd.DataFrame:
    """Build one row per user with behaviour, session, and preference features."""
    log.info("Building user feature table …")

    df = df.sort_values(["user_id", "datetime"]).copy()
    df["hour_of_day"] = df["datetime"].dt.hour
    df["is_weekend"] = df["datetime"].dt.dayofweek.isin([5, 6]).astype(int)

    # Core aggregates
    user_agg = df.groupby("user_id").agg(
        user_n_clicks=("item_id", "count"),
        user_n_unique_items=("item_id", "nunique"),
        first_seen=("datetime", "min"),
        last_seen=("datetime", "max"),
        user_mean_hour=("hour_of_day", "mean"),
        user_weekend_ratio=("is_weekend", "mean"),
    ).reset_index()

    user_agg["user_repeat_ratio"] = 1 - (user_agg["user_n_unique_items"] / user_agg["user_n_clicks"].replace(0, np.nan))
    user_agg["user_tenure_days"] = (user_agg["last_seen"] - user_agg["first_seen"]).dt.days
    user_agg["user_active_days_span"] = user_agg["user_tenure_days"]

    # Session features
    df_sessions = add_sessions(df, gap_minutes=30)
    session_stats = df_sessions.groupby(["user_id", "session_id"]).size().reset_index(name="session_len")
    session_agg = session_stats.groupby("user_id").agg(
        user_n_sessions=("session_id", "nunique"),
        user_avg_session_len=("session_len", "mean"),
    ).reset_index()

    # Mean gap between clicks
    def mean_gap_hours(group: pd.DataFrame) -> float:
        times = group["datetime"].sort_values()
        gaps = times.diff().dropna().dt.total_seconds() / 3600
        return float(gaps.mean()) if len(gaps) > 0 else np.nan

    gap_agg = df.groupby("user_id").apply(mean_gap_hours).reset_index()
    gap_agg.columns = ["user_id", "user_avg_session_gap_hours"]

    # Genre preference vector: average clicked genres per user
    genre_cols_present = [c for c in item_features.columns if c.startswith("genre_")]
    item_genre_lookup = item_features[["item_id"] + genre_cols_present].copy()
    df_with_genres = df.merge(item_genre_lookup, on="item_id", how="left")

    for c in genre_cols_present:
        df_with_genres[c] = df_with_genres[c].fillna(0)

    genre_affinity = df_with_genres.groupby("user_id")[genre_cols_present].mean().reset_index()
    genre_affinity = genre_affinity.rename(columns={c: f"user_pref_{c}" for c in genre_cols_present})

    # Release year / decade preferences
    df_with_meta = df.merge(
        item_features[["item_id", "release_year", "decade_bucket"]],
        on="item_id",
        how="left",
    )
    year_pref = df_with_meta.groupby("user_id").agg(
        user_avg_release_year=("release_year", "mean"),
        user_avg_decade_bucket=("decade_bucket", "mean"),
    ).reset_index()

    # Preferred hour bucket
    user_hour_pref = df.groupby("user_id")["hour_of_day"].agg(lambda x: x.mode().iloc[0] if len(x.mode()) else np.nan).reset_index()
    user_hour_pref.columns = ["user_id", "user_preferred_hour"]
    user_hour_pref["user_preferred_hour_bucket"] = user_hour_pref["user_preferred_hour"].apply(hour_to_bucket)

    # Engagement tier from click volume
    user_agg["user_engagement_tier"] = pd.qcut(
        user_agg["user_n_clicks"],
        q=4,
        labels=False,
        duplicates="drop",
    )

    # Merge all parts
    user_features = (
        user_agg
        .merge(session_agg, on="user_id", how="left")
        .merge(gap_agg, on="user_id", how="left")
        .merge(genre_affinity, on="user_id", how="left")
        .merge(year_pref, on="user_id", how="left")
        .merge(user_hour_pref[["user_id", "user_preferred_hour_bucket"]], on="user_id", how="left")
    )

    # Fill defaults where needed
    user_features["user_n_sessions"] = user_features["user_n_sessions"].fillna(0).astype(int)
    user_features["user_avg_session_len"] = user_features["user_avg_session_len"].fillna(0)
    user_features["user_avg_session_gap_hours"] = user_features["user_avg_session_gap_hours"].fillna(0)
    user_features["user_preferred_hour_bucket"] = user_features["user_preferred_hour_bucket"].fillna(-1).astype(int)
    user_features["user_engagement_tier"] = user_features["user_engagement_tier"].fillna(-1).astype(int)

    # Integer index via shared mapping
    user_features["user_idx"] = user_features["user_id"].map(user2idx)
    user_features["user_idx"] = user_features["user_idx"].astype(int)

    # Order columns
    base_cols = [
        "user_idx",
        "user_id",
        "user_n_clicks",
        "user_n_unique_items",
        "user_repeat_ratio",
        "user_n_sessions",
        "user_avg_session_len",
        "user_mean_hour",
        "user_weekend_ratio",
        "user_active_days_span",
        "first_seen",
        "last_seen",
        "user_tenure_days",
        "user_engagement_tier",
        "user_preferred_hour_bucket",
        "user_avg_session_gap_hours",
        "user_avg_release_year",
        "user_avg_decade_bucket",
    ]

    # Include all genre preference columns
    pref_cols = [c for c in user_features.columns if c.startswith("user_pref_genre_")]
    keep_cols = base_cols + sorted(pref_cols)

    user_features = user_features[keep_cols].copy()
    log.info(f"  User feature table shape: {user_features.shape}")
    return user_features


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    raw = load_raw(INPUT_CSV)
    user2idx, item2idx = build_id_maps(raw)

    # 1) Interaction log
    interactions = build_interaction_log(raw, user2idx, item2idx)
    interactions.to_csv(OUT_DATA, index=False)
    log.info(f"Saved interaction log -> {OUT_DATA} ({len(interactions):,} rows)")

    # 2) Item features
    item_feats = build_item_features(raw, item2idx)
    item_feats.to_csv(OUT_ITEMS, index=False)
    log.info(f"Saved item features   -> {OUT_ITEMS} ({len(item_feats):,} items)")

    # 3) User features
    user_feats = build_user_features(raw, item_feats, user2idx)
    user_feats.to_csv(OUT_USERS, index=False)
    log.info(f"Saved user features   -> {OUT_USERS} ({len(user_feats):,} users)")

    # Sanity checks
    log.info("\n── Sanity checks ──────────────────────────────────────────")
    log.info(f"data.csv columns      : {interactions.columns.tolist()}")
    log.info(f"item_features columns : {item_feats.columns.tolist()[:12]} …")
    log.info(f"user_features columns : {user_feats.columns.tolist()[:12]} …")

    genre_cols_present = [c for c in item_feats.columns if c.startswith("genre_")]
    genre_coverage = (item_feats[genre_cols_present].sum(axis=1) > 0).mean() if genre_cols_present else 0.0
    log.info(f"Items with >=1 genre  : {genre_coverage:.1%}")

    date_coverage = item_feats["release_year"].notna().mean()
    log.info(f"Items with year       : {date_coverage:.1%}")
    log.info("Done.")


if __name__ == "__main__":
    main()
