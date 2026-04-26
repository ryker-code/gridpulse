"""
Fetches real-time LMP data from GridStatus.io API with 5-minute caching
and exponential backoff on 429 rate limit errors.

Also fetches supplemental signal data via the gridstatus Python library:
  - Actual load & load forecast (Signal 3)
  - Fuel mix / total generation (Signal 4 reserve margin proxy)
  - Day-ahead LMP (Signal 5 DART)
  - 30-day load history for percentile scoring (Signal 3, cached 1 hour)

Verified column/dataset facts (confirmed via live API probes):
  PJM  : pjm_lmp_real_time_5_min          | lmp, energy, congestion | location_type=HUB
         Hubs: WESTERN HUB, DOMINION HUB
  CAISO: caiso_lmp_real_time_5_min         | lmp, energy, congestion | location_type=Trading Hub
         Zones: TH_NP15_GEN-APND, TH_SP15_GEN-APND, TH_ZP26_GEN-APND
  ERCOT: ercot_lmp_by_settlement_point     | lmp only (no components) | location_type=Trading Hub
         Hubs: HB_NORTH, HB_HOUSTON, HB_SOUTH, HB_WEST, HB_BUSAVG
  MISO : miso_lmp_real_time_5_min_ex_ante  | lmp, energy, congestion | location_type=Hub
         Hubs: INDIANA.HUB, ILLINOIS.HUB
  ISONE: isone_lmp_real_time_5_min_prelim  | lmp, energy, congestion | filter by location=.H.INTERNAL_HUB
  NYISO: nyiso_lmp_real_time_5_min         | lmp, energy, congestion | location_type=Zone
         Zones: N.Y.C., LONGIL
"""

import time
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Optional

logging.getLogger("gridstatus").setLevel(logging.WARNING)

def _get_api_key() -> str:
    try:
        import streamlit as st
        return st.secrets["GRIDSTATUS_API_KEY"]
    except Exception:
        return "ef58d55f5a5f4f54a58102984ca32925"

API_KEY = _get_api_key()
BASE_URL = "https://api.gridstatus.io/v1/datasets"

# In-memory cache: {cache_key: (epoch_ts, data)}
_cache: dict = {}
CACHE_TTL_SECONDS = 300      # 5 minutes for live data
CACHE_TTL_HISTORY = 3600     # 1 hour for 30-day load history

# ── Dataset configs per ISO ───────────────────────────────────────────────��───

ISO_CONFIG = {
    "PJM": {
        "dataset": "pjm_lmp_real_time_5_min",
        "lmp_col": "lmp", "energy_col": "energy", "congestion_col": "congestion",
        "time_col": "interval_start_utc", "location_col": "location",
        "hub_filter": {"filter_column": "location_type", "filter_value": "HUB", "filter_operator": "="},
        "target_hubs": ["WESTERN HUB", "DOMINION HUB"],
        "aux_hubs": [],
        "da_dataset": "pjm_lmp_day_ahead_hourly", "da_lmp_col": "lmp",
        "display_name": "PJM",
    },
    "CAISO": {
        "dataset": "caiso_lmp_real_time_5_min",
        "lmp_col": "lmp", "energy_col": "energy", "congestion_col": "congestion",
        "time_col": "interval_start_utc", "location_col": "location",
        "hub_filter": {"filter_column": "location_type", "filter_value": "Trading Hub", "filter_operator": "="},
        "target_hubs": ["TH_NP15_GEN-APND", "TH_SP15_GEN-APND"],
        "aux_hubs": [],
        "da_dataset": "caiso_lmp_day_ahead_hourly", "da_lmp_col": "lmp",
        "display_name": "CAISO",
    },
    "ERCOT": {
        "dataset": "ercot_lmp_by_settlement_point",
        "lmp_col": "lmp", "energy_col": None, "congestion_col": None,
        "time_col": "interval_start_utc", "location_col": "location",
        "hub_filter": {"filter_column": "location_type", "filter_value": "Trading Hub", "filter_operator": "="},
        "target_hubs": ["HB_NORTH", "HB_HOUSTON", "HB_SOUTH"],
        "aux_hubs": ["HB_BUSAVG"],  # fetched for congestion proxy but not scored separately
        "da_dataset": "ercot_spp_day_ahead_hourly", "da_lmp_col": "spp",
        "display_name": "ERCOT",
    },
    "MISO": {
        "dataset": "miso_lmp_real_time_5_min_ex_ante",
        "lmp_col": "lmp", "energy_col": "energy", "congestion_col": "congestion",
        "time_col": "interval_start_utc", "location_col": "location",
        "hub_filter": {"filter_column": "location_type", "filter_value": "Hub", "filter_operator": "="},
        "target_hubs": ["INDIANA.HUB", "ILLINOIS.HUB"],
        "aux_hubs": [],
        "da_dataset": "miso_lmp_day_ahead_hourly_ex_ante", "da_lmp_col": "lmp",
        "display_name": "MISO",
    },
    "ISONE": {
        "dataset": "isone_lmp_real_time_5_min_prelim",
        "lmp_col": "lmp", "energy_col": "energy", "congestion_col": "congestion",
        "time_col": "interval_start_utc", "location_col": "location",
        "hub_filter": {"filter_column": "location", "filter_value": ".H.INTERNAL_HUB", "filter_operator": "="},
        "target_hubs": [".H.INTERNAL_HUB"],
        "aux_hubs": [],
        "da_dataset": "isone_lmp_day_ahead_hourly", "da_lmp_col": "lmp",
        "display_name": "ISONE",
    },
    "NYISO": {
        "dataset": "nyiso_lmp_real_time_5_min",
        "lmp_col": "lmp", "energy_col": "energy", "congestion_col": "congestion",
        "time_col": "interval_start_utc", "location_col": "location",
        "hub_filter": {"filter_column": "location_type", "filter_value": "Zone", "filter_operator": "="},
        "target_hubs": ["N.Y.C.", "LONGIL"],
        "aux_hubs": [],
        "da_dataset": "nyiso_lmp_day_ahead_hourly", "da_lmp_col": "lmp",
        "display_name": "NYISO",
    },
    "SPP": {
        "dataset": "spp_lmp_real_time_5_min",
        "lmp_col": "lmp", "energy_col": "energy", "congestion_col": "congestion",
        "time_col": "interval_start_utc", "location_col": "location",
        "hub_filter": {"filter_column": "location_type", "filter_value": "HUB", "filter_operator": "="},
        "target_hubs": ["SPPNORTH_HUB", "SPPSOUTH_HUB"],
        "aux_hubs": [],
        "da_dataset": "spp_lmp_day_ahead_hourly", "da_lmp_col": "lmp",
        "display_name": "SPP",
    },
}

# Forecast column name varies per ISO in the gridstatus library
_FORECAST_COL = {
    "CAISO": "Load Forecast",
    "ERCOT": "System Total",
    "MISO":  "MISO MTLF",
    "ISONE": "Load Forecast",
    "NYISO": "Load Forecast",
    "PJM":   "Load Forecast",
    "SPP":   "Load Forecast",
}

# ── REST API helpers ──────────────────────────────────────────────────────────

_last_request_time: float = 0.0


def _rate_limited_get(url: str, params: dict, max_retries: int = 5) -> dict:
    """GET with 1 req/s rate limiting and exponential backoff on 429."""
    global _last_request_time
    delay = 1.0
    for attempt in range(max_retries):
        elapsed = time.time() - _last_request_time
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)
        try:
            _last_request_time = time.time()
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                wait = delay + attempt * 0.5
                print(f"Rate limited, waiting {wait:.1f}s (attempt {attempt + 1})")
                time.sleep(wait)
                delay = min(delay * 2, 60)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as e:
            if attempt == max_retries - 1:
                raise
            print(f"HTTP error: {e}, retrying in {delay:.1f}s")
            time.sleep(delay)
            delay = min(delay * 2, 60)
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise
            print(f"Request error: {e}, retrying in {delay:.1f}s")
            time.sleep(delay)
            delay = min(delay * 2, 60)
    raise RuntimeError(f"Failed after {max_retries} attempts")


def _is_fresh(key: str, ttl: float = CACHE_TTL_SECONDS) -> bool:
    return key in _cache and (time.time() - _cache[key][0]) < ttl


def _query(dataset: str, params: dict) -> Optional[pd.DataFrame]:
    url = f"{BASE_URL}/{dataset}/query"
    params = {**params, "api_key": API_KEY}
    try:
        data = _rate_limited_get(url, params)
        records = data.get("data", [])
        return pd.DataFrame(records) if records else None
    except Exception as e:
        print(f"[{dataset}] query failed: {e}")
        return None


# ── LMP fetch (Signals 1 & 2) ─────────────────────────────────────────────────

def fetch_latest(iso: str) -> Optional[pd.DataFrame]:
    """Latest LMP for all target + aux hubs. Normalized to time/location/lmp columns."""
    cfg = ISO_CONFIG.get(iso)
    if not cfg:
        raise ValueError(f"Unknown ISO: {iso}")

    cache_key = f"{iso}_latest"
    if _is_fresh(cache_key):
        return _cache[cache_key][1].copy()

    params = {"time": "latest", "limit": 500}
    if cfg["hub_filter"]:
        params.update(cfg["hub_filter"])

    df = _query(cfg["dataset"], params)
    if df is None or df.empty:
        print(f"[{iso}] No latest data returned")
        return None

    df = _normalize(df, iso, cfg)
    _cache[cache_key] = (time.time(), df)
    return df.copy()


def fetch_history(iso: str, hours: int = 24) -> Optional[pd.DataFrame]:
    """Historical LMP for target + aux hubs over the trailing `hours`."""
    cfg = ISO_CONFIG.get(iso)
    if not cfg:
        raise ValueError(f"Unknown ISO: {iso}")

    cache_key = f"{iso}_history_{hours}h"
    if _is_fresh(cache_key):
        return _cache[cache_key][1].copy()

    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=hours)
    params = {
        "start_time": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end_time":   end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "limit": 5000,
    }
    if cfg["hub_filter"]:
        params.update(cfg["hub_filter"])

    df = _query(cfg["dataset"], params)
    if df is None or df.empty:
        return None

    df = _normalize(df, iso, cfg)
    _cache[cache_key] = (time.time(), df)
    return df.copy()


def _normalize(df: pd.DataFrame, iso: str, cfg: dict) -> pd.DataFrame:
    """Rename to standard schema and filter to target+aux hubs."""
    col_map = {cfg["time_col"]: "time", cfg["location_col"]: "location", cfg["lmp_col"]: "lmp"}
    if cfg.get("energy_col") and cfg["energy_col"] in df.columns:
        col_map[cfg["energy_col"]] = "energy"
    if cfg.get("congestion_col") and cfg["congestion_col"] in df.columns:
        col_map[cfg["congestion_col"]] = "congestion"
    df = df.rename(columns=col_map)

    all_hubs = (cfg.get("target_hubs") or []) + (cfg.get("aux_hubs") or [])
    if all_hubs and "location" in df.columns:
        mask = df["location"].str.upper().isin([h.upper() for h in all_hubs])
        df = df[mask]
        if df.empty:
            print(f"[{iso}] Hub filter matched 0 rows; returning unfiltered")

    if "lmp" in df.columns:
        df["lmp"] = pd.to_numeric(df["lmp"], errors="coerce")
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")

    return df


def get_region_lmp(iso: str) -> Optional[dict]:
    """
    Returns {hub: {lmp, energy, congestion, time}} for target hubs only.
    Aux hubs (e.g. HB_BUSAVG) are returned under their own key for use as
    the ERCOT congestion proxy but are not treated as scored regions.
    """
    df = fetch_latest(iso)
    if df is None or df.empty or "location" not in df.columns:
        return None

    cfg = ISO_CONFIG[iso]
    all_hubs = (cfg.get("target_hubs") or []) + (cfg.get("aux_hubs") or [])
    result = {}
    for hub in all_hubs:
        hub_df = df[df["location"].str.upper() == hub.upper()]
        if hub_df.empty:
            continue
        row = hub_df.sort_values("time", ascending=False).iloc[0] if "time" in hub_df.columns else hub_df.iloc[0]
        result[hub] = {
            "lmp": round(float(row["lmp"]), 2) if pd.notna(row.get("lmp")) else None,
            "energy": round(float(row["energy"]), 2) if "energy" in row.index and pd.notna(row.get("energy")) else None,
            "congestion": round(float(row["congestion"]), 4) if "congestion" in row.index and pd.notna(row.get("congestion")) else None,
            "time": row.get("time"),
        }
    return result if result else None


def get_all_regions() -> dict:
    """Fetch latest LMP data for all ISOs. Returns {iso: {hub: {...}}}."""
    result = {}
    for iso in ["PJM", "CAISO", "ERCOT", "MISO", "ISONE", "NYISO", "SPP"]:
        data = get_region_lmp(iso)
        if data:
            result[iso] = data
        else:
            print(f"[{iso}] Failed to retrieve data")
    return result


# ── gridstatus library helpers (Signals 3, 4) ─────────────────────────────────

_iso_instances: dict = {}


def _get_iso_lib(iso: str):
    """Return a cached gridstatus library instance for the given ISO."""
    if iso in _iso_instances:
        return _iso_instances[iso]
    try:
        import gridstatus
        class_map = {
            "CAISO": ("CAISO", {}),
            "ERCOT": ("Ercot", {}),
            "MISO":  ("MISO", {}),
            "ISONE": ("ISONE", {}),
            "NYISO": ("NYISO", {}),
            "PJM":   ("PJM",   {"api_key": API_KEY}),
            "SPP":   ("SPP",   {}),
        }
        if iso not in class_map:
            return None
        cls_name, kwargs = class_map[iso]
        obj = getattr(gridstatus, cls_name)(**kwargs)
        _iso_instances[iso] = obj
        return obj
    except Exception as e:
        print(f"[{iso}] gridstatus init error: {e}")
        return None


def _closest_value(df: pd.DataFrame, time_col: str, val_col: str) -> Optional[float]:
    """Return the value in val_col from the row with time_col closest to now."""
    if df is None or df.empty or val_col not in df.columns:
        return None
    now = datetime.now(timezone.utc)
    df = df.copy()
    times = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    idx = (times - now).abs().argmin()
    val = df[val_col].iloc[idx]
    try:
        return float(val) if pd.notna(val) else None
    except (TypeError, ValueError):
        return None


def fetch_current_load(iso: str) -> Optional[float]:
    """Latest actual system load in MW from gridstatus library."""
    cache_key = f"{iso}_load_actual"
    if _is_fresh(cache_key):
        return _cache[cache_key][1]

    lib = _get_iso_lib(iso)
    if lib is None:
        return None
    try:
        df = lib.get_load("today")
        val = _closest_value(df, "Time", "Load")
        _cache[cache_key] = (time.time(), val)
        return val
    except Exception as e:
        print(f"[{iso}] fetch_current_load error: {e}")
        return None


def fetch_current_forecast(iso: str) -> Optional[float]:
    """Latest load forecast in MW from gridstatus library."""
    cache_key = f"{iso}_load_forecast"
    if _is_fresh(cache_key):
        return _cache[cache_key][1]

    lib = _get_iso_lib(iso)
    if lib is None:
        return None
    try:
        df = lib.get_load_forecast("today")
        fc_col = _FORECAST_COL.get(iso, "Load Forecast")
        # Try Time column first, fall back to Interval Start
        time_col = "Time" if "Time" in df.columns else "Interval Start"
        val = _closest_value(df, time_col, fc_col)
        _cache[cache_key] = (time.time(), val)
        return val
    except Exception as e:
        print(f"[{iso}] fetch_current_forecast error: {e}")
        return None


def fetch_fuel_mix_gen(iso: str) -> Optional[dict]:
    """
    Returns {"current_gen_mw": float, "max_gen_48h_mw": float} for reserve margin.
    Sums all positive-value fuel columns (excludes batteries/storage which can be negative).
    """
    cache_key = f"{iso}_fuel_gen"
    if _is_fresh(cache_key):
        return _cache[cache_key][1]

    lib = _get_iso_lib(iso)
    if lib is None:
        return None
    try:
        df = lib.get_fuel_mix("today")
        skip = {"Time", "Interval Start", "Interval End"}
        fuel_cols = [c for c in df.columns if c not in skip]
        numeric = df[fuel_cols].apply(pd.to_numeric, errors="coerce")
        # Sum only positive generation (negative = storage charging or net imports)
        gen_series = numeric.clip(lower=0).sum(axis=1)
        current_gen = float(gen_series.iloc[-1]) if not gen_series.empty else None
        max_gen = float(gen_series.max()) if not gen_series.empty else None
        result = {"current_gen_mw": current_gen, "max_gen_48h_mw": max_gen}
        _cache[cache_key] = (time.time(), result)
        return result
    except Exception as e:
        print(f"[{iso}] fetch_fuel_mix_gen error: {e}")
        return None


def fetch_load_history_30d(iso: str) -> Optional[pd.Series]:
    """
    Returns a Series of load values over the trailing 30 days for percentile scoring.
    Uses date-range or day-by-day fetching depending on what the ISO library supports.
    Cached for 1 hour. Falls back gracefully to None on any error.
    """
    cache_key = f"{iso}_load_30d"
    if _is_fresh(cache_key, ttl=CACHE_TTL_HISTORY):
        return _cache[cache_key][1]

    lib = _get_iso_lib(iso)
    if lib is None:
        return None

    import inspect
    sig = inspect.signature(lib.get_load)
    params = set(sig.parameters.keys())

    try:
        end = datetime.now(timezone.utc).date()
        start = end - timedelta(days=30)

        if "start" in params and "end" in params:
            # Most ISOs: CAISO, ISONE, NYISO support start/end kwargs
            df = lib.get_load(start=str(start), end=str(end))
        elif "date" in params:
            # ISOs with single-day API (MISO, ERCOT): fetch day-by-day, collect all
            frames = []
            for d in pd.date_range(start, end, freq="D"):
                try:
                    day_df = lib.get_load(date=d)
                    if day_df is not None and not day_df.empty:
                        frames.append(day_df)
                except Exception:
                    continue
            df = pd.concat(frames, ignore_index=True) if frames else None
        else:
            # Fallback: try positional "today" and accept partial data
            df = lib.get_load("today")

        if df is None or df.empty or "Load" not in df.columns:
            return None

        series = pd.to_numeric(df["Load"], errors="coerce").dropna()
        _cache[cache_key] = (time.time(), series)
        return series
    except Exception as e:
        print(f"[{iso}] fetch_load_history_30d error: {e}")
        return None


# ── DA LMP fetch (Signal 5 DART) ──────────────────────────────────────────────

def fetch_da_lmp(iso: str, hub: str) -> Optional[float]:
    """
    Day-ahead LMP for the current UTC hour from the GridStatus REST API.
    Returns None if the ISO has no DA dataset configured.
    """
    cfg = ISO_CONFIG.get(iso, {})
    da_dataset = cfg.get("da_dataset")
    da_lmp_col = cfg.get("da_lmp_col")
    if not da_dataset or not da_lmp_col:
        return None

    cache_key = f"{iso}_{hub}_da_lmp"
    if _is_fresh(cache_key):
        return _cache[cache_key][1]

    now = datetime.now(timezone.utc)
    hour_start = now.replace(minute=0, second=0, microsecond=0)
    # Fetch a 2-hour window around the current hour to ensure we catch it
    start_s = (hour_start - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    end_s   = (hour_start + timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")

    params = {
        "start_time": start_s,
        "end_time": end_s,
        "limit": 50,
        "filter_column": "location",
        "filter_value": hub,
        "filter_operator": "=",
    }
    df = _query(da_dataset, params)
    if df is None or df.empty or da_lmp_col not in df.columns:
        return None

    df["_t"] = pd.to_datetime(df.get("interval_start_utc", df.columns[0]), utc=True, errors="coerce")
    df["_lmp"] = pd.to_numeric(df[da_lmp_col], errors="coerce")
    df = df.dropna(subset=["_t", "_lmp"])
    if df.empty:
        return None
    idx = (df["_t"] - now).abs().argmin()
    val = float(df["_lmp"].iloc[idx])
    _cache[cache_key] = (time.time(), val)
    return val


# ── DA LMP forecast (next N hours) ───────────────────────────────────────────

def fetch_da_lmp_forecast(iso: str, hub: str, hours: int = 12) -> Optional[pd.DataFrame]:
    """
    Day-ahead LMP for the next `hours` hours, used as forward stress proxy.
    Returns DataFrame[time, lmp] sorted ascending, or None.
    """
    cfg = ISO_CONFIG.get(iso, {})
    da_dataset = cfg.get("da_dataset")
    da_lmp_col = cfg.get("da_lmp_col")
    if not da_dataset or not da_lmp_col:
        return None

    cache_key = f"{iso}_{hub}_da_forecast_{hours}h"
    if _is_fresh(cache_key, ttl=1800):   # 30-min cache — DA is published for the full day
        return _cache[cache_key][1].copy()

    now = datetime.now(timezone.utc)
    hour_start = now.replace(minute=0, second=0, microsecond=0)
    start_s = hour_start.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_s   = (hour_start + timedelta(hours=hours + 2)).strftime("%Y-%m-%dT%H:%M:%SZ")

    params = {
        "start_time": start_s,
        "end_time": end_s,
        "limit": 50,
        "filter_column": "location",
        "filter_value": hub,
        "filter_operator": "=",
    }
    df = _query(da_dataset, params)
    if df is None or df.empty or da_lmp_col not in df.columns:
        return None

    time_col = "interval_start_utc" if "interval_start_utc" in df.columns else df.columns[0]
    df = df.rename(columns={time_col: "time", da_lmp_col: "lmp"})
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df["lmp"] = pd.to_numeric(df["lmp"], errors="coerce")
    df = df[["time", "lmp"]].dropna().sort_values("time").reset_index(drop=True)
    if df.empty:
        return None

    _cache[cache_key] = (time.time(), df)
    return df.copy()


def fetch_load_forecast_series(iso: str, hours: int = 12) -> Optional[pd.DataFrame]:
    """
    Load forecast for the next `hours` hours from the gridstatus library.
    Returns DataFrame[time, load] or None.
    """
    cache_key = f"{iso}_load_forecast_fwd_{hours}h"
    if _is_fresh(cache_key, ttl=1800):
        return _cache[cache_key][1].copy()

    lib = _get_iso_lib(iso)
    if lib is None:
        return None

    try:
        df = lib.get_load_forecast("today")
        if df is None or df.empty:
            return None

        fc_col = _FORECAST_COL.get(iso, "Load Forecast")
        time_col = "Time" if "Time" in df.columns else (
            "Interval Start" if "Interval Start" in df.columns else None
        )
        if time_col is None or fc_col not in df.columns:
            return None

        result = pd.DataFrame({
            "time": pd.to_datetime(df[time_col], utc=True, errors="coerce"),
            "load": pd.to_numeric(df[fc_col], errors="coerce"),
        }).dropna().sort_values("time")

        now = datetime.now(timezone.utc)
        result = result[result["time"] >= now].head(hours + 2).reset_index(drop=True)
        if result.empty:
            return None

        _cache[cache_key] = (time.time(), result)
        return result.copy()
    except Exception as e:
        print(f"[{iso}] fetch_load_forecast_series error: {e}")
        return None


# ── Bundled extended data fetch ───────────────────────────────────────────────

def get_extended_data(iso: str) -> dict:
    """
    Fetch all supplemental signal data for one ISO in a single call.
    Returns dict with keys: actual_load, forecast_load, fuel_gen, load_history_30d.
    DA LMP is fetched per-hub in get_all_stress_scores since it's hub-specific.
    All values are None on error (callers must handle gracefully).
    """
    return {
        "actual_load":     fetch_current_load(iso),
        "forecast_load":   fetch_current_forecast(iso),
        "fuel_gen":        fetch_fuel_mix_gen(iso),
        "load_history_30d": fetch_load_history_30d(iso),
    }


if __name__ == "__main__":
    print("Fetching latest LMP data...\n")
    all_data = get_all_regions()
    for iso, hubs in all_data.items():
        print(f"=== {iso} ===")
        for hub, vals in hubs.items():
            print(f"  {hub}: LMP={vals['lmp']} $/MWh  @ {vals['time']}")
    print()
    print("Fetching extended data for CAISO and ERCOT...")
    for iso in ["CAISO", "ERCOT"]:
        ext = get_extended_data(iso)
        print(f"  [{iso}] load={ext['actual_load']} MW  forecast={ext['forecast_load']} MW  fuel_gen={ext['fuel_gen']}")
