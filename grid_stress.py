"""
Grid Stress Score computation (0-100 scale, higher = more stressed).

Five signals (weights sum to 1.0):
  1. LMP Deviation Score    (30%) — how far current LMP is from recent mean
  2. LMP Velocity Score     (15%) — rate of LMP change over recent intervals
  3. Load vs Forecast Dev   (20%) — actual vs forecast load + 30d percentile
  4. Reserve Margin Score   (25%) — (peak_cap_72h * 1.05 - load) / load vs NERC thresholds
  5. DART Score             (10%) — RT_LMP - DA_LMP as % of DA_LMP

Signals with unavailable data return None and are excluded from composite;
weights are renormalized over available signals only.
"""

import numpy as np
import pandas as pd
from typing import Optional
from data_cache import (
    fetch_latest, fetch_history, fetch_da_lmp,
    get_extended_data, ISO_CONFIG
)

WEIGHTS = {
    "lmp_deviation": 0.30,
    "lmp_velocity":  0.15,
    "load_forecast": 0.20,
    "reserve_margin": 0.25,
    "dart":          0.10,
}

# Regions we compute scores for: (iso, hub_label)
REGIONS = [
    ("PJM",   "WESTERN HUB"),
    ("PJM",   "DOMINION HUB"),
    ("CAISO", "TH_NP15_GEN-APND"),
    ("CAISO", "TH_SP15_GEN-APND"),
    ("ERCOT", "HB_NORTH"),
    ("ERCOT", "HB_HOUSTON"),
    ("MISO",  "INDIANA.HUB"),
    ("ISONE", ".H.INTERNAL_HUB"),
    ("NYISO", "N.Y.C."),
    ("SPP",   "SPPNORTH_HUB"),
]


def _clamp(val: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, val))


# ── Signal 1: LMP Deviation ──────────────────────────────────────────────────

def lmp_deviation_score(
    current_lmp: float,
    history_lmps: pd.Series,
    congestion_proxy: Optional[float] = None,
) -> float:
    """
    Deviation from recent mean baseline.
    For ERCOT hubs, congestion_proxy = hub_lmp - HB_BUSAVG_lmp is blended in
    with 1.5x weight since ERCOT has no published LMP components.
    """
    if history_lmps.empty or len(history_lmps) < 3:
        return 50.0

    baseline = history_lmps.mean()
    if baseline <= 0:
        return 50.0

    lmp_deviation = (current_lmp - baseline) / baseline

    if congestion_proxy is not None:
        cong_deviation = congestion_proxy / max(abs(baseline), 1.0)
        # Blend: 1× LMP deviation + 1.5× congestion component, normalized
        blended = (lmp_deviation + 1.5 * cong_deviation) / 2.5
        score = 30.0 + blended * 100.0
    else:
        score = 30.0 + lmp_deviation * 100.0

    return _clamp(score)


# ── Signal 2: LMP Velocity ───────────────────────────────────────────────────

def lmp_velocity_score(recent_lmps: pd.Series) -> float:
    """% change over trailing ~15 minutes (last 4 intervals of 5-min data)."""
    if len(recent_lmps) < 2:
        return 50.0
    tail = recent_lmps.dropna().tail(4)
    if len(tail) < 2:
        return 50.0
    oldest, newest = float(tail.iloc[0]), float(tail.iloc[-1])
    if oldest <= 0:
        return 50.0
    pct_change = (newest - oldest) / oldest
    return _clamp(40.0 + pct_change * 150.0)


# ── Signal 3: Load vs Forecast Deviation ─────────────────────────────────────

def load_forecast_score(
    actual_load: Optional[float],
    forecast_load: Optional[float],
    history_30d: Optional[pd.Series] = None,
) -> Optional[float]:
    """
    load_score = 0.5 * deviation_component + 0.5 * percentile_component

    deviation_component: clamp(load_deviation / 0.08, 0, 1) * 100
      — reaches 100 when actual exceeds forecast by 8%+
    percentile_component: where today's load sits in 30-day same-hour distribution * 100
      — falls back to 0 if history unavailable, leaving only deviation component
    """
    if actual_load is None or forecast_load is None or forecast_load <= 0:
        return None

    load_deviation = (actual_load - forecast_load) / forecast_load
    deviation_component = _clamp(load_deviation / 0.08, 0.0, 1.0) * 100.0

    if history_30d is not None and len(history_30d) >= 10:
        try:
            from scipy.stats import percentileofscore
            pct = percentileofscore(history_30d.dropna().values, actual_load, kind="rank")
            percentile_component = pct  # already 0–100
            return 0.5 * deviation_component + 0.5 * percentile_component
        except Exception:
            pass

    # No history: use deviation component alone (effectively doubled weight)
    return deviation_component


# ── Signal 4: Reserve Margin ─────────────────────────────────────────────────

def reserve_margin_score(
    current_gen_mw: Optional[float],
    max_gen_48h_mw: Optional[float],
    current_load_mw: Optional[float],
) -> Optional[float]:
    """
    reserve_margin = (peak_capacity - current_load) / current_load
    where peak_capacity = max_gen_48h * 1.05  (NERC planning buffer)

    NERC threshold mapping:
      >= 20% →  0  (adequate)
      15–20% → 25  (elevated concern)
      10–15% → 60  (high)
       5–10% → 90  (emergency)
        < 5% → 100 (critical)
    """
    if any(v is None or v <= 0 for v in [current_gen_mw, max_gen_48h_mw, current_load_mw]):
        return None

    peak_capacity = max_gen_48h_mw * 1.05
    reserve_margin = (peak_capacity - current_load_mw) / current_load_mw

    if reserve_margin >= 0.20:
        return 0.0
    elif reserve_margin >= 0.15:
        return 25.0
    elif reserve_margin >= 0.10:
        return 60.0
    elif reserve_margin >= 0.05:
        return 90.0
    else:
        return 100.0


# ── Signal 5: DART ───────────────────────────────────────────────────────────

def dart_score(rt_lmp: float, da_lmp: Optional[float]) -> Optional[float]:
    """
    DART = RT_LMP - DA_LMP
    dart_score = clamp(DART / (DA_LMP * 0.20), 0, 1) * 100
    Reaches 100 when RT exceeds DA by 20%+.
    Negative DART (RT < DA) → 0 (RT cheaper than DA is a low-stress signal).
    """
    if da_lmp is None or da_lmp <= 0:
        return None
    dart = rt_lmp - da_lmp
    return _clamp(dart / (da_lmp * 0.20), 0.0, 1.0) * 100.0


# ── Composite Score ───────────────────────────────────────────────────────────

def compute_stress_score(
    iso: str,
    hub: str,
    current_lmp: float,
    history_df: Optional[pd.DataFrame] = None,
    congestion_proxy: Optional[float] = None,
    actual_load: Optional[float] = None,
    forecast_load: Optional[float] = None,
    fuel_gen: Optional[dict] = None,
    load_history_30d: Optional[pd.Series] = None,
    da_lmp: Optional[float] = None,
) -> dict:
    """
    Compute composite Grid Stress Score for one hub.
    Returns a dict with per-signal scores and composite (weights renormalized over live signals).
    """
    # ── Extract hub history series ────────────────────────────────────────────
    history_lmps = pd.Series(dtype=float)
    if history_df is not None and not history_df.empty and "lmp" in history_df.columns:
        hub_upper = hub.upper()
        if "location" in history_df.columns:
            hub_rows = history_df[history_df["location"].str.upper() == hub_upper]
        else:
            hub_rows = history_df
        if not hub_rows.empty:
            if "time" in hub_rows.columns:
                hub_rows = hub_rows.sort_values("time")
            history_lmps = hub_rows["lmp"].dropna()

    recent_lmps = history_lmps.tail(8)

    # ── Compute each signal ───────────────────────────────────────────────────
    s1 = lmp_deviation_score(current_lmp, history_lmps, congestion_proxy)
    s2 = lmp_velocity_score(recent_lmps)
    s3 = load_forecast_score(actual_load, forecast_load, load_history_30d)
    s4 = reserve_margin_score(
        fuel_gen.get("current_gen_mw") if fuel_gen else None,
        fuel_gen.get("max_gen_48h_mw") if fuel_gen else None,
        actual_load,
    )
    s5 = dart_score(current_lmp, da_lmp)

    signal_values = {
        "lmp_deviation": s1,
        "lmp_velocity":  s2,
        "load_forecast": s3,
        "reserve_margin": s4,
        "dart":          s5,
    }

    # ── Renormalize weights over available signals ────────────────────────────
    available = {k: v for k, v in signal_values.items() if v is not None}
    total_weight = sum(WEIGHTS[k] for k in available)
    if total_weight <= 0:
        composite = 50.0
    else:
        composite = sum(v * WEIGHTS[k] / total_weight for k, v in available.items())

    return {
        "iso": iso,
        "hub": hub,
        "lmp": float(current_lmp),
        "stress_score": round(_clamp(composite), 1),
        "signals": {k: round(float(v), 1) if v is not None else None for k, v in signal_values.items()},
        "active_signals": len(available),
        "total_signals": 5,
        "signal_weights": {k: round(WEIGHTS[k] / total_weight, 3) for k in available},
    }


def stress_label(score: float) -> tuple[str, str]:
    """Returns (label, hex_color) for a stress score."""
    if score < 25:
        return "LOW", "#22c55e"
    elif score < 50:
        return "MODERATE", "#eab308"
    elif score < 75:
        return "HIGH", "#f97316"
    else:
        return "CRITICAL", "#ef4444"


def historical_stress_series(history_df: pd.DataFrame, iso: str, hub: str) -> pd.DataFrame:
    """
    Vectorized stress score computation over a historical DataFrame.
    Uses Signals 1+2 (deviation + velocity) only since load/DART data isn't historical.
    Returns DataFrame[time, stress_score, lmp] sorted ascending.
    """
    if history_df is None or history_df.empty or "lmp" not in history_df.columns:
        return pd.DataFrame(columns=["time", "stress_score", "lmp"])

    hub_upper = hub.upper()
    if "location" in history_df.columns:
        ts = history_df[history_df["location"].str.upper() == hub_upper].copy()
    else:
        ts = history_df.copy()
    if "time" in ts.columns:
        ts = ts.sort_values("time")

    ts = ts[["time", "lmp"]].dropna().reset_index(drop=True)
    if len(ts) < 2:
        return pd.DataFrame(columns=["time", "stress_score", "lmp"])

    lmps = ts["lmp"].astype(float)

    expanding_mean = lmps.expanding(min_periods=3).mean().fillna(lmps.mean())
    deviation_pct = (lmps - expanding_mean) / expanding_mean.clip(lower=0.01)
    deviation_score = (30.0 + deviation_pct * 100.0).clip(0.0, 100.0)

    periods = min(4, max(1, len(lmps) // 8))
    velocity_pct = lmps.pct_change(periods=periods).fillna(0.0)
    velocity_score = (40.0 + velocity_pct * 150.0).clip(0.0, 100.0)

    active_weight = WEIGHTS["lmp_deviation"] + WEIGHTS["lmp_velocity"]
    stress = (
        (WEIGHTS["lmp_deviation"] * deviation_score + WEIGHTS["lmp_velocity"] * velocity_score)
        / active_weight
    ).clip(0.0, 100.0).round(1)

    ts["stress_score"] = stress
    return ts[["time", "stress_score", "lmp"]]


def get_all_stress_scores(
    latest_data: dict,
    history_data: dict,
    extended_data: Optional[dict] = None,
) -> list[dict]:
    """
    Compute stress scores for all configured REGIONS.

    latest_data   : {iso: {hub: {lmp, energy, congestion, time}}}
    history_data  : {iso: DataFrame}
    extended_data : {iso: {actual_load, forecast_load, fuel_gen, load_history_30d}}
    """
    scores = []
    for iso, hub in REGIONS:
        iso_latest = latest_data.get(iso, {})
        hub_data = iso_latest.get(hub)
        if hub_data is None or hub_data.get("lmp") is None:
            continue

        current_lmp = hub_data["lmp"]
        history_df  = history_data.get(iso)
        ext = (extended_data or {}).get(iso, {})

        # ERCOT congestion proxy: hub_lmp - HB_BUSAVG_lmp
        congestion_proxy = None
        if iso == "ERCOT":
            busavg = iso_latest.get("HB_BUSAVG", {})
            busavg_lmp = busavg.get("lmp") if busavg else None
            if busavg_lmp is not None:
                congestion_proxy = current_lmp - busavg_lmp

        # DA LMP for DART signal
        try:
            da_lmp = fetch_da_lmp(iso, hub)
        except Exception:
            da_lmp = None

        score = compute_stress_score(
            iso=iso,
            hub=hub,
            current_lmp=current_lmp,
            history_df=history_df,
            congestion_proxy=congestion_proxy,
            actual_load=ext.get("actual_load"),
            forecast_load=ext.get("forecast_load"),
            fuel_gen=ext.get("fuel_gen"),
            load_history_30d=ext.get("load_history_30d"),
            da_lmp=da_lmp,
        )

        score["energy"]     = hub_data.get("energy")
        score["congestion"] = hub_data.get("congestion")
        score["data_time"]  = hub_data.get("time")
        lbl, color = stress_label(score["stress_score"])
        score["stress_label"] = lbl
        score["stress_color"] = color
        scores.append(score)

    return scores


if __name__ == "__main__":
    from data_cache import get_all_regions

    print("Fetching data...")
    latest = get_all_regions()

    print("Fetching 24h history...")
    history = {}
    for iso in ["PJM", "CAISO", "ERCOT", "MISO", "ISONE", "NYISO"]:
        h = fetch_history(iso, hours=24)
        if h is not None:
            history[iso] = h
            print(f"  [{iso}] {len(h)} rows")

    print("Fetching extended signal data (load, fuel, DA)...")
    extended = {}
    for iso in ["CAISO", "ERCOT", "MISO", "ISONE", "NYISO"]:
        extended[iso] = get_extended_data(iso)
        ext = extended[iso]
        print(f"  [{iso}] load={ext['actual_load']} MW  fc={ext['forecast_load']} MW  "
              f"gen={ext['fuel_gen']}")

    print("\nStress Scores:")
    scores_list = get_all_stress_scores(latest, history, extended)
    for s in scores_list:
        lbl, _ = stress_label(s["stress_score"])
        sigs = s["signals"]
        print(f"  {s['iso']:6} / {s['hub']:25} {s['stress_score']:5.1f} [{lbl:8}]  "
              f"LMP=${s['lmp']:7.2f}  "
              f"s1={sigs['lmp_deviation']}  s2={sigs['lmp_velocity']}  "
              f"s3={sigs['load_forecast']}  s4={sigs['reserve_margin']}  s5={sigs['dart']}  "
              f"({s['active_signals']}/5 signals)")
