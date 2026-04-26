"""
Forward stress score computation for the next 12 hours.

Uses Day-Ahead LMP as a proxy for future RT conditions, combined with
the gridstatus load forecast. Three-signal composite:

  Signal 1 — LMP Deviation (50 %): (DA_LMP(t) − history_baseline) / baseline
  Signal 2 — Load Score    (35 %): deviation of forecast load from 30-day median / 0.08
  Signal 3 — DART persist. (15 %): last observed DART stress score held constant

Weights are renormalized if a signal is unavailable (same pattern as grid_stress.py).
"""

from __future__ import annotations

import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Optional

from data_cache import fetch_da_lmp_forecast, fetch_load_forecast_series
from grid_stress import _clamp, stress_label

# Chart line colours per ISO
ISO_LINE_COLORS: dict[str, str] = {
    "PJM":   "#818cf8",   # indigo
    "CAISO": "#fbbf24",   # amber
    "ERCOT": "#f87171",   # red
    "MISO":  "#34d399",   # green
    "ISONE": "#60a5fa",   # blue
    "NYISO": "#c084fc",   # purple
    "SPP":   "#fb923c",   # orange
}

# Primary forecast hub per ISO
ISO_FORECAST_HUB: dict[str, str] = {
    "PJM":   "WESTERN HUB",
    "CAISO": "TH_NP15_GEN-APND",
    "ERCOT": "HB_NORTH",
    "MISO":  "INDIANA.HUB",
    "ISONE": ".H.INTERNAL_HUB",
    "NYISO": "N.Y.C.",
    "SPP":   "SPPNORTH_HUB",
}

_FORECAST_WEIGHTS = {"lmp": 0.50, "load": 0.35, "dart": 0.15}


def _lookup_nearest(
    df: Optional[pd.DataFrame],
    t: datetime,
    val_col: str,
    max_gap_hours: float = 2.0,
) -> Optional[float]:
    """Return the value in val_col from the row in df whose 'time' column is closest to t."""
    if df is None or df.empty or val_col not in df.columns or "time" not in df.columns:
        return None
    times = pd.to_datetime(df["time"], utc=True, errors="coerce")
    diffs = (times - t).abs()
    if diffs.empty:
        return None
    min_diff = diffs.min()
    if min_diff.total_seconds() > max_gap_hours * 3600:
        return None
    idx = diffs.argmin()
    v = df[val_col].iloc[idx]
    return float(v) if pd.notna(v) else None


def compute_iso_forward_stress(
    da_lmp_df: Optional[pd.DataFrame],
    load_fc_df: Optional[pd.DataFrame],
    history_lmps: Optional[pd.Series] = None,
    history_loads: Optional[pd.Series] = None,
    current_dart_score: float = 50.0,
    hours: int = 12,
) -> pd.DataFrame:
    """
    Compute forward stress for a single ISO / hub over the next `hours` hours.
    Returns DataFrame[time, stress_score, lmp, load_forecast].
    """
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    future_hours = [now + timedelta(hours=h) for h in range(1, hours + 1)]

    lmp_baseline = (
        float(history_lmps.mean())
        if history_lmps is not None and len(history_lmps) >= 3
        else None
    )
    load_median = (
        float(history_loads.median())
        if history_loads is not None and len(history_loads) >= 10
        else None
    )

    rows = []
    for t in future_hours:
        da_lmp = _lookup_nearest(da_lmp_df, t, "lmp")
        fc_load = _lookup_nearest(load_fc_df, t, "load")

        # Signal 1: LMP deviation score (0–100)
        lmp_dev_score: Optional[float] = None
        if da_lmp is not None and lmp_baseline is not None and lmp_baseline > 0:
            dev = (da_lmp - lmp_baseline) / lmp_baseline
            lmp_dev_score = _clamp(30.0 + dev * 100.0)

        # Signal 2: load vs median score (0–100)
        load_score: Optional[float] = None
        if fc_load is not None and load_median is not None and load_median > 0:
            load_dev = (fc_load - load_median) / load_median
            load_score = _clamp(load_dev / 0.08, 0.0, 1.0) * 100.0

        # Signal 3: DART persistence (always available)
        dart_score = current_dart_score

        available: dict[str, tuple[float, float]] = {}
        if lmp_dev_score is not None:
            available["lmp"] = (lmp_dev_score, _FORECAST_WEIGHTS["lmp"])
        if load_score is not None:
            available["load"] = (load_score, _FORECAST_WEIGHTS["load"])
        available["dart"] = (dart_score, _FORECAST_WEIGHTS["dart"])

        total_w = sum(w for _, w in available.values())
        composite = (
            sum(v * w / total_w for v, w in available.values())
            if total_w > 0 else 50.0
        )

        rows.append({
            "time": t,
            "stress_score": round(_clamp(composite), 1),
            "lmp": round(da_lmp, 2) if da_lmp is not None else None,
            "load_forecast": round(fc_load, 0) if fc_load is not None else None,
        })

    return pd.DataFrame(rows)


def get_all_forward_stress(
    scores: list[dict],
    history: dict,
    extended: dict,
    hours: int = 12,
) -> dict[str, pd.DataFrame]:
    """
    Compute forward stress for each ISO's primary forecast hub.
    Returns {iso: DataFrame[time, stress_score, lmp, load_forecast]}.
    """
    result: dict[str, pd.DataFrame] = {}

    for iso in ["PJM", "CAISO", "ERCOT", "MISO", "ISONE", "NYISO", "SPP"]:
        hub = ISO_FORECAST_HUB.get(iso)
        if not hub:
            continue

        da_df = fetch_da_lmp_forecast(iso, hub, hours=hours)
        load_fc_df = fetch_load_forecast_series(iso, hours=hours)

        # Historical LMPs for baseline (from 24h history already loaded)
        hist_df = history.get(iso)
        history_lmps: Optional[pd.Series] = None
        if hist_df is not None and not hist_df.empty and "lmp" in hist_df.columns:
            if "location" in hist_df.columns:
                rows_h = hist_df[hist_df["location"].str.upper() == hub.upper()]
            else:
                rows_h = hist_df
            if not rows_h.empty:
                history_lmps = rows_h["lmp"].dropna()

        # 30-day load history for median
        history_loads: Optional[pd.Series] = (extended.get(iso) or {}).get("load_history_30d")

        # DART persistence: current DART stress score for this hub
        current_dart: float = 50.0
        for s in scores:
            if s["iso"] == iso and s["hub"] == hub:
                dart_val = (s.get("signals") or {}).get("dart")
                if dart_val is not None:
                    current_dart = float(dart_val)
                break

        df = compute_iso_forward_stress(
            da_lmp_df=da_df,
            load_fc_df=load_fc_df,
            history_lmps=history_lmps,
            history_loads=history_loads,
            current_dart_score=current_dart,
            hours=hours,
        )
        if not df.empty:
            result[iso] = df

    return result


def find_routing_opportunities(
    forward_stress: dict[str, pd.DataFrame],
    source_iso: str = "PJM",
    threshold_source: float = 50.0,
    threshold_dest: float = 25.0,
) -> list[dict]:
    """
    Find future hours where source stress > threshold_source and the best
    alternative ISO has stress < threshold_dest simultaneously.

    Returns a list of dicts suitable for a DataFrame display.
    """
    source_df = forward_stress.get(source_iso)
    if source_df is None or source_df.empty:
        return []

    opps: list[dict] = []
    for _, row in source_df.iterrows():
        t = pd.Timestamp(row["time"])
        src_stress = float(row["stress_score"])
        if src_stress < threshold_source:
            continue

        # Find best destination at this hour
        best_iso: Optional[str] = None
        best_stress = 100.0
        for iso, df in forward_stress.items():
            if iso == source_iso:
                continue
            dest_row = df[df["time"] == t]
            if dest_row.empty:
                continue
            dest_stress = float(dest_row["stress_score"].iloc[0])
            if dest_stress < best_stress:
                best_stress = dest_stress
                best_iso = iso

        if best_iso is None:
            continue

        # LMPs
        src_lmp = _lookup_nearest(source_df, t, "lmp", max_gap_hours=1.5)
        dest_lmp = _lookup_nearest(forward_stress.get(best_iso), t, "lmp", max_gap_hours=1.5)
        savings = round(src_lmp - dest_lmp, 2) if src_lmp is not None and dest_lmp is not None else None

        lbl_src, _ = stress_label(src_stress)
        lbl_dst, _ = stress_label(best_stress)

        opps.append({
            "time": t,
            "source_iso": source_iso,
            "source_stress": src_stress,
            "source_stress_label": lbl_src,
            "best_dest": best_iso,
            "dest_stress": best_stress,
            "dest_stress_label": lbl_dst,
            "savings_per_mwh": savings,
        })

    return opps


def compute_batch_schedule(
    forward_stress: dict[str, pd.DataFrame],
    workload_mw: float,
    duration_hours: int,
    source_iso: str = "PJM",
) -> tuple[list[dict], float, float]:
    """
    Greedily route each hour of a `duration_hours` batch workload to the
    lowest-stress eligible ISO (stress ≤ STRESS_HARD_LIMIT = 75).

    Returns:
      schedule     — list[dict] with hour, time, iso, stress_score, lmp
      total_savings_usd  — estimated cost savings vs always using source_iso
      carbon_improvement_pct  — renewable fraction improvement (× 100)
    """
    from router import STATIC_RENEWABLE_PCT, STRESS_HARD_LIMIT

    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    future_times = [now + timedelta(hours=h) for h in range(1, duration_hours + 1)]

    src_df = forward_stress.get(source_iso)
    baseline_renewable = STATIC_RENEWABLE_PCT.get(source_iso, 0.4)

    schedule: list[dict] = []
    total_savings = 0.0
    total_renewable_weighted = 0.0

    for i, t in enumerate(future_times):
        candidates: list[tuple[float, float, str]] = []
        for iso, df in forward_stress.items():
            r = df[df["time"] == t]
            if r.empty:
                continue
            stress = float(r["stress_score"].iloc[0])
            lmp_v = r["lmp"].iloc[0]
            lmp = float(lmp_v) if pd.notna(lmp_v) else 999.0
            if stress <= STRESS_HARD_LIMIT:
                candidates.append((stress, lmp, iso))

        if candidates:
            candidates.sort()   # lowest stress first, then LMP
            best_stress, best_lmp, best_iso = candidates[0]
        else:
            # All ISOs stressed — fall back to source
            best_iso = source_iso
            r = src_df[src_df["time"] == t] if src_df is not None else pd.DataFrame()
            best_stress = float(r["stress_score"].iloc[0]) if not r.empty else 50.0
            v = r["lmp"].iloc[0] if not r.empty else None
            best_lmp = float(v) if pd.notna(v) else None  # type: ignore[arg-type]

        # Cost savings vs source ISO at this hour
        src_lmp = _lookup_nearest(src_df, t, "lmp", max_gap_hours=1.5) if src_df is not None else None
        if src_lmp is not None and best_lmp is not None and best_lmp < 900:
            total_savings += (src_lmp - best_lmp) * workload_mw

        total_renewable_weighted += STATIC_RENEWABLE_PCT.get(best_iso, 0.4)

        lbl, _ = stress_label(best_stress)
        schedule.append({
            "hour": i + 1,
            "time": t,
            "iso": best_iso,
            "stress_score": best_stress,
            "stress_label": lbl,
            "lmp": best_lmp if (best_lmp is not None and best_lmp < 900) else None,
        })

    avg_renewable = total_renewable_weighted / duration_hours if duration_hours > 0 else baseline_renewable
    carbon_improvement = (avg_renewable - baseline_renewable) * 100

    return schedule, round(total_savings, 2), round(carbon_improvement, 1)
