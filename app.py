"""GridPulse — Grid-Aware AI Inference Router dashboard."""

import time
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from data_cache import get_all_regions, fetch_history, get_extended_data, ISO_CONFIG
from grid_stress import get_all_stress_scores, stress_label, historical_stress_series
from router import compute_routing, STATIC_RENEWABLE_PCT
from stress_forecast import (
    get_all_forward_stress, find_routing_opportunities,
    compute_batch_schedule, ISO_LINE_COLORS, ISO_FORECAST_HUB,
)
from ercot_2021_scenario import (
    ERCOT_2021, CAISO_2020, ALL_SCENARIOS,
    build_scenario_animation, build_snapshot_map, build_gauge_figure,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GridPulse — Real-Time Grid Stress Monitor",
    page_icon="⚡",
    layout="wide",
)

# ── Dark mode CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Force dark background */
  .stApp { background-color: #0e1117; color: #fafafa; }
  section[data-testid="stSidebar"] { background-color: #161b22; }

  /* Reduce top padding */
  .block-container { padding-top: 0.75rem !important; padding-bottom: 0.5rem !important; }
  header[data-testid="stHeader"] { height: 0; }
  div[data-testid="stToolbar"] { display: none; }

  /* Metric card styles */
  .stress-card {
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 10px;
    border-left: 4px solid;
  }
  .stress-card h4 { margin: 0 0 4px 0; font-size: 0.75rem; opacity: 0.7; letter-spacing: 0.06em; text-transform: uppercase; }
  .stress-card .score { font-size: 2rem; font-weight: 700; line-height: 1; }
  .stress-card .lmp { font-size: 0.85rem; opacity: 0.8; margin-top: 4px; }

  .card-low      { background: rgba(34,197,94,0.12);  border-color: #22c55e; }
  .card-moderate { background: rgba(234,179,8,0.12);  border-color: #eab308; }
  .card-high     { background: rgba(249,115,22,0.12); border-color: #f97316; }
  .card-critical { background: rgba(239,68,68,0.15);  border-color: #ef4444; }

  .badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    margin-top: 6px;
  }
  .badge-low      { background: #22c55e22; color: #22c55e; border: 1px solid #22c55e44; }
  .badge-moderate { background: #eab30822; color: #eab308; border: 1px solid #eab30844; }
  .badge-high     { background: #f9731622; color: #f97316; border: 1px solid #f9731644; }
  .badge-critical { background: #ef444422; color: #ef4444; border: 1px solid #ef444444; }

  /* Routing recommendation card */
  .rec-card {
    background: rgba(99,102,241,0.12);
    border: 1px solid rgba(99,102,241,0.4);
    border-radius: 10px;
    padding: 20px;
    margin-top: 10px;
  }
  .rec-card h2 { color: #818cf8; margin: 0 0 8px 0; }

  /* Section headers */
  .section-header {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #6b7280;
    margin: 20px 0 8px 0;
    border-bottom: 1px solid #1f2937;
    padding-bottom: 4px;
  }

  /* Router panel */
  .router-panel {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 16px;
  }

  /* Animated winner card */
  @keyframes winner-pulse {
    0%   { box-shadow: 0 0 8px rgba(99,102,241,0.25); }
    50%  { box-shadow: 0 0 28px rgba(99,102,241,0.55), 0 0 50px rgba(99,102,241,0.15); }
    100% { box-shadow: 0 0 8px rgba(99,102,241,0.25); }
  }
  .winner-card {
    background: linear-gradient(135deg, rgba(99,102,241,0.14) 0%, rgba(139,92,246,0.10) 100%);
    border: 1px solid rgba(99,102,241,0.45);
    border-radius: 12px;
    padding: 22px;
    animation: winner-pulse 2.5s ease-in-out infinite;
  }
  .winner-card .winner-title {
    font-size: 0.7rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #818cf8;
    margin-bottom: 4px;
  }
  .winner-card .winner-region {
    font-size: 1.6rem;
    font-weight: 800;
    color: #e0e7ff;
    margin: 0 0 4px 0;
    line-height: 1.2;
  }
  .winner-card .winner-hub {
    font-size: 0.9rem;
    color: #a5b4fc;
    margin-bottom: 16px;
  }
  .winner-card .winner-stats {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin-bottom: 16px;
  }
  .winner-card .stat-box {
    background: rgba(255,255,255,0.05);
    border-radius: 6px;
    padding: 8px 12px;
  }
  .winner-card .stat-label { font-size: 0.65rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.08em; }
  .winner-card .stat-value { font-size: 1.1rem; font-weight: 700; color: #f9fafb; }
  .winner-card .rationale {
    font-size: 0.82rem;
    color: #9ca3af;
    line-height: 1.6;
    border-top: 1px solid rgba(255,255,255,0.08);
    padding-top: 12px;
    margin-top: 4px;
  }

  /* No-route error */
  .no-route {
    background: rgba(239,68,68,0.1);
    border: 1px solid rgba(239,68,68,0.35);
    border-radius: 10px;
    padding: 18px;
    text-align: center;
    color: #fca5a5;
  }

  /* Weight budget bar */
  .budget-bar-outer {
    height: 6px;
    background: #1f2937;
    border-radius: 3px;
    margin: 6px 0 14px 0;
    overflow: hidden;
  }
  .budget-bar-inner {
    height: 100%;
    border-radius: 3px;
    transition: width 0.2s ease;
  }

  /* Comparison table styling */
  .rank-badge {
    display: inline-block;
    width: 22px; height: 22px;
    border-radius: 50%;
    text-align: center; line-height: 22px;
    font-size: 0.75rem; font-weight: 700;
  }
  .rank-1 { background: #818cf8; color: #0e1117; }
  .rank-other { background: #1f2937; color: #6b7280; }
  .rank-filtered { background: #1f2937; color: #374151; }
</style>
""", unsafe_allow_html=True)


# ── Constants ─────────────────────────────────────────────────────────────────

# ISO → states mapping for choropleth
ISO_STATES = {
    "PJM":   ["PA", "OH", "WV", "MD", "DE", "NJ", "VA", "KY", "MI", "IN", "IL"],
    "CAISO": ["CA"],
    "ERCOT": ["TX"],
    "MISO":  ["MN", "IA", "MO", "WI", "ND", "SD", "NE", "AR", "LA", "MS", "MT"],
    "ISONE": ["ME", "NH", "VT", "MA", "RI", "CT"],
    "NYISO": ["NY"],
    "SPP":   ["KS", "OK"],
}

# ISO marker centroids [lat, lon]
ISO_CENTROIDS = {
    "PJM":   (39.5, -77.5),
    "CAISO": (37.4, -121.9),
    "ERCOT": (32.8, -97.3),
    "MISO":  (41.9, -87.7),
    "ISONE": (42.4, -71.1),
    "NYISO": (40.7, -74.0),
    "SPP":   (37.2, -97.5),
}

# Estimated DC capacity MW — sets marker size on map
ISO_DC_CAPACITY_MW = {
    "PJM":   2800,
    "CAISO": 1200,
    "ERCOT":  900,
    "MISO":   700,
    "NYISO":  500,
    "ISONE":  300,
    "SPP":    200,
}

# ISO full display names
ISO_DISPLAY_NAMES = {
    "PJM":   "PJM — N.Virginia / Mid-Atlantic",
    "CAISO": "CAISO — Silicon Valley / LA",
    "ERCOT": "ERCOT — Dallas / Houston",
    "MISO":  "MISO — Chicago / Indianapolis",
    "ISONE": "ISONE — Boston / New England",
    "NYISO": "NYISO — New York City",
    "SPP":   "SPP — Oklahoma City / Kansas City",
}

# Per-ISO chart colors for history
_ISO_HIST_COLORS = {
    "PJM":   "#818cf8",
    "CAISO": "#fbbf24",
    "ERCOT": "#34d399",
    "MISO":  "#f97316",
    "ISONE": "#60a5fa",
    "NYISO": "#c084fc",
    "SPP":   "#fb923c",
}

# Short display names for hubs
_HUB_SHORT = {
    "WESTERN HUB":       "Western Hub",
    "DOMINION HUB":      "Dominion Hub",
    "TH_NP15_GEN-APND": "NP15",
    "TH_SP15_GEN-APND": "SP15",
    "HB_NORTH":          "HB North",
    "HB_HOUSTON":        "HB Houston",
    "HB_SOUTH":          "HB South",
    "INDIANA.HUB":       "Indiana Hub",
    "ILLINOIS.HUB":      "Illinois Hub",
    ".H.INTERNAL_HUB":   "Hub",
    "N.Y.C.":            "NYC",
    "LONGIL":            "Long Island",
    "SPPNORTH_HUB":      "North Hub",
    "SPPSOUTH_HUB":      "South Hub",
}

# ISO local timezones for "Local Time" column
ISO_TIMEZONES = {
    "PJM":   ZoneInfo("America/New_York"),
    "CAISO": ZoneInfo("America/Los_Angeles"),
    "ERCOT": ZoneInfo("America/Chicago"),
    "MISO":  ZoneInfo("America/Chicago"),
    "ISONE": ZoneInfo("America/New_York"),
    "NYISO": ZoneInfo("America/New_York"),
    "SPP":   ZoneInfo("America/Chicago"),
}

# Step colorscale: green → yellow → orange → red
_STRESS_COLORSCALE = [
    [0.000, "#22c55e"],
    [0.249, "#22c55e"],
    [0.250, "#eab308"],
    [0.499, "#eab308"],
    [0.500, "#f97316"],
    [0.749, "#f97316"],
    [0.750, "#ef4444"],
    [1.000, "#ef4444"],
]


def _stress_to_hex(score: float) -> str:
    if score < 25:   return "#22c55e"
    elif score < 50: return "#eab308"
    elif score < 75: return "#f97316"
    else:            return "#ef4444"


def score_to_card_class(label: str) -> str:
    return {"LOW": "low", "MODERATE": "moderate", "HIGH": "high", "CRITICAL": "critical"}.get(label, "moderate")


# ── Data loading (cached 5 min) ───────────────────────────────────────────────

_CACHE_TTL = 300  # seconds

def _data_is_fresh() -> bool:
    return (
        "grid_data" in st.session_state
        and (time.time() - st.session_state.get("_loaded_at", 0)) < _CACHE_TTL
    )


# ── Map builder ───────────────────────────────────────────────────────────────

def build_us_map(scores: list, winner=None, fallback_isos: set = None) -> go.Figure:
    fallback_isos = fallback_isos or set()

    iso_scores: dict = {}
    for s in scores:
        iso = s["iso"]
        if iso not in iso_scores or s["stress_score"] > iso_scores[iso]["stress_score"]:
            iso_scores[iso] = s

    state_codes, state_z, state_hover = [], [], []
    for iso, states in ISO_STATES.items():
        data = iso_scores.get(iso)
        score = data["stress_score"] if data else 0
        lmp   = data["lmp"] if data else 0
        lbl   = data["stress_label"] if data else "—"
        for st_code in states:
            state_codes.append(st_code)
            state_z.append(score)
            state_hover.append(
                f"<b>{ISO_DISPLAY_NAMES.get(iso, iso)}</b><br>"
                f"Stress: {score:.0f}/100  [{lbl}]<br>"
                f"LMP: ${lmp:.2f}/MWh"
            )

    fig = go.Figure()
    fig.add_trace(go.Choropleth(
        locations=state_codes,
        z=state_z,
        locationmode="USA-states",
        colorscale=_STRESS_COLORSCALE,
        zmin=0, zmax=100,
        text=state_hover,
        hovertemplate="%{text}<extra></extra>",
        marker_line_color="#0e1117",
        marker_line_width=1.5,
        colorbar=dict(
            title=dict(text="Stress", font=dict(color="#9ca3af", size=11)),
            tickvals=[12, 37, 62, 87],
            ticktext=["LOW", "MOD", "HIGH", "CRIT"],
            tickfont=dict(color="#9ca3af", size=10),
            thickness=10, len=0.5, x=1.01,
            bgcolor="rgba(0,0,0,0)", outlinewidth=0,
        ),
    ))

    import math
    _min_mw = min(ISO_DC_CAPACITY_MW.values())
    _max_mw = max(ISO_DC_CAPACITY_MW.values())

    def _marker_size(mw: float) -> float:
        t = (math.sqrt(mw) - math.sqrt(_min_mw)) / (math.sqrt(_max_mw) - math.sqrt(_min_mw))
        return 14 + t * 22

    lats, lons, colors, sizes, hovers, isos_ordered = [], [], [], [], [], []
    for iso, (lat, lon) in ISO_CENTROIDS.items():
        data = iso_scores.get(iso)
        if data is None:
            continue
        score = data["stress_score"]
        lmp   = data["lmp"]
        lbl   = data["stress_label"]
        active = data.get("active_signals", "?")
        mw    = ISO_DC_CAPACITY_MW.get(iso, 500)
        fb    = " ⚠ fallback" if iso in fallback_isos else ""

        lats.append(lat)
        lons.append(lon)
        colors.append(_stress_to_hex(score))
        sizes.append(_marker_size(mw))
        hovers.append(
            f"<b>{ISO_DISPLAY_NAMES.get(iso, iso)}</b>{fb}<br>"
            f"Stress: <b>{score:.0f}/100</b>  [{lbl}]<br>"
            f"LMP: ${lmp:.2f}/MWh<br>"
            f"DC capacity: ~{mw:,} MW<br>"
            f"Signals live: {active}/5"
        )
        isos_ordered.append(iso)

    fig.add_trace(go.Scattergeo(
        lat=lats, lon=lons,
        mode="markers+text",
        marker=dict(
            size=sizes,
            color=colors,
            symbol="circle",
            line=dict(width=2, color="rgba(255,255,255,0.7)"),
            opacity=0.92,
        ),
        text=[f"<b>{iso}</b>" for iso in isos_ordered],
        textfont=dict(size=9, color="white"),
        textposition="top center",
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hovers,
        showlegend=False,
    ))

    fb_lats, fb_lons, fb_hover = [], [], []
    for iso in fallback_isos:
        if iso in ISO_CENTROIDS and iso in iso_scores:
            lat, lon = ISO_CENTROIDS[iso]
            fb_lats.append(lat - 1.5)
            fb_lons.append(lon)
            fb_hover.append(f"<b>{iso}</b> — using REST API fallback<br>gridstatus library unavailable")

    if fb_lats:
        fig.add_trace(go.Scattergeo(
            lat=fb_lats, lon=fb_lons,
            mode="markers",
            marker=dict(size=10, color="#f97316", symbol="triangle-up",
                        line=dict(width=1, color="#ffffff")),
            hovertemplate="%{customdata}<extra></extra>",
            customdata=fb_hover,
            showlegend=False,
        ))

    if winner and winner.iso in ISO_CENTROIDS and winner.iso in iso_scores:
        lat, lon = ISO_CENTROIDS[winner.iso]
        fig.add_trace(go.Scattergeo(
            lat=[lat], lon=[lon],
            mode="markers",
            marker=dict(
                size=_marker_size(ISO_DC_CAPACITY_MW.get(winner.iso, 500)) + 14,
                color="rgba(99,102,241,0.0)",
                symbol="circle",
                line=dict(width=3, color="#818cf8"),
            ),
            hovertemplate=f"<b>Recommended: {winner.iso}</b><extra></extra>",
            showlegend=False,
        ))

    fig.update_layout(
        geo=dict(
            scope="usa",
            projection_type="albers usa",
            showland=True, landcolor="#1a1f2e",
            showlakes=False,
            showocean=True, oceancolor="#0e1117",
            showcoastlines=True, coastlinecolor="#374151",
            showsubunits=True, subunitcolor="#374151",
            bgcolor="#0e1117",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=0, b=0, l=0, r=0),
        height=390,
        showlegend=False,
    )
    return fig


# ── Load data ─────────────────────────────────────────────────────────────────

if not _data_is_fresh():
    st.markdown("""
    <style>
    @keyframes pulse-ring {
        0%   { transform: scale(0.8); opacity: 0.8; }
        50%  { transform: scale(1.1); opacity: 0.4; }
        100% { transform: scale(0.8); opacity: 0.8; }
    }
    .gp-loading-wrap {
        display: flex; flex-direction: column; align-items: center;
        justify-content: center; padding: 3rem 0 1rem;
    }
    .gp-ring {
        width: 56px; height: 56px; border-radius: 50%;
        border: 4px solid #818cf8; border-top-color: transparent;
        animation: spin 0.9s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    .gp-loading-title {
        margin-top: 1.2rem; font-size: 1.3rem; font-weight: 700;
        color: #e5e7eb; letter-spacing: 0.03em;
    }
    .gp-loading-sub {
        margin-top: 0.4rem; font-size: 0.85rem; color: #6b7280;
    }
    </style>
    <div class="gp-loading-wrap">
      <div class="gp-ring"></div>
      <div class="gp-loading-title">⚡ GridPulse</div>
      <div class="gp-loading-sub">Fetching live grid data from 7 ISOs…</div>
    </div>
    """, unsafe_allow_html=True)

    _isos = ["PJM", "CAISO", "ERCOT", "MISO", "ISONE", "NYISO", "SPP"]
    _failed_isos: list[str] = []
    with st.status("Loading live grid data…", expanded=True) as _status:
        st.write("📡 Pulling real-time LMP prices…")
        try:
            _latest = get_all_regions()
        except Exception:
            _latest = {}
        _missing_rt = [iso for iso in _isos if iso not in _latest or not _latest[iso]]
        if _missing_rt:
            _failed_isos.extend(_missing_rt)

        _history: dict = {}
        _extended: dict = {}
        for _iso in _isos:
            st.write(f"📊 {_iso} — history & extended signals…")
            try:
                _h = fetch_history(_iso, hours=24)
                if _h is not None:
                    _history[_iso] = _h
            except Exception:
                pass
            try:
                _extended[_iso] = get_extended_data(_iso)
            except Exception:
                _extended[_iso] = {}

        st.write("🧮 Computing stress scores…")
        _scores = []
        try:
            _scores = get_all_stress_scores(_latest, _history, _extended)
        except Exception as e:
            st.error(f"Error computing stress scores: {e}")

        st.write("📈 Building 12-hour forward forecast…")
        _fwd = {}
        try:
            _fwd = get_all_forward_stress(_scores, _history, _extended, hours=12)
        except Exception as e:
            st.error(f"Error building forecast: {e}")

        _fetched_at = datetime.now(timezone.utc)
        st.session_state["grid_data"] = (_scores, _history, _fetched_at, _extended, _fwd)
        st.session_state["_loaded_at"] = time.time()
        st.session_state["_failed_isos"] = _failed_isos
        _status.update(label="✅ Grid data loaded", state="complete", expanded=False)
    st.rerun()

scores, history, fetched_at, extended, forward_stress = st.session_state["grid_data"]

if not scores:
    st.error("No stress scores available — GridStatus API may be rate-limited. Try refreshing.")
    st.stop()

_failed_isos = st.session_state.get("_failed_isos", [])
if _failed_isos:
    _ttl_remaining = max(0, int(_CACHE_TTL - (time.time() - st.session_state.get("_loaded_at", time.time()))))
    for _fi in _failed_isos:
        st.warning(
            f"⚠️ {_fi} data temporarily unavailable. "
            f"Showing last cached values. Retry in {_ttl_remaining}s."
        )


# ── Info banner ───────────────────────────────────────────────────────────────

st.markdown("""
<div style='background:#0D3D34; padding:8px 16px; border-radius:6px;
            border-left:3px solid #00BFA5; margin-bottom:12px;'>
⚡ <b>GridPulse</b> &nbsp;·&nbsp; Live data &nbsp;·&nbsp;
Updates every 5 min &nbsp;·&nbsp; 7 ISOs monitored &nbsp;·&nbsp;
<span style='color:#9CA3AF'>PJM · CAISO · ERCOT · MISO · NYISO · ISONE · SPP</span>
</div>
""", unsafe_allow_html=True)

# ── Page header ───────────────────────────────────────────────────────────────

_hdr_left, _hdr_right = st.columns([6, 2])
with _hdr_left:
    st.markdown("# ⚡ GridPulse")
    st.markdown("**Real-time grid stress monitoring & AI inference routing**")
with _hdr_right:
    st.markdown(
        f"<div style='text-align:right;padding-top:10px;'>"
        f"<div style='font-size:0.72rem;color:#6b7280;margin-bottom:6px;'>"
        f"Data pulled at {fetched_at.strftime('%H:%M UTC')}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    if st.button("🔄 Refresh", key="btn_refresh"):
        st.session_state.pop("grid_data", None)
        st.session_state.pop("_loaded_at", None)
        st.rerun()

st.markdown("---")

tab_monitor, tab_forecast, tab_crisis = st.tabs([
    "⚡ Live Monitor", "📈 Stress Forecast", "🔴 Crisis Replay",
])

with tab_monitor:
    # Default routing for map highlight
    default_weights = {"stress": 0.4, "price": 0.3, "renewable": 0.2, "forward": 0.1}
    default_result = compute_routing(scores, 50, 80, default_weights)
    default_winner = default_result["winner"]

    # ── US Map ────────────────────────────────────────────────────────────────

    col_map, col_bar = st.columns([3, 2])

    with col_map:
        st.markdown("### Grid Stress Map")
        st.caption("ISO footprint states colored by peak hub stress score")
        fallback_isos = {
            iso for iso, ext in extended.items()
            if iso != "PJM" and ext.get("actual_load") is None
        }
        fig_map = build_us_map(scores, default_winner, fallback_isos=fallback_isos)
        if fallback_isos:
            st.caption(f"⚠ {', '.join(sorted(fallback_isos))}: REST-only data (gridstatus lib unavailable)")
        st.plotly_chart(fig_map, use_container_width=True)

    with col_bar:
        # ── Stress by Hub Bar Chart ───────────────────────────────────────────
        st.markdown("### Stress by Hub")
        st.caption(
            "Composite of LMP Deviation, LMP Velocity, Load vs Forecast Deviation, "
            "Reserve Margin Score, DART Score"
        )

        # Sort lowest → highest stress (left to right)
        sorted_scores = sorted(scores, key=lambda s: s["stress_score"])
        region_labels = [
            f"{s['iso']}·{_HUB_SHORT.get(s['hub'], s['hub'].split()[0])}"
            for s in sorted_scores
        ]
        stress_values = [s["stress_score"] for s in sorted_scores]
        bar_colors    = [s["stress_color"] for s in sorted_scores]

        fig_bar = go.Figure(go.Bar(
            x=region_labels,
            y=stress_values,
            marker_color=bar_colors,
            marker_line_width=0,
            text=[f"{v:.0f}" for v in stress_values],
            textposition="outside",
            textfont=dict(color="white", size=13),
        ))
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e5e7eb"),
            yaxis=dict(range=[0, 115], gridcolor="#1f2937", title="Stress Score (0–100)"),
            xaxis=dict(gridcolor="#1f2937"),
            margin=dict(t=20, b=10, l=10, r=10),
            height=380,
            showlegend=False,
        )
        fig_bar.add_hline(y=75, line_dash="dot", line_color="#ef4444",
                          annotation_text="CRITICAL", annotation_font_color="#ef4444")
        fig_bar.add_hline(y=50, line_dash="dot", line_color="#f97316",
                          annotation_text="HIGH", annotation_font_color="#f97316")
        fig_bar.add_hline(y=25, line_dash="dot", line_color="#eab308",
                          annotation_text="MODERATE", annotation_font_color="#eab308")
        st.plotly_chart(fig_bar, use_container_width=True)


    # ── 24-Hour Grid Stress History ───────────────────────────────────────────

    st.markdown("### 24-Hour Grid Stress History")

    # Build label list for the selectbox (only hubs that have history data)
    _hist_labels: list[str] = []
    _hist_keys: list[tuple[str, str]] = []
    for s in scores:
        _iso_h, _hub_h = s["iso"], s["hub"]
        _hdf = history.get(_iso_h)
        if _hdf is None:
            continue
        _ts_chk = historical_stress_series(_hdf, _iso_h, _hub_h)
        if _ts_chk.empty:
            continue
        _hub_short_h = _HUB_SHORT.get(_hub_h, _hub_h)
        _hist_labels.append(f"{_iso_h} · {_hub_short_h}")
        _hist_keys.append((_iso_h, _hub_h))

    _default_hl_idx = 0
    for _i, (_ki, _kh) in enumerate(_hist_keys):
        if (_ki, _kh) == ("PJM", "WESTERN HUB"):
            _default_hl_idx = _i
            break

    if _hist_labels:
        _hl_label = st.selectbox(
            "Highlight hub",
            options=_hist_labels,
            index=_default_hl_idx,
            key="hist_highlight_hub",
            label_visibility="collapsed",
        )
        _hl_idx = _hist_labels.index(_hl_label) if _hl_label in _hist_labels else _default_hl_idx
        HIGHLIGHT_KEY = _hist_keys[_hl_idx]
    else:
        HIGHLIGHT_KEY = ("PJM", "WESTERN HUB")

    st.caption("Stress score per hub over the last 24 hours  ·  select a hub above to highlight it")

    fig_hist = go.Figure()
    any_hist_data = False

    for s in scores:
        iso = s["iso"]
        hub = s["hub"]
        is_hl = (iso, hub) == HIGHLIGHT_KEY

        color_hex = _ISO_HIST_COLORS.get(iso, "#9ca3af")
        r_c = int(color_hex[1:3], 16)
        g_c = int(color_hex[3:5], 16)
        b_c = int(color_hex[5:7], 16)

        hist_df = history.get(iso)
        if hist_df is None:
            continue
        ts_series = historical_stress_series(hist_df, iso, hub)
        if ts_series.empty:
            continue

        any_hist_data = True
        hub_short = _HUB_SHORT.get(hub, hub)
        label_name = f"{iso} · {hub_short}"

        if is_hl:
            line_color  = color_hex
            fill_mode   = "tozeroy"
            fill_color  = f"rgba({r_c},{g_c},{b_c},0.07)"
        else:
            line_color  = f"rgba({r_c},{g_c},{b_c},0.25)"
            fill_mode   = None
            fill_color  = None

        fig_hist.add_trace(go.Scatter(
            x=ts_series["time"],
            y=ts_series["stress_score"],
            mode="lines",
            name=label_name,
            line=dict(color=line_color, width=2),
            fill=fill_mode,
            fillcolor=fill_color,
            hovertemplate=(
                f"<b>{label_name}</b><br>"
                "Time: %{x|%H:%M UTC}<br>"
                "Stress: %{y:.1f}/100<br>"
                "<extra></extra>"
            ),
        ))

    fig_hist.add_hline(
        y=75, line_dash="dot", line_color="#ef4444", line_width=1.5,
        annotation_text="CRITICAL (75)", annotation_position="top right",
        annotation_font_color="#ef4444",
    )
    fig_hist.add_hrect(
        y0=75, y1=100, fillcolor="rgba(239,68,68,0.06)", layer="below", line_width=0,
    )
    fig_hist.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5e7eb", size=12),
        xaxis=dict(gridcolor="#1f2937", title="Time (UTC)", tickformat="%H:%M", showgrid=True),
        yaxis=dict(gridcolor="#1f2937", title="Stress Score", range=[0, 105],
                   showgrid=True, zeroline=False),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            bgcolor="rgba(0,0,0,0)", font=dict(size=11),
        ),
        margin=dict(t=40, b=40, l=50, r=20),
        height=340,
        hovermode="x unified",
    )

    if any_hist_data:
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("Historical data not yet available. Refresh in a moment.")


    # ── LMP Snapshot table ────────────────────────────────────────────────────

    st.markdown("### Current LMP Snapshot")
    st.caption(f"Data extracted at {fetched_at.strftime('%H:%M UTC')}")

    rows = []
    for s in scores:
        label, color = stress_label(s["stress_score"])

        # Convert data_time to ISO local time
        local_time = "—"
        data_time = s.get("data_time")
        if data_time is not None:
            try:
                ts = pd.Timestamp(data_time)
                if ts.tzinfo is None:
                    ts = ts.tz_localize("UTC")
                tz = ISO_TIMEZONES.get(s["iso"])
                if tz:
                    local_ts = ts.tz_convert(tz)
                    local_time = local_ts.strftime("%H:%M %Z")
            except Exception:
                pass

        rows.append({
            "ISO": s["iso"],
            "Hub": s["hub"],
            "LMP ($/MWh)": s["lmp"],
            "Energy": s.get("energy") or "—",
            "Congestion": s.get("congestion") or "—",
            "Local Time": local_time,
            "Stress Score": s["stress_score"],
            "Status": label,
        })

    df_table = pd.DataFrame(rows)
    st.dataframe(
        df_table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "LMP ($/MWh)": st.column_config.NumberColumn(format="$%.2f"),
            "Stress Score": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f"),
        },
    )


    # ── Inference Router Panel ────────────────────────────────────────────────

    st.markdown("---")
    st.markdown("## 🔀 Inference Router")
    st.caption("Configure workload parameters and weights, then click **Route Now** for an optimal placement recommendation.")

    if "routing_result" not in st.session_state:
        st.session_state.routing_result = None

    router_left, router_right = st.columns([2, 3], gap="large")

    with router_left:
        st.markdown('<div class="section-header">Workload Parameters</div>', unsafe_allow_html=True)

        source_iso = st.selectbox(
            "Workload Location (source ISO)",
            options=["PJM", "CAISO", "ERCOT", "MISO", "NYISO", "ISONE", "SPP"],
            index=0,
            help="ISO region where the inference workload currently runs",
        )
        workload_mw = st.slider(
            "Workload Size (MW)", min_value=1, max_value=500, value=50, step=1,
        )
        max_latency_ms = st.slider(
            "Max Latency SLA (ms)", min_value=10, max_value=200, value=80, step=5,
        )

        st.markdown('<div class="section-header">Routing Weights</div>', unsafe_allow_html=True)
        st.caption("Adjust priorities — auto-normalized to 100%.")

        raw_stress    = st.slider("⚡ Grid Stress Benefit", 0, 100, 40, 5)
        raw_price     = st.slider("💰 LMP Price Score",     0, 100, 30, 5)
        raw_renewable = st.slider("🌱 Renewable Mix",        0, 100, 20, 5)
        raw_forward   = st.slider("📈 DART Bonus",           0, 100, 10, 5)

        raw_total = raw_stress + raw_price + raw_renewable + raw_forward or 1
        eff_stress    = raw_stress    / raw_total
        eff_price     = raw_price     / raw_total
        eff_renewable = raw_renewable / raw_total
        eff_forward   = raw_forward   / raw_total

        bar_pct   = min(100, int(raw_total))
        bar_color = "#22c55e" if raw_total == 100 else "#eab308" if raw_total < 100 else "#f97316"
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;font-size:0.72rem;color:#6b7280;margin-top:8px;">
          <span>Effective weights</span>
          <span style="color:{bar_color}">
            ⚡{eff_stress:.0%} · 💰{eff_price:.0%} · 🌱{eff_renewable:.0%} · 📈{eff_forward:.0%}
          </span>
        </div>
        <div class="budget-bar-outer">
          <div class="budget-bar-inner" style="width:{bar_pct}%;background:{bar_color};"></div>
        </div>
        """, unsafe_allow_html=True)

        weights = {
            "stress": eff_stress, "price": eff_price,
            "renewable": eff_renewable, "forward": eff_forward,
        }

        if st.button("⚡ Route Now", use_container_width=True, type="primary"):
            st.session_state.routing_result = compute_routing(
                scores, workload_mw, max_latency_ms, weights,
                source_iso=source_iso, extended_data=extended,
            )

        if st.session_state.routing_result is None:
            st.session_state.routing_result = compute_routing(
                scores, workload_mw, max_latency_ms, weights,
                source_iso=source_iso, extended_data=extended,
            )

    result = st.session_state.routing_result

    with router_right:
        st.markdown('<div class="section-header">Recommendation</div>', unsafe_allow_html=True)

        n_eligible = result["n_eligible"]
        n_total    = result["n_total"]

        if result["winner"] is None:
            st.markdown("""
            <div class="no-route">
              <div style="font-size:2rem;margin-bottom:8px;">⛔</div>
              <div style="font-size:1rem;font-weight:700;margin-bottom:6px;">No eligible regions</div>
              <div style="font-size:0.82rem;">All regions exceed stress threshold or latency SLA.<br>
              Try increasing the max latency SLA or wait for grid conditions to improve.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            winner = result["winner"]
            cost_delta = result["cost_delta_per_mwh"]
            cost_color = "#6ee7b7" if cost_delta <= 0 else "#fca5a5"
            cost_sign  = "−" if cost_delta < 0 else "+"

            _dart_badges = {
                "improving": '<span style="color:#22c55e">▼ improving</span>',
                "stable":    '<span style="color:#eab308">→ stable</span>',
                "worsening": '<span style="color:#ef4444">▲ worsening</span>',
                "unknown":   '<span style="color:#6b7280">? unknown</span>',
            }
            dart_badge = _dart_badges.get(result["dart_signal"], "—")

            runner_note = ""
            if n_eligible > 1:
                runner = result["all_candidates"][1]
                gap = winner.routing_score - runner.routing_score
                runner_note = f"  ·  {gap:.3f} pts ahead of {runner.iso}/{runner.hub.split()[0]}"

            st.markdown(f"""
            <div class="winner-card">
              <div class="winner-title">✅ Recommended Region · {n_eligible}/{n_total} eligible</div>
              <div class="winner-region">{winner.iso} — {winner.hub}</div>
              <div class="winner-hub">{winner.dc_location}{runner_note}</div>
              <div class="winner-stats">
                <div class="stat-box">
                  <div class="stat-label">Routing Score</div>
                  <div class="stat-value">{winner.routing_score:.3f}</div>
                </div>
                <div class="stat-box">
                  <div class="stat-label">Grid Stress</div>
                  <div class="stat-value" style="color:{winner.stress_color}">{winner.stress_score:.0f}/100 {winner.stress_label}</div>
                </div>
                <div class="stat-box">
                  <div class="stat-label">Stress Avoided</div>
                  <div class="stat-value" style="color:#6ee7b7">~{result['stress_delta_mw']:.1f} MW</div>
                </div>
                <div class="stat-box">
                  <div class="stat-label">LMP vs Eligible Avg</div>
                  <div class="stat-value">${winner.lmp:.2f} <span style="color:{cost_color};font-size:0.75rem">{cost_sign}${abs(cost_delta):.2f}/MWh</span></div>
                </div>
                <div class="stat-box">
                  <div class="stat-label">Latency · Renewable</div>
                  <div class="stat-value">~{winner.latency_ms}ms · {winner.renewable_fraction:.0%} 🌱</div>
                </div>
                <div class="stat-box">
                  <div class="stat-label">DART Signal</div>
                  <div class="stat-value" style="font-size:0.9rem">{dart_badge}</div>
                </div>
              </div>
              <div class="rationale"><em>{winner.rationale}</em></div>
            </div>
            """, unsafe_allow_html=True)

    # ── Why Not Table ─────────────────────────────────────────────────────────

    _all_cands = result["all_candidates"] if result else []
    _excluded  = result["excluded"]       if result else []
    _n_runners = max(0, len(_all_cands) - 1)

    with st.expander(f"Why not other regions?  ({_n_runners} runner-up{'s' if _n_runners != 1 else ''} · {len(_excluded)} excluded)", expanded=False):
        _rows = []
        for rank_i, r in enumerate(_all_cands):
            _rows.append({
                "Rank": rank_i + 1,
                "ISO": r.iso, "Hub": r.hub, "DC Location": r.dc_location,
                "Routing Score": round(r.routing_score, 3),
                "Stress": r.stress_score, "LMP $/MWh": r.lmp,
                "Renewable": f"{r.renewable_fraction:.0%}",
                "Latency ms": r.latency_ms,
                "⚡": round(r.stress_component, 3),
                "💰": round(r.price_component, 3),
                "🌱": round(r.renewable_component, 3),
                "DART": r.dart_signal, "Status": r.stress_label,
            })
        for r in _excluded:
            _rows.append({
                "Rank": "—",
                "ISO": r.iso, "Hub": r.hub, "DC Location": r.dc_location,
                "Routing Score": None,
                "Stress": r.stress_score, "LMP $/MWh": r.lmp,
                "Renewable": f"{r.renewable_fraction:.0%}",
                "Latency ms": r.latency_ms,
                "⚡": round(r.stress_component, 3),
                "💰": round(r.price_component, 3),
                "🌱": round(r.renewable_component, 3),
                "DART": r.dart_signal,
                "Status": f"FILTERED ({r.filter_reason})",
            })
        if _rows:
            df_cmp = pd.DataFrame(_rows)
            st.dataframe(
                df_cmp, use_container_width=True, hide_index=True,
                column_config={
                    "Routing Score": st.column_config.ProgressColumn(
                        min_value=0, max_value=1, format="%.3f"),
                    "Stress": st.column_config.ProgressColumn(
                        min_value=0, max_value=100, format="%.1f"),
                    "LMP $/MWh": st.column_config.NumberColumn(format="$%.2f"),
                    "⚡": st.column_config.NumberColumn(format="%.3f"),
                    "💰": st.column_config.NumberColumn(format="%.3f"),
                    "🌱": st.column_config.NumberColumn(format="%.3f"),
                },
            )


    # ── Signal Breakdown by Region ────────────────────────────────────────────

    st.markdown("---")
    st.markdown('<div class="section-header">Signal Breakdown by Region</div>', unsafe_allow_html=True)

    _SIG_LABELS = {
        "lmp_deviation":  ("LMP Deviation",  "30%"),
        "lmp_velocity":   ("LMP Velocity",   "15%"),
        "load_forecast":  ("Load vs Fcst",   "20%"),
        "reserve_margin": ("Reserve Margin", "25%"),
        "dart":           ("DART",           "10%"),
    }
    _SIG_COLORS = {
        "lmp_deviation":  "#818cf8",
        "lmp_velocity":   "#a78bfa",
        "load_forecast":  "#34d399",
        "reserve_margin": "#fbbf24",
        "dart":           "#f87171",
    }

    _selected_region = st.selectbox(
        "Select region to inspect",
        options=[f"{s['iso']} / {s['hub']}" for s in scores],
        key="sig_breakdown_select",
    )

    _sel_score = next(
        (s for s in scores if f"{s['iso']} / {s['hub']}" == _selected_region), None
    )

    if _sel_score:
        sig = _sel_score["signals"]
        sig_weights = _sel_score.get("signal_weights", {})
        active_n = _sel_score["active_signals"]

        _bc1, _bc2 = st.columns([3, 2])
        with _bc1:
            sig_names_ordered = ["lmp_deviation", "lmp_velocity", "load_forecast", "reserve_margin", "dart"]
            bar_vals, bar_colors_sig, bar_labels, bar_statuses = [], [], [], []
            for k in sig_names_ordered:
                v = sig.get(k)
                label, base_w = _SIG_LABELS[k]
                if v is not None:
                    bar_vals.append(v)
                    bar_colors_sig.append(_SIG_COLORS[k])
                    eff_w = sig_weights.get(k, 0)
                    bar_labels.append(f"{label} ({base_w} → {eff_w:.0%} eff.)")
                    bar_statuses.append("live")
                else:
                    bar_vals.append(0)
                    bar_colors_sig.append("#374151")
                    bar_labels.append(f"{label} ({base_w} — unavailable)")
                    bar_statuses.append("unavailable")

            fig_sig = go.Figure(go.Bar(
                x=bar_vals,
                y=bar_labels,
                orientation="h",
                marker_color=bar_colors_sig,
                text=[f"{v:.0f}" if bar_statuses[i] == "live" else "N/A"
                      for i, v in enumerate(bar_vals)],
                textposition="inside",
                textfont=dict(color="#f9fafb", size=11),
            ))
            fig_sig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e5e7eb", size=11),
                xaxis=dict(range=[0, 100], gridcolor="#1f2937", title="Score (0–100)"),
                yaxis=dict(gridcolor="rgba(0,0,0,0)", autorange="reversed"),
                margin=dict(t=10, b=30, l=10, r=20),
                height=210,
                showlegend=False,
            )
            st.plotly_chart(fig_sig, use_container_width=True)

        with _bc2:
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.03);border:1px solid #1f2937;
                        border-radius:8px;padding:14px;margin-top:4px;">
              <div style="font-size:0.65rem;color:#6b7280;text-transform:uppercase;
                          letter-spacing:0.1em;margin-bottom:8px;">Region details</div>
              <div style="font-size:1.5rem;font-weight:800;color:{_sel_score['stress_color']}">
                {_sel_score['stress_score']:.1f}
              </div>
              <div style="font-size:0.75rem;color:#9ca3af;margin-bottom:10px;">
                Composite stress score<br>
                <span style="color:#6b7280">{active_n}/5 signals live</span>
              </div>
              <div style="font-size:0.72rem;color:#9ca3af;line-height:1.8;">
                LMP: <b style="color:#e5e7eb">${_sel_score['lmp']:.2f}/MWh</b><br>
                ISO: <b style="color:#e5e7eb">{_sel_score['iso']}</b><br>
                Hub: <b style="color:#e5e7eb">{_sel_score['hub']}</b>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Data freshness
            data_time = _sel_score.get("data_time")
            if data_time is not None:
                try:
                    ts_fresh = pd.Timestamp(data_time)
                    if ts_fresh.tzinfo is None:
                        ts_fresh = ts_fresh.tz_localize("UTC")
                    age_min = int(
                        (datetime.now(timezone.utc) - ts_fresh.to_pydatetime())
                        .total_seconds() // 60
                    )
                    freshness_color = (
                        "#22c55e" if age_min < 10 else
                        "#eab308" if age_min < 30 else
                        "#ef4444"
                    )
                    st.markdown(f"""
                    <div style="font-size:0.68rem;color:#6b7280;margin-top:10px;">
                      LMP data as of<br>
                      <span style="color:{freshness_color}">
                        {ts_fresh.strftime('%H:%M UTC')} ({age_min}m ago)
                      </span>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception:
                    pass

            iso_ext = extended.get(_sel_score["iso"], {})
            load_ts_str = "—"
            act_load = iso_ext.get("actual_load")
            fc_load  = iso_ext.get("forecast_load")
            if act_load is not None:
                load_ts_str = f"{act_load:,.0f} MW actual"
            if fc_load is not None:
                load_ts_str += f" / {fc_load:,.0f} MW fcst"
            st.markdown(f"""
            <div style="font-size:0.68rem;color:#6b7280;margin-top:6px;">
              System load: <span style="color:#9ca3af">{load_ts_str}</span>
            </div>
            """, unsafe_allow_html=True)


# ── Stress Forecast tab ───────────────────────────────────────────────────────

with tab_forecast:
    st.markdown("### 📈 12-Hour Grid Stress Forecast")
    st.caption(
        "Forward stress computed from Day-Ahead LMP + load forecast as a proxy for future "
        "real-time conditions. DA LMP deviation (50%), forecast load vs 30d median (35%), "
        "current DART signal persisted (15%)."
    )

    _fwd = forward_stress  # use cached value from load_data()

    if not _fwd:
        st.warning(
            "Forward stress data unavailable — DA LMP may not be published yet "
            "or API rate limits were hit. Try refreshing in a moment."
        )
    else:
        try:
            _now_ts = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
            _fig_fwd = go.Figure()

            for y0, y1, col in [(0, 25, "rgba(34,197,94,0.07)"),
                                 (25, 50, "rgba(234,179,8,0.07)"),
                                 (50, 75, "rgba(249,115,22,0.07)"),
                                 (75, 100, "rgba(239,68,68,0.10)")]:
                _fig_fwd.add_hrect(y0=y0, y1=y1, fillcolor=col, layer="below", line_width=0)

            for iso, df_fwd in _fwd.items():
                _fig_fwd.add_trace(go.Scatter(
                    x=df_fwd["time"],
                    y=df_fwd["stress_score"],
                    mode="lines+markers",
                    name=f"{iso} ({ISO_FORECAST_HUB.get(iso, '')})",
                    line=dict(color=ISO_LINE_COLORS.get(iso, "#9ca3af"), width=2.5, dash="dot"),
                    marker=dict(size=6),
                    hovertemplate=(
                        f"<b>{iso}</b><br>"
                        "Time: %{x|%H:%M UTC}<br>"
                        "Fwd Stress: %{y:.1f}/100<br>"
                        "<extra></extra>"
                    ),
                ))

            _fig_fwd.add_vline(
                x=_now_ts.isoformat(),
                line_dash="solid", line_color="#6b7280", line_width=1.5,
                annotation_text="Now", annotation_position="top left",
                annotation_font_color="#9ca3af",
            )
            _fig_fwd.add_hline(y=75, line_dash="dot", line_color="#ef4444", line_width=1,
                               annotation_text="CRITICAL", annotation_position="top right",
                               annotation_font_color="#ef4444")
            _fig_fwd.add_hline(y=50, line_dash="dot", line_color="#f97316", line_width=1,
                               annotation_text="HIGH", annotation_position="top right",
                               annotation_font_color="#f97316")

            _fig_fwd.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e5e7eb", size=12),
                xaxis=dict(gridcolor="#1f2937", tickformat="%H:%M", title="Time (UTC)"),
                yaxis=dict(gridcolor="#1f2937", range=[0, 105],
                           title="Forward Stress Score (0–100)", zeroline=False),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                            bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
                margin=dict(t=40, b=40, l=55, r=20),
                height=360,
                hovermode="x unified",
            )
            st.plotly_chart(_fig_fwd, use_container_width=True)

        except Exception as e:
            st.error(f"Forecast chart error: {e}")

        # ── Proactive Routing Opportunities ──────────────────────────────────
        st.markdown("---")
        st.markdown("#### 🎯 Proactive Routing Opportunities")

        _opp_col1, _opp_col2 = st.columns([1, 2])
        with _opp_col1:
            if _fwd:
                _opp_source = st.selectbox(
                    "Your home ISO",
                    options=list(_fwd.keys()),
                    index=0,
                    key="opp_source_iso",
                )
            else:
                _opp_source = "PJM"
                st.info("No forecast data available")

            _opp_thresh_src = st.slider("Alert when home stress >", 25, 90, 50, 5, key="opp_thresh_src")
            _opp_thresh_dst = st.slider("Show destinations with stress <", 10, 75, 35, 5, key="opp_thresh_dst")

        try:
            opps = find_routing_opportunities(
                _fwd, source_iso=_opp_source,
                threshold_source=float(_opp_thresh_src),
                threshold_dest=float(_opp_thresh_dst),
            )
        except Exception:
            opps = []

        with _opp_col2:
            if not opps:
                st.markdown("""
                <div style="background:rgba(34,197,94,0.10);border:1px solid rgba(34,197,94,0.35);
                            border-radius:8px;padding:14px;color:#86efac;font-size:0.88rem;">
                  ✅ No stress windows detected in the next 12 hours at the selected thresholds.
                </div>
                """, unsafe_allow_html=True)
            else:
                _opp_rows = []
                for o in opps:
                    _opp_rows.append({
                        "Time (UTC)": pd.Timestamp(o["time"]).strftime("%H:%M"),
                        f"{o['source_iso']} Stress": f"{o['source_stress']:.0f} [{o['source_stress_label']}]",
                        "Best Destination": o["best_dest"] or "—",
                        "Dest Stress": f"{o['dest_stress']:.0f} [{o['dest_stress_label']}]" if o["best_dest"] else "—",
                        "Savings $/MWh": f"${o['savings_per_mwh']:.2f}" if o["savings_per_mwh"] is not None else "—",
                    })
                st.dataframe(pd.DataFrame(_opp_rows), use_container_width=True, hide_index=True)
                st.caption(f"{len(opps)} window{'s' if len(opps) != 1 else ''} found — consider pre-scheduling batch workloads.")

        # ── Batch Workload Scheduler ──────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 📅 Schedule Batch Workload")
        st.caption("Optimally distribute a batch job across ISOs to minimize cost and grid stress.")

        _sched_c1, _sched_c2, _sched_c3 = st.columns(3)
        with _sched_c1:
            _sched_mw = st.slider("Workload size (MW)", 1, 500, 50, 5, key="sched_mw")
        with _sched_c2:
            _sched_dur = st.slider("Duration (hours)", 1, 12, 4, 1, key="sched_dur")
        with _sched_c3:
            if _fwd:
                _sched_src = st.selectbox("Source ISO (baseline)", options=list(_fwd.keys()), index=0, key="sched_src_iso")
            else:
                _sched_src = "PJM"

        try:
            _schedule, _total_savings, _carbon_delta = compute_batch_schedule(
                _fwd, workload_mw=float(_sched_mw),
                duration_hours=int(_sched_dur),
                source_iso=_sched_src,
            )
        except Exception as e:
            st.error(f"Batch scheduler error: {e}")
            _schedule, _total_savings, _carbon_delta = [], 0.0, 0.0

        _m1, _m2, _m3 = st.columns(3)
        _savings_color = "#22c55e" if _total_savings >= 0 else "#ef4444"
        _carbon_color  = "#22c55e" if _carbon_delta >= 0 else "#f97316"
        _box_base = "border-radius:8px;padding:14px;text-align:center;min-height:110px;display:flex;flex-direction:column;justify-content:center;"
        with _m1:
            st.markdown(f"""
            <div style="background:rgba(34,197,94,0.10);border:1px solid rgba(34,197,94,0.3);{_box_base}">
              <div style="font-size:0.65rem;color:#6b7280;text-transform:uppercase;letter-spacing:0.1em;">Est. Cost Savings</div>
              <div style="font-size:1.6rem;font-weight:800;color:{_savings_color}">${abs(_total_savings):,.0f}</div>
              <div style="font-size:0.72rem;color:#6b7280;">vs always using {_sched_src}<br>over {_sched_dur}h @ {_sched_mw} MW</div>
            </div>
            """, unsafe_allow_html=True)
        with _m2:
            st.markdown(f"""
            <div style="background:rgba(96,165,250,0.10);border:1px solid rgba(96,165,250,0.3);{_box_base}">
              <div style="font-size:0.65rem;color:#6b7280;text-transform:uppercase;letter-spacing:0.1em;">Carbon Intensity Δ</div>
              <div style="font-size:1.6rem;font-weight:800;color:{_carbon_color}">{'+' if _carbon_delta >= 0 else ''}{_carbon_delta:.1f}%</div>
              <div style="font-size:0.72rem;color:#6b7280;">renewable fraction vs {_sched_src} baseline</div>
            </div>
            """, unsafe_allow_html=True)
        with _m3:
            _isos_used = list({item["iso"] for item in _schedule})
            st.markdown(f"""
            <div style="background:rgba(139,92,246,0.10);border:1px solid rgba(139,92,246,0.3);{_box_base}">
              <div style="font-size:0.65rem;color:#6b7280;text-transform:uppercase;letter-spacing:0.1em;">ISOs Used</div>
              <div style="font-size:1.6rem;font-weight:800;color:#c084fc">{len(_isos_used)}</div>
              <div style="font-size:0.72rem;color:#6b7280;">{' · '.join(_isos_used) if _isos_used else '—'}</div>
            </div>
            """, unsafe_allow_html=True)

        if _schedule:
            st.markdown("")
            _sched_rows = []
            for item in _schedule:
                _lbl, _ = stress_label(item["stress_score"])
                _sched_rows.append({
                    "Hour": f"T+{item['hour']}h",
                    "Time (UTC)": pd.Timestamp(item["time"]).strftime("%H:%M"),
                    "Route To": item["iso"],
                    "Stress": item["stress_score"],
                    "LMP $/MWh": item["lmp"] if item["lmp"] is not None else float("nan"),
                    "Label": _lbl,
                })
            _df_sched = pd.DataFrame(_sched_rows)
            st.dataframe(
                _df_sched, use_container_width=True, hide_index=True,
                column_config={
                    "Stress": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f"),
                    "LMP $/MWh": st.column_config.NumberColumn(format="$%.2f"),
                },
            )


# ── Crisis Replay tab ─────────────────────────────────────────────────────────

with tab_crisis:
    sc_col1, sc_col2, sc_col3 = st.columns([1, 1, 4])
    with sc_col1:
        if st.button(
            "🔴 ERCOT Feb 2021", use_container_width=True,
            type="primary" if st.session_state.get("crisis_scenario", "ERCOT_2021") == "ERCOT_2021" else "secondary",
        ):
            st.session_state.crisis_scenario = "ERCOT_2021"
            st.session_state.crisis_hour = 0
    with sc_col2:
        if st.button(
            "🌡 CAISO Aug 2020", use_container_width=True,
            type="primary" if st.session_state.get("crisis_scenario") == "CAISO_2020" else "secondary",
        ):
            st.session_state.crisis_scenario = "CAISO_2020"
            st.session_state.crisis_hour = 0

    if "crisis_scenario" not in st.session_state:
        st.session_state.crisis_scenario = "ERCOT_2021"
    if "crisis_hour" not in st.session_state:
        st.session_state.crisis_hour = 0

    scenario = ERCOT_2021 if st.session_state.crisis_scenario == "ERCOT_2021" else CAISO_2020

    st.markdown(f"""
    <div style="background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.25);
                border-radius:10px;padding:14px 18px;margin:8px 0 14px 0;">
      <div style="font-size:1.05rem;font-weight:700;color:#fca5a5;margin-bottom:6px;">{scenario.name}</div>
      <div style="font-size:0.84rem;color:#d1d5db;line-height:1.55;">{scenario.description}</div>
      <div style="font-size:0.72rem;color:#6b7280;margin-top:8px;">
        ⚠ Stress scores computed from synthetic-but-historically-accurate LMP data using the GridPulse algorithm.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Animated stress chart
    st.markdown("#### Grid Stress Over Time")
    st.caption("Press ▶ Play to watch stress scores evolve — or drag the slider")
    try:
        fig_scenario = build_scenario_animation(scenario)
        st.plotly_chart(fig_scenario, use_container_width=True, key=f"crisis_anim_{scenario.short_name}")
    except Exception as e:
        st.error(f"Animation error: {e}")

    # Key moment jump buttons
    st.markdown("**Jump to key moment:**")
    km_cols = st.columns(len(scenario.key_moments))
    for kmc, km in zip(km_cols, scenario.key_moments):
        with kmc:
            if st.button(f"{km.icon} {km.label}", use_container_width=True,
                         key=f"km_{scenario.short_name}_{km.hour_idx}"):
                st.session_state.crisis_hour = km.hour_idx

    # Timeline slider
    timestamps = scenario.timestamps
    slider_h = st.slider(
        "Timeline",
        min_value=0,
        max_value=scenario.hours - 1,
        value=int(st.session_state.crisis_hour),
        format="Hour %d",
        key=f"crisis_slider_{scenario.short_name}",
    )
    st.session_state.crisis_hour = slider_h
    crisis_h = slider_h

    ts_label = timestamps[crisis_h].strftime("%b %d, %H:%M UTC")
    st.caption(f"Snapshot at **{ts_label}**")

    # Snapshot: map | gauges | router recommendation
    map_col, gauge_col, router_col = st.columns([5, 5, 4])

    with map_col:
        st.markdown("##### ISO Locations")
        try:
            fig_snap_map = build_snapshot_map(scenario, crisis_h)
            st.plotly_chart(fig_snap_map, use_container_width=True,
                            key=f"crisis_map_{scenario.short_name}_{crisis_h}")
        except Exception as e:
            st.error(f"Map error: {e}")

    with gauge_col:
        st.markdown("##### Stress Gauges")
        try:
            fig_gauges = build_gauge_figure(scenario, crisis_h)
            st.plotly_chart(fig_gauges, use_container_width=True,
                            key=f"crisis_gauges_{scenario.short_name}_{crisis_h}")
        except Exception as e:
            st.error(f"Gauge error: {e}")

    with router_col:
        st.markdown("##### Router Recommendation")
        try:
            routing = scenario.routing_at(crisis_h, max_latency_ms=120)
            winner_cr = routing.get("winner")
            src_iso = scenario.primary_iso
            iso_data = scenario.iso_data_at(crisis_h)
            src_data = iso_data.get(src_iso, {})
            src_stress = src_data.get("stress_score", 0)
            src_lmp    = src_data.get("lmp", 0)
            src_lbl    = src_data.get("stress_label", "—")

            src_color_map = {"LOW": "#22c55e", "MODERATE": "#eab308", "HIGH": "#f97316", "CRITICAL": "#ef4444"}
            src_border = src_color_map.get(src_lbl, "#6b7280")

            if winner_cr and winner_cr.iso != src_iso:
                savings_mwh = (src_lmp - winner_cr.lmp) if winner_cr.lmp else 0
                savings_hr  = savings_mwh * scenario.workload_mw
                st.markdown(f"""
                <div style="background:rgba(34,197,94,0.10);border:1px solid #22c55e;
                            border-radius:8px;padding:12px;margin-bottom:8px;">
                  <div style="font-size:0.65rem;color:#6b7280;text-transform:uppercase;letter-spacing:0.1em;">
                    Route Away From {src_iso}
                  </div>
                  <div style="font-size:1.3rem;font-weight:800;color:#86efac;margin:4px 0;">
                    → {winner_cr.iso}
                  </div>
                  <div style="font-size:0.78rem;color:#6b7280;">{winner_cr.hub}</div>
                  <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:10px;">
                    <div style="background:rgba(255,255,255,0.04);border-radius:6px;padding:6px;text-align:center;">
                      <div style="font-size:0.6rem;color:#6b7280;">SOURCE LMP</div>
                      <div style="font-size:0.9rem;font-weight:700;color:#fca5a5;">${src_lmp:,.0f}</div>
                    </div>
                    <div style="background:rgba(255,255,255,0.04);border-radius:6px;padding:6px;text-align:center;">
                      <div style="font-size:0.6rem;color:#6b7280;">DEST LMP</div>
                      <div style="font-size:0.9rem;font-weight:700;color:#86efac;">${winner_cr.lmp:,.0f}</div>
                    </div>
                    <div style="background:rgba(255,255,255,0.04);border-radius:6px;padding:6px;text-align:center;">
                      <div style="font-size:0.6rem;color:#6b7280;">SAVINGS/MWH</div>
                      <div style="font-size:0.9rem;font-weight:700;color:#e5e7eb;">${savings_mwh:,.0f}</div>
                    </div>
                    <div style="background:rgba(255,255,255,0.04);border-radius:6px;padding:6px;text-align:center;">
                      <div style="font-size:0.6rem;color:#6b7280;">$/HOUR ({scenario.workload_mw:.0f} MW)</div>
                      <div style="font-size:0.9rem;font-weight:700;color:#e5e7eb;">${savings_hr:,.0f}</div>
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)
            elif winner_cr and winner_cr.iso == src_iso:
                st.markdown(f"""
                <div style="background:rgba(99,102,241,0.10);border:1px solid #818cf8;
                            border-radius:8px;padding:12px;margin-bottom:8px;">
                  <div style="font-size:0.65rem;color:#6b7280;text-transform:uppercase;letter-spacing:0.1em;">
                    Stay in {src_iso}
                  </div>
                  <div style="font-size:1.1rem;font-weight:700;color:#a5b4fc;margin:4px 0;">
                    {src_iso} is best eligible ISO
                  </div>
                  <div style="font-size:0.78rem;color:#9ca3af;">Stress: {src_stress:.0f}/100  LMP: ${src_lmp:,.0f}/MWh</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background:rgba(239,68,68,0.12);border:1px solid #ef4444;
                            border-radius:8px;padding:12px;margin-bottom:8px;">
                  <div style="font-size:0.65rem;color:#6b7280;text-transform:uppercase;letter-spacing:0.1em;">
                    No Eligible ISO
                  </div>
                  <div style="font-size:1.0rem;font-weight:700;color:#fca5a5;">⛔ All ISOs exceed stress or latency limit</div>
                  <div style="font-size:0.78rem;color:#9ca3af;margin-top:6px;">
                    {src_iso} stress: {src_stress:.0f}/100  LMP: ${src_lmp:,.0f}/MWh
                  </div>
                </div>
                """, unsafe_allow_html=True)

            pill_bg = {"LOW": "rgba(34,197,94,0.15)", "MODERATE": "rgba(234,179,8,0.15)",
                       "HIGH": "rgba(249,115,22,0.15)", "CRITICAL": "rgba(239,68,68,0.18)"}
            st.markdown(f"""
            <div style="background:{pill_bg.get(src_lbl,'rgba(99,102,241,0.12)')};
                        border:1px solid {src_border};border-radius:6px;padding:8px 12px;">
              <span style="font-size:0.65rem;color:#6b7280;text-transform:uppercase;">
                {src_iso} Status at {ts_label}
              </span><br>
              <span style="font-weight:700;color:{src_border};">{src_lbl}</span>
              <span style="color:#9ca3af;font-size:0.8rem;">  {src_stress:.0f}/100  ·  ${src_lmp:,.0f}/MWh</span>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Router recommendation error: {e}")

    # Key moment callout
    for km in scenario.key_moments:
        if abs(crisis_h - km.hour_idx) <= 2:
            km_bg     = {"⚠": "rgba(234,179,8,0.10)", "🔴": "rgba(239,68,68,0.12)", "📊": "rgba(192,132,252,0.10)"}
            km_border = {"⚠": "#eab308", "🔴": "#ef4444", "📊": "#c084fc"}
            st.markdown(f"""
            <div style="background:{km_bg.get(km.icon,'rgba(99,102,241,0.10)')};
                        border-left:3px solid {km_border.get(km.icon,'#818cf8')};
                        border-radius:0 8px 8px 0;padding:10px 14px;margin:10px 0;">
              <b style="color:{km_border.get(km.icon,'#a5b4fc')};">{km.label}</b><br>
              <span style="font-size:0.82rem;color:#d1d5db;">{km.description}</span>
            </div>
            """, unsafe_allow_html=True)
            break

    st.markdown("---")
    st.markdown("#### What Happened vs What GridPulse Would Have Done")
    cmp_left, cmp_right = st.columns(2)
    with cmp_left:
        st.markdown(f"""
        <div style="background:rgba(239,68,68,0.10);border:1px solid rgba(239,68,68,0.3);
                    border-radius:10px;padding:16px;">
          <div style="font-size:0.75rem;font-weight:700;color:#fca5a5;text-transform:uppercase;
                      letter-spacing:0.08em;margin-bottom:8px;">📋 What Happened</div>
          <div style="font-size:0.84rem;color:#d1d5db;line-height:1.6;">{scenario.what_happened}</div>
        </div>
        """, unsafe_allow_html=True)
    with cmp_right:
        st.markdown(f"""
        <div style="background:rgba(34,197,94,0.08);border:1px solid rgba(34,197,94,0.3);
                    border-radius:10px;padding:16px;">
          <div style="font-size:0.75rem;font-weight:700;color:#86efac;text-transform:uppercase;
                      letter-spacing:0.08em;margin-bottom:8px;">⚡ GridPulse Impact</div>
          <div style="font-size:0.84rem;color:#d1d5db;line-height:1.6;">{scenario.gridpulse_impact}</div>
        </div>
        """, unsafe_allow_html=True)


# ── Auto-refresh & footer ─────────────────────────────────────────────────────

st_autorefresh(interval=300_000, key="gridpulse_refresh")

st.markdown("---")
st.caption(
    "Built with GridStatus API · SCSP Hackathon 2026 · "
    "Data for informational purposes only · "
    "Team: Satwak & Sushmita"
)

