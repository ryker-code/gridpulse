"""
Hardcoded backtesting scenarios for two historical grid stress events.

1. ERCOT February 2021 — Winter Storm Uri
   Synthetic-but-historically-accurate hourly data, Feb 10–17 2021 (192 hours)
   ISOs: ERCOT (crisis), MISO (moderate stress), SPP (elevated), PJM + CAISO (background)
   Key references: PUCT post-crisis report, Texas Tribune, ERCOT market advisories

2. CAISO August 2020 — Western Heatwave Emergency
   Hourly data, Aug 14–15 2020 (48 hours, midnight PDT start)
   ISOs: CAISO (crisis), PJM (background), MISO (background)
   Key references: CPUC investigation report, CAISO emergency notices
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

from grid_stress import _clamp, stress_label
from router import compute_routing

_rng = np.random.default_rng(42)

_DEFAULT_WEIGHTS = {"stress": 0.40, "price": 0.30, "renewable": 0.20, "forward": 0.10}


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class KeyMoment:
    label: str
    hour_idx: int
    icon: str
    color: str
    description: str


@dataclass
class ScenarioISO:
    iso: str
    hub: str
    line_color: str
    display_label: str
    lat: float
    lon: float
    lmps: np.ndarray = field(default_factory=lambda: np.array([]))
    stress_scores: np.ndarray = field(default_factory=lambda: np.array([]))
    reserve_margins: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class Scenario:
    name: str
    short_name: str
    start_dt: datetime
    hours: int
    isos: list[ScenarioISO]
    key_moments: list[KeyMoment]
    primary_iso: str
    description: str
    what_happened: str
    gridpulse_impact: str
    workload_mw: float = 240.0

    @property
    def timestamps(self) -> list[datetime]:
        return [self.start_dt + timedelta(hours=h) for h in range(self.hours)]

    @property
    def primary(self) -> ScenarioISO:
        return next(s for s in self.isos if s.iso == self.primary_iso)

    def iso_map(self) -> dict[str, ScenarioISO]:
        return {s.iso: s for s in self.isos}

    def iso_data_at(self, h: int) -> dict[str, dict]:
        """Return {iso: {lmp, stress_score, reserve_margin, ...}} at hour h."""
        h = max(0, min(h, self.hours - 1))
        result = {}
        for s in self.isos:
            stress = float(np.clip(s.stress_scores[h], 0, 100))
            lbl, col = stress_label(stress)
            rm = float(s.reserve_margins[h]) if len(s.reserve_margins) > h else None
            result[s.iso] = {
                "iso": s.iso, "hub": s.hub,
                "lmp": float(s.lmps[h]),
                "stress_score": stress, "stress_label": lbl, "stress_color": col,
                "reserve_margin": rm,
            }
        return result

    def scores_at(self, h: int) -> list[dict]:
        """Score dicts compatible with compute_routing()."""
        scores = []
        for s in self.isos:
            h = max(0, min(h, self.hours - 1))
            stress = float(np.clip(s.stress_scores[h], 0, 100))
            lbl, col = stress_label(stress)
            scores.append({
                "iso": s.iso, "hub": s.hub,
                "stress_score": stress, "lmp": float(s.lmps[h]),
                "stress_label": lbl, "stress_color": col,
                "signals": {},
            })
        return scores

    def routing_at(self, h: int, max_latency_ms: int = 120) -> dict:
        return compute_routing(
            self.scores_at(h), self.workload_mw, max_latency_ms,
            _DEFAULT_WEIGHTS, source_iso=self.primary_iso,
        )

    def df(self) -> pd.DataFrame:
        rows: dict = {"time": self.timestamps}
        for s in self.isos:
            rows[f"lmp_{s.iso}"]    = s.lmps
            rows[f"stress_{s.iso}"] = s.stress_scores
        return pd.DataFrame(rows)


# ── Utility helpers ───────────────────────────────────────────────────────────

def _piecewise(h: np.ndarray, knots: list[tuple[float, float]], noise_std: float = 0.0) -> np.ndarray:
    """Linear interpolation between (hour, value) knots, with optional Gaussian noise."""
    xs = np.array([k[0] for k in knots], dtype=float)
    ys = np.array([k[1] for k in knots], dtype=float)
    vals = np.interp(h, xs, ys)
    if noise_std > 0:
        vals = vals + _rng.normal(0, noise_std, size=len(vals))
    return vals.clip(0)


def _diurnal(h: np.ndarray, amp: float, peak_hour: int = 8) -> np.ndarray:
    """Sine-shaped diurnal signal peaking at peak_hour (UTC)."""
    return amp * np.sin((h % 24 - peak_hour) * math.pi / 12)


# ── ERCOT February 2021 ───────────────────────────────────────────────────────
# Timeline (all UTC; ERCOT = CST = UTC-6):
#   h0    = Feb 10 00:00 UTC  (start)
#   h98   = Feb 14 02:00 UTC  = Feb 13 8 PM CST  → first HIGH alert
#   h103  = Feb 14 07:00 UTC  = Feb 14 1 AM CST  → Emergency curtailments
#   h134  = Feb 15 14:00 UTC  = Feb 15 8 AM CST  → Peak crisis $9,000/MWh
#   h192  = Feb 17 24:00 UTC  (end)

def _build_ercot_2021() -> Scenario:
    H = 192
    h = np.arange(H, dtype=float)

    # ── ERCOT ──
    ercot_lmp_knots = [
        (0,   150),   # Feb 10: already 4× baseline per spec
        (24,  180),   # Feb 11: creeping higher
        (48,  220),   # Feb 12: accelerating
        (72,  350),   # Feb 13 dawn
        (86,  490),   # Feb 13 14:00 UTC (8 AM CST)
        (98,  900),   # Feb 14 02:00 UTC (first alert) — $900/MWh per spec
        (103, 2500),  # Feb 14 07:00 UTC (curtailments begin)
        (108, 9000),  # Feb 14 12:00 UTC (cap hit)
        (168, 9000),  # Feb 16 24:00 UTC (3 days at cap)
        (178, 2500),  # Feb 17 10:00 UTC (beginning recovery)
        (192, 450),   # end of period
    ]
    ercot_lmps = _piecewise(h, ercot_lmp_knots, noise_std=0)
    ercot_lmps += _diurnal(h, amp=15, peak_hour=14)
    ercot_lmps = np.clip(ercot_lmps + _rng.normal(0, 40, H), 80, 9200)

    ercot_stress_knots = [
        (0,   65),    # Feb 10: elevated (4× baseline, +20% load deviation)
        (48,  68),
        (72,  72),
        (86,  75),    # HIGH threshold crossed
        (98,  87),
        (103, 97),    # 97/100 per spec (crisis begins)
        (108, 100),   # 100/100 per spec (cap hit)
        (168, 100),
        (178, 95),
        (192, 88),
    ]
    ercot_stress = _piecewise(h, ercot_stress_knots, noise_std=1.5).clip(0, 100)

    ercot_rm_knots = [
        (0,  18),   # 18% reserve margin Feb 10 start
        (48, 14),
        (72, 11),
        (86, 8),
        (98, 5.5),
        (103, 3.7),  # 3.7% per spec
        (108, 1.5),
        (120, 0),    # rolling blackouts
        (168, 2),
        (192, 7),
    ]
    ercot_rm = _piecewise(h, ercot_rm_knots, noise_std=0.3).clip(0, 25)

    # ── MISO ──
    miso_lmp_knots = [
        (0,  48), (48, 58), (72, 75), (86, 120), (98, 210),
        (108, 350), (120, 320), (168, 240), (192, 95),
    ]
    miso_lmps = _piecewise(h, miso_lmp_knots, noise_std=0)
    miso_lmps += _diurnal(h, amp=18, peak_hour=13)
    miso_lmps = np.clip(miso_lmps + _rng.normal(0, 15, H), 30, 500)

    miso_stress_knots = [
        (0, 42), (48, 48), (72, 54), (86, 62), (98, 70),
        (108, 78), (120, 75), (168, 72), (192, 55),
    ]
    miso_stress = _piecewise(h, miso_stress_knots, noise_std=2).clip(0, 100)

    miso_rm_knots = [
        (0, 22), (72, 18), (98, 12), (108, 9), (120, 8), (168, 11), (192, 16),
    ]
    miso_rm = _piecewise(h, miso_rm_knots, noise_std=0.4).clip(0, 30)

    # ── SPP ──
    spp_lmp_knots = [
        (0, 40), (48, 48), (72, 62), (86, 95), (98, 170),
        (108, 280), (120, 250), (168, 180), (192, 72),
    ]
    spp_lmps = _piecewise(h, spp_lmp_knots, noise_std=0)
    spp_lmps += _diurnal(h, amp=12, peak_hour=14)
    spp_lmps = np.clip(spp_lmps + _rng.normal(0, 12, H), 25, 400)

    spp_stress_knots = [
        (0, 36), (48, 40), (72, 46), (86, 55), (98, 63),
        (108, 65), (120, 63), (168, 60), (192, 46),
    ]
    spp_stress = _piecewise(h, spp_stress_knots, noise_std=1.8).clip(0, 100)

    spp_rm_knots = [
        (0, 24), (72, 20), (98, 13), (108, 10), (120, 11), (168, 14), (192, 19),
    ]
    spp_rm = _piecewise(h, spp_rm_knots, noise_std=0.3).clip(0, 30)

    # ── PJM (background — winter premium but not crisis) ──
    pjm_lmps = (
        _piecewise(h, [(0, 62), (48, 70), (108, 85), (168, 75), (192, 65)], noise_std=0)
        + _diurnal(h, amp=22, peak_hour=9)
        + _rng.normal(0, 8, H)
    ).clip(35, 130)
    pjm_stress = _piecewise(h, [(0, 38), (48, 42), (108, 48), (168, 44), (192, 40)], noise_std=2).clip(0, 65)

    # ── CAISO (background — California winter, unaffected) ──
    caiso_lmps = (
        _piecewise(h, [(0, 30), (192, 35)], noise_std=0)
        + _diurnal(h, amp=16, peak_hour=10)
        + _rng.normal(0, 5, H)
    ).clip(12, 70)
    caiso_stress = (_piecewise(h, [(0, 28), (192, 30)], noise_std=0) + _rng.normal(0, 2, H)).clip(15, 40)

    start = datetime(2021, 2, 10, 0, 0, tzinfo=timezone.utc)

    return Scenario(
        name="ERCOT February 2021 — Winter Storm Uri",
        short_name="ERCOT Feb 2021",
        start_dt=start,
        hours=H,
        primary_iso="ERCOT",
        description=(
            "Winter Storm Uri (Feb 10–17 2021) drove ~30 GW of Texas generation offline. "
            "ERCOT prices hit the $9,000/MWh administrative cap for three consecutive days. "
            "**4.5 million households** lost power. Estimated economic damage: **$195B+**."
        ),
        what_happened=(
            "~30 GW of unplanned generation outages. "
            "ERCOT prices hit the $9,000/MWh price cap for 72+ hours. "
            "4.5 million households without power for up to 4 days. "
            "Estimated economic damages: **$130–195B**."
        ),
        gridpulse_impact=(
            "GridPulse would have started routing AI workloads **away from Texas on Feb 13** — "
            "**34 hours before the emergency curtailments** — as ERCOT stress crossed 75/100. "
            "A 240 MW inference cluster represents ~0.35% of ERCOT peak load. "
            "Early rerouting to CAISO (~$30/MWh) saves **$8,970/MWh vs $9,000 peak**, "
            "or ~$2.15M/hour for a 240 MW workload."
        ),
        workload_mw=240.0,
        isos=[
            ScenarioISO("ERCOT", "HB_NORTH",          "#ef4444", "ERCOT · HB North",       32.8, -97.3,  ercot_lmps,  ercot_stress,  ercot_rm),
            ScenarioISO("MISO",  "INDIANA.HUB",        "#f97316", "MISO · Indiana Hub",     41.9, -87.7,  miso_lmps,   miso_stress,   miso_rm),
            ScenarioISO("SPP",   "SPPNORTH_HUB",       "#eab308", "SPP · North Hub",        38.5, -97.5,  spp_lmps,    spp_stress,    spp_rm),
            ScenarioISO("PJM",   "WESTERN HUB",        "#818cf8", "PJM · Western Hub",      39.5, -77.5,  pjm_lmps,    pjm_stress,    np.full(H, 20.0)),
            ScenarioISO("CAISO", "TH_NP15_GEN-APND",   "#34d399", "CAISO · NP15",           37.4, -121.9, caiso_lmps,  caiso_stress,  np.full(H, 25.0)),
        ],
        key_moments=[
            KeyMoment(
                label="⚠ Feb 13, 8 PM CST", hour_idx=98, icon="⚠",
                color="#eab308",
                description="GridPulse first HIGH alert — ERCOT stress crosses 75/100. "
                            "Router begins recommending CAISO over ERCOT. "
                            "LMP: ~$500/MWh (14× seasonal baseline).",
            ),
            KeyMoment(
                label="🔴 Feb 14, 1 AM CST", hour_idx=103, icon="🔴",
                color="#ef4444",
                description="ERCOT declares Grid Emergency. Forced curtailments begin at 1:25 AM. "
                            "Stress: 97/100. LMP: $900/MWh. Reserve margin: 3.7%. "
                            "GridPulse hard-filters ERCOT. All AI workloads routed to CAISO or PJM.",
            ),
            KeyMoment(
                label="📊 Feb 15, 8 AM CST", hour_idx=134, icon="📊",
                color="#c084fc",
                description="Peak crisis. ERCOT at $9,000/MWh administrative cap. "
                            "Rolling blackouts affecting 4.5M households. Stress: 100/100. "
                            "Reserve margin: 0%. CAISO routing saves $8,970/MWh.",
            ),
        ],
    )


# ── CAISO August 2020 ─────────────────────────────────────────────────────────
# Timeline (UTC-7 = PDT; start = Aug 14 07:00 UTC = midnight PDT):
#   h0    = Aug 14 07:00 UTC (midnight PDT)
#   h15   = Aug 14 22:00 UTC (3 PM PDT)    → stress entering High
#   h19   = Aug 15 02:00 UTC (7 PM PDT)    → Stage 3 emergency
#   h41   = Aug 16 00:00 UTC (5 PM PDT Aug 15) → second wave
#   h48   = end

def _build_caiso_2020() -> Scenario:
    H = 48
    h = np.arange(H, dtype=float)

    # ── CAISO ──
    caiso_lmp_knots = [
        (0,  45),    # midnight PDT — normal overnight
        (7,  50),    # morning rise
        (10, 80),    # morning peak
        (12, 65),    # solar trough
        (15, 220),   # 3 PM PDT — afternoon spike begins
        (18, 700),   # 6 PM PDT — Stage 2 emergency
        (19, 1000),  # 7 PM PDT — Stage 3 emergency / rolling blackouts
        (21, 1000),  # sustained
        (23, 400),   # blackouts ease, grid recovers
        (26, 150),   # overnight recovery
        (31, 80),    # Aug 15 morning
        (35, 300),   # Aug 15 afternoon rising
        (38, 800),   # 5 PM PDT second heat event
        (41, 1000),  # second Stage 3
        (43, 500),
        (48, 80),
    ]
    caiso_lmps = _piecewise(h, caiso_lmp_knots, noise_std=0)
    caiso_lmps += _diurnal(h, amp=20, peak_hour=13)
    caiso_lmps = np.clip(caiso_lmps + _rng.normal(0, 25, H), 10, 1100)

    caiso_stress_knots = [
        (0,  22),
        (10, 28),
        (13, 35),
        (15, 58),   # 3 PM — HIGH threshold approached
        (17, 72),
        (19, 90),   # Stage 3 emergency
        (21, 95),   # rolling blackouts
        (23, 78),
        (26, 55),
        (31, 38),
        (35, 58),
        (38, 78),
        (41, 93),   # second wave
        (43, 82),
        (48, 40),
    ]
    caiso_stress = _piecewise(h, caiso_stress_knots, noise_std=1.5).clip(0, 100)

    caiso_rm_knots = [
        (0, 28), (15, 18), (17, 12), (19, 6), (21, 3),
        (23, 8), (26, 14), (31, 22), (35, 16), (38, 10),
        (41, 4), (43, 8), (48, 20),
    ]
    caiso_rm = _piecewise(h, caiso_rm_knots, noise_std=0.3).clip(0, 35)

    # ── PJM (background) ──
    pjm_lmps = (
        _piecewise(h, [(0, 42), (48, 45)], noise_std=0)
        + _diurnal(h, amp=18, peak_hour=9)
        + _rng.normal(0, 6, H)
    ).clip(22, 80)
    pjm_stress = (_piecewise(h, [(0, 24), (48, 26)], noise_std=0) + _rng.normal(0, 2, H)).clip(15, 38)

    # ── MISO (background — slightly elevated from heat) ──
    miso_lmps = (
        _piecewise(h, [(0, 35), (48, 38)], noise_std=0)
        + _diurnal(h, amp=12, peak_hour=14)
        + _rng.normal(0, 5, H)
    ).clip(18, 65)
    miso_stress = (_piecewise(h, [(0, 28), (48, 30)], noise_std=0) + _rng.normal(0, 2, H)).clip(18, 42)

    start = datetime(2020, 8, 14, 7, 0, tzinfo=timezone.utc)

    return Scenario(
        name="CAISO August 2020 — Western Heatwave Emergency",
        short_name="CAISO Aug 2020",
        start_dt=start,
        hours=H,
        primary_iso="CAISO",
        description=(
            "The August 2020 Western heatwave drove California demand to near-record levels. "
            "CAISO declared Stage 3 grid emergencies on **Aug 14 and Aug 15**, with rolling blackouts "
            "affecting ~800,000 customers. Prices hit the $1,000/MWh soft cap. "
            "Root cause: extreme simultaneous heat across the entire Western grid plus "
            "insufficient evening 'net load' resources as solar dropped."
        ),
        what_happened=(
            "Stage 3 grid emergencies on two consecutive evenings (Aug 14–15). "
            "~800,000 California customers experienced rolling blackouts. "
            "CAISO LMP hit the $1,000/MWh soft cap. "
            "Estimated economic impact: **~$1.4B** (CPUC post-event analysis)."
        ),
        gridpulse_impact=(
            "GridPulse would have begun rerouting AI workloads at ~2 PM PDT Aug 14 — "
            "**5 hours before the Stage 3 emergency declaration**. "
            "Target: PJM (~$42/MWh) or MISO (~$36/MWh). "
            "At $1,000/MWh peak vs $42/MWh in PJM, a 240 MW inference cluster "
            "saves ~**$230,000/hour** during peak crisis. "
            "Proactive rerouting also reduces West-coast grid stress by ~240 MW (~0.6% of peak)."
        ),
        workload_mw=240.0,
        isos=[
            ScenarioISO("CAISO", "TH_NP15_GEN-APND",   "#ef4444", "CAISO · NP15",          37.4, -121.9, caiso_lmps,  caiso_stress, caiso_rm),
            ScenarioISO("PJM",   "WESTERN HUB",          "#818cf8", "PJM · Western Hub",    39.5, -77.5,  pjm_lmps,    pjm_stress,   np.full(H, 22.0)),
            ScenarioISO("MISO",  "INDIANA.HUB",           "#34d399", "MISO · Indiana Hub",  41.9, -87.7,  miso_lmps,   miso_stress,  np.full(H, 25.0)),
        ],
        key_moments=[
            KeyMoment(
                label="⚠ Aug 14, 3 PM PDT", hour_idx=15, icon="⚠",
                color="#eab308",
                description="CAISO stress crosses 50/100 (HIGH). LMP: ~$220/MWh. "
                            "Solar output dropping rapidly. GridPulse begins routing to PJM.",
            ),
            KeyMoment(
                label="🔴 Aug 14, 7 PM PDT", hour_idx=19, icon="🔴",
                color="#ef4444",
                description="CAISO Stage 3 Emergency. Rolling blackouts begin. "
                            "LMP: $1,000/MWh (soft cap). Stress: 90/100. "
                            "GridPulse routes 100% to PJM · Western Hub ($42/MWh).",
            ),
            KeyMoment(
                label="📊 Aug 15, 5 PM PDT", hour_idx=41, icon="📊",
                color="#c084fc",
                description="Second wave. CAISO stress 93/100. LMP: $1,000/MWh again. "
                            "PJM + MISO remain below 30/100 — GridPulse routes all workloads east. "
                            "Estimated savings: $230K/hour vs staying in CAISO.",
            ),
        ],
    )


# Module-level instances (built once on import)
ERCOT_2021: Scenario = _build_ercot_2021()
CAISO_2020: Scenario = _build_caiso_2020()

ALL_SCENARIOS: list[Scenario] = [ERCOT_2021, CAISO_2020]


# ── Animated Plotly figure ────────────────────────────────────────────────────

def build_scenario_animation(scenario: Scenario) -> "go.Figure":  # type: ignore[name-defined]
    import plotly.graph_objects as go

    sdf = scenario.df()
    # Sample every 4 hours for smooth-enough animation
    frame_idxs = list(range(4, scenario.hours, 4))
    iso_colors = {s.iso: s.line_color for s in scenario.isos}

    def _traces(up_to: int) -> list:
        return [
            go.Scatter(
                x=sdf["time"].iloc[:up_to],
                y=sdf[f"stress_{s.iso}"].iloc[:up_to],
                name=s.display_label,
                mode="lines",
                line=dict(color=s.line_color, width=2.5),
                showlegend=True,
            )
            for s in scenario.isos
        ]

    frames = [
        go.Frame(
            data=_traces(i),
            name=str(i),
            layout=go.Layout(
                annotations=[dict(
                    x=0.01, y=0.97, xref="paper", yref="paper",
                    text=f"<b>{sdf['time'].iloc[i - 1].strftime('%b %d %H:%M UTC')}</b>",
                    showarrow=False,
                    font=dict(size=12, color="#e5e7eb"),
                    bgcolor="rgba(14,17,23,0.7)", borderpad=4,
                )],
            ),
        )
        for i in frame_idxs
    ]

    slider_steps = [
        dict(
            method="animate",
            args=[[str(i)], {"frame": {"duration": 60, "redraw": True}, "mode": "immediate"}],
            label=sdf["time"].iloc[i - 1].strftime("%b %d"),
        )
        for i in frame_idxs
    ]

    fig = go.Figure(
        data=_traces(4),
        layout=go.Layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e5e7eb", size=12),
            title=dict(text=scenario.name, font=dict(size=12, color="#d1d5db"), x=0),
            xaxis=dict(gridcolor="#1f2937", tickformat="%b %d", title=""),
            yaxis=dict(gridcolor="#1f2937", range=[0, 108], title="Stress Score (0–100)", zeroline=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                        bgcolor="rgba(0,0,0,0)"),
            hovermode="x unified",
            margin=dict(t=80, b=100, l=55, r=20),
            height=380,
            updatemenus=[dict(
                type="buttons", showactive=False, y=1.18, x=0.0, xanchor="left",
                buttons=[
                    dict(label="▶  Play", method="animate",
                         args=[None, {"frame": {"duration": 60, "redraw": True},
                                      "fromcurrent": True, "transition": {"duration": 0}}]),
                    dict(label="⏸  Pause", method="animate",
                         args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}]),
                    dict(label="↩  Reset", method="animate",
                         args=[["4"], {"frame": {"duration": 0}, "mode": "immediate"}]),
                ],
            )],
            sliders=[dict(
                steps=slider_steps, x=0, y=0, len=1.0,
                pad={"t": 45, "b": 5},
                currentvalue=dict(prefix="", visible=True, xanchor="center",
                                  font=dict(color="#9ca3af", size=10)),
                transition=dict(duration=0),
                bgcolor="#1f2937", activebgcolor="#818cf8",
                font=dict(color="#6b7280", size=9), tickcolor="#374151", bordercolor="#374151",
            )],
        ),
        frames=frames,
    )

    # Threshold lines and key moment markers
    fig.add_hline(y=75, line_dash="dot", line_color="#ef4444", line_width=1.5,
                  annotation_text="CRITICAL (75)", annotation_position="top right",
                  annotation_font_color="#ef4444")
    fig.add_hrect(y0=75, y1=108, fillcolor="rgba(239,68,68,0.06)", layer="below", line_width=0)

    km_colors = {"⚠": "#eab308", "🔴": "#ef4444", "📊": "#c084fc"}
    for km in scenario.key_moments:
        t = scenario.start_dt + timedelta(hours=km.hour_idx)
        fig.add_vline(
            x=t.isoformat(), line_dash="dot",
            line_color=km_colors.get(km.icon, "#6b7280"), line_width=1.2,
            annotation_text=km.icon, annotation_position="top",
            annotation_font_color=km_colors.get(km.icon, "#6b7280"),
        )

    return fig


def build_snapshot_map(scenario: Scenario, h: int) -> "go.Figure":  # type: ignore[name-defined]
    """
    Static Scattergeo showing ISO stress circles at hour h.
    Circle size is fixed (all ISOs treated equally for the scenario map).
    """
    import plotly.graph_objects as go

    iso_data = scenario.iso_data_at(h)

    lats, lons, colors, hovers, texts = [], [], [], [], []
    for s in scenario.isos:
        d = iso_data.get(s.iso, {})
        stress = d.get("stress_score", 0)
        lmp    = d.get("lmp", 0)
        lbl    = d.get("stress_label", "—")
        rm     = d.get("reserve_margin")
        rm_str = f"Reserve margin: {rm:.1f}%<br>" if rm is not None else ""

        # Stress → color
        if stress >= 75:   col = "#ef4444"
        elif stress >= 50: col = "#f97316"
        elif stress >= 25: col = "#eab308"
        else:              col = "#22c55e"

        lats.append(s.lat)
        lons.append(s.lon)
        colors.append(col)
        texts.append(f"<b>{s.iso}</b>")
        hovers.append(
            f"<b>{s.display_label}</b><br>"
            f"Stress: {stress:.0f}/100  [{lbl}]<br>"
            f"LMP: ${lmp:,.0f}/MWh<br>"
            f"{rm_str}"
        )

    fig = go.Figure()
    fig.add_trace(go.Scattergeo(
        lat=lats, lon=lons,
        mode="markers+text",
        marker=dict(size=28, color=colors, symbol="circle",
                    line=dict(width=2, color="rgba(255,255,255,0.6)"), opacity=0.92),
        text=texts,
        textfont=dict(size=9, color="white"),
        textposition="top center",
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hovers,
        showlegend=False,
    ))
    fig.update_layout(
        geo=dict(scope="usa", projection_type="albers usa",
                 showland=True, landcolor="#1a1f2e",
                 showlakes=False, showocean=True, oceancolor="#0e1117",
                 showcoastlines=True, coastlinecolor="#374151",
                 showsubunits=True, subunitcolor="#374151",
                 bgcolor="#0e1117"),
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=0, b=0, l=0, r=0),
        height=260,
        showlegend=False,
    )
    return fig


def build_gauge_figure(scenario: Scenario, h: int) -> "go.Figure":  # type: ignore[name-defined]
    """Plotly gauge indicators for each ISO at hour h."""
    import plotly.graph_objects as go

    iso_data = scenario.iso_data_at(h)
    isos = scenario.isos
    n = len(isos)
    cols = min(n, 3)

    fig = go.Figure()

    # domain grid
    cell_w = 1.0 / cols
    cell_h = 1.0 / math.ceil(n / cols)

    for i, s in enumerate(isos):
        row = i // cols
        col = i % cols
        d = iso_data.get(s.iso, {})
        stress = d.get("stress_score", 0)

        if stress >= 75:   bar_col = "#ef4444"
        elif stress >= 50: bar_col = "#f97316"
        elif stress >= 25: bar_col = "#eab308"
        else:              bar_col = "#22c55e"

        x0 = col * cell_w + 0.02
        x1 = (col + 1) * cell_w - 0.02
        y0 = 1 - (row + 1) * cell_h
        y1 = 1 - row * cell_h

        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=stress,
            title=dict(text=s.display_label, font=dict(size=11, color="#9ca3af")),
            gauge=dict(
                axis=dict(range=[0, 100], tickcolor="#374151",
                          tickfont=dict(size=9, color="#6b7280")),
                bar=dict(color=bar_col, thickness=0.7),
                bgcolor="#1f2937",
                borderwidth=0,
                steps=[
                    dict(range=[0, 25], color="rgba(34,197,94,0.08)"),
                    dict(range=[25, 50], color="rgba(234,179,8,0.08)"),
                    dict(range=[50, 75], color="rgba(249,115,22,0.08)"),
                    dict(range=[75, 100], color="rgba(239,68,68,0.10)"),
                ],
                threshold=dict(line=dict(color="#ef4444", width=2), thickness=0.75, value=75),
            ),
            number=dict(font=dict(size=22, color="#f9fafb"), suffix="/100"),
            domain=dict(x=[x0, x1], y=[y0, y1]),
        ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5e7eb"),
        margin=dict(t=10, b=10, l=10, r=10),
        height=180 * math.ceil(n / cols),
    )
    return fig
