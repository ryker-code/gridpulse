"""
Inference routing decision engine — 3-tier algorithm.

Tier 1: Hard constraint filter
  - stress_score > STRESS_HARD_LIMIT (75) → excluded
  - estimated_latency > max_latency_ms → excluded
  Latency model: same-ISO = 10 ms, adjacent = 30 ms, cross-country = 70 ms
  Adjacency: PJM ↔ {MISO, NYISO, ISONE},  NYISO ↔ ISONE,  CAISO isolated,  ERCOT isolated

Tier 2: Multi-objective scoring (weights auto-normalized)
  stress_benefit  = (100 − stress_score) / 100
  lmp_score       = (max_lmp − lmp) / (max_lmp − min_lmp)
  renewable_score = static EIA 2024 annual fraction (live breakdown requires per-fuel API)
  dart_bonus      = 1.0 − (dart_stress_signal / 100)   [inverted: RT < DA → high bonus]
  routing_score   = w_stress·stress_benefit + w_price·lmp_score
                  + w_renewable·renewable_score + w_dart·dart_bonus

Tier 3: Rich result dict
  recommendation, winner (RegionResult), routing_score, destination_stress,
  source_stress, stress_delta_mw, cost_delta_per_mwh, renewable_fraction,
  estimated_latency_ms, dart_signal, rationale, all_candidates, excluded,
  n_eligible, n_total
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

STRESS_HARD_LIMIT = 75

# Bidirectional ISO adjacency graph for latency model
ADJACENCY: dict[str, set[str]] = {
    "PJM":   {"MISO", "NYISO", "ISONE"},
    "MISO":  {"PJM"},
    "NYISO": {"PJM", "ISONE"},
    "ISONE": {"PJM", "NYISO"},
    "CAISO": set(),
    "ERCOT": set(),
    "SPP":   set(),
}

# Static renewable fraction fallback (EIA 2024 annual averages)
STATIC_RENEWABLE_PCT: dict[str, float] = {
    "CAISO": 0.72,
    "ERCOT": 0.55,
    "MISO":  0.42,
    "ISONE": 0.40,
    "NYISO": 0.38,
    "SPP":   0.38,
    "PJM":   0.28,
}

REGION_DC_HUBS: dict[str, str] = {
    "PJM / WESTERN HUB":          "Northern Virginia / Ohio",
    "PJM / DOMINION HUB":         "Northern Virginia",
    "CAISO / TH_NP15_GEN-APND":   "Silicon Valley / Bay Area (NP15)",
    "CAISO / TH_SP15_GEN-APND":   "Los Angeles / Southern CA (SP15)",
    "ERCOT / HB_NORTH":           "Dallas / DFW",
    "ERCOT / HB_HOUSTON":         "Houston",
    "ERCOT / HB_SOUTH":           "San Antonio",
    "MISO / INDIANA.HUB":         "Indianapolis / Chicago",
    "ISONE / .H.INTERNAL_HUB":    "Boston / New England",
    "NYISO / N.Y.C.":             "New York City",
    "SPP / SPPNORTH_HUB":         "Oklahoma City / Kansas City",
    "SPP / SPPSOUTH_HUB":         "Dallas (SPP South)",
}


def _latency_ms(source_iso: str, dest_iso: str) -> int:
    if source_iso == dest_iso:
        return 10
    if dest_iso in ADJACENCY.get(source_iso, set()):
        return 30
    return 70


@dataclass
class RegionResult:
    iso: str
    hub: str
    region_key: str

    stress_score: float
    lmp: float
    stress_label: str
    stress_color: str

    latency_ms: int
    renewable_fraction: float

    # Component scores (0–1, higher = better)
    stress_component: float = 0.0
    price_component: float = 0.0
    renewable_component: float = 0.0
    dart_bonus: float = 0.5

    routing_score: float = 0.0

    filtered_stress: bool = False
    filtered_latency: bool = False

    dart_signal: str = "unknown"
    signals: dict = field(default_factory=dict)

    rationale: str = ""
    dc_location: str = ""

    @property
    def eligible(self) -> bool:
        return not self.filtered_stress and not self.filtered_latency

    @property
    def filter_reason(self) -> str:
        parts = []
        if self.filtered_stress:
            parts.append(f"stress {self.stress_score:.0f} > {STRESS_HARD_LIMIT}")
        if self.filtered_latency:
            parts.append(f"latency {self.latency_ms}ms > SLA")
        return "; ".join(parts) if parts else ""

    # Backwards-compat alias used in some display code
    @property
    def renewable_pct(self) -> float:
        return self.renewable_fraction


def compute_routing(
    scores: list[dict],
    workload_mw: float,
    max_latency_ms: int,
    weights: dict[str, float],
    source_iso: str = "PJM",
    extended_data: Optional[dict] = None,
) -> dict:
    """
    Score and rank all regions for the given workload.

    weights keys: stress, price, renewable, forward (dart bonus).
    Weights are auto-normalized so they need not sum to 1.0.

    Returns a rich routing result dict.
    """
    w_total = sum(weights.values()) or 1.0
    w = {k: v / w_total for k, v in weights.items()}

    results: list[RegionResult] = []

    for s in scores:
        iso = s["iso"]
        hub = s["hub"]
        key = f"{iso} / {hub}"

        lat = _latency_ms(source_iso, iso)
        ren = STATIC_RENEWABLE_PCT.get(iso, 0.40)
        dc_loc = REGION_DC_HUBS.get(key, "—")

        # Inverted DART bonus: low dart-stress (RT much cheaper than DA) → high routing bonus
        sig_dart = (s.get("signals") or {}).get("dart")
        if sig_dart is not None:
            dart_bonus = 1.0 - (sig_dart / 100.0)
        else:
            dart_bonus = 0.5  # neutral when DA LMP unavailable

        if dart_bonus > 0.65:
            dart_signal = "improving"    # RT meaningfully below DA
        elif dart_bonus < 0.35:
            dart_signal = "worsening"   # RT at or above DA
        else:
            dart_signal = "stable"

        r = RegionResult(
            iso=iso,
            hub=hub,
            region_key=key,
            stress_score=s["stress_score"],
            lmp=s["lmp"],
            stress_label=s["stress_label"],
            stress_color=s["stress_color"],
            latency_ms=lat,
            renewable_fraction=ren,
            dart_bonus=dart_bonus,
            dart_signal=dart_signal,
            signals=s.get("signals") or {},
            dc_location=dc_loc,
        )
        r.filtered_stress = r.stress_score > STRESS_HARD_LIMIT
        r.filtered_latency = lat > max_latency_ms
        results.append(r)

    lmps = [r.lmp for r in results]
    min_lmp, max_lmp = min(lmps), max(lmps)
    lmp_range = max_lmp - min_lmp or 1.0

    for r in results:
        r.stress_component = (100 - r.stress_score) / 100
        r.price_component = (max_lmp - r.lmp) / lmp_range
        r.renewable_component = r.renewable_fraction
        r.routing_score = (
            w.get("stress", 0.4) * r.stress_component
            + w.get("price", 0.3) * r.price_component
            + w.get("renewable", 0.2) * r.renewable_component
            + w.get("forward", 0.1) * r.dart_bonus
        )

    eligible = sorted([r for r in results if r.eligible], key=lambda r: r.routing_score, reverse=True)
    excluded = sorted([r for r in results if not r.eligible], key=lambda r: r.stress_score)

    # Source stress: first hub found in source ISO
    source_stress: Optional[float] = next(
        (s["stress_score"] for s in scores if s["iso"] == source_iso), None
    )

    if not eligible:
        return {
            "recommendation": None,
            "winner": None,
            "routing_score": 0.0,
            "destination_stress": None,
            "source_stress": source_stress,
            "stress_delta_mw": 0.0,
            "cost_delta_per_mwh": 0.0,
            "renewable_fraction": 0.0,
            "estimated_latency_ms": 0,
            "dart_signal": "unknown",
            "rationale": "No eligible regions. All regions exceed stress threshold or latency SLA.",
            "all_candidates": [],
            "excluded": excluded,
            "n_eligible": 0,
            "n_total": len(results),
        }

    winner = eligible[0]
    avg_lmp = sum(r.lmp for r in eligible) / len(eligible)
    dest_stress = winner.stress_score
    stress_delta = workload_mw * ((source_stress or dest_stress) - dest_stress) / 100

    winner.rationale = _build_rationale(winner, eligible, workload_mw, w, source_iso, source_stress)

    return {
        "recommendation": winner.region_key,
        "winner": winner,
        "routing_score": round(winner.routing_score, 4),
        "destination_stress": dest_stress,
        "source_stress": source_stress,
        "stress_delta_mw": round(stress_delta, 1),
        "cost_delta_per_mwh": round(winner.lmp - avg_lmp, 2),
        "renewable_fraction": winner.renewable_fraction,
        "estimated_latency_ms": winner.latency_ms,
        "dart_signal": winner.dart_signal,
        "rationale": winner.rationale,
        "all_candidates": eligible,
        "excluded": excluded,
        "n_eligible": len(eligible),
        "n_total": len(results),
    }


def _build_rationale(
    winner: RegionResult,
    eligible: list[RegionResult],
    workload_mw: float,
    w: dict[str, float],
    source_iso: str,
    source_stress: Optional[float],
) -> str:
    parts = []

    if len(eligible) > 1:
        runner = eligible[1]
        gap = winner.routing_score - runner.routing_score
        parts.append(f"Outscored {runner.iso}/{runner.hub.split()[0]} by {gap:.3f} pts.")

    if winner.stress_score < 25:
        parts.append(f"Grid stress LOW ({winner.stress_score:.0f}/100) — minimal congestion risk.")
    elif winner.stress_score < 50:
        parts.append(f"Grid stress MODERATE ({winner.stress_score:.0f}/100) — within safe operating bounds.")
    else:
        parts.append(f"Grid stress elevated ({winner.stress_score:.0f}/100) but below CRITICAL threshold.")

    if source_stress is not None and source_iso != winner.iso:
        delta = source_stress - winner.stress_score
        if delta > 5:
            parts.append(
                f"Routing from {source_iso} (stress {source_stress:.0f}) to {winner.iso} "
                f"(stress {winner.stress_score:.0f}) reduces exposure by {delta:.0f} pts."
            )

    avg_lmp = sum(r.lmp for r in eligible) / len(eligible)
    lmp_delta = winner.lmp - avg_lmp
    if lmp_delta < -2:
        parts.append(f"LMP ${winner.lmp:.2f}/MWh is ${abs(lmp_delta):.2f} below eligible average — cost-efficient.")
    elif lmp_delta > 2:
        parts.append(f"LMP ${winner.lmp:.2f}/MWh is ${lmp_delta:.2f} above average, offset by other factors.")
    else:
        parts.append(f"LMP ${winner.lmp:.2f}/MWh is near the eligible average.")

    if w.get("renewable", 0) > 0.05:
        above = winner.renewable_fraction > 0.45
        parts.append(
            f"~{winner.renewable_fraction:.0%} renewable generation "
            f"({'above' if above else 'below'} grid median)."
        )

    if winner.dart_signal == "improving":
        parts.append("RT prices trending below DA forward — favorable near-term cost signal.")
    elif winner.dart_signal == "worsening":
        parts.append("RT prices at or above DA forward — monitor for congestion build-up.")

    stress_avoided = workload_mw * (100 - winner.stress_score) / 100
    parts.append(f"Placing {workload_mw:.0f} MW here avoids ~{stress_avoided:.1f} MW of stressed-grid exposure.")

    return " ".join(parts)
