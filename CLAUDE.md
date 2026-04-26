# GridPulse — Grid-Aware AI Inference Router

## Project Purpose
A real-time grid stress monitoring and AI inference routing system for the SCSP Hackathon (Electric Grid Optimization track) and Claude Code Hackathon (April 21-28 2026).

## What We're Building
A Streamlit dashboard with two components:
1. **Grid Stress Monitor**: Pulls real-time LMP data from 8 ISOs via GridStatus API, computes a composite Grid Stress Score (0-100) per region using a 5-signal algorithm
2. **Inference Router**: Given an incoming AI workload (MW size, max latency SLA), recommends optimal routing target across ISO regions

## GridStatus API
- API Key: ef58d55f5a5f4f54a58102984ca32925
- Base URL: https://api.gridstatus.io/v1/datasets
- Rate limit: 1 req/second (free tier)
- Key datasets: pjm_lmp_real_time_5_min, caiso_lmp_real_time_5_min, ercot_lmp_by_bus, miso_lmp_real_time_5_min, isone_lmp_real_time_5_min, nyiso_lmp_real_time_5_min, spp_lmp_real_time_5_min

## Grid Stress Score Algorithm (5 signals, weighted)
- LMP Deviation Score (30%): (RT_LMP - seasonal_baseline) / seasonal_baseline, where baseline = mean LMP for same hour-of-week over trailing 30 days. Use congestion component if available (2x weight vs energy).
- LMP Velocity Score (15%): % change in LMP over trailing 15 minutes (3 intervals)
- Load vs Forecast Deviation (20%): actual load vs forecast load deviation + load percentile vs historical
- Reserve Margin Score (25%): approximated as (max_gen_72h * 1.05 - current_load) / current_load, mapped to 0-100 with NERC thresholds (>20%=0, <10%=100)
- DART Score (10%): RT_LMP - DA_LMP for current interval as % of DA_LMP

## Routing Algorithm
For a given workload {mw: float, max_latency_ms: int}:
1. Hard filter: exclude regions with stress_score > 75 or estimated latency > max_latency_ms
2. Score remaining: routing_score = 0.40*stress_benefit + 0.30*lmp_score + 0.20*renewable_score + 0.10*dart_bonus
3. User-configurable weights via sliders
4. Output: recommended region + rationale text + grid stress avoided (MW) + cost delta ($/MWh)

## Key Data Center Hubs per ISO
- PJM: WESTERN HUB, DOM HUB (Northern Virginia DC corridor)
- CAISO: NP15, SP15 (Silicon Valley, Southern CA)
- ERCOT: HB_NORTH, HB_HOUSTON, HB_SOUTH (Dallas, Houston, San Antonio)
- MISO: INDIANA.HUB, ILLINOIS.HUB (Chicago)
- ISONE: .H.INTERNAL_HUB (Boston)
- NYISO: N.Y.C., LONGIL (NYC metro)
- SPP: SPPNORTH_HUB, SPPSOUTH_HUB

## Tech Stack
- Streamlit (dashboard)
- gridstatus Python library + direct requests (data fetching)
- Plotly (charts)
- pandas (data processing)
- All in a single app.py file

## Files
- app.py: main Streamlit application
- grid_stress.py: stress score computation module
- router.py: inference routing decision engine
- data_cache.py: ISO data fetching with caching to avoid repeat API calls
- requirements.txt

## Design Requirements
- Dark mode dashboard
- Real-time auto-refresh (every 5 minutes)
- US map showing 3-5 key ISO regions color-coded by stress level
- Live routing decision panel with animated recommendation
- Historical stress chart (last 24 hours per region)
- Sidebar with routing weight sliders (user-configurable)

## Corrected Dataset Names (from GridStatus docs)
IMPORTANT: These are the exact dataset names to use:

### Real-Time 5-Min LMP
- PJM: pjm_lmp_real_time_5_min (hubs: location_type='HUB')
- CAISO: caiso_lmp_real_time_5_min (zones: NP15, SP15, ZP26)
- ERCOT RT hubs: ercot_lmp_by_settlement_point (NOT ercot_lmp_by_bus - hubs are settlement points, location_type='Trading Hub', location names: HB_NORTH, HB_SOUTH, HB_WEST, HB_HOUSTON, HB_BUSAVG)
- MISO RT: miso_lmp_real_time_5_min_ex_ante (hubs: location_type='HUB')
- ISONE RT: isone_lmp_real_time_5_min_prelim (zones: .H.INTERNAL_HUB)
- NYISO RT: nyiso_lmp_real_time_5_min (zones: N.Y.C., LONGIL, CAPITL)
- SPP RT: spp_lmp_real_time_5_min (hubs: location_type='HUB')
- IESO RT: ieso_lmp_real_time_5_min_ontario_zonal (Ontario-wide zonal)

### Day-Ahead Hourly LMP (for DART signal)
- PJM DA: pjm_lmp_day_ahead_hourly
- CAISO DA: caiso_lmp_day_ahead_hourly
- ERCOT DA: ercot_spp_day_ahead_hourly (settlement points, same location names)
- MISO DA: miso_lmp_day_ahead_hourly_ex_ante
- ISONE DA: isone_lmp_day_ahead_hourly
- NYISO DA: nyiso_lmp_day_ahead_hourly
- SPP DA: spp_lmp_day_ahead_hourly

### NYISO Congestion Note
GridStatus FLIPS the sign on NYISO congestion to match other ISOs convention.
LMP = Energy + Congestion + Loss (consistent across all ISOs in GridStatus)

### MISO LMP Type
Use Ex-Ante for real-time (published in near-real-time from SCED).
Ex-Post final is settlement quality but published 3 days later — too late for real-time use.

### ERCOT Congestion Note
ERCOT does not publish LMP components. Congestion can be approximated as:
congestion_proxy = location_lmp - HB_BUSAVG_lmp
