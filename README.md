# ⚡ GridPulse — Real-Time Grid Stress Monitor & AI Inference Router

**SCSP Hackathon 2026 · Electric Grid Optimization Track**

## Team
- Satwak (AI/ML, Architecture, Full-Stack)
- Sushmita (Grid Domain Expert, Research, Validation)

## Track
Electric Grid Optimization

## What We Built
GridPulse is a real-time grid stress monitoring and AI inference routing
system that helps data center operators shift AI workloads to regions
where the electrical grid is least stressed.

It continuously monitors 7 U.S. electricity markets (ISOs) — PJM, CAISO,
ERCOT, MISO, NYISO, ISONE, and SPP — and scores each region on a
5-signal composite Grid Stress Score (0–100). When a data center's local
grid is under stress, GridPulse recommends routing inference workloads to
a healthier region using a 3-tier routing algorithm that balances grid
stress, energy price, renewable fraction, and latency SLA.

**Key Features:**
- Live 5-signal Grid Stress Score per ISO (LMP Deviation, Price Velocity,
  Load vs Forecast, Reserve Margin, DART Spread)
- 3-tier routing algorithm: hard constraint filter → multi-objective
  scoring → ranked output with rationale
- ERCOT February 2021 crisis replay with historical stress animation
- 12-hour forward stress forecast using Day-Ahead LMP + load forecasts
- Batch workload scheduler for proactive demand shifting

**The Problem It Solves:**
AI data centers are the fastest-growing grid load in the U.S. (130+ GW
by 2030), yet they operate with zero real-time visibility into grid
conditions. GridPulse turns them into grid-aware assets — reducing
infrastructure costs by $16B–$40B, accelerating interconnection, and
creating tens of GW of instant flexible capacity using software, not steel.

## Datasets & APIs Used
| Source | Data Used | Access |
|--------|-----------|--------|
| GridStatus.io API | Real-time LMP, load, fuel mix, load forecast for all 7 ISOs | API key (free tier) |
| PJM DataMiner 2 | Real-time 5-min LMP, reserve margin | Public REST API |
| CAISO OASIS | Real-time interval LMP (NP15, SP15, ZP26) | Public REST API |
| ERCOT Public API | Settlement point prices, fuel mix | Free registration |
| MISO API | Real-time Ex-Ante LMP | Public REST API |
| NYISO API | Real-time 5-min LMP | Public REST API |
| ISONE API | Real-time 5-min preliminary LMP | Public REST API |
| SPP API | Real-time 5-min LMP | Public REST API |
| NERC (hardcoded) | Feb 2021 ERCOT crisis historical data | Public reports |

## How to Run

### Option 1: Live App
👉 [https://gridpulse.streamlit.app](https://gridpulse.streamlit.app)

### Option 2: Run Locally
```bash
# 1. Clone the repo
git clone https://github.com/ryker-code/gridpulse.git
cd gridpulse

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your GridStatus API key
mkdir -p .streamlit
echo 'GRIDSTATUS_API_KEY = "YOUR_KEY_HERE"' > .streamlit/secrets.toml

# 4. Run
streamlit run app.py
```

### Environment
- Python 3.10+
- No other infrastructure required (all APIs are public/free-tier)

## Architecture

```
app.py                  — Streamlit dashboard (UI, routing UI, forecast UI)
grid_stress.py          — 5-signal Grid Stress Score algorithm
router.py               — 3-tier inference routing engine
data_cache.py           — ISO data fetching with in-memory + session-state caching
stress_forecast.py      — 12-hour forward stress from DA LMP + load forecast
ercot_2021_scenario.py  — ERCOT Feb 2021 crisis replay scenario
```

### Grid Stress Score (5 signals, weighted composite)

| Signal | Weight | Description |
|--------|--------|-------------|
| LMP Deviation | 30% | How far current price is from the recent baseline |
| LMP Velocity | 15% | Rate of price change over trailing 15 minutes |
| Load vs Forecast | 20% | Actual load deviation from forecast + 30-day percentile |
| Reserve Margin | 25% | Available headroom vs NERC planning thresholds |
| DART Spread | 10% | Real-time vs day-ahead LMP divergence |

### Routing Algorithm (3 tiers)

1. **Hard filter** — exclude ISOs with stress > 75 or latency > SLA
2. **Multi-objective score** — weighted: stress benefit + LMP score + renewable fraction + DART bonus
3. **Output** — recommended region, routing score, cost delta ($/MWh), grid stress avoided (MW), rationale

### Data Flow

```
7 ISO APIs (GridStatus.io)
│
▼
data_cache.py ← 5-min TTL cache, graceful fallback
│
▼
grid_stress.py ← 5-signal composite score per ISO
│
▼
router.py ← 3-tier routing algorithm
│
▼
app.py (Streamlit) ← Dashboard, map, replay, forecast
```

## Built With Claude Code
This project was built end-to-end using Claude Code (Anthropic) as the
primary development agent during the SCSP Hackathon 2026. Claude Code
generated, iterated, and debugged all backend and frontend code through
natural language prompts.

## Security
- API keys are stored in `.streamlit/secrets.toml` (excluded from git via `.gitignore`)
- No credentials are hardcoded in source files

## License
MIT
