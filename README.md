# DDoS Defense System — 5-Day Real-Time Testing
# ================================================================
# Complete setup and run instructions for Mac
# ================================================================


## FOLDER STRUCTURE

    ddos-system/
    ├── requirements.txt          # Python dependencies
    ├── step1_download.py         # Download and prepare dataset
    ├── step2_train.py            # Train all 4 models
    ├── step3_engine.py           # Real-time fusion engine (keep running)
    ├── step4_attacker.py         # Traffic generator (keep running)
    ├── step5_dashboard.py        # Live web dashboard (keep running)
    ├── step6_daily_report.py     # Generate daily CSV + report
    ├── data/
    │   ├── dataset.csv           # Created by step1
    │   ├── feature_cols.json     # Feature names
    │   └── decisions.db          # SQLite — all decisions stored here
    ├── models/
    │   ├── rf_model.pkl          # Random Forest
    │   ├── dt_model.pkl          # Decision Tree
    │   ├── iso_model.pkl         # Isolation Forest
    │   ├── scaler.pkl            # StandardScaler
    │   └── model_meta.json       # Weights, thresholds, metadata
    └── logs/
        ├── day_1_YYYY-MM-DD_data.csv     # Day 1 full decision log
        ├── day_1_YYYY-MM-DD_report.txt   # Day 1 text summary
        └── ... (days 2-5 same pattern)


## FIRST-TIME SETUP (do this once)

    # 1. Create project folder
    mkdir ~/ddos-system
    cd ~/ddos-system

    # 2. Copy all .py files into this folder

    # 3. Install dependencies
    pip install -r requirements.txt

    # 4. Download dataset and prepare it (takes ~5 min)
    python step1_download.py

    # 5. Train all 4 models (takes ~5-10 min)
    python step2_train.py


## RUNNING THE SYSTEM (do this each day)

    Open 3 separate Terminal windows, all in ~/ddos-system/

    Terminal 1 — Fusion Engine (keep running for all 5 days)
        python step3_engine.py

    Terminal 2 — Traffic Generator (keep running for all 5 days)
        python step4_attacker.py

    Terminal 3 — Dashboard (keep running, open browser)
        python step5_dashboard.py
        Then open: http://localhost:5000


## DAILY REPORT (run once per day or at end)

    # Replace N with the day number (1-5)
    python step6_daily_report.py --day N

    # Or for a specific date:
    python step6_daily_report.py --day 1 --date 2026-06-24


## WHAT YOU WILL SEE

    Dashboard at http://localhost:5000 shows:
        - Live decision feed (last 50 decisions)
        - Detection rate chart (updates every 3 seconds)
        - False positive rate trend
        - Traffic type breakdown pie chart
        - Daily summary: F1, FPR, latency, totals
        - 5-day trend bar chart (builds up over 5 days)

    Terminal 2 (attacker) prints every 100 flows:
        [HH:MM:SS] Flows: 1,000 | Active attacks: syn_flood, http_flood

    Logs folder after 5 days:
        day_1_data.csv    ~50MB  (~864,000 rows)
        day_1_report.txt  ~5KB   human-readable summary
        ...same for days 2-5


## TRAFFIC SCHEDULE

    Each 30-minute cycle (randomised to prevent timing exploitation):
        - Normal traffic:  always active in background
        - SYN flood:       starts at random minute 5-20
        - UDP flood:       starts at random minute 10-25
        - HTTP flood:      starts at random minute 8-22
        - Slow-rate:       starts at random minute 15-28

    Attacks overlap — multiple attack types can run simultaneously.
    The model never learns timing patterns because start times
    are re-randomised each cycle.


## TROUBLESHOOTING

    "Engine not responding"
        → Make sure step3_engine.py is running in Terminal 1
        → Check it shows "Engine running at http://localhost:8000"

    "Model file not found"
        → Run step2_train.py first

    "Dataset not found"
        → Run step1_download.py first

    "No data found for today"
        → Make sure step3_engine.py and step4_attacker.py are both running

    Dashboard shows no data
        → Wait 30 seconds for first flows to be processed
        → Check that step4_attacker.py is running and showing flow counts
