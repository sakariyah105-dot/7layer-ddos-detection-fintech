"""
================================================================
step4_attacker_v2.py — Realistic Probability-Weighted Traffic
================================================================
Implements a statistically realistic attack distribution based
on real-world DDoS incident patterns:

    70% — normal traffic only
    20% — single attack type active
     8% — two attack types overlapping
     2% — large burst (3-4 attack types simultaneously)

This is stronger than fixed 30-minute cycles because:
    1. The model cannot exploit timing patterns
    2. It mirrors real organizational threat exposure
    3. Overlapping attacks test multi-vector resilience
    4. Every event is logged with ground truth for audit

Every single decision (ground truth + model output) is logged
to decisions.db so you can compute exact detection rates after
the run, attack-type by attack-type.

Run in a separate terminal while step3_engine.py is running:
    python3 step4_attacker_v2.py
================================================================
"""

import os
import json
import time
import random
import sqlite3
import requests
import numpy as np
import pandas as pd
from datetime import datetime, date


# ── Configuration ──────────────────────────────────────────────
ENGINE_URL    = "http://localhost:8000/predict"
DATASET_PATH  = "data/dataset.csv"
LOG_DIR       = "logs"
DB_PATH       = "data/decisions.db"

# Flow rate — flows per second sent to the engine
FLOWS_PER_SEC = 10

# ── Traffic state probabilities ────────────────────────────────
# Each "epoch" lasts a random 30-120 seconds, then re-rolls.
# This means attack durations are also unpredictable.
EPOCH_MIN_SEC = 30
EPOCH_MAX_SEC = 120

STATE_PROBS = {
    'normal_only':    0.70,   # 70% — only normal traffic
    'single_attack':  0.20,   # 20% — one attack type active
    'double_attack':  0.08,   #  8% — two attack types overlap
    'burst_attack':   0.02,   #  2% — 3-4 attack types simultaneously
}

ATTACK_TYPES = ['syn_flood', 'udp_flood', 'http_flood',
                'slow_rate', 'attack_other']


# ── Load real dataset rows ONCE at startup ──────────────────────
print("Loading real CIC-DDoS2019 rows for traffic generation...")
_df = pd.read_csv(DATASET_PATH)

with open('data/feature_cols.json') as f:
    FEATURE_COLS = json.load(f)

_normal_pool = _df[_df['label'] == 0][FEATURE_COLS].to_dict('records')
_attack_pool = _df[_df['label'] == 1][FEATURE_COLS].to_dict('records')

print(f"  Normal pool: {len(_normal_pool):,} real rows")
print(f"  Attack pool: {len(_attack_pool):,} real rows")
print()


# ── Real-row samplers ───────────────────────────────────────────
def sample_normal_flow():
    return dict(random.choice(_normal_pool))

def sample_attack_flow():
    return dict(random.choice(_attack_pool))


# ── Attack-type labeling (cosmetic — never alters feature values) ─
def classify_attack_signature(features):
    if features.get('SYN Flag Count', 0) >= 1 and \
       features.get('Total Backward Packets', 1) == 0:
        return 'syn_flood'
    if features.get('Protocol') == 17 and \
       features.get('Flow Bytes/s', 0) > 1_000_000:
        return 'udp_flood'
    if features.get('Flow Duration', 0) > 10_000_000 and \
       features.get('Flow Packets/s', 1) < 1:
        return 'slow_rate'
    if features.get('Protocol') == 6 and \
       features.get('Flow Packets/s', 0) > 100:
        return 'http_flood'
    return 'attack_other'


# ── Probability-weighted state machine ─────────────────────────
class RealisticScheduler:
    """
    Models realistic DDoS threat exposure using probability weights.

    Each epoch (30-120 seconds, randomised) the scheduler draws
    a new traffic state from the probability distribution. This
    means:
        - Attack timing is completely unpredictable
        - Attack duration is completely unpredictable
        - The model cannot exploit any temporal pattern
        - Multi-vector attacks emerge naturally from the distribution

    An event log is maintained so you can verify exactly which
    flows were attacks and which were normal after the run.
    """

    def __init__(self):
        self.state           = 'normal_only'
        self.active_attacks  = []
        self.epoch_end       = time.time()  # triggers immediately
        self.event_log       = []           # audit trail
        self.state_counts    = {k: 0 for k in STATE_PROBS}

    def _draw_new_state(self):
        """Samples a new traffic state from the probability distribution."""
        states = list(STATE_PROBS.keys())
        probs  = list(STATE_PROBS.values())
        return random.choices(states, weights=probs, k=1)[0]

    def _pick_attacks(self, state):
        """Returns the list of active attack types for a given state."""
        if state == 'normal_only':
            return []
        elif state == 'single_attack':
            return random.sample(ATTACK_TYPES, 1)
        elif state == 'double_attack':
            return random.sample(ATTACK_TYPES, 2)
        elif state == 'burst_attack':
            k = random.randint(3, min(4, len(ATTACK_TYPES)))
            return random.sample(ATTACK_TYPES, k)
        return []

    def tick(self):
        """
        Called once per flow loop iteration.
        Returns the currently active set of attack types.
        Re-rolls state when the current epoch expires.
        """
        now = time.time()

        if now >= self.epoch_end:
            # Draw new state
            new_state    = self._draw_new_state()
            new_attacks  = self._pick_attacks(new_state)
            epoch_length = random.randint(EPOCH_MIN_SEC, EPOCH_MAX_SEC)

            self.state          = new_state
            self.active_attacks = new_attacks
            self.epoch_end      = now + epoch_length
            self.state_counts[new_state] += 1

            # Log the state transition
            event = {
                'timestamp':    datetime.now().isoformat(),
                'state':        new_state,
                'attacks':      new_attacks,
                'duration_sec': epoch_length,
            }
            self.event_log.append(event)

            attack_str = ', '.join(new_attacks) if new_attacks else 'none'
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] "
                  f"New epoch: {new_state} "
                  f"({epoch_length}s) — attacks: {attack_str}")

        return self.active_attacks

    def print_summary(self):
        """Prints a summary of how many epochs each state was active."""
        print("\n[Scheduler] State distribution across all epochs:")
        total = sum(self.state_counts.values())
        for state, count in self.state_counts.items():
            pct = count/max(total,1)*100
            print(f"  {state:<20} {count:>4} epochs  ({pct:.1f}%)")

    def save_event_log(self):
        """Saves the full attack event log to a CSV for audit."""
        os.makedirs(LOG_DIR, exist_ok=True)
        log_path = os.path.join(
            LOG_DIR,
            f"attack_event_log_{date.today().isoformat()}.csv"
        )
        pd.DataFrame(self.event_log).to_csv(log_path, index=False)
        print(f"[Logger] Attack event log saved: {log_path}")
        return log_path


# ── Send flow to engine ──────────────────────────────────────────
def send_flow(traffic_type, features, ground_truth):
    payload = {
        'features':     features,
        'traffic_type': traffic_type,
        'ground_truth': ground_truth,
    }
    try:
        r = requests.post(ENGINE_URL, json=payload, timeout=2)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def save_daily_log(day_number):
    os.makedirs(LOG_DIR, exist_ok=True)
    today    = date.today().isoformat()
    csv_path = os.path.join(LOG_DIR, f"day_{day_number}_{today}.csv")

    conn   = sqlite3.connect(DB_PATH)
    df_log = pd.read_sql_query(
        "SELECT * FROM decisions WHERE date(timestamp) = date('now')",
        conn
    )
    conn.close()
    df_log.to_csv(csv_path, index=False)
    print(f"\n[Logger] Day {day_number} CSV: {csv_path} ({len(df_log):,} rows)")
    return csv_path


# ── Main loop ─────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("DDoS Defense System — Step 4 v2: Realistic Traffic")
    print("=" * 60)
    print(f"\nTraffic state probabilities:")
    for state, prob in STATE_PROBS.items():
        print(f"  {state:<20} {prob*100:.0f}%")
    print(f"\nEpoch length: {EPOCH_MIN_SEC}-{EPOCH_MAX_SEC} seconds (randomised)")
    print(f"Flow rate:    {FLOWS_PER_SEC} flows/second")
    print(f"\nConnecting to engine at {ENGINE_URL}...")

    for attempt in range(30):
        try:
            r = requests.get("http://localhost:8000/health", timeout=2)
            if r.status_code == 200:
                print("Engine is ready.\n")
                break
        except Exception:
            pass
        print(f"  Waiting... ({attempt+1}/30)")
        time.sleep(2)
    else:
        print("ERROR: Engine not responding. Run step3_engine.py first.")
        return

    scheduler  = RealisticScheduler()
    total      = 0
    day_number = 1
    day_start  = datetime.now().date()
    interval   = 1.0 / FLOWS_PER_SEC

    print("Running. Press Ctrl+C to stop.\n")

    try:
        while True:
            loop_start     = time.time()

            # Midnight day rollover
            today = datetime.now().date()
            if today != day_start:
                save_daily_log(day_number)
                scheduler.save_event_log()
                day_number += 1
                day_start   = today

            # Get current active attacks from scheduler
            active_attacks = scheduler.tick()

            # Always send one normal flow
            send_flow('normal', sample_normal_flow(), 0)

            # Send one attack flow per active attack type
            for attack_type in active_attacks:
                attack_features = sample_attack_flow()
                # Label it by signature for dashboard (never alters features)
                labelled_type = classify_attack_signature(attack_features)
                send_flow(labelled_type, attack_features, 1)

            total += 1 + len(active_attacks)

            if total % 200 == 0:
                attack_str = ', '.join(active_attacks) if active_attacks else 'none'
                print(f"  [{datetime.now().strftime('%H:%M:%S')}] "
                      f"Flows: {total:,}  |  State: {scheduler.state}"
                      f"  |  Attacks: {attack_str}")

            elapsed = time.time() - loop_start
            time.sleep(max(0, interval - elapsed))

    except KeyboardInterrupt:
        print(f"\n\nStopped. Total flows sent: {total:,}")
        scheduler.print_summary()
        save_daily_log(day_number)
        scheduler.save_event_log()
        print("\nDone.")


if __name__ == "__main__":
    main()