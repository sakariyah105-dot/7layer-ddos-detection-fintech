"""
================================================================
step6_daily_report.py — Daily Report Generator
================================================================
Reads the SQLite database and generates a comprehensive
daily report for each completed day. Run manually or schedule
it to run at midnight each day.

Produces:
    logs/day_N_YYYY-MM-DD_report.txt  — human-readable summary
    logs/day_N_YYYY-MM-DD_data.csv    — full decision log

Run any time:
    python step6_daily_report.py

Or run for a specific date:
    python step6_daily_report.py --date 2026-06-24
================================================================
"""

import os
import sys
import sqlite3
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta


# ── Configuration ──────────────────────────────────────────────
DB_PATH  = "data/decisions.db"
LOG_DIR  = "logs"


def load_day_data(target_date):
    """
    Loads all decisions for a given date from SQLite.
    Returns a pandas DataFrame.
    """
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql_query(
        "SELECT * FROM decisions WHERE date(timestamp) = ?",
        conn,
        params=(target_date,)
    )
    conn.close()
    return df


def compute_metrics(df):
    """
    Computes all classification metrics from a day's decisions.
    Returns a dict of metric name → value.
    """
    if len(df) == 0:
        return {}

    # Treat QUARANTINE + BLOCK as attack prediction
    df['predicted_attack'] = (df['decision'] != 'ALLOW').astype(int)

    tp = ((df['predicted_attack'] == 1) & (df['ground_truth'] == 1)).sum()
    tn = ((df['predicted_attack'] == 0) & (df['ground_truth'] == 0)).sum()
    fp = ((df['predicted_attack'] == 1) & (df['ground_truth'] == 0)).sum()
    fn = ((df['predicted_attack'] == 0) & (df['ground_truth'] == 1)).sum()

    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-8)
    fpr       = fp / max(fp + tn, 1)
    accuracy  = (tp + tn) / max(len(df), 1)

    return {
        'total':          len(df),
        'attacks':        int(df['ground_truth'].sum()),
        'normal':         int((df['ground_truth'] == 0).sum()),
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
        'precision':      round(float(precision), 4),
        'recall':         round(float(recall),    4),
        'f1':             round(float(f1),         4),
        'fpr':            round(float(fpr),        5),
        'accuracy':       round(float(accuracy),   4),
        'allowed':        int((df['decision'] == 'ALLOW').sum()),
        'quarantine':     int((df['decision'] == 'QUARANTINE').sum()),
        'blocked':        int((df['decision'] == 'BLOCK').sum()),
        'avg_latency_ms': round(float(df['latency_ms'].mean()), 2),
        'min_latency_ms': round(float(df['latency_ms'].min()),  2),
        'max_latency_ms': round(float(df['latency_ms'].max()),  2),
        'p99_latency_ms': round(float(df['latency_ms'].quantile(0.99)), 2),
    }


def compute_attack_breakdown(df):
    """
    Computes per-attack-type detection rates.
    Shows how well the model performed on each attack type.
    """
    results = []
    for traffic_type in df['traffic_type'].unique():
        subset = df[df['traffic_type'] == traffic_type]

        if traffic_type == 'normal':
            # For normal traffic: measure false positive rate
            fp  = (subset['decision'] != 'ALLOW').sum()
            fpr = fp / max(len(subset), 1)
            results.append({
                'traffic_type':   traffic_type,
                'total_flows':    len(subset),
                'correctly_handled': int((subset['decision'] == 'ALLOW').sum()),
                'incorrectly_handled': int(fp),
                'rate':           round(float(1 - fpr), 4),
                'metric_name':    'Correct allow rate',
            })
        else:
            # For attacks: measure detection rate
            detected = (subset['decision'] != 'ALLOW').sum()
            det_rate = detected / max(len(subset), 1)
            results.append({
                'traffic_type':   traffic_type,
                'total_flows':    len(subset),
                'correctly_handled': int(detected),
                'incorrectly_handled': int(len(subset) - detected),
                'rate':           round(float(det_rate), 4),
                'metric_name':    'Detection rate',
            })

    return sorted(results, key=lambda x: x['traffic_type'])


def compute_hourly_trend(df):
    """
    Computes F1 and FPR per hour of the day.
    Shows how performance varied throughout the day.
    """
    df = df.copy()
    df['hour']             = pd.to_datetime(df['timestamp']).dt.hour
    df['predicted_attack'] = (df['decision'] != 'ALLOW').astype(int)

    hourly = []
    for hour in sorted(df['hour'].unique()):
        h = df[df['hour'] == hour]
        tp = ((h['predicted_attack']==1) & (h['ground_truth']==1)).sum()
        tn = ((h['predicted_attack']==0) & (h['ground_truth']==0)).sum()
        fp = ((h['predicted_attack']==1) & (h['ground_truth']==0)).sum()
        fn = ((h['predicted_attack']==0) & (h['ground_truth']==1)).sum()
        precision = tp / max(tp+fp, 1)
        recall    = tp / max(tp+fn, 1)
        f1  = 2*precision*recall / max(precision+recall, 1e-8)
        fpr = fp / max(fp+tn, 1)
        hourly.append({
            'hour':    f"{int(hour):02d}:00",
            'flows':   len(h),
            'f1':      round(float(f1),  4),
            'fpr':     round(float(fpr), 5),
            'avg_lat': round(float(h['latency_ms'].mean()), 2),
        })

    return hourly


def write_text_report(target_date, day_number, metrics,
                      attack_breakdown, hourly_trend, output_path):
    """
    Writes a human-readable text report summarising the day's results.
    """
    lines = []
    sep   = "=" * 60

    lines.append(sep)
    lines.append(f"DDoS Defense System — Day {day_number} Report")
    lines.append(f"Date: {target_date}")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(sep)

    # Overall metrics
    lines.append("\nOVERALL METRICS")
    lines.append("-" * 40)
    lines.append(f"  Total flows processed: {metrics['total']:,}")
    lines.append(f"  Attack flows:          {metrics['attacks']:,}")
    lines.append(f"  Normal flows:          {metrics['normal']:,}")
    lines.append(f"")
    lines.append(f"  F1 Score:    {metrics['f1']}")
    lines.append(f"  Precision:   {metrics['precision']}")
    lines.append(f"  Recall:      {metrics['recall']}")
    lines.append(f"  Accuracy:    {metrics['accuracy']}")
    lines.append(f"  FPR:         {metrics['fpr']}")
    lines.append(f"")
    lines.append(f"  True Positives:  {metrics['tp']:,}")
    lines.append(f"  True Negatives:  {metrics['tn']:,}")
    lines.append(f"  False Positives: {metrics['fp']:,}")
    lines.append(f"  False Negatives: {metrics['fn']:,}")

    # Decision breakdown
    lines.append("\nDECISION BREAKDOWN")
    lines.append("-" * 40)
    lines.append(f"  ALLOW:      {metrics['allowed']:,}")
    lines.append(f"  QUARANTINE: {metrics['quarantine']:,}")
    lines.append(f"  BLOCK:      {metrics['blocked']:,}")

    # Latency
    lines.append("\nINFERENCE LATENCY")
    lines.append("-" * 40)
    lines.append(f"  Average: {metrics['avg_latency_ms']} ms")
    lines.append(f"  Min:     {metrics['min_latency_ms']} ms")
    lines.append(f"  Max:     {metrics['max_latency_ms']} ms")
    lines.append(f"  P99:     {metrics['p99_latency_ms']} ms")

    # Per-attack breakdown
    lines.append("\nPER-ATTACK-TYPE PERFORMANCE")
    lines.append("-" * 40)
    lines.append(f"  {'Traffic type':<20} {'Flows':>8} {'Correct':>8} {'Rate':>8}  Metric")
    lines.append(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8}  ------")
    for r in attack_breakdown:
        lines.append(
            f"  {r['traffic_type']:<20} "
            f"{r['total_flows']:>8,} "
            f"{r['correctly_handled']:>8,} "
            f"{r['rate']:>8.4f}  {r['metric_name']}"
        )

    # Hourly trend
    lines.append("\nHOURLY PERFORMANCE TREND")
    lines.append("-" * 40)
    lines.append(f"  {'Hour':<8} {'Flows':>8} {'F1':>8} {'FPR':>10} {'Avg lat':>10}")
    lines.append(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*10}")
    for h in hourly_trend:
        lines.append(
            f"  {h['hour']:<8} "
            f"{h['flows']:>8,} "
            f"{h['f1']:>8.4f} "
            f"{h['fpr']:>10.5f} "
            f"{h['avg_lat']:>8.2f} ms"
        )

    lines.append(f"\n{sep}")
    lines.append(f"End of Day {day_number} Report")
    lines.append(sep)

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"  Report saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate daily DDoS report')
    parser.add_argument('--date', default=None,
                        help='Date to report (YYYY-MM-DD). Default: today.')
    parser.add_argument('--day',  default=1, type=int,
                        help='Day number for filename. Default: 1.')
    args = parser.parse_args()

    target_date = args.date or date.today().isoformat()
    day_number  = args.day

    print("=" * 60)
    print(f"DDoS Defense System — Day {day_number} Report")
    print(f"Date: {target_date}")
    print("=" * 60)

    # Load data
    print("\nLoading data from database...")
    df = load_day_data(target_date)

    if len(df) == 0:
        print(f"No data found for {target_date}")
        print("Make sure step3_engine.py and step4_attacker.py are running.")
        return

    print(f"  Loaded {len(df):,} decisions")

    # Compute metrics
    metrics          = compute_metrics(df)
    attack_breakdown = compute_attack_breakdown(df)
    hourly_trend     = compute_hourly_trend(df)

    # Save CSV
    os.makedirs(LOG_DIR, exist_ok=True)
    csv_path    = os.path.join(LOG_DIR, f"day_{day_number}_{target_date}_data.csv")
    report_path = os.path.join(LOG_DIR, f"day_{day_number}_{target_date}_report.txt")

    df.to_csv(csv_path, index=False)
    print(f"  Data CSV saved: {csv_path}")

    # Write text report
    write_text_report(
        target_date, day_number, metrics,
        attack_breakdown, hourly_trend, report_path
    )

    # Print summary to terminal
    print(f"\n{'='*40}")
    print(f"Day {day_number} Summary")
    print(f"{'='*40}")
    print(f"  Total flows: {metrics['total']:,}")
    print(f"  F1:          {metrics['f1']}")
    print(f"  FPR:         {metrics['fpr']}")
    print(f"  Recall:      {metrics['recall']}")
    print(f"  Avg latency: {metrics['avg_latency_ms']} ms")
    print(f"{'='*40}")
    print(f"\nFiles saved in {LOG_DIR}/")


if __name__ == "__main__":
    main()
