import sqlite3
from datetime import datetime
from flask import Flask, render_template_string, jsonify


# ── Configuration ──────────────────────────────────────────────
DB_PATH         = "data/decisions.db"
DASHBOARD_PORT  = 5000
ENGINE_URL      = "http://localhost:8000"

app = Flask(__name__)


# ── Database queries ───────────────────────────────────────────
def query_db(sql, args=()):
    """Helper: run a SQL query and return all rows."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(sql, args).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── API endpoints for dashboard charts ────────────────────────
@app.route('/api/recent')
def api_recent():
    """Last 50 decisions for the live feed table."""
    rows = query_db("""
        SELECT timestamp, traffic_type, fusion_score,
               decision, ground_truth, correct, latency_ms
        FROM decisions
        ORDER BY id DESC
        LIMIT 50
    """)
    return jsonify(rows)


@app.route('/api/summary')
def api_summary():
    """Today's overall statistics for the summary cards."""
    rows = query_db("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN decision='ALLOW'      THEN 1 ELSE 0 END) as allowed,
            SUM(CASE WHEN decision='QUARANTINE' THEN 1 ELSE 0 END) as quarantine,
            SUM(CASE WHEN decision='BLOCK'      THEN 1 ELSE 0 END) as blocked,
            SUM(CASE WHEN ground_truth=1 AND decision!='ALLOW' THEN 1 ELSE 0 END) as tp,
            SUM(CASE WHEN ground_truth=0 AND decision='ALLOW'  THEN 1 ELSE 0 END) as tn,
            SUM(CASE WHEN ground_truth=0 AND decision!='ALLOW' THEN 1 ELSE 0 END) as fp,
            SUM(CASE WHEN ground_truth=1 AND decision='ALLOW'  THEN 1 ELSE 0 END) as fn,
            AVG(latency_ms) as avg_latency,
            MIN(latency_ms) as min_latency,
            MAX(latency_ms) as max_latency
        FROM decisions
        WHERE date(timestamp) = date('now')
    """)

    if not rows or rows[0]['total'] == 0:
        return jsonify({'total': 0})

    r  = rows[0]
    tp = r['tp'] or 0
    tn = r['tn'] or 0
    fp = r['fp'] or 0
    fn = r['fn'] or 0

    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-8)
    fpr       = fp / max(fp + tn, 1)

    return jsonify({
        'total':      r['total'],
        'allowed':    r['allowed'],
        'quarantine': r['quarantine'],
        'blocked':    r['blocked'],
        'f1':         round(f1,       4),
        'precision':  round(precision,4),
        'recall':     round(recall,   4),
        'fpr':        round(fpr,      5),
        'avg_latency':round(r['avg_latency'] or 0, 2),
        'min_latency':round(r['min_latency'] or 0, 2),
        'max_latency':round(r['max_latency'] or 0, 2),
    })


@app.route('/api/timeline')
def api_timeline():
    """Detection rate and FPR per 5-minute window for line charts."""
    rows = query_db("""
        SELECT
            strftime('%H:%M', timestamp, 'localtime') as window,
            COUNT(*) as total,
            SUM(CASE WHEN ground_truth=1 AND decision!='ALLOW' THEN 1 ELSE 0 END) as tp,
            SUM(CASE WHEN ground_truth=0 AND decision!='ALLOW' THEN 1 ELSE 0 END) as fp,
            SUM(CASE WHEN ground_truth=1 THEN 1 ELSE 0 END) as total_attacks,
            SUM(CASE WHEN ground_truth=0 THEN 1 ELSE 0 END) as total_normal,
            AVG(latency_ms) as avg_latency
        FROM decisions
        WHERE date(timestamp) = date('now')
        GROUP BY strftime('%H:%M', timestamp, 'localtime')
        ORDER BY window DESC
        LIMIT 60
    """)

    # Compute detection rate and FPR per window
    for r in rows:
        r['detection_rate'] = round(
            r['tp'] / max(r['total_attacks'], 1) * 100, 1)
        r['fpr'] = round(
            r['fp'] / max(r['total_normal'], 1), 4)
        r['avg_latency'] = round(r['avg_latency'] or 0, 2)

    rows.reverse()  # oldest first for chart x-axis
    return jsonify(rows)


@app.route('/api/breakdown')
def api_breakdown():
    """Traffic type counts for the pie chart."""
    rows = query_db("""
        SELECT traffic_type, COUNT(*) as count
        FROM decisions
        WHERE date(timestamp) = date('now')
        GROUP BY traffic_type
        ORDER BY count DESC
    """)
    return jsonify(rows)


@app.route('/api/daily_trend')
def api_daily_trend():
    """Per-day metrics for the 5-day trend bar chart."""
    rows = query_db("""
        SELECT
            date(timestamp, 'localtime') as day,
            COUNT(*) as total,
            SUM(CASE WHEN ground_truth=1 AND decision!='ALLOW' THEN 1 ELSE 0 END) as tp,
            SUM(CASE WHEN ground_truth=0 AND decision!='ALLOW' THEN 1 ELSE 0 END) as fp,
            SUM(CASE WHEN ground_truth=1 THEN 1 ELSE 0 END) as total_attacks,
            SUM(CASE WHEN ground_truth=0 THEN 1 ELSE 0 END) as total_normal,
            AVG(latency_ms) as avg_latency
        FROM decisions
        GROUP BY date(timestamp, 'localtime')
        ORDER BY day
    """)

    for r in rows:
        tp = r['tp'] or 0
        fp = r['fp'] or 0
        fn = r['total_attacks'] - tp
        tn = r['total_normal']  - fp
        precision = tp / max(tp + fp, 1)
        recall    = tp / max(tp + fn, 1)
        r['f1']   = round(2*precision*recall / max(precision+recall,1e-8), 4)
        r['fpr']  = round(fp / max(fp + tn, 1), 5)
        r['detection_rate'] = round(tp / max(r['total_attacks'], 1) * 100, 1)
        r['avg_latency'] = round(r['avg_latency'] or 0, 2)

    return jsonify(rows)


# ── Main HTML dashboard ────────────────────────────────────────
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DDoS Defense System — Live Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #0f1117;
            color: #e2e8f0;
            min-height: 100vh;
        }

        /* ── Header ── */
        header {
            background: #1a1f2e;
            border-bottom: 1px solid #2d3748;
            padding: 16px 24px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        header h1 { font-size: 18px; font-weight: 600; color: #fff; }
        header h1 span { color: #63b3ed; }
        #live-indicator {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 13px;
            color: #68d391;
        }
        #live-dot {
            width: 8px; height: 8px;
            border-radius: 50%;
            background: #68d391;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50%       { opacity: 0.4; }
        }

        /* ── Layout ── */
        main {
            padding: 20px 24px;
            display: grid;
            gap: 16px;
            grid-template-columns: repeat(4, 1fr);
        }

        /* ── Cards ── */
        .card {
            background: #1a1f2e;
            border: 1px solid #2d3748;
            border-radius: 12px;
            padding: 16px;
        }
        .card-title {
            font-size: 12px;
            font-weight: 500;
            color: #718096;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 12px;
        }

        /* ── Metric cards ── */
        .metric-card {
            text-align: center;
        }
        .metric-value {
            font-size: 32px;
            font-weight: 700;
            line-height: 1;
            margin-bottom: 4px;
        }
        .metric-label {
            font-size: 12px;
            color: #718096;
        }
        .green  { color: #68d391; }
        .amber  { color: #f6ad55; }
        .red    { color: #fc8181; }
        .blue   { color: #63b3ed; }

        /* ── Grid spans ── */
        .span-4 { grid-column: span 4; }
        .span-2 { grid-column: span 2; }

        /* ── Live feed table ── */
        .feed-table {
            width: 100%;
            font-size: 12px;
            border-collapse: collapse;
        }
        .feed-table th {
            text-align: left;
            padding: 6px 8px;
            color: #718096;
            border-bottom: 1px solid #2d3748;
            font-weight: 500;
        }
        .feed-table td {
            padding: 5px 8px;
            border-bottom: 1px solid #1a1f2e;
            font-family: monospace;
        }
        .feed-table tr:hover { background: #232b3e; }

        .badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 99px;
            font-size: 11px;
            font-weight: 600;
        }
        .badge-allow     { background: #1a3a2a; color: #68d391; }
        .badge-quarantine{ background: #3a2e1a; color: #f6ad55; }
        .badge-block     { background: #3a1a1a; color: #fc8181; }

        /* ── Chart containers ── */
        .chart-container {
            position: relative;
            height: 200px;
        }

        /* ── Status bar ── */
        #status-bar {
            background: #0d111a;
            border-top: 1px solid #2d3748;
            padding: 8px 24px;
            font-size: 12px;
            color: #4a5568;
            display: flex;
            gap: 24px;
        }
    </style>
</head>
<body>

<header>
    <h1>DDoS Defense System — <span>Live Dashboard</span></h1>
    <div id="live-indicator">
        <div id="live-dot"></div>
        <span id="last-update">Connecting...</span>
    </div>
</header>

<main>
    <!-- Row 1: Summary metric cards -->
    <div class="card metric-card">
        <div class="card-title">Total Flows</div>
        <div class="metric-value blue" id="m-total">—</div>
        <div class="metric-label">processed today</div>
    </div>
    <div class="card metric-card">
        <div class="card-title">F1 Score</div>
        <div class="metric-value green" id="m-f1">—</div>
        <div class="metric-label">fusion engine</div>
    </div>
    <div class="card metric-card">
        <div class="card-title">False Positive Rate</div>
        <div class="metric-value" id="m-fpr">—</div>
        <div class="metric-label">normal traffic blocked</div>
    </div>
    <div class="card metric-card">
        <div class="card-title">Avg Latency</div>
        <div class="metric-value blue" id="m-latency">—</div>
        <div class="metric-label">ms per decision</div>
    </div>

    <!-- Row 2: Decision counts -->
    <div class="card metric-card">
        <div class="card-title">Allowed</div>
        <div class="metric-value green" id="m-allowed">—</div>
        <div class="metric-label">legitimate flows</div>
    </div>
    <div class="card metric-card">
        <div class="card-title">Quarantine</div>
        <div class="metric-value amber" id="m-quarantine">—</div>
        <div class="metric-label">borderline flows</div>
    </div>
    <div class="card metric-card">
        <div class="card-title">Blocked</div>
        <div class="metric-value red" id="m-blocked">—</div>
        <div class="metric-label">attack flows</div>
    </div>
    <div class="card metric-card">
        <div class="card-title">Recall</div>
        <div class="metric-value green" id="m-recall">—</div>
        <div class="metric-label">attacks detected</div>
    </div>

    <!-- Row 3: Detection rate + FPR charts -->
    <div class="card span-2">
        <div class="card-title">Detection rate % over time</div>
        <div class="chart-container">
            <canvas id="chart-detection"></canvas>
        </div>
    </div>
    <div class="card span-2">
        <div class="card-title">False positive rate over time</div>
        <div class="chart-container">
            <canvas id="chart-fpr"></canvas>
        </div>
    </div>

    <!-- Row 4: Traffic breakdown + latency -->
    <div class="card span-2">
        <div class="card-title">Traffic type breakdown (today)</div>
        <div class="chart-container">
            <canvas id="chart-breakdown"></canvas>
        </div>
    </div>
    <div class="card span-2">
        <div class="card-title">5-day F1 trend</div>
        <div class="chart-container">
            <canvas id="chart-daily"></canvas>
        </div>
    </div>

    <!-- Row 5: Live decision feed -->
    <div class="card span-4">
        <div class="card-title">Live decision feed (last 50)</div>
        <div style="overflow-x: auto; max-height: 280px; overflow-y: auto;">
            <table class="feed-table">
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Traffic type</th>
                        <th>Fusion score</th>
                        <th>Decision</th>
                        <th>Ground truth</th>
                        <th>Correct</th>
                        <th>Latency (ms)</th>
                    </tr>
                </thead>
                <tbody id="feed-body"></tbody>
            </table>
        </div>
    </div>
</main>

<div id="status-bar">
    <span id="sb-engine">Engine: connecting...</span>
    <span id="sb-flows">Flows today: —</span>
    <span id="sb-uptime">Started: —</span>
</div>

<script>
// ── Chart defaults ──────────────────────────────────────────
Chart.defaults.color          = '#718096';
Chart.defaults.borderColor    = '#2d3748';
Chart.defaults.font.family    = "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
Chart.defaults.font.size      = 11;

// ── Initialise charts ───────────────────────────────────────
const makeLineChart = (id, label, color) => new Chart(
    document.getElementById(id).getContext('2d'), {
        type: 'line',
        data: { labels: [], datasets: [{
            label, data: [],
            borderColor: color, backgroundColor: color + '22',
            borderWidth: 2, pointRadius: 2, fill: true, tension: 0.3,
        }]},
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { grid: { color: '#1a2035' }, ticks: { maxTicksLimit: 8 } },
                y: { grid: { color: '#1a2035' }, min: 0 },
            },
            animation: { duration: 300 },
        }
    }
);

const detectionChart = makeLineChart('chart-detection', 'Detection %', '#68d391');
const fprChart       = makeLineChart('chart-fpr',       'FPR',        '#fc8181');

const breakdownChart = new Chart(
    document.getElementById('chart-breakdown').getContext('2d'), {
        type: 'doughnut',
        data: { labels: [], datasets: [{ data: [],
            backgroundColor: ['#63b3ed','#fc8181','#f6ad55','#9f7aea','#68d391'],
            borderWidth: 0,
        }]},
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { position: 'right', labels: { boxWidth: 12 } } },
        }
    }
);

const dailyChart = new Chart(
    document.getElementById('chart-daily').getContext('2d'), {
        type: 'bar',
        data: { labels: [], datasets: [{
            label: 'F1 Score', data: [],
            backgroundColor: '#63b3ed', borderRadius: 4,
        }]},
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: { min: 0, max: 1, grid: { color: '#1a2035' } },
                x: { grid: { color: '#1a2035' } },
            },
        }
    }
);

// ── Update functions ────────────────────────────────────────
function updateSummary(data) {
    if (!data || data.total === 0) return;

    document.getElementById('m-total').textContent     = data.total.toLocaleString();
    document.getElementById('m-f1').textContent        = data.f1.toFixed(4);
    document.getElementById('m-fpr').textContent       = data.fpr.toFixed(5);
    document.getElementById('m-latency').textContent   = data.avg_latency.toFixed(1) + 'ms';
    document.getElementById('m-allowed').textContent   = data.allowed.toLocaleString();
    document.getElementById('m-quarantine').textContent= data.quarantine.toLocaleString();
    document.getElementById('m-blocked').textContent   = data.blocked.toLocaleString();
    document.getElementById('m-recall').textContent    = data.recall.toFixed(4);

    // Colour FPR red if above 0.01
    const fprEl = document.getElementById('m-fpr');
    fprEl.className = 'metric-value ' + (data.fpr > 0.01 ? 'red' : 'green');
}

function updateTimeline(rows) {
    detectionChart.data.labels   = rows.map(r => r.window);
    detectionChart.data.datasets[0].data = rows.map(r => r.detection_rate);
    detectionChart.update();

    fprChart.data.labels = rows.map(r => r.window);
    fprChart.data.datasets[0].data = rows.map(r => r.fpr * 100);
    fprChart.update();
}

function updateBreakdown(rows) {
    breakdownChart.data.labels = rows.map(r => r.traffic_type);
    breakdownChart.data.datasets[0].data = rows.map(r => r.count);
    breakdownChart.update();
}

function updateDailyTrend(rows) {
    dailyChart.data.labels = rows.map(r => r.day);
    dailyChart.data.datasets[0].data = rows.map(r => r.f1);
    dailyChart.update();
}

function updateFeed(rows) {
    const tbody = document.getElementById('feed-body');
    tbody.innerHTML = rows.map(r => {
        const badgeClass = {
            'ALLOW':      'badge-allow',
            'QUARANTINE': 'badge-quarantine',
            'BLOCK':      'badge-block',
        }[r.decision] || '';

        const time    = r.timestamp ? r.timestamp.slice(11, 19) : '—';
        const correct = r.correct === 1
            ? '<span class="green">✓</span>'
            : '<span class="red">✗</span>';

        return `<tr>
            <td>${time}</td>
            <td>${r.traffic_type}</td>
            <td>${r.fusion_score.toFixed(4)}</td>
            <td><span class="badge ${badgeClass}">${r.decision}</span></td>
            <td>${r.ground_truth === 1 ? 'Attack' : 'Normal'}</td>
            <td>${correct}</td>
            <td>${r.latency_ms.toFixed(1)}</td>
        </tr>`;
    }).join('');
}

// ── Main refresh loop ───────────────────────────────────────
async function refresh() {
    try {
        const [summary, timeline, breakdown, daily, feed] = await Promise.all([
            fetch('/api/summary').then(r => r.json()),
            fetch('/api/timeline').then(r => r.json()),
            fetch('/api/breakdown').then(r => r.json()),
            fetch('/api/daily_trend').then(r => r.json()),
            fetch('/api/recent').then(r => r.json()),
        ]);

        updateSummary(summary);
        updateTimeline(timeline);
        updateBreakdown(breakdown);
        updateDailyTrend(daily);
        updateFeed(feed);

        document.getElementById('last-update').textContent =
            'Live · ' + new Date().toLocaleTimeString();
        document.getElementById('sb-flows').textContent =
            'Flows today: ' + (summary.total || 0).toLocaleString();
        document.getElementById('sb-engine').textContent = 'Engine: connected';

    } catch (err) {
        document.getElementById('last-update').textContent = 'Error — retrying...';
        document.getElementById('sb-engine').textContent = 'Engine: disconnected';
    }
}

// Refresh every 3 seconds
refresh();
setInterval(refresh, 3000);
</script>

</body>
</html>
"""


@app.route('/')
def dashboard():
    return render_template_string(DASHBOARD_HTML)


def main():
    print("=" * 60)
    print("DDoS Defense System — Step 5: Dashboard")
    print("=" * 60)
    print(f"\n  Dashboard: http://localhost:{DASHBOARD_PORT}")
    print(f"  Database:  {DB_PATH}")
    print(f"\n  Open http://localhost:{DASHBOARD_PORT} in your browser.")
    print(f"  Updates every 3 seconds automatically.")
    print("=" * 60)

    app.run(host='0.0.0.0', port=DASHBOARD_PORT, debug=False)


if __name__ == "__main__":
    main()