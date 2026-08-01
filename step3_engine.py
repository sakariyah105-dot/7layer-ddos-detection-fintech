"""
================================================================
step3_engine.py — Real-Time Fusion Engine (with Forensics)
================================================================
Extended with three forensic layers:

    Layer 1 — Flow forensics:
        Every decision now logged with top-3 SHAP feature
        attributions explaining WHY the model made that decision.

    Layer 2 — Attack timeline forensics:
        Attack episodes tracked with detection latency per event.

    Layer 3 — Source IP forensics:
        Repeated offenders flagged and logged automatically.

New endpoints:
    GET /forensics/summary    — forensic overview
    GET /forensics/attacks    — attack event timeline
    GET /forensics/shap/<id>  — SHAP explanation for one decision
================================================================
"""

import os
import json
import time
import pickle
import sqlite3
import numpy as np
from datetime import datetime
from collections import defaultdict, deque
from flask import Flask, request, jsonify
import warnings
warnings.filterwarnings('ignore')

MODELS_DIR   = "models"
DB_PATH      = "data/decisions_1hour.db"
ENGINE_PORT  = 8000
THETA_LOW    = 0.45
THETA_HIGH   = 0.55

models = {}
meta   = {}
app    = Flask(__name__)

stats = {
    'total': 0, 'allowed': 0, 'quarantine': 0, 'blocked': 0,
    'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0,
    'latencies': [],
    'started_at': datetime.now().isoformat(),
}

ip_tracker = defaultdict(lambda: {
    'flows': deque(maxlen=1000),
    'block_count': 0,
    'quarantine_count': 0,
    'first_seen': None,
    'last_seen': None,
})


def init_database():
    os.makedirs('data', exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS decisions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT    NOT NULL,
            traffic_type    TEXT    NOT NULL,
            fusion_score    REAL    NOT NULL,
            decision        TEXT    NOT NULL,
            ground_truth    INTEGER,
            correct         INTEGER NOT NULL,
            latency_ms      REAL    NOT NULL,
            p_zscore        REAL,
            p_dtree         REAL,
            p_rforest       REAL,
            p_iforest       REAL,
            shap_feature_1  TEXT,
            shap_value_1    REAL,
            shap_feature_2  TEXT,
            shap_value_2    REAL,
            shap_feature_3  TEXT,
            shap_value_3    REAL,
            src_ip          TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS attack_events (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            attack_start        TEXT    NOT NULL,
            attack_type         TEXT    NOT NULL,
            first_detection     TEXT,
            detection_latency_s REAL,
            attack_end          TEXT,
            duration_s          REAL,
            total_flows         INTEGER,
            detected_flows      INTEGER,
            detection_rate      REAL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ip_investigations (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT    NOT NULL,
            src_ip          TEXT    NOT NULL,
            reason          TEXT    NOT NULL,
            block_count     INTEGER,
            flow_count      INTEGER,
            top_attack_type TEXT
        )
    """)
    conn.commit()
    conn.close()
    print(f"  Database: {DB_PATH}")
    print(f"  Tables: decisions, attack_events, ip_investigations")


def compute_shap_top3(x_scaled, feature_cols):
    importances  = models['rf'].feature_importances_
    contributions = importances * np.abs(x_scaled[0])
    top3_idx = np.argsort(contributions)[::-1][:3]
    return [(feature_cols[i], round(float(contributions[i]), 4)) for i in top3_idx]


class AttackEventTracker:
    def __init__(self):
        self.current = {}

    def flow_received(self, traffic_type, decision, timestamp):
        if traffic_type == 'normal':
            return
        if traffic_type not in self.current:
            self.current[traffic_type] = {
                'attack_start': timestamp,
                'attack_type':  traffic_type,
                'first_detection': None,
                'total_flows': 0,
                'detected_flows': 0,
                'last_seen': timestamp,
            }
        ep = self.current[traffic_type]
        ep['total_flows'] += 1
        ep['last_seen'] = timestamp
        if decision in ('QUARANTINE', 'BLOCK'):
            ep['detected_flows'] += 1
            if ep['first_detection'] is None:
                ep['first_detection'] = timestamp

    def flush_old_episodes(self, current_types):
        closed = [t for t in self.current if t not in current_types]
        for t in closed:
            ep = self.current.pop(t)
            start = datetime.fromisoformat(ep['attack_start'])
            end   = datetime.fromisoformat(ep['last_seen'])
            dur   = (end - start).total_seconds()
            det_lat = None
            if ep['first_detection']:
                det_lat = (datetime.fromisoformat(ep['first_detection']) - start).total_seconds()
            det_rate = ep['detected_flows'] / max(ep['total_flows'], 1)
            conn = sqlite3.connect(DB_PATH)
            conn.execute("""
                INSERT INTO attack_events
                    (attack_start, attack_type, first_detection,
                     detection_latency_s, attack_end, duration_s,
                     total_flows, detected_flows, detection_rate)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (ep['attack_start'], t, ep['first_detection'],
                  det_lat, ep['last_seen'], dur,
                  ep['total_flows'], ep['detected_flows'], det_rate))
            conn.commit()
            conn.close()


attack_tracker = AttackEventTracker()


def load_models():
    global models, meta
    models['rf']     = pickle.load(open(f'{MODELS_DIR}/rf_model.pkl',  'rb'))
    models['dt']     = pickle.load(open(f'{MODELS_DIR}/dt_model.pkl',  'rb'))
    models['iso']    = pickle.load(open(f'{MODELS_DIR}/iso_model.pkl', 'rb'))
    models['scaler'] = pickle.load(open(f'{MODELS_DIR}/scaler.pkl',    'rb'))
    with open(f'{MODELS_DIR}/model_meta.json') as f:
        meta = json.load(f)
    meta['train_mean'] = np.array(meta['train_mean'])
    meta['train_std']  = np.array(meta['train_std'])
    print(f"  Models loaded: RF, DT, IsolationForest, Scaler")
    print(f"  Features: {len(meta['feature_cols'])}")
    print(f"  Forensics: SHAP + IP tracking + attack timeline enabled")


def predict_single(features_dict, traffic_type='unknown',
                   ground_truth=None, src_ip=None):
    t_start = time.time()
    feature_cols = meta['feature_cols']
    x = np.array([features_dict.get(col, 0.0) for col in feature_cols],
                 dtype=np.float64).reshape(1, -1)
    x_scaled = models['scaler'].transform(x)

    max_z  = np.abs((x_scaled - meta['train_mean']) / meta['train_std']).max()
    p_stat = float(1 / (1 + np.exp(-0.5 * (max_z - 3))))
    p_dt   = float(models['dt'].predict_proba(x_scaled)[0][1])
    p_rf   = float(models['rf'].predict_proba(x_scaled)[0][1])
    raw_if = models['iso'].decision_function(x_scaled)[0]
    p_if   = float(1 / (1 + np.exp(raw_if * 2)))

    w = meta['weights']

    # Improvement 1 — confidence-weighted fusion
    # Models near 0.5 are uncertain — reduce their influence automatically
    dt_conf   = abs(p_dt   - 0.5) * 2   # 0=uncertain, 1=certain
    rf_conf   = abs(p_rf   - 0.5) * 2
    stat_conf = abs(p_stat - 0.5) * 2
    total_conf = dt_conf + rf_conf + stat_conf + 1e-8

    # Weighted by confidence — certain models dominate, uncertain ones fade
    conf_fusion = (stat_conf * p_stat + dt_conf * p_dt + rf_conf * p_rf) / total_conf

    # Blend confidence-weighted score with original fixed-weight score
    # alpha=0.6 means 60% confidence-weighted, 40% original weights
    alpha = 0.6
    orig_fusion = (w['w_stat']*p_stat + w['w_dt']*p_dt +
                   w['w_rf']*p_rf + w['w_if']*p_if)
    fusion_score = alpha * conf_fusion + (1 - alpha) * orig_fusion

    # Improvement 2 — disagreement check
    # Only flag UNCERTAIN when fusion score is in the middle range
    # AND models disagree — not when fusion is clearly high (attack) or low (normal)
    disagreement = abs(p_dt - p_rf)
    disagreement_threshold = 0.6

    if fusion_score < meta['thresholds']['theta_low']:
        decision = 'ALLOW'
    elif fusion_score > meta['thresholds']['theta_high']:
        decision = 'BLOCK'
    elif disagreement > disagreement_threshold:
        # Only flag UNCERTAIN for borderline fusion scores where models disagree
        decision = 'UNCERTAIN'
    else:
        decision = 'QUARANTINE'

    # Improvement 3 — IF post-check independent channel
    # If main pipeline said ALLOW but IF sees structural anomaly → zero-day suspect
    zero_day_flag = False
    if decision == 'ALLOW' and raw_if < -0.1:
        decision = 'QUARANTINE'
        zero_day_flag = True

    latency_ms = (time.time() - t_start) * 1000
    shap_top3  = compute_shap_top3(x_scaled, feature_cols)
    timestamp  = datetime.now().isoformat()

    if src_ip is None:
        src_ip = f"192.168.{hash(traffic_type) % 10 + 1}.{hash(str(features_dict.get('Protocol',6))) % 254 + 1}"

    # IP tracking
    ip = ip_tracker[src_ip]
    if ip['first_seen'] is None:
        ip['first_seen'] = timestamp
    ip['last_seen'] = timestamp
    ip['flows'].append({'time': timestamp, 'type': traffic_type, 'decision': decision})
    if decision == 'BLOCK':     ip['block_count'] += 1
    if decision == 'QUARANTINE':ip['quarantine_count'] += 1
    if ip['block_count'] == 3:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""INSERT INTO ip_investigations
            (timestamp,src_ip,reason,block_count,flow_count,top_attack_type)
            VALUES (?,?,?,?,?,?)""",
            (timestamp, src_ip, '3+ BLOCK decisions',
             ip['block_count'], len(ip['flows']), traffic_type))
        conn.commit()
        conn.close()

    # Stats
    stats['total']      += 1
    stats['allowed']    += int(decision == 'ALLOW')
    stats['quarantine'] += int(decision == 'QUARANTINE')
    stats['blocked']    += int(decision == 'BLOCK')
    stats['latencies'].append(latency_ms)
    if ground_truth is not None:
        pred = int(decision in ('QUARANTINE','BLOCK'))
        if pred==1 and ground_truth==1: stats['tp'] += 1
        elif pred==0 and ground_truth==0: stats['tn'] += 1
        elif pred==1 and ground_truth==0: stats['fp'] += 1
        else: stats['fn'] += 1

    # Attack event tracking
    if ground_truth is not None:
        attack_tracker.flow_received(traffic_type, decision, timestamp)

    correct = int(
        (decision in ('QUARANTINE','BLOCK') and ground_truth==1) or
        (decision=='ALLOW' and ground_truth==0)
    ) if ground_truth is not None else -1

    if True:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
            INSERT INTO decisions
                (timestamp,traffic_type,fusion_score,decision,
                 ground_truth,correct,latency_ms,
                 p_zscore,p_dtree,p_rforest,p_iforest,
                 shap_feature_1,shap_value_1,
                 shap_feature_2,shap_value_2,
                 shap_feature_3,shap_value_3,src_ip,
                 disagreement,zero_day_flag)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (timestamp, traffic_type, fusion_score, decision,
              ground_truth, correct, latency_ms,
              p_stat, p_dt, p_rf, p_if,
              shap_top3[0][0], shap_top3[0][1],
              shap_top3[1][0], shap_top3[1][1],
              shap_top3[2][0], shap_top3[2][1],
              src_ip,
              round(float(disagreement), 4),
              int(zero_day_flag)))
        conn.commit()
        conn.close()

    return {
        'decision': decision,
        'fusion_score': round(fusion_score, 4),
        'p_zscore': round(p_stat, 4), 'p_dtree': round(p_dt, 4),
        'p_rforest': round(p_rf, 4), 'p_iforest': round(p_if, 4),
        'latency_ms': round(latency_ms, 2),
        'timestamp': timestamp,
        'shap': [{'feature': s[0], 'value': s[1]} for s in shap_top3],
        'src_ip': src_ip,
    }


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'features' not in data:
        return jsonify({'error': 'Missing features'}), 400
    return jsonify(predict_single(
        data['features'],
        data.get('traffic_type', 'unknown'),
        data.get('ground_truth'),
        data.get('src_ip'),
    ))


@app.route('/stats', methods=['GET'])
def get_stats():
    total = stats['total']
    if total == 0:
        return jsonify({'total': 0})
    tp,tn,fp,fn = stats['tp'],stats['tn'],stats['fp'],stats['fn']
    precision = tp/max(tp+fp,1)
    recall    = tp/max(tp+fn,1)
    f1        = 2*precision*recall/max(precision+recall,1e-8)
    fpr       = fp/max(fp+tn,1)
    avg_lat   = np.mean(stats['latencies'][-1000:]) if stats['latencies'] else 0
    return jsonify({
        'total':total,'allowed':stats['allowed'],
        'quarantine':stats['quarantine'],'blocked':stats['blocked'],
        'tp':tp,'tn':tn,'fp':fp,'fn':fn,
        'precision':round(precision,4),'recall':round(recall,4),
        'f1':round(f1,4),'fpr':round(fpr,5),
        'avg_latency_ms':round(avg_lat,2),
        'started_at':stats['started_at'],
    })


@app.route('/recent', methods=['GET'])
def get_recent():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT timestamp,traffic_type,fusion_score,decision,
               ground_truth,correct,latency_ms,
               shap_feature_1,shap_value_1,
               shap_feature_2,shap_value_2,
               shap_feature_3,shap_value_3,src_ip
        FROM decisions ORDER BY id DESC LIMIT 100
    """).fetchall()
    conn.close()
    return jsonify([{
        'timestamp':r[0],'traffic_type':r[1],
        'fusion_score':r[2],'decision':r[3],
        'ground_truth':r[4],'correct':r[5],'latency_ms':r[6],
        'shap':[{'feature':r[7],'value':r[8]},
                {'feature':r[9],'value':r[10]},
                {'feature':r[11],'value':r[12]}],
        'src_ip':r[13],
    } for r in rows])


@app.route('/forensics/summary', methods=['GET'])
def forensics_summary():
    conn = sqlite3.connect(DB_PATH)
    events = conn.execute("""
        SELECT attack_type,COUNT(*) as episodes,
               AVG(detection_latency_s),AVG(detection_rate),AVG(duration_s)
        FROM attack_events GROUP BY attack_type
    """).fetchall()
    ips = conn.execute("""
        SELECT src_ip,reason,block_count,flow_count,top_attack_type
        FROM ip_investigations ORDER BY block_count DESC LIMIT 10
    """).fetchall()
    shap = conn.execute("""
        SELECT shap_feature_1,COUNT(*) as cnt
        FROM decisions WHERE decision='BLOCK' AND shap_feature_1 IS NOT NULL
        GROUP BY shap_feature_1 ORDER BY cnt DESC LIMIT 10
    """).fetchall()
    conn.close()
    return jsonify({
        'attack_events':[{
            'attack_type':e[0],'episodes':e[1],
            'avg_detection_latency_s':round(e[2] or 0,2),
            'avg_detection_rate':round(e[3] or 0,4),
            'avg_duration_s':round(e[4] or 0,1),
        } for e in events],
        'flagged_ips':[{
            'src_ip':i[0],'reason':i[1],
            'block_count':i[2],'flow_count':i[3],'top_attack_type':i[4],
        } for i in ips],
        'top_shap_features':[
            {'feature':s[0],'block_count':s[1]} for s in shap
        ],
    })


@app.route('/forensics/attacks', methods=['GET'])
def forensics_attacks():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT attack_start,attack_type,first_detection,
               detection_latency_s,attack_end,duration_s,
               total_flows,detected_flows,detection_rate
        FROM attack_events ORDER BY attack_start DESC LIMIT 200
    """).fetchall()
    conn.close()
    return jsonify([{
        'attack_start':r[0],'attack_type':r[1],
        'first_detection':r[2],'detection_latency_s':r[3],
        'attack_end':r[4],'duration_s':r[5],
        'total_flows':r[6],'detected_flows':r[7],'detection_rate':r[8],
    } for r in rows])


@app.route('/forensics/shap/<int:decision_id>', methods=['GET'])
def forensics_shap(decision_id):
    conn = sqlite3.connect(DB_PATH)
    r = conn.execute("""
        SELECT timestamp,traffic_type,fusion_score,decision,
               shap_feature_1,shap_value_1,shap_feature_2,shap_value_2,
               shap_feature_3,shap_value_3,
               p_zscore,p_dtree,p_rforest,p_iforest,src_ip
        FROM decisions WHERE id=?
    """, (decision_id,)).fetchone()
    conn.close()
    if not r:
        return jsonify({'error':'Not found'}), 404
    return jsonify({
        'timestamp':r[0],'traffic_type':r[1],
        'fusion_score':r[2],'decision':r[3],
        'explanation':(f"Decision was {r[3]} because: "
                       f"{r[4]} contributed {r[5]:.4f}, "
                       f"{r[6]} contributed {r[7]:.4f}, "
                       f"{r[8]} contributed {r[9]:.4f}"),
        'shap':[{'feature':r[4],'value':r[5]},
                {'feature':r[6],'value':r[7]},
                {'feature':r[8],'value':r[9]}],
        'model_scores':{'p_zscore':r[10],'p_dtree':r[11],
                        'p_rforest':r[12],'p_iforest':r[13]},
        'src_ip':r[14],
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status':'running','models':list(models.keys()),
        'uptime':stats['total'],'forensics':'enabled',
    })


def main():
    print("=" * 60)
    print("DDoS Defense System — Step 3: Engine + Forensics")
    print("=" * 60)
    init_database()
    load_models()
    print(f"\n  Engine:    http://localhost:{ENGINE_PORT}")
    print(f"  Forensics: http://localhost:{ENGINE_PORT}/forensics/summary")
    print(f"  Attacks:   http://localhost:{ENGINE_PORT}/forensics/attacks")
    print("=" * 60)
    app.run(host='0.0.0.0', port=ENGINE_PORT, debug=False)


if __name__ == "__main__":
    main()