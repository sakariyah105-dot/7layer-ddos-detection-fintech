import sys
sys.path.insert(0, '/Users/hemantsakariya/ddos-system')
from scapy.all import rdpcap, IP, TCP, UDP
import requests
import json
import numpy as np
from collections import defaultdict

PCAP = '/Users/hemantsakariya/ddos-system/data/attack_traffic.pcap'
ENGINE = 'http://localhost:8000/predict'

with open('/Users/hemantsakariya/ddos-system/data/feature_cols.json') as f:
    COLS = json.load(f)

print("Loading pcap...")
pkts = rdpcap(PCAP)
print(f"Total packets: {len(pkts):,}")

# ── Classify packets by attack type ───────────────────────────
syn_pkts   = []
udp_pkts   = []
http_pkts  = []
slow_pkts  = []

for pkt in pkts:
    if not pkt.haslayer(IP):
        continue
    if pkt.haslayer(TCP):
        tcp = pkt[TCP]
        flags = tcp.flags
        # SYN flood: SYN set, no ACK, no payload
        if flags & 0x02 and not flags & 0x10 and len(bytes(tcp.payload)) == 0:
            syn_pkts.append(pkt)
        # HTTP or Slowloris: port 80, has payload
        elif tcp.dport == 80 and len(bytes(tcp.payload)) > 0:
            payload = bytes(tcp.payload).decode('utf-8', errors='ignore')
            if 'GET' in payload or 'POST' in payload:
                if len(bytes(tcp.payload)) < 100:
                    slow_pkts.append(pkt)  # Slowloris: incomplete headers
                else:
                    http_pkts.append(pkt)  # Normal HTTP flood
    elif pkt.haslayer(UDP):
        udp_pkts.append(pkt)

print(f"\nPacket classification:")
print(f"  SYN flood packets:   {len(syn_pkts):,}")
print(f"  UDP flood packets:   {len(udp_pkts):,}")
print(f"  HTTP flood packets:  {len(http_pkts):,}")
print(f"  Slowloris packets:   {len(slow_pkts):,}")

# ── Build flow features per attack type ───────────────────────
def build_flow_features(pkts_subset, label):
    """Build simple flow feature vectors from packet subset."""
    flows = []
    if len(pkts_subset) < 10:
        print(f"  Warning: only {len(pkts_subset)} packets for {label}")
        return flows

    # Group into flows of 50 packets each
    chunk_size = 50
    for i in range(0, min(len(pkts_subset), 5000), chunk_size):
        chunk = pkts_subset[i:i+chunk_size]
        if len(chunk) < 5:
            continue

        # Extract basic stats
        lengths = [len(p) for p in chunk]
        times = [float(p.time) for p in chunk]

        fwd_lengths = lengths[:len(lengths)//2]
        bwd_lengths = lengths[len(lengths)//2:]

        flow_dur = (times[-1] - times[0]) * 1e6 if len(times) > 1 else 1.0
        flow_dur = max(flow_dur, 1.0)

        features = {col: 0.0 for col in COLS}

        # Fill known features
        features['Flow Duration']              = flow_dur
        features['Total Fwd Packets']          = len(fwd_lengths)
        features['Total Backward Packets']     = len(bwd_lengths)
        features['Fwd Packets Length Total']   = sum(fwd_lengths)
        features['Bwd Packets Length Total']   = sum(bwd_lengths)
        features['Fwd Packet Length Max']      = max(fwd_lengths) if fwd_lengths else 0
        features['Fwd Packet Length Min']      = min(fwd_lengths) if fwd_lengths else 0
        features['Fwd Packet Length Mean']     = np.mean(fwd_lengths) if fwd_lengths else 0
        features['Fwd Packet Length Std']      = np.std(fwd_lengths) if fwd_lengths else 0
        features['Bwd Packet Length Max']      = max(bwd_lengths) if bwd_lengths else 0
        features['Bwd Packet Length Min']      = min(bwd_lengths) if bwd_lengths else 0
        features['Bwd Packet Length Mean']     = np.mean(bwd_lengths) if bwd_lengths else 0
        features['Flow Bytes/s']               = sum(lengths) / (flow_dur/1e6)
        features['Flow Packets/s']             = len(chunk) / (flow_dur/1e6)
        features['Packet Length Mean']         = np.mean(lengths)
        features['Packet Length Std']          = np.std(lengths)
        features['Packet Length Min']          = min(lengths)
        features['Packet Length Max']          = max(lengths)
        features['Packet Length Variance']     = np.var(lengths)

        # Protocol
        if chunk[0].haslayer(TCP):
            features['Protocol'] = 6
            tcp = chunk[0][TCP]
            flags = tcp.flags
            features['SYN Flag Count']  = sum(1 for p in chunk if p.haslayer(TCP) and p[TCP].flags & 0x02)
            features['ACK Flag Count']  = sum(1 for p in chunk if p.haslayer(TCP) and p[TCP].flags & 0x10)
            features['FIN Flag Count']  = sum(1 for p in chunk if p.haslayer(TCP) and p[TCP].flags & 0x01)
            features['RST Flag Count']  = sum(1 for p in chunk if p.haslayer(TCP) and p[TCP].flags & 0x04)
        elif chunk[0].haslayer(UDP):
            features['Protocol'] = 17

        # IAT features
        if len(times) > 1:
            iats = [times[i+1]-times[i] for i in range(len(times)-1)]
            iats_us = [x*1e6 for x in iats]
            features['Flow IAT Mean']  = np.mean(iats_us)
            features['Flow IAT Std']   = np.std(iats_us)
            features['Flow IAT Max']   = max(iats_us)
            features['Flow IAT Min']   = min(iats_us)

        flows.append(features)
    return flows

# ── Test each attack type ──────────────────────────────────────
attack_types = [
    ('SYN Flood (hping3)',   syn_pkts),
    ('UDP Flood (hping3)',   udp_pkts),
    ('HTTP Flood (curl)',    http_pkts),
    ('Slowloris',            slow_pkts),
]

print("\n" + "="*60)
print("PER-ATTACK-TYPE DETECTION RESULTS")
print("="*60)

results = {}
for name, pkts_subset in attack_types:
    flows = build_flow_features(pkts_subset, name)
    if not flows:
        print(f"\n{name}: insufficient packets")
        continue

    decisions = defaultdict(int)
    latencies = []

    for feat in flows:
        try:
            r = requests.post(ENGINE,
                json={'features': feat, 'traffic_type': name.lower().replace(' ','_'), 'ground_truth': 1},
                timeout=5)
            if r.status_code == 200:
                d = r.json()
                decisions[d['decision']] += 1
                latencies.append(d.get('latency_ms', 0))
        except:
            pass

    total = sum(decisions.values())
    if total == 0:
        continue

    blocked    = decisions.get('BLOCK', 0) + decisions.get('QUARANTINE', 0) + decisions.get('UNCERTAIN', 0)
    allowed    = decisions.get('ALLOW', 0)
    zero_day   = decisions.get('ZERO_DAY_SUSPECT', 0)
    detection  = blocked / total * 100

    results[name] = {
        'flows_tested': total,
        'detection_rate': detection,
        'blocked': decisions.get('BLOCK', 0),
        'quarantined': decisions.get('QUARANTINE', 0),
        'allowed': allowed,
        'zero_day': zero_day,
        'avg_latency': np.mean(latencies) if latencies else 0
    }

    print(f"\n{name}:")
    print(f"  Flows tested:    {total:,}")
    print(f"  Detection rate:  {detection:.1f}%")
    print(f"  BLOCK:           {decisions.get('BLOCK',0):,}")
    print(f"  QUARANTINE:      {decisions.get('QUARANTINE',0):,}")
    print(f"  ALLOW:           {allowed:,}  ← missed attacks")
    print(f"  ZERO_DAY:        {zero_day:,}")
    print(f"  Avg latency:     {np.mean(latencies):.2f}ms")

print("\n" + "="*60)
print("SUMMARY TABLE")
print("="*60)
print(f"{'Attack Type':<25} {'Flows':>8} {'Detection':>10} {'Missed':>8} {'Latency':>10}")
print("-"*65)
for name, r in results.items():
    print(f"{name:<25} {r['flows_tested']:>8,} {r['detection_rate']:>9.1f}% {r['allowed']:>8,} {r['avg_latency']:>9.2f}ms")

print("\nDone.")
