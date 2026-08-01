import sys
sys.path.insert(0, '/Users/hemantsakariya/ddos-system')
from scapy.all import rdpcap, IP, TCP, UDP
import requests, json, numpy as np
from collections import defaultdict

PCAP   = '/Users/hemantsakariya/ddos-system/data/attack_traffic.pcap'
ENGINE = 'http://localhost:8000/predict'

with open('/Users/hemantsakariya/ddos-system/data/feature_cols.json') as f:
    COLS = json.load(f)

print("Loading pcap...")
pkts = rdpcap(PCAP)
print(f"Total packets: {len(pkts):,}")

# Classify packets
syn_pkts, udp_pkts, http_pkts, slow_pkts = [], [], [], []
for pkt in pkts:
    if not pkt.haslayer(IP): continue
    if pkt.haslayer(TCP):
        tcp = pkt[TCP]
        flags = int(tcp.flags)
        if flags & 0x02 and not flags & 0x10 and len(bytes(tcp.payload)) == 0:
            syn_pkts.append(pkt)
        elif tcp.dport == 80 and len(bytes(tcp.payload)) > 0:
            payload = bytes(tcp.payload).decode('utf-8', errors='ignore')
            if 'GET' in payload or 'POST' in payload:
                if len(bytes(tcp.payload)) < 150:
                    slow_pkts.append(pkt)
                else:
                    http_pkts.append(pkt)
    elif pkt.haslayer(UDP):
        udp_pkts.append(pkt)

print(f"\nPacket counts:")
print(f"  SYN:      {len(syn_pkts):,}")
print(f"  UDP:      {len(udp_pkts):,}")
print(f"  HTTP:     {len(http_pkts):,}")
print(f"  Slowloris:{len(slow_pkts):,}")

def build_flows(pkts_subset, chunk_size=20, max_flows=200):
    flows = []
    subset = pkts_subset[:chunk_size*max_flows]
    for i in range(0, len(subset), chunk_size):
        chunk = subset[i:i+chunk_size]
        if len(chunk) < 3: continue
        lengths = [len(p) for p in chunk]
        times   = [float(p.time) for p in chunk]
        fwd = lengths[:len(lengths)//2] or [0]
        bwd = lengths[len(lengths)//2:] or [0]
        dur = max((times[-1]-times[0])*1e6, 1.0)

        feat = {c: 0.0 for c in COLS}
        feat['Flow Duration']           = dur
        feat['Total Fwd Packets']       = len(fwd)
        feat['Total Backward Packets']  = len(bwd)
        feat['Fwd Packets Length Total']= sum(fwd)
        feat['Bwd Packets Length Total']= sum(bwd)
        feat['Fwd Packet Length Max']   = max(fwd)
        feat['Fwd Packet Length Min']   = min(fwd)
        feat['Fwd Packet Length Mean']  = float(np.mean(fwd))
        feat['Fwd Packet Length Std']   = float(np.std(fwd))
        feat['Bwd Packet Length Max']   = max(bwd)
        feat['Bwd Packet Length Min']   = min(bwd)
        feat['Bwd Packet Length Mean']  = float(np.mean(bwd))
        feat['Packet Length Mean']      = float(np.mean(lengths))
        feat['Packet Length Std']       = float(np.std(lengths))
        feat['Packet Length Min']       = min(lengths)
        feat['Packet Length Max']       = max(lengths)
        feat['Packet Length Variance']  = float(np.var(lengths))
        feat['Flow Bytes/s']            = sum(lengths)/(dur/1e6)
        feat['Flow Packets/s']          = len(chunk)/(dur/1e6)
        feat['Avg Packet Size']         = float(np.mean(lengths))
        feat['Avg Fwd Segment Size']    = float(np.mean(fwd))
        feat['Avg Bwd Segment Size']    = float(np.mean(bwd))

        if chunk[0].haslayer(TCP):
            feat['Protocol'] = 6.0
            syn  = sum(1 for p in chunk if p.haslayer(TCP) and int(p[TCP].flags)&0x02)
            ack  = sum(1 for p in chunk if p.haslayer(TCP) and int(p[TCP].flags)&0x10)
            fin  = sum(1 for p in chunk if p.haslayer(TCP) and int(p[TCP].flags)&0x01)
            rst  = sum(1 for p in chunk if p.haslayer(TCP) and int(p[TCP].flags)&0x04)
            feat['SYN Flag Count'] = syn
            feat['ACK Flag Count'] = ack
            feat['FIN Flag Count'] = fin
            feat['RST Flag Count'] = rst
        elif chunk[0].haslayer(UDP):
            feat['Protocol'] = 17.0

        if len(times) > 1:
            iats = [(times[j+1]-times[j])*1e6 for j in range(len(times)-1)]
            feat['Flow IAT Mean'] = float(np.mean(iats))
            feat['Flow IAT Std']  = float(np.std(iats))
            feat['Flow IAT Max']  = float(max(iats))
            feat['Flow IAT Min']  = float(min(iats))
            feat['Fwd IAT Total'] = float(sum(iats))
            feat['Bwd IAT Total'] = 0.0

        flows.append(feat)
    return flows

def test_flows(flows, attack_name):
    decisions = defaultdict(int)
    latencies = []
    for feat in flows:
        try:
            r = requests.post(ENGINE,
                json={'features': feat, 'traffic_type': attack_name, 'ground_truth': 1},
                timeout=5)
            if r.status_code == 200:
                d = r.json()
                decisions[d['decision']] += 1
                latencies.append(d.get('latency_ms', 0))
        except Exception as e:
            pass
    return decisions, latencies

print("\n" + "="*60)
print("PER-ATTACK-TYPE DETECTION RESULTS")
print("="*60)

attack_types = [
    ('SYN Flood',  syn_pkts,  20),
    ('UDP Flood',  udp_pkts,  20),
    ('HTTP Flood', http_pkts, 5),
    ('Slowloris',  slow_pkts, 5),
]

results = {}
for name, pkts_sub, chunk in attack_types:
    print(f"\nTesting {name}...")
    flows = build_flows(pkts_sub, chunk_size=chunk, max_flows=200)
    print(f"  Built {len(flows)} flows from {len(pkts_sub):,} packets")
    if not flows:
        print(f"  SKIP — no flows built")
        continue
    decisions, latencies = test_flows(flows, name)
    total = sum(decisions.values())
    if total == 0:
        print(f"  SKIP — no decisions returned")
        continue
    blocked   = decisions.get('BLOCK',0) + decisions.get('QUARANTINE',0) + decisions.get('UNCERTAIN',0)
    allowed   = decisions.get('ALLOW',0)
    zeroday   = decisions.get('ZERO_DAY_SUSPECT',0)
    detection = blocked/total*100
    results[name] = {
        'flows': total, 'detection': detection,
        'block': decisions.get('BLOCK',0),
        'quarantine': decisions.get('QUARANTINE',0),
        'allow': allowed, 'zero_day': zeroday,
        'latency': float(np.mean(latencies)) if latencies else 0
    }
    print(f"  Total flows:    {total}")
    print(f"  Detection rate: {detection:.1f}%")
    print(f"  BLOCK:          {decisions.get('BLOCK',0)}")
    print(f"  QUARANTINE:     {decisions.get('QUARANTINE',0)}")
    print(f"  ALLOW (missed): {allowed}")
    print(f"  ZERO_DAY:       {zeroday}")
    print(f"  Avg latency:    {float(np.mean(latencies)):.2f}ms")

print("\n" + "="*60)
print("SUMMARY TABLE")
print("="*60)
print(f"{'Attack Type':<15} {'Flows':>7} {'Detection':>10} {'BLOCK':>7} {'QUAR':>6} {'ALLOW':>7} {'ZeroDay':>8} {'Latency':>9}")
print("-"*75)
for name, r in results.items():
    print(f"{name:<15} {r['flows']:>7} {r['detection']:>9.1f}% {r['block']:>7} {r['quarantine']:>6} {r['allow']:>7} {r['zero_day']:>8} {r['latency']:>8.2f}ms")
print("\nDone.")
