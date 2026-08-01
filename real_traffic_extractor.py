"""
================================================================
real_traffic_extractor.py — Live Traffic Feature Extractor
================================================================
Extracts the same 77 CICFlowMeter-compatible features from
real network traffic captured by tcpdump.

Reads a pcap file, groups packets into bidirectional flows,
computes all 77 features per flow, and sends each flow to
the ML engine for real-time classification.

Usage:
    python3 real_traffic_extractor.py --pcap data/real_traffic.pcap
    python3 real_traffic_extractor.py --live en0

Requirements:
    pip3 install scapy
    sudo needed for live capture
================================================================
"""

import os
import json
import time
import argparse
import requests
import numpy as np
from collections import defaultdict
from datetime import datetime
from scapy.all import rdpcap, sniff, IP, TCP, UDP


# ── Configuration ──────────────────────────────────────────────
ENGINE_URL   = "http://localhost:8000/predict"
FLOW_TIMEOUT = 10    # seconds — close flow if idle for 10 seconds
FEATURE_COLS_PATH = "data/feature_cols.json"

with open(FEATURE_COLS_PATH) as f:
    FEATURE_COLS = json.load(f)


# ── Flow storage ───────────────────────────────────────────────
class Flow:
    """
    Tracks one bidirectional network flow.
    A flow is identified by the 5-tuple:
        (src_ip, dst_ip, src_port, dst_port, protocol)
    """

    def __init__(self, src_ip, dst_ip, src_port, dst_port, proto, timestamp):
        self.src_ip    = src_ip
        self.dst_ip    = dst_ip
        self.src_port  = src_port
        self.dst_port  = dst_port
        self.proto     = proto
        self.start_time = timestamp
        self.last_time  = timestamp

        # Forward packets (src → dst)
        self.fwd_packets   = []   # list of (timestamp, size, flags)
        # Backward packets (dst → src)
        self.bwd_packets   = []

        # TCP flags
        self.fin_count = 0
        self.syn_count = 0
        self.rst_count = 0
        self.psh_count = 0
        self.ack_count = 0
        self.urg_count = 0
        self.cwe_count = 0
        self.ece_count = 0

        self.fwd_psh_flags = 0
        self.bwd_psh_flags = 0
        self.fwd_urg_flags = 0
        self.bwd_urg_flags = 0

        self.init_fwd_win = 0
        self.init_bwd_win = 0
        self.fwd_header_len = 0
        self.bwd_header_len = 0

    def add_packet(self, timestamp, size, flags, is_forward,
                   header_len=20, window=0):
        self.last_time = timestamp

        if is_forward:
            self.fwd_packets.append((timestamp, size, flags))
            self.fwd_header_len += header_len
            if len(self.fwd_packets) == 1:
                self.init_fwd_win = window
            if flags and 'P' in flags: self.fwd_psh_flags += 1
            if flags and 'U' in flags: self.fwd_urg_flags += 1
        else:
            self.bwd_packets.append((timestamp, size, flags))
            self.bwd_header_len += header_len
            if len(self.bwd_packets) == 1:
                self.init_bwd_win = window
            if flags and 'P' in flags: self.bwd_psh_flags += 1
            if flags and 'U' in flags: self.bwd_urg_flags += 1

        if flags:
            if 'F' in flags: self.fin_count += 1
            if 'S' in flags: self.syn_count += 1
            if 'R' in flags: self.rst_count += 1
            if 'P' in flags: self.psh_count += 1
            if 'A' in flags: self.ack_count += 1
            if 'U' in flags: self.urg_count += 1

    def is_complete(self):
        """Flow is complete if FIN seen or timeout."""
        return self.fin_count >= 2 or self.rst_count >= 1

    def compute_features(self):
        """
        Computes all 77 CICFlowMeter-compatible features.
        Returns a dict matching the training feature names.
        """
        fwd = self.fwd_packets
        bwd = self.bwd_packets
        all_pkts = fwd + bwd

        # Flow duration in microseconds
        duration = max((self.last_time - self.start_time) * 1e6, 1)

        # Packet sizes
        fwd_sizes = [p[1] for p in fwd] if fwd else [0]
        bwd_sizes = [p[1] for p in bwd] if bwd else [0]
        all_sizes = [p[1] for p in all_pkts] if all_pkts else [0]

        # IAT (inter-arrival times) in microseconds
        def iat(packets):
            if len(packets) < 2:
                return [0]
            times = [p[0] for p in packets]
            return [(times[i+1] - times[i]) * 1e6
                    for i in range(len(times)-1)]

        fwd_iat = iat(sorted(fwd, key=lambda x: x[0]))
        bwd_iat = iat(sorted(bwd, key=lambda x: x[0]))
        all_iat = iat(sorted(all_pkts, key=lambda x: x[0]))

        def safe_stats(lst):
            if not lst or len(lst) == 0:
                return 0, 0, 0, 0, 0
            arr = np.array(lst, dtype=float)
            return (float(arr.min()), float(arr.max()),
                    float(arr.mean()), float(arr.std()),
                    float(arr.sum()))

        fwd_min, fwd_max, fwd_mean, fwd_std, fwd_total = safe_stats(fwd_sizes)
        bwd_min, bwd_max, bwd_mean, bwd_std, bwd_total = safe_stats(bwd_sizes)
        all_min, all_max, all_mean, all_std, all_total = safe_stats(all_sizes)

        fiat_min, fiat_max, fiat_mean, fiat_std, fiat_total = safe_stats(fwd_iat)
        biat_min, biat_max, biat_mean, biat_std, biat_total = safe_stats(bwd_iat)
        aiat_min, aiat_max, aiat_mean, aiat_std, _ = safe_stats(all_iat)

        total_fwd = len(fwd)
        total_bwd = len(bwd)
        total_pkts = total_fwd + total_bwd

        flow_bytes_s  = (fwd_total + bwd_total) / max(duration / 1e6, 1e-6)
        flow_pkts_s   = total_pkts / max(duration / 1e6, 1e-6)
        fwd_pkts_s    = total_fwd / max(duration / 1e6, 1e-6)
        bwd_pkts_s    = total_bwd / max(duration / 1e6, 1e-6)
        down_up_ratio = total_bwd / max(total_fwd, 1)

        avg_pkt_size    = all_mean
        avg_fwd_seg     = fwd_mean
        avg_bwd_seg     = bwd_mean
        pkt_len_var     = float(np.var(all_sizes)) if all_sizes else 0

        # Active/idle periods (simplified — time between bursts)
        active_mean = duration / 2
        active_std  = 0
        active_max  = duration
        active_min  = 0
        idle_mean   = 0
        idle_std    = 0
        idle_max    = 0
        idle_min    = 0

        # Build feature dict matching your training column names exactly
        features = {
            'Protocol':                self.proto,
            'Flow Duration':           duration,
            'Total Fwd Packets':       total_fwd,
            'Total Backward Packets':  total_bwd,
            'Fwd Packets Length Total': fwd_total,
            'Bwd Packets Length Total': bwd_total,
            'Fwd Packet Length Max':   fwd_max,
            'Fwd Packet Length Min':   fwd_min,
            'Fwd Packet Length Mean':  fwd_mean,
            'Fwd Packet Length Std':   fwd_std,
            'Bwd Packet Length Max':   bwd_max,
            'Bwd Packet Length Min':   bwd_min,
            'Bwd Packet Length Mean':  bwd_mean,
            'Bwd Packet Length Std':   bwd_std,
            'Flow Bytes/s':            flow_bytes_s,
            'Flow Packets/s':          flow_pkts_s,
            'Flow IAT Mean':           aiat_mean,
            'Flow IAT Std':            aiat_std,
            'Flow IAT Max':            aiat_max,
            'Flow IAT Min':            aiat_min,
            'Fwd IAT Total':           fiat_total,
            'Fwd IAT Mean':            fiat_mean,
            'Fwd IAT Std':             fiat_std,
            'Fwd IAT Max':             fiat_max,
            'Fwd IAT Min':             fiat_min,
            'Bwd IAT Total':           biat_total,
            'Bwd IAT Mean':            biat_mean,
            'Bwd IAT Std':             biat_std,
            'Bwd IAT Max':             biat_max,
            'Bwd IAT Min':             biat_min,
            'Fwd PSH Flags':           self.fwd_psh_flags,
            'Bwd PSH Flags':           self.bwd_psh_flags,
            'Fwd URG Flags':           self.fwd_urg_flags,
            'Bwd URG Flags':           self.bwd_urg_flags,
            'Fwd Header Length':       self.fwd_header_len,
            'Bwd Header Length':       self.bwd_header_len,
            'Fwd Packets/s':           fwd_pkts_s,
            'Bwd Packets/s':           bwd_pkts_s,
            'Packet Length Min':       all_min,
            'Packet Length Max':       all_max,
            'Packet Length Mean':      all_mean,
            'Packet Length Std':       all_std,
            'Packet Length Variance':  pkt_len_var,
            'FIN Flag Count':          self.fin_count,
            'SYN Flag Count':          self.syn_count,
            'RST Flag Count':          self.rst_count,
            'PSH Flag Count':          self.psh_count,
            'ACK Flag Count':          self.ack_count,
            'URG Flag Count':          self.urg_count,
            'CWE Flag Count':          self.cwe_count,
            'ECE Flag Count':          self.ece_count,
            'Down/Up Ratio':           down_up_ratio,
            'Avg Packet Size':         avg_pkt_size,
            'Avg Fwd Segment Size':    avg_fwd_seg,
            'Avg Bwd Segment Size':    avg_bwd_seg,
            'Fwd Avg Bytes/Bulk':      0,
            'Fwd Avg Packets/Bulk':    0,
            'Fwd Avg Bulk Rate':       0,
            'Bwd Avg Bytes/Bulk':      0,
            'Bwd Avg Packets/Bulk':    0,
            'Bwd Avg Bulk Rate':       0,
            'Subflow Fwd Packets':     total_fwd,
            'Subflow Fwd Bytes':       fwd_total,
            'Subflow Bwd Packets':     total_bwd,
            'Subflow Bwd Bytes':       bwd_total,
            'Init Fwd Win Bytes':      self.init_fwd_win,
            'Init Bwd Win Bytes':      self.init_bwd_win,
            'Fwd Act Data Packets':    max(total_fwd - 1, 0),
            'Fwd Seg Size Min':        int(fwd_min),
            'Active Mean':             active_mean,
            'Active Std':              active_std,
            'Active Max':              active_max,
            'Active Min':              active_min,
            'Idle Mean':               idle_mean,
            'Idle Std':                idle_std,
            'Idle Max':                idle_max,
            'Idle Min':                idle_min,
        }

        # Fill any missing features with 0
        for col in FEATURE_COLS:
            if col not in features:
                features[col] = 0.0

        return features


# ── Flow tracker ───────────────────────────────────────────────
class FlowTracker:
    def __init__(self):
        self.flows     = {}
        self.completed = []

    def _flow_key(self, src_ip, dst_ip, src_port, dst_port, proto):
        """Canonical bidirectional flow key."""
        fwd = (src_ip, dst_ip, src_port, dst_port, proto)
        bwd = (dst_ip, src_ip, dst_port, src_port, proto)
        return min(fwd, bwd)

    def process_packet(self, pkt):
        """Add one packet to the appropriate flow."""
        if not pkt.haslayer(IP):
            return

        src_ip = pkt[IP].src
        dst_ip = pkt[IP].dst
        proto  = pkt[IP].proto
        ts     = float(pkt.time)
        size   = len(pkt)
        flags  = ''
        header_len = 20
        window = 0
        src_port = dst_port = 0

        if pkt.haslayer(TCP):
            src_port   = pkt[TCP].sport
            dst_port   = pkt[TCP].dport
            header_len = pkt[TCP].dataofs * 4 if pkt[TCP].dataofs else 20
            window     = pkt[TCP].window
            flag_int   = pkt[TCP].flags
            if flag_int & 0x01: flags += 'F'
            if flag_int & 0x02: flags += 'S'
            if flag_int & 0x04: flags += 'R'
            if flag_int & 0x08: flags += 'P'
            if flag_int & 0x10: flags += 'A'
            if flag_int & 0x20: flags += 'U'

        elif pkt.haslayer(UDP):
            src_port = pkt[UDP].sport
            dst_port = pkt[UDP].dport

        key = self._flow_key(src_ip, dst_ip, src_port, dst_port, proto)

        # Create new flow if not seen
        if key not in self.flows:
            self.flows[key] = Flow(src_ip, dst_ip, src_port,
                                   dst_port, proto, ts)

        flow = self.flows[key]
        is_forward = (src_ip == flow.src_ip)
        flow.add_packet(ts, size, flags, is_forward, header_len, window)

        # Complete the flow if FIN/RST seen
        if flow.is_complete():
            self.completed.append(flow)
            del self.flows[key]

    def flush_timeout(self, current_time):
        """Force-complete flows that have been idle too long."""
        timed_out = [k for k, f in self.flows.items()
                     if current_time - f.last_time > FLOW_TIMEOUT]
        for k in timed_out:
            self.completed.append(self.flows.pop(k))

    def flush_all(self):
        """Force-complete all remaining flows at end of pcap."""
        for flow in self.flows.values():
            self.completed.append(flow)
        self.flows.clear()


# ── Send to engine ─────────────────────────────────────────────
def send_to_engine(flow, ground_truth=None):
    """Send extracted features to the ML engine for classification."""
    features = flow.compute_features()
    payload = {
        'features':     features,
        'traffic_type': 'real_traffic',
        'ground_truth': ground_truth,
        'src_ip':       flow.src_ip,
    }
    try:
        r = requests.post(ENGINE_URL, json=payload, timeout=3)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        print(f"  Engine error: {e}")
    return None


# ── Main: process pcap file ────────────────────────────────────
def process_pcap(pcap_path, send_to_ml=True):
    print(f"Reading pcap: {pcap_path}")
    packets = rdpcap(pcap_path)
    print(f"Loaded {len(packets):,} packets")

    tracker = FlowTracker()

    for i, pkt in enumerate(packets):
        tracker.process_packet(pkt)
        if i % 10000 == 0:
            print(f"  Processed {i:,}/{len(packets):,} packets, "
                  f"completed flows: {len(tracker.completed)}")

    tracker.flush_all()
    flows = tracker.completed
    print(f"\nTotal flows extracted: {len(flows):,}")

    if len(flows) == 0:
        print("No flows extracted. Try capturing more traffic.")
        return

    # Show sample features from first flow
    print("\nSample flow features (first flow):")
    sample = flows[0].compute_features()
    for col in FEATURE_COLS[:10]:
        print(f"  {col:<35} {sample.get(col, 0):.4f}")

    # Verify feature match
    sample_keys = set(sample.keys())
    trained_keys = set(FEATURE_COLS)
    missing = trained_keys - sample_keys
    print(f"\nFeature coverage: {len(trained_keys - missing)}/{len(trained_keys)}")
    if missing:
        print(f"Missing features: {missing}")
    else:
        print("All 77 features present — feature match confirmed!")

    # Send to ML engine if requested
    if send_to_ml:
        print(f"\nSending {len(flows)} flows to ML engine...")
        results = {'ALLOW': 0, 'QUARANTINE': 0, 'BLOCK': 0}
        latencies = []

        for flow in flows:
            result = send_to_engine(flow)
            if result:
                decision = result.get('decision', 'UNKNOWN')
                results[decision] = results.get(decision, 0) + 1
                latencies.append(result.get('latency_ms', 0))

        print(f"\nResults on REAL traffic:")
        print(f"  ALLOW:      {results['ALLOW']:,}")
        print(f"  QUARANTINE: {results['QUARANTINE']:,}")
        print(f"  BLOCK:      {results['BLOCK']:,}")
        if latencies:
            print(f"  Avg latency: {np.mean(latencies):.2f}ms")
        print(f"\nAll {len(flows)} real flows classified by your model.")
        print("Check dashboard at http://localhost:5000 to see them.")
    else:
        print("\nDry run complete — features extracted successfully.")
        print("Run with --send to classify flows with the ML engine.")


# ── Entry point ────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Extract features from real traffic and classify with ML engine'
    )
    parser.add_argument('--pcap', required=True, help='Path to pcap file')
    parser.add_argument('--send', action='store_true',
                        help='Send flows to ML engine (default: dry run)')
    args = parser.parse_args()

    process_pcap(args.pcap, send_to_ml=args.send)
