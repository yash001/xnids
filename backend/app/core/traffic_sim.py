"""
traffic_sim.py
================
Synthetic network traffic generator used to train/evaluate both the
Kitsune-style autoencoder (paper baseline) and our LSTM-Autoencoder
deep-learning alternative.

We simulate 23 packet/flow-level features per time-window, loosely
inspired by Kitsune's feature extractor described in xNIDS (Sec. 5,
Appendix C.1): packet size, inter-arrival time, protocol one-hots,
flag one-hots, jitter, byte-rate, etc. Five attack families are
modeled, matching the ones discussed in the xNIDS paper's
evaluation/case studies (DDoS, Port/OS Scan (Recon), Botnet, MITM,
HTTP Flood).
"""

import numpy as np

FEATURE_NAMES = [
    "pkt_size_mean", "pkt_size_std", "iat_mean", "iat_std",
    "byte_rate", "pkt_rate", "tcp_flag_syn", "tcp_flag_ack",
    "tcp_flag_fin", "tcp_flag_rst", "is_tcp", "is_udp", "is_icmp",
    "is_arp", "is_http", "src_port_entropy", "dst_port_entropy",
    "unique_dst_ip_count", "unique_dst_port_count", "http_referer_repeat",
    "ttl_mean", "window_size_mean", "payload_entropy",
]
N_FEATURES = len(FEATURE_NAMES)

ATTACK_TYPES = ["DDoS", "PortScan", "Botnet", "MITM", "HTTPFlood"]

RNG = np.random.default_rng(42)


def _benign_window(rng):
    f = rng.normal(loc=0.5, scale=0.08, size=N_FEATURES)
    f[6:10] = rng.uniform(0.1, 0.4, size=4)          # tcp flags moderate
    f[10] = rng.uniform(0.4, 0.7)                     # is_tcp
    f[11] = 1 - f[10] - rng.uniform(0, 0.1)            # is_udp
    f[17] = rng.uniform(1, 6)                          # unique dst ip
    f[18] = rng.uniform(1, 10)                         # unique dst port
    f[19] = rng.uniform(0, 0.2)                        # http referer repeat
    return np.clip(f, 0, None)


def _attack_window(rng, attack_type):
    f = _benign_window(rng)
    if attack_type == "DDoS":
        f[5] *= rng.uniform(8, 20)      # pkt_rate spike
        f[4] *= rng.uniform(8, 20)      # byte_rate spike
        f[6] = rng.uniform(0.8, 1.0)    # SYN flood
        f[17] = rng.uniform(1, 3)       # one victim
    elif attack_type == "PortScan":
        f[18] = rng.uniform(40, 120)    # many unique dst ports
        f[16] = rng.uniform(0.8, 1.0)   # high dst port entropy
        f[12] = rng.uniform(0.2, 0.6)   # icmp probing
        f[13] = rng.uniform(0.2, 0.6)   # arp probing
    elif attack_type == "Botnet":
        f[19] = rng.uniform(0.6, 1.0)   # repeated C2 referer
        f[14] = rng.uniform(0.7, 1.0)   # is_http
        f[17] = rng.uniform(8, 20)      # multiple dst (C2 + targets)
    elif attack_type == "MITM":
        f[20] = rng.uniform(0.0, 0.05)  # abnormal ttl (spoof)
        f[21] *= rng.uniform(2, 5)      # window size anomaly
        f[22] = rng.uniform(0.7, 1.0)   # payload entropy (tampering)
    elif attack_type == "HTTPFlood":
        f[14] = 1.0
        f[19] = rng.uniform(0.85, 1.0)  # same referer repeated
        f[5] *= rng.uniform(5, 15)
    return np.clip(f, 0, None)


def generate_dataset(n_benign=4000, n_attack_per_type=600, seed=42, with_history=True, history_len=5):
    """Generates a labeled dataset of (sequence, label, attack_type).
    Each sample is a sequence of `history_len` windows ending at the
    current window, matching the xNIDS notion of x_t with history
    inputs X_{t,k} (Sec 3.1).
    """
    rng = np.random.default_rng(seed)
    X, y, atk = [], [], []

    def make_sequence(window_fn, *args):
        seq = [window_fn(rng, *args) if args else window_fn(rng) for _ in range(history_len)]
        return np.stack(seq, axis=0)

    for _ in range(n_benign):
        X.append(make_sequence(_benign_window))
        y.append(0)
        atk.append("Benign")

    for t in ATTACK_TYPES:
        for _ in range(n_attack_per_type):
            # mix: a few benign history windows followed by attack windows,
            # mirroring the paper's point that attacks build up over a
            # number of history inputs (Sec 2.3 / Ch1).
            n_attack_steps = rng.integers(1, history_len + 1)
            n_benign_steps = history_len - n_attack_steps
            seq = [_benign_window(rng) for _ in range(n_benign_steps)] + \
                  [_attack_window(rng, t) for _ in range(n_attack_steps)]
            X.append(np.stack(seq, axis=0))
            y.append(1)
            atk.append(t)

    X = np.stack(X, axis=0)  # (N, history_len, N_FEATURES)
    y = np.array(y)
    atk = np.array(atk)

    idx = rng.permutation(len(X))
    return X[idx], y[idx], atk[idx]


def live_packet_window(attack_type=None, rng=None):
    """Generate one realtime window for simulation/demo endpoints."""
    rng = rng or RNG
    if attack_type is None or attack_type == "Benign":
        return _benign_window(rng)
    return _attack_window(rng, attack_type)
