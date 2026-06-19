"""
detect.py
==========
Implements the `/detect_attack` endpoint exactly as specified in the
project brief's Example 1: accepts raw packet-ish data, runs it through
the selected model (paper-baseline Kitsune-style AE, or our LSTM-AE),
and returns prediction + confidence + attack_type + model_used.

Since a single packet has no history, we synthesize a short history
window by repeating/perturbing the packet's derived feature vector --
in a production deployment this would instead pull the last k real
windows for that flow/host from a feature store.
"""

import numpy as np
from fastapi import APIRouter

from ..core.schemas import DetectRequest
from ..core.model_store import store
from ..core.traffic_sim import FEATURE_NAMES, N_FEATURES

router = APIRouter()

PROTOCOL_MAP = {"TCP": 0, "UDP": 1, "ICMP": 2, "ARP": 3, "HTTP": 4}


def packet_to_feature_vector(packet):
    f = np.zeros(N_FEATURES)
    f[0] = min(packet.payload_size / 1500.0, 1.0)        # pkt_size_mean (normalized)
    f[1] = 0.05
    f[4] = min(packet.payload_size / 500.0, 5.0)          # byte_rate proxy
    f[5] = 0.3
    flags = (packet.flags or "").upper()
    f[6] = 1.0 if "S" in flags else 0.1
    f[7] = 1.0 if "A" in flags else 0.1
    f[8] = 1.0 if "F" in flags else 0.05
    f[9] = 1.0 if "R" in flags else 0.05
    proto = packet.protocol.upper()
    f[10] = 1.0 if proto == "TCP" else 0.0
    f[11] = 1.0 if proto == "UDP" else 0.0
    f[12] = 1.0 if proto == "ICMP" else 0.0
    f[13] = 1.0 if proto == "ARP" else 0.0
    f[14] = 1.0 if proto == "HTTP" else 0.0
    f[17] = 1.0
    f[18] = 1.0
    f[20] = 0.5
    f[21] = 0.5
    return f


@router.post("/detect_attack")
def detect_attack(req: DetectRequest):
    model_name = req.model if req.model in ("kitsune", "lstm_ae") else "lstm_ae"
    detector = store.get_detector(model_name)

    vec = packet_to_feature_vector(req.packet_data)
    history_len = 5
    rng = np.random.default_rng()
    seq = np.stack([vec + rng.normal(0, 0.02, size=N_FEATURES) for _ in range(history_len)])

    if model_name == "kitsune":
        labels, scores, conf = detector.predict(seq[-1:, :])
    else:
        labels, scores, conf = detector.predict(seq[np.newaxis, :, :])

    pred_label = int(labels[0])
    attack_type = None
    if pred_label == 1:
        flags = (req.packet_data.flags or "").upper()
        if "S" in flags and req.packet_data.payload_size == 0:
            attack_type = "DDoS"
        elif req.packet_data.protocol.upper() in ("ICMP", "ARP"):
            attack_type = "PortScan"
        else:
            attack_type = "Unclassified Anomaly"

    return {
        "prediction": "attack" if pred_label == 1 else "normal",
        "confidence": round(float(conf[0]), 4),
        "attack_type": attack_type,
        "anomaly_score": round(float(scores[0]), 4),
        "model_used": "deep_learning_lstm_autoencoder" if model_name == "lstm_ae" else "paper_solution_kitsune_ae",
    }
