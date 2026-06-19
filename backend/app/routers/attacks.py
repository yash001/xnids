"""
attacks.py
===========
Endpoints powering the "Attack Screen": simulate an attack of a chosen
type, run it through the selected detector, generate an xNIDS-style
explanation, persist the alert, and (optionally) auto-generate a
defense rule via the rule generator.
"""

import json
import random
from datetime import datetime

import numpy as np
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..core.schemas import SimulateAttackRequest
from ..core.model_store import store
from ..core.traffic_sim import live_packet_window, ATTACK_TYPES
from ..core.database import get_db, AttackLog, DefenseRuleRecord, WhitelistEntry
from ..core.rule_generator import generate_defense_response
from ..core.explainer import compute_statistical_info

router = APIRouter()

SEVERITY_MAP = {
    "DDoS": "high", "Botnet": "high", "PortScan": "medium",
    "MITM": "high", "HTTPFlood": "medium",
}


def _random_ip(prefix="192.168.1."):
    return f"{prefix}{random.randint(2, 250)}"


def _random_mac():
    return ":".join(f"{random.randint(0,255):02x}" for _ in range(6))


@router.post("/attacks/simulate")
def simulate_attack(req: SimulateAttackRequest, db: Session = Depends(get_db)):
    if req.attack_type not in ATTACK_TYPES:
        return {"error": f"attack_type must be one of {ATTACK_TYPES}"}

    rng = np.random.default_rng()
    history_len = 5
    n_attack_steps = rng.integers(2, history_len + 1)
    n_benign_steps = history_len - n_attack_steps
    seq = [live_packet_window("Benign") for _ in range(n_benign_steps)] + \
          [live_packet_window(req.attack_type) for _ in range(n_attack_steps)]
    seq = np.stack(seq, axis=0)

    detector = store.get_detector(req.model)
    if req.model == "kitsune":
        labels, scores, conf = detector.predict(seq[-1:, :])
    else:
        labels, scores, conf = detector.predict(seq[np.newaxis, :, :])

    explainer = store.get_explainer(req.model)
    explanation = explainer.explain(seq)

    src_ip = req.src_ip or _random_ip()
    src_mac = _random_mac()
    dst_port = {"DDoS": 80, "PortScan": random.randint(1, 65535), "Botnet": 6667,
                "MITM": 443, "HTTPFlood": 80}.get(req.attack_type, 80)
    flow_sample = {"src_ip": src_ip, "dst_ip": req.dst_ip, "src_mac": src_mac,
                    "dst_port": dst_port, "protocol": "TCP" if req.attack_type != "PortScan" else "ICMP"}

    # simulate a handful of related malicious flows for statistical info S
    related_flows = [flow_sample] + [
        {**flow_sample, "dst_port": dst_port if req.attack_type != "PortScan" else random.randint(1, 65535)}
        for _ in range(random.randint(3, 15))
    ]
    stat_info = compute_statistical_info(related_flows)

    defense = generate_defense_response(explanation, flow_sample, stat_info,
                                         req.attack_type, strategy=req.block_strategy)

    log = AttackLog(
        src_ip=src_ip, dst_ip=req.dst_ip, src_mac=src_mac, dst_port=dst_port,
        protocol=flow_sample["protocol"], attack_type=req.attack_type,
        severity=SEVERITY_MAP.get(req.attack_type, "medium"),
        status="active", model_used=req.model, confidence=float(conf[0]),
        anomaly_score=float(scores[0]),
        top_features_json=json.dumps(explanation["top_features"]),
    )
    db.add(log)
    db.commit()
    db.refresh(log)

    rule = DefenseRuleRecord(
        attack_log_id=log.id, scope=defense["scope"], strategy=defense["strategy"],
        openflow_rule=defense["openflow_rule"], iptables_rule=defense["iptables_rule"],
        pfsense_rule=defense["pfsense_rule"], active=True,
    )
    db.add(rule)
    db.commit()
    db.refresh(rule)

    return {
        "alert_id": log.id,
        "prediction": "attack" if int(labels[0]) == 1 else "normal",
        "attack_type": req.attack_type,
        "confidence": round(float(conf[0]), 4),
        "anomaly_score": round(float(scores[0]), 4),
        "model_used": req.model,
        "flow": flow_sample,
        "explanation": explanation,
        "defense_rule": {**defense, "rule_db_id": rule.id},
        "timestamp": log.timestamp.isoformat(),
    }


@router.get("/attacks")
def list_attacks(limit: int = 50, db: Session = Depends(get_db)):
    rows = db.query(AttackLog).order_by(AttackLog.id.desc()).limit(limit).all()
    out = []
    for r in rows:
        out.append({
            "id": r.id,
            "timestamp": r.timestamp.isoformat(),
            "source_ip": r.src_ip,
            "destination_ip": r.dst_ip,
            "attack_type": r.attack_type,
            "severity": r.severity,
            "status": r.status,
            "model_used": r.model_used,
            "confidence": r.confidence,
            "anomaly_score": r.anomaly_score,
            "top_features": json.loads(r.top_features_json) if r.top_features_json else [],
        })
    return out


@router.post("/attacks/{attack_id}/mitigate")
def mitigate_attack(attack_id: int, db: Session = Depends(get_db)):
    log = db.query(AttackLog).filter(AttackLog.id == attack_id).first()
    if not log:
        return {"error": "not found"}
    log.status = "mitigated"
    db.commit()
    return {"id": attack_id, "status": "mitigated"}
