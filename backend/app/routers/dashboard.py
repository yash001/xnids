"""
dashboard.py
=============
Endpoints powering the main Dashboard: overall stats, attack-type
breakdown, and the head-to-head comparison between the paper baseline
(Kitsune-style AE) and our LSTM-Autoencoder DL alternative -- including
a simplified reproduction of the paper's own fidelity-style evaluation
idea (Descriptive Accuracy, Sec 6.1.1): we zero out the top-k important
features identified by the explainer and measure how much the
detector's anomaly score drops, exactly mirroring Fig. 3 of the paper.
"""

import numpy as np
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func

from ..core.database import get_db, AttackLog, DefenseRuleRecord
from ..core.model_store import store
from ..core.traffic_sim import generate_dataset, FEATURE_NAMES, ATTACK_TYPES
from ..core.explainer import XNIDSExplainer

router = APIRouter()


@router.get("/dashboard/stats")
def dashboard_stats(db: Session = Depends(get_db)):
    total = db.query(func.count(AttackLog.id)).scalar() or 0
    active = db.query(func.count(AttackLog.id)).filter(AttackLog.status == "active").scalar() or 0
    mitigated = db.query(func.count(AttackLog.id)).filter(AttackLog.status == "mitigated").scalar() or 0
    by_type = dict(db.query(AttackLog.attack_type, func.count(AttackLog.id))
                    .group_by(AttackLog.attack_type).all())
    by_severity = dict(db.query(AttackLog.severity, func.count(AttackLog.id))
                        .group_by(AttackLog.severity).all())
    avg_conf = db.query(func.avg(AttackLog.confidence)).scalar() or 0
    rules_active = db.query(func.count(DefenseRuleRecord.id)).filter(DefenseRuleRecord.active == True).scalar() or 0  # noqa

    return {
        "total_alerts": total,
        "active_alerts": active,
        "mitigated_alerts": mitigated,
        "by_attack_type": by_type,
        "by_severity": by_severity,
        "avg_confidence": round(float(avg_conf), 4) if avg_conf else 0,
        "active_defense_rules": rules_active,
        "model_ready": store.ready,
        "training_metrics": store.metrics,
    }


@router.get("/dashboard/model_comparison")
def model_comparison():
    """Returns headline metrics + a live fidelity test (paper Sec 6.1.1)
    comparing how 'sharp' the explanation-driven feature ablation curve
    is for each model -- a steeper drop means a more faithful, more
    explainable, higher-quality detector+explainer pairing.
    """
    if not store.ready:
        return {"error": "models not trained yet"}

    X, y, atk = generate_dataset(n_benign=200, n_attack_per_type=40, history_len=5, seed=99)
    anomaly_idx = np.where(y == 1)[0][:30]

    results = {}
    for name, key in [("kitsune", "kitsune"), ("lstm_ae", "lstm_ae")]:
        detector = store.get_detector(key)
        explainer = XNIDSExplainer(detector, history_len=5, n_samples=80)
        scores_curve = []
        for k in range(0, 11, 2):
            dropped_scores = []
            for idx in anomaly_idx[:10]:
                seq = X[idx].copy()
                exp = explainer.explain(seq)
                top_feats = [f for f, _ in exp["top_features"]][:k]
                seq_mod = seq.copy()
                for f in top_feats:
                    fi = FEATURE_NAMES.index(f)
                    seq_mod[:, fi] = 0.0
                if key == "kitsune":
                    _, s, _ = detector.predict(seq_mod[-1:, :])
                else:
                    _, s, _ = detector.predict(seq_mod[np.newaxis, :, :])
                dropped_scores.append(float(s[0]))
            scores_curve.append(round(float(np.mean(dropped_scores)), 4))
        results[name] = {
            "ada_curve": scores_curve,  # x-axis: 0,2,4,...,10 modified features
            "metrics": store.metrics.get(key, {}),
        }

    return {
        "ada_curve_x": list(range(0, 11, 2)),
        "results": results,
        "per_attack_recall": store.metrics.get("per_attack_recall", {}),
        "note": ("Steeper decline in anomaly score (ADA-style curve) as more "
                 "top-ranked features are zeroed indicates a more faithful "
                 "explanation, mirroring xNIDS paper Fig. 3 / Table 3."),
    }


@router.get("/dashboard/feature_catalog")
def feature_catalog():
    return {"features": FEATURE_NAMES, "attack_types": ATTACK_TYPES}
