"""
defense.py
===========
Endpoints powering the "Defence Screen": list generated defense rules
(across OpenFlow / iptables / Pfsense translations), manage the
whitelist (Sec 4.2 security constraint), and toggle/expire rules.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..core.database import get_db, DefenseRuleRecord, WhitelistEntry, AttackLog
from ..core.schemas import WhitelistRequest

router = APIRouter()


@router.get("/defense/rules")
def list_rules(limit: int = 50, db: Session = Depends(get_db)):
    rows = db.query(DefenseRuleRecord).order_by(DefenseRuleRecord.id.desc()).limit(limit).all()
    out = []
    for r in rows:
        attack = db.query(AttackLog).filter(AttackLog.id == r.attack_log_id).first()
        out.append({
            "id": r.id,
            "timestamp": r.timestamp.isoformat(),
            "attack_log_id": r.attack_log_id,
            "attack_type": attack.attack_type if attack else None,
            "src_ip": attack.src_ip if attack else None,
            "scope": r.scope,
            "strategy": r.strategy,
            "openflow_rule": r.openflow_rule,
            "iptables_rule": r.iptables_rule,
            "pfsense_rule": r.pfsense_rule,
            "active": r.active,
        })
    return out


@router.post("/defense/rules/{rule_id}/toggle")
def toggle_rule(rule_id: int, db: Session = Depends(get_db)):
    rule = db.query(DefenseRuleRecord).filter(DefenseRuleRecord.id == rule_id).first()
    if not rule:
        return {"error": "not found"}
    rule.active = not rule.active
    db.commit()
    return {"id": rule_id, "active": rule.active}


@router.get("/defense/whitelist")
def list_whitelist(db: Session = Depends(get_db)):
    rows = db.query(WhitelistEntry).order_by(WhitelistEntry.id.desc()).all()
    return [{"id": r.id, "ip": r.ip, "label": r.label, "created_at": r.created_at.isoformat()} for r in rows]


@router.post("/defense/whitelist")
def add_whitelist(req: WhitelistRequest, db: Session = Depends(get_db)):
    existing = db.query(WhitelistEntry).filter(WhitelistEntry.ip == req.ip).first()
    if existing:
        return {"error": "already whitelisted", "id": existing.id}
    entry = WhitelistEntry(ip=req.ip, label=req.label)
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return {"id": entry.id, "ip": entry.ip, "label": entry.label}


@router.delete("/defense/whitelist/{entry_id}")
def remove_whitelist(entry_id: int, db: Session = Depends(get_db)):
    entry = db.query(WhitelistEntry).filter(WhitelistEntry.id == entry_id).first()
    if not entry:
        return {"error": "not found"}
    db.delete(entry)
    db.commit()
    return {"deleted": entry_id}
