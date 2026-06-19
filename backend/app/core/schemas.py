from typing import Optional, List
from pydantic import BaseModel


class PacketData(BaseModel):
    source_ip: str
    destination_ip: str
    protocol: str
    flags: Optional[str] = "*"
    payload_size: Optional[int] = 0
    destination_port: Optional[int] = 80


class DetectRequest(BaseModel):
    packet_data: PacketData
    model: Optional[str] = "lstm_ae"  # "kitsune" | "lstm_ae"


class SimulateAttackRequest(BaseModel):
    attack_type: str          # "DDoS" | "PortScan" | "Botnet" | "MITM" | "HTTPFlood"
    model: Optional[str] = "lstm_ae"
    src_ip: Optional[str] = None
    dst_ip: Optional[str] = "10.0.0.5"
    block_strategy: Optional[str] = "assertive"


class WhitelistRequest(BaseModel):
    ip: str
    label: Optional[str] = ""


class RuleActionRequest(BaseModel):
    rule_id: int
