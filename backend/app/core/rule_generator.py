"""
rule_generator.py
===================
Implements xNIDS's defense-rule generation pipeline (paper Sec 4):
  - Defense Rule Scope (per-flow / per-host / multi-hosts) decided from
    statistical info S (Table 2).
  - Security Constraints: whitelist + block strategy (passive /
    assertive / aggressive), Sec 4.2.
  - Unified Defense Rule representation <entity, action, priority, timeout>
    (Sec 4.3, Appendix B), translated into OpenFlow, iptables, and Pfsense
    style rules (Table 9).
"""

import time
from dataclasses import dataclass, field
from typing import Optional


BLOCK_STRATEGIES = ("passive", "assertive", "aggressive")


@dataclass
class UnifiedRule:
    src_ip: str
    dst_ip: Optional[str] = None
    src_mac: Optional[str] = None
    dst_port: Optional[int] = None
    protocol: str = "*"
    flags: str = "*"
    action: str = "drop"
    priority: int = 10
    timeout: int = 600
    scope: str = "per-flow"
    rule_id: str = field(default_factory=lambda: f"R{int(time.time()*1000) % 1000000}")

    def to_openflow(self):
        parts = [f"nw_src={self.src_ip}"]
        if self.dst_ip:
            parts.append(f"nw_dst={self.dst_ip}")
        if self.protocol != "*":
            parts.append(self.protocol.lower())
        if self.flags != "*":
            parts.append(self.flags.lower())
        body = ", ".join(parts)
        return f"<{body}, actions=drop, priority={self.priority}, hard_timeout={self.timeout}>"

    def to_iptables(self):
        cmd = f"-A INPUT -s {self.src_ip}"
        if self.dst_port:
            cmd += f" --dport {self.dst_port}"
        if self.protocol != "*":
            cmd += f" -p {self.protocol.lower()}"
        if self.flags != "*" and "syn" in self.flags.lower():
            cmd += " --syn -m limit --limit-burst 3"
        cmd += " -j DROP"
        return f"iptables {cmd}"

    def to_pfsense(self):
        return (f"block in proto {self.protocol.lower() if self.protocol != '*' else 'any'} "
                f"from {self.src_ip} to {self.dst_ip or 'any'} "
                f"port {self.dst_port or 'any'}")


def determine_scope(stat_info: dict) -> str:
    """Paper Sec 4.1: decide scope by checking which field in S has the
    greatest value."""
    if stat_info is None:
        return "unknown"
    fields = {
        "IP_n": stat_info["IP_n"],
        "Port_n": stat_info["Port_n"],
        "Protocol_n": stat_info["Protocol_n"],
    }
    dominant = max(fields, key=fields.get)
    if dominant in ("Protocol_n", "Port_n") and len(stat_info["IP_pool"]) > 1:
        return "multi-hosts"
    if len(stat_info["IP_pool"]) == 1:
        # same host: per-host if multiple protocols/ports involved else per-flow
        return "per-host" if stat_info["Protocol_n"] > 1 or stat_info["Port_n"] > 1 else "per-flow"
    return "multi-hosts"


def apply_block_strategy(scope: str, strategy: str) -> str:
    """Sec 4.2: modify operation (drop_flow / drop_host) per block strategy."""
    base_op = "drop_host" if scope in ("per-host", "multi-hosts") else "drop_flow"
    if strategy == "passive":
        return "drop_flow"
    if strategy == "aggressive":
        return "drop_host"
    return base_op  # assertive: unchanged


def generate_unified_rule(top_features: dict, flow_sample: dict, scope: str,
                           strategy: str, attack_type: str, priority: int = 10) -> UnifiedRule:
    """Builds a unified defense rule entity from the explanation's
    important features + the representative flow sample, following the
    operation->entity construction logic in Sec 4.3.
    """
    operation = apply_block_strategy(scope, strategy)
    timeout = 6000 if scope == "per-flow" else (600 if scope == "per-host" else 999999)

    flags = "*"
    if top_features.get("tcp_flag_syn", 0) > 0.05:
        flags = "TCP.SYN"
    protocol = "tcp" if flow_sample.get("protocol") == "TCP" else flow_sample.get("protocol", "*").lower()

    rule = UnifiedRule(
        src_ip=flow_sample["src_ip"] if operation == "drop_host" else flow_sample["src_ip"],
        dst_ip=None if operation == "drop_host" else flow_sample.get("dst_ip"),
        src_mac=flow_sample.get("src_mac"),
        dst_port=flow_sample.get("dst_port") if scope == "multi-hosts" else None,
        protocol=protocol if scope != "multi-hosts" else "*",
        flags=flags,
        action="drop",
        priority=priority,
        timeout=timeout,
        scope=scope,
    )
    return rule


def generate_defense_response(explanation: dict, flow_sample: dict, stat_info: dict,
                               attack_type: str, strategy: str = "assertive") -> dict:
    """End-to-end pipeline: explanation -> scope -> unified rule -> actionable rules."""
    scope = determine_scope(stat_info)
    rule = generate_unified_rule(explanation["feature_importance"], flow_sample, scope, strategy, attack_type)
    return {
        "scope": scope,
        "strategy": strategy,
        "unified_rule": rule.__dict__,
        "openflow_rule": rule.to_openflow(),
        "iptables_rule": rule.to_iptables(),
        "pfsense_rule": rule.to_pfsense(),
    }
