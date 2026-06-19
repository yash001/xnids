import { useEffect, useState } from "react";
import { api } from "../lib/api";
import { Panel, Badge, Button } from "../components/ui";
import "./DefenceScreen.css";

export default function DefenceScreen() {
  const [rules, setRules] = useState([]);
  const [whitelist, setWhitelist] = useState([]);
  const [newIp, setNewIp] = useState("");
  const [newLabel, setNewLabel] = useState("");

  async function refresh() {
    setRules(await api.listRules(30));
    setWhitelist(await api.listWhitelist());
  }

  useEffect(() => { refresh(); }, []);

  async function toggle(id) {
    await api.toggleRule(id);
    refresh();
  }

  async function addWhitelist(e) {
    e.preventDefault();
    if (!newIp) return;
    await api.addWhitelist({ ip: newIp, label: newLabel });
    setNewIp(""); setNewLabel("");
    refresh();
  }

  async function removeWhitelist(id) {
    await api.removeWhitelist(id);
    refresh();
  }

  const activeCount = rules.filter((r) => r.active).length;

  return (
    <div className="def">
      <header className="def__header">
        <div>
          <div className="eyebrow">Active Response</div>
          <h1 className="def__title">Defence Screen</h1>
        </div>
        <Badge tone="mitigated">{activeCount} rule{activeCount !== 1 ? "s" : ""} enforced</Badge>
      </header>

      <Panel eyebrow="Unified Defense Rules" title="Generated rules (per xNIDS Sec 4)">
        {rules.length === 0 ? (
          <div className="def__empty mono">No defense rules generated yet — simulate an attack first.</div>
        ) : (
          <table className="def__table">
            <thead>
              <tr>
                <th>Time</th><th>Attack</th><th>Source</th><th>Scope</th><th>Strategy</th><th>OpenFlow rule</th><th>Status</th><th />
              </tr>
            </thead>
            <tbody>
              {rules.map((r) => (
                <tr key={r.id} className={!r.active ? "is-inactive" : ""}>
                  <td className="mono">{new Date(r.timestamp).toLocaleTimeString()}</td>
                  <td>{r.attack_type}</td>
                  <td className="mono">{r.src_ip}</td>
                  <td><span className="def__scope mono">{r.scope}</span></td>
                  <td className="mono">{r.strategy}</td>
                  <td className="mono def__rule-cell">{r.openflow_rule}</td>
                  <td><Badge tone={r.active ? "active" : "mitigated"}>{r.active ? "enforced" : "lifted"}</Badge></td>
                  <td>
                    <Button variant="ghost" onClick={() => toggle(r.id)}>
                      {r.active ? "Lift" : "Re-enforce"}
                    </Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Panel>

      <Panel eyebrow="Security Constraint" title="Whitelist (critical services exempt from rules)">
        <form className="def__whitelist-form" onSubmit={addWhitelist}>
          <input
            className="def__input mono"
            placeholder="IP address, e.g. 10.0.0.1"
            value={newIp}
            onChange={(e) => setNewIp(e.target.value)}
          />
          <input
            className="def__input"
            placeholder="Label (optional)"
            value={newLabel}
            onChange={(e) => setNewLabel(e.target.value)}
          />
          <Button type="submit">Add to whitelist</Button>
        </form>

        {whitelist.length === 0 ? (
          <div className="def__empty mono">No whitelisted hosts.</div>
        ) : (
          <ul className="def__whitelist-list">
            {whitelist.map((w) => (
              <li key={w.id}>
                <span className="mono">{w.ip}</span>
                <span className="def__whitelist-label">{w.label}</span>
                <button className="def__remove" onClick={() => removeWhitelist(w.id)}>remove</button>
              </li>
            ))}
          </ul>
        )}
      </Panel>
    </div>
  );
}
