import { useEffect, useState } from "react";
import { api } from "../lib/api";
import { Panel, Badge, Button, Select } from "../components/ui";
import GroupBars from "../components/GroupBars";
import "./AttackScreen.css";

const ATTACK_TYPES = [
  { value: "DDoS", label: "DDoS — SYN flood" },
  { value: "PortScan", label: "Port / OS Scan" },
  { value: "Botnet", label: "Botnet C2 beacon" },
  { value: "MITM", label: "Man-in-the-Middle" },
  { value: "HTTPFlood", label: "HTTP Flood" },
];

const MODEL_OPTIONS = [
  { value: "lstm_ae", label: "LSTM-Autoencoder (our DL alt.)" },
  { value: "kitsune", label: "Kitsune-style AE (paper baseline)" },
];

const STRATEGY_OPTIONS = [
  { value: "passive", label: "Passive block" },
  { value: "assertive", label: "Assertive block" },
  { value: "aggressive", label: "Aggressive block" },
];

export default function AttackScreen() {
  const [attackType, setAttackType] = useState("DDoS");
  const [model, setModel] = useState("lstm_ae");
  const [strategy, setStrategy] = useState("assertive");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [log, setLog] = useState([]);
  const [error, setError] = useState(null);

  async function refreshLog() {
    try {
      setLog(await api.listAttacks(15));
    } catch {
      // ignore — log list will simply stay empty until next refresh
    }
  }

  useEffect(() => { refreshLog(); }, []);

  async function launch() {
    setLoading(true);
    setError(null);
    try {
      const res = await api.simulateAttack({
        attack_type: attackType,
        model,
        block_strategy: strategy,
      });
      setResult(res);
      refreshLog();
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  async function mitigate(id) {
    await api.mitigateAttack(id);
    refreshLog();
  }

  return (
    <div className="atk">
      <header className="atk__header">
        <div>
          <div className="eyebrow">Adversarial Simulation</div>
          <h1 className="atk__title">Attack Screen</h1>
        </div>
      </header>

      <Panel eyebrow="Configure" title="Launch a simulated intrusion">
        <div className="atk__controls">
          <div className="atk__field">
            <label className="eyebrow">Attack type</label>
            <Select value={attackType} onChange={(e) => setAttackType(e.target.value)} options={ATTACK_TYPES} />
          </div>
          <div className="atk__field">
            <label className="eyebrow">Detector</label>
            <Select value={model} onChange={(e) => setModel(e.target.value)} options={MODEL_OPTIONS} />
          </div>
          <div className="atk__field">
            <label className="eyebrow">Block strategy</label>
            <Select value={strategy} onChange={(e) => setStrategy(e.target.value)} options={STRATEGY_OPTIONS} />
          </div>
          <Button onClick={launch} disabled={loading}>
            {loading ? "Simulating…" : "Launch simulation"}
          </Button>
        </div>
        {error && <div className="atk__error mono">{error}</div>}
      </Panel>

      {result && (
        <div className="atk__result-grid">
          <Panel eyebrow={`Alert #${result.alert_id}`} title="Detection result">
            <div className="atk__result-row">
              <Badge tone={result.prediction === "attack" ? "high" : "mitigated"}>
                {result.prediction}
              </Badge>
              <span className="mono atk__score">
                score {result.anomaly_score.toFixed(3)} · confidence {(result.confidence * 100).toFixed(1)}%
              </span>
            </div>
            <dl className="atk__deflist">
              <dt>Attack type</dt><dd>{result.attack_type}</dd>
              <dt>Model used</dt><dd className="mono">{result.model_used}</dd>
              <dt>Source IP</dt><dd className="mono">{result.flow.src_ip}</dd>
              <dt>Destination</dt><dd className="mono">{result.flow.dst_ip}:{result.flow.dst_port}</dd>
              <dt>Protocol</dt><dd className="mono">{result.flow.protocol}</dd>
            </dl>
          </Panel>

          <Panel eyebrow="xNIDS-style explanation" title="Feature group importance">
            <GroupBars groupImportance={result.explanation.group_importance} />
            <div className="atk__top-features">
              <div className="eyebrow" style={{ marginBottom: 8 }}>Top individual features</div>
              {result.explanation.top_features.map(([f, v]) => (
                <div key={f} className="atk__feature-row mono">
                  <span>{f}</span><span>{(v * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
          </Panel>

          <Panel eyebrow={`Scope: ${result.defense_rule.scope}`} title="Generated defense rule">
            <div className="atk__rule-block">
              <div className="eyebrow">OpenFlow</div>
              <code className="mono atk__rule-code">{result.defense_rule.openflow_rule}</code>
            </div>
            <div className="atk__rule-block">
              <div className="eyebrow">iptables</div>
              <code className="mono atk__rule-code">{result.defense_rule.iptables_rule}</code>
            </div>
            <div className="atk__rule-block">
              <div className="eyebrow">Pfsense</div>
              <code className="mono atk__rule-code">{result.defense_rule.pfsense_rule}</code>
            </div>
          </Panel>
        </div>
      )}

      <Panel eyebrow="History" title="Attack log">
        {log.length === 0 ? (
          <div className="atk__empty mono">No attacks simulated yet.</div>
        ) : (
          <table className="atk__table">
            <thead>
              <tr><th>Time</th><th>Source</th><th>Type</th><th>Severity</th><th>Confidence</th><th>Status</th><th /></tr>
            </thead>
            <tbody>
              {log.map((a) => (
                <tr key={a.id}>
                  <td className="mono">{new Date(a.timestamp).toLocaleTimeString()}</td>
                  <td className="mono">{a.source_ip}</td>
                  <td>{a.attack_type}</td>
                  <td><Badge tone={a.severity}>{a.severity}</Badge></td>
                  <td className="mono">{(a.confidence * 100).toFixed(1)}%</td>
                  <td><Badge tone={a.status}>{a.status}</Badge></td>
                  <td>
                    {a.status === "active" && (
                      <Button variant="ghost" onClick={() => mitigate(a.id)}>Mitigate</Button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Panel>
    </div>
  );
}
