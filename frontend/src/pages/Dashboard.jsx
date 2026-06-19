import { useEffect, useState } from "react";
import { api } from "../lib/api";
import { Panel, StatCard, Badge } from "../components/ui";
import Waveform from "../components/Waveform";
import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid,
  PieChart, Pie, Cell,
} from "recharts";
import "./Dashboard.css";

const SEVERITY_COLORS = { high: "#ff5468", medium: "#ffb020", low: "#4fd6d2" };

export default function Dashboard() {
  const [stats, setStats] = useState(null);
  const [attacks, setAttacks] = useState([]);
  const [error, setError] = useState(null);

  async function refresh() {
    try {
      const [s, a] = await Promise.all([api.dashboardStats(), api.listAttacks(8)]);
      setStats(s);
      setAttacks(a);
      setError(null);
    } catch (e) {
      setError(e.message);
    }
  }

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 5000);
    return () => clearInterval(id);
  }, []);

  const recentActive = attacks.filter((a) => a.status === "active").length;
  const pulse = Math.min(1, recentActive / 4);

  const byType = stats?.by_attack_type
    ? Object.entries(stats.by_attack_type).map(([name, count]) => ({ name, count }))
    : [];
  const bySeverity = stats?.by_severity
    ? Object.entries(stats.by_severity).map(([name, value]) => ({ name, value }))
    : [];

  return (
    <div className="dash">
      <header className="dash__header">
        <div>
          <div className="eyebrow">Network Operations</div>
          <h1 className="dash__title">Signal Overview</h1>
        </div>
        <Badge tone={stats?.model_ready ? "mitigated" : "active"}>
          {stats?.model_ready ? "Detectors online" : "Training…"}
        </Badge>
      </header>

      {error && <div className="dash__error mono">API error: {error}. Is the backend running on :8000?</div>}

      <Panel
        eyebrow="Live anomaly trace"
        title="Composite detection signal"
        className="dash__waveform-panel"
      >
        <Waveform height={70} pulse={pulse} />
        <div className="dash__waveform-foot mono">
          {recentActive > 0
            ? `${recentActive} active alert${recentActive > 1 ? "s" : ""} elevating the signal`
            : "Baseline — no active alerts"}
        </div>
      </Panel>

      <div className="dash__stats">
        <StatCard label="Total Alerts" value={stats?.total_alerts ?? "—"} tone="default" />
        <StatCard label="Active" value={stats?.active_alerts ?? "—"} tone="amber" />
        <StatCard label="Mitigated" value={stats?.mitigated_alerts ?? "—"} tone="cyan" />
        <StatCard label="Avg. Confidence" value={stats ? `${(stats.avg_confidence * 100).toFixed(1)}%` : "—"} />
        <StatCard label="Active Defense Rules" value={stats?.active_defense_rules ?? "—"} tone="cyan" />
      </div>

      <div className="dash__grid">
        <Panel eyebrow="Breakdown" title="Alerts by attack type">
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={byType} margin={{ top: 4, right: 8, left: -16, bottom: 0 }}>
              <CartesianGrid stroke="var(--hairline)" vertical={false} />
              <XAxis dataKey="name" tick={{ fill: "var(--ink-dim)", fontSize: 11 }} axisLine={{ stroke: "var(--hairline)" }} tickLine={false} />
              <YAxis tick={{ fill: "var(--ink-dim)", fontSize: 11 }} axisLine={false} tickLine={false} allowDecimals={false} />
              <Tooltip contentStyle={{ background: "var(--bg-panel-raised)", border: "1px solid var(--hairline-bright)", fontSize: 12 }} />
              <Bar dataKey="count" fill="var(--signal-cyan)" radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Panel>

        <Panel eyebrow="Breakdown" title="Severity distribution">
          {bySeverity.length === 0 ? (
            <div className="dash__empty mono">No alerts yet — try the Attack Screen.</div>
          ) : (
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={bySeverity} dataKey="value" nameKey="name" innerRadius={48} outerRadius={78} paddingAngle={3}>
                  {bySeverity.map((entry) => (
                    <Cell key={entry.name} fill={SEVERITY_COLORS[entry.name] || "var(--ink-faint)"} stroke="var(--bg-panel)" />
                  ))}
                </Pie>
                <Tooltip contentStyle={{ background: "var(--bg-panel-raised)", border: "1px solid var(--hairline-bright)", fontSize: 12 }} />
              </PieChart>
            </ResponsiveContainer>
          )}
        </Panel>
      </div>

      <Panel eyebrow="Latest" title="Recent activity">
        {attacks.length === 0 ? (
          <div className="dash__empty mono">No activity logged yet.</div>
        ) : (
          <table className="dash__table">
            <thead>
              <tr>
                <th>Time</th><th>Source</th><th>Type</th><th>Severity</th><th>Model</th><th>Status</th>
              </tr>
            </thead>
            <tbody>
              {attacks.map((a) => (
                <tr key={a.id}>
                  <td className="mono">{new Date(a.timestamp).toLocaleTimeString()}</td>
                  <td className="mono">{a.source_ip}</td>
                  <td>{a.attack_type}</td>
                  <td><Badge tone={a.severity}>{a.severity}</Badge></td>
                  <td className="mono">{a.model_used}</td>
                  <td><Badge tone={a.status}>{a.status}</Badge></td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Panel>
    </div>
  );
}
