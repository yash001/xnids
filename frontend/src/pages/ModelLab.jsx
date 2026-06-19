import { useEffect, useState } from "react";
import { api } from "../lib/api";
import { Panel } from "../components/ui";
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, Legend,
  BarChart, Bar,
} from "recharts";
import "./ModelLab.css";

export default function ModelLab() {
  const [data, setData] = useState(null);

  useEffect(() => {
    api.modelComparison().then(setData).catch(() => {});
  }, []);

  const adaChart = data
    ? data.ada_curve_x.map((x, i) => ({
        x,
        kitsune: data.results.kitsune.ada_curve[i],
        lstm_ae: data.results.lstm_ae.ada_curve[i],
      }))
    : [];

  const recallChart = data
    ? Object.entries(data.per_attack_recall).map(([type, v]) => ({
        type,
        kitsune: +(v.kitsune_recall * 100).toFixed(1),
        lstm_ae: +(v.lstm_ae_recall * 100).toFixed(1),
      }))
    : [];

  return (
    <div className="lab">
      <header className="lab__header">
        <div className="eyebrow">Gap Analysis & DL Alternative</div>
        <h1 className="lab__title">Model Lab</h1>
      </header>

      <div className="lab__cards">
        <Panel eyebrow="Paper baseline" title="Kitsune-style Autoencoder" className="lab__card">
          <p>Reproduces the detection paradigm of one of xNIDS's four target DL-NIDS: a shallow autoencoder trained on benign traffic, scored by reconstruction error on a <em>single current window</em>.</p>
          {data && (
            <dl className="lab__metrics">
              <dt>Precision</dt><dd>{(data.results.kitsune.metrics.precision * 100).toFixed(1)}%</dd>
              <dt>Recall</dt><dd>{(data.results.kitsune.metrics.recall * 100).toFixed(1)}%</dd>
              <dt>F1</dt><dd>{(data.results.kitsune.metrics.f1 * 100).toFixed(1)}%</dd>
              <dt>AUC</dt><dd>{data.results.kitsune.metrics.auc?.toFixed(3)}</dd>
            </dl>
          )}
        </Panel>

        <Panel eyebrow="Our proposal" title="LSTM-Autoencoder" className="lab__card lab__card--highlight">
          <p>Sequence-to-sequence LSTM autoencoder reconstructing the <em>full history window</em>, with native per-timestep error exposed as an importance signal — directly targeting the paper's Ch1 history-input gap instead of only approximating it post-hoc.</p>
          {data && (
            <dl className="lab__metrics">
              <dt>Precision</dt><dd>{(data.results.lstm_ae.metrics.precision * 100).toFixed(1)}%</dd>
              <dt>Recall</dt><dd>{(data.results.lstm_ae.metrics.recall * 100).toFixed(1)}%</dd>
              <dt>F1</dt><dd>{(data.results.lstm_ae.metrics.f1 * 100).toFixed(1)}%</dd>
              <dt>AUC</dt><dd>{data.results.lstm_ae.metrics.auc?.toFixed(3)}</dd>
            </dl>
          )}
        </Panel>
      </div>

      <Panel eyebrow="Fidelity test (paper Sec 6.1.1, Fig. 3)" title="Descriptive-accuracy-style ablation curve">
        <p className="lab__note">
          We zero out the top-k features the explainer ranks as most important and re-score
          the same malicious sequences. A steeper decline means the explanation is more faithful
          to what the detector actually relies on.
        </p>
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={adaChart} margin={{ top: 8, right: 16, left: -10, bottom: 0 }}>
            <CartesianGrid stroke="var(--hairline)" />
            <XAxis dataKey="x" tick={{ fill: "var(--ink-dim)", fontSize: 11 }} axisLine={{ stroke: "var(--hairline)" }} tickLine={false}
                   label={{ value: "features zeroed", position: "insideBottom", offset: -4, fill: "var(--ink-faint)", fontSize: 11 }} />
            <YAxis tick={{ fill: "var(--ink-dim)", fontSize: 11 }} axisLine={false} tickLine={false} />
            <Tooltip contentStyle={{ background: "var(--bg-panel-raised)", border: "1px solid var(--hairline-bright)", fontSize: 12 }} />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <Line type="monotone" dataKey="kitsune" name="Kitsune-AE" stroke="var(--signal-amber)" strokeWidth={2} dot={false} />
            <Line type="monotone" dataKey="lstm_ae" name="LSTM-AE" stroke="var(--signal-cyan)" strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </Panel>

      <Panel eyebrow="Per-attack recall" title="Detection recall by attack family">
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={recallChart} margin={{ top: 8, right: 16, left: -10, bottom: 0 }}>
            <CartesianGrid stroke="var(--hairline)" vertical={false} />
            <XAxis dataKey="type" tick={{ fill: "var(--ink-dim)", fontSize: 11 }} axisLine={{ stroke: "var(--hairline)" }} tickLine={false} />
            <YAxis tick={{ fill: "var(--ink-dim)", fontSize: 11 }} axisLine={false} tickLine={false} unit="%" />
            <Tooltip contentStyle={{ background: "var(--bg-panel-raised)", border: "1px solid var(--hairline-bright)", fontSize: 12 }} />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <Bar dataKey="kitsune" name="Kitsune-AE" fill="var(--signal-amber)" radius={[3,3,0,0]} />
            <Bar dataKey="lstm_ae" name="LSTM-AE" fill="var(--signal-cyan)" radius={[3,3,0,0]} />
          </BarChart>
        </ResponsiveContainer>
      </Panel>

      <Panel eyebrow="Why this addresses the gap" title="Summary">
        <ol className="lab__gap-list">
          <li><strong>Gap identified:</strong> xNIDS explains existing DL-NIDS post-hoc by sampling around approximated history inputs (Sec 3.2-3.3) because the target models (Kitsune, ODDS, RNN-IDS, AE-IDS) don't natively expose history-aware importance.</li>
          <li><strong>Our response:</strong> an LSTM-Autoencoder that reconstructs the entire history sequence end-to-end, so per-timestep / per-feature reconstruction error is a first-class, gradient-free signal — reducing reliance on expensive sampling-based approximation for the history dimension.</li>
          <li><strong>Still complementary:</strong> we keep the XNIDS-style sparse-group-lasso explainer (Sec 3.4) for cross-checking and for the rule-generation pipeline (Sec 4), since group-level feature dependency reasoning is orthogonal to the detector architecture.</li>
        </ol>
      </Panel>
    </div>
  );
}
