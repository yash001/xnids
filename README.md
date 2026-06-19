# xNIDS Demo Platform

A full-stack demonstration platform built around **xNIDS: Explaining
Deep Learning-based Network Intrusion Detection Systems for Active
Intrusion Responses** (Wei, Li, Zhao, Hu — USENIX Security 2023).

It reproduces the paper's core ideas (history-aware, feature-dependency-aware
explanation of DL-NIDS, and automated defense-rule generation), reproduces
one of the paper's own target detectors (a Kitsune-style autoencoder), and
adds a **from-scratch LSTM-Autoencoder** as a deep-learning alternative
designed specifically to close a gap we identified in the paper.

```
xnids-project/
├── backend/         FastAPI service: detectors, explainer, rule generator, DB
├── frontend/        React (Vite) console: Dashboard / Attack / Defence / Model Lab
└── docs/            Presentation source notes (see project root for .pptx)
```

---

## 1. Paper Analysis

### 1.1 Problem the paper addresses
Deep-Learning NIDS (DL-NIDS) detect attacks well but output little more
than a label/score. Network operators cannot safely act on a bare score:
they don't know *which* host/flow/feature triggered it, so they either
ignore it or spend significant manual effort investigating before
responding — by which time an attacker may already be inside the network.

### 1.2 The paper's proposed solution
**xNIDS** explains *any* black-box DL-NIDS by:
1. **Approximating history inputs** (Sec 3.2) — since DL-NIDS decide using
   `x_t` *and* `k` history inputs, xNIDS searches for a small, relevant
   subset of history windows instead of using a fixed window or ignoring
   history altogether.
2. **Sampling around history inputs with decay-weighted WRS** (Sec 3.3) —
   recent inputs are perturbed more heavily since they influence the
   decision more (motivated by LSTM forget-gates).
3. **Capturing feature dependencies via sparse group lasso** (Sec 3.4) —
   features are grouped by domain knowledge/correlation (e.g. all TCP-flag
   sub-features) so the surrogate model selects/excludes whole groups,
   instead of assuming feature independence like LIME/SHAP/IG/LRP.
4. **Generating actionable defense rules** (Sec 4) from the explanation:
   a *defense rule scope* (per-flow / per-host / multi-hosts) is derived
   from statistical info about the attributed flows, modified by a
   *block strategy* (passive / assertive / aggressive) and a *whitelist*,
   then translated into a unified rule and finally into OpenFlow,
   iptables, Pfsense, or Squid syntax.

### 1.3 Gap we identified
xNIDS treats the target DL-NIDS as an immutable black box and *approximates*
history-awareness from the outside via repeated sampling (Ch1 in the paper).
This is necessary because three of its four target systems (Kitsune,
RNN-IDS, AE-IDS) only weakly or shallowly model sequence history, and the
fourth (ODDS) is LSTM-based but still opaque to the explainer. Two
consequences:

- Sampling-based history approximation is comparatively expensive
  (the paper reports up to ~900ms explanation latency, Fig. 5) and its
  fidelity is bounded by how well a handful of sampled history windows can
  stand in for the model's true temporal reasoning.
- Because the detector itself was not designed with explanation in mind,
  *all* of the burden of surfacing "why" falls on the post-hoc explainer.

### 1.4 Our proposed Deep Learning solution
We add an **LSTM-Autoencoder** (`backend/app/models/lstm_autoencoder.py`)
as a second target detector, implemented from scratch in NumPy (manual
forward pass + full backpropagation-through-time, Adam-free SGD) — no
PyTorch/TensorFlow dependency required:

- **Architecture.** Encoder LSTM consumes the `k`-window history sequence;
  Decoder LSTM reconstructs it. Reconstruction error is the anomaly score
  (unsupervised, trained only on benign traffic — consistent with the
  paper's target systems).
- **Why LSTM-AE.** It directly tackles Ch1: the model *natively* attends
  over the full history sequence instead of a single window (unlike
  Kitsune), and it exposes **per-timestep, per-feature reconstruction
  error** as a free, gradient-free importance signal that doesn't require
  expensive post-hoc sampling — directly complementing (not replacing) the
  XNIDS-style explainer, which we keep for cross-validation and for
  feeding the rule-generation pipeline.
- **Training & evaluation.** Both detectors are trained on the same
  synthetic, paper-inspired traffic generator (`core/traffic_sim.py`,
  5 attack families: DDoS, Port/OS Scan, Botnet, MITM, HTTP Flood — chosen
  to match the case studies in the paper's Sec 6.3) and evaluated with
  precision/recall/F1/AUC plus a live reproduction of the paper's
  Descriptive-Accuracy fidelity test (Sec 6.1.1, Fig. 3) — see the **Model
  Lab** screen in the app.

> **Why synthetic traffic, not CICIDS2017/NSL-KDD directly?** This sandbox
> has no internet access to large dataset mirrors. The generator in
> `traffic_sim.py` is explicitly modeled on the feature semantics Kitsune
> and the paper describe (packet size/rate, TCP flags, protocol one-hots,
> port/destination entropy, HTTP referer repetition, TTL/window anomalies)
> so detection logic, explanation, and rule generation are all exercised
> realistically end-to-end. Swapping in a real PCAP-derived feature
> pipeline only requires changing `traffic_sim.py`'s output shape — every
> downstream component (detectors, explainer, rule generator, API,
> frontend) is dataset-agnostic.

---

## 2. High-Level Design (HLD)

```
┌─────────────────────────┐        ┌───────────────────────────────────────┐
│        Frontend          │        │                Backend                │
│   React + Vite (SPA)     │  HTTP  │              FastAPI                  │
│                           │◄──────►│                                       │
│  Dashboard                │  JSON  │  ┌─────────────┐   ┌────────────────┐ │
│  Attack Screen            │        │  │ traffic_sim  │   │  model_store   │ │
│  Defence Screen           │        │  │ (synthetic   │──▶│ (trains/caches │ │
│  Model Lab                │        │  │  generator)  │   │  both models)  │ │
└─────────────────────────┘        │  └─────────────┘   └────────┬───────┘ │
                                     │                              │        │
                                     │   ┌──────────────────────────▼─────┐ │
                                     │   │  Detectors                      │ │
                                     │   │  • KitsuneAutoencoder (baseline) │ │
                                     │   │  • LSTMAutoencoder (our DL alt.) │ │
                                     │   └──────────────────┬──────────────┘ │
                                     │                       │                │
                                     │   ┌──────────────────▼──────────────┐ │
                                     │   │  XNIDSExplainer                  │ │
                                     │   │  (history WRS + sparse grp lasso)│ │
                                     │   └──────────────────┬──────────────┘ │
                                     │                       │                │
                                     │   ┌──────────────────▼──────────────┐ │
                                     │   │  rule_generator                  │ │
                                     │   │  (scope, strategy, unified rule, │ │
                                     │   │   OpenFlow/iptables/Pfsense)     │ │
                                     │   └──────────────────┬──────────────┘ │
                                     │                       │                │
                                     │   ┌──────────────────▼──────────────┐ │
                                     │   │  SQLAlchemy ORM (SQLite/MySQL)   │ │
                                     │   │  AttackLog · DefenseRule · WL    │ │
                                     │   └──────────────────────────────────┘ │
                                     └───────────────────────────────────────┘
```

**Components**

| Layer | Technology | Responsibility |
|---|---|---|
| Frontend | React 19 + Vite + react-router + recharts | Dashboard, attack simulation UI, defense management UI, model comparison |
| API | FastAPI + Pydantic | REST endpoints, request validation, CORS |
| ML core | NumPy + scikit-learn | Two detectors, XNIDS-style explainer, synthetic traffic |
| Rule engine | Pure Python | Defense rule scope/strategy/translation |
| Persistence | SQLAlchemy ORM | SQLite by default; MySQL via `DATABASE_URL` |

**Data flow for one simulated attack**
1. Frontend POSTs `{attack_type, model, block_strategy}` to `/api/attacks/simulate`.
2. Backend synthesizes a `(history_len, n_features)` sequence containing a
   benign→attack transition.
3. The selected detector scores the sequence → `(label, anomaly_score, confidence)`.
4. `XNIDSExplainer` perturbs the *history* via decay-weighted WRS, fits a
   sparse-group-aware ElasticNet surrogate, returns per-feature and
   per-group importance.
5. `rule_generator` computes statistical info `S` from synthetic related
   flows, determines rule **scope**, applies the **block strategy**, and
   emits a **unified rule** + OpenFlow/iptables/Pfsense translations.
6. Everything is persisted (`AttackLog`, `DefenseRuleRecord`) and returned
   to the frontend, which renders the explanation as segmented group bars
   and the rule as copyable code blocks.

---

## 3. Low-Level Design (LLD)

### 3.1 Feature vector (23 dims) — `core/traffic_sim.py`
`pkt_size_mean, pkt_size_std, iat_mean, iat_std, byte_rate, pkt_rate,
tcp_flag_{syn,ack,fin,rst}, is_{tcp,udp,icmp,arp,http},
src_port_entropy, dst_port_entropy, unique_dst_ip_count,
unique_dst_port_count, http_referer_repeat, ttl_mean,
window_size_mean, payload_entropy`

Each sample is a `(history_len=5, 23)` sequence; attacks are injected into
a *suffix* of the sequence (1–5 of the 5 steps) so the dataset has
realistic, variable-length attack onset, mirroring the paper's Ch1
observation that "different attacks rely on different numbers of inputs."

### 3.2 KitsuneAutoencoder — `models/kitsune_model.py`
- Shallow MLP autoencoder (`23 → 8 → 23`), ReLU hidden layer, manual SGD.
- Trained on the **last window only** of benign sequences (no history) —
  intentionally reproducing the paper baseline's single-window limitation.
- Anomaly score = RMSE; threshold = 99th percentile of benign RMSE.

### 3.3 LSTMAutoencoder — `models/lstm_autoencoder.py`
- `LSTMCell` class: full gate math (`i, f, o, g`), manual forward + BPTT,
  forget-gate bias initialized > 0 (Gers et al., as cited by the paper).
- Encoder (`23 → 12`) → Decoder (`12 → 23`), sequence-to-sequence
  reconstruction across all 5 history steps.
- Anomaly score = sequence-mean RMSE; `feature_importance()` exposes a
  decay-weighted (recent-step-biased) native importance signal.

### 3.4 XNIDSExplainer — `core/explainer.py`
- `FEATURE_GROUPS`: 8 semantic groups (packet shape, timing, TCP flags,
  protocol, port entropy, destination spread, HTTP behavior, link layer).
- `_decay_weights`: Gaussian decay centered on the most recent timestep
  (Eq. 6 analogue).
- `_weighted_random_sample_history`: perturbs the *entire* history
  sequence with timestep-scaled noise (Eq. 7 analogue).
- Surrogate fit: `sklearn.linear_model.ElasticNet` on flattened,
  decay-weighted samples, approximating sparse group lasso's group +
  feature sparsity (a from-scratch BCD solver matching Eq. 13-16 is also
  included as `_sparse_group_lasso_fallback` for reference).
- `compute_statistical_info`: builds Table 2's `S = {IP_pool, IP_n,
  MAC_n, Port_n, Protocol_n}` from attributed flows.

### 3.5 rule_generator — `core/rule_generator.py`
- `determine_scope(S)`: per-flow / per-host / multi-hosts, by the
  dominant-field logic in Sec 4.1.
- `apply_block_strategy(scope, strategy)`: passive → always `drop_flow`;
  aggressive → always `drop_host`; assertive → scope-determined default.
- `UnifiedRule` dataclass with `.to_openflow() / .to_iptables() /
  .to_pfsense()` — Appendix B syntax, Table 9 capability matrix.

### 3.6 REST API — `routers/*.py`

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/detect_attack` | Single-packet inference (spec example 1) |
| POST | `/api/attacks/simulate` | Full simulate → detect → explain → defend pipeline |
| GET | `/api/attacks` | Attack log (spec example 2 shape) |
| POST | `/api/attacks/{id}/mitigate` | Mark alert mitigated |
| GET | `/api/defense/rules` | List generated defense rules |
| POST | `/api/defense/rules/{id}/toggle` | Enforce/lift a rule |
| GET/POST/DELETE | `/api/defense/whitelist` | Whitelist CRUD (Sec 4.2) |
| GET | `/api/dashboard/stats` | Aggregate counts for Dashboard |
| GET | `/api/dashboard/model_comparison` | Live ADA-style fidelity curve, Kitsune vs LSTM-AE |

### 3.7 Database schema
- `attack_log(id, timestamp, src_ip, dst_ip, src_mac, dst_port, protocol,
  attack_type, severity, status, model_used, confidence, anomaly_score,
  top_features_json)`
- `defense_rule(id, timestamp, attack_log_id, scope, strategy,
  openflow_rule, iptables_rule, pfsense_rule, active)`
- `whitelist(id, ip, label, created_at)`

Both the **database itself** and these tables are created automatically
at startup (`core/database.py: init_db()` → `ensure_database_exists()` +
`Base.metadata.create_all()`) — no manual `CREATE DATABASE` or migration
step is needed for either SQLite or MySQL. See Section 4.4.

### 3.8 Frontend structure
```
src/
├── lib/api.js              fetch wrapper, one function per endpoint
├── components/
│   ├── Shell.jsx/.css      sidebar nav + page frame
│   ├── Waveform.jsx        signature oscilloscope-style live signal
│   ├── GroupBars.jsx/.css  sparse-group-lasso importance visualization
│   └── ui.jsx/.css         Panel, StatCard, Badge, Button, Select
└── pages/
    ├── Dashboard.jsx        stats, charts, recent activity
    ├── AttackScreen.jsx      simulate attacks, view explanation + rule
    ├── DefenceScreen.jsx     manage rules + whitelist
    └── ModelLab.jsx          Kitsune vs LSTM-AE comparison & gap narrative
```

**Design language ("SIGNAL" console).** Dark charcoal/navy background,
phosphor-amber for "active/alert" state, cyan for "defended/calm" state,
red for high severity — a signal-analysis console aesthetic (oscilloscope
trace, segmented group bars) chosen because the subject matter *is*
signal detection and explanation, not a generic admin template.

---

## 4. Setup & Execution

### 4.1 Backend

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate     # optional but recommended
pip install -r requirements.txt

# (optional) use MySQL instead of the default SQLite — no manual
# database/table creation needed, see Section 4.4:
# export DATABASE_URL="mysql+pymysql://user:password@localhost:3306/xnids"

uvicorn app.main:app --reload --port 8000
```
On first startup the app trains both detectors (~5-10s) and caches them to
`backend/app/trained_models.pkl`. Delete that file to force retraining
(e.g. after editing `traffic_sim.py`).

Verify: `curl http://localhost:8000/api/health` → `{"status":"healthy","models_ready":true}`

Interactive API docs: `http://localhost:8000/docs`

### 4.2 Frontend

```bash
cd frontend
npm install
cp .env.example .env   # adjust VITE_API_URL if the backend isn't on :8000
npm run dev            # http://localhost:5173
```

Production build: `npm run build` → output in `frontend/dist/`.

### 4.3 Quick smoke test
1. Start backend, then frontend.
2. Open `http://localhost:5173` → Dashboard should show "Detectors online".
3. Go to **Attack Screen** → pick "DDoS — SYN flood", detector "LSTM-Autoencoder",
   strategy "Assertive" → **Launch simulation**.
4. Confirm an explanation (group bars) and a generated OpenFlow/iptables/Pfsense
   rule appear, and the alert shows up on the Dashboard and Defence Screen.
5. Go to **Model Lab** to see the Kitsune-vs-LSTM-AE fidelity curve and recall
   comparison.

### 4.4 Switching to MySQL

No manual setup required — the app creates both the **database** and its
**tables** automatically on startup. Just point it at a MySQL server and a
user that's allowed to create databases:

```bash
export DATABASE_URL="mysql+pymysql://xnids_user:password@localhost:3306/xnids"
uvicorn app.main:app --reload --port 8000
```

On startup, `init_db()` (in `core/database.py`) does two things, in order:

1. **`ensure_database_exists()`** — opens a throwaway connection to the
   MySQL *server* (no database selected) and runs
   `CREATE DATABASE IF NOT EXISTS xnids CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci`,
   then disposes that connection. This means the `xnids` database itself
   does **not** need to exist beforehand — only the server and a user
   with `CREATE` privilege (or full privileges) do.
2. **`Base.metadata.create_all()`** — creates `attack_log`, `defense_rule`,
   and `whitelist` inside that database if they don't already exist.

If you'd rather create the database yourself (e.g. a shared MySQL
instance where your user can't run `CREATE DATABASE`), that's still fine
— step 1 simply finds it already exists and moves on:

```sql
CREATE DATABASE xnids CHARACTER SET utf8mb4;
```

This auto-creation only applies to MySQL. SQLite (the default) never
needed it — the `.db` file is created automatically on first connection.


---

## 5. Limitations & Honest Notes

- Both detectors are trained on **synthetic, paper-inspired traffic**, not
  CICIDS2017/NSL-KDD directly (no internet access to dataset mirrors in
  this environment). The architectures, training procedures, and
  explanation/rule-generation pipeline are real and fully wired end to
  end; swapping in real PCAP-derived features is a drop-in change to
  `traffic_sim.py`'s output contract.
- The LSTM-Autoencoder is implemented from scratch in NumPy (not
  PyTorch/TensorFlow) to keep the dependency footprint minimal in this
  sandbox; it is trained with real backpropagation-through-time, not a
  relabeled shallow model.
- The XNIDS explainer here is a faithful **simplification**: it uses
  ElasticNet as a fast stand-in for the paper's sparse-group-lasso BCD
  solver (a from-scratch BCD implementation is included for reference but
  not used by default), and the "fidelity test" in the Model Lab is a
  simplified single-run reproduction of Fig. 3, not the full multi-seed
  evaluation in the paper.

## 6. Team Member Contributions

See `presentation.pptx`, final slide, for the contributions breakdown.
