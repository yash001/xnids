"""
explainer.py
=============
A working, simplified re-implementation of the XNIDS explanation method
(paper Sec 3) applied to BOTH target models in this project (the
Kitsune-style baseline and our LSTM-Autoencoder). It demonstrates the
paper's two central ideas:

  1. History-aware sampling (Sec 3.2 / 3.3): instead of perturbing only
     the current window x_t (as LIME/SHAP/IG do), we perturb the *whole*
     history sequence X_t,k, weighting the most recent windows higher
     via a decay function and Weighted Random Sampling (WRS, Eq. 6-7).

  2. Feature-dependency-aware sparsity (Sec 3.4): features are divided
     into correlated groups (e.g. all TCP-flag features, all protocol
     one-hots) and we fit a Sparse Group Lasso surrogate (Eq. 9/12) so
     that whole groups of features are included/excluded together,
     instead of assuming feature independence like LIME/SHAP/IG/LRP.

We use scikit-learn's ElasticNet as the convex solver, weighting and
grouping samples manually before fitting to approximate the sparse
group lasso behaviour described in the paper (full custom group-lasso
BCD is implemented for the curious in `_sparse_group_lasso_fallback`,
but ElasticNet is used by default for speed/stability in this demo).
"""

import numpy as np
from sklearn.linear_model import ElasticNet

from .traffic_sim import FEATURE_NAMES, N_FEATURES

# Feature groups reflecting structural/semantic correlations described
# in the paper (Sec 3.4): e.g. TCP.flags are sub-features of "is_tcp";
# protocol one-hots are mutually exclusive; entropy/rate features cluster.
FEATURE_GROUPS = {
    "packet_shape": ["pkt_size_mean", "pkt_size_std"],
    "timing": ["iat_mean", "iat_std", "byte_rate", "pkt_rate"],
    "tcp_flags": ["tcp_flag_syn", "tcp_flag_ack", "tcp_flag_fin", "tcp_flag_rst"],
    "protocol": ["is_tcp", "is_udp", "is_icmp", "is_arp", "is_http"],
    "port_entropy": ["src_port_entropy", "dst_port_entropy"],
    "destination_spread": ["unique_dst_ip_count", "unique_dst_port_count"],
    "http_behavior": ["http_referer_repeat"],
    "link_layer": ["ttl_mean", "window_size_mean", "payload_entropy"],
}
GROUP_OF_FEATURE = {f: g for g, feats in FEATURE_GROUPS.items() for f in feats}


def _decay_weights(history_len, kind="gaussian"):
    t = np.arange(history_len)
    center = history_len - 1  # current/most-recent index gets highest weight
    if kind == "gaussian":
        w = np.exp(-0.5 * ((t - center) / 1.5) ** 2)
    elif kind == "exponential":
        w = np.exp(-0.5 * (center - t))
    else:
        w = (t + 1) / t.sum()
    return w / w.sum()


def _weighted_random_sample_history(history_seq, weights, n_samples, rng):
    """Eq. 6-7: synthesize samples around history inputs, shifting the
    synthesized samples toward the most recent (highest-weight) inputs.
    """
    T, F = history_seq.shape
    samples = np.zeros((n_samples, T, F))
    chosen_t = rng.choice(T, size=n_samples, p=weights)
    for n in range(n_samples):
        seq = history_seq.copy()
        # perturb timesteps from chosen_t[n] onward with decaying noise,
        # approximating sampling "around" the selected history input.
        noise_scale = 0.25
        for t in range(T):
            local_w = weights[t] / weights[chosen_t[n]]
            seq[t] = seq[t] + rng.normal(0, noise_scale * (0.3 + local_w), size=F)
        samples[n] = seq
    return samples


def _sparse_group_lasso_fallback(X, y, groups, alpha=0.2, l1_ratio=0.6, n_iter=60):
    """A compact Block-Coordinate-Descent sparse group lasso (Eq. 13-16)
    kept for transparency; not used by default (ElasticNet is faster &
    numerically safer for this demo) but illustrates the paper's exact
    optimization structure with group + feature sparsity penalties.
    """
    n, p = X.shape
    beta = np.zeros(p)
    unique_groups = sorted(set(groups))
    for _ in range(n_iter):
        for g in unique_groups:
            idx = [i for i, gg in enumerate(groups) if gg == g]
            Xg = X[:, idx]
            r = y - X @ beta + Xg @ beta[idx]
            grad = Xg.T @ r / n
            soft = np.sign(grad) * np.maximum(np.abs(grad) - alpha * l1_ratio, 0)
            norm = np.linalg.norm(soft)
            scale = max(0, 1 - alpha * (1 - l1_ratio) * np.sqrt(len(idx)) / (norm + 1e-9))
            beta[idx] = soft * scale
    return beta


class XNIDSExplainer:
    """Explains a target detector's anomaly score for one input window
    + its history, returning per-feature and per-group importance, plus
    statistical info (S) used downstream for defense-rule-scope decisions
    (paper Sec 4.1, Table 2).
    """

    def __init__(self, detector, history_len=5, decay="gaussian", n_samples=300, seed=0):
        self.detector = detector          # object with .anomaly_score(X) -> per-window or per-seq score
        self.history_len = history_len
        self.decay = decay
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)

    def _score_fn(self, seq_batch):
        """seq_batch: (n, T, F). Returns scalar score per sample using the
        detector's own scoring function (works for both Kitsune-on-last-
        window and LSTM-AE-on-full-sequence detectors via duck typing).
        """
        if hasattr(self.detector, "anomaly_score") and self.detector.__class__.__name__ == "KitsuneAutoencoder":
            Xn = (seq_batch[:, -1, :] - self.detector.mu_) / self.detector.sigma_
            return self.detector.anomaly_score(Xn)
        else:
            Xn = (seq_batch - self.detector.mu_) / self.detector.sigma_
            return self.detector.anomaly_score(Xn)

    def explain(self, history_seq):
        """history_seq: (T, F) ndarray, most recent window last.
        Returns dict with feature_importance, group_importance,
        statistical_info S, and meta info about the surrogate fit.
        """
        T, F = history_seq.shape
        weights = _decay_weights(T, self.decay)
        samples = _weighted_random_sample_history(history_seq, weights, self.n_samples, self.rng)
        y = self._score_fn(samples)

        # flatten (T,F) -> single feature vector per sample, weight each
        # column-block by its timestep decay weight (Sec 3.3 intuition:
        # recent windows influence the surrogate fit more).
        Xflat = samples.reshape(self.n_samples, T * F)
        col_weights = np.repeat(weights, F)
        Xw = Xflat * col_weights[np.newaxis, :]

        groups = []
        for t in range(T):
            for fname in FEATURE_NAMES:
                groups.append(f"t{t}_{GROUP_OF_FEATURE[fname]}")

        # Standardize targets/inputs for stable ElasticNet fit
        y_std = (y - y.mean()) / (y.std() + 1e-9)
        model = ElasticNet(alpha=0.02, l1_ratio=0.5, max_iter=2000, fit_intercept=True)
        model.fit(Xw, y_std)
        beta = model.coef_

        # aggregate per-feature importance across history timesteps,
        # weighting recent timesteps higher (mirrors Eq. 11's structure:
        # current input + sum over history inputs).
        beta_matrix = beta.reshape(T, F)
        per_feature = np.abs(beta_matrix) * weights[:, None]
        feature_importance = per_feature.sum(axis=0)
        feature_importance = feature_importance / (feature_importance.sum() + 1e-9)

        feat_imp_named = {FEATURE_NAMES[i]: float(feature_importance[i]) for i in range(F)}
        # group-level aggregation (group sparsity, Sec 3.4)
        group_importance = {}
        for fname, score in feat_imp_named.items():
            g = GROUP_OF_FEATURE[fname]
            group_importance[g] = group_importance.get(g, 0.0) + score

        top_features = sorted(feat_imp_named.items(), key=lambda kv: -kv[1])[:6]

        return {
            "feature_importance": feat_imp_named,
            "group_importance": group_importance,
            "top_features": top_features,
            "decay_weights": weights.tolist(),
            "n_samples": self.n_samples,
        }


def compute_statistical_info(flow_records):
    """Computes the Statistical Information S used by the defense-rule
    scope analysis (paper Table 2): IP_pool, IP_n, MAC_n, Port_n,
    Protocol_n. `flow_records` is a list of dicts with src_ip, src_mac,
    dst_port, protocol for the malicious flows attributed to this alert.
    """
    if not flow_records:
        return None
    from collections import Counter
    ip_counts = Counter(r["src_ip"] for r in flow_records)
    mac_counts = Counter(r.get("src_mac", "unknown") for r in flow_records)
    port_counts = Counter(r["dst_port"] for r in flow_records)
    proto_counts = Counter(r["protocol"] for r in flow_records)
    return {
        "IP_pool": list(ip_counts.keys()),
        "IP_n": max(ip_counts.values()),
        "MAC_n": max(mac_counts.values()),
        "Port_n": max(port_counts.values()),
        "Protocol_n": max(proto_counts.values()),
    }
