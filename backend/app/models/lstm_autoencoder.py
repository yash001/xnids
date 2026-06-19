"""
lstm_autoencoder.py
=====================
Our proposed Deep-Learning alternative addressing the gap we identified
in the xNIDS paper (see docs/README.md "Gap Analysis"): xNIDS explains
*existing* DL-NIDS (Kitsune, ODDS, RNN-IDS, AE-IDS) post-hoc, but those
target models themselves only use a *fixed, shallow* notion of history
(a sliding window or a single-layer RNN), and the underlying detectors
are not natively designed to (a) jointly model sequential dependence
AND (b) expose attention-style importance over history inputs that an
explainer could exploit directly, instead of approximating it
post-hoc with sampling (Sec 3.2-3.3 of the paper).

We implement a compact LSTM-Autoencoder (sequence-to-sequence) trained
on benign traffic only:
  - Encoder LSTM compresses a window of `history_len` feature vectors
    into a latent state.
  - Decoder LSTM reconstructs the sequence from the latent state.
  - Reconstruction error across the sequence is the anomaly score.
  - We additionally expose per-timestep, per-feature reconstruction
    error as a *native* (gradient-free) importance signal -- giving
    network operators a head start that complements our XNIDS-inspired
    explainer (see core/explainer.py), directly tackling Ch1 (history
    inputs) since the model already attends over the full sequence
    instead of needing an explainer to approximate it.

This is a from-scratch implementation (numpy only, manual BPTT) so the
project has zero heavy DL-framework dependency while remaining a true
recurrent deep network (multiple LSTM gates, nonlinear activations,
trained via gradient descent), not a relabeled MLP.
"""

import numpy as np

from ..core.traffic_sim import N_FEATURES


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class LSTMCell:
    """A single LSTM layer with manual forward/backward (BPTT)."""

    def __init__(self, input_dim, hidden_dim, seed=0):
        rng = np.random.default_rng(seed)
        z = input_dim + hidden_dim
        scale = np.sqrt(1.0 / z)
        # combined weight matrices for gates: i, f, o, g (candidate)
        self.Wi = rng.normal(0, scale, (z, hidden_dim))
        self.Wf = rng.normal(0, scale, (z, hidden_dim))
        self.Wo = rng.normal(0, scale, (z, hidden_dim))
        self.Wg = rng.normal(0, scale, (z, hidden_dim))
        self.bi = np.zeros(hidden_dim)
        self.bf = np.ones(hidden_dim) * 0.5  # forget-gate bias init > 0 helps memory (Gers et al.)
        self.bo = np.zeros(hidden_dim)
        self.bg = np.zeros(hidden_dim)
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

    def forward(self, X):
        """X: (batch, T, input_dim) -> caches list, outputs (batch,T,hidden)"""
        b, T, _ = X.shape
        h = np.zeros((b, self.hidden_dim))
        c = np.zeros((b, self.hidden_dim))
        cache = []
        H = np.zeros((b, T, self.hidden_dim))
        for t in range(T):
            xt = X[:, t, :]
            z = np.concatenate([xt, h], axis=1)
            i = sigmoid(z @ self.Wi + self.bi)
            f = sigmoid(z @ self.Wf + self.bf)
            o = sigmoid(z @ self.Wo + self.bo)
            g = np.tanh(z @ self.Wg + self.bg)
            c_new = f * c + i * g
            h_new = o * np.tanh(c_new)
            cache.append((z, i, f, o, g, c, c_new, h_new))
            c, h = c_new, h_new
            H[:, t, :] = h
        return H, cache

    def backward(self, dH, cache, lr):
        """dH: (batch, T, hidden) gradient w.r.t outputs at every timestep."""
        b, T, _ = dH.shape
        dWi = np.zeros_like(self.Wi); dWf = np.zeros_like(self.Wf)
        dWo = np.zeros_like(self.Wo); dWg = np.zeros_like(self.Wg)
        dbi = np.zeros_like(self.bi); dbf = np.zeros_like(self.bf)
        dbo = np.zeros_like(self.bo); dbg = np.zeros_like(self.bg)

        dh_next = np.zeros((b, self.hidden_dim))
        dc_next = np.zeros((b, self.hidden_dim))
        dX = np.zeros((b, T, self.input_dim))

        for t in reversed(range(T)):
            z, i, f, o, g, c_prev, c_new, h_new = cache[t]
            dh = dH[:, t, :] + dh_next
            do = dh * np.tanh(c_new)
            dc = dh * o * (1 - np.tanh(c_new) ** 2) + dc_next
            di = dc * g
            dg = dc * i
            df = dc * c_prev

            di_raw = di * i * (1 - i)
            df_raw = df * f * (1 - f)
            do_raw = do * o * (1 - o)
            dg_raw = dg * (1 - g ** 2)

            dWi += z.T @ di_raw; dbi += di_raw.sum(0)
            dWf += z.T @ df_raw; dbf += df_raw.sum(0)
            dWo += z.T @ do_raw; dbo += do_raw.sum(0)
            dWg += z.T @ dg_raw; dbg += dg_raw.sum(0)

            dz = (di_raw @ self.Wi.T + df_raw @ self.Wf.T +
                  do_raw @ self.Wo.T + dg_raw @ self.Wg.T)
            dx = dz[:, :self.input_dim]
            dh_prev = dz[:, self.input_dim:]

            dX[:, t, :] = dx
            dh_next = dh_prev
            dc_next = dc * f

        for p, dp in [(self.Wi, dWi), (self.Wf, dWf), (self.Wo, dWo), (self.Wg, dWg),
                      (self.bi, dbi), (self.bf, dbf), (self.bo, dbo), (self.bg, dbg)]:
            p -= lr * dp / b
        return dX


class LSTMAutoencoder:
    """Encoder-decoder LSTM trained to reconstruct benign sequences.
    Anomaly score = mean squared reconstruction error over the sequence.
    """

    def __init__(self, input_dim=N_FEATURES, hidden_dim=16, seed=11):
        self.encoder = LSTMCell(input_dim, hidden_dim, seed=seed)
        self.decoder = LSTMCell(hidden_dim, input_dim, seed=seed + 1)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.threshold_ = None
        self.mu_ = None
        self.sigma_ = None

    def _forward(self, X):
        H_enc, cache_enc = self.encoder.forward(X)
        latent_seq = H_enc  # use full encoder hidden sequence as decoder input
        H_dec, cache_dec = self.decoder.forward(latent_seq)
        return H_dec, cache_enc, cache_dec, H_enc

    def reconstruct(self, X):
        H_dec, *_ = self._forward(X)
        return H_dec

    def per_step_error(self, X):
        recon = self.reconstruct(X)
        return (X - recon) ** 2  # (batch, T, features)

    def anomaly_score(self, X):
        err = self.per_step_error(X)
        return np.sqrt(np.mean(err, axis=(1, 2)))

    def fit(self, X_benign, epochs=60, lr=0.03, batch_size=64, percentile=99.0, verbose=False):
        b0 = X_benign.shape[0]
        self.mu_ = X_benign.mean(axis=(0, 1))
        self.sigma_ = X_benign.std(axis=(0, 1)) + 1e-6
        Xn = (X_benign - self.mu_) / self.sigma_

        for epoch in range(epochs):
            perm = np.random.permutation(b0)
            epoch_loss = 0.0
            for i in range(0, b0, batch_size):
                idx = perm[i:i + batch_size]
                xb = Xn[idx]
                H_dec, cache_enc, cache_dec, H_enc = self._forward(xb)
                err = H_dec - xb
                loss = np.mean(err ** 2)
                epoch_loss += loss * len(idx)

                dH_dec = 2 * err / (xb.shape[0] * xb.shape[1] * xb.shape[2])
                dLatent = self.decoder.backward(dH_dec, cache_dec, lr)
                # gradient flows from decoder input (=encoder hidden seq) back into encoder
                self.encoder.backward(dLatent, cache_enc, lr)
            if verbose and epoch % 10 == 0:
                print(f"[LSTM-AE] epoch {epoch} loss={epoch_loss / b0:.5f}")

        scores = self.anomaly_score(Xn)
        self.threshold_ = float(np.percentile(scores, percentile))
        return self

    def predict(self, X):
        Xn = (X - self.mu_) / self.sigma_
        scores = self.anomaly_score(Xn)
        labels = (scores > self.threshold_).astype(int)
        margin = (scores - self.threshold_) / (self.threshold_ + 1e-6)
        conf = 0.5 + 0.49 * (1 / (1 + np.exp(-3 * margin)))
        return labels, scores, conf

    def feature_importance(self, X):
        """Native importance: per-feature contribution to reconstruction
        error, averaged with higher weight on later (closer to t)
        timesteps -- mirrors xNIDS's intuition (Sec 3.3) that recent
        history inputs influence the decision more than older ones.
        """
        Xn = (X - self.mu_) / self.sigma_
        err = self.per_step_error(Xn)  # (batch, T, F)
        T = err.shape[1]
        decay = np.exp(-0.5 * np.arange(T)[::-1])  # most recent weighted highest
        decay = decay / decay.sum()
        weighted = np.einsum('btf,t->bf', err, decay)
        return weighted  # (batch, F) importance score per feature
