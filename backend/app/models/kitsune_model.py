"""
kitsune_model.py
==================
Re-implementation of the *paper's* baseline detection paradigm: an
autoencoder-based DL-NIDS in the style of Kitsune [Mirsky et al., NDSS'18],
which is one of the four target DL-NIDS evaluated by xNIDS (Sec 5,
Appendix C.1). xNIDS treats this model (and three others) as a black box
and explains *why* it raises an alert -- it does not change how the
model itself detects intrusions.

For this project we reproduce the essential idea: a single
under-complete autoencoder trained ONLY on benign traffic. Reconstruction
error (RMSE) on unseen input is the anomaly score; large error => attack.
This mirrors KitNET's ensemble-of-autoencoders concept but uses a single
compact autoencoder for clarity, trained on flattened per-window features
(no history) -- exactly the limitation the xNIDS paper identifies as Ch1
("DL-NIDS make decisions using k history inputs, but most explanation/
analysis treats a single xt").
"""

import numpy as np

from ..core.traffic_sim import N_FEATURES


class KitsuneAutoencoder:
    """A minimal shallow autoencoder (encoder-decoder MLP) trained with
    numpy + manual gradient descent. No external DL framework required.
    """

    def __init__(self, input_dim=N_FEATURES, hidden_dim=8, seed=7):
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / input_dim)
        self.W1 = rng.normal(0, scale, size=(input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.normal(0, scale, size=(hidden_dim, input_dim))
        self.b2 = np.zeros(input_dim)
        self.threshold_ = None
        self.mu_ = None
        self.sigma_ = None

    @staticmethod
    def _relu(x):
        return np.maximum(0, x)

    @staticmethod
    def _relu_grad(x):
        return (x > 0).astype(x.dtype)

    def _forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = self._relu(z1)
        z2 = a1 @ self.W2 + self.b2
        return z1, a1, z2

    def reconstruct(self, X):
        _, _, z2 = self._forward(X)
        return z2

    def anomaly_score(self, X):
        """Per-sample RMSE reconstruction error."""
        recon = self.reconstruct(X)
        return np.sqrt(np.mean((X - recon) ** 2, axis=-1))

    def fit(self, X_benign, epochs=120, lr=0.05, batch_size=128, percentile=99.0, verbose=False):
        self.mu_ = X_benign.mean(axis=0)
        self.sigma_ = X_benign.std(axis=0) + 1e-6
        Xn = (X_benign - self.mu_) / self.sigma_

        n = Xn.shape[0]
        for epoch in range(epochs):
            perm = np.random.permutation(n)
            epoch_loss = 0.0
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                xb = Xn[idx]
                z1, a1, z2 = self._forward(xb)
                err = z2 - xb
                loss = np.mean(err ** 2)
                epoch_loss += loss * len(idx)

                # backprop
                dz2 = 2 * err / xb.shape[0]
                dW2 = a1.T @ dz2
                db2 = dz2.sum(axis=0)
                da1 = dz2 @ self.W2.T
                dz1 = da1 * self._relu_grad(z1)
                dW1 = xb.T @ dz1
                db1 = dz1.sum(axis=0)

                self.W1 -= lr * dW1
                self.b1 -= lr * db1
                self.W2 -= lr * dW2
                self.b2 -= lr * db2
            if verbose and epoch % 20 == 0:
                print(f"[Kitsune-AE] epoch {epoch} loss={epoch_loss / n:.5f}")

        # set detection threshold from benign reconstruction error distribution
        scores = self.anomaly_score(X_benign)
        self.threshold_ = float(np.percentile(scores, percentile))
        return self

    def predict(self, X):
        """Returns (labels, scores, confidences). label 1 = attack."""
        Xn = (X - self.mu_) / self.sigma_
        scores = self.anomaly_score(Xn)
        labels = (scores > self.threshold_).astype(int)
        # confidence: distance from threshold, squashed to [0.5, 0.99]
        margin = (scores - self.threshold_) / (self.threshold_ + 1e-6)
        conf = 0.5 + 0.49 * (1 / (1 + np.exp(-3 * margin)))
        return labels, scores, conf
