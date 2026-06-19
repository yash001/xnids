"""
model_store.py
================
Singleton-style holder that trains (once) and exposes both target
detectors -- the paper-baseline Kitsune-style autoencoder and our
LSTM-Autoencoder DL alternative -- plus an XNIDSExplainer bound to
whichever model is currently selected for an inference request.

Training happens at FastAPI startup. With the dataset sizes used here
(~6k sequences) training both models takes a few seconds on CPU.
"""

import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from .traffic_sim import generate_dataset, ATTACK_TYPES, FEATURE_NAMES
from ..models.kitsune_model import KitsuneAutoencoder
from ..models.lstm_autoencoder import LSTMAutoencoder
from .explainer import XNIDSExplainer

CACHE_PATH = Path(__file__).resolve().parent.parent / "trained_models.pkl"


class ModelStore:
    def __init__(self):
        self.kitsune: Optional[KitsuneAutoencoder] = None
        self.lstm_ae: Optional[LSTMAutoencoder] = None
        self.metrics: dict = {}
        self.ready = False

    def train_or_load(self, force_retrain=False):
        if not force_retrain and CACHE_PATH.exists():
            try:
                with open(CACHE_PATH, "rb") as f:
                    data = pickle.load(f)
                self.kitsune = data["kitsune"]
                self.lstm_ae = data["lstm_ae"]
                self.metrics = data["metrics"]
                self.ready = True
                return
            except Exception:
                pass  # fall through to retrain on any cache issue

        X, y, atk = generate_dataset(n_benign=1500, n_attack_per_type=250, history_len=5, seed=42)
        benign_mask = y == 0

        kit = KitsuneAutoencoder(hidden_dim=8).fit(X[benign_mask][:, -1, :], epochs=100, lr=0.08, verbose=False)
        lstm = LSTMAutoencoder(hidden_dim=12).fit(X[benign_mask], epochs=50, lr=0.05, verbose=False)

        kit_labels, kit_scores, _ = kit.predict(X[:, -1, :])
        lstm_labels, lstm_scores, _ = lstm.predict(X)

        from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
        kp, kr, kf, _ = precision_recall_fscore_support(y, kit_labels, average="binary", zero_division=0)
        lp, lr_, lf, _ = precision_recall_fscore_support(y, lstm_labels, average="binary", zero_division=0)
        try:
            k_auc = roc_auc_score(y, kit_scores)
            l_auc = roc_auc_score(y, lstm_scores)
        except Exception:
            k_auc, l_auc = float("nan"), float("nan")

        per_attack = {}
        for t in ATTACK_TYPES:
            mask = atk == t
            per_attack[t] = {
                "kitsune_recall": float((kit_labels[mask] == 1).mean()) if mask.any() else None,
                "lstm_ae_recall": float((lstm_labels[mask] == 1).mean()) if mask.any() else None,
            }

        self.kitsune = kit
        self.lstm_ae = lstm
        self.metrics = {
            "kitsune": {"precision": kp, "recall": kr, "f1": kf, "auc": k_auc},
            "lstm_ae": {"precision": lp, "recall": lr_, "f1": lf, "auc": l_auc},
            "per_attack_recall": per_attack,
            "n_train_benign": int(benign_mask.sum()),
            "n_train_total": int(len(y)),
        }
        self.ready = True

        with open(CACHE_PATH, "wb") as f:
            pickle.dump({"kitsune": kit, "lstm_ae": lstm, "metrics": self.metrics}, f)

    def get_explainer(self, model_name: str) -> XNIDSExplainer:
        detector = self.kitsune if model_name == "kitsune" else self.lstm_ae
        return XNIDSExplainer(detector, history_len=5)

    def get_detector(self, model_name: str):
        return self.kitsune if model_name == "kitsune" else self.lstm_ae


store = ModelStore()
