from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from mne.filter import filter_data, notch_filter
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import EEGConfig, MentalCommandModelConfig


def filter_window(X: np.ndarray, eeg_cfg: EEGConfig, sfreq: float) -> np.ndarray:
    """Bandpass/notch filter for one or many windows.

    X shape: (n_ch, n_samp) or (n_win, n_ch, n_samp).
    """
    Xf = np.asarray(X, dtype=np.float64, order="C")
    if eeg_cfg.notch is not None:
        Xf = notch_filter(Xf, Fs=sfreq, freqs=[float(eeg_cfg.notch)], verbose="ERROR")
    Xf = filter_data(
        Xf,
        sfreq=float(sfreq),
        l_freq=float(eeg_cfg.l_freq),
        h_freq=float(eeg_cfg.h_freq),
        verbose="ERROR",
    )
    return Xf.astype(np.float32, copy=False)


def split_windows(
    block: np.ndarray,
    sfreq: float,
    window_s: float,
    step_s: float,
) -> np.ndarray:
    """Split one continuous block into fixed windows.

    block shape: (n_ch, n_samples).
    returns shape: (n_windows, n_ch, n_window_samples).
    """
    n_ch, n_samples = block.shape
    w = int(round(float(window_s) * float(sfreq)))
    h = int(round(float(step_s) * float(sfreq)))
    if w <= 0 or h <= 0:
        raise ValueError("window_s and step_s must produce at least one sample")
    if n_samples < w:
        return np.empty((0, n_ch, w), dtype=np.float32)
    starts = np.arange(0, n_samples - w + 1, h, dtype=int)
    out = np.empty((len(starts), n_ch, w), dtype=np.float32)
    for i, s in enumerate(starts):
        out[i] = block[:, s : s + w]
    return out


def make_mental_command_classifier(model_cfg: MentalCommandModelConfig) -> Pipeline:
    return Pipeline(
        [
            ("cov", Covariances(estimator="oas")),
            ("ts", TangentSpace(metric="riemann")),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "clf",
                LogisticRegression(
                    C=float(model_cfg.C),
                    solver="lbfgs",
                    multi_class="multinomial",
                    max_iter=int(model_cfg.max_iter),
                    class_weight=model_cfg.class_weight,
                    random_state=42,
                ),
            ),
        ]
    )


@dataclass
class MCQuality:
    balanced_accuracy: float
    macro_f1: float
    n_samples: int
    n_per_class: dict[str, int]


def evaluate_cv_quality(
    X: np.ndarray,
    y: np.ndarray,
    model_cfg: MentalCommandModelConfig,
    cv_splits_max: int,
) -> MCQuality:
    classes, counts = np.unique(y, return_counts=True)
    min_count = int(np.min(counts))
    n_splits = min(int(cv_splits_max), min_count)
    if n_splits < 2:
        raise ValueError(f"Not enough class samples for CV: counts={dict(zip(classes, counts))}")

    clf = make_mental_command_classifier(model_cfg)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_pred = cross_val_predict(clf, X, y, cv=cv, method="predict")

    return MCQuality(
        balanced_accuracy=float(balanced_accuracy_score(y, y_pred)),
        macro_f1=float(f1_score(y, y_pred, average="macro")),
        n_samples=int(len(y)),
        n_per_class={str(int(c)): int(n) for c, n in zip(classes, counts)},
    )


class EMAProbSmoother:
    def __init__(self, alpha: float, n_classes: int):
        self.alpha = float(alpha)
        self.n_classes = int(n_classes)
        self._state: np.ndarray | None = None

    def update(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=np.float64)
        if p.shape != (self.n_classes,):
            raise ValueError(f"Expected probs shape ({self.n_classes},), got {p.shape}")
        if self._state is None:
            self._state = p.copy()
        else:
            self._state = (1.0 - self.alpha) * self._state + self.alpha * p
        s = float(np.sum(self._state))
        if s > 0:
            self._state = self._state / s
        return self._state.copy()
