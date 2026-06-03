"""Интегральный балл S — мера качества смены для ранжирования

    S = w_V · V + w_M · M + w_P · P,   V, M, P ∈ [0, 1]

  V — стабильность по CV:
        V = 1 - min(1, (CV_current + CV_weight) / (2 · CV*))
  M — запас до технологических ограничений (safety_margin), [0, 1]
  P — объём переработки за смену, нормированный по p95 распределения смен

Назначение балла — ранжирования смен и выделение наблюдаемой устойчивой
рабочей области. Балл не является вероятностной оценкой риска
аварии — для этой задачи в модуле обучается отдельная мультивариантная
модель (`predictive.py`)

Нормировки CV* и tons_p95 фиксируются на тренировочной (хронологически
ранней) выборке и применяются ко всем сменам — это исключает data leakage
при последующей валидации
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .config import CONFIG


@dataclass
class ScoreNorms:
    cv_star: float
    tons_p95: float


def _safe_p(arr: np.ndarray, q: float, fallback: float) -> float:
    arr = arr[np.isfinite(arr)]
    if not len(arr):
        return fallback
    return float(np.percentile(arr, q))


def fit_norms(shifts: List[Dict]) -> ScoreNorms:
    """Подгонка нормировок на (тренировочной) выборке смен"""
    if not shifts:
        return ScoreNorms(cv_star=0.05, tons_p95=1.0)
    cvs = np.array([s["current_cv"] + s["weight_cv"] for s in shifts], dtype=float)
    tons = np.array([s["tons"] for s in shifts], dtype=float)

    cv_star = max(_safe_p(cvs, 90, 0.1) / 2.0, 1e-6)
    tons_p95 = max(_safe_p(tons, 95, 1.0), 1e-6)
    return ScoreNorms(cv_star=cv_star, tons_p95=tons_p95)


def compute_score(shift: Dict, norms: ScoreNorms,
                  weights: Optional[Dict[str, float]] = None) -> Dict:
    cv_sum = shift["current_cv"] + shift["weight_cv"]
    if not np.isfinite(cv_sum):
        cv_sum = 2.0 * norms.cv_star  # худший случай => V = 0
    V = 1.0 - min(1.0, cv_sum / (2.0 * norms.cv_star))
    V = float(np.clip(V, 0.0, 1.0))

    M = float(np.clip(shift["safety_margin"], 0.0, 1.0))

    tons = shift["tons"]
    P = 0.0 if not np.isfinite(tons) else min(1.0, tons / norms.tons_p95)
    P = float(np.clip(P, 0.0, 1.0))

    w = weights or CONFIG["score_weights"]
    S = w["V"] * V + w["M"] * M + w["P"] * P
    return {"V": V, "M": M, "P": P, "score": float(S)}


def score_shifts(shifts: List[Dict], norms: ScoreNorms,
                 weights: Optional[Dict[str, float]] = None) -> List[Dict]:
    out = []
    for s in shifts:
        merged = dict(s)
        merged.update(compute_score(s, norms, weights=weights))
        out.append(merged)
    return out
