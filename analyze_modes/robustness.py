"""Устойчивость ранжирования смен

Цель — показать, что итоговое ранжирование по интегральному баллу не
является артефактом (а) случайной выборки данных и (б) субъективного
выбора весов. Это закрывает два главных вопроса комиссии к балловой
модели качества.

Реализованы три проверки:

1. Устойчивость ранга при разных нормировках. Все смены делятся на
   две хронологические половины. На каждой половине отдельно
   подгоняются нормировки CV* и tons_p95, после чего ВСЕ смены
   перенормируются под обе системы. Spearman ρ между двумя
   ранжированиями — мера независимости порядка от выбора cohort'а
   нормировки.

2. Bootstrap-CI для Spearman ρ ранжирования при ресэмплировании
   смен — мера статистической устойчивости порядка.

3. Чувствительность top-K к возмущениям весов. На симплексе весов
   вокруг номинальной точки случайно сэмплируются N комбинаций;
   для каждой пересчитывается top-K. Доля смен, удержавшихся в
   top-K при всех возмущениях, и средний размер пересечения — мера
   независимости top-K от субъективного выбора весов.
"""

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

from .config import CONFIG
from .scoring import ScoreNorms, fit_norms, score_shifts


def _scores_with_norms(shifts: List[Dict], norms: ScoreNorms) -> np.ndarray:
    return np.array([s["score"] for s in score_shifts(shifts, norms)], dtype=float)


def ranking_stability_temporal(shifts: List[Dict]) -> Dict:
    """Устойчивость рангов при разных хронологических нормировочных cohort'ах"""
    if len(shifts) < 20:
        return {"error": "недостаточно смен"}

    sorted_shifts = sorted(shifts, key=lambda s: s["end_time"])
    half = len(sorted_shifts) // 2
    first, second = sorted_shifts[:half], sorted_shifts[half:]

    norms_first = fit_norms(first)
    norms_second = fit_norms(second)

    s_a = _scores_with_norms(sorted_shifts, norms_first)
    s_b = _scores_with_norms(sorted_shifts, norms_second)

    rho, p = stats.spearmanr(s_a, s_b)
    tau, p_tau = stats.kendalltau(s_a, s_b)

    return {
        "n_shifts": len(sorted_shifts),
        "spearman_rho": float(rho),
        "spearman_p": float(p),
        "kendall_tau": float(tau),
        "kendall_p": float(p_tau),
        "norms_first":  {"cv_star": norms_first.cv_star,  "tons_p95": norms_first.tons_p95},
        "norms_second": {"cv_star": norms_second.cv_star, "tons_p95": norms_second.tons_p95},
        "ranks_a": stats.rankdata(s_a).tolist(),
        "ranks_b": stats.rankdata(s_b).tolist(),
    }


def ranking_stability_bootstrap(shifts: List[Dict]) -> Dict:
    """Bootstrap-CI для Spearman ρ между двумя независимыми ресэмплами"""
    if len(shifts) < 20:
        return {"error": "недостаточно смен"}

    rng = np.random.default_rng(CONFIG["random_seed"])
    iters = CONFIG["ranking_stability_iters"]
    n = len(shifts)

    norms = fit_norms(shifts)
    base_scores = _scores_with_norms(shifts, norms)

    rhos = []
    for _ in range(iters):
        idx = rng.integers(0, n, n)
        sub = [shifts[i] for i in idx]
        sub_norms = fit_norms(sub)
        sub_scores = _scores_with_norms(shifts, sub_norms)
        rho, _ = stats.spearmanr(base_scores, sub_scores)
        if np.isfinite(rho):
            rhos.append(rho)

    if not rhos:
        return {"error": "не удалось посчитать"}
    rhos = np.array(rhos)
    return {
        "iters": int(len(rhos)),
        "mean_rho": float(rhos.mean()),
        "ci_low":  float(np.percentile(rhos, 2.5)),
        "ci_high": float(np.percentile(rhos, 97.5)),
        "rhos": rhos.tolist(),
    }


def _random_weights_on_simplex(rng: np.random.Generator,
                               base: np.ndarray,
                               radius: float) -> np.ndarray:
    """Возмущение весов в окрестности base (на симплексе)"""
    delta = rng.uniform(-radius, radius, size=base.shape)
    delta -= delta.mean()  # сохранить сумму
    w = base + delta
    w = np.clip(w, 1e-3, None)
    w /= w.sum()
    return w


def weight_sensitivity(shifts: List[Dict], top_n: int) -> Dict:
    """Доля смен, остающихся в top-N при возмущениях весов S"""
    if len(shifts) < top_n:
        return {"error": "недостаточно смен"}

    rng = np.random.default_rng(CONFIG["random_seed"])
    iters = CONFIG["weight_perturbation_iters"]
    radius = CONFIG["weight_perturbation_radius"]
    base = CONFIG["score_weights"]
    keys = ("V", "M", "P")
    base_arr = np.array([base[k] for k in keys], dtype=float)

    norms = fit_norms(shifts)
    base_scored = score_shifts(shifts, norms)
    base_order = np.argsort([-s["score"] for s in base_scored])
    base_top = set(int(i) for i in base_order[:top_n])

    overlaps = []
    rho_list = []
    membership_count = {i: 0 for i in base_top}
    base_scores = np.array([s["score"] for s in base_scored])

    for _ in range(iters):
        w = _random_weights_on_simplex(rng, base_arr, radius)
        weights = {k: float(w[i]) for i, k in enumerate(keys)}
        scored = score_shifts(shifts, norms, weights=weights)
        order = np.argsort([-s["score"] for s in scored])
        top_set = set(int(i) for i in order[:top_n])
        overlaps.append(len(base_top & top_set) / top_n)
        for i in base_top:
            if i in top_set:
                membership_count[i] += 1
        new_scores = np.array([s["score"] for s in scored])
        rho, _ = stats.spearmanr(base_scores, new_scores)
        if np.isfinite(rho):
            rho_list.append(rho)

    overlaps = np.array(overlaps)
    rho_arr = np.array(rho_list) if rho_list else np.array([np.nan])

    persistent = sum(1 for c in membership_count.values() if c == iters)

    return {
        "iters": iters,
        "radius": radius,
        "top_n": top_n,
        "mean_overlap": float(overlaps.mean()),
        "min_overlap": float(overlaps.min()),
        "overlap_ci": [float(np.percentile(overlaps, 2.5)),
                       float(np.percentile(overlaps, 97.5))],
        "mean_spearman_rho": float(np.nanmean(rho_arr)),
        "spearman_ci": [float(np.nanpercentile(rho_arr, 2.5)),
                        float(np.nanpercentile(rho_arr, 97.5))],
        "persistent_in_top_n": int(persistent),
        "spearman_rhos": rho_arr.tolist(),
        "overlaps": overlaps.tolist(),
    }

def assess_robustness(shifts: List[Dict], top_n: int) -> Dict:
    """Полный пакет проверок устойчивости ранжирования"""
    return {
        "temporal_stability": ranking_stability_temporal(shifts),
        "bootstrap_stability": ranking_stability_bootstrap(shifts),
        "weight_sensitivity": weight_sensitivity(shifts, top_n=top_n),
    }
