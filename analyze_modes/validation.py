"""Статистическая валидация интегрального балла и наблюдаемой рабочей области

Дизайн валидации (исключение data leakage):

1. Все смены сортируются по времени окончания и делятся в отношении
   train_split_frac : (1 − train_split_frac)
2. На train-выборке подгоняются нормировки балла (CV*, tons_p95) и
   извлекается наблюдаемая рабочая область (top-K% по баллу)
3. На test-выборке проверяется:
   - однофакторная разделимость классов «авария в горизонте H часов»
     по компонентам балла и исходным признакам (ROC-AUC + bootstrap-CI
     + permutation p-value);
   - relative risk аварии для смен, средние параметры которых попадают
     внутрь рабочей области, против попавших вне (с bootstrap-CI)

Полная мультивариантная модель риска обучается отдельно (`predictive.py`)
Цель этого модуля — обеспечить статистические гарантии качества
ранжирования и рабочей области
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from .config import CONFIG


def chronological_split(shifts: List[Dict],
                        frac: Optional[float] = None) -> Tuple[List[Dict], List[Dict]]:
    """хронологический split"""
    if frac is None:
        frac = CONFIG["train_split_frac"]
    ordered = sorted(shifts, key=lambda s: s["end_time"])
    n = int(len(ordered) * frac)
    return ordered[:n], ordered[n:]




def label_shifts(shifts: List[Dict], stops: pd.DataFrame,
                 horizon_h: Optional[float] = None) -> List[Dict]:
    """разметка y по аварийным остановкам"""
    """y_h<H> = 1, если в течение H часов после конца смены произошла аварийная остановка"""
    horizons = list(CONFIG["validation_horizons_h"])
    if horizon_h is not None and horizon_h not in horizons:
        horizons = sorted(set(horizons + [horizon_h]))

    primary = CONFIG["primary_horizon_h"]
    out = []
    if stops is None or stops.empty:
        for s in shifts:
            s2 = dict(s)
            for h in horizons:
                s2[f"y_h{h}"] = 0
                s2[f"n_emerg_h{h}"] = 0
            s2["y"] = 0
            out.append(s2)
        return out

    emerg = stops[stops["stop_type"] == "АВАРИЙНАЯ"].copy()
    for s in shifts:
        end = s["end_time"]
        same_file = emerg[emerg["filename"] == s["source_file"]]
        s2 = dict(s)
        for h in horizons:
            in_h = same_file[(same_file["stop_start"] > end)
                             & (same_file["stop_start"] <= end + pd.Timedelta(hours=h))]
            s2[f"y_h{h}"] = int(len(in_h) > 0)
            s2[f"n_emerg_h{h}"] = int(len(in_h))
        s2["y"] = s2[f"y_h{primary}"]
        s2["n_emerg_after"] = s2[f"n_emerg_h{primary}"]
        out.append(s2)
    return out


def _auc_with_direction(x: np.ndarray, y: np.ndarray) -> Tuple[float, str]:
    """однофакторная валидация признаков"""
    """возвращает max(AUC(x→y), AUC(−x→y)) и направление 'high=risk' / 'low=risk'"""
    a_pos = roc_auc_score(y, x)
    a_neg = roc_auc_score(y, -x)
    if a_pos >= a_neg:
        return float(a_pos), "high=risk"
    return float(a_neg), "low=risk"


def _bootstrap_auc_ci(x: np.ndarray, y: np.ndarray, sign: float,
                      iters: int, seed: int) -> Optional[Dict[str, float]]:
    rng = np.random.default_rng(seed)
    n = len(x)
    aucs = []
    for _ in range(iters):
        idx = rng.integers(0, n, n)
        ys = y[idx]
        if ys.min() == ys.max():
            continue
        aucs.append(roc_auc_score(ys, sign * x[idx]))
    if not aucs:
        return None
    aucs = np.array(aucs)
    return {
        "mean": float(aucs.mean()),
        "ci_low":  float(np.percentile(aucs, 2.5)),
        "ci_high": float(np.percentile(aucs, 97.5)),
    }


def _permutation_p_auc(x: np.ndarray, y: np.ndarray, sign: float,
                       observed_auc: float, iters: int, seed: int) -> float:
    """двусторонний permutation test для отклонения AUC от 0.5"""
    rng = np.random.default_rng(seed + 1)
    n = len(x)
    if iters <= 0:
        return float("nan")
    extreme = 0
    obs = abs(observed_auc - 0.5)
    for _ in range(iters):
        ys = y[rng.permutation(n)]
        if ys.min() == ys.max():
            continue
        a = roc_auc_score(ys, sign * x)
        if abs(a - 0.5) >= obs:
            extreme += 1
    return float((extreme + 1) / (iters + 1))


def _cliffs_delta(x_pos: np.ndarray, x_neg: np.ndarray) -> float:
    """Effect size для непараметрического сравнения (−1..+1)"""
    if not len(x_pos) or not len(x_neg):
        return float("nan")
    diffs = x_pos[:, None] - x_neg[None, :]
    return float((np.sign(diffs).sum()) / diffs.size)


def univariate_validation(scored_shifts: List[Dict],
                          horizon_h: int,
                          features: Optional[List[str]] = None) -> List[Dict]:
    """однофакторный AUC + bootstrap-CI + permutation p + Cliff's delta для каждого признака"""
    if not scored_shifts:
        return []
    df = pd.DataFrame(scored_shifts)
    label_col = f"y_h{horizon_h}"
    if label_col not in df.columns:
        return []
    y = df[label_col].to_numpy().astype(int)
    if y.min() == y.max():
        return []

    if features is None:
        features = [
            "current_mean", "current_cv", "weight_mean", "weight_cv",
            "specific_current_mean", "safety_margin",
            "late_current_cv", "late_weight_cv",
            "late_safety_margin", "current_slope_A_per_h",
            "tons", "V", "M", "P", "score",
        ]

    iters_b = CONFIG["bootstrap_iters"]
    iters_p = CONFIG["permutation_iters"]
    seed = CONFIG["random_seed"]

    rows = []
    for col in features:
        if col not in df.columns:
            continue
        x = df[col].to_numpy().astype(float)
        m = np.isfinite(x)
        if m.sum() < len(x) * 0.9:
            continue
        x_m = x[m]; y_m = y[m]
        if y_m.min() == y_m.max():
            continue
        auc, direction = _auc_with_direction(x_m, y_m)
        sign = 1.0 if direction == "high=risk" else -1.0
        ci = _bootstrap_auc_ci(x_m, y_m, sign, iters_b, seed)
        p_perm = _permutation_p_auc(x_m, y_m, sign, auc, iters_p, seed)
        delta = _cliffs_delta(x_m[y_m == 1], x_m[y_m == 0])
        rows.append({
            "feature": col,
            "direction": direction,
            "auc": auc,
            "auc_ci_low":  ci["ci_low"]  if ci else None,
            "auc_ci_high": ci["ci_high"] if ci else None,
            "permutation_p": p_perm,
            "cliffs_delta": delta,
            "n": int(m.sum()),
            "n_pos": int(y_m.sum()),
        })
    rows.sort(key=lambda r: r["auc"], reverse=True)
    return rows



def empirical_working_area(scored_shifts: List[Dict],
                           top_frac: Optional[float] = None) -> Dict:
    """эмпирическая рабочая область + её риск-валидация"""
    """наблюдаемая устойчивая рабочая область — диапазоны параметров top-K смен"""
    if not scored_shifts:
        return {}
    if top_frac is None:
        top_frac = CONFIG["working_area_top_frac"]
    df = pd.DataFrame(scored_shifts).sort_values("score", ascending=False)
    k = max(3, int(len(df) * top_frac))
    top = df.head(k)

    rng = np.random.default_rng(CONFIG["random_seed"])
    iters = CONFIG["bootstrap_iters"]

    def _bands(values: np.ndarray) -> Dict:
        p25 = float(np.percentile(values, 25))
        p75 = float(np.percentile(values, 75))
        boots_low, boots_high = [], []
        n = len(values)
        for _ in range(iters):
            sample = values[rng.integers(0, n, n)]
            boots_low.append(np.percentile(sample, 25))
            boots_high.append(np.percentile(sample, 75))
        return {
            "p25": p25,
            "p75": p75,
            "p25_ci": [float(np.percentile(boots_low, 2.5)),
                       float(np.percentile(boots_low, 97.5))],
            "p75_ci": [float(np.percentile(boots_high, 2.5)),
                       float(np.percentile(boots_high, 97.5))],
        }

    return {
        "n_top_shifts": int(k),
        "top_frac": top_frac,
        "current": _bands(top["current_mean"].to_numpy()),
        "weight":  _bands(top["weight_mean"].to_numpy()),
        "specific_current": _bands(top["specific_current_mean"].to_numpy()),
    }


def _params_in_area(s: Dict, area: Dict) -> int:
    """количество параметров (из 3), попадающих в IQR рабочей области"""
    n = 0
    if area["current"]["p25"]          <= s["current_mean"]          <= area["current"]["p75"]:          n += 1
    if area["weight"]["p25"]           <= s["weight_mean"]           <= area["weight"]["p75"]:           n += 1
    if area["specific_current"]["p25"] <= s["specific_current_mean"] <= area["specific_current"]["p75"]: n += 1
    return n


def _shift_in_area(s: Dict, area: Dict, min_params: int = 3) -> bool:
    return _params_in_area(s, area) >= min_params


def working_area_risk_test(test_shifts: List[Dict],
                           area: Dict,
                           horizon_h: int,
                           min_params: int = 2) -> Dict:
    """доля аварий в горизонте H для смен внутри/снаружи рабочей области

    рабочая область строится на train-выборке; этот тест применяется к test-выборке.
    `min_params` — сколько из 3 параметров должно одновременно находиться в IQR
    рабочей области, чтобы смена считалась попавшей внутрь. По умолчанию 2 из 3
    (relaxed): жёсткий критерий «3 из 3» при IQR-полосе 50% и 3-кратном AND
    отсекает 87.5% выборки и резко теряет статистическую мощность.
    возвращает relative risk с bootstrap-CI и точный тест Фишера
    """
    if not test_shifts or not area:
        return {"error": "нет данных"}
    label_col = f"y_h{horizon_h}"
    inside_y, outside_y = [], []
    for s in test_shifts:
        y = int(s.get(label_col, 0))
        if _shift_in_area(s, area, min_params=min_params):
            inside_y.append(y)
        else:
            outside_y.append(y)

    n_in, n_out = len(inside_y), len(outside_y)
    if n_in == 0 or n_out == 0:
        return {"error": "одна из групп пуста",
                "n_inside": n_in, "n_outside": n_out}
    p_in  = float(np.mean(inside_y))
    p_out = float(np.mean(outside_y))
    rr_point = (p_in / p_out) if p_out > 0 else float("inf")

    # katz log-CI для относительного риска — устойчив при малых частотах положительных меток
    a = int(sum(inside_y));  b = n_in - a
    c = int(sum(outside_y)); d = n_out - c
    rr_ci_katz = None
    if a > 0 and c > 0 and rr_point > 0 and np.isfinite(rr_point):
        se_log = float(np.sqrt(1/a - 1/(a + b) + 1/c - 1/(c + d)))
        log_rr = float(np.log(rr_point))
        rr_ci_katz = [float(np.exp(log_rr - 1.96 * se_log)),
                      float(np.exp(log_rr + 1.96 * se_log))]

    # bootstrap CI остаётся как дополнительный контроль
    rng = np.random.default_rng(CONFIG["random_seed"])
    iters = CONFIG["bootstrap_iters"]
    inside_arr  = np.array(inside_y,  dtype=int)
    outside_arr = np.array(outside_y, dtype=int)
    rrs = []
    for _ in range(iters):
        aa = inside_arr[rng.integers(0, n_in, n_in)]
        bb = outside_arr[rng.integers(0, n_out, n_out)]
        if bb.mean() <= 0:
            continue
        rrs.append(aa.mean() / bb.mean())
    rr_ci_boot = None
    if rrs:
        rrs = np.array(rrs)
        rr_ci_boot = [float(np.percentile(rrs, 2.5)),
                      float(np.percentile(rrs, 97.5))]

    # точный тест Фишера
    table = np.array([
        [a, b],
        [c, d],
    ])
    odds_ratio, p_fisher = stats.fisher_exact(table, alternative="less")

    return {
        "horizon_h": horizon_h,
        "min_params_inside": min_params,
        "n_inside":  n_in,
        "n_outside": n_out,
        "n_emerg_inside":  a,
        "n_emerg_outside": c,
        "rate_inside":  p_in,
        "rate_outside": p_out,
        "relative_risk":            rr_point,
        "relative_risk_ci_katz":      rr_ci_katz,
        "relative_risk_ci_bootstrap": rr_ci_boot,
        "relative_risk_ci":           rr_ci_katz or rr_ci_boot,
        "risk_reduction":     (p_out - p_in) / p_out if p_out > 0 else float("nan"),
        "fisher_odds_ratio":  float(odds_ratio),
        "fisher_p_one_sided": float(p_fisher),
    }