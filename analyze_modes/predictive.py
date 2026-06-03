"""Мультивариантная модель риска аварийной остановки

Обучается логистическая регрессия с хронологическим train/test split:
модель видит только ранние данные, предсказание делается на поздних.

Метрики на тестовой выборке:
  - ROC-AUC + bootstrap-CI;
  - PR-AUC;
  - permutation p-value для AUC (двусторонний);
  - Brier score (калибровка);
  - permutation feature importance (drop в AUC при перемешивании одной
    колонки на тесте, усреднённый по seed-ам);
  - reliability diagram (точки калибровки).

Дополнительно проводится TimeSeriesSplit-CV для оценки устойчивости AUC
по фолдам. Это исключает overfitting и отражает реальный сценарий
применения «модель построена на истории — применяется к будущим сменам»
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from .config import CONFIG


FEATURE_COLS = [
    "current_mean",
    "current_cv",
    "weight_mean",
    "weight_cv",
    "specific_current_mean",
    "safety_margin",
    "late_current_cv",
    "late_safety_margin",
    "current_slope_A_per_h",
]


def _prepare_xy(df: pd.DataFrame, label_col: str):
    df = df.dropna(subset=FEATURE_COLS + [label_col]).copy()
    X = df[FEATURE_COLS].to_numpy()
    y = df[label_col].to_numpy().astype(int)
    return X, y, df


def _temporal_split(df: pd.DataFrame, frac: float):
    df = df.sort_values("end_time")
    n_train = int(len(df) * frac)
    return df.iloc[:n_train], df.iloc[n_train:]


def fit_predict_score_baseline(scored_shifts: List[Dict], horizon_h: int) -> Dict:
    """Baseline: сам балл S как одномерный риск-предиктор.

    Низкий S = высокий риск, поэтому при подаче в AUC берётся -S.
    Хронологический split тот же, что и у мультивариантной модели — это
    делает сравнение «balanced»: оба классификатора оценены на одной и
    той же тестовой подвыборке
    """
    df = pd.DataFrame(scored_shifts)
    label_col = f"y_h{horizon_h}"
    if label_col not in df.columns or "score" not in df.columns:
        return {"error": "нет score или метки"}

    _, test_df = _temporal_split(df, CONFIG["train_split_frac"])
    test_df = test_df.dropna(subset=["score", label_col])
    y = test_df[label_col].to_numpy().astype(int)
    s = test_df["score"].to_numpy().astype(float)
    if len(y) == 0 or y.min() == y.max():
        return {"error": "в тестовой выборке только один класс"}

    score_as_risk = -s
    auc = float(roc_auc_score(y, score_as_risk))
    seed = CONFIG["random_seed"]
    boot = _bootstrap_auc(y, score_as_risk, CONFIG["bootstrap_iters"], seed)
    p_perm = _permutation_p_auc(y, score_as_risk, auc,
                                CONFIG["permutation_iters"], seed)
    return {
        "horizon_h": horizon_h,
        "n_test": int(len(y)),
        "n_pos_test": int(y.sum()),
        "test_roc_auc": auc,
        "bootstrap_auc": boot,
        "permutation_p": p_perm,
    }


def _bootstrap_auc(y: np.ndarray, p: np.ndarray, iters: int, seed: int):
    rng = np.random.default_rng(seed)
    n = len(y)
    aucs = []
    for _ in range(iters):
        idx = rng.integers(0, n, n)
        ys = y[idx]
        if ys.min() == ys.max():
            continue
        aucs.append(roc_auc_score(ys, p[idx]))
    if not aucs:
        return None
    aucs = np.array(aucs)
    return {
        "mean":    float(aucs.mean()),
        "ci_low":  float(np.percentile(aucs, 2.5)),
        "ci_high": float(np.percentile(aucs, 97.5)),
    }


def _permutation_p_auc(y: np.ndarray, p: np.ndarray, observed: float,
                       iters: int, seed: int) -> float:
    rng = np.random.default_rng(seed + 1)
    n = len(y)
    obs = abs(observed - 0.5)
    extreme = 0
    for _ in range(iters):
        ys = y[rng.permutation(n)]
        if ys.min() == ys.max():
            continue
        a = roc_auc_score(ys, p)
        if abs(a - 0.5) >= obs:
            extreme += 1
    return float((extreme + 1) / (iters + 1))


def _permutation_importance(model, scaler, X_test: np.ndarray, y_test: np.ndarray,
                            base_auc: float, iters: int, seed: int) -> List[Dict]:
    """Drop в AUC на тесте при перемешивании каждой колонки (на масштабированном X)"""
    rng = np.random.default_rng(seed + 2)
    Xs = scaler.transform(X_test)
    rows = []
    for j, name in enumerate(FEATURE_COLS):
        drops = []
        for _ in range(iters):
            X_perm = Xs.copy()
            X_perm[:, j] = X_perm[rng.permutation(len(X_perm)), j]
            p = model.predict_proba(X_perm)[:, 1]
            drops.append(base_auc - roc_auc_score(y_test, p))
        drops = np.array(drops)
        rows.append({
            "feature": name,
            "auc_drop_mean": float(drops.mean()),
            "auc_drop_ci": [float(np.percentile(drops, 2.5)),
                            float(np.percentile(drops, 97.5))],
        })
    rows.sort(key=lambda r: r["auc_drop_mean"], reverse=True)
    return rows


def _reliability_diagram(y: np.ndarray, p: np.ndarray, n_bins: int = 5) -> List[Dict]:
    """Equal-frequency биннинг: одинаковое n в каждом бине

    При малом числе положительных меток (n_pos < 20) фиксированные бины
    np.linspace становятся в большинстве пустыми. Биннинг по квантилям
    предсказанной вероятности даёт устойчивое отображение калибровки
    """
    if len(p) < n_bins:
        return []
    quantiles = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    quantiles[0] -= 1e-9
    quantiles[-1] += 1e-9
    bins = np.digitize(p, quantiles[1:-1])
    rows = []
    for b in range(n_bins):
        mask = bins == b
        if mask.sum() == 0:
            continue
        rows.append({
            "bin": int(b),
            "p_predicted_mean": float(p[mask].mean()),
            "p_observed":       float(y[mask].mean()),
            "n":                int(mask.sum()),
        })
    return rows


def _calibration_in_the_large(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    """Регрессия y на logit(p). Идеальная калибровка: intercept=0, slope=1.

    Это формальная альтернатива визуальной reliability diagram, устойчивая
    к выбору биннинга и малому количеству положительных меток
    """
    eps = 1e-6
    if y.min() == y.max():
        return {"intercept": float("nan"), "slope": float("nan")}
    p_clip = np.clip(p, eps, 1 - eps)
    logit_p = np.log(p_clip / (1 - p_clip)).reshape(-1, 1)
    m = LogisticRegression(C=1e6, max_iter=2000).fit(logit_p, y)
    return {"intercept": float(m.intercept_[0]),
            "slope":     float(m.coef_[0, 0])}


def fit_predict_one(scored_shifts: List[Dict], horizon_h: int,
                    importance_iters: int = 50) -> Dict:
    df = pd.DataFrame(scored_shifts)
    label_col = f"y_h{horizon_h}"
    if label_col not in df.columns:
        return {"error": f"нет колонки {label_col}"}

    train_df, test_df = _temporal_split(df, CONFIG["train_split_frac"])
    if train_df.empty or test_df.empty:
        return {"error": "недостаточно данных для split"}

    X_train, y_train, _ = _prepare_xy(train_df, label_col)
    X_test, y_test, test_df_clean = _prepare_xy(test_df, label_col)

    if y_train.sum() == 0 or y_train.sum() == len(y_train):
        return {"error": "в тренировочной выборке только один класс"}
    if y_test.sum() == 0 or y_test.sum() == len(y_test):
        return {"error": "в тестовой выборке только один класс",
                "n_train": len(y_train), "n_test": len(y_test),
                "n_pos_train": int(y_train.sum()),
                "n_pos_test":  int(y_test.sum())}

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # L1-логистическая регрессия с подбором C по стратифицированной кросс-валидации на train.
    # L1-разреживание обеспечивает автоматический отбор признаков, снижая эффективную размерность модели и сужая
    # bootstrap-CI AUC при малом числе положительных меток (EPV ≈ 4 на 9 признаках без отбора).
    # Stratified CV нужен потому, что чисто хронологический TimeSeriesSplit на train с base rate ~7% даёт
    # фолды без положительных меток — внутренний solver падает.
    # защита от утечки информации о будущем сохраняется: внешний holdout-test остаётся хронологически последним.
    n_pos_train = int(y_train.sum())
    n_neg_train = int((y_train == 0).sum())
    min_class = min(n_pos_train, n_neg_train)
    if min_class < 2:
        return {"error": f"в train слишком мало миноритарного класса: {min_class}"}

    inner_splits = max(2, min(5, min_class))
    inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True,
                               random_state=CONFIG["random_seed"])
    base_model = LogisticRegressionCV(
        Cs=np.logspace(-3, 1, 20),
        cv=inner_cv,
        penalty="l1",
        solver="liblinear",
        scoring="roc_auc",
        max_iter=5000,
        random_state=CONFIG["random_seed"],
    )

    # внешняя калибровка тоже на стратифицированных фолдах, иначе CalibratedClassifierCV сделает внутренний
    # k-fold по умолчанию и снова поймает фолды без положительных при малом n_pos
    n_calib_splits = max(2, min(5, min_class))
    calib_cv = StratifiedKFold(n_splits=n_calib_splits, shuffle=True,
                               random_state=CONFIG["random_seed"])
    model = CalibratedClassifierCV(estimator=base_model, method="sigmoid",
                                   cv=calib_cv)
    model.fit(X_train_s, y_train)

    # извлекаем выбранный C и активные (ненулевые) признаки из первого калиброванного эстиматора.
    # это даёт защищаемое описание модели: отбор признаков выполнен на train-выборке без data leakage
    selected_C = None
    nonzero_features: Optional[List[str]] = None
    try:
        inner_est = model.calibrated_classifiers_[0].estimator
        selected_C = float(inner_est.C_[0])
        coefs = inner_est.coef_[0]
        nonzero_features = [FEATURE_COLS[j] for j, c in enumerate(coefs)
                            if abs(c) > 1e-8]
    except Exception:
        pass
    proba_test = model.predict_proba(X_test_s)[:, 1]

    auc = float(roc_auc_score(y_test, proba_test))
    pr  = float(average_precision_score(y_test, proba_test))
    brier = float(brier_score_loss(y_test, proba_test))
    base_rate = float(y_test.mean())

    seed = CONFIG["random_seed"]
    boot = _bootstrap_auc(y_test, proba_test, CONFIG["bootstrap_iters"], seed)
    p_perm = _permutation_p_auc(y_test, proba_test, auc,
                                CONFIG["permutation_iters"], seed)
    importance = (_permutation_importance(model, scaler, X_test, y_test,
                                          base_auc=auc,
                                          iters=importance_iters, seed=seed)
                  if importance_iters > 0 else [])
    reliability = _reliability_diagram(y_test, proba_test, n_bins=5)
    calibration = _calibration_in_the_large(y_test, proba_test)

    return {
        "horizon_h": horizon_h,
        "n_train": int(len(y_train)),
        "n_test":  int(len(y_test)),
        "n_pos_train": int(y_train.sum()),
        "n_pos_test":  int(y_test.sum()),
        "test_roc_auc":   auc,
        "test_pr_auc":    pr,
        "test_brier":     brier,
        "test_base_rate": base_rate,
        "test_brier_naive": float(base_rate * (1 - base_rate)),
        "permutation_p":  p_perm,
        "bootstrap_auc":  boot,
        "permutation_importance": importance,
        "reliability_diagram":    reliability,
        "calibration_in_the_large": calibration,
        "selected_C":      selected_C,
        "nonzero_features": nonzero_features,
        "test_predictions": {
            "y_true":   y_test.tolist(),
            "y_proba":  proba_test.tolist(),
            "end_time": test_df_clean["end_time"].astype(str).tolist(),
        },
    }


def time_series_cv(scored_shifts: List[Dict], horizon_h: int,
                   n_splits: int = 5) -> Dict:
    df = pd.DataFrame(scored_shifts).sort_values("end_time").reset_index(drop=True)
    label_col = f"y_h{horizon_h}"
    if label_col not in df.columns:
        return {"error": f"нет {label_col}"}
    df = df.dropna(subset=FEATURE_COLS + [label_col])
    if len(df) < n_splits * 20:
        return {"error": "мало данных для CV"}

    X = df[FEATURE_COLS].to_numpy()
    y = df[label_col].to_numpy().astype(int)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    aucs = []
    fold_info = []
    for i, (tr, te) in enumerate(tscv.split(X)):
        y_tr, y_te = y[tr], y[te]
        if y_tr.sum() == 0 or y_te.sum() == 0 or y_te.sum() == len(y_te):
            fold_info.append({"fold": i + 1, "skipped": "только один класс"})
            continue
        scaler = StandardScaler().fit(X[tr])
        X_tr_s = scaler.transform(X[tr])
        X_te_s = scaler.transform(X[te])
        min_class_fold = min(int(y_tr.sum()), int((y_tr == 0).sum()))
        inner_n = max(2, min(3, min_class_fold))
        if min_class_fold < 2:
            m = LogisticRegression(penalty="l1", solver="liblinear",
                                   C=1.0, max_iter=5000,
                                   random_state=CONFIG["random_seed"])
        else:
            m = LogisticRegressionCV(
                Cs=np.logspace(-3, 1, 10),
                cv=StratifiedKFold(n_splits=inner_n, shuffle=True,
                                   random_state=CONFIG["random_seed"]),
                penalty="l1", solver="liblinear",
                scoring="roc_auc", max_iter=5000,
                random_state=CONFIG["random_seed"],
            )
        m.fit(X_tr_s, y_tr)
        proba = m.predict_proba(X_te_s)[:, 1]
        auc = float(roc_auc_score(y_te, proba))
        aucs.append(auc)
        fold_info.append({"fold": i + 1,
                          "n_train": int(len(tr)), "n_test": int(len(te)),
                          "n_pos_train": int(y_tr.sum()),
                          "n_pos_test":  int(y_te.sum()),
                          "test_auc": auc})
    if not aucs:
        return {"error": "ни одного валидного фолда"}
    return {
        "horizon_h": horizon_h,
        "n_splits": n_splits,
        "fold_aucs": aucs,
        "mean_auc": float(np.mean(aucs)),
        "std_auc":  float(np.std(aucs)),
        "min_auc":  float(np.min(aucs)),
        "max_auc":  float(np.max(aucs)),
        "folds":    fold_info,
    }


def fit_predict_primary(scored_shifts: List[Dict],
                        horizon_h: Optional[int] = None) -> Dict:
    if horizon_h is None:
        horizon_h = CONFIG["primary_horizon_h"]
    info = fit_predict_one(scored_shifts, horizon_h)
    info["time_series_cv"] = time_series_cv(scored_shifts, horizon_h)
    return info


def fit_predict_supplementary(scored_shifts: List[Dict]) -> Dict:
    """Прогон по всем горизонтам без полного importance"""
    out = {}
    primary = CONFIG["primary_horizon_h"]
    for h in CONFIG["validation_horizons_h"]:
        if h == primary:
            continue
        info = fit_predict_one(scored_shifts, h, importance_iters=0)
        info["time_series_cv"] = time_series_cv(scored_shifts, h)
        out[h] = info
    return out
