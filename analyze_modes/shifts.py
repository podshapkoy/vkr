"""Сегментация на смены и расчёт метрик каждой смены (без формулы балла)

Кроме базовых статистик (среднее, CV, объём, safety_margin) считаем
признаки **конца смены** — нестабильность в последние N часов и дрейф тока
Они нужны для R-балла раннего предупреждения
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import CONFIG
from .models import SafetyThresholds


def _shift_label(hour: int) -> str:
    day_start = CONFIG["shift_day_start_hour"]
    night_start = CONFIG["shift_night_start_hour"]
    return "День" if day_start <= hour < night_start else "Ночь"


def _safety_margin(df: pd.DataFrame, th: SafetyThresholds) -> float:
    parts = []

    cur = df["current_filtered"]
    cur_top = (th.current_max - cur.max()) / th.current_max
    cur_bot = (cur.min() - th.current_min) / th.current_min
    parts.append(max(0.0, min(cur_top, cur_bot)))

    wgt = df["weight_filtered"]
    wgt_top = (th.weight_max - wgt.max()) / th.weight_max
    wgt_bot = (wgt.min() - th.weight_min) / th.weight_min
    parts.append(max(0.0, min(wgt_top, wgt_bot)))

    spc = df["specific_current"]
    spc_top = (th.specific_current_max - spc.max()) / th.specific_current_max
    spc_bot = (spc.min() - th.specific_current_min) / th.specific_current_min
    parts.append(max(0.0, min(spc_top, spc_bot)))

    return float(np.clip(np.mean(parts), 0.0, 1.0))


def _late_window(g: pd.DataFrame, hours: float) -> pd.DataFrame:
    end = g["time"].iloc[-1]
    return g[g["time"] >= end - pd.Timedelta(hours=hours)]


def _current_slope_per_hour(g: pd.DataFrame) -> float:
    """Линейный наклон тока (А / ч) — дрейф через всю смену"""
    if len(g) < 2:
        return 0.0
    t = (g["time"] - g["time"].iloc[0]).dt.total_seconds().to_numpy() / 3600.0
    y = g["current_filtered"].to_numpy()
    finite = np.isfinite(t) & np.isfinite(y)
    if finite.sum() < 2 or t[finite].std() == 0:
        return 0.0
    slope, _ = np.polyfit(t[finite], y[finite], 1)
    return float(slope) if np.isfinite(slope) else 0.0


def _late_cv(g_late: pd.DataFrame) -> Tuple[float, float]:
    if g_late.empty:
        return float("nan"), float("nan")
    cm = float(g_late["current_filtered"].mean())
    wm = float(g_late["weight_filtered"].mean())
    cs = float(g_late["current_filtered"].std())
    ws = float(g_late["weight_filtered"].std())
    cur_cv = cs / cm if cm and np.isfinite(cm) and np.isfinite(cs) else float("nan")
    wgt_cv = ws / wm if wm and np.isfinite(wm) and np.isfinite(ws) else float("nan")
    return cur_cv, wgt_cv


def _late_safety_margin(g: pd.DataFrame, th: SafetyThresholds) -> float:
    """Запас в последний 1 ч смены — насколько мельница близка к пределу перед концом"""
    g_last = _late_window(g, hours=1.0)
    if g_last.empty:
        return float("nan")
    return _safety_margin(g_last, th)


def split_into_shifts(data: pd.DataFrame, thresholds: SafetyThresholds) -> List[Dict]:
    """Возвращает список словарей с сырыми метриками смены — без балла"""
    if data.empty:
        return []

    df = data.sort_values(["source_file", "time"]).copy()
    df["shift_type"] = df["time"].dt.hour.map(_shift_label)
    df["_grp"] = (
        (df["source_file"] != df["source_file"].shift())
        | (df["shift_type"] != df["shift_type"].shift())
    ).cumsum()

    min_dur = CONFIG["min_shift_duration_h"]
    shifts = []
    for _, g in df.groupby("_grp"):
        start = g["time"].iloc[0]
        end = g["time"].iloc[-1]
        dt = g["time"].diff().dt.total_seconds()
        gap_threshold_s = 120.0
        duration = float(dt[(dt > 0) & (dt <= gap_threshold_s)].sum()) / 3600.0
        if duration < min_dur:
            continue

        cur = g["current_filtered"]
        wgt = g["weight_filtered"]
        spc = g["specific_current"]

        cur_mean = float(cur.mean())
        wgt_mean = float(wgt.mean())
        if (not np.isfinite(cur_mean) or not np.isfinite(wgt_mean)
                or cur_mean <= 0 or wgt_mean <= 0):
            continue

        cur_std = float(cur.std())
        wgt_std = float(wgt.std())
        if not np.isfinite(cur_std) or not np.isfinite(wgt_std):
            continue
        cur_cv = cur_std / cur_mean
        wgt_cv = wgt_std / wgt_mean

        tons = float(wgt_mean * duration)

        late_g = _late_window(g, CONFIG["late_window_hours"])
        late_cur_cv, late_wgt_cv = _late_cv(late_g)
        slope = _current_slope_per_hour(g)
        late_margin = _late_safety_margin(g, thresholds)

        shifts.append({
            "source_file": g["source_file"].iloc[0],
            "start_time": start,
            "end_time": end,
            "shift_type": g["shift_type"].iloc[0],
            "duration_h": duration,
            "current_mean": cur_mean,
            "current_cv": cur_cv,
            "weight_mean": wgt_mean,
            "weight_cv": wgt_cv,
            "specific_current_mean": float(spc.mean()),
            "tons": tons,
            "safety_margin": _safety_margin(g, thresholds),
            "late_current_cv": late_cur_cv,
            "late_weight_cv": late_wgt_cv,
            "current_slope_A_per_h": slope,
            "late_safety_margin": late_margin,
        })
    return shifts
