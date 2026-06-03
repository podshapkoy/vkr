"""Загрузка временных рядов и журнала остановок, вырезание окон вокруг аварий"""

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from .config import CONFIG
from .filtering import apply_adaptive_filter
from .models import SafetyThresholds


def setup_directories() -> None:
    Path(CONFIG["reports_dir"]).mkdir(parents=True, exist_ok=True)
    Path(CONFIG["plots_dir"]).mkdir(parents=True, exist_ok=True)


def _load_csv_dir(csv_dir: str) -> pd.DataFrame:
    files = sorted(Path(csv_dir).glob("1sect_*.csv"))
    if not files:
        raise FileNotFoundError(f"Нет файлов в {csv_dir}")
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f, parse_dates=["time"])
            df["source_file"] = f.name
            frames.append(df)
        except Exception as e:
            print(f"[WARN] Пропускаю {f.name}: {e}")
    if not frames:
        raise RuntimeError("Не удалось прочитать ни одного CSV")
    return pd.concat(frames, ignore_index=True)


def _load_stops(stops_file: str) -> pd.DataFrame:
    p = Path(stops_file)
    if not p.exists():
        print(f"[WARN] Файл остановок не найден: {stops_file}")
        return pd.DataFrame()
    stops = pd.read_csv(p, parse_dates=["drop_start", "stop_start"])
    stops["stop_duration_h"] = pd.to_numeric(stops["stop_duration"], errors="coerce")
    return stops


def _compute_safety_thresholds(df: pd.DataFrame) -> SafetyThresholds:
    cur = df["current"].describe(percentiles=[0.05, 0.95, 0.99])
    wgt = df["weight"].describe(percentiles=[0.05, 0.95, 0.99])
    spc = (df["current"] / df["weight"]).describe(percentiles=[0.05, 0.95, 0.99])
    return SafetyThresholds(
        current_max=cur["95%"] * 1.10,
        current_min=cur["5%"] * 0.90,
        weight_max=wgt["95%"] * 1.10,
        weight_min=wgt["5%"] * 0.90,
        specific_current_max=spc["95%"] * 1.15,
        specific_current_min=spc["5%"] * 0.85,
        current_emergency=cur["99%"] * 1.05,
        weight_emergency=wgt["99%"] * 1.05,
        specific_current_emergency=spc["99%"] * 1.10,
    )


def _cut_stop_windows(data: pd.DataFrame, stops: pd.DataFrame) -> pd.DataFrame:
    """Вырезаем окно [drop_start - δ, stop_start + ε] для каждой остановки"""
    if stops.empty:
        return data
    pre = pd.Timedelta(minutes=CONFIG["stop_pre_window_min"])
    post = pd.Timedelta(minutes=CONFIG["stop_post_window_min"])

    mask = pd.Series(True, index=data.index)
    for fname, grp in stops.groupby("filename"):
        file_mask = data["source_file"] == fname
        if not file_mask.any():
            continue
        sub_idx = data.index[file_mask]
        sub_time = data.loc[sub_idx, "time"]
        keep = pd.Series(True, index=sub_idx)
        for _, row in grp.iterrows():
            d_start = row["drop_start"]
            s_start = row["stop_start"]
            if pd.isna(d_start) or pd.isna(s_start):
                continue
            window_start = d_start - pre
            window_end = s_start + post
            in_window = (sub_time >= window_start) & (sub_time <= window_end)
            keep.loc[in_window[in_window].index] = False
        mask.loc[sub_idx] = keep
    cleaned = data.loc[mask].copy()
    removed = len(data) - len(cleaned)
    print(f"[INFO] Вырезано {removed} точек ({removed / max(1, len(data)) * 100:.2f}%) "
          f"в окнах вокруг {len(stops)} остановок")
    return cleaned


def load_all(csv_dir: Optional[str] = None,
             stops_file: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, SafetyThresholds]:
    csv_dir = csv_dir or CONFIG["csv_dir"]
    stops_file = stops_file or CONFIG["stops_file"]

    raw = _load_csv_dir(csv_dir)
    raw = raw[(raw["current"] > 0) & (raw["weight"] > 0)].copy()
    raw = raw.sort_values(["source_file", "time"]).reset_index(drop=True)

    stops = _load_stops(stops_file)
    cleaned = _cut_stop_windows(raw, stops)

    thresholds = _compute_safety_thresholds(cleaned)

    cleaned["current_filtered"] = (
        cleaned.groupby("source_file", group_keys=False)["current"]
        .apply(lambda s: apply_adaptive_filter(s, thresholds.current_max, thresholds.current_min))
    )
    cleaned["weight_filtered"] = (
        cleaned.groupby("source_file", group_keys=False)["weight"]
        .apply(lambda s: apply_adaptive_filter(s, thresholds.weight_max, thresholds.weight_min))
    )
    cleaned["specific_current"] = cleaned["current_filtered"] / cleaned["weight_filtered"]

    return cleaned, stops, thresholds
