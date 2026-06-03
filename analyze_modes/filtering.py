from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from statsmodels.tsa.arima.model import ARIMA

from .models import KalmanFilter


def apply_kalman_filter(data: pd.Series, initial_state: Optional[float] = None,
                        process_variance: float = 1e-5, measurement_variance: float = 0.1 ** 2) -> pd.Series:
    """Оптимизированная версия фильтра Калмана для одномерных данных"""
    if initial_state is None:
        initial_state = data.iloc[0]

    kf = KalmanFilter(initial_state, process_variance, measurement_variance)
    estimates = np.zeros(len(data))
    estimates[0] = initial_state

    for i in range(1, len(data)):
        estimates[i] = kf.update(data.iloc[i])

    return pd.Series(estimates, index=data.index)

def apply_savitzky_golay(data: pd.Series, window: int = 51, order: int = 3) -> pd.Series:
    """Применение фильтра Savitzky-Golay для сглаживания данных"""
    if len(data) < window:
        window = len(data) if len(data) % 2 == 1 else len(data) - 1
        if window < order:
            return data

    return pd.Series(
        savgol_filter(data, window_length=window, polyorder=order),
        index=data.index
    )

def apply_adaptive_filter(data: pd.Series, max_threshold: float, min_threshold: float) -> pd.Series:
    """
    Адаптивная фильтрация данных с применением фильтра Калмана и порогов безопасности
    """
    kalman_filtered = apply_kalman_filter(data)

    smoothed = apply_savitzky_golay(kalman_filtered)

    filtered = np.where(
        smoothed > max_threshold,
        np.minimum(smoothed, max_threshold * 0.95),
        smoothed
    )

    filtered = np.where(
        filtered < min_threshold,
        np.maximum(filtered, min_threshold * 1.05),
        filtered
    )

    return pd.Series(filtered, index=data.index)

