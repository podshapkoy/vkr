import pandas as pd
from config import CONFIG


def detect_stops(df_current):
    """Обнаружение периодов остановки с улучшенной логикой"""
    df_clean = df_current.copy()

    if isinstance(df_clean.index, pd.DatetimeIndex):
        df_clean['time'] = df_clean.index
    elif 'time' not in df_clean.columns:
        raise KeyError("В DataFrame нет столбца 'time' или datetime-индекса")

    df_clean['smoothed'] = df_clean['current'].rolling(window=3, min_periods=1).mean()

    mean_current = df_clean['smoothed'].mean()
    zero_threshold = max(0.01, 0.01 * mean_current)

    low_current_mask = df_clean['smoothed'] <= zero_threshold
    low_points = df_clean[low_current_mask].copy()

    if low_points.empty:
        return pd.DataFrame(columns=['start_time', 'end_time', 'duration'])

    low_points = low_points.reset_index(drop=True)
    time_diff = low_points['time'].diff().dt.total_seconds()
    low_points['group'] = (time_diff > CONFIG["merge_threshold"]).cumsum()

    stops = low_points.groupby('group').agg(
        start_time=('time', 'first'),
        end_time=('time', 'last'),
        min_value=('smoothed', 'min'),
        mean_value=('smoothed', 'mean')
    )

    stops['duration'] = (stops['end_time'] - stops['start_time']).dt.total_seconds()
    meaningful_stops = stops[
        (stops['duration'] >= CONFIG["min_stop_duration"]) &
        (stops['mean_value'] <= zero_threshold * 1.5)
        ]

    final_stops = []
    for _, stop in meaningful_stops.iterrows():
        stop_period = df_clean[(df_clean['time'] >= stop['start_time']) &
                               (df_clean['time'] <= stop['end_time'])]

        if (stop_period['smoothed'] <= zero_threshold).mean() > 0.95:
            final_stops.append(stop)

    if final_stops:
        return pd.DataFrame(final_stops).reset_index(drop=True)
    return pd.DataFrame(columns=['start_time', 'end_time', 'duration'])