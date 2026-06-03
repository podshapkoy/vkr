import pandas as pd
import numpy as np
from config import CONFIG, MIN_PRE_STOP_WINDOW, MAX_PRE_STOP_WINDOW, PRE_STOP_WINDOW_RATIO, HISTORICAL_DROPS_FILE
from stop_detector import detect_stops
from visualizer import plot_overall_current, plot_stop_dynamics


def determine_stop_type(pre_stop, stop_start, historical_durations=None, window_size=None):
    """Определение типа остановки"""
    pre_stop_duration = (pre_stop.index[-1] - pre_stop.index[0]).total_seconds() / 60

    min_duration = max(1, min(10, 0.15 * window_size if window_size else 0.1 * pre_stop_duration))

    if historical_durations is not None and len(historical_durations) > 0:
        min_drop_duration = np.quantile(historical_durations, 0.05)
    else:
        min_drop_duration = 1

    normal_data = pre_stop.iloc[:int(0.8 * len(pre_stop))]
    mean_normal = normal_data['current'].mean()
    std_normal = normal_data['current'].std()

    start_threshold = mean_normal - 1.5 * std_normal
    final_threshold = mean_normal * 0.1

    below_threshold = pre_stop[pre_stop['current'] < start_threshold]
    drop_start = None
    min_duration_td = pd.Timedelta(minutes=min_duration)

    for i in range(len(below_threshold)):
        current_time = below_threshold.index[i]
        next_period = pre_stop.loc[current_time:current_time + min_duration_td]
        if all(next_period['current'] < start_threshold):
            drop_start = current_time
            break

    if drop_start is None:
        return "Не соответствует критериям", "Нет устойчивого снижения", None, None, None, None

    drop_zone = pre_stop.loc[drop_start:stop_start]
    final_value = drop_zone['current'].iloc[-1]
    drop_duration = (stop_start - drop_start).total_seconds() / 60

    if final_value <= final_threshold:
        if drop_duration >= min_drop_duration:
            stop_type = "ПЛАНОВАЯ"
            reason = f"Устойчивый спад до {final_value:.1f} А за {drop_duration:.1f} мин"
        else:
            stop_type = "АВАРИЙНАЯ"
            reason = f"Резкое падение до {final_value:.1f} А за {drop_duration:.1f} мин"
    else:
        stop_type = "Не соответствует критериям"
        reason = f"Не достигнут пороговый ток {final_threshold:.1f} А (конечное значение: {final_value:.1f} А)"

    return stop_type, reason, drop_start, drop_duration, start_threshold, final_threshold


def save_historical_drops(drop_info, stop_duration):
    """Сохраняет информацию о спадах в CSV файл"""
    import pandas as pd
    from pathlib import Path

    file_path = Path(CONFIG["reports_dir"]) / HISTORICAL_DROPS_FILE

    try:
        drop_info['stop_duration'] = stop_duration
        existing_data = pd.read_csv(file_path) if file_path.exists() else pd.DataFrame()

        new_data = pd.DataFrame([drop_info])
        updated_data = pd.concat([existing_data, new_data], ignore_index=True)

        updated_data.to_csv(file_path, index=False)
    except Exception as e:
        print(f"Ошибка при сохранении исторических данных: {str(e)}")


def load_historical_drops():
    """Загружает исторические данные о спадах"""
    from pathlib import Path
    import pandas as pd

    file_path = Path(CONFIG["reports_dir"]) / HISTORICAL_DROPS_FILE

    if file_path.exists():
        try:
            df = pd.read_csv(file_path, parse_dates=['drop_start', 'stop_start'])
            if not df.empty and 'drop_duration' in df.columns:
                return df['drop_duration'].dropna().values
        except Exception as e:
            print(f"Ошибка загрузки исторических данных: {str(e)}")
    return None


def analyze_file(file, df_current, historical_durations=None):
    """Анализ остановок в одном файле"""
    if historical_durations is None:
        historical_durations = load_historical_drops()

    file_data = df_current[df_current['source'] == file].copy()
    stops = detect_stops(file_data)

    print(f"\nАнализ остановок для {file}")
    print(f"Всего обнаружено остановок: {len(stops)}")

    report = {
        'filename': file,
        'total_stops': len(stops),
        'stops': [],
        'plots': []
    }

    try:
        overall_plot = plot_overall_current(file_data, stops, file)
        if overall_plot:
            report['plots'].append(overall_plot)
    except Exception as e:
        print(f"Ошибка при построении общего графика: {str(e)}")

    if stops.empty:
        print("В файле не обнаружено остановок")
        drop_info = {
            'filename': file,
            'stop_number': 0,
            'drop_start': None,
            'stop_start': None,
            'drop_duration': 0,
            'start_current': None,
            'end_current': None,
            'stop_type': "Нет остановок",
            'window_size': 0
        }
        save_historical_drops(drop_info, 0)
        return report, historical_durations

    for i, stop in stops.head(CONFIG["max_stops_to_show"]).iterrows():
        stop_start = stop['start_time']
        stop_end = stop['end_time']
        duration_min = stop['duration'] / 60

        window_size = min(MAX_PRE_STOP_WINDOW,
                          max(MIN_PRE_STOP_WINDOW,
                              duration_min * PRE_STOP_WINDOW_RATIO))

        pre_stop_start = stop_start - pd.Timedelta(minutes=window_size)
        pre_stop = file_data.loc[max(pre_stop_start, file_data.index[0]):stop_start]

        print(f"\nОстановка #{i + 1}: {stop_start} → {stop_end} ({duration_min:.1f} мин)")
        print(f"Анализируемое окно: {window_size:.1f} мин")

        result = determine_stop_type(pre_stop, stop_start, historical_durations, window_size)
        stop_type, reason, drop_start, drop_duration, start_threshold, final_threshold = result

        if None in (start_threshold, final_threshold):
            print("Не удалось определить пороги спада")
            continue

        print(f"Тип: {stop_type}")
        print(f"Причина: {reason}")
        if drop_start:
            print(f"Спад: {drop_start} (длительность: {drop_duration:.1f} мин)")
            print(f"Пороги: старт < {start_threshold:.1f} А, финал < 0.0 А")

            drop_info = {
                'filename': file,
                'stop_number': i + 1,
                'drop_start': drop_start,
                'stop_start': stop_start,
                'drop_duration': drop_duration,
                'start_current': pre_stop.loc[drop_start]['current'],
                'end_current': pre_stop.loc[stop_start]['current'],
                'stop_type': stop_type,
                'window_size': window_size
            }
            save_historical_drops(drop_info, duration_min)

        if drop_start:
            try:
                plot_path = plot_stop_dynamics(
                    pre_stop,
                    stop['start_time'],
                    stop['end_time'],
                    drop_start,
                    pre_stop['current'].median(),
                    start_threshold,
                    final_threshold,
                    file,
                    i + 1,
                    stop_type
                )
                report['plots'].append(plot_path)
            except Exception as e:
                print(f"Ошибка при построении графика: {str(e)}")

        report['stops'].append({
            'number': i + 1,
            'start': stop_start.strftime('%Y-%m-%d %H:%M:%S'),
            'end': stop_end.strftime('%Y-%m-%d %H:%M:%S'),
            'duration': duration_min,
            'type': stop_type,
            'reason': reason,
            'drop_start': drop_start.strftime('%Y-%m-%d %H:%M:%S') if drop_start else "",
            'drop_duration': drop_duration if drop_duration else ""
        })

    return report, historical_durations
