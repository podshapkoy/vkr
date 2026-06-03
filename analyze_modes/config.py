"""Единая конфигурация модуля analyze_modes"""

CONFIG = {
    "csv_dir": "data/csv_dir",
    "stops_file": "reports_type/historical_drops.csv",
    "reports_dir": "reports_mode",
    "plots_dir": "reports_mode/plots",
    "top_n_shifts": 10,

    "min_shift_duration_h": 11.0,
    "shift_day_start_hour": 8,
    "shift_night_start_hour": 20,

    # только восстановление после остановки вырезаем (downtime + рестарт)
    # pre-окно не режем, иначе теряется предаварийный сигнал в CV смены
    "stop_pre_window_min": 0,
    "stop_post_window_min": 30,

    # окно конца смены, по которому считаются признаки риска
    "late_window_hours": 2,

    # веса S-балла
    # V — стабильность, M — запас до пределов, P — объём переработки
    "score_weights": {"V": 0.4, "M": 0.3, "P": 0.3},

    # хронологический split: первые train_split_frac смен — train (для нормировок,
    # рабочей области и обучения модели), остальные — test (валидация)
    "train_split_frac": 0.70,

    # основной горизонт валидации (часы после конца смены)
    "primary_horizon_h": 24,
    "validation_horizons_h": [12, 24, 48],

    # доля топ-смен для рабочей области
    "working_area_top_frac": 0.15,

    # параметры robustness-анализа
    "weight_perturbation_iters": 200,   # число случайных весов на симплексе
    "weight_perturbation_radius": 0.10,  # макс отклонение от заданных весов
    "ranking_stability_iters": 1000,     # bootstrap для Spearman ρ

    # bootstrap и permutation
    "bootstrap_iters": 1000,
    "permutation_iters": 2000,
    "random_seed": 42,
}
