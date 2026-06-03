import json
from pathlib import Path
import pandas as pd
from config import CONFIG


def analyze_drop_history():
    """Анализ всех исторических данных о спадах"""
    history_files = list(Path(CONFIG["history_dir"]).glob("*_history.json"))

    all_drops = []
    for file in history_files:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            all_drops.extend(data)

    if not all_drops:
        print("Нет исторических данных для анализа")
        return None

    df = pd.DataFrame(all_drops)

    type_stats = df.groupby('type')['drop_duration'].describe()

    df['hour'] = pd.to_datetime(df['drop_start']).dt.hour
    hourly_stats = df.groupby('hour')['drop_duration'].mean()

    return {
        'type_stats': type_stats.to_dict(),
        'hourly_stats': hourly_stats.to_dict(),
        'total_drops': len(df),
        'avg_drop_duration': df['drop_duration'].mean(),
        'most_common_type': df['type'].mode()[0]
    }
