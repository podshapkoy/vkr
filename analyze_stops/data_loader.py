from pathlib import Path
import pandas as pd
from datetime import datetime
import re
from .config import CONFIG


def setup_directories():
    """Создание необходимых директорий"""
    Path(CONFIG["plots_dir"]).mkdir(exist_ok=True)
    Path(CONFIG["reports_dir"]).mkdir(exist_ok=True)


def extract_file_dates(file_path):
    """Извлекает даты из имени файла для сортировки"""
    date_pattern = r'(\d{2}\.\d{2})(?:\.(\d{2}))?'
    dates = re.findall(date_pattern, file_path.name)

    if not dates:
        return datetime.max

    day, month = map(int, dates[0][0].split('.'))
    year_part = dates[0][1]

    if year_part:
        year = int(year_part)
        full_year = 2000 + year if year < 100 else year
    else:
        full_year = 2024 if "24" in file_path.name else 2025

    return datetime(year=full_year, month=month, day=day)


def load_and_prepare_data():
    """Загрузка и подготовка данных из CSV файлов с сортировкой по датам"""
    all_data = {
        'current': pd.DataFrame(),
        'weight': pd.DataFrame()
    }

    csv_files = sorted(
        Path(CONFIG["csv_dir"]).glob("*.csv"),
        key=lambda x: extract_file_dates(x)
    )

    for csv_file in csv_files:
        try:
            print(f"Обработка файла: {csv_file.name}")

            df = pd.read_csv(csv_file, parse_dates=['time'])

            df = df[df['current'].notna()]
            df['current'] = pd.to_numeric(df['current'], errors='coerce')
            df['weight'] = pd.to_numeric(df['weight'], errors='coerce')

            df['source'] = csv_file.name

            df = df.sort_values('time')

            current_df = df[['time', 'current', 'source']].copy()
            weight_df = df[['time', 'weight', 'source']].copy()

            current_df = current_df.set_index('time')
            weight_df = weight_df.set_index('time')

            all_data['current'] = pd.concat([all_data['current'], current_df])
            all_data['weight'] = pd.concat([all_data['weight'], weight_df])

        except Exception as e:
            print(f"Ошибка обработки файла {csv_file.name}: {str(e)}")
            continue

    for key in all_data:
        if not all_data[key].empty and not isinstance(all_data[key].index, pd.DatetimeIndex):
            try:
                all_data[key].index = pd.to_datetime(all_data[key].index)
            except Exception as e:
                print(f"Ошибка преобразования индекса для {key}: {str(e)}")

    return all_data
