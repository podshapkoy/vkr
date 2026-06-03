import json
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict
from config import CONFIG


def extract_dates_from_filename(filename: str) -> List[datetime]:
    """Извлекает даты из имени файла в формате DD.MM(.YY)"""
    date_pattern = r'(\d{2}\.\d{2})(?:\.(\d{2}))?'
    dates = re.findall(date_pattern, filename)

    parsed_dates = []
    for date_part, year_part in dates:
        day, month = map(int, date_part.split('.'))

        if year_part:
            year = int(year_part)
            full_year = 2000 + year if year < 100 else year
        else:
            if "24" in filename:
                full_year = 2024
            else:
                full_year = 2025

        parsed_dates.append(datetime(year=full_year, month=month, day=day))

    return parsed_dates


def sort_reports_by_date(reports: List[Dict]) -> List[Dict]:
    """Сортирует отчеты по датам из имен файлов"""
    report_date_pairs = []
    for report in reports:
        dates = extract_dates_from_filename(report['filename'])
        if dates:
            start_date = dates[0]
            report_date_pairs.append((start_date, report))

    reports_2024 = []
    reports_2025 = []

    for date, report in report_date_pairs:
        if date.year == 2024:
            reports_2024.append((date, report))
        else:
            reports_2025.append((date, report))

    reports_2024.sort(key=lambda x: x[0])
    reports_2025.sort(key=lambda x: x[0])

    sorted_reports = [r[1] for r in reports_2024] + [r[1] for r in reports_2025]
    return sorted_reports


def generate_json_report(reports: List[Dict]):
    """Генерация отчета в формате JSON с сортировкой файлов по датам"""
    summary_path = Path(CONFIG["reports_dir"]) / "stop_summary.json"

    formatted_reports = []
    for report in reports:
        formatted_report = {
            'filename': report['filename'],
            'total_stops': report['total_stops'],
            'stops': [],
            'plots': report.get('plots', [])
        }
        for stop in report['stops']:
            formatted_stop = {
                'number': stop['number'],
                'start_time': stop['start'],
                'end_time': stop['end'],
                'duration_min': stop['duration'],
                'type': stop.get('type', ""),
                'reason': stop.get('reason', ""),
                'drop_start': stop.get('drop_start', ""),
                'drop_duration': stop.get('drop_duration', "")
            }
            formatted_report['stops'].append(formatted_stop)

        formatted_reports.append(formatted_report)

    sorted_reports = sort_reports_by_date(formatted_reports)

    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(sorted_reports, f, ensure_ascii=False, indent=2)

    print(f"\nJSON отчет сохранен в {summary_path}")