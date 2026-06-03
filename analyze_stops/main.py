from data_loader import setup_directories, load_and_prepare_data
from stop_analyzer import analyze_file
from report_generator import generate_json_report


def main():
    """Основная функция обработки"""
    setup_directories()

    print("Загрузка данных...")
    all_data = load_and_prepare_data()

    if all_data['current'].empty:
        print("Нет данных для анализа")
        return

    reports = []
    historical_durations = None
    unique_files = all_data['current']['source'].unique()

    for file in unique_files:
        report, historical_durations = analyze_file(file, all_data['current'], historical_durations)
        reports.append(report)

    generate_json_report(reports)

    print("\nОбработка завершена")


if __name__ == "__main__":
    main()
