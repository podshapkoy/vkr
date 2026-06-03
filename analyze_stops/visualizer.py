import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from pathlib import Path
from config import CONFIG


def plot_overall_current(df_current, stops, filename):
    """Построение общего графика тока с остановками"""
    try:
        fig, ax = plt.subplots(figsize=(14, 6))

        ax.plot(df_current.index, df_current['current'],
                'b-', label='Ток', linewidth=1)

        for _, stop in stops.iterrows():
            ax.axvspan(stop['start_time'], stop['end_time'],
                       color='red', alpha=0.3, label='Остановка' if _ == 0 else "")

        ax.set_title(f'Общий график тока ({filename}) - {len(stops)} остановок')
        ax.set_ylabel('Ток, А')
        ax.set_xlabel('Время')
        ax.legend()
        ax.grid(True)

        date_form = DateFormatter("%Y-%m-%d %H:%M")
        ax.xaxis.set_major_formatter(date_form)

        plt.tight_layout()
        plot_path = Path(CONFIG["plots_dir"]) / f"{Path(filename).stem}_overall.pdf"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        return str(plot_path)

    except Exception as e:
        print(f"Ошибка при построении общего графика: {str(e)}")
        plt.close('all')
        return None


def plot_stop_dynamics(pre_stop, stop_start, stop_end, drop_start, median_current,
                       start_threshold, final_threshold, filename, stop_num, stop_type):
    """Построение графика динамики остановки"""
    try:
        duration = (stop_end - stop_start).total_seconds()
        if duration >= 3600:
            hours = int(duration // 3600)
            minutes = int((duration % 3600) // 60)
            duration_str = f"{hours} ч {minutes} мин"
        else:
            duration_str = f"{int(duration // 60)} мин"

        fig = plt.figure(figsize=(14, 12))
        main_title = (
            f"Тип остановки: {stop_type}\n"
            f"Дата и время: {stop_start.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Длительность: {duration_str}"
        )
        plt.suptitle(main_title, y=1.02, fontsize=14, fontweight='bold')

        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(pre_stop.index, pre_stop['current'], 'b-', label='Ток', linewidth=2)

        if drop_start is not None:
            ax1.axvspan(drop_start, stop_start, color='orange', alpha=0.3, label='Зона спада')

        ax1.axhline(y=median_current, color='g', linestyle='--',
                    label=f'Медиана тока ({median_current:.1f} А)')
        ax1.axhline(y=start_threshold, color='y', linestyle=':',
                    label=f'Порог спада ({start_threshold:.1f} А)')
        ax1.axhline(y=final_threshold, color='r', linestyle='-.',
                    label=f'Порог остановки (0 А)')
        ax1.axvline(x=stop_start, color='r', linestyle='--', linewidth=2, label='Начало остановки')

        ax1.set_title(f'Динамика тока перед остановкой ({filename}) - Остановка #{stop_num}')
        ax1.set_ylabel('Ток, А')
        ax1.set_xlabel('Время')
        ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
        ax1.grid(True)

        if drop_start is not None:
            ax2 = plt.subplot(2, 1, 2)
            drop_zone = pre_stop.loc[drop_start:stop_start]
            ax2.plot(drop_zone.index, drop_zone['current'], 'b-', label='Ток', linewidth=2)

            time_diff = (stop_start - drop_start).total_seconds()
            current_diff = drop_zone['current'].iloc[0] - drop_zone['current'].iloc[-1]
            drop_speed = current_diff / time_diff if time_diff > 0 else 0

            if time_diff >= 3600:
                td_hours = int(time_diff // 3600)
                td_minutes = int((time_diff % 3600) // 60)
                time_diff_str = f"{td_hours} ч {td_minutes} мин"
            else:
                time_diff_str = f"{int(time_diff // 60)} мин"

            ax2.annotate(
                f'Тип: {stop_type}\n'
                f'Дата: {stop_start.strftime("%Y-%m-%d %H:%M:%S")}\n'
                f'Длит. спада: {time_diff_str}\n'
                f'Скорость: {drop_speed * 60:.2f} А/мин\n'
                f'Падение: {current_diff:.1f} А\n'
                f'Начало: {drop_zone["current"].iloc[0]:.1f} А\n'
                f'Конец: {drop_zone["current"].iloc[-1]:.1f} А',
                xy=(drop_start + (stop_start - drop_start) / 2,
                    (drop_zone['current'].max() + drop_zone['current'].min()) / 2),
                bbox=dict(boxstyle='round', fc='w', alpha=0.8),
                fontsize=10
            )

            ax2.fill_between(drop_zone.index, drop_zone['current'], alpha=0.2, color='orange')
            ax2.axhline(y=final_threshold, color='r', linestyle='-.')
            ax2.axvline(stop_start, color='r', linestyle='--', linewidth=2)
            ax2.set_title('Детализация фазы спада')
            ax2.set_ylabel('Ток, А')
            ax2.set_xlabel('Время')
            ax2.grid(True)
            ax2.legend()

        plt.tight_layout()

        plot_filename = f"{Path(filename).stem}_stop_{stop_num}.pdf"
        plot_path = Path(CONFIG["plots_dir"]) / plot_filename
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        return str(plot_path)

    except Exception as e:
        print(f"Ошибка при построении графика: {str(e)}")
        plt.close('all')
        return None