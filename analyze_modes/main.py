"""Точка входа: прогоняет весь пайплайн analyze_modes

Логика:

  1. Загрузка временных рядов и журнала остановок, фильтрация
  2. Сегментация на смены (≥ min_shift_duration_h часов)
  3. Хронологический split: train (первые train_split_frac смен) /
     test (остальные). Все нормировки и рабочая область определяются
     ТОЛЬКО по train. Это устраняет data leakage.
  4. Подгонка нормировок CV*, tons_p95 на train; расчёт балла S для всех смен
  5. Разметка y по аварийным остановкам в горизонтах H часов после смены
  6. Анализ устойчивости ранжирования (temporal + bootstrap + sensitivity по весам)
  7. Извлечение наблюдаемой рабочей области из train top-K%
  8. Валидация рабочей области как риск-индикатора на test (relative risk)
  9. Однофакторная валидация признаков на test (AUC + bootstrap-CI + permutation p)
 10. Мультивариантная модель (logreg, train→test) с калибровкой и importance
 11. Сохранение CSV/JSON, графиков и сводного отчёта
"""

from .config import CONFIG
from .data_loader import load_all, setup_directories
from .predictive import (fit_predict_primary, fit_predict_score_baseline,
                         fit_predict_supplementary)
from .reporting import (
    build_summary_text,
    plot_calibration,
    plot_file_overview,
    plot_predictive_roc,
    plot_score_distributions,
    save_all_shifts_csv,
    save_summary,
    save_top_shifts_csv,
)
from .robustness import assess_robustness
from .scoring import fit_norms, score_shifts
from .shifts import split_into_shifts
from .validation import (
    chronological_split,
    empirical_working_area,
    label_shifts,
    univariate_validation,
    working_area_risk_test,
)


def main() -> None:
    setup_directories()

    print("[1/8] Загрузка данных и фильтрация окон остановок...")
    data, stops, thresholds = load_all()

    print("[2/8] Сегментация на смены...")
    raw_shifts = split_into_shifts(data, thresholds)
    print(f"      Получено {len(raw_shifts)} смен длительностью >= "
          f"{CONFIG['min_shift_duration_h']} ч")
    if not raw_shifts:
        print("Нет валидных смен. Проверь данные.")
        return

    print("[3/8] Хронологический train/test split...")
    train_raw, test_raw = chronological_split(raw_shifts)
    print(f"      train: {len(train_raw)} смен, test: {len(test_raw)} смен "
          f"(frac={CONFIG['train_split_frac']:.2f})")

    print("[4/8] Подгонка нормировок на train, расчёт балла S для всех смен...")
    norms = fit_norms(train_raw)
    print(f"      Нормировки (train): CV* = {norms.cv_star:.4f}, "
          f"tons_p95 = {norms.tons_p95:.1f}")
    scored_all = score_shifts(raw_shifts, norms)

    print("[5/8] Разметка y по аварийным остановкам и устойчивость ранга...")
    scored_all = label_shifts(scored_all, stops)
    train_scored = [s for s in scored_all if s["end_time"] <= train_raw[-1]["end_time"]]
    test_scored  = [s for s in scored_all if s["end_time"] >  train_raw[-1]["end_time"]]

    robustness = assess_robustness(scored_all, top_n=CONFIG["top_n_shifts"])
    if "spearman_rho" in robustness.get("temporal_stability", {}):
        ts = robustness["temporal_stability"]
        print(f"      Stability: ρ={ts['spearman_rho']:.3f}, τ={ts['kendall_tau']:.3f}")
    if "mean_overlap" in robustness.get("weight_sensitivity", {}):
        ws = robustness["weight_sensitivity"]
        print(f"      Top-{ws['top_n']} overlap при ±{ws['radius']:.2f} весов: "
              f"mean={ws['mean_overlap']:.2f}")

    print("[6/8] Рабочая область (train) и её валидация на test...")
    working_area = empirical_working_area(train_scored)
    primary = CONFIG["primary_horizon_h"]
    wa_risk_loose  = working_area_risk_test(test_scored, working_area,
                                            horizon_h=primary, min_params=1)
    wa_risk        = working_area_risk_test(test_scored, working_area,
                                            horizon_h=primary, min_params=2)
    wa_risk_strict = working_area_risk_test(test_scored, working_area,
                                            horizon_h=primary, min_params=3)
    if "relative_risk" in wa_risk_loose:
        print(f"      RR(≥1/3, h={primary}): {wa_risk_loose['relative_risk']:.2f}, "
              f"p_Fisher={wa_risk_loose['fisher_p_one_sided']:.4f}")
    if "relative_risk" in wa_risk:
        print(f"      RR(≥2/3, h={primary}): {wa_risk['relative_risk']:.2f}, "
              f"p_Fisher={wa_risk['fisher_p_one_sided']:.4f}")
    if "relative_risk" in wa_risk_strict:
        print(f"      RR(3/3, h={primary}): {wa_risk_strict['relative_risk']:.2f}, "
              f"p_Fisher={wa_risk_strict['fisher_p_one_sided']:.4f}")

    print("[7/8] Однофакторная валидация и мультивариантная модель (test)...")
    univ_primary = univariate_validation(test_scored, horizon_h=primary)
    predictive = fit_predict_primary(scored_all)
    predictive_supp = fit_predict_supplementary(scored_all)
    score_baseline = fit_predict_score_baseline(scored_all, horizon_h=primary)
    predictive["score_baseline"] = score_baseline
    if "test_roc_auc" in score_baseline:
        ci_sb = score_baseline.get("bootstrap_auc") or {}
        print(f"      Baseline S (h={primary}): AUC = {score_baseline['test_roc_auc']:.3f}, "
              f"CI [{ci_sb.get('ci_low', float('nan')):.3f}; "
              f"{ci_sb.get('ci_high', float('nan')):.3f}], "
              f"p = {score_baseline['permutation_p']:.4f}")

    print("[8/8] Сохранение CSV, JSON, графиков, отчёта...")
    top_path = save_top_shifts_csv(scored_all, CONFIG["top_n_shifts"])
    all_path = save_all_shifts_csv(scored_all)
    print(f"      {top_path}")
    print(f"      {all_path}")

    summary = build_summary_text(
        scored_all, train_scored, test_scored,
        working_area=working_area, working_area_risk=wa_risk,
        working_area_risk_strict=wa_risk_strict,
        working_area_risk_loose=wa_risk_loose,
        robustness=robustness,
        univariate=univ_primary,
        predictive=predictive, predictive_supp=predictive_supp,
    )
    summary_path = save_summary(
        summary,
        working_area=working_area, working_area_risk=wa_risk,
        working_area_risk_strict=wa_risk_strict,
        working_area_risk_loose=wa_risk_loose,
        robustness=robustness,
        univariate=univ_primary,
        predictive=predictive, predictive_supp=predictive_supp,
    )
    plot_score_distributions(scored_all, horizon_h=primary)
    plot_predictive_roc(predictive)
    plot_calibration(predictive)
    print(f"      {summary_path}")

    for fname, df_file in data.groupby("source_file"):
        file_shifts = [s for s in scored_all if s["source_file"] == fname]
        out = plot_file_overview(fname, df_file, file_shifts, stops)
        print(f"      {out}")

    print()
    print(summary)


if __name__ == "__main__":
    main()
