"""Последовательный запуск и сохранение всех проверяемых результатов."""

import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone

from .config import Config
from .data import load_tables
from .diagnostics import break_even_costs, descriptive_tables, economic_scenarios, sensitivity
from .metrics import dictionary
from .preprocessing import prepare, user_mart
from .quality import audit, require_valid
from .report import write_report
from .segments import balance, catalog_breakdown, segment_effects
from .statistics import analyze_metrics, bootstrap_checks, distribution_checks, srm_checks
from .visualization import make_figures


def source_hashes(config):
    result = {}
    for path in sorted(config.data_dir.glob("*.csv")):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b""):
                digest.update(chunk)
        result[path.name] = digest.hexdigest()
    return result


def run_analysis(config: Config | None = None, progress=print) -> dict:
    config = config or Config()
    config.validate()
    table_dir = config.output_dir/"tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    hashes = source_hashes(config)
    progress("1/8 Загрузка и аудит CSV")
    tables = load_tables(config.data_dir)
    results = audit(tables, config)
    for name, frame in results.items():
        frame.to_csv(table_dir/f"{name}.csv", index=False, encoding="utf-8-sig")
    require_valid(results)
    progress("2/8 Выборки и пользовательская витрина")
    prepared = prepare(tables, config)
    marts = {name: user_mart(cohort, prepared["events"], config) for name, cohort in prepared["cohorts"].items()}
    main = marts["main"]
    for name in ["cleaning_log", "population_counts", "assignment_membership", "duplicate_impact"]:
        results[name] = prepared[name]
    results["user_metrics"] = main
    results["metric_definitions"] = dictionary(config)
    results["srm"] = srm_checks(prepared["population_counts"], config)
    results["baseline_balance"], results["baseline_balance_levels"] = balance(main, config)
    progress("3/8 Основные метрики и множественные поправки")
    results["metrics"] = analyze_metrics(main, config)
    progress("4/8 Bootstrap и проверки распределений")
    results["bootstrap"] = bootstrap_checks(main, config)
    results["distribution_tests"] = distribution_checks(main, config)
    progress("5/8 Сегменты, взаимодействия и каталог")
    results["segment_effects"], results["interactions"], results["standardized_effects"] = segment_effects(main, config)
    results["category_breakdown"], results["category_effects"], results["seller_breakdown"] = catalog_breakdown(main, prepared["events"], tables, config)
    progress("6/8 Диагностика, чувствительность и экономика")
    results.update(descriptive_tables(main, prepared, tables, config))
    results["economic_scenarios"] = economic_scenarios(main, config)
    results["break_even_costs"] = break_even_costs(main, config)
    results["sensitivity"] = sensitivity(marts, config)
    for name, frame in results.items():
        frame.to_csv(table_dir/f"{name}.csv", index=False, encoding="utf-8-sig")
    progress("7/8 Графики и отчёт")
    paths = make_figures(main, results, config.output_dir/"figures", config)
    report = write_report(results, config, paths)
    progress("8/8 Проверка исходников и метаданные воспроизводимости")
    if hashes != source_hashes(config):
        raise RuntimeError("Исходные CSV изменились во время анализа")
    packages = ["numpy", "pandas", "scipy", "statsmodels", "matplotlib", "seaborn", "nbformat", "nbclient", "ipykernel"]
    versions = {name: importlib.metadata.version(name) for name in packages}
    metadata = {"created_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
                "config": config.serializable(), "packages": versions, "source_sha256": hashes,
                "ambiguous_first_assignments": prepared["ambiguous_first_assignments"],
                "assumptions": ["Randomization by user; 50/50 configurable", "Complete event logging for structural zeros",
                                "Date-only initial assignment at start of day; ITT sensitivity needs confirmation",
                                "No snapshot date; post-window outcomes are recorded follow-up only",
                                "Margin already includes recorded returns; additional expense scenarios only"]}
    (config.output_dir/"run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (config.output_dir/"requirements-tested.txt").write_text("\n".join(f"{name}=={version}" for name, version in versions.items())+"\n", encoding="utf-8")
    progress(f"Готово: {report}")
    return {"config": config, "tables": results, "figures": paths, "report": report}
