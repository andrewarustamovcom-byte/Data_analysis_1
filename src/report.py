"""Короткий бизнес-отчёт с явными ограничениями и ссылками на графики."""

from pathlib import Path

import numpy as np


def markdown_table(frame, columns=None, max_rows=20):
    part = frame[columns] if columns else frame
    part = part.head(max_rows)
    header = "| " + " | ".join(part.columns) + " |"
    divider = "| " + " | ".join(["---"]*len(part.columns)) + " |"
    rows = []
    for values in part.itertuples(index=False, name=None):
        cells = [f"{v:.4g}" if isinstance(v, (float, np.floating)) else str(v).replace("|", "/").replace("\n", " ") for v in values]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, divider, *rows])


def write_report(results: dict, config, paths: list[Path]) -> Path:
    confidence = f"{100*(1-config.alpha):g}%"
    guard_upper = f"{100*(1-config.alpha/2):g}%"
    metrics = results["metrics"].set_index("metric")
    gmv, margin = (metrics.loc[k] for k in ["gmv_user", "margin_user"])
    counts = results["population_counts"]
    main = counts[counts.population.eq("main")].set_index("group")
    srm = results["srm"][results["srm"].population.eq("initial_unique_assignment") & results["srm"].count_field.eq("users")]
    srm_flag = bool(srm.srm_flag.any())
    if srm_flag:
        decision = "Решение о запуске отложено: выявлен SRM назначенной аудитории, нужно исследовать механизм назначения."
    elif gmv.significant_adjusted and gmv.effect < 0:
        decision = "Общий запуск не рекомендуется: основная метрика статистически значимо ухудшается."
    elif margin.significant_adjusted and margin.effect < 0:
        decision = "Общий запуск не рекомендуется: записанная валовая маржа статистически значимо ухудшается."
    elif gmv.significant_adjusted and gmv.effect > 0:
        decision = "Рост основной метрики подтверждается в выбранной выборке. Решение о запуске условно: необходимо проверить допустимый вред и неизвестные расходы."
    else:
        decision = "Доказательств для общего запуска недостаточно: основная метрика не подтверждает улучшение после поправки. Нужен новый тест или уточнение минимально полезного эффекта."
    guardrails = []
    for name, threshold in [("return_user_window", config.max_return_increase_pp), ("support_user_window", config.max_support_increase_pp)]:
        row = metrics.loc[name]
        upper_pp = 100*row.ci_high
        # Верхние 97.5% границы дают односторонний Bonferroni для двух guardrail.
        verdict = "Порог не задан — безопасность не установлена" if threshold is None else (
            "В пределах порога по верхней границе" if upper_pp <= threshold else "Допустимость ухудшения не подтверждена")
        guardrails.append(f"- {row.label}: эффект {100*row.effect:.3f} п.п.; верхняя граница {upper_pp:.3f} п.п.; {verdict}.")
    display = results["metrics"].copy()
    display["effect_display"] = [f"{r.effect*100:.3f} п.п." if r.unit == "доля" else f"{r.effect:.3f} {r.unit}" for r in display.itertuples()]
    display["ci_display"] = [f"[{r.ci_low*100:.3f}; {r.ci_high*100:.3f}] п.п." if r.unit == "доля" else f"[{r.ci_low:.3f}; {r.ci_high:.3f}]" for r in display.itertuples()]
    key = display[display.metric.isin(["gmv_user", "margin_user", "ctr", "buyer_conversion", "return_rate_window", "return_rate_observed", "support_user_window", "tickets_per_order_observed"])]
    interactions = results["interactions"]
    significant = interactions[interactions.significant_adjusted]
    interaction_note = ("После FDR обнаружены взаимодействия: " + ", ".join(f"{r.segment}/{r.metric}" for r in significant.itertuples())
                        if len(significant) else "После FDR убедительных различий эффекта между исходными сегментами не выявлено.")
    missing_business = any(v is None for v in [config.minimum_gmv_lift, config.max_return_increase_pp,
                            config.max_support_increase_pp, config.support_cost_rub, config.ai_cost_per_impression_rub])
    business_note = "Бизнес-пороги и расходы заданы не полностью; безусловное разрешение на запуск не формируется." if missing_business else "Бизнес-пороги заданы; оценка применима к указанным допущениям об учёте затрат."
    if config.minimum_gmv_lift is not None:
        threshold = config.minimum_gmv_lift*gmv.control
        business_note += f" Минимальный выигрыш {threshold:.3f} руб./польз.; нижняя граница эффекта {gmv.ci_low:.3f} руб./польз."
    text = f"""# Результат A/B-эксперимента AI-рекомендаций

{decision}

{business_note}

## Основные результаты

Основная выборка: control — {main.loc['control', 'users']:,}, test — {main.loc['test', 'users']:,} пользователей.
GMV/user: control {gmv.control:.2f} руб., test {gmv.test:.2f} руб.; разница {gmv.effect:.2f} руб.
({gmv.relative_effect_pct:.2f}%), {confidence} ДИ [{gmv.ci_low:.2f}; {gmv.ci_high:.2f}], p после Holm = {gmv.p_adjusted:.4g}.
Записанная маржа/user: разница {margin.effect:.2f} руб., {confidence} ДИ [{margin.ci_low:.2f}; {margin.ci_high:.2f}].
Это валовая маржа с записанными возвратами, не чистая прибыль и не маржа на дату 30 июля.

{markdown_table(key, ['label', 'effect_display', 'ci_display', 'p_adjusted', 'method'])}

Интервалы в таблице — индивидуальные, до поправки. Подтверждающее семейство Holm:
GMV/user, маржа/user, доли пользователей с возвратом и обращением внутри эксперимента.
Вторичные проверки имеют отдельную FDR-поправку. p-value не измеряет размер бизнес-выигрыша.

## Ограничения по вреду

{chr(10).join(guardrails)}

Верхние {guard_upper} границы для двух guardrail соответствуют односторонней поправке Bonferroni;
незначимое ухудшение само по себе не подтверждает безопасность.
Возвраты на заказ и обращения на заказ оцениваются с кластеризацией по пользователю.

## Данные и допущения

- Окно заказов и основного воздействия: {config.start} — {config.end}, включительно.
- Рандомизация предполагается по user_id; ожидаемые группы {1-config.expected_test_share:.0%}/{config.expected_test_share:.0%} — допущение.
- Исключены конфликты назначения и is_bot_candidate=1. Выборка после исключений не является полным ITT;
  фильтрация может влиять на причинную интерпретацию. Первичное назначение и включение ботов — чувствительность.
- Временная точность назначения — день. Для первичного назначения предполагается старт в начале дня;
  причинный ITT для конфликтующих пользователей требует подтверждения журнала рандомизации.
- Нули неактивных пользователей основаны на допущении о полноте журнала событий.
- Флаги дублей имеют самостоятельные ID и связанные клики: уникальные события сохранены;
  CTR без флага вынесен в чувствительность. Восстановимого оригинала по естественному ключу нет.
- Дата полной выгрузки: {config.observation_end or 'неизвестна'}. Последнее событие не доказывает полноту follow-up.
  Записанные поздние возвраты описательны; без даты выгрузки нет сопоставимого подтверждённого 30-дневного окна после каждого заказа.
- gross_margin_rub уже отражает возвраты; повторное вычитание возвратов не применяется.
- delivery_fee_rub не заполняется для основной метрики. Суммы доставки со сценарными заполнениями
  сохранены отдельно; экономический знак и включение доставки в маржу не установлены.
- Стоимость возвращённых заказов не объявляется суммой возмещения. Условный net GMV включается только отдельной настройкой.

## Сегменты и чувствительность

{interaction_note}
Сегменты исследовательские. Для ограниченного запуска нужны воспроизводимость эффекта,
допустимый вред и подтверждающий эксперимент в выбранном исходном сегменте.
Категории — вклад исходов на всей аудитории; покупатели категории не объявляются исходным сегментом.

Чувствительность: включение ботов, первичное назначение, CTR без помеченных показов,
верхняя отсечка GMV/маржи на общей 99-й перцентили. Усечённая метрика имеет другую гипотезу.
Bootstrap: {config.bootstrap_repeats} повторений, seed {config.seed}; KS: {config.permutation_repeats} перестановок.
Для KS используются перестановки пользовательских меток из-за нулей и совпадений.
MDE — нормальное приближение чувствительности при мощности {config.power:.0%}; для редких событий
и небольших сегментов требует отдельного планирования. Наблюдаемая мощность не рассчитывается.

## Экономика и следующий шаг

Сценарий = записанная маржа − дополнительная стоимость записанных обращений
− стоимость AI-показов в test. Учитывается предположение, что эти расходы ещё не включены в маржу.
Расходы доставки, возвраты и подписная выручка повторно не добавляются.
Результат не включает все расходы бизнеса и поэтому не является чистой прибылью.

Предельная дополнительная цена AI-показа, при которой сценарная разница маржи
равна нулю (руб./показ test; интервалы дельта-метода):

{markdown_table(results['break_even_costs'], ['support_cost_rub', 'ai_break_even_rub_per_impression', 'ci_low', 'ci_high'])}

При цене выше точечной границы ожидаемый сценарный выигрыш становится отрицательным;
нижняя граница интервала показывает более консервативный ориентир с учётом неопределённости.

Перед решением задать минимально полезный рост и допустимое ухудшение в п.п.,
подтвердить полноту последующего наблюдения, состав gross_margin_rub и стоимость AI/поддержки.
Для сегментного запуска повторить эксперимент в заранее выбранном сегменте.
Один синтетический эксперимент не доказывает долгосрочный эффект, удержание или LTV.

## Графики

"""
    captions = [
        "Размеры групп показывают объём и асимметрию исключений; не доказывают баланс сами по себе.",
        "Воронки показывают охват и этапы; событийные и пользовательские знаменатели различаются.",
        "Дневные значения описательны; день остановки по ним не выбирается. Маржа привязана к заказу, но уже содержит записанные возвраты.",
        "ECDF включает нули, гистограмма — ненулевые значения. Форма распределения не подменяет проверку среднего.",
        f"Линии — индивидуальные {confidence} ДИ; единицы разделены на панели, множественная поправка — в таблице.",
        "Сегментные интервалы показывают точность; вывод о неоднородности требует теста взаимодействия и FDR.",
        "GMV категорий нормирован на всю аудиторию. Записанная доля возвратов ограничена неизвестной полнотой наблюдения.",
        "Причины и задержки описаны среди наблюдаемых событий; нулевая частота не доказывает отсутствие будущих обращений/возвратов.",
        "Знак сценарной разницы меняется при неизвестных расходах; сетка значений не является согласованным бюджетом.",
    ]
    for path, caption in zip(paths, captions):
        text += f"![{path.stem}](figures/{path.name})\n\n{caption}\n\n"
    text += "Все расчёты: [tables/](tables/). Настройки и хэши исходников: [run_metadata.json](run_metadata.json).\n"
    path = config.output_dir / "analysis_report.md"
    path.write_text(text, encoding="utf-8")
    return path
