"""Графики с русскими подписями; каждый сохраняется в PNG и SVG."""

import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "marketplace-matplotlib"))
import matplotlib
matplotlib.use("Agg")  # Работает в терминале без графического интерфейса.
import matplotlib.pyplot as plt
import numpy as np


COLORS = {"control": "#2864A8", "test": "#E37A28"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def make_figures(mart, results, folder: Path, config) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    confidence = f"{100*(1-config.alpha):g}%"

    def save(fig, name):
        fig.tight_layout()
        for ext in ["png", "svg"]:
            path = folder/f"{name}.{ext}"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            if ext == "png":
                paths.append(path)
        plt.close(fig)

    counts = results["population_counts"]
    fig, ax = plt.subplots(figsize=(9, 4))
    stages = ["initial_unique_assignment", "include_bots", "main"]
    for group, shift in [("control", -.18), ("test", .18)]:
        values = counts[counts.group.eq(group)].set_index("population").users.reindex(stages)
        bars = ax.bar(np.arange(3)+shift, values, .36, label=group, color=COLORS[group])
        ax.bar_label(bars, fontsize=9)
    ax.set_xticks(range(3), ["Первичное назначение", "Без конфликтов", "Без конфликтов и ботов"])
    ax.set_ylabel("Пользователи")
    ax.set_title("Размеры групп и правила очистки")
    ax.legend()
    save(fig, "01_population")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, kind, title in zip(axes, ["users", "events"], ["Воронка пользователей", "Воронка событий"]):
        part = results["funnel"][results["funnel"].type.eq(kind)]
        stages = list(part.stage.unique())
        for group, shift in [("control", -.18), ("test", .18)]:
            values = part[part.group.eq(group)].set_index("stage")["count"].reindex(stages)
            bars = ax.bar(np.arange(len(stages))+shift, values, .36, label=group, color=COLORS[group])
            ax.bar_label(bars, fontsize=8, rotation=45, padding=2)
        ax.set_xticks(range(len(stages)), stages)
        ax.set_title(title)
        ax.set_ylabel("Количество")
        if kind == "events":
            ax.set_yscale("log")
            ax.set_ylabel("Количество, логарифмическая шкала")
        ax.legend()
    save(fig, "02_funnel")

    fig, axes = plt.subplots(3, 2, figsize=(13, 10))
    for ax, field, title, scale in zip(axes.flat,
        ["ctr", "gmv_user", "margin_user", "returns_user", "tickets_user", "orders"],
        ["CTR, %", "GMV, руб./назначенного пользователя", "Записанная маржа, руб./польз. по дню заказа",
         "Возвраты по дате события, на пользователя", "Обращения по дате события, на пользователя", "Заказы"],
        [100, 1, 1, 1, 1, 1]):
        for group, part in results["daily_metrics"].groupby("group"):
            ax.plot(part.date, part[field]*scale, label=group, color=COLORS[group], alpha=.85)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(alpha=.2)
        ax.legend()
    save(fig, "03_daily")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for idx, (field, label) in enumerate([("gmv", "GMV, руб./польз."), ("margin", "Маржа, руб./польз.")]):
        for group, part in mart.groupby("group"):
            values = part[field].to_numpy()
            sorted_values = np.sort(values)
            axes[idx, 0].step(sorted_values, np.arange(1, len(values)+1)/len(values), color=COLORS[group], label=group)
            # Ненулевые значения не скрывают долю нулей: она явно указана в легенде.
            nonzero = values[values != 0]
            axes[idx, 1].hist(nonzero, bins=40, histtype="step", density=True, color=COLORS[group],
                              label=f"{group}; нулей {(values == 0).mean():.1%}")
        axes[idx, 0].set_xlabel(label)
        axes[idx, 0].set_ylabel("ECDF, все пользователи")
        axes[idx, 0].set_xscale("symlog", linthresh=10)
        axes[idx, 0].set_title("Полный хвост; симметричная логарифмическая шкала")
        axes[idx, 1].set_xlabel(label)
        axes[idx, 1].set_ylabel("Плотность ненулевых значений")
        for ax in axes[idx]:
            ax.legend(fontsize=8)
            ax.grid(alpha=.2)
    save(fig, "04_distributions")

    score = results["metrics"].set_index("metric")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    panels = [(["gmv_user", "margin_user"], "руб./польз.", 1),
              (["buyer_conversion", "ctr", "return_user_window", "support_user_window"], "процентные пункты", 100),
              (["orders_user", "tickets_user_window"], "события/польз.", 1)]
    short_labels = {"gmv_user": "GMV", "margin_user": "Маржа", "buyer_conversion": "Конверсия", "ctr": "CTR",
                    "return_user_window": "Пользователи с возвратом", "support_user_window": "С обращением",
                    "orders_user": "Заказы", "tickets_user_window": "Обращения"}
    for ax, (names, unit, scale) in zip(axes, panels):
        for y, name in enumerate(names):
            row = score.loc[name]
            ax.plot([row.ci_low*scale, row.ci_high*scale], [y, y], color="#2864A8", lw=2)
            ax.scatter(row.effect*scale, y, color="#2864A8")
        ax.set_yticks(range(len(names)), [short_labels[n] for n in names])
        ax.axvline(0, color="gray", ls="--")
        ax.set_xlabel(f"test − control, {unit}")
        ax.set_title(f"Эффекты и {confidence} ДИ до поправки")
    save(fig, "05_effects")

    segments = results["segment_effects"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax, field in zip(axes.flat, ["device", "region", "is_premium", "tenure"]):
        part = segments[segments.segment.eq(field) & segments.metric.eq("gmv_user")]
        for y, row in enumerate(part.itertuples()):
            ax.plot([row.ci_low, row.ci_high], [y, y], lw=2, color="#2864A8")
            ax.scatter(row.effect, y, color="#2864A8")
        labels = [f"{r.level} (C={r.n_control}, T={r.n_test})" for r in part.itertuples()]
        ax.set_yticks(range(len(part)), labels)
        ax.axvline(0, color="gray", ls="--")
        ax.set_xlabel(f"Разница GMV, руб./польз.; {confidence} ДИ до поправки")
        ax.set_title({"device": "Устройство", "region": "Регион", "is_premium": "Premium на старте", "tenure": "Стаж, дней"}[field])
    save(fig, "06_segments")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cats = results["category_breakdown"]
    categories = sorted(cats.category.unique())
    for ax, field, title, scale in zip(axes, ["gmv_user", "return_rate_observed"],
                                     ["Вклад категории в GMV, руб./польз.", "Записанные возвраты, % заказов; follow-up неполон"], [1, 100]):
        for group, shift in [("control", -.18), ("test", .18)]:
            values = cats[cats.group.eq(group)].set_index("category")[field].reindex(categories)
            ax.barh(np.arange(len(categories))+shift, values*scale, .36, label=group, color=COLORS[group])
        ax.set_yticks(range(len(categories)), categories)
        ax.set_title(title)
        ax.legend()
    save(fig, "07_categories")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    support = results["support_reasons"]
    reasons = sorted(support.reason.unique())
    for group, shift in [("control", -.18), ("test", .18)]:
        values = support[support.group.eq(group)].set_index("reason").tickets.reindex(reasons, fill_value=0)
        axes[0].barh(np.arange(len(reasons))+shift, values, .36, color=COLORS[group], label=group)
        lags = results["return_lags"]
        axes[1].hist(lags.loc[lags.group.eq(group), "lag_days"], bins=np.arange(0, 71, 5), histtype="step", color=COLORS[group], label=group)
    axes[0].set_yticks(range(len(reasons)), reasons)
    axes[0].set_title("Причины записанных обращений")
    axes[1].set_title("Задержка записанных возвратов")
    axes[1].set_xlabel("Дней от заказа; только вернувшиеся заказы")
    for ax in axes:
        ax.legend()
    save(fig, "08_support_returns")

    economics = results["economic_scenarios"]
    pivot = economics.pivot(index="support_cost_rub", columns="ai_cost_per_impression_rub", values="effect")
    fig, ax = plt.subplots(figsize=(9, 5))
    vmax = max(abs(pivot.to_numpy()).max(), 1e-9)
    chart = ax.imshow(pivot, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    for y in range(len(pivot)):
        for x in range(len(pivot.columns)):
            ax.text(x, y, f"{pivot.iloc[y, x]:.1f}", ha="center", va="center", color="black")
    ax.set_xticks(range(len(pivot.columns)), [f"{v:g}" for v in pivot.columns])
    ax.set_yticks(range(len(pivot)), [f"{v:g}" for v in pivot.index])
    ax.set_xlabel("Дополнительная стоимость AI, руб./показ test")
    ax.set_ylabel("Дополнительная стоимость обращения, руб.")
    ax.set_title("Сценарная разница маржи, руб./польз.; не чистая прибыль")
    fig.colorbar(chart, ax=ax)
    save(fig, "09_economic_scenarios")
    return paths
