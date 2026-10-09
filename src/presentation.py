"""Компактные таблицы и три графика для итогового ноутбука."""

from pathlib import Path

import pandas as pd

from .visualization import COLORS, plt


def metric_table(metrics, names):
    rows = []
    labels = {
        "gmv_user": "GMV на пользователя, ₽",
        "margin_user": "Записанная маржа на пользователя, ₽",
        "ctr": "CTR, %", "buyer_conversion": "Конверсия в покупку, %",
        "aov": "Средний чек, ₽",
        "return_user_window": "Пользователи с возвратом за 30 дней, %",
        "support_user_window": "Пользователи с обращением за 30 дней, %",
    }
    for name in names:
        row = metrics.loc[name]
        scale = 100 if row.unit == "доля" else 1
        unit = "п.п." if scale == 100 else "₽"
        p = "< 0,001" if row.p_adjusted < .001 else f"{row.p_adjusted:.3f}"
        rows.append({"Метрика": labels[name], "Control": f"{row.control*scale:.2f}",
                     "Test": f"{row.test*scale:.2f}",
                     "Разница": f"{row.effect*scale:+.2f} {unit}",
                     "95% ДИ разницы": f"[{row.ci_low*scale:.2f}; {row.ci_high*scale:.2f}] {unit}",
                     "p с поправкой": p})
    return pd.DataFrame(rows).set_index("Метрика")


def main_figure(metrics):
    names = ["gmv_user", "margin_user", "ctr", "buyer_conversion"]
    titles = ["GMV на пользователя, ₽", "Записанная маржа на пользователя, ₽", "CTR, %", "Конверсия в покупку, %"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    for ax, name, title in zip(axes.flat, names, titles):
        row = metrics.loc[name]
        scale = 100 if row.unit == "доля" else 1
        values = [row.control*scale, row.test*scale]
        bars = ax.bar(["Control", "Test"], values, color=[COLORS["control"], COLORS["test"]], width=.55)
        ax.bar_label(bars, labels=[f"{value:.2f}" for value in values], padding=4)
        ax.set_ylim(0, max(values)*1.3)
        ax.set_title(f"{title}\nИзменение: {row.relative_effect_pct:+.1f}%")
    return fig


def risk_figure(metrics):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, name, title in zip(axes, ["return_user_window", "support_user_window"], ["С возвратом за 30 дней", "С обращением за 30 дней"]):
        row = metrics.loc[name]
        values = [row.control*1000, row.test*1000]
        bars = ax.bar(["Control", "Test"], values, color=[COLORS["control"], COLORS["test"]], width=.55)
        ax.bar_label(bars, labels=[f"{v:.1f}" for v in values], padding=4)
        ax.set_ylim(0, max(values)*1.3)
        ax.set_ylabel("Пользователей на 1 000 назначенных")
        ax.set_title(f"{title}\nДополнительно в test: {row.effect*1000:.1f} на 1 000")
    return fig


def device_figure(segments):
    devices = ["android", "ios", "web"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, metric, title in zip(axes, ["gmv_user", "margin_user"], ["Прирост GMV на пользователя", "Прирост записанной маржи на пользователя"]):
        part = segments[segments.segment.eq("device") & segments.metric.eq(metric)].set_index("level")
        for y, device in enumerate(devices):
            row = part.loc[device]
            ax.plot([row.ci_low, row.ci_high], [y, y], color=COLORS["test"], lw=3)
            ax.scatter(row.effect, y, color=COLORS["test"], zorder=3)
        ax.set_yticks(range(len(devices)), ["Android", "iOS", "Web"])
        ax.invert_yaxis()
        ax.axvline(0, color="gray", ls="--", lw=1)
        ax.set_xlabel("Test − control, ₽; 95% доверительный интервал")
        ax.set_title(title)
    return fig


def show_figure(fig, name, root: Path):
    from IPython.display import Image, display
    folder = root / "reports" / "figures" / "summary"
    folder.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for extension in ["png", "svg"]:
        fig.savefig(folder / f"{name}.{extension}", dpi=150, bbox_inches="tight")
    display(Image(filename=str(folder / f"{name}.png"), width=900))
    plt.close(fig)
