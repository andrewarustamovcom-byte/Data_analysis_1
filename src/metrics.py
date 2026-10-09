"""Определения метрик: формула, единица и семейство гипотез."""

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class Metric:
    name: str
    label: str
    kind: str
    numerator: str
    denominator: str | None = None
    family: str = "secondary"
    unit: str = "на пользователя"
    window: str = "30 дней эксперимента"


def definitions(config) -> list[Metric]:
    metrics = [
        Metric("gmv_user", "GMV на пользователя", "mean", "gmv", family="confirmatory", unit="руб./польз."),
        Metric("margin_user", "Записанная валовая маржа на пользователя", "mean", "margin", family="confirmatory", unit="руб./польз.", window="Заказы эксперимента; маржа уже учитывает записанные возвраты"),
        Metric("return_user_window", "Доля пользователей с возвратом в эксперименте", "binary", "return_user_window", family="confirmatory", unit="доля"),
        Metric("support_user_window", "Доля пользователей с обращением в эксперименте", "binary", "support_user_window", family="confirmatory", unit="доля"),
        Metric("buyer_conversion", "Конверсия пользователя в покупку", "binary", "buyer", unit="доля"),
        Metric("reach", "Охват рекомендациями", "binary", "reached", unit="доля"),
        Metric("ctr", "CTR уникальных записей показов", "ratio", "clicks", "impressions", unit="доля"),
        Metric("ctr_unflagged", "CTR без помеченных показов", "ratio", "clicks_unflagged", "impressions_unflagged", family="sensitivity", unit="доля"),
        Metric("orders_user", "Заказы на пользователя", "mean", "orders"),
        Metric("clicks_user", "Клики на пользователя", "mean", "clicks"),
        Metric("sessions_user", "Сессии на пользователя", "mean", "sessions"),
        Metric("duration_user", "Время в сессиях на пользователя", "mean", "duration_total", unit="сек./польз."),
        Metric("aov", "Средний чек", "ratio", "gmv", "orders", unit="руб./заказ"),
        Metric("click_to_order", "Заказы на клик", "ratio", "orders", "clicks", unit="доля"),
        Metric("return_rate_window", "Возвраты в эксперименте / заказы", "ratio", "returns_window", "orders", unit="доля"),
        Metric("return_rate_observed", "Записанные возвраты / заказы", "ratio", "returns_observed", "orders", unit="доля", window="Наблюдаемый follow-up; полнота не подтверждена"),
        Metric("returned_gmv_user", "GMV возвращённых заказов на пользователя", "mean", "returned_gmv_observed", unit="руб./польз.", window="Записанный follow-up; это стоимость заказов, не сумма возмещения"),
        Metric("tickets_user_window", "Обращения на пользователя в эксперименте", "mean", "tickets_window"),
        Metric("tickets_per_order_observed", "Записанные обращения на заказ", "ratio", "tickets_observed", "orders", window="Наблюдаемый follow-up; полнота не подтверждена"),
        Metric("support_user_observed", "Доля пользователей с записанным обращением", "binary", "support_user_observed", unit="доля", window="Наблюдаемый follow-up; полнота не подтверждена"),
    ]
    if config.observation_end:
        metrics.append(Metric("return_rate_fixed_followup", "Возвраты за фиксированный срок / созревшие заказы", "ratio", "returns_fixed_followup", "mature_orders", unit="доля", window=f"{config.followup_days} дней после заказа; только зрелые заказы"))
    if config.assume_full_refund:
        metrics.append(Metric("net_gmv_user", "GMV после условного полного возмещения", "mean", "net_gmv_full_refund", family="sensitivity", unit="руб./польз.", window="Сценарий полного возврата стоимости"))
    return metrics


def dictionary(config) -> pd.DataFrame:
    return pd.DataFrame([asdict(metric) for metric in definitions(config)])
