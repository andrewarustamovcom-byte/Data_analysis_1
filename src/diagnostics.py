"""Описательные таблицы, временные ряды и сценарии неизвестных расходов."""

import numpy as np
import pandas as pd

from .statistics import analyze_metrics


def daily_metrics(mart, events, config):
    days = pd.date_range(config.start, config.end)
    sizes = mart.groupby("group").size()
    rows = []
    for group in ["control", "test"]:
        ids = set(mart.loc[mart.group.eq(group), "user_id"])
        frame = pd.DataFrame(index=days)
        for name, date, aggregations in [
            ("impressions", "event_time", {"impressions": ("impression_id", "size")}),
            ("clicks", "event_time", {"clicks": ("click_id", "size")}),
            ("orders", "order_date", {"gmv": ("gmv_rub", "sum"), "margin": ("gross_margin_rub", "sum"), "orders": ("order_id", "size")}),
            ("tickets", "ticket_date", {"tickets": ("ticket_id", "size")}),
        ]:
            part = events[name][events[name].user_id.isin(ids)].copy()
            part["date"] = part[date].dt.normalize()
            frame = frame.join(part.groupby("date").agg(**aggregations))
        returns = events["orders"][events["orders"].user_id.isin(ids) & events["orders"].return_observed]
        frame = frame.join(returns.groupby("return_date").size().rename("returns"))
        frame = frame.fillna(0)
        frame["ctr"] = frame.clicks/frame.impressions.replace(0, np.nan)
        frame["gmv_user"] = frame.gmv/sizes[group]
        frame["margin_user"] = frame.margin/sizes[group]
        frame["tickets_user"] = frame.tickets/sizes[group]
        frame["returns_user"] = frame.returns/sizes[group]
        frame["group"], frame["assigned_users"] = group, sizes[group]
        rows.append(frame.rename_axis("date").reset_index())
    return pd.concat(rows, ignore_index=True)


def descriptive_tables(mart, prepared, tables, config):
    ids = set(mart.user_id)
    events = prepared["events"]
    groups = mart[["user_id", "group"]]
    orders = events["orders"].merge(groups, on="user_id", how="inner", validate="many_to_one")
    tickets = events["tickets"].merge(groups, on="user_id", how="inner", validate="many_to_one")
    fees = orders.assign(missing_fee=orders.delivery_fee_rub.isna()).groupby(["group", "category"]).agg(
        orders=("order_id", "size"), missing_fee=("missing_fee", "sum"), median_fee=("delivery_fee_rub", "median")).reset_index()
    fees["missing_share"] = fees.missing_fee/fees.orders
    # Медиана общая для групп: заполнение не использует знание эффекта treatment.
    medians = orders.groupby("category").delivery_fee_rub.median()
    median_global = orders.delivery_fee_rub.median()
    fee_scenarios = []
    imputed = orders.delivery_fee_rub.fillna(orders.category.map(medians)).fillna(median_global)
    for group in ["control", "test"]:
        select = orders.group.eq(group)
        fee_scenarios.append({"group": group, "missing_orders": int(orders.loc[select, "delivery_fee_rub"].isna().sum()),
                              "observed_fee_sum": orders.loc[select, "delivery_fee_rub"].sum(min_count=1),
                              "zero_fill_sum": orders.loc[select, "delivery_fee_rub"].fillna(0).sum(),
                              "pooled_category_median_fill_sum": imputed[select].sum(min_count=1),
                              "note": "Сумма платежей доставки; знак влияния на прибыль неизвестен"})
    return_cohorts = orders.groupby(["group", "order_date"]).agg(orders=("order_id", "size"),
        returns_window=("return_in_window", "sum"), returns_observed=("return_observed", "sum")).reset_index()
    return_cohorts["observed_return_rate"] = return_cohorts.returns_observed/return_cohorts.orders
    returned = orders[orders.return_observed].copy()
    returned["lag_days"] = (returned.return_date-returned.order_date).dt.days
    return_lags = returned[["group", "order_id", "order_date", "return_date", "lag_days"]]
    support = tickets.groupby(["group", "reason"]).agg(tickets=("ticket_id", "size"), sentiment=("sentiment_score", "mean"),
        resolved_24h_share=("resolved_within_24h", "mean")).reset_index()
    support["interpretation"] = "Описательно среди обратившихся; состав зависит от воздействия"
    activity = []
    base = prepared["base"]
    s = events["sessions"].groupby("user_id").size()
    i = events["impressions"].groupby("user_id").size()
    for flag, part in base.groupby("is_bot_candidate"):
        counts = part.user_id.map(i).fillna(0)
        activity.append({"is_bot_candidate": flag, "users": len(part), "impressions_mean": counts.mean(),
                         "impressions_p99": counts.quantile(.99), "impressions_max": counts.max(),
                         "sessions_mean": part.user_id.map(s).fillna(0).mean()})
    funnel = []
    for group, part in mart.groupby("group"):
        counts = [len(part), int(part.reached.sum()), int(part.clicker.sum()), int(part.buyer.sum())]
        for idx, (stage, n) in enumerate(zip(["Назначены", "Увидели показ", "Кликнули", "Купили"], counts)):
            funnel.append({"group": group, "type": "users", "stage": stage, "count": n,
                           "denominator": counts[idx-1] if idx else counts[0],
                           "step_rate": n/counts[idx-1] if idx and counts[idx-1] else 1.0})
        for stage, num, den in [("Показы", "impressions", "impressions"), ("Клики", "clicks", "impressions"), ("Заказы", "orders", "clicks")]:
            denominator = part[den].sum()
            funnel.append({"group": group, "type": "events", "stage": stage, "count": int(part[num].sum()),
                           "denominator": int(denominator), "step_rate": part[num].sum()/denominator if denominator else np.nan})
    new = events["new_premium"].merge(mart[["user_id", "group", "is_premium"]], on="user_id", how="inner", validate="many_to_one")
    premium = new.groupby(["group", "is_premium", "is_trial", "premium_status"]).agg(records=("user_id", "size"), users=("user_id", "nunique")).reset_index()
    margin_profile = orders.groupby(["group", "is_returned"]).agg(orders=("order_id", "size"),
        margin_sum=("gross_margin_rub", "sum"), margin_mean=("gross_margin_rub", "mean"),
        margin_min=("gross_margin_rub", "min"), margin_max=("gross_margin_rub", "max")).reset_index()
    return {"daily_metrics": daily_metrics(mart, events, config), "delivery_missing": fees,
            "delivery_scenarios": pd.DataFrame(fee_scenarios), "return_cohorts": return_cohorts,
            "return_lags": return_lags, "support_reasons": support, "bot_activity": pd.DataFrame(activity),
            "funnel": pd.DataFrame(funnel), "premium_records": premium, "margin_profile": margin_profile}


def economic_scenarios(mart, config):
    rows = []
    # Это сценарная маржа после двух дополнительных расходов, не чистая прибыль.
    support_costs = sorted(set([0., 100., 300., 500.] + ([config.support_cost_rub] if config.support_cost_rub is not None else [])))
    ai_costs = sorted(set([0., .01, .05, .10, .50, 1.] + ([config.ai_cost_per_impression_rub] if config.ai_cost_per_impression_rub is not None else [])))
    from .statistics import mean_effect
    for support in support_costs:
        for ai in ai_costs:
            adjusted = mart.margin - support*mart.tickets_observed - ai*mart.impressions*mart.group.eq("test")
            effect = mean_effect(adjusted[mart.group.eq("control")], adjusted[mart.group.eq("test")], config.alpha)
            rows.append({"support_cost_rub": support, "ai_cost_per_impression_rub": ai, **effect,
                         "note": "Дополнительные расходы сверх записанной маржи; AI только test; полнота follow-up неизвестна"})
    return pd.DataFrame(rows)


def break_even_costs(mart, config):
    """Предельная дополнительная цена AI-показа при заданной цене обращения."""
    from scipy import stats
    c, t = (mart[mart.group.eq(group)] for group in ["control", "test"])
    rows = []
    for cost in sorted(set([0., 100., 300., 500.] + ([config.support_cost_rub] if config.support_cost_rub is not None else []))):
        cm = c.margin-cost*c.tickets_observed
        tm = t.margin-cost*t.tickets_observed
        numerator, denominator = tm.mean()-cm.mean(), t.impressions.mean()
        if denominator <= 0:
            threshold, se = np.nan, np.nan
        else:
            threshold = numerator/denominator
            # Дельта-метод учитывает ковариацию маржи, обращений и показов.
            c_influence = -cm/denominator
            t_influence = tm/denominator-numerator*t.impressions/denominator**2
            se = np.sqrt(c_influence.var(ddof=1)/len(c)+t_influence.var(ddof=1)/len(t))
        z = stats.norm.ppf(1-config.alpha/2)
        rows.append({"support_cost_rub": cost, "ai_break_even_rub_per_impression": threshold,
                     "ci_low": threshold-z*se, "ci_high": threshold+z*se,
                     "note": "Сценарная граница записанной маржи; дополнительные расходы, не чистая прибыль"})
    return pd.DataFrame(rows)


def sensitivity(marts, config):
    frames = []
    for name, mart in marts.items():
        frame = analyze_metrics(mart, config)
        frame["sample"] = name
        frames.append(frame)
    # Верхняя отсечка — только устойчивость, основной эффект остаётся неусечённым.
    for field in ["gmv", "margin"]:
        mart = marts["main"].copy()
        limit = mart[field].quantile(.99)
        mart[field] = mart[field].clip(upper=limit)
        frame = analyze_metrics(mart, config)
        metric = "gmv_user" if field == "gmv" else "margin_user"
        frame = frame[frame.metric.eq(metric)].copy()
        frame["sample"] = f"winsorized_{field}_pooled_p99"
        frame["threshold"] = limit
        frame["note"] = "Проверка усечённой метрики; отличается от исходной гипотезы"
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)
