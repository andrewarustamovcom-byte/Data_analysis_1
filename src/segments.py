"""Баланс, эффекты исходных сегментов и вклад категорий/продавцов."""

import numpy as np
import pandas as pd
from scipy import stats

from .statistics import analyze_metrics, correct_family, mean_effect


SEGMENTS = ["device", "region", "is_premium", "tenure"]
CONFIRMATORY = ["gmv_user", "margin_user", "return_user_window", "support_user_window"]


def balance(mart, config):
    rows, levels = [], []
    fields = [f for f in SEGMENTS + ["age_group", "traffic_source"] if f in mart]
    for field in fields:
        tab = pd.crosstab(mart[field].fillna("missing"), mart.group).reindex(columns=["control", "test"], fill_value=0)
        tab = tab[tab.sum(axis=1).gt(0)]
        if len(tab) < 2:
            continue
        chi, p, df, expected = stats.chi2_contingency(tab, correction=False)
        c, t = tab.control/tab.control.sum(), tab.test/tab.test.sum()
        sd = np.sqrt((c*(1-c)+t*(1-t))/2)
        smd = (t-c)/sd.replace(0, np.nan)
        rows.append({"field": field, "chi2": chi, "p_value": p, "df": df,
                     "min_expected_count": expected.min(), "max_abs_smd": smd.abs().max(), "family": "balance"})
        for level in tab.index:
            levels.append({"field": field, "level": str(level), "control_share": c[level], "test_share": t[level], "smd": smd[level]})
    return correct_family(pd.DataFrame(rows), config), pd.DataFrame(levels)


def segment_effects(mart, config):
    effects, interactions, standardized = [], [], []
    overall = analyze_metrics(mart, config).set_index("metric")
    for field in SEGMENTS:
        field_rows = []
        sparse_metrics = set()
        for level, part in mart.groupby(field, observed=True):
            sizes = part.groupby("group").size()
            if min(sizes.get("control", 0), sizes.get("test", 0)) < 2:
                continue
            selected = analyze_metrics(part, config)
            selected = selected[selected.metric.isin(CONFIRMATORY)].copy()
            selected["segment"] = field
            selected["level"] = str(level)
            selected["weight"] = len(part)/len(mart)
            selected["family"] = "segment_exploration"
            for metric in ["return_user_window", "support_user_window"]:
                rate = part[metric].mean()
                if min(sizes.min()*rate, sizes.min()*(1-rate)) < 5:
                    sparse_metrics.add(metric)
            effects.append(selected)
            field_rows.append(selected)
        if not field_rows:
            continue
        frame = pd.concat(field_rows, ignore_index=True)
        for metric, part in frame.groupby("metric"):
            if metric in sparse_metrics:
                interactions.append({"segment": field, "metric": metric, "statistic": np.nan,
                                     "df": np.nan, "p_value": np.nan, "family": "interaction",
                                     "method": "Not tested: rare events in at least one level; Wald calibration unreliable"})
                continue
            # Совместная проверка равенства эффектов по непересекающимся уровням.
            usable = part[part.se.gt(0) & part.effect.notna()]
            if len(usable) >= 2:
                weights = 1/usable.se.pow(2)
                common = np.average(usable.effect, weights=weights)
                q = float((weights*(usable.effect-common).pow(2)).sum())
                p = stats.chi2.sf(q, len(usable)-1)
                interactions.append({"segment": field, "metric": metric, "statistic": q,
                                     "df": len(usable)-1, "p_value": p, "family": "interaction",
                                     "method": "Asymptotic Wald interaction, independent strata"})
            std = float((part.weight*part.effect).sum())
            standardized.append({"segment": field, "metric": metric, "overall_effect": overall.loc[metric, "effect"],
                                 "standardized_effect": std, "weight_sum": part.weight.sum(),
                                 "sign_differs": bool(np.sign(std) != np.sign(overall.loc[metric, "effect"]))})
    return (correct_family(pd.concat(effects, ignore_index=True), config),
            correct_family(pd.DataFrame(interactions), config), pd.DataFrame(standardized))


def catalog_breakdown(mart, events, tables, config):
    orders = events["orders"].merge(mart[["user_id", "group"]], on="user_id", how="inner", validate="many_to_one")
    catalog = tables["products"][["product_id", "seller_id"]]
    orders = orders.merge(catalog, on="product_id", validate="many_to_one")
    category = orders.groupby(["group", "category"]).agg(orders=("order_id", "size"), buyers=("user_id", "nunique"),
                gmv=("gmv_rub", "sum"), margin=("gross_margin_rub", "sum"), returns=("return_observed", "sum")).reset_index()
    size_map = mart.groupby("group").size()
    category["gmv_user"] = category.gmv/category.group.map(size_map)
    category["return_rate_observed"] = category.returns/category.orders
    effects = []
    for name, part in orders.groupby("category"):
        sums = part.groupby("user_id").agg(gmv=("gmv_rub", "sum"), margin=("gross_margin_rub", "sum"))
        all_users = mart[["user_id", "group"]].merge(sums, left_on="user_id", right_index=True, how="left", validate="one_to_one").fillna({"gmv": 0, "margin": 0})
        for field in ["gmv", "margin"]:
            c, t = (all_users.loc[all_users.group.eq(g), field] for g in ["control", "test"])
            effects.append({"category": name, "metric": field, **mean_effect(c, t, config.alpha),
                            "n_control": len(c), "n_test": len(t), "family": "category_exploration"})
    seller = orders.groupby(["group", "seller_id"]).agg(orders=("order_id", "size"), gmv=("gmv_rub", "sum"),
                margin=("gross_margin_rub", "sum"), returns=("return_observed", "sum")).reset_index()
    seller = seller.merge(tables["sellers"], on="seller_id", validate="many_to_one")
    return category, correct_family(pd.DataFrame(effects), config), seller
