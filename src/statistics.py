"""Проверки эффектов; независимая единица всегда пользователь."""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportions_ztest

from .metrics import definitions


def mean_effect(control, test, alpha=0.05) -> dict:
    c, t = np.asarray(control, dtype=float), np.asarray(test, dtype=float)
    if min(len(c), len(t)) < 2 or not (np.isfinite(c).all() and np.isfinite(t).all()):
        raise ValueError("Нужны ≥2 конечных пользовательских значения в каждой группе")
    vc, vt = c.var(ddof=1) / len(c), t.var(ddof=1) / len(t)
    se = np.sqrt(vc + vt)
    delta = t.mean() - c.mean()
    df = (vc + vt) ** 2 / (vc ** 2 / (len(c)-1) + vt ** 2 / (len(t)-1)) if se else np.inf
    critical = stats.t.ppf(1-alpha/2, df)
    p = 2 * stats.t.sf(abs(delta / se), df) if se else float(delta == 0)
    return {"control": c.mean(), "test": t.mean(), "effect": delta, "se": se,
            "ci_low": delta-critical*se, "ci_high": delta+critical*se,
            "p_value": p, "method": "Welch t-test"}


def binary_effect(control, test, alpha=0.05) -> dict:
    c, t = np.asarray(control), np.asarray(test)
    if not np.isin(c, [0, 1]).all() or not np.isin(t, [0, 1]).all():
        raise ValueError("Бинарная метрика должна содержать только 0 и 1")
    c, t = c.astype(int), t.astype(int)
    pc, pt = c.mean(), t.mean()
    table = np.array([[t.sum(), len(t)-t.sum()], [c.sum(), len(c)-c.sum()]])
    pooled = (c.sum()+t.sum())/(len(c)+len(t))
    expected = np.outer(table.sum(axis=1), table.sum(axis=0))/table.sum()
    if expected.min() < 5:
        p = stats.fisher_exact(table).pvalue
        method = "Fisher exact; Newcombe CI"
    else:
        p = proportions_ztest([t.sum(), c.sum()], [len(t), len(c)])[1]
        method = "Two-proportion Z-test; Newcombe CI"
    if pooled in [0, 1]:
        p = 1.0
    # Newcombe (Wilson) сохраняет ненулевую неопределённость при нуле событий.
    from statsmodels.stats.proportion import confint_proportions_2indep
    lo, hi = confint_proportions_2indep(t.sum(), len(t), c.sum(), len(c), method="newcomb", compare="diff", alpha=alpha)
    return {"control": pc, "test": pt, "effect": pt-pc,
            "se": np.sqrt(pt*(1-pt)/len(t)+pc*(1-pc)/len(c)),
            "ci_low": lo, "ci_high": hi, "p_value": p, "method": method}


def ratio_estimate(numerator, denominator) -> tuple[float, float]:
    x, y = np.asarray(numerator, dtype=float), np.asarray(denominator, dtype=float)
    if len(x) < 2 or y.mean() <= 0:
        return np.nan, np.nan
    value = x.sum()/y.sum()
    # Линеаризация сохраняет ковариацию числителя и знаменателя внутри пользователя.
    influence = (x-value*y)/y.mean()
    return value, influence.var(ddof=1)/len(x)


def ratio_effect(c_num, c_den, t_num, t_den, alpha=0.05) -> dict:
    pc, vc = ratio_estimate(c_num, c_den)
    pt, vt = ratio_estimate(t_num, t_den)
    se, delta = np.sqrt(vc+vt), pt-pc
    z = stats.norm.ppf(1-alpha/2)
    p = 2*stats.norm.sf(abs(delta/se)) if se > 0 else (float(delta == 0) if np.isfinite(delta) else np.nan)
    return {"control": pc, "test": pt, "effect": delta, "se": se,
            "ci_low": delta-z*se, "ci_high": delta+z*se,
            "p_value": p, "method": "User-cluster delta method"}


def correct_family(frame: pd.DataFrame, config) -> pd.DataFrame:
    frame = frame.copy()
    if frame.empty:
        for column in ["p_value", "family", "p_adjusted", "correction", "significant_adjusted"]:
            frame[column] = pd.Series(dtype="object")
        return frame
    frame["p_adjusted"] = np.nan
    frame["correction"] = ""
    for family, group in frame.groupby("family", sort=False):
        valid = group.index[group.p_value.notna()]
        if not len(valid):
            continue
        method = "holm" if family == "confirmatory" else "fdr_bh"
        frame.loc[valid, "p_adjusted"] = multipletests(frame.loc[valid, "p_value"], alpha=config.alpha, method=method)[1]
        frame.loc[valid, "correction"] = method
    frame["significant_adjusted"] = frame.p_adjusted.lt(config.alpha)
    return frame


def analyze_metrics(mart: pd.DataFrame, config) -> pd.DataFrame:
    c, t = (mart[mart.group.eq(g)] for g in ["control", "test"])
    rows = []
    for m in definitions(config):
        if m.kind == "ratio":
            estimate = ratio_effect(c[m.numerator], c[m.denominator], t[m.numerator], t[m.denominator], config.alpha)
        elif m.kind == "binary":
            estimate = binary_effect(c[m.numerator], t[m.numerator], config.alpha)
        else:
            estimate = mean_effect(c[m.numerator], t[m.numerator], config.alpha)
        estimate.update(metric=m.name, label=m.label, family=m.family, unit=m.unit, window=m.window,
                        n_control=len(c), n_test=len(t), numerator_control=c[m.numerator].sum(), numerator_test=t[m.numerator].sum(),
                        denominator_control=c[m.denominator].sum() if m.denominator else len(c),
                        denominator_test=t[m.denominator].sum() if m.denominator else len(t))
        estimate["relative_effect_pct"] = 100*estimate["effect"]/estimate["control"] if estimate["control"] != 0 else np.nan
        # MDE — приближение плановой чувствительности, не post-hoc power.
        multiplier = stats.norm.ppf(1-config.alpha/2) + stats.norm.ppf(config.power)
        estimate["mde_absolute_approx"] = multiplier*estimate["se"]
        estimate["mde_note"] = "Normal approximation; rare events require dedicated power planning"
        if m.kind == "binary" and estimate["se"] == 0:
            estimate["mde_absolute_approx"] = np.nan
            estimate["mde_note"] = "Degenerate observed rate: cannot estimate MDE from zero variance"
        estimate["mde_relative_pct_approx"] = 100*estimate["mde_absolute_approx"]/abs(estimate["control"]) if estimate["control"] else np.nan
        rows.append(estimate)
    return correct_family(pd.DataFrame(rows), config)


def bootstrap_values(values, repeats, rng, ratio=False):
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    unique, counts = np.unique(values, axis=0, return_counts=True)
    results = []
    # Multinomial по одинаковым строкам точно эквивалентен bootstrap пользователей.
    for offset in range(0, repeats, 64):
        weights = rng.multinomial(len(values), counts/len(values), size=min(64, repeats-offset))
        sums = weights @ unique
        if ratio:
            result = np.divide(sums[:, 0], sums[:, 1], out=np.full(len(sums), np.nan), where=sums[:, 1] > 0)
        else:
            result = sums[:, 0]/len(values)
        results.append(result)
    return np.concatenate(results)


def bootstrap_checks(mart, config) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed)
    c, t = (mart[mart.group.eq(g)] for g in ["control", "test"])
    selected = {"gmv_user", "margin_user", "ctr", "aov", "return_rate_observed", "tickets_per_order_observed"}
    rows = []
    for metric in definitions(config):
        if metric.name not in selected:
            continue
        cols = [metric.numerator] + ([metric.denominator] if metric.denominator else [])
        cb = bootstrap_values(c[cols], config.bootstrap_repeats, rng, metric.kind == "ratio")
        tb = bootstrap_values(t[cols], config.bootstrap_repeats, rng, metric.kind == "ratio")
        diff = tb-cb
        finite = diff[np.isfinite(diff)]
        lo, hi = np.quantile(finite, [config.alpha/2, 1-config.alpha/2]) if len(finite) else (np.nan, np.nan)
        rows.append({"metric": metric.name, "ci_low": lo, "ci_high": hi,
                     "repeats": config.bootstrap_repeats, "valid_repeats": len(finite), "seed": config.seed,
                     "method": "Independent user bootstrap, percentile CI; no bootstrap p-value"})
    return pd.DataFrame(rows)


def discrete_ks_permutation(control, test, repeats, rng) -> dict:
    c, t = np.asarray(control), np.asarray(test)
    pooled = np.concatenate([c, t])
    _, inverse, totals = np.unique(pooled, return_inverse=True, return_counts=True)

    def statistic(counts_c):
        counts_t = totals-counts_c
        return np.max(np.abs(counts_c.cumsum()/len(c)-counts_t.cumsum()/len(t)))

    observed = statistic(np.bincount(inverse[:len(c)], minlength=len(totals)))
    extreme = 0
    for _ in range(repeats):
        chosen = rng.permutation(len(pooled))[:len(c)]
        value = statistic(np.bincount(inverse[chosen], minlength=len(totals)))
        extreme += value >= observed-1e-12
    return {"statistic": observed, "p_value": (extreme+1)/(repeats+1), "repeats": repeats}


def distribution_checks(mart, config) -> pd.DataFrame:
    c, t = (mart[mart.group.eq(g)] for g in ["control", "test"])
    rng = np.random.default_rng(config.seed+1)
    rows = []
    for field in ["gmv", "margin"]:
        mw = stats.mannwhitneyu(t[field], c[field], alternative="two-sided", method="asymptotic")
        rows.append({"metric": field, "method": "Mann–Whitney with tie correction", "statistic": mw.statistic,
                     "p_value": mw.pvalue, "rank_biserial": 2*mw.statistic/(len(c)*len(t))-1, "family": "distribution"})
        ks = discrete_ks_permutation(c[field], t[field], config.permutation_repeats, rng)
        rows.append({"metric": field, "method": "KS statistic; user-label permutations (zeros/ties allowed)",
                     **ks, "family": "distribution"})
    return correct_family(pd.DataFrame(rows), config)


def srm_checks(populations: pd.DataFrame, config) -> pd.DataFrame:
    rows = []
    for name, part in populations.groupby("population"):
        for count_field in ["users", "active_users"]:
            counts = part.set_index("group")[count_field].reindex(["control", "test"], fill_value=0).to_numpy()
            expected = counts.sum()*np.array([1-config.expected_test_share, config.expected_test_share])
            stat, p = stats.chisquare(counts, expected) if counts.sum() else (np.nan, np.nan)
            rows.append({"population": name, "count_field": count_field, "control": counts[0], "test": counts[1],
                         "chi2": stat, "p_value": p, "srm_flag": bool(p < config.srm_alpha),
                         "note": "Активность — исход воздействия; эта строка не проверяет рандомизацию" if count_field == "active_users" else f"Ожидаемая доля test {config.expected_test_share:.1%} — допущение"})
    return pd.DataFrame(rows)
