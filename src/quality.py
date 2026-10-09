"""Аудит исходных данных до расчёта эффектов."""

import pandas as pd

from .data import DATES, EXPECTED_ROWS, KEYS


def audit(tables: dict, config) -> dict[str, pd.DataFrame]:
    profiles, missing, dates, checks = [], [], [], []

    def check(name, count, severity="error", note=""):
        checks.append({"check": name, "issues": int(count), "severity": severity, "note": note})

    for name, df in tables.items():
        profiles.append({"table": name, "rows": len(df), "columns": len(df.columns),
                         "expected_rows": EXPECTED_ROWS.get(name),
                         "exact_duplicate_rows": int(df.duplicated().sum())})
        for col in df:
            missing.append({"table": name, "field": col, "dtype": str(df[col].dtype),
                            "missing": int(df[col].isna().sum()), "missing_share": df[col].isna().mean()})
        if name in KEYS:
            key = KEYS[name]
            check(f"{name}: empty {key}", df[key].isna().sum())
            check(f"{name}: duplicate {key}", df[key].duplicated().sum())
        for col in DATES.get(name, []):
            dates.append({"table": name, "field": col, "min": df[col].min(), "max": df[col].max()})
            check(f"{name}: invalid {col}", df.attrs.get(f"invalid_{col}", 0))
            if col != "return_date":
                check(f"{name}: empty {col}", df[col].isna().sum())

    # many_to_one защищает от незаметного размножения строк при join.
    def relation(child, parent, fk, parent_key, pairs=()):
        c, p = tables[child], tables[parent]
        if p[parent_key].duplicated().any():
            return
        selected = list(dict.fromkeys([parent_key] + [pc for _, pc in pairs]))
        ref = p[selected].rename(columns={col: f"ref_{col}" for col in selected})
        joined = c.merge(ref, left_on=fk, right_on=f"ref_{parent_key}", how="left", validate="many_to_one")
        check(f"{child} → {parent}: missing reference", joined[f"ref_{parent_key}"].isna().sum())
        for cc, pc in pairs:
            check(f"{child} → {parent}: inconsistent {cc}",
                  (joined[f"ref_{parent_key}"].notna() & joined[cc].ne(joined[f"ref_{pc}"])).sum())
        return joined

    for name in ["experiment_assignments", "sessions", "impressions", "clicks", "orders", "support_tickets", "premium_subscriptions"]:
        relation(name, "users", "user_id", "user_id")
    relation("impressions", "sessions", "session_id", "session_id", [("user_id", "user_id")])
    relation("clicks", "sessions", "session_id", "session_id", [("user_id", "user_id")])
    ci = relation("clicks", "impressions", "impression_id", "impression_id",
                  [("user_id", "user_id"), ("session_id", "session_id"), ("product_id", "product_id")])
    # Даты заказа точнее дня не известны: сравниваем дни, а не полуночь с часом клика.
    oc = relation("orders", "clicks", "click_id", "click_id",
                  [("user_id", "user_id"), ("session_id", "session_id"), ("product_id", "product_id")])
    relation("orders", "sessions", "session_id", "session_id", [("user_id", "user_id")])
    relation("orders", "products", "product_id", "product_id", [("category", "category")])
    relation("impressions", "products", "product_id", "product_id")
    relation("clicks", "products", "product_id", "product_id")
    relation("products", "sellers", "seller_id", "seller_id")
    so = relation("support_tickets", "orders", "order_id", "order_id", [("user_id", "user_id")])
    if ci is not None:
        times = tables["impressions"][["impression_id", "event_time", "is_clicked"]]
        ci = tables["clicks"].merge(times, on="impression_id", suffixes=("", "_impression"), validate="many_to_one")
        check("click before impression", (ci.event_time < ci.event_time_impression).sum())
        check("click from unclicked impression", ci.is_clicked.ne(1).sum())
        check("multiple clicks for same impression", ci.impression_id.duplicated().sum())
    if oc is not None:
        ot = tables["orders"].merge(tables["clicks"][["click_id", "event_time"]], on="click_id", validate="many_to_one")
        check("order day before click day", (ot.order_date < ot.event_time.dt.normalize()).sum())
    if so is not None:
        st = tables["support_tickets"].merge(tables["orders"][["order_id", "order_date"]], on="order_id", validate="many_to_one")
        check("ticket before order", (st.ticket_date < st.order_date).sum())

    i, s, o, u = (tables[x] for x in ["impressions", "sessions", "orders", "users"])
    clicked_ids = set(tables["clicks"].impression_id)
    check("is_clicked differs from click log", i.is_clicked.ne(i.impression_id.isin(clicked_ids).astype(int)).sum())
    if not s.session_id.duplicated().any():
        joined = i.merge(s[["session_id", "session_date", "experiment_group"]], on="session_id", validate="many_to_one")
        expected = joined.experiment_group.map({"control": "baseline_v1", "test": "ai_v2"})
        check("model_version differs from session group", joined.model_version.ne(expected).sum())
        check("impression day differs from session day", joined.event_time.dt.normalize().ne(joined.session_date).sum(), "warning")
    check("return without return_date", (o.is_returned.eq(1) & o.return_date.isna()).sum())
    check("return_date without return flag", (o.is_returned.eq(0) & o.return_date.notna()).sum())
    check("return before order", (o.return_date < o.order_date).sum())
    check("negative GMV", o.gmv_rub.lt(0).sum())
    for name, cols in {"users": ["is_premium", "is_bot_candidate"], "impressions": ["is_clicked", "is_duplicate_event"], "orders": ["is_returned"]}.items():
        for col in cols:
            check(f"{name}: invalid binary {col}", (~tables[name][col].isin([0, 1])).sum())
    for name, cols in {"orders": ["gmv_rub", "gross_margin_rub"], "sessions": ["session_duration_sec"]}.items():
        for col in cols:
            check(f"{name}: missing {col}", tables[name][col].isna().sum())
    a = tables["experiment_assignments"]
    a = a[a.experiment_name.eq(config.experiment_name)]
    check("assignment: unknown group", (~a.experiment_group.isin(["control", "test"])).sum())
    check("session: unknown group", (~s.experiment_group.isin(["control", "test"])).sum())
    check("users without experiment assignment", (~u.user_id.isin(a.user_id)).sum(), "warning")
    check("registration after experiment start", u.registration_date.gt(pd.Timestamp(config.start)).sum())
    check("missing delivery fee", o.delivery_fee_rub.isna().sum(), "warning")
    check("flagged duplicate impressions", i.is_duplicate_event.eq(1).sum(), "warning", "Флаг не доказывает наличие восстановимого оригинала")
    natural = ["session_id", "user_id", "event_time", "product_id", "position", "model_version"]
    check("duplicate natural impression key", i.duplicated(natural).sum(), "warning", "Совпадение ключа требует проверки смысла события")
    check("returns after experiment", o.return_date.gt(pd.Timestamp(config.end)).sum(), "warning")
    check("tickets after experiment", tables["support_tickets"].ticket_date.gt(pd.Timestamp(config.end)).sum(), "warning")
    check("nonnegative margin on returned order", (o.is_returned.eq(1) & o.gross_margin_rub.ge(0)).sum(), "warning")
    return {"profile": pd.DataFrame(profiles), "missing_values": pd.DataFrame(missing),
            "date_bounds": pd.DataFrame(dates), "quality_checks": pd.DataFrame(checks)}


def require_valid(audit_tables: dict) -> None:
    checks = audit_tables["quality_checks"]
    failed = checks[checks.severity.eq("error") & checks.issues.gt(0)]
    if not failed.empty:
        raise ValueError("Проверки данных не пройдены; см. quality_checks.csv:\n" + failed.to_string(index=False))
