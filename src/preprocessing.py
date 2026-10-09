"""Аудит очистки, выборки пользователей и витрина без размножения событий."""

import numpy as np
import pandas as pd


def prepare(tables: dict, config) -> dict:
    if not config.assume_complete_event_log:
        raise ValueError("Нули требуют полного журнала событий. Сначала уточните полноту выгрузки.")
    start, end = pd.Timestamp(config.start), pd.Timestamp(config.end)
    a = tables["experiment_assignments"].copy()
    a = a[a.experiment_name.eq(config.experiment_name)]
    if a.empty:
        raise ValueError("Назначения указанного эксперимента отсутствуют")
    first_date = a.groupby("user_id").assignment_date.transform("min")
    earliest = a[a.assignment_date.eq(first_date)]
    ambiguous = earliest.groupby("user_id").experiment_group.nunique().gt(1)
    # Берём первое назначение только при единственной группе в первую дату.
    initial = earliest[~earliest.user_id.isin(ambiguous[ambiguous].index)].drop_duplicates("user_id")
    initial = initial.rename(columns={"experiment_group": "group", "assignment_date": "first_assignment"})
    base = tables["users"].merge(initial[["user_id", "group", "first_assignment"]], on="user_id", how="inner", validate="one_to_one")
    conflicts = a.groupby("user_id").experiment_group.nunique().gt(1)
    base["assignment_conflict"] = base.user_id.map(conflicts).fillna(False)
    base["tenure_days"] = (start - base.registration_date).dt.days
    base["tenure"] = pd.cut(base.tenure_days, [-1, 30, 90, 365, np.inf], labels=["0–30", "31–90", "91–365", ">365"]).astype(str)

    sessions = tables["sessions"]
    first_session = sessions.groupby("user_id").session_date.min()
    base["first_session"] = base.user_id.map(first_session)
    base["eligible_initial"] = base.first_assignment.le(start) & (base.first_session.isna() | base.first_assignment.le(base.first_session))
    # Для чистой группы фактический вариант обязан совпадать с назначением.
    exposure = sessions.merge(base[["user_id", "group", "assignment_conflict"]], on="user_id", validate="many_to_one")
    mismatch = exposure[~exposure.assignment_conflict & exposure.experiment_group.ne(exposure.group)]
    if not mismatch.empty:
        raise ValueError("Сессии чистых пользователей противоречат назначениям")
    eligible = ~base.assignment_conflict & base.eligible_initial
    cohorts = {
        "main": base[eligible & base.is_bot_candidate.eq(0)].copy(),
        "include_bots": base[eligible].copy(),
        "initial_assignment_itt": base[base.eligible_initial & base.is_bot_candidate.eq(0)].copy(),
    }
    audit_rows = []
    for rule, mask in {
        "initial_unique_assignment": pd.Series(True, index=base.index),
        "exclude_conflicts": ~base.assignment_conflict,
        "exclude_conflicts_and_bots": eligible & base.is_bot_candidate.eq(0),
        "initial_assignment_sensitivity": base.eligible_initial & base.is_bot_candidate.eq(0),
    }.items():
        for group in ["control", "test"]:
            group_mask = base.group.eq(group)
            ids = set(base.loc[group_mask & ~mask, "user_id"])
            audit_rows.append({"rule": rule, "group": group, "users_kept": int((group_mask & mask).sum()),
                               "users_excluded": len(ids),
                               "sessions_excluded": int(sessions.user_id.isin(ids).sum()),
                               "impressions_excluded": int(tables["impressions"].user_id.isin(ids).sum()),
                               "orders_excluded": int(tables["orders"].user_id.isin(ids).sum())})
    counts = []
    for name, cohort in {"initial_unique_assignment": base, **cohorts}.items():
        for group in ["control", "test"]:
            ids = cohort.loc[cohort.group.eq(group), "user_id"]
            counts.append({"population": name, "group": group, "users": len(ids),
                           "active_users": int(ids.isin(sessions.user_id).sum())})
    membership = a.groupby("experiment_group").agg(rows=("user_id", "size"), unique_users=("user_id", "nunique")).reset_index()

    # В этом датасете нет восстановимого оригинала для помеченных дублей.
    # Сохраняем уникальные события; альтернативный CTR исключает флаг и его клики.
    impressions = tables["impressions"].copy()
    clicks = tables["clicks"].copy()
    flagged_ids = set(impressions.loc[impressions.is_duplicate_event.eq(1), "impression_id"])
    clicks["flagged_impression"] = clicks.impression_id.isin(flagged_ids)
    clicked_flagged = set(clicks.loc[clicks.flagged_impression, "click_id"])
    duplicate_impact = pd.DataFrame([{
        "flagged_impressions": len(flagged_ids), "linked_clicks": int(clicks.flagged_impression.sum()),
        "linked_orders": int(tables["orders"].click_id.isin(clicked_flagged).sum()),
        "natural_key_duplicates": int(impressions.duplicated(["session_id", "user_id", "event_time", "product_id", "position", "model_version"]).sum()),
        "policy": "keep_unique_events; flag-excluded CTR is sensitivity only",
    }])
    events = {
        "sessions": sessions[sessions.session_date.between(start, end)].copy(),
        "impressions": impressions[impressions.event_time.dt.normalize().between(start, end)].copy(),
        "clicks": clicks[clicks.event_time.dt.normalize().between(start, end)].copy(),
        "orders": tables["orders"][tables["orders"].order_date.between(start, end)].copy(),
    }
    orders = events["orders"]
    observed_end = pd.Timestamp(config.observation_end) if config.observation_end else None
    orders["return_in_window"] = orders.is_returned.eq(1) & orders.return_date.le(end)
    orders["return_observed"] = orders.is_returned.eq(1)
    if observed_end is not None:
        if observed_end < end:
            raise ValueError("observation_end не может предшествовать концу эксперимента")
        latest_recorded = pd.concat([tables["orders"].return_date, tables["support_tickets"].ticket_date]).max()
        if pd.notna(latest_recorded) and observed_end < latest_recorded:
            raise ValueError("observation_end раньше записанных последствий. Историческую маржу из итогового поля восстановить нельзя.")
        orders["return_observed"] &= orders.return_date.le(observed_end)
        # Явная единица D избегает устаревающего generic-timedelta в pandas 2.x.
        maturity_date = pd.Series(orders.order_date.to_numpy(dtype="datetime64[ns]") + np.timedelta64(config.followup_days, "D"), index=orders.index)
        orders["mature_order"] = maturity_date.le(observed_end)
        orders["return_fixed_followup"] = orders.mature_order & orders.is_returned.eq(1) & orders.return_date.le(maturity_date)
    orders["return_amount_observed"] = orders.gmv_rub.where(orders.return_observed, 0)
    orders["return_amount_window"] = orders.gmv_rub.where(orders.return_in_window, 0)
    tickets = tables["support_tickets"]
    tickets = tickets[tickets.order_id.isin(orders.order_id) & tickets.ticket_date.ge(start)].copy()
    if observed_end is not None:
        tickets = tickets[tickets.ticket_date.le(observed_end)].copy()
    tickets["ticket_in_window"] = tickets.ticket_date.le(end)
    events["tickets"] = tickets
    subs = tables["premium_subscriptions"]
    events["new_premium"] = subs[subs.premium_start_date.between(start, end)].copy()
    return {"base": base, "cohorts": cohorts, "events": events,
            "cleaning_log": pd.DataFrame(audit_rows), "population_counts": pd.DataFrame(counts),
            "assignment_membership": membership, "duplicate_impact": duplicate_impact,
            "ambiguous_first_assignments": int(ambiguous.sum())}


def user_mart(cohort: pd.DataFrame, events: dict, config) -> pd.DataFrame:
    result = cohort.copy()
    added = []

    def attach(agg):
        nonlocal result
        added.extend(agg.columns)
        result = result.merge(agg, left_on="user_id", right_index=True, how="left", validate="one_to_one")

    s, i, c, o, t = (events[x] for x in ["sessions", "impressions", "clicks", "orders", "tickets"])
    attach(s.groupby("user_id").agg(sessions=("session_id", "size"), duration_total=("session_duration_sec", "sum")))
    i = i.assign(unflagged=i.is_duplicate_event.eq(0))
    attach(i.groupby("user_id").agg(impressions=("impression_id", "size"), impressions_unflagged=("unflagged", "sum")))
    c = c.assign(unflagged=~c.flagged_impression)
    attach(c.groupby("user_id").agg(clicks=("click_id", "size"), clicks_unflagged=("unflagged", "sum")))
    attach(o.groupby("user_id").agg(orders=("order_id", "size"), gmv=("gmv_rub", "sum"), margin=("gross_margin_rub", "sum"),
          returns_observed=("return_observed", "sum"), returns_window=("return_in_window", "sum"),
          returned_gmv_observed=("return_amount_observed", "sum"), returned_gmv_window=("return_amount_window", "sum")))
    if "mature_order" in o:
        attach(o.groupby("user_id").agg(mature_orders=("mature_order", "sum"), returns_fixed_followup=("return_fixed_followup", "sum")))
    attach(t.groupby("user_id").agg(tickets_observed=("ticket_id", "size"), tickets_window=("ticket_in_window", "sum")))
    # Подписки считаем как пользовательский исход, не как выручку без данных о платежах.
    attach(events["new_premium"].groupby("user_id").size().to_frame("new_premium_records"))
    result[added] = result[added].fillna(0)
    for name, source in [("buyer", "orders"), ("reached", "impressions"), ("clicker", "clicks"),
                         ("return_user_window", "returns_window"), ("return_user_observed", "returns_observed"),
                         ("support_user_window", "tickets_window"), ("support_user_observed", "tickets_observed"),
                         ("new_premium_user", "new_premium_records")]:
        result[name] = result[source].gt(0).astype(int)
    if config.assume_full_refund:
        result["net_gmv_full_refund"] = result.gmv - result.returned_gmv_observed
    # Контроль сумм исключает ошибки join и потери покупателей.
    selected_orders = o[o.user_id.isin(result.user_id)]
    if not np.isclose(result.gmv.sum(), selected_orders.gmv_rub.sum()):
        raise AssertionError("Сумма GMV витрины не совпадает с заказами")
    if result.user_id.duplicated().any():
        raise AssertionError("Витрина должна содержать одну строку на пользователя")
    return result
