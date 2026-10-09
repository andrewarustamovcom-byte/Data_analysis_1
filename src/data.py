"""Загрузка CSV и контроль обязательных полей; исходные файлы не изменяются."""

from pathlib import Path

import pandas as pd


KEYS = {
    "users": "user_id", "experiment_assignments": "assignment_id",
    "sessions": "session_id", "impressions": "impression_id",
    "clicks": "click_id", "orders": "order_id", "products": "product_id",
    "sellers": "seller_id", "support_tickets": "ticket_id",
}
DATES = {
    "users": ["registration_date"],
    "experiment_assignments": ["assignment_date"],
    "sessions": ["session_date"], "impressions": ["event_time"],
    "clicks": ["event_time"], "orders": ["order_date", "return_date"],
    "products": ["created_date"], "sellers": ["joined_date"],
    "support_tickets": ["ticket_date"], "premium_subscriptions": ["premium_start_date"],
}
REQUIRED = {
    "users": ["user_id", "registration_date", "device", "region", "is_premium", "is_bot_candidate"],
    "experiment_assignments": ["assignment_id", "user_id", "experiment_name", "experiment_group", "assignment_date"],
    "sessions": ["session_id", "user_id", "session_date", "experiment_group", "device", "region", "session_duration_sec"],
    "impressions": ["impression_id", "session_id", "user_id", "event_time", "product_id", "position", "model_version", "is_clicked", "is_duplicate_event"],
    "clicks": ["click_id", "impression_id", "session_id", "user_id", "event_time", "product_id"],
    "orders": ["order_id", "user_id", "session_id", "click_id", "order_date", "product_id", "category", "gmv_rub", "gross_margin_rub", "delivery_fee_rub", "is_returned", "return_date"],
    "products": ["product_id", "seller_id", "category"],
    "sellers": ["seller_id"],
    "support_tickets": ["ticket_id", "user_id", "order_id", "ticket_date", "reason", "sentiment_score", "resolved_within_24h"],
    "premium_subscriptions": ["user_id", "premium_start_date", "premium_status", "is_trial"],
    "data_dictionary": ["table", "field", "type", "description"],
}
EXPECTED_ROWS = {
    "users": 50000, "products": 4500, "sellers": 700, "sessions": 164739,
    "impressions": 1066434, "clicks": 91266, "orders": 7407,
    "support_tickets": 337, "premium_subscriptions": 9127,
}


def load_tables(data_dir: Path) -> dict[str, pd.DataFrame]:
    tables = {}
    for name, required in REQUIRED.items():
        path = data_dir / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Нет таблицы: {path}")
        df = pd.read_csv(path)
        missing = set(required) - set(df.columns)
        if missing:
            raise ValueError(f"{name}: отсутствуют поля {sorted(missing)}")
        for column in DATES.get(name, []):
            original = df[column]
            df[column] = pd.to_datetime(original, errors="coerce")
            # Не маскируем неверные строки дат под обычные пропуски.
            df.attrs[f"invalid_{column}"] = int((original.notna() & df[column].isna()).sum())
        tables[name] = df
    return tables
