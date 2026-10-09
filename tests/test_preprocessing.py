"""Проверки отбора пользователей, поздних возвратов и связанных событий."""

import unittest
from dataclasses import replace

import pandas as pd

from src.config import Config
from src.preprocessing import prepare, user_mart


def fixture():
    users = pd.DataFrame({"user_id": ["u1", "u2", "u3", "u4", "u5"],
                          "registration_date": pd.to_datetime(["2026-01-01"]*5),
                          "is_bot_candidate": [0, 0, 0, 0, 1], "is_premium": [0]*5})
    assignments = pd.DataFrame({"user_id": ["u1", "u2", "u3", "u4", "u5", "u4"],
        "experiment_name": ["ai_recommendations_homepage"]*6,
        "experiment_group": ["control", "control", "test", "control", "test", "test"],
        "assignment_date": pd.to_datetime(["2026-07-01"]*5 + ["2026-07-04"])})
    sessions = pd.DataFrame({"session_id": ["s1", "s3"], "user_id": ["u1", "u3"],
        "session_date": pd.to_datetime(["2026-07-01", "2026-07-05"]),
        "experiment_group": ["control", "test"], "session_duration_sec": [50, 60]})
    impressions = pd.DataFrame({"impression_id": ["i1", "i3"], "session_id": ["s1", "s3"], "user_id": ["u1", "u3"],
        "event_time": pd.to_datetime(["2026-07-01", "2026-07-05"]), "product_id": ["p1", "p1"],
        "position": [1, 1], "model_version": ["baseline_v1", "ai_v2"], "is_duplicate_event": [1, 0]})
    clicks = pd.DataFrame({"click_id": ["c1", "c3"], "impression_id": ["i1", "i3"], "user_id": ["u1", "u3"],
        "event_time": pd.to_datetime(["2026-07-01", "2026-07-05"])})
    orders = pd.DataFrame({"order_id": ["o1", "o3"], "user_id": ["u1", "u3"], "click_id": ["c1", "c3"],
        "order_date": pd.to_datetime(["2026-07-01", "2026-07-05"]), "gmv_rub": [100., 200.],
        "gross_margin_rub": [-5., 40.], "is_returned": [1, 0], "return_date": pd.to_datetime(["2026-07-31", None])})
    tickets = pd.DataFrame({"ticket_id": ["t1"], "order_id": ["o1"], "user_id": ["u1"], "ticket_date": pd.to_datetime(["2026-08-01"])})
    subs = pd.DataFrame({"user_id": pd.Series(dtype=str), "premium_start_date": pd.Series(dtype="datetime64[ns]")})
    return {"users": users, "experiment_assignments": assignments, "sessions": sessions, "impressions": impressions,
            "clicks": clicks, "orders": orders, "support_tickets": tickets, "premium_subscriptions": subs}


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.tables = fixture()
        self.config = Config()
        self.prepared = prepare(self.tables, self.config)
        self.mart = user_mart(self.prepared["cohorts"]["main"], self.prepared["events"], self.config).set_index("user_id")

    def test_inactive_assigned_user_is_kept(self):
        self.assertIn("u2", self.mart.index)
        self.assertEqual(self.mart.loc["u2", "gmv"], 0)
        self.assertEqual(len(self.mart), 3)

    def test_conflicts_and_bots_have_separate_sensitivity_samples(self):
        self.assertNotIn("u4", self.mart.index)
        self.assertNotIn("u5", self.mart.index)
        self.assertIn("u5", set(self.prepared["cohorts"]["include_bots"].user_id))
        initial = self.prepared["cohorts"]["initial_assignment_itt"].set_index("user_id")
        self.assertEqual(initial.loc["u4", "group"], "control")

    def test_flagged_impression_does_not_delete_real_order(self):
        self.assertEqual(self.mart.loc["u1", "orders"], 1)
        self.assertEqual(self.mart.loc["u1", "gmv"], 100)
        self.assertEqual(self.mart.loc["u1", "clicks_unflagged"], 0)

    def test_late_events_are_separated_and_margin_not_subtracted_twice(self):
        self.assertEqual(self.mart.loc["u1", "returns_window"], 0)
        self.assertEqual(self.mart.loc["u1", "returns_observed"], 1)
        self.assertEqual(self.mart.loc["u1", "tickets_window"], 0)
        self.assertEqual(self.mart.loc["u1", "tickets_observed"], 1)
        self.assertEqual(self.mart.loc["u1", "margin"], -5)
        self.assertNotIn("mature_orders", self.mart.columns)

    def test_followup_requires_confirmed_snapshot(self):
        config = replace(self.config, observation_end="2026-08-04")
        prepared = prepare(self.tables, config)
        mart = user_mart(prepared["cohorts"]["main"], prepared["events"], config).set_index("user_id")
        self.assertEqual(mart.loc["u1", "mature_orders"], 1)
        self.assertEqual(mart.loc["u1", "returns_fixed_followup"], 1)

    def test_snapshot_before_recorded_events_is_rejected(self):
        config = replace(self.config, observation_end="2026-07-30")
        with self.assertRaisesRegex(ValueError, "раньше записанных"):
            prepare(self.tables, config)


if __name__ == "__main__":
    unittest.main()
