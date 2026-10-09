"""Все допущения и изменяемые параметры анализа."""

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    data_dir: Path = Path(__file__).resolve().parents[1]
    output_dir: Path = Path(__file__).resolve().parents[1] / "reports"
    experiment_name: str = "ai_recommendations_homepage"
    start: str = "2026-07-01"
    end: str = "2026-07-30"  # Обе даты включаются в окно.
    alpha: float = 0.05
    power: float = 0.80
    expected_test_share: float = 0.50  # Допущение о дизайне, не результат проверки.
    srm_alpha: float = 0.0005  # Отдельный диагностический порог.
    seed: int = 2026
    bootstrap_repeats: int = 2000
    permutation_repeats: int = 1999
    followup_days: int = 30
    observation_end: str | None = None  # Дата полной выгрузки неизвестна.
    assume_complete_event_log: bool = True  # Для нулей у неактивных пользователей.
    minimum_gmv_lift: float | None = None  # Доля: 0.03 означает +3%.
    max_return_increase_pp: float | None = None
    max_support_increase_pp: float | None = None
    support_cost_rub: float | None = None
    ai_cost_per_impression_rub: float | None = None
    assume_full_refund: bool = False  # Без суммы возмещения net GMV условен.

    def validate(self) -> None:
        if not 0 < self.alpha < 1 or not 0 < self.expected_test_share < 1:
            raise ValueError("alpha и expected_test_share должны быть между 0 и 1")
        if not 0 < self.power < 1:
            raise ValueError("power должен быть между 0 и 1")
        if self.bootstrap_repeats < 100 or self.permutation_repeats < 99:
            raise ValueError("Слишком мало повторений для устойчивых интервалов")
        from datetime import date
        if date.fromisoformat(self.end) < date.fromisoformat(self.start):
            raise ValueError("Конец эксперимента раньше начала")
        if self.followup_days <= 0:
            raise ValueError("followup_days должен быть положительным")

    def serializable(self) -> dict:
        return {k: str(v) if isinstance(v, Path) else v for k, v in asdict(self).items()}
