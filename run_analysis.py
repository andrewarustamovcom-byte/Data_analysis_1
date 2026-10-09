"""Точка входа: python run_analysis.py. Коммитов и отправки в Git нет."""

import argparse
from dataclasses import replace
from pathlib import Path

from src.config import Config
from src.pipeline import run_analysis


def main():
    parser = argparse.ArgumentParser(description="Анализ эксперимента AI-рекомендаций")
    parser.add_argument("--data-dir", type=Path, help="Каталог исходных CSV")
    parser.add_argument("--output-dir", type=Path, help="Каталог результатов")
    parser.add_argument("--bootstrap-repeats", type=int)
    parser.add_argument("--permutation-repeats", type=int)
    parser.add_argument("--observation-end", help="Подтверждённая дата полной выгрузки YYYY-MM-DD")
    args = parser.parse_args()
    changes = {k: v for k, v in vars(args).items() if v is not None}
    config = replace(Config(), **changes)
    run_analysis(config)


if __name__ == "__main__":
    main()
