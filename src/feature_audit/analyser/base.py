from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class AnalyzerResult:
    """Результат работы analyzer.

    Attributes:
        name: Системное имя analyzer.
        payload: Основные данные анализа.
        meta: Технические метаданные шага.
    """

    name: str
    payload: dict[str, Any]
    meta: dict[str, Any] = field(default_factory=dict)


class BaseAnalyzer:
    """Базовый интерфейс для analyzer-компонентов."""

    name = "base_analyzer"

    def analyze(self, df: pd.DataFrame) -> AnalyzerResult:
        raise NotImplementedError

    def apply(
        self,
        df: pd.DataFrame,
        result: AnalyzerResult,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Применить одно преобразование к DataFrame.

        По умолчанию analyzer ничего не меняет. Это позволяет держать правило:
        один analyzer = одно преобразование или один изолированный анализ.
        """
        return df.copy(), {}
