from __future__ import annotations

from typing import Any
from pathlib import Path

import pandas as pd

from .analyser.cardinality_analyser import HighCardinalityAnalyzer
from .analyser.categorical_target_correlation_analyser import (
    CategoricalTargetCorrelationAnalyzer,
)
from .analyser.date_analyser import DateNormalizationAnalyzer
from .analyser.date_candidates_analyser import DateCandidateAnalyzer
from .analyser.dominant_analyser import DominantValueAnalyzer
from .analyser.duplicates_analyser import DuplicateFeatureAnalyzer
from .analyser.empty_features_analyser import EmptyFeatureAnalyzer
from .analyser.manual_dropper_analyser import ManualDropAnalyzer
from .analyser.feature_analyser_5_structured import LeadFeatureEngineeringAnalyzer
from .analyser.numeric_feature_correlation_analyser import (
    NumericFeatureCorrelationAnalyzer,
)
from .analyser.numeric_target_correlation_analyser import NumericTargetCorrelationAnalyzer
from .analyser.order_analyser import DateOrderAnalyzer
from .logger import AuditLogger
from .utils import build_step_report, save_report


class FeatureCleaningPipeline:
    """Пайплайн очистки и аудита признаков.

    Ключевая идея: отчет пополняется сразу после каждого analyzer,
    а не только в конце обработки.
    """

    def __init__(
        self,
        target_column: str = "buyout_flag",
        manual_drop_map: dict[str, str] | None = None,
        drop_exact_duplicates: bool = True,
    ):
        self.target_column = target_column
        self.manual_drop_map = manual_drop_map or {
            "current_status_id": "удалить вручную",
            "lead_Сумма наложенного платежа (руб)": (
                "дубликат или нецелевой признак, удалить вручную"
            ),
        }
        # Удалить, утечка данных
        # received_dt: corr=1.0, n=17966
        # rejected_dt: corr=-0.8629, n=17966
        # returned_dt: corr=-0.8497, n=17966

        self.drop_exact_duplicates = drop_exact_duplicates
        self.report: dict[str, Any] = {}

    def _init_report(self, df: pd.DataFrame) -> None:
        self.report = {
            "target_column": self.target_column,
            "initial_shape": list(df.shape),
            "steps": [],
        }

    def _append_step(
        self,
        *,
        analyzer_name: str,
        action: str,
        before_shape: tuple[int, int],
        after_shape: tuple[int, int],
        result_payload: dict[str, Any],
        transform_payload: dict[str, Any] | None = None,
    ) -> None:
        self.report["steps"].append(
            build_step_report(
                analyzer_name=analyzer_name,
                action=action,
                before_shape=before_shape,
                after_shape=after_shape,
                result=result_payload,
                transform=transform_payload,
            )
        )

    def _finalize_report(self, df: pd.DataFrame) -> None:
        self.report["final_shape"] = list(df.shape)

    def _run_transform_step(
        self,
        *,
        work_df: pd.DataFrame,
        analyzer,
        title: str,
        action: str,
        log_fn,
        apply_transform: bool = True,
    ) -> pd.DataFrame:
        AuditLogger.step(title)
        before_shape = work_df.shape
        result = analyzer.analyze(work_df)
        log_fn(result)

        transform_payload = {}
        if apply_transform:
            work_df, transform_payload = analyzer.apply(work_df, result)

        after_shape = work_df.shape
        self._append_step(
            analyzer_name=analyzer.name,
            action=action,
            before_shape=before_shape,
            after_shape=after_shape,
            result_payload=result.payload,
            transform_payload=transform_payload,
        )
        return work_df

    def _run_analyze_step(
        self,
        *,
        work_df: pd.DataFrame,
        analyzer,
        title: str,
        action: str,
        log_fn,
    ):
        AuditLogger.step(title)
        before_shape = work_df.shape
        result = analyzer.analyze(work_df)
        log_fn(result)
        self._append_step(
            analyzer_name=analyzer.name,
            action=action,
            before_shape=before_shape,
            after_shape=work_df.shape,
            result_payload=result.payload,
            transform_payload={},
        )
        return result

    @staticmethod
    def _log_empty_features(result) -> None:
        AuditLogger.info(f"Почти пустых колонок: {len(result.payload['columns'])}")
        for item in result.payload["columns"]:
            AuditLogger.info(f"  - {item['column']}: {item['empty_share']:.2%} пустых")

    @staticmethod
    def _log_high_cardinality(result) -> None:
        AuditLogger.info(f"High-cardinality колонок: {len(result.payload['columns'])}")
        for item in result.payload["columns"]:
            AuditLogger.info(
                f"  - {item['column']}: nunique={item['nunique']}, "
                f"non_empty={item['non_empty_count']}, "
                f"unique_share={item['unique_share']:.2%}"
            )

    @staticmethod
    def _log_dominant_values(result) -> None:
        AuditLogger.info(f"Найдено почти константных колонок: {len(result.payload['columns'])}")
        for item in result.payload["columns"]:
            AuditLogger.info(
                f"  - {item['column']}: top_value={repr(item['top_value'])}, "
                f"top_count={item['top_count']}, non_empty={item['non_empty_count']}, "
                f"top_share={item['top_share']:.2%}, nunique={item['nunique']}"
            )

    @staticmethod
    def _log_duplicates(result) -> None:
        AuditLogger.info(f"Совпадающих полей: {len(result.payload['exact_matches'])}")
        for item in result.payload["exact_matches"]:
            AuditLogger.info(f"  - {item['left']} == {item['right']}: {item['similarity']:.2%}")

        AuditLogger.info(f"Почти одинаковых полей: {len(result.payload['near_matches'])}")
        for item in result.payload["near_matches"]:
            AuditLogger.info(f"  - {item['left']} ~ {item['right']}: {item['similarity']:.2%}")

    @staticmethod
    def _log_manual_drop(result) -> None:
        AuditLogger.info(f"Колонок для ручного удаления: {len(result.payload['columns'])}")
        for item in result.payload["columns"]:
            AuditLogger.info(f"  - {item['column']}: reason={item['reason']}")

    @staticmethod
    def _log_date_candidates(result) -> None:
        AuditLogger.info(f"Колонки с суффиксом _ts: {len(result.payload['ts_suffix_columns'])}")
        for col in result.payload["ts_suffix_columns"]:
            AuditLogger.info(f"  - {col}")

        AuditLogger.info(
            f"Кандидатов на дату найдено: {len(result.payload['detected_date_columns'])}"
        )
        for item in result.payload["detected_date_columns"]:
            suffix_mark = " [ts]" if item["has_ts_suffix"] else ""
            unit_mark = f", unit={item['best_unit']}" if item.get("best_unit") else ""
            AuditLogger.info(
                f"  - {item['column']}: {item['kind']}, "
                f"parsed={item['parse_success_share']:.2%}{suffix_mark}{unit_mark}"
            )

    @staticmethod
    def _log_date_normalization(result) -> None:
        AuditLogger.info(
            f"Колонок для преобразования в дату: {len(result.payload['columns'])}"
        )
        for item in result.payload["columns"]:
            unit_mark = f", unit={item['best_unit']}" if item.get("best_unit") else ""
            AuditLogger.info(f"  - {item['column']} -> {item['new_column']}{unit_mark}")

    @staticmethod
    def _log_date_order(result) -> None:
        ordered = result.payload["ordered_columns"]
        AuditLogger.info(f"Колонок с датами для анализа порядка: {len(ordered)}")
        AuditLogger.info("Предполагаемый хронологический порядок:")
        for i, col in enumerate(ordered, start=1):
            AuditLogger.info(f"  {i}. {col}")

    @staticmethod
    def _log_numeric_target_corr(result) -> None:
        if result.payload.get("error") == "target_not_found":
            AuditLogger.info("Target buyout_flag не найден. Анализ numeric -> target пропущен.")
            return
        AuditLogger.info(
            f"Числовых полей с заметной корреляцией с target: {len(result.payload['columns'])}"
        )
        for item in result.payload["columns"][:30]:
            AuditLogger.info(
                f"  - {item['column']}: corr={item['correlation']}, n={item['pair_count']}"
            )

    @staticmethod
    def _log_categorical_target_corr(result) -> None:
        if result.payload.get("error") == "target_not_found":
            AuditLogger.info("Target buyout_flag не найден. Анализ categorical -> target пропущен.")
            return
        AuditLogger.info(
            f"Категориальных полей с заметной связью с target: {len(result.payload['columns'])}"
        )
        for item in result.payload["columns"][:30]:
            AuditLogger.info(
                f"  - {item['column']}: Cramer's V={item['cramers_v']}, "
                f"n={item['pair_count']}, nunique={item['nunique']}"
            )

    @staticmethod
    def _log_numeric_feature_corr(result) -> None:
        AuditLogger.info(
            f"Пар числовых полей с высокой корреляцией: {len(result.payload['pairs'])}"
        )
        for item in result.payload["pairs"][:50]:
            AuditLogger.info(
                f"  - {item['left']} ~ {item['right']}: "
                f"corr={item['correlation']}, n={item['pair_count']}"
            )

    def run(
        self,
        df: pd.DataFrame,
        report_path: (str | Path) | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        work_df = df.copy()
        self._init_report(work_df)

        work_df = self._run_transform_step(
            work_df=work_df,
            analyzer=EmptyFeatureAnalyzer(),
            title="ПОИСК И УДАЛЕНИЕ ПОЧТИ ПУСТЫХ ПОЛЕЙ",
            action="drop_empty_features",
            log_fn=self._log_empty_features,
        )

        work_df = self._run_transform_step(
            work_df=work_df,
            analyzer=HighCardinalityAnalyzer(),
            title="ПОИСК И УДАЛЕНИЕ HIGH-CARDINALITY ПОЛЕЙ",
            action="drop_high_cardinality_features",
            log_fn=self._log_high_cardinality,
        )

        work_df = self._run_transform_step(
            work_df=work_df,
            analyzer=DominantValueAnalyzer(),
            title="ПОИСК И УДАЛЕНИЕ ПОЧТИ КОНСТАНТНЫХ ПОЛЕЙ",
            action="drop_dominant_features",
            log_fn=self._log_dominant_values,
        )

        duplicate_analyzer = DuplicateFeatureAnalyzer()
        if not self.drop_exact_duplicates:
            duplicate_analyzer.apply = lambda df_, result_: (df_.copy(), {"dropped_columns": []})
        work_df = self._run_transform_step(
            work_df=work_df,
            analyzer=duplicate_analyzer,
            title="ПОИСК ДУБЛИКАТОВ И ПОЧТИ СОВПАДАЮЩИХ ПОЛЕЙ",
            action="drop_exact_duplicate_features",
            log_fn=self._log_duplicates,
        )

        work_df = self._run_transform_step(
            work_df=work_df,
            analyzer=ManualDropAnalyzer(drop_map=self.manual_drop_map),
            title="РУЧНОЕ УДАЛЕНИЕ ЗАДАННЫХ ПОЛЕЙ",
            action="drop_manual_features",
            log_fn=self._log_manual_drop,
        )

        date_candidate_result = self._run_analyze_step(
            work_df=work_df,
            analyzer=DateCandidateAnalyzer(),
            title="АНАЛИЗ КАНДИДАТОВ НА ДАТУ",
            action="detect_date_candidates",
            log_fn=self._log_date_candidates,
        )

        date_specs = date_candidate_result.payload.get("detected_date_columns", [])
        date_normalization_analyzer = DateNormalizationAnalyzer(candidate_specs=date_specs)
        work_df = self._run_transform_step(
            work_df=work_df,
            analyzer=date_normalization_analyzer,
            title="НОРМАЛИЗАЦИЯ ДАТ",
            action="normalize_dates",
            log_fn=self._log_date_normalization,
        )

        normalized_date_columns = [
            item["new_column"]
            for item in date_specs
            if item.get("new_column")
        ]
        self._run_analyze_step(
            work_df=work_df,
            analyzer=DateOrderAnalyzer(candidate_columns=normalized_date_columns),
            title="АНАЛИЗ ХРОНОЛОГИЧЕСКОГО ПОРЯДКА ДАТ",
            action="analyze_date_order",
            log_fn=self._log_date_order,
        )

        self._run_analyze_step(
            work_df=work_df,
            analyzer=NumericTargetCorrelationAnalyzer(target_column=self.target_column),
            title="КОРРЕЛЯЦИЯ ЧИСЛОВЫХ ПРИЗНАКОВ С TARGET",
            action="analyze_numeric_target_correlation",
            log_fn=self._log_numeric_target_corr,
        )

        self._run_analyze_step(
            work_df=work_df,
            analyzer=CategoricalTargetCorrelationAnalyzer(target_column=self.target_column),
            title="СВЯЗЬ КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ С TARGET",
            action="analyze_categorical_target_correlation",
            log_fn=self._log_categorical_target_corr,
        )

        self._run_analyze_step(
            work_df=work_df,
            analyzer=NumericFeatureCorrelationAnalyzer(),
            title="КОРРЕЛЯЦИЯ ЧИСЛОВЫХ ПРИЗНАКОВ МЕЖДУ СОБОЙ",
            action="analyze_numeric_feature_correlation",
            log_fn=self._log_numeric_feature_corr,
        )

        work_df = self._run_transform_step(
            work_df=work_df,
            analyzer=LeadFeatureEngineeringAnalyzer(),
            title="LEAD FEATURE ENGINEERING",
            action="lead_feature_analyzer",
            log_fn=lambda result: None,
        )

        self._finalize_report(work_df)
        if report_path is not None:
            save_report(self.report, report_path)

        return work_df, self.report
