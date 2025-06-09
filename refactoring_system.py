# =============================================================================
# Главная система рефакторинга
# =============================================================================
import time
from typing import List, Dict, Any

from loguru import logger

from base import AnalysisResult, RefactoringResult
from code_analyzer import CodeAnalyzer
from code_validator import CodeGenerator
from llm_manager import LLMConfig, LLMInterface


class RefactoringSystem:
    """Главная система рефакторинга"""

    def __init__(self, llm_config: LLMConfig = None):
        if llm_config is None:
            llm_config = LLMConfig()

        self.analyzer = CodeAnalyzer()
        self.llm_interface = LLMInterface(llm_config)
        self.generator = CodeGenerator(self.llm_interface)
        self.logger = logger

    def analyze_file(self, file_path: str) -> AnalysisResult:
        """Анализ файла"""
        return self.analyzer.analyze_file(file_path)

    def refactor_file(self, file_path: str, max_issues: int = 3) -> List[RefactoringResult]:
        """Рефакторинг файла"""
        self.logger.info(f"Начало рефакторинга файла: {file_path}")

        # Анализ файла
        analysis = self.analyze_file(file_path)

        if not analysis.code_smells:
            self.logger.info("Проблемы не обнаружены")
            return []

        # Чтение исходного кода
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()

        results = []

        # Рефакторинг наиболее критичных проблем
        sorted_smells = sorted(analysis.code_smells,
                               key=lambda x: (x.severity.value, x.confidence),
                               reverse=True)

        for smell in sorted_smells[:max_issues]:
            self.logger.info(
                f"Рефакторинг: {smell.type} в строках {smell.location.line_start}-{smell.location.line_end}")

            result = self.generator.refactor_code(code, smell)
            results.append(result)

            # Если рефакторинг успешен, используем улучшенный код для следующих итераций
            if result.success:
                code = result.refactored_code

        self.logger.info(f"Рефакторинг завершен. Обработано {len(results)} проблем")
        return results

    def get_metrics_comparison(self, results: List[RefactoringResult]) -> Dict[str, Any]:
        """Сравнение метрик до и после рефакторинга"""
        if not results:
            return {}

        # Берем первый и последний результат для сравнения
        first_result = results[0]
        last_result = results[-1]

        before = first_result.metrics_before
        after = last_result.metrics_after

        def calculate_improvement(old_val, new_val):
            if old_val == 0:
                return 0.0
            return ((old_val - new_val) / old_val) * 100

        return {
            "metrics_before": {
                "cyclomatic_complexity": before.cyclomatic_complexity,
                "cognitive_complexity": before.cognitive_complexity,
                "lines_of_code": before.lines_of_code,
                "maintainability_index": before.maintainability_index,
                "number_of_methods": before.number_of_methods,
                "number_of_classes": before.number_of_classes
            },
            "metrics_after": {
                "cyclomatic_complexity": after.cyclomatic_complexity,
                "cognitive_complexity": after.cognitive_complexity,
                "lines_of_code": after.lines_of_code,
                "maintainability_index": after.maintainability_index,
                "number_of_methods": after.number_of_methods,
                "number_of_classes": after.number_of_classes
            },
            "improvements": {
                "cyclomatic_complexity": calculate_improvement(
                    before.cyclomatic_complexity, after.cyclomatic_complexity
                ),
                "cognitive_complexity": calculate_improvement(
                    before.cognitive_complexity, after.cognitive_complexity
                ),
                "lines_of_code": calculate_improvement(
                    before.lines_of_code, after.lines_of_code
                ),
                "maintainability_index": calculate_improvement(
                    after.maintainability_index, before.maintainability_index  # Обратное для MI
                ),
                "duplication_ratio": calculate_improvement(
                    before.duplication_ratio, after.duplication_ratio
                )
            }
        }

    def save_refactored_code(self, file_path: str, results: List[RefactoringResult],
                             backup: bool = True) -> bool:
        """Сохранение рефакторенного кода"""
        try:
            if not results or not any(r.success for r in results):
                self.logger.warning("Нет успешных результатов для сохранения")
                return False

            # Создание бэкапа
            if backup:
                backup_path = f"{file_path}.backup"
                with open(file_path, 'r', encoding='utf-8') as original:
                    with open(backup_path, 'w', encoding='utf-8') as backup_file:
                        backup_file.write(original.read())
                self.logger.info(f"Создан бэкап: {backup_path}")

            # Поиск последнего успешного результата
            final_code = None
            for result in reversed(results):
                if result.success:
                    final_code = result.refactored_code
                    break

            if final_code:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(final_code)
                self.logger.info(f"Сохранен рефакторенный код: {file_path}")
                return True

            return False

        except Exception as e:
            self.logger.error(f"Ошибка сохранения: {e}")
            return False

    def generate_report(self, file_path: str, analysis: AnalysisResult,
                        results: List[RefactoringResult]) -> str:
        """Генерация отчета о рефакторинге"""
        report = []
        report.append(f"# Отчет о рефакторинге: {file_path}")
        report.append(f"Дата: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Исходные метрики
        report.append("## Анализ исходного кода")
        report.append(f"- Время анализа: {analysis.parse_time:.2f} сек")
        report.append(f"- Обнаружено проблем: {len(analysis.code_smells)}")
        report.append("")

        if analysis.code_smells:
            report.append("### Обнаруженные проблемы:")
            for smell in analysis.code_smells:
                report.append(f"- **{smell.type}** (строки {smell.location.line_start}-{smell.location.line_end})")
                report.append(f"  - Серьезность: {smell.severity.value}")
                report.append(f"  - Описание: {smell.description}")
                report.append(f"  - Рекомендация: {smell.suggested_refactoring}")
                report.append("")

        # Результаты рефакторинга
        if results:
            report.append("## Результаты рефакторинга")
            successful = [r for r in results if r.success]
            report.append(f"- Успешно обработано: {len(successful)}/{len(results)}")
            report.append("")

            for i, result in enumerate(results, 1):
                status = "✅ Успешно" if result.success else "❌ Ошибка"
                report.append(f"### Рефакторинг {i}: {result.refactoring_type.value} {status}")

                if result.success:
                    report.append(f"**Объяснение:** {result.explanation}")
                    report.append("")

                    # Diff
                    if result.diff:
                        report.append("**Изменения:**")
                        report.append("```diff")
                        report.append(result.diff)
                        report.append("```")
                        report.append("")
                else:
                    report.append("**Предупреждения:**")
                    for warning in result.warnings:
                        report.append(f"- {warning}")
                    report.append("")

            # Сравнение метрик
            metrics_comparison = self.get_metrics_comparison(results)
            if metrics_comparison:
                report.append("## Сравнение метрик")
                report.append("| Метрика | До | После | Улучшение |")
                report.append("|---------|-------|-------|-----------|")

                before = metrics_comparison["metrics_before"]
                after = metrics_comparison["metrics_after"]
                improvements = metrics_comparison["improvements"]

                for metric in before.keys():
                    improvement = improvements.get(metric, 0)
                    improvement_str = f"{improvement:+.1f}%" if improvement != 0 else "0%"
                    report.append(f"| {metric} | {before[metric]} | {after[metric]} | {improvement_str} |")

                report.append("")

        # Рекомендации
        if analysis.suggestions:
            report.append("## Рекомендации")
            for suggestion in analysis.suggestions:
                report.append(f"- {suggestion}")
            report.append("")

        return "\n".join(report)
