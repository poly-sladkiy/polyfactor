import time
from typing import List

from loguru import logger

from base import AnalysisResult, Metrics, CodeSmell
from code_smell_detector import CodeSmellDetector
from metrics_calculator import MetricsCalculator


class CodeAnalyzer:
    """Основной анализатор кода"""

    def __init__(self):
        self.metrics_calc = MetricsCalculator()
        self.smell_detector = CodeSmellDetector()
        self.logger = logger

    def analyze_file(self, file_path: str) -> AnalysisResult:
        """Анализ файла Python"""
        start_time = time.time()

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()

            # Вычисление метрик
            metrics = self.metrics_calc.calculate_metrics(code)

            # Обнаружение запахов кода
            code_smells = self.smell_detector.detect_smells(code, file_path)

            # Генерация предложений
            suggestions = self._generate_suggestions(metrics, code_smells)

            parse_time = time.time() - start_time

            return AnalysisResult(
                file_path=file_path,
                metrics=metrics,
                code_smells=code_smells,
                suggestions=suggestions,
                parse_time=parse_time
            )

        except Exception as e:
            self.logger.error(f"Error analyzing file {file_path}: {e}")
            return AnalysisResult(
                file_path=file_path,
                metrics=Metrics(),
                code_smells=[],
                suggestions=[],
                parse_time=time.time() - start_time
            )

    def _generate_suggestions(self, metrics: Metrics, smells: List[CodeSmell]) -> List[str]:
        """Генерация предложений по улучшению"""
        suggestions = []

        if metrics.cyclomatic_complexity > 15:
            suggestions.append("Рассмотрите разбиение сложных методов на более простые")

        if metrics.maintainability_index < 40:
            suggestions.append("Код требует рефакторинга для улучшения сопровождаемости")

        if len(smells) > 5:
            suggestions.append("Обнаружено много проблем - рекомендуется комплексный рефакторинг")

        return suggestions
