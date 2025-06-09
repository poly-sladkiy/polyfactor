import ast
from typing import List

from loguru import logger

from base import CodeSmell, Location, Severity
from metrics_calculator import MetricsCalculator


class CodeSmellDetector:
    """Детектор запахов кода"""

    def __init__(self):
        self.metrics_calc = MetricsCalculator()

    def detect_smells(self, code: str, file_path: str) -> List[CodeSmell]:
        """Обнаружение запахов кода"""
        smells = []

        try:
            tree = ast.parse(code)
            metrics = self.metrics_calc.calculate_metrics(code)

            # Длинные методы
            smells.extend(self._detect_long_methods(tree, file_path))

            # Большие классы
            smells.extend(self._detect_large_classes(tree, file_path))

            # Длинные списки параметров
            smells.extend(self._detect_long_parameter_lists(tree, file_path))

            # Дублирование кода
            if metrics.duplication_ratio > 0.3:
                smells.append(CodeSmell(
                    type="duplicate_code",
                    location=Location(file_path, 1, metrics.lines_of_code),
                    severity=Severity.MEDIUM,
                    description=f"Высокий уровень дублирования: {metrics.duplication_ratio:.1%}",
                    suggested_refactoring="extract_method",
                    confidence=0.8
                ))

        except Exception as e:
            logger.warning(f"Error detecting smells: {e}")

        return smells

    def _detect_long_methods(self, tree: ast.AST, file_path: str) -> List[CodeSmell]:
        """Обнаружение длинных методов"""
        smells = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                lines = (node.end_lineno or node.lineno) - node.lineno
                complexity = self._method_complexity(node)

                if lines > 20 or complexity > 10:
                    severity = Severity.HIGH if lines > 40 else Severity.MEDIUM

                    smells.append(CodeSmell(
                        type="long_method",
                        location=Location(file_path, node.lineno, node.end_lineno or node.lineno),
                        severity=severity,
                        description=f"Метод {node.name}: {lines} строк, сложность {complexity}",
                        suggested_refactoring="extract_method",
                        confidence=0.9
                    ))

        return smells

    def _detect_large_classes(self, tree: ast.AST, file_path: str) -> List[CodeSmell]:
        """Обнаружение больших классов"""
        smells = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
                method_count = len(methods)

                if method_count > 15:
                    smells.append(CodeSmell(
                        type="large_class",
                        location=Location(file_path, node.lineno, node.end_lineno or node.lineno),
                        severity=Severity.HIGH,
                        description=f"Класс {node.name} содержит {method_count} методов",
                        suggested_refactoring="extract_class",
                        confidence=0.8
                    ))

        return smells

    def _detect_long_parameter_lists(self, tree: ast.AST, file_path: str) -> List[CodeSmell]:
        """Обнаружение длинных списков параметров"""
        smells = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                param_count = len(node.args.args)

                if param_count > 5:
                    smells.append(CodeSmell(
                        type="long_parameter_list",
                        location=Location(file_path, node.lineno, node.lineno),
                        severity=Severity.MEDIUM,
                        description=f"Метод {node.name} имеет {param_count} параметров",
                        suggested_refactoring="parameter_object",
                        confidence=0.9
                    ))

        return smells

    def _method_complexity(self, node: ast.FunctionDef) -> int:
        """Сложность метода"""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For)):
                complexity += 1
        return complexity

