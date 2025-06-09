# =============================================================================
# Модуль анализа кода
# =============================================================================
import ast
import math

from loguru import logger

from base import Metrics


class MetricsCalculator:
    """Калькулятор метрик качества кода"""

    def calculate_metrics(self, code: str) -> Metrics:
        """Вычисление всех метрик для кода"""
        try:
            tree = ast.parse(code)

            return Metrics(
                cyclomatic_complexity=self._cyclomatic_complexity(tree),
                cognitive_complexity=self._cognitive_complexity(tree),
                lines_of_code=self._lines_of_code(code),
                number_of_methods=self._count_methods(tree),
                number_of_classes=self._count_classes(tree),
                maintainability_index=self._maintainability_index(tree, code),
                halstead_volume=self._halstead_volume(tree),
                duplication_ratio=self._duplication_ratio(code)
            )
        except Exception as er:
            logger.exception(er)
            return Metrics()

    def _cyclomatic_complexity(self, tree: ast.AST) -> int:
        """Цикломатическая сложность"""
        complexity = 1
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
            elif isinstance(node, (ast.ExceptHandler, ast.With)):
                complexity += 1
        return complexity

    def _cognitive_complexity(self, tree: ast.AST) -> int:
        """Когнитивная сложность"""
        complexity = 0

        def calculate_recursive(node, nesting_level=0):
            nonlocal complexity

            for child in ast.iter_child_nodes(node):
                increment = 0
                new_nesting = nesting_level

                if isinstance(child, (ast.If, ast.While, ast.For)):
                    increment = 1 + nesting_level
                    new_nesting = nesting_level + 1
                elif isinstance(child, ast.BoolOp):
                    increment = 1
                elif isinstance(child, (ast.Break, ast.Continue)):
                    increment = 1

                complexity += increment
                calculate_recursive(child, new_nesting)

        calculate_recursive(tree)
        return complexity

    def _lines_of_code(self, code: str) -> int:
        """Количество строк кода (без пустых и комментариев)"""
        lines = code.split('\n')
        return len([line for line in lines if line.strip() and not line.strip().startswith('#')])

    def _count_methods(self, tree: ast.AST) -> int:
        """Количество методов"""
        return len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)])

    def _count_classes(self, tree: ast.AST) -> int:
        """Количество классов"""
        return len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)])

    def _maintainability_index(self, tree: ast.AST, code: str) -> float:
        """Индекс сопровождаемости"""
        try:
            volume = self._halstead_volume(tree)
            complexity = self._cyclomatic_complexity(tree)
            loc = self._lines_of_code(code)

            if volume == 0 or loc == 0:
                return 100.0

            mi = 171 - 5.2 * math.log(volume) - 0.23 * complexity - 16.2 * math.log(loc)
            return max(0, min(100, mi))
        except:
            return 50.0

    def _halstead_volume(self, tree: ast.AST) -> float:
        """Объем Холстеда"""
        operators = set()
        operands = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp):
                operators.add(type(node.op).__name__)
            elif isinstance(node, ast.Name):
                operands.add(node.id)
            elif isinstance(node, ast.Constant):
                operands.add(str(node.value))

        n = len(operators) + len(operands)
        N = len(list(ast.walk(tree)))

        if n == 0:
            return 0.0

        return N * math.log2(n)

    def _duplication_ratio(self, code: str) -> float:
        """Коэффициент дублирования"""
        lines = [line.strip() for line in code.split('\n') if line.strip()]
        if len(lines) < 2:
            return 0.0

        unique_lines = len(set(lines))
        total_lines = len(lines)

        return 1.0 - (unique_lines / total_lines)

