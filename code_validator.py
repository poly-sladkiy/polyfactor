# =============================================================================
# Модуль генерации и валидации
# =============================================================================
import ast
import difflib
from typing import Tuple, Dict

import black
import isort
from loguru import logger

from base import RefactoringType, CodeSmell, RefactoringResult
from llm_manager import LLMInterface
from metrics_calculator import MetricsCalculator


class CodeValidator:
    """Валидатор кода"""

    def validate_syntax(self, code: str) -> Tuple[bool, str]:
        """Проверка синтаксиса"""
        try:
            ast.parse(code)
            compile(code, '<string>', 'exec')
            return True, ""
        except SyntaxError as e:
            return False, f"Синтаксическая ошибка: {e.msg} на строке {e.lineno}"
        except Exception as e:
            return False, f"Ошибка компиляции: {str(e)}"

    def validate_functionality(self, original_code: str, refactored_code: str) -> Tuple[bool, str]:
        """Базовая проверка сохранения функциональности"""
        try:
            # Проверяем, что основные определения сохранились
            original_tree = ast.parse(original_code)
            refactored_tree = ast.parse(refactored_code)

            original_defs = self._extract_definitions(original_tree)
            refactored_defs = self._extract_definitions(refactored_tree)

            # Проверяем, что публичные определения сохранились
            missing_defs = []
            for name, def_type in original_defs.items():
                if not name.startswith('_') and name not in refactored_defs:
                    missing_defs.append(f"{def_type} {name}")

            if missing_defs:
                return False, f"Отсутствуют определения: {', '.join(missing_defs)}"

            return True, ""

        except Exception as e:
            return False, f"Ошибка валидации: {str(e)}"

    def _extract_definitions(self, tree: ast.AST) -> Dict[str, str]:
        """Извлечение определений функций и классов"""
        definitions = {}

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                definitions[node.name] = "function"
            elif isinstance(node, ast.ClassDef):
                definitions[node.name] = "class"

        return definitions


class CodeGenerator:
    """Генератор улучшенного кода"""

    def __init__(self, llm_interface: LLMInterface):
        self.llm = llm_interface
        self.validator = CodeValidator()
        self.metrics_calc = MetricsCalculator()
        self.logger = logger

    def refactor_code(self, code: str, code_smell: CodeSmell) -> RefactoringResult:
        """Рефакторинг кода"""
        # Метрики до рефакторинга
        metrics_before = self.metrics_calc.calculate_metrics(code)

        # Определение типа рефакторинга
        refactoring_type = self._map_smell_to_refactoring(code_smell.type)

        # Рефакторинг с помощью LLM
        refactored_code, explanation = self.llm.refactor_code(
            code, code_smell.type, refactoring_type.value
        )

        # Валидация
        warnings = []

        # Проверка синтаксиса
        syntax_valid, syntax_error = self.validator.validate_syntax(refactored_code)
        if not syntax_valid:
            warnings.append(syntax_error)
            refactored_code = code  # Возврат к исходному коду

        # Проверка функциональности
        func_valid, func_error = self.validator.validate_functionality(code, refactored_code)
        if not func_valid:
            warnings.append(func_error)

        # Форматирование кода
        if syntax_valid:
            refactored_code = self._format_code(refactored_code)

        # Метрики после рефакторинга
        metrics_after = self.metrics_calc.calculate_metrics(refactored_code)

        # Генерация diff
        diff = self._generate_diff(code, refactored_code)

        success = syntax_valid and func_valid and len(warnings) == 0

        return RefactoringResult(
            original_code=code,
            refactored_code=refactored_code,
            refactoring_type=refactoring_type,
            success=success,
            metrics_before=metrics_before,
            metrics_after=metrics_after,
            explanation=explanation,
            diff=diff,
            warnings=warnings
        )

    def _map_smell_to_refactoring(self, smell_type: str) -> RefactoringType:
        """Сопоставление типа запаха с типом рефакторинга"""
        mapping = {
            "long_method": RefactoringType.EXTRACT_METHOD,
            "large_class": RefactoringType.EXTRACT_CLASS,
            "long_parameter_list": RefactoringType.RENAME_VARIABLE,
            "duplicate_code": RefactoringType.REMOVE_DUPLICATE,
            "complex_conditional": RefactoringType.SIMPLIFY_CONDITIONAL
        }
        return mapping.get(smell_type, RefactoringType.EXTRACT_METHOD)

    def _format_code(self, code: str) -> str:
        """Форматирование кода"""
        try:
            # Форматирование с black
            formatted = black.format_str(code, mode=black.FileMode())

            # Сортировка импортов
            formatted = isort.code(formatted)

            return formatted
        except Exception as e:
            self.logger.warning(f"Ошибка форматирования: {e}")
            return code

    def _generate_diff(self, original: str, refactored: str) -> str:
        """Генерация diff"""
        diff = difflib.unified_diff(
            original.splitlines(keepends=True),
            refactored.splitlines(keepends=True),
            fromfile="original.py",
            tofile="refactored.py"
        )
        return ''.join(diff)
