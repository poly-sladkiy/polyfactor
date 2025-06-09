import ast
import re
import json
import time
import hashlib
import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
from enum import Enum
from pathlib import Path
import difflib
import math

import torch
from loguru import logger
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import black
import isort

# =============================================================================
# Базовые структуры данных
# =============================================================================

class RefactoringType(Enum):
    EXTRACT_METHOD = "extract_method"
    EXTRACT_CLASS = "extract_class"
    RENAME_VARIABLE = "rename_variable"
    SIMPLIFY_CONDITIONAL = "simplify_conditional"
    REMOVE_DUPLICATE = "remove_duplicate"

class Severity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class Location:
    file_path: str
    line_start: int
    line_end: int

@dataclass
class CodeSmell:
    type: str
    location: Location
    severity: Severity
    description: str
    suggested_refactoring: str
    confidence: float

@dataclass
class Metrics:
    cyclomatic_complexity: int = 0
    cognitive_complexity: int = 0
    lines_of_code: int = 0
    number_of_methods: int = 0
    number_of_classes: int = 0
    maintainability_index: float = 0.0
    halstead_volume: float = 0.0
    duplication_ratio: float = 0.0

@dataclass
class AnalysisResult:
    file_path: str
    metrics: Metrics
    code_smells: List[CodeSmell]
    suggestions: List[str]
    parse_time: float

@dataclass
class RefactoringResult:
    original_code: str
    refactored_code: str
    refactoring_type: RefactoringType
    success: bool
    metrics_before: Metrics
    metrics_after: Metrics
    explanation: str
    diff: str
    warnings: List[str] = field(default_factory=list)

# =============================================================================
# Модуль анализа кода
# =============================================================================

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

# =============================================================================
# Модуль LLM интерфейса
# =============================================================================

@dataclass
class LLMConfig:
    """Конфигурация LLM"""
    model_name: str = "deepseek-ai/deepseek-coder-1.3b-instruct"
    device: str = "auto"
    max_length: int = 4096
    temperature: float = 0.1
    top_p: float = 0.95
    do_sample: bool = True


class PromptManager:
    """Менеджер промптов"""
    
    def __init__(self):
        self.templates = {
            "system": """Ты эксперт по рефакторингу Python кода. Твоя задача - улучшить качество кода, сохранив его функциональность.

Требования:
- Сохраняй функциональность кода
- Следуй PEP 8
- Улучшай читаемость
- Применяй принципы SOLID
- Добавляй типизацию при необходимости

Формат ответа:
```python
# Улучшенный код
```

Объяснение: [краткое описание изменений]""",
            
            "refactoring": """Проблема: {problem_type}
Тип рефакторинга: {refactoring_type}

Исходный код:
```python
{code}
```

Инструкции:
1. Исправь проблему: {problem_type}
2. Примени рефакторинг: {refactoring_type}
3. Сохрани функциональность
4. Улучши качество кода"""
        }
    
    def build_prompt(self, code: str, problem_type: str, refactoring_type: str) -> str:
        """Построение промпта для рефакторинга"""
        system_prompt = self.templates["system"]
        task_prompt = self.templates["refactoring"].format(
            problem_type=problem_type,
            refactoring_type=refactoring_type,
            code=code
        )
        
        return f"{system_prompt}\n\n{task_prompt}"


class LLMInterface:
    """Интерфейс для работы с LLM"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.prompt_manager = PromptManager()
        self.logger = logger
        self._load_model()
    
    def _load_model(self):
        """Загрузка модели"""
        try:
            self.logger.info(f"Загрузка модели {self.config.model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map=self.config.device
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.logger.info("Модель загружена успешно")
            
        except Exception as e:
            self.logger.error(f"Ошибка загрузки модели: {e}")
            raise
    
    def refactor_code(self, code: str, problem_type: str, refactoring_type: str) -> Tuple[str, str]:
        """Рефакторинг кода с помощью LLM"""
        try:
            # Построение промпта
            prompt = self.prompt_manager.build_prompt(code, problem_type, refactoring_type)
            
            # Токенизация
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            
            # Генерация
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=1024,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=self.config.do_sample,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # Декодирование
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Извлечение кода и объяснения
            refactored_code, explanation = self._parse_response(response, prompt)
            
            return refactored_code, explanation
            
        except Exception as e:
            self.logger.error(f"Ошибка рефакторинга: {e}")
            return code, f"Ошибка: {str(e)}"
    
    def _parse_response(self, response: str, prompt: str) -> Tuple[str, str]:
        """Парсинг ответа LLM"""
        # Удаляем промпт из ответа
        response = response[len(prompt):].strip()
        
        # Ищем код в блоках ```python
        code_pattern = r'```python\s*(.*?)\s*```'
        code_match = re.search(code_pattern, response, re.DOTALL)
        
        if code_match:
            code = code_match.group(1).strip()
        else:
            # Если нет блоков, берем все как код
            code = response.split("Объяснение:")[0].strip()
        
        # Ищем объяснение
        explanation_pattern = r'Объяснение:\s*(.*?)$'
        explanation_match = re.search(explanation_pattern, response, re.DOTALL)
        
        explanation = explanation_match.group(1).strip() if explanation_match else "Объяснение не предоставлено"
        
        return code, explanation

# =============================================================================
# Модуль генерации и валидации
# =============================================================================

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

# =============================================================================
# Главная система рефакторинга
# =============================================================================

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
            self.logger.info(f"Рефакторинг: {smell.type} в строках {smell.location.line_start}-{smell.location.line_end}")
            
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
            )   ,
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

# =============================================================================
# Утилиты и вспомогательные функции
# =============================================================================

class ProjectScanner:
    """Сканер проектов для поиска Python файлов"""
    
    def __init__(self, exclude_patterns: List[str] = None):
        self.exclude_patterns = exclude_patterns or [
            "__pycache__", ".git", ".venv", "venv", "env",
            "node_modules", ".pytest_cache", "build", "dist"
        ]
    
    def scan_directory(self, directory: str) -> List[str]:
        """Сканирование директории для поиска Python файлов"""
        python_files = []
        directory_path = Path(directory)
        
        for file_path in directory_path.rglob("*.py"):
            # Проверка исключений
            if any(pattern in str(file_path) for pattern in self.exclude_patterns):
                continue
            
            python_files.append(str(file_path))
        
        return sorted(python_files)


class ConfigManager:
    """Менеджер конфигурации"""
    
    @staticmethod
    def load_config(config_path: str = "refactor_config.json") -> Dict[str, Any]:
        """Загрузка конфигурации из файла"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return ConfigManager.get_default_config()
    
    @staticmethod
    def save_config(config: Dict[str, Any], config_path: str = "refactor_config.json"):
        """Сохранение конфигурации в файл"""
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """Получение конфигурации по умолчанию"""
        return {
            "llm": {
                "model_name": "deepseek-ai/deepseek-coder-1.3b-instruct",
                "device": "auto",
                "temperature": 0.1,
                "max_length": 4096
            },
            "analysis": {
                "max_issues_per_file": 3,
                "min_confidence": 0.7,
                "severity_filter": ["medium", "high", "critical"]
            },
            "output": {
                "create_backup": True,
                "generate_report": True,
                "format_code": True
            }
        }


class CLIInterface:
    """Интерфейс командной строки"""
    
    def __init__(self):
        self.system = None
        self.scanner = ProjectScanner()
    
    def run(self, args: List[str]):
        """Запуск CLI"""
        if len(args) < 2:
            self.print_help()
            return
        
        command = args[1]
        
        if command == "analyze":
            self.analyze_command(args[2:])
        elif command == "refactor":
            self.refactor_command(args[2:])
        elif command == "scan":
            self.scan_command(args[2:])
        elif command == "config":
            self.config_command(args[2:])
        else:
            self.print_help()
    
    def analyze_command(self, args: List[str]):
        """Команда анализа"""
        if not args:
            print("Использование: analyze <file_path>")
            return
        
        file_path = args[0]
        if not Path(file_path).exists():
            print(f"Файл не найден: {file_path}")
            return
        
        print(f"Анализ файла: {file_path}")
        
        if self.system is None:
            self.system = RefactoringSystem()
        
        analysis = self.system.analyze_file(file_path)
        
        print(f"\nРезультаты анализа:")
        print(f"- Время анализа: {analysis.parse_time:.2f} сек")
        print(f"- Строк кода: {analysis.metrics.lines_of_code}")
        print(f"- Цикломатическая сложность: {analysis.metrics.cyclomatic_complexity}")
        print(f"- Индекс сопровождаемости: {analysis.metrics.maintainability_index:.1f}")
        print(f"- Обнаружено проблем: {len(analysis.code_smells)}")
        
        if analysis.code_smells:
            print("\nОбнаруженные проблемы:")
            for smell in analysis.code_smells:
                print(f"- {smell.type} (строки {smell.location.line_start}-{smell.location.line_end}): {smell.description}")
    
    def refactor_command(self, args: List[str]):
        """Команда рефакторинга"""
        if not args:
            print("Использование: refactor <file_path> [--save] [--report]")
            return
        
        file_path = args[0]
        save_changes = "--save" in args
        generate_report = "--report" in args
        
        if not Path(file_path).exists():
            print(f"Файл не найден: {file_path}")
            return
        
        print(f"Рефакторинг файла: {file_path}")
        
        if self.system is None:
            self.system = RefactoringSystem()
        
        # Анализ
        analysis = self.system.analyze_file(file_path)
        
        if not analysis.code_smells:
            print("Проблемы не обнаружены. Рефакторинг не требуется.")
            return
        
        # Рефакторинг
        results = self.system.refactor_file(file_path)
        
        successful = [r for r in results if r.success]
        print(f"\nРефакторинг завершен: {len(successful)}/{len(results)} успешно")
        
        # Сохранение
        if save_changes and successful:
            if self.system.save_refactored_code(file_path, results):
                print(f"Изменения сохранены в {file_path}")
            else:
                print("Ошибка сохранения изменений")
        
        # Отчет
        if generate_report:
            report = self.system.generate_report(file_path, analysis, results)
            report_path = f"{file_path}.refactor_report.md"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"Отчет сохранен в {report_path}")
        
        # Метрики
        metrics_comparison = self.system.get_metrics_comparison(results)
        if metrics_comparison:
            print("\nСравнение метрик:")
            improvements = metrics_comparison["improvements"]
            for metric, improvement in improvements.items():
                if improvement != 0:
                    print(f"- {metric}: {improvement:+.1f}%")
    
    def scan_command(self, args: List[str]):
        """Команда сканирования директории"""
        if not args:
            print("Использование: scan <directory>")
            return
        
        directory = args[0]
        if not Path(directory).exists():
            print(f"Директория не найдена: {directory}")
            return
        
        print(f"Сканирование директории: {directory}")
        files = self.scanner.scan_directory(directory)
        
        print(f"\nНайдено {len(files)} Python файлов:")
        for file_path in files:
            print(f"- {file_path}")
    
    def config_command(self, args: List[str]):
        """Команда работы с конфигурацией"""
        if not args or args[0] == "show":
            config = ConfigManager.load_config()
            print("Текущая конфигурация:")
            print(json.dumps(config, indent=2, ensure_ascii=False))
        elif args[0] == "create":
            config = ConfigManager.get_default_config()
            ConfigManager.save_config(config)
            print("Создан файл конфигурации по умолчанию: refactor_config.json")
        else:
            print("Использование: config [show|create]")
    
    def print_help(self):
        """Вывод справки"""
        help_text = """
    Система автоматического рефакторинга Python кода с использованием LLM

    Использование:
    python refactor.py <command> [options]

    Команды:
    analyze <file>           - Анализ файла Python
    refactor <file> [flags]  - Рефакторинг файла
    --save                 - Сохранить изменения
    --report               - Создать отчет
    scan <directory>         - Сканирование директории
    config [show|create]     - Работа с конфигурацией

    Примеры:
    python refactor.py analyze my_script.py
    python refactor.py refactor my_script.py --save --report
    python refactor.py scan ./src
    python refactor.py config create
        """
        print(help_text.strip())

# =============================================================================
# Демонстрационные примеры
# =============================================================================

def demo_simple_refactoring():
    """Демонстрация простого рефакторинга"""
    print("=== Демонстрация системы рефакторинга ===\n")
    
    # Пример проблемного кода
    problem_code = '''
def process_user_data(name, email, age, address, phone, is_active, role, department):
    if name is None or name == "":
        return False
    if email is None or "@" not in email:
        return False
    if age is None or age < 0 or age > 150:
        return False
    if address is None or len(address) < 5:
        return False
    if phone is None or len(phone) < 10:
        return False

    user_data = {}
    user_data["name"] = name.strip().lower()
    user_data["email"] = email.strip().lower()
    user_data["age"] = age
    user_data["address"] = address.strip()
    user_data["phone"] = phone.strip()
    user_data["is_active"] = is_active
    user_data["role"] = role
    user_data["department"] = department

    return user_data
    '''
    
    # Создание временного файла
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(problem_code)
        temp_file = f.name
    
    try:
        # Инициализация системы
        system = RefactoringSystem()
        
        # Анализ
        print("1. Анализ исходного кода...")
        analysis = system.analyze_file(temp_file)
        
        print(f"   Обнаружено проблем: {len(analysis.code_smells)}")
        for smell in analysis.code_smells:
            print(f"   - {smell.type}: {smell.description}")
        
        # Рефакторинг
        print("\n2. Рефакторинг...")
        results = system.refactor_file(temp_file, max_issues=2)
        
        successful = [r for r in results if r.success]
        print(f"   Успешно обработано: {len(successful)}/{len(results)} проблем")
        
        # Сравнение метрик
        print("\n3. Сравнение метрик:")
        metrics_comparison = system.get_metrics_comparison(results)
        if metrics_comparison:
            improvements = metrics_comparison["improvements"]
            for metric, improvement in improvements.items():
                if improvement != 0:
                    print(f"   {metric}: {improvement:+.1f}%")
        
        # Показать результат
        if successful:
            print("\n4. Результат рефакторинга:")
            print("```python")
            print(successful[-1].refactored_code)
            print("```")
            
            print(f"\nОбъяснение: {successful[-1].explanation}")
    
    finally:
        # Удаление временного файла
        Path(temp_file).unlink()


def create_sample_project():
    """Создание примера проекта для демонстрации"""
    project_dir = Path("sample_project")
    project_dir.mkdir(exist_ok=True)
    
    # Пример с длинным методом
    long_method_code = '''
class UserManager:
    def process_registration(self, user_data):
        # Валидация email
        if not user_data.get("email"):
            raise ValueError("Email is required")
        if "@" not in user_data["email"]:
            raise ValueError("Invalid email format")
        if len(user_data["email"]) > 100:
            raise ValueError("Email too long")
        
        # Валидация пароля
        password = user_data.get("password", "")
        if len(password) < 8:
            raise ValueError("Password too short")
        if not any(c.isupper() for c in password):
            raise ValueError("Password must contain uppercase")
        if not any(c.islower() for c in password):
            raise ValueError("Password must contain lowercase")
        if not any(c.isdigit() for c in password):
            raise ValueError("Password must contain digit")
        
        # Обработка данных
        processed_data = {}
        processed_data["email"] = user_data["email"].lower().strip()
        processed_data["password_hash"] = self.hash_password(password)
        processed_data["created_at"] = self.get_current_time()
        processed_data["is_active"] = True
        processed_data["role"] = "user"
        
        # Сохранение в базу данных
        user_id = self.save_to_database(processed_data)
        
        # Отправка email подтверждения
        self.send_confirmation_email(processed_data["email"])
        
        # Логирование
        self.log_user_registration(user_id, processed_data["email"])
        
        return user_id

    def hash_password(self, password):
        return f"hashed_{password}"

    def get_current_time(self):
        import time
        return time.time()

    def save_to_database(self, data):
        return 12345

    def send_confirmation_email(self, email):
        print(f"Sending email to {email}")

    def log_user_registration(self, user_id, email):
        print(f"User {user_id} registered with email {email}")
    '''
    
    # Сохранение примера
    with open(project_dir / "user_manager.py", "w", encoding="utf-8") as f:
        f.write(long_method_code)
    
    print(f"Создан пример проекта в директории: {project_dir}")
    return str(project_dir)


# =============================================================================
# Точка входа
# =============================================================================

def main():
    """Главная функция"""
    import sys
    
    if len(sys.argv) > 1:
        # CLI режим
        cli = CLIInterface()
        cli.run(sys.argv)
    else:
        # Демонстрационный режим
        print("Запуск в демонстрационном режиме...\n")
        
        # Создание примера проекта
        project_dir = create_sample_project()
        
        # Демонстрация
        demo_simple_refactoring()
        
        print(f"\nДля тестирования CLI используйте:")
        print(f"python refactor.py analyze {project_dir}/user_manager.py")
        print(f"python refactor.py refactor {project_dir}/user_manager.py --save --report")


if __name__ == "__main__":
    main()