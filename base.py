# =============================================================================
# Базовые структуры данных
# =============================================================================
from dataclasses import dataclass, field
from enum import Enum
from typing import List


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