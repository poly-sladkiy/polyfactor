# =============================================================================
# Утилиты и вспомогательные функции
# =============================================================================
import json
from pathlib import Path
from typing import List, Dict, Any

from refactoring_system import RefactoringSystem


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
                print(
                    f"- {smell.type} (строки {smell.location.line_start}-{smell.location.line_end}): {smell.description}")

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
