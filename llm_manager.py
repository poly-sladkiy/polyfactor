# =============================================================================
# Модуль LLM интерфейса
# =============================================================================
import re
from dataclasses import dataclass
from typing import Tuple

import torch
from loguru import logger
from transformers import AutoTokenizer, AutoModelForCausalLM


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
