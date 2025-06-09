# =============================================================================
# Демонстрационные примеры
# =============================================================================
import tempfile
from pathlib import Path

from refactoring_system import RefactoringSystem


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

