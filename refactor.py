from demo import create_sample_project, demo_simple_refactoring
from utils import CLIInterface


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