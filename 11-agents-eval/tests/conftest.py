"""
Fixtures для E2E тестов агента

Переиспользуем существующую логику инициализации из src/
"""
import pytest_asyncio
import sys
from pathlib import Path

# Добавляем src/ в путь для импортов
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest_asyncio.fixture(scope="session")
async def agent_fixture():
    """
    Создает агента для тестирования
    
    Переиспользуем функции из src/:
    - bot.py: setup_indexing() - логика индексации
    - agent.py: initialize_agent() - создание агента
    
    Scope=session - создаем один раз для всех тестов
    Каждый тест использует уникальный thread_id для изоляции
    """
    # Импортируем существующие функции
    from bot import setup_indexing
    from agent import initialize_agent
    
    # Используем ту же логику индексации что и в боте
    await setup_indexing()
    
    # Используем ту же функцию инициализации агента
    agent = await initialize_agent()
    
    return agent

