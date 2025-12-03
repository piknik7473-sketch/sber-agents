"""
E2E тесты агента с LLM-as-Judge evaluators

Медленные тесты с использованием LLM для оценки траекторий.
Требуют хорошую модель (gpt-4o, claude-3.5-sonnet) и дороже в запуске.

Рекомендуется запускать отдельно от основных тестов:
    pytest tests/test_agent_llm_judge.py -v
    # или через marker
    pytest -m llm_judge -v
"""
import pytest
import sys
import logging
from pathlib import Path
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from agentevals.trajectory.llm import (
    create_async_trajectory_llm_as_judge, 
    TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE
)
from tests.helpers import extract_trajectory, print_trajectory

# Импортируем config для доступа к AGENTEVALS_LLM_MODEL
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from config import config

logger = logging.getLogger(__name__)


@pytest.mark.llm_judge
@pytest.mark.asyncio
async def test_combined_scenario_with_ref(agent_fixture):
    """
    Тест: Combined Scenario (LLM-as-Judge с референсом)
    
    Проверяет многошаговый диалог с комбинацией инструментов:
    1. Вопрос требующий search_products + deposit_income_calculator
    2. Последовательность вызовов инструментов
    3. Финальный ответ с комбинированной информацией
    4. LLM сравнивает фактическую траекторию с референсной
    
    ⚠️ ВНИМАНИЕ: Требует хорошую модель для LLM-as-Judge (gpt-4o, claude-3.5-sonnet)
    """
    agent = agent_fixture
    
    # Запрос требующий комбинацию инструментов
    user_message = "Найди ставку по вкладу Пополняй и посчитай доходность для суммы 500000 на год по минимальной ставке"
    
    # Запускаем агента и получаем траекторию
    actual_trajectory = await extract_trajectory(agent, "test_combined_4", user_message)
    
    # Для отладки выводим траекторию
    print_trajectory(actual_trajectory)
    
    # Создаем референсную траекторию
    # Агент должен:
    # 1. Найти информацию о вкладе Пополняй (search_products)
    # 2. Найти общие условия в документах (rag_search) - опционально
    # 3. Рассчитать доход (deposit_income_calculator)
    # 4. Дать комбинированный ответ
    reference_trajectory = [
        HumanMessage(content=user_message),
        
        # Шаг 1: Поиск информации о вкладе Пополняй
        AIMessage(
            content="",
            tool_calls=[{
                "name": "search_products",
                "args": {"product_type": "deposit", "keyword": "Пополняй"},
                "id": "call_1"
            }]
        ),
        ToolMessage(
            content='[{"name": "Пополняй", "type": "deposit", "rate": 16.0, "min_amount": 1000}]',
            name="search_products",
            tool_call_id="call_1"
        ),
        
        # Шаг 2: Расчет доходности
        AIMessage(
            content="",
            tool_calls=[{
                "name": "deposit_income_calculator",
                "args": {"amount": 500000, "rate": 16, "term_months": 12},
                "id": "call_2"
            }]
        ),
        ToolMessage(
            content='{"income": 80000, "final_amount": 580000, "calculation_type": "simple"}',
            name="deposit_income_calculator",
            tool_call_id="call_2"
        ),
        
        # Финальный ответ
        AIMessage(content="Вклад Пополняй имеет ставку 16% годовых. При размещении 500000 рублей на год вы заработаете 80000 рублей...")
    ]
    
    # Создаем async LLM-as-Judge evaluator С РЕФЕРЕНСОМ
    # LLM сравнивает фактическую траекторию с эталонной
    # Оценивает логику выбора инструментов и последовательность действий
    evaluator = create_async_trajectory_llm_as_judge(
        prompt=TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE,
        model=config.AGENTEVALS_LLM_MODEL
    )
    
    # Оцениваем траекторию с референсом
    result = await evaluator(
        outputs=actual_trajectory,
        reference_outputs=reference_trajectory
    )
    
    # Проверяем score
    score = result.get("score", 0)
    comment = result.get("comment", "No comment")
    
    # Выводим результат evaluator в лог
    logger.info("=" * 60)
    logger.info("📊 LLM-AS-JUDGE EVALUATOR RESULT (WITH REFERENCE)")
    logger.info(f"   Score: {score}")
    logger.info(f"   Comment: {comment}")
    logger.info(f"   Trajectory length: {len(actual_trajectory)}")
    logger.info(f"   Reference length: {len(reference_trajectory)}")
    logger.info("=" * 60)
    
    # Assert с информативным сообщением
    assert score > 0.7, (
        f"Expected LLM-as-Judge score > 0.7 for combined scenario.\n"
        f"Actual score: {score}\n"
        f"Comment: {comment}\n"
        f"Trajectory length: {len(actual_trajectory)}\n"
        f"Reference length: {len(reference_trajectory)}"
    )

