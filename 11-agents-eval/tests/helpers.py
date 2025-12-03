"""
Вспомогательные функции для работы с траекториями агента
"""
import logging
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)


async def extract_trajectory(agent, thread_id: str, user_message: str):
    """
    Запуск агента и извлечение траектории
    
    Args:
        agent: ReAct агент
        thread_id: Уникальный ID для изоляции теста
        user_message: Сообщение пользователя
    
    Returns:
        list: Список сообщений (траектория)
    
    Raises:
        Exception: Пробрасывает все исключения включая RateLimitError для обработки в тестах
    """
    config = {"configurable": {"thread_id": thread_id}}
    
    # Запуск агента (пробрасываем все исключения для обработки в тестах)
    await agent.ainvoke(
        {"messages": [HumanMessage(content=user_message)]},
        config=config
    )
    
    # Извлечение полной траектории из состояния
    state = agent.get_state(config)
    trajectory = state.values["messages"]
    
    return trajectory


def print_trajectory(trajectory):
    """
    Вывод траектории для отладки
    
    Args:
        trajectory: Список сообщений
    """
    logger.info("\n" + "="*60)
    logger.info("🔍 AGENT TRAJECTORY")
    logger.info("="*60)
    
    for i, msg in enumerate(trajectory):
        msg_type = type(msg).__name__
        logger.info(f"\n{i+1}. {msg_type}")
        
        if hasattr(msg, "content"):
            content = str(msg.content)[:150]
            if content:
                logger.info(f"   Content: {content}...")
        
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                logger.info(f"   🔧 Tool: {tc['name']}")
                logger.info(f"      Args: {tc['args']}")
        
        if hasattr(msg, "name"):
            logger.info(f"   📦 Tool name: {msg.name}")
    
    logger.info("\n" + "="*60 + "\n")



