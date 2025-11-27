"""
Инструменты для ReAct агента

Инструменты - это функции, которые агент может вызывать для получения информации.
Декоратор @tool из LangChain автоматически создает описание для LLM.
"""
import json
import logging
from langchain_core.tools import tool
import rag

logger = logging.getLogger(__name__)

@tool
def rag_search(query: str) -> str:
    """
    Ищет информацию в документах Сбербанка (условия кредитов, вкладов и других банковских продуктов).
    
    Возвращает JSON со списком источников, где каждый источник содержит:
    - source: имя файла
    - page: номер страницы (только для PDF)
    - page_content: текст документа
    """
    try:
        # Получаем релевантные документы через RAG (retrieval + reranking)
        documents = rag.retrieve_documents(query)
        
        if not documents:
            return json.dumps({"sources": []}, ensure_ascii=False)
        
        # Формируем структурированный ответ для агента
        sources = []
        for doc in documents:
            source_data = {
                "source": doc.metadata.get("source", "Unknown"),
                "page_content": doc.page_content  # Полный текст документа
            }
            # page только для PDF (у JSON документов его нет)
            if "page" in doc.metadata:
                source_data["page"] = doc.metadata["page"]
            sources.append(source_data)
        
        # ensure_ascii=False для корректной кириллицы
        return json.dumps({"sources": sources}, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"Error in rag_search: {e}", exc_info=True)
        return json.dumps({"sources": []}, ensure_ascii=False)

@tool
def currency_converter(amount: float, from_currency: str, to_currency: str) -> str:
    """
    Конвертирует сумму из одной валюты в другую.
    
    Args:
        amount: Сумма для конвертации
        from_currency: Исходная валюта (USD, EUR, RUB)
        to_currency: Целевая валюта (USD, EUR, RUB)
    
    Returns:
        Строка с результатом конвертации
    """
    try:
        # Фиксированные курсы валют (относительно RUB)
        exchange_rates = {
            "RUB": 1.0,
            "USD": 100.0,  # 1 USD = 100 RUB
            "EUR": 110.0,  # 1 EUR = 110 RUB
        }
        
        # Нормализуем названия валют к верхнему регистру
        from_currency = from_currency.upper()
        to_currency = to_currency.upper()
        
        # Проверяем, что валюты поддерживаются
        if from_currency not in exchange_rates:
            return f"Ошибка: валюта {from_currency} не поддерживается. Доступные валюты: {', '.join(exchange_rates.keys())}"
        
        if to_currency not in exchange_rates:
            return f"Ошибка: валюта {to_currency} не поддерживается. Доступные валюты: {', '.join(exchange_rates.keys())}"
        
        # Конвертируем через RUB как базовую валюту
        # Сначала конвертируем в RUB, затем в целевую валюту
        amount_in_rub = amount * exchange_rates[from_currency]
        result = amount_in_rub / exchange_rates[to_currency]
        
        return f"{amount} {from_currency} = {result:.2f} {to_currency}"
        
    except Exception as e:
        logger.error(f"Error in currency_converter: {e}", exc_info=True)
        return f"Ошибка при конвертации валют: {e}"