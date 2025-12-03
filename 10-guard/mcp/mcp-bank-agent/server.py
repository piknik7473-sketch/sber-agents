#!/usr/bin/env python3
"""
Bank Agent MCP Server

Предоставляет два инструмента для банковского агента:
1. search_products - поиск актуальных продуктов банка (вклады, кредиты, карты)
2. currency_converter - конвертация валют по курсам ЦБ РФ

Транспорт: streamable-http (HTTP MCP server)
Порт: 8000 (по умолчанию для FastMCP)
"""
import json
import logging
import os
from pathlib import Path
from typing import Annotated, Literal
import requests
from pydantic import Field

from mcp.server.fastmcp import FastMCP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp-bank-agent")

# Path to the products database
PRODUCTS_DB_PATH = Path(__file__).parent / "data" / "bank_products.json"

# CBR API endpoint
CBR_API_URL = "https://www.cbr-xml-daily.ru/latest.js"

# Mock номер карты для демонстрации (константа)
MOCK_CARD_NUMBER = "5105-1051-0510-5100"


def load_products() -> list[dict]:
    """Загрузка продуктов банка из JSON файла."""
    try:
        if not PRODUCTS_DB_PATH.exists():
            logger.error(f"Products database not found at {PRODUCTS_DB_PATH}")
            return []
        
        with open(PRODUCTS_DB_PATH, 'r', encoding='utf-8') as f:
            products = json.load(f)
        
        logger.info(f"Loaded {len(products)} products from database")
        return products
    except Exception as e:
        logger.error(f"Error loading products: {e}")
        return []


def filter_products(
    products: list[dict],
    product_type: str | None = None,
    keyword: str | None = None,
    min_amount: int | None = None,
    max_amount: int | None = None,
    min_rate: float | None = None,
    max_rate: float | None = None,
    currency: str | None = None
) -> list[dict]:
    """
    Фильтрация продуктов по параметрам
    
    Использует list comprehension для простоты (следуя принципу KISS).
    """
    filtered = products
    
    # Фильтр по типу продукта
    if product_type:
        filtered = [p for p in filtered if p.get('product_type') == product_type]
    
    # Поиск по ключевому слову (в названии и описании)
    if keyword:
        keyword_lower = keyword.lower()
        filtered = [
            p for p in filtered
            if keyword_lower in p.get('name', '').lower() or 
               keyword_lower in p.get('description', '').lower()
        ]
    
    # Фильтр по минимальной сумме
    if min_amount is not None:
        filtered = [p for p in filtered if p.get('amount_min', 0) <= min_amount]
    
    # Фильтр по максимальной сумме
    if max_amount is not None:
        filtered = [p for p in filtered if p.get('amount_max', float('inf')) >= max_amount]
    
    # Фильтр по минимальной ставке
    if min_rate is not None:
        filtered = [p for p in filtered if p.get('rate_max', 0) >= min_rate]
    
    # Фильтр по максимальной ставке
    if max_rate is not None:
        filtered = [p for p in filtered if p.get('rate_min', float('inf')) <= max_rate]
    
    # Фильтр по валюте
    if currency:
        filtered = [p for p in filtered if currency in p.get('currency', '')]
    
    return filtered


def format_products(products: list[dict], limit: int = 10) -> str:
    """
    Форматирование списка продуктов для агента
    
    Возвращает топ-N продуктов с основной информацией.
    """
    if not products:
        return "Продукты не найдены по заданным критериям."
    
    # Ограничиваем количество результатов
    products = products[:limit]
    
    result = f"Найдено {len(products)} продукт(ов):\n\n"
    
    for i, product in enumerate(products, 1):
        result += f"{i}. {product.get('name')}\n"
        result += f"   Описание: {product.get('description')}\n"
        
        # Ставка (для вкладов и кредитов)
        rate_min = product.get('rate_min', 0)
        rate_max = product.get('rate_max', 0)
        if rate_min > 0 or rate_max > 0:
            if rate_min == rate_max:
                result += f"   Ставка: {rate_min} процентов годовых\n"
            else:
                result += f"   Ставка: от {rate_min} до {rate_max} процентов годовых\n"
        
        # Сумма
        amount_min = product.get('amount_min', 0)
        amount_max = product.get('amount_max', 0)
        if amount_min > 0 or amount_max > 0:
            if amount_max > 0:
                result += f"   Сумма: от {amount_min:,} до {amount_max:,} {product.get('currency', 'RUB')}\n"
            else:
                result += f"   Сумма: от {amount_min:,} {product.get('currency', 'RUB')}\n"
        
        # Срок
        term = product.get('term_months', '')
        if term:
            result += f"   Срок: {term} месяцев\n"
        
        # Особенности
        features = product.get('features', [])
        if features:
            result += f"   Особенности: {', '.join(features)}\n"
        
        result += "\n"
    
    return result


def get_exchange_rates() -> dict:
    """
    Получение курсов валют от ЦБ РФ
    
    API возвращает курсы относительно рубля (base: RUB).
    Например: {"USD": 0.0124} означает 1 RUB = 0.0124 USD (или 1 USD ≈ 80.6 RUB)
    """
    try:
        response = requests.get(CBR_API_URL, timeout=5)
        response.raise_for_status()
        data = response.json()
        return data.get('rates', {})
    except requests.RequestException as e:
        logger.error(f"Error fetching exchange rates: {e}")
        return {}


def convert_currency(
    from_currency: str,
    to_currency: str,
    amount: float | None,
    rates: dict
) -> tuple[float | None, str]:
    """
    Конвертация валюты через рубль
    
    Логика:
    - RUB → другая валюта: amount * rates[to_currency]
    - другая валюта → RUB: amount / rates[from_currency]
    - валюта1 → валюта2: amount / rates[from] * rates[to] (через рубли)
    
    Returns:
        (converted_amount, formatted_string)
    """
    if not rates:
        return None, "Не удалось получить курсы валют от ЦБ РФ"
    
    # Проверка поддержки валют
    if from_currency != "RUB" and from_currency not in rates:
        return None, f"Валюта {from_currency} не поддерживается"
    
    if to_currency != "RUB" and to_currency not in rates:
        return None, f"Валюта {to_currency} не поддерживается"
    
    # Одинаковые валюты
    if from_currency == to_currency:
        rate_str = f"1 {from_currency} = 1 {to_currency}"
        if amount:
            return amount, f"{amount:,.2f} {from_currency} = {amount:,.2f} {to_currency}"
        return 1.0, rate_str
    
    # Конвертация через рубль
    if from_currency == "RUB":
        # RUB → другая валюта
        rate = rates[to_currency]
        rate_str = f"1 RUB = {rate:.6f} {to_currency} (или 1 {to_currency} ≈ {1/rate:.2f} RUB)"
        if amount:
            converted = amount * rate
            return converted, f"{amount:,.2f} RUB = {converted:,.2f} {to_currency}\n\nТекущий курс: {rate_str}"
        return rate, rate_str
    
    elif to_currency == "RUB":
        # другая валюта → RUB
        rate = rates[from_currency]
        rate_str = f"1 {from_currency} = {1/rate:.2f} RUB (или 1 RUB = {rate:.6f} {from_currency})"
        if amount:
            converted = amount / rate
            return converted, f"{amount:,.2f} {from_currency} = {converted:,.2f} RUB\n\nТекущий курс: {rate_str}"
        return 1/rate, rate_str
    
    else:
        # валюта1 → валюта2 (через рубль)
        rate_from = rates[from_currency]  # from → RUB
        rate_to = rates[to_currency]      # RUB → to
        rate = (1 / rate_from) * rate_to  # итоговый курс from → to
        
        rate_str = f"1 {from_currency} = {rate:.4f} {to_currency}"
        if amount:
            converted = amount * rate
            return converted, f"{amount:,.2f} {from_currency} = {converted:,.2f} {to_currency}\n\nТекущий курс: {rate_str}"
        return rate, rate_str


def calculate_simple_interest(
    amount: float,
    rate: float,
    term_months: int
) -> tuple[float, float]:
    """
    Расчет простого процента (без капитализации)
    
    Формула: доход = сумма * ставка * (месяцы / 12)
    
    Args:
        amount: начальная сумма вклада
        rate: годовая процентная ставка
        term_months: срок вклада в месяцах
    
    Returns:
        (income, total) - доход и итоговая сумма
    """
    income = amount * (rate / 100) * (term_months / 12)
    total = amount + income
    return income, total


def calculate_compound_interest(
    amount: float,
    rate: float,
    term_months: int,
    capitalization_months: int = 1
) -> tuple[float, float, list]:
    """
    Расчет сложного процента с капитализацией
    
    Логика: начисляем проценты каждые capitalization_months месяцев
    и добавляем их к основной сумме для следующего периода
    
    Args:
        amount: начальная сумма
        rate: годовая ставка в процентах
        term_months: срок вклада в месяцах
        capitalization_months: период капитализации (1, 3, 6, 12)
    
    Returns:
        (income, total, breakdown) - доход, итоговая сумма, помесячная разбивка
    """
    current_amount = amount
    breakdown = []
    
    # Начисляем проценты пошагово
    periods = term_months // capitalization_months
    remaining_months = term_months % capitalization_months
    
    for period in range(periods):
        period_income = current_amount * (rate / 100) * (capitalization_months / 12)
        current_amount += period_income
        breakdown.append({
            "period": period + 1,
            "months": capitalization_months,
            "income": period_income,
            "total": current_amount
        })
    
    # Остаток месяцев (если есть)
    if remaining_months > 0:
        period_income = current_amount * (rate / 100) * (remaining_months / 12)
        current_amount += period_income
        breakdown.append({
            "period": periods + 1,
            "months": remaining_months,
            "income": period_income,
            "total": current_amount
        })
    
    total_income = current_amount - amount
    return total_income, current_amount, breakdown


def calculate_tax(income: float) -> float:
    """
    Расчет НДФЛ на доход с вклада
    
    По закону РФ: налог 13% на доход свыше 150,000₽ за год
    Для упрощения: применяем к общему доходу независимо от срока
    
    Args:
        income: доход по вкладу
    
    Returns:
        сумма налога
    """
    if income <= 150_000:
        return 0.0
    return (income - 150_000) * 0.13


def format_deposit_calculation(
    amount: float,
    rate: float,
    term_months: int,
    income: float,
    total: float,
    calculation_type: str,
    tax: float = 0,
    breakdown: list = None,
    detailed: bool = False
) -> str:
    """
    Форматирование результата расчета для агента
    
    Args:
        amount: начальная сумма
        rate: процентная ставка
        term_months: срок в месяцах
        income: доход
        total: итоговая сумма
        calculation_type: тип расчета (simple/compound)
        tax: сумма налога
        breakdown: помесячная разбивка
        detailed: показывать детальную разбивку
    
    Returns:
        форматированная строка с результатом
    """
    result = f"Расчет доходности вклада\n\n"
    result += f"Начальная сумма: {amount:,.0f} руб.\n"
    result += f"Ставка: {rate} процентов годовых\n"
    result += f"Срок: {term_months} мес.\n"
    result += f"Тип: {'с капитализацией' if calculation_type == 'compound' else 'без капитализации'}\n\n"
    
    result += f"Результат:\n"
    result += f"Доход: {income:,.2f} руб.\n"
    
    if tax > 0:
        result += f"Налог (НДФЛ 13 процентов): {tax:,.2f} руб.\n"
        result += f"Чистый доход: {income - tax:,.2f} руб.\n"
    
    result += f"Итоговая сумма: {total:,.2f} руб.\n"
    
    # Детализированная разбивка для compound
    if detailed and breakdown:
        result += f"\nПомесячная разбивка:\n"
        for b in breakdown:
            result += f"Период {b['period']} ({b['months']} мес.): +{b['income']:,.2f} руб. = {b['total']:,.2f} руб.\n"
    
    return result


# Create FastMCP server
mcp = FastMCP("mcp-bank-agent", dependencies=["requests>=2.31.0"])


@mcp.tool(
    name="search_products",
    description="Универсальный поиск актуальных продуктов банка (вклады, кредиты, карты, счета) с гибкой фильтрацией",
)
async def search_products(
    product_type: Annotated[
        Literal["deposit", "credit", "debit_card", "credit_card", "account"] | None,
        Field(
            description="Тип продукта для фильтрации",
        )
    ] = None,
    keyword: Annotated[
        str | None,
        Field(
            description="Ключевое слово для поиска в названии и описании продукта",
            min_length=2,
            max_length=100,
            examples=["вклад", "кредит", "карта", "кешбэк"]
        )
    ] = None,
    min_amount: Annotated[
        int | None,
        Field(
            description="Минимальная сумма (ищет продукты доступные от этой суммы)",
            ge=0,
            examples=[10000, 50000, 100000]
        )
    ] = None,
    max_amount: Annotated[
        int | None,
        Field(
            description="Максимальная сумма (ищет продукты доступные до этой суммы)",
            ge=0,
            examples=[1000000, 5000000]
        )
    ] = None,
    min_rate: Annotated[
        float | None,
        Field(
            description="Минимальная процентная ставка (для вкладов и кредитов)",
            ge=0,
            le=100,
            examples=[10.0, 15.0, 20.0]
        )
    ] = None,
    max_rate: Annotated[
        float | None,
        Field(
            description="Максимальная процентная ставка (для вкладов и кредитов)",
            ge=0,
            le=100,
            examples=[15.0, 20.0, 25.0]
        )
    ] = None,
    currency: Annotated[
        Literal["RUB", "USD", "EUR"] | None,
        Field(
            description="Валюта продукта"
        )
    ] = None
) -> str:
    """
    Поиск актуальных банковских продуктов с фильтрацией
    
    Этот инструмент ищет текущие продукты банка с актуальными ставками и условиями.
    В отличие от rag_search (статические PDF), здесь динамические данные о продуктах.
    
    Args:
        product_type: Тип продукта (вклад, кредит, карта, счёт)
        keyword: Поиск по ключевому слову
        min_amount: Минимальная сумма
        max_amount: Максимальная сумма
        min_rate: Минимальная ставка
        max_rate: Максимальная ставка
        currency: Валюта
    
    Returns:
        Форматированный список найденных продуктов (топ-10)
    """
    logger.info(f"search_products called with: type={product_type}, keyword={keyword}, "
                f"amount={min_amount}-{max_amount}, rate={min_rate}-{max_rate}, currency={currency}")
    
    # Загружаем продукты
    products = load_products()
    if not products:
        return "Не удалось загрузить базу продуктов банка"
    
    # Фильтруем
    filtered = filter_products(
        products,
        product_type=product_type,
        keyword=keyword,
        min_amount=min_amount,
        max_amount=max_amount,
        min_rate=min_rate,
        max_rate=max_rate,
        currency=currency
    )
    
    # Форматируем результат
    return format_products(filtered)


@mcp.tool(
    name="currency_converter",
    description="Конвертация валют по актуальным курсам ЦБ РФ с поддержкой всех основных валют",
)
async def currency_converter(
    from_currency: Annotated[
        Literal["RUB", "USD", "EUR", "CNY", "GBP", "CHF", "JPY", "TRY"],
        Field(
            description="Исходная валюта для конвертации"
        )
    ] = "USD",
    to_currency: Annotated[
        Literal["RUB", "USD", "EUR", "CNY", "GBP", "CHF", "JPY", "TRY"],
        Field(
            description="Целевая валюта для конвертации"
        )
    ] = "RUB",
    amount: Annotated[
        float | None,
        Field(
            description="Сумма для конвертации (если не указана, вернется только курс)",
            ge=0,
            examples=[100, 1000, 10000]
        )
    ] = None
) -> str:
    """
    Конвертация валют по актуальным курсам ЦБ РФ
    
    Поддерживает конвертацию между любыми валютами (не только с рублями).
    Данные обновляются ежедневно ЦБ РФ.
    
    Args:
        from_currency: Исходная валюта
        to_currency: Целевая валюта
        amount: Сумма для конвертации (опционально)
    
    Returns:
        Результат конвертации с текущим курсом
    """
    logger.info(f"currency_converter called: {amount} {from_currency} -> {to_currency}")
    
    # Получаем актуальные курсы
    rates = get_exchange_rates()
    
    # Конвертируем
    converted_amount, result_str = convert_currency(from_currency, to_currency, amount, rates)
    
    if converted_amount is None:
        return result_str  # Сообщение об ошибке
    
    return result_str


@mcp.tool(
    name="deposit_income_calculator",
    description="Расчет доходности по вкладу с учетом простого или сложного процента и опциональных налогов",
)
async def deposit_income_calculator(
    amount: Annotated[
        float,
        Field(description="Сумма вклада в рублях", ge=1000, examples=[100000, 500000, 1000000])
    ],
    rate: Annotated[
        float,
        Field(description="Процентная ставка годовых", ge=0.1, le=100, examples=[12.0, 15.5, 18.0])
    ],
    term_months: Annotated[
        int,
        Field(description="Срок вклада в месяцах", ge=1, le=120, examples=[6, 12, 24, 36])
    ],
    calculation_type: Annotated[
        Literal["simple", "compound"],
        Field(description="Тип расчета: simple (простой процент) или compound (с капитализацией)")
    ] = "simple",
    capitalization_months: Annotated[
        Literal[1, 3, 6, 12] | None,
        Field(description="Период капитализации в месяцах (только для compound)")
    ] = None,
    include_tax: Annotated[
        bool,
        Field(description="Учитывать НДФЛ 13% на доход свыше 150,000₽")
    ] = False,
    detailed: Annotated[
        bool,
        Field(description="Детализированный расчет с помесячной разбивкой (только для compound)")
    ] = False
) -> str:
    """
    Расчет доходности по вкладу
    
    Поддерживает два типа расчета:
    - simple: простой процент без капитализации
    - compound: сложный процент с капитализацией
    
    Опционально учитывает налоги (НДФЛ 13% на доход свыше 150 тыс.)
    
    Args:
        amount: Сумма вклада
        rate: Процентная ставка годовых
        term_months: Срок вклада в месяцах
        calculation_type: Тип расчета (simple/compound)
        capitalization_months: Период капитализации для compound
        include_tax: Учитывать налоги
        detailed: Показывать детальную разбивку
    
    Returns:
        Форматированный результат расчета
    """
    logger.info(f"deposit_income_calculator called: amount={amount}, rate={rate}, "
                f"term={term_months}, type={calculation_type}, tax={include_tax}")
    
    # Валидация capitalization_months для compound
    if calculation_type == "compound" and capitalization_months is None:
        capitalization_months = 1  # По умолчанию ежемесячная капитализация
    
    # Расчет
    if calculation_type == "simple":
        income, total = calculate_simple_interest(amount, rate, term_months)
        breakdown = None
    else:  # compound
        income, total, breakdown = calculate_compound_interest(
            amount, rate, term_months, capitalization_months
        )
    
    # Налоги
    tax = 0.0
    if include_tax:
        tax = calculate_tax(income)
        total = total - tax  # Вычитаем налог из итоговой суммы
    
    # Форматирование результата
    result = format_deposit_calculation(
        amount, rate, term_months, income, total,
        calculation_type, tax, breakdown, detailed
    )
    
    return result


@mcp.tool(
    name="open_credit_card",
    description="Открытие новой дебетовой или кредитной карты для клиента",
)
async def open_credit_card(
    card_type: Annotated[
        Literal["debit", "credit"],
        Field(description="Тип карты: debit (дебетовая) или credit (кредитная)")
    ],
    client_name: Annotated[
        str,
        Field(
            description="Имя владельца карты латиницей (как будет напечатано на карте)",
            min_length=3,
            max_length=26,
            examples=["IVAN PETROV", "MARIA KOZLOVA", "PETR SIDOROV"]
        )
    ]
) -> str:
    """
    Открытие новой дебетовой или кредитной карты
       
    Возвращает данные карты БЕЗ CVV кода (для безопасности).
    CVV будет отправлен клиенту отдельным СМС.
    
    Args:
        card_type: Тип карты (debit - дебетовая, credit - кредитная)
        client_name: Имя на карте латиницей (будет автоматически приведено к верхнему регистру)
    
    Returns:
        Форматированная информация о новой карте
    """
    logger.info(f"🔐 open_credit_card called: type={card_type}, client={client_name}")
    
    # Форматируем имя на карте (всегда в верхнем регистре как на настоящих картах)
    card_holder_name = client_name.upper()
    
    # Определяем платежную систему по первой цифре номера карты
    first_digit = MOCK_CARD_NUMBER[0]
    if first_digit == "4":
        payment_system = "Visa"
    elif first_digit == "5":
        payment_system = "Mastercard"
    elif first_digit == "2":
        payment_system = "МИР"
    else:
        payment_system = "Unknown"
    
    # Генерируем срок действия: 3 года с текущей даты
    from datetime import datetime, timedelta
    expiration_date = (datetime.now() + timedelta(days=3*365)).strftime("%m/%y")
    
    # Форматируем тип карты для вывода
    card_type_ru = "дебетовая" if card_type == "debit" else "кредитная"
    
    # Формируем результат (БЕЗ CVV!)
    result = (
        "✅ Карта успешно открыта!\n\n"
        "📋 Детали карты:\n"
        f"   Тип: {card_type_ru.capitalize()} карта\n"
        f"   Платежная система: {payment_system}\n"
        f"   Номер карты: {MOCK_CARD_NUMBER}\n"
        f"   Срок действия: {expiration_date}\n"
        f"   Владелец: {card_holder_name}\n"
        "   Статус: Активна\n\n"
        "💳 Карта готова к использованию.\n"
        "🔐 CVV код будет отправлен на номер телефона клиента отдельным СМС-сообщением.\n"
    )
    
    logger.info(f"✓ Card opened successfully: {card_type} for {card_holder_name}")
    
    return result

@mcp.tool(
    name="open_deposit",
    description="Открытие нового вклада для клиента",
)
async def open_deposit(
    client_name: Annotated[
        str,
        Field(
            description="Имя клиента (владельца вклада)",
            min_length=2,
            max_length=100,
            examples=["Иван Петров", "Мария Козлова", "Петр Сидоров"]
        )
    ],
    amount: Annotated[
        float,
        Field(
            description="Сумма вклада в рублях",
            ge=1000,
            examples=[100000, 500000, 1000000]
        )
    ],
    rate: Annotated[
        float,
        Field(
            description="Процентная ставка годовых",
            ge=0.1,
            le=100,
            examples=[12.0, 15.5, 18.0]
        )
    ],
    term_months: Annotated[
        int,
        Field(
            description="Срок вклада в месяцах",
            ge=1,
            le=120,
            examples=[6, 12, 24, 36]
        )
    ],
    calculation_type: Annotated[
        Literal["simple", "compound"],
        Field(description="Тип расчета: simple (простой процент) или compound (с капитализацией)")
    ] = "simple",
    capitalization_months: Annotated[
        Literal[1, 3, 6, 12] | None,
        Field(description="Период капитализации в месяцах (только для compound)")
    ] = None
) -> str:
    """
    Открытие нового вклада
    
    Возвращает номер договора и детали вклада с расчетом доходности.
    
    Args:
        client_name: Имя клиента
        amount: Сумма вклада в рублях
        rate: Процентная ставка годовых
        term_months: Срок вклада в месяцах
        calculation_type: Тип расчета (simple/compound)
        capitalization_months: Период капитализации для compound
    
    Returns:
        Форматированная информация о вкладе с номером договора
    """
    logger.info(f"💰 open_deposit called: client={client_name}, amount={amount}, "
                f"rate={rate}, term={term_months}, type={calculation_type}")
    
    # Генерируем номер договора в формате DEP-YYYYMMDD-XXXX
    from datetime import datetime, timedelta
    import random
    now = datetime.now()
    contract_number = f"DEP-{now.strftime('%Y%m%d')}-{random.randint(1000, 9999)}"
    
    # Валидация capitalization_months для compound
    if calculation_type == "compound" and capitalization_months is None:
        capitalization_months = 1  # По умолчанию ежемесячная капитализация
    
    # Расчет доходности
    if calculation_type == "simple":
        income, total = calculate_simple_interest(amount, rate, term_months)
    else:  # compound
        income, total, _ = calculate_compound_interest(
            amount, rate, term_months, capitalization_months
        )
    
    # Вычисляем дату окончания вклада
    end_date = (now + timedelta(days=term_months * 30)).strftime("%d.%m.%Y")
    
    # Форматируем тип вклада для вывода (аналогично open_credit_card)
    deposit_type = "с капитализацией" if calculation_type == "compound" else "без капитализации"
    
    # Форматируем результат (аналогично open_credit_card - без форматирования чисел с запятыми)
    result = (
        "✅ Вклад успешно открыт!\n\n"
        "📋 Детали вклада:\n"
        f"   Номер договора: {contract_number}\n"
        f"   Клиент: {client_name}\n"
        f"   Сумма: {int(amount)} руб.\n"
        f"   Ставка: {rate} процентов годовых\n"
        f"   Срок: {term_months} мес.\n"
        f"   Тип: {deposit_type}\n"
        f"   Дата окончания: {end_date}\n"
        f"   Ожидаемый доход: {income:.2f} руб.\n"
        f"   Итоговая сумма: {total:.2f} руб.\n\n"
        "💰 Вклад активен и начал начисление процентов.\n"
    )
    
    logger.info(f"✓ Deposit opened: {contract_number} for {client_name}")
    return result

if __name__ == "__main__":
    logger.info("Starting Bank Agent MCP Server...")
    logger.info(f"Products database: {PRODUCTS_DB_PATH}")
    logger.info(f"Currency API: {CBR_API_URL}")
    
    # Проверяем наличие базы продуктов
    if not PRODUCTS_DB_PATH.exists():
        logger.error(f"Products database not found at {PRODUCTS_DB_PATH}")
        logger.error("Please create data/bank_products.json before starting the server")
        exit(1)
    
    # Получаем порт из переменной окружения (по умолчанию 8000)
    port = int(os.getenv("PORT", "8000"))
    logger.info(f"Server will be available at: http://localhost:{port}/mcp")
    
    # Запускаем сервер
    mcp.run(transport="streamable-http")

