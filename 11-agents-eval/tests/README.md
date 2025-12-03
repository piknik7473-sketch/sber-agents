# E2E Agent Tests

Тесты агента разделены на две группы по типу evaluator'ов:

## 📊 Типы тестов

### 1. Детерминированные тесты (`test_agent_deterministic.py`)

**Характеристики:**
- ✅ Быстрые (< 10 сек)
- ✅ Надежные (детерминированные)
- ✅ Дешевые (только вызовы агента)
- ✅ Работают с любой моделью

**Evaluators:**
- `superset` - агент должен вызвать минимум указанные инструменты
- `subset` - агент не должен вызывать лишние инструменты
- `exact` - агент должен вызвать точно указанные инструменты

**Запуск:**
```bash
make test-deterministic              # Через Makefile
pytest tests/test_agent_deterministic.py -v  # Через pytest
pytest -m deterministic -v           # Через marker
```

### 2. LLM-as-Judge тесты (`test_agent_llm_judge.py`)

**Характеристики:**
- ⏱️ Медленные (20-60 сек)
- 🎲 Недетерминированные
- 💰 Дорогие (LLM для оценки)
- ⚠️ Требуют хорошую модель (gpt-4o, claude-3.5-sonnet)

**Evaluators:**
- LLM оценивает разумность траектории целиком
- Проверяет логику выбора инструментов
- Оценивает корректность обработки HITL

**Запуск:**
```bash
make test-llm-judge                  # Через Makefile
pytest tests/test_agent_llm_judge.py -v  # Через pytest
pytest -m llm_judge -v               # Через marker
```

## 🚀 Команды запуска

```bash
# Быстрые тесты (рекомендуется для CI/CD)
make test-deterministic

# Медленные тесты (перед релизом)
make test-llm-judge

# Все тесты
make test-all
```

## 📝 Конфигурация

### Модели для evaluators

Настройки в `.env`:
```bash
# Модель для основного агента (любая с tool calling)
MODEL=openai/gpt-4o-mini

# Модель для LLM-as-Judge (рекомендуется gpt-4o или claude-3.5-sonnet)
AGENTEVALS_LLM_MODEL=openai:gpt-4o
```

### Pytest markers

Настройки в `pytest.ini`:
```ini
[pytest]
markers =
    deterministic: tests that use deterministic match-based evaluators (fast, reliable)
    llm_judge: tests that use LLM-as-Judge evaluators (slow, requires good model)
```

## 🎯 Рекомендации

**Для разработки:**
- Запускайте `make test-deterministic` после каждого изменения
- Быстрая обратная связь, работают с любой моделью

**Перед коммитом:**
- Запускайте `make test-all` для полной проверки
- Убедитесь что LLM-as-Judge тесты проходят

**В CI/CD:**
- Запускайте только `make test-deterministic`
- LLM-as-Judge тесты можно запускать отдельным джобом

## 📚 Структура

```
tests/
├── README.md                      # Этот файл
├── conftest.py                    # Фикстуры (agent_fixture)
├── helpers.py                     # Утилиты (extract_trajectory, print_trajectory)
├── test_agent_deterministic.py   # Детерминированные тесты (match-based)
└── test_agent_llm_judge.py       # LLM-as-Judge тесты
```

## 🔍 Добавление новых тестов

### Детерминированный тест

```python
@pytest.mark.deterministic
@pytest.mark.asyncio
async def test_my_scenario(agent_fixture):
    """Описание теста"""
    agent = agent_fixture
    
    # Запрос
    user_message = "..."
    
    # Получение траектории
    actual = await extract_trajectory(agent, "test_id", user_message)
    
    # Референсная траектория
    reference = [...]
    
    # Создание evaluator
    evaluator = create_trajectory_match_evaluator(
        trajectory_match_mode="superset",  # superset/subset/exact
        tool_args_match_mode="ignore"      # ignore/exact
    )
    
    # Оценка
    result = evaluator(outputs=actual, reference_outputs=reference)
    assert result["score"], f"Test failed: {result.get('comment')}"
```

### LLM-as-Judge тест

```python
@pytest.mark.llm_judge
@pytest.mark.asyncio
async def test_my_complex_scenario(agent_fixture):
    """Описание теста"""
    agent = agent_fixture
    
    # Получение траектории
    actual = await extract_trajectory(agent, "test_id", "...")
    
    # Создание evaluator
    evaluator = create_async_trajectory_llm_as_judge(
        prompt=TRAJECTORY_ACCURACY_PROMPT,
        model=config.AGENTEVALS_LLM_MODEL
    )
    
    # Оценка
    result = await evaluator(outputs=actual)
    assert result["score"] > 0.7, f"Score: {result['score']}"
```

## ⚙️ Troubleshooting

**Проблема:** LLM-as-Judge тесты падают с низким score

**Решение:**
1. Проверьте модель в `.env`: `AGENTEVALS_LLM_MODEL=openai:gpt-4o`
2. Убедитесь что модель агента корректно работает с tool calling
3. Проверьте траекторию через `print_trajectory()`

**Проблема:** Детерминированные тесты падают из-за invalid tool calls

**Решение:**
1. Модель плохо работает с tool calling
2. Используйте более надежную модель: `MODEL=openai/gpt-4o-mini`
3. Проверьте что MCP сервер запущен: `make run-mcp-bank`

