from pydantic import BaseModel, Field, field_validator, ConfigDict
from datetime import date, time
from enum import Enum
from typing import Optional

class TransactionType(str, Enum):
    INCOME = "income"      # доход
    EXPENSE = "expense"    # расход

class TransactionFrequency(str, Enum):
    DAILY = "daily"           # повседневные
    PERIODIC = "periodic"     # периодические
    ONE_TIME = "one_time"     # разовые

class Transaction(BaseModel):
    date: date                           # дата транзакции
    time: Optional[str] = Field(default=None, description="Время в формате HH:MM или HH:MM:SS")  # время как строка в JSON
    type: TransactionType                # доход/расход
    amount: float = Field(gt=0)          # сумма (строго положительная)
    frequency: TransactionFrequency       # тип (повседневные, периодические, разовые)
    category: str                        # категория (продукты, рестораны, такси и т.д.)
    description: str = ""                # описание транзакции (подробная информация о товарах, услугах, источнике, контрагенте и т.п.)
    
    @field_validator('time', mode='after')
    @classmethod
    def parse_time(cls, v):
        """Преобразует строку времени в объект time после валидации."""
        if v is None or v == "":
            return None
        if isinstance(v, time):
            return v
        if isinstance(v, str):
            v = v.strip()
            if not v:
                return None
            try:
                parts = v.split(':')
                if len(parts) == 2:
                    hour, minute = int(parts[0]), int(parts[1])
                    if 0 <= hour < 24 and 0 <= minute < 60:
                        return time(hour, minute)
                elif len(parts) == 3:
                    hour, minute, second = int(parts[0]), int(parts[1]), int(parts[2])
                    if 0 <= hour < 24 and 0 <= minute < 60 and 0 <= second < 60:
                        return time(hour, minute, second)
            except (ValueError, IndexError, TypeError):
                return None
        return None
    
    @property
    def time_obj(self) -> Optional[time]:
        """Возвращает время как объект time."""
        return self.time if isinstance(self.time, time) else None

class TransactionResponse(BaseModel):
    transactions: list[Transaction]  # список транзакций (всегда должен быть, пустой [] если не найдено)
    answer: str                     # текстовый ответ пользователю (обязателен)

