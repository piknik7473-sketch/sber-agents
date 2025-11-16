import json
import logging
from pathlib import Path
from typing import Dict, List
from models import Transaction

# Путь к корню проекта
PROJECT_ROOT = Path(__file__).parent.parent

logger = logging.getLogger(__name__)

# Путь к файлу с транзакциями
TRANSACTIONS_FILE = PROJECT_ROOT / "data" / "transactions.json"

def ensure_data_dir():
    """Создает директорию для данных, если её нет."""
    TRANSACTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)

def save_transactions(transactions: Dict[int, List[Transaction]]):
    """
    Сохраняет транзакции в JSON файл.
    
    Args:
        transactions: Словарь {chat_id: [список транзакций]}
    """
    try:
        ensure_data_dir()
        
        # Преобразуем транзакции в JSON-совместимый формат
        transactions_dict = {}
        for chat_id, txns in transactions.items():
            transactions_dict[str(chat_id)] = [t.model_dump(mode='json') for t in txns]
        
        # Сохраняем в файл
        with open(TRANSACTIONS_FILE, 'w', encoding='utf-8') as f:
            json.dump(transactions_dict, f, ensure_ascii=False, indent=2, default=str)
        
        total_count = sum(len(txns) for txns in transactions.values())
        logger.info(f"Saved {total_count} transactions for {len(transactions)} users to {TRANSACTIONS_FILE}")
    
    except Exception as e:
        logger.error(f"Error saving transactions: {e}", exc_info=True)

def load_transactions() -> Dict[int, List[Transaction]]:
    """
    Загружает транзакции из JSON файла.
    
    Returns:
        Словарь {chat_id: [список транзакций]}
    """
    transactions: Dict[int, List[Transaction]] = {}
    
    try:
        if not TRANSACTIONS_FILE.exists():
            logger.info(f"Transactions file not found: {TRANSACTIONS_FILE}, starting with empty transactions")
            return transactions
        
        with open(TRANSACTIONS_FILE, 'r', encoding='utf-8') as f:
            transactions_dict = json.load(f)
        
        # Преобразуем обратно в объекты Transaction
        for chat_id_str, txns_data in transactions_dict.items():
            chat_id = int(chat_id_str)
            transactions[chat_id] = [
                Transaction.model_validate(txn_data) for txn_data in txns_data
            ]
        
        total_count = sum(len(txns) for txns in transactions.values())
        logger.info(f"Loaded {total_count} transactions for {len(transactions)} users from {TRANSACTIONS_FILE}")
    
    except Exception as e:
        logger.error(f"Error loading transactions: {e}", exc_info=True)
    
    return transactions

def add_transactions(chat_id: int, new_transactions: List[Transaction], all_transactions: Dict[int, List[Transaction]]):
    """
    Добавляет новые транзакции и сохраняет в файл.
    
    Args:
        chat_id: ID чата
        new_transactions: Новые транзакции для добавления
        all_transactions: Словарь всех транзакций
    """
    if chat_id not in all_transactions:
        all_transactions[chat_id] = []
    
    all_transactions[chat_id].extend(new_transactions)
    save_transactions(all_transactions)
    
    logger.info(f"Added {len(new_transactions)} transactions for chat {chat_id}, total: {len(all_transactions[chat_id])}")

