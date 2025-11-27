import os
from pathlib import Path
from dotenv import load_dotenv

# Путь к корню проекта (где находится pyproject.toml)
# config.py находится в src/, поэтому поднимаемся на уровень вверх
PROJECT_ROOT = Path(__file__).parent.parent
env_path = PROJECT_ROOT / ".env"

# Загружаем .env файл
# override=True означает, что переменные из .env будут перезаписывать системные переменные
result = load_dotenv(dotenv_path=env_path, override=True)

# Отладочная информация (только для диагностики)
import logging
logger = logging.getLogger(__name__)
if result:
    logger.info(f"✅ .env file loaded successfully from: {env_path}")
    # Проверяем, что файл существует и читаем его размер
    if env_path.exists():
        logger.info(f"   File size: {env_path.stat().st_size} bytes")
else:
    logger.warning(f"⚠️  Failed to load .env file from: {env_path}")
    if not env_path.exists():
        logger.warning(f"   File does not exist!")

class Config:
    # Используем .strip() для удаления пробелов в начале и конце значений
    # Для токена также удаляем все пробелы и невидимые символы
    _raw_telegram_token = os.getenv("TELEGRAM_TOKEN", "").strip()
    # Удаляем все пробелы, табы, переносы строк и другие whitespace символы
    if _raw_telegram_token:
        # Удаляем все whitespace символы (пробелы, табы, переносы строк и т.д.)
        import re
        TELEGRAM_TOKEN = re.sub(r'\s+', '', _raw_telegram_token)
    else:
        TELEGRAM_TOKEN = ""
    
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "").strip()
    MODEL = os.getenv("MODEL", "").strip()
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    DATA_DIR = os.getenv("DATA_DIR", "data")
    PROMPTS_DIR = os.getenv("PROMPTS_DIR", "prompts")
    AGENT_SYSTEM_PROMPT_FILE = os.getenv("AGENT_SYSTEM_PROMPT_FILE", "agent_system.txt")
    SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT")
    
    # Embeddings Configuration
    EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai")  # openai/huggingface
    HUGGINGFACE_EMBEDDING_MODEL = os.getenv("HUGGINGFACE_EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
    HUGGINGFACE_DEVICE = os.getenv("HUGGINGFACE_DEVICE", "cpu")  # cpu/cuda/mps
    
    # Retrieval Configuration
    RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "semantic")  # semantic/hybrid/hybrid_reranker
    SEMANTIC_RETRIEVER_K = int(os.getenv("SEMANTIC_RETRIEVER_K", "10"))
    BM25_RETRIEVER_K = int(os.getenv("BM25_RETRIEVER_K", "10"))
    ENSEMBLE_SEMANTIC_WEIGHT = float(os.getenv("ENSEMBLE_SEMANTIC_WEIGHT", "0.5"))
    ENSEMBLE_BM25_WEIGHT = float(os.getenv("ENSEMBLE_BM25_WEIGHT", "0.5"))
    
    # Cross-Encoder Reranking Configuration
    CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    RERANKER_TOP_K = int(os.getenv("RERANKER_TOP_K", "3"))
    
    # Отображение источников
    SHOW_SOURCES = os.getenv("SHOW_SOURCES", "false").lower() == "true"
    
    # LangSmith настройки
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
    # Поддержка обеих переменных для совместимости (стандартная - LANGSMITH_TRACING_V2)
    _tracing = os.getenv("LANGSMITH_TRACING_V2") or os.getenv("LANGSMITH_TRACING") or "false"
    LANGSMITH_TRACING_V2 = _tracing.lower() == "true"
    LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "rag-assistant")
    LANGSMITH_DATASET = os.getenv("LANGSMITH_DATASET", "06-rag-qa-dataset")
    
    # RAGAS evaluation настройки (фиксированные модели для единообразной оценки)
    RAGAS_LLM_MODEL = os.getenv("RAGAS_LLM_MODEL", "gpt-4o")
    RAGAS_EMBEDDING_MODEL = os.getenv("RAGAS_EMBEDDING_MODEL", "text-embedding-3-large")
    RAGAS_EMBEDDING_PROVIDER = os.getenv("RAGAS_EMBEDDING_PROVIDER", EMBEDDING_PROVIDER)  # По умолчанию = основному провайдеру
    # Для HuggingFace используем те же настройки что и для основных embeddings
    RAGAS_HUGGINGFACE_EMBEDDING_MODEL = os.getenv("RAGAS_HUGGINGFACE_EMBEDDING_MODEL", HUGGINGFACE_EMBEDDING_MODEL)
    RAGAS_HUGGINGFACE_DEVICE = os.getenv("RAGAS_HUGGINGFACE_DEVICE", HUGGINGFACE_DEVICE)
    
    @classmethod
    def load_prompt(cls, filename: str) -> str:
        """Загрузка промпта из файла"""
        # Если путь абсолютный, используем его как есть, иначе относительно корня проекта
        if os.path.isabs(cls.PROMPTS_DIR):
            prompt_path = Path(cls.PROMPTS_DIR) / filename
        else:
            prompt_path = PROJECT_ROOT / cls.PROMPTS_DIR / filename
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        return prompt_path.read_text(encoding='utf-8')
    
    @classmethod
    def get_data_dir(cls) -> Path:
        """Возвращает абсолютный путь к директории с данными"""
        if os.path.isabs(cls.DATA_DIR):
            return Path(cls.DATA_DIR)
        return PROJECT_ROOT / cls.DATA_DIR
    
    @classmethod
    def get_prompts_dir(cls) -> Path:
        """Возвращает абсолютный путь к директории с промптами"""
        if os.path.isabs(cls.PROMPTS_DIR):
            return Path(cls.PROMPTS_DIR)
        return PROJECT_ROOT / cls.PROMPTS_DIR
    
    @classmethod
    def validate(cls):
        """Валидация конфигурации"""
        # Валидация обязательных переменных
        required_vars = {
            "TELEGRAM_TOKEN": cls.TELEGRAM_TOKEN,
            "OPENAI_API_KEY": cls.OPENAI_API_KEY,
            "OPENAI_BASE_URL": cls.OPENAI_BASE_URL,
            "MODEL": cls.MODEL,
        }
        missing_vars = [var for var, value in required_vars.items() if not value]
        if missing_vars:
            # Проверяем, существует ли файл .env
            env_exists = env_path.exists()
            env_info = f"File exists: {env_exists}"
            if env_exists:
                env_info += f", size: {env_path.stat().st_size} bytes"
            
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing_vars)}\n"
                f"Please check your .env file at: {env_path}\n"
                f"{env_info}\n"
                f"Make sure the variables are set in the format: VARIABLE_NAME=value\n"
                f"Without spaces around the '=' sign."
            )
        
        # Валидация RETRIEVAL_MODE
        valid_retrieval_modes = ["semantic", "hybrid", "hybrid_reranker"]
        if cls.RETRIEVAL_MODE not in valid_retrieval_modes:
            raise ValueError(
                f"Invalid RETRIEVAL_MODE: {cls.RETRIEVAL_MODE}. "
                f"Must be one of: {', '.join(valid_retrieval_modes)}"
            )
        
        # Валидация EMBEDDING_PROVIDER
        valid_embedding_providers = ["openai", "huggingface"]
        if cls.EMBEDDING_PROVIDER not in valid_embedding_providers:
            raise ValueError(
                f"Invalid EMBEDDING_PROVIDER: {cls.EMBEDDING_PROVIDER}. "
                f"Must be one of: {', '.join(valid_embedding_providers)}"
            )
        
        # Валидация RAGAS_EMBEDDING_PROVIDER
        if cls.RAGAS_EMBEDDING_PROVIDER not in valid_embedding_providers:
            raise ValueError(
                f"Invalid RAGAS_EMBEDDING_PROVIDER: {cls.RAGAS_EMBEDDING_PROVIDER}. "
                f"Must be one of: {', '.join(valid_embedding_providers)}"
            )

config = Config()

# Отладочная информация для токена (после создания экземпляра)
logger.info("=" * 70)
logger.info("🔍 Token Debug Information:")
logger.info(f"   Raw token from env: {repr(config._raw_telegram_token[:20])}..." if config._raw_telegram_token else "   Raw token: EMPTY")
logger.info(f"   Processed token length: {len(config.TELEGRAM_TOKEN)}")
if config.TELEGRAM_TOKEN:
    token_preview = f"{config.TELEGRAM_TOKEN[:4]}...{config.TELEGRAM_TOKEN[-4:]}" if len(config.TELEGRAM_TOKEN) > 8 else config.TELEGRAM_TOKEN
    logger.info(f"   Token preview: {token_preview}")
    if len(config._raw_telegram_token) != len(config.TELEGRAM_TOKEN):
        removed = len(config._raw_telegram_token) - len(config.TELEGRAM_TOKEN)
        logger.info(f"   Removed {removed} whitespace characters")
    # Проверяем формат токена (обычно формат: число:строка)
    if ':' not in config.TELEGRAM_TOKEN:
        logger.error("❌ TELEGRAM_TOKEN doesn't contain ':' - INVALID FORMAT!")
        logger.error("   Telegram tokens should be in format: NUMBER:STRING")
    elif len(config.TELEGRAM_TOKEN.split(':')) != 2:
        logger.error("❌ TELEGRAM_TOKEN has unexpected format!")
    else:
        parts = config.TELEGRAM_TOKEN.split(':')
        logger.info(f"   Token format: OK (bot_id: {parts[0]}, token_part: {parts[1][:4]}...)")
else:
    logger.error("❌ TELEGRAM_TOKEN is EMPTY!")
logger.info("=" * 70)

# Валидация конфигурации при загрузке
config.validate()

