import asyncio
import logging
import sys
from pathlib import Path
from aiogram import Bot, Dispatcher
from aiogram.client.session.aiohttp import AiohttpSession
from aiohttp import ClientError, ClientConnectorError
from handlers import router
from config import config
#import indexer
import rag
from indexer_with_json import reindex_all

# Создаем директорию для логов
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Настройка логирования в консоль и файл
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Вывод в консоль
        logging.FileHandler(log_dir / "bot.log", encoding='utf-8')  # Запись в файл
    ]
)
logger = logging.getLogger(__name__)

async def main():
    logger.info("=" * 50)
    logger.info("Bot starting...")
    
    # Проверка наличия токена
    if not config.TELEGRAM_TOKEN:
        logger.error("TELEGRAM_TOKEN not found in environment variables!")
        logger.error("Please create a .env file with TELEGRAM_TOKEN=your_token")
        sys.exit(1)
    
    # Индексация при старте
    logger.info("Starting indexing...")
    rag.vector_store = await reindex_all()
    if rag.vector_store:
        # Инициализируем retriever
        rag.initialize_retriever()
        stats = rag.get_vector_store_stats()
        logger.info(f"Indexing completed successfully: {stats['count']} documents indexed")
    else:
        logger.warning("Indexing completed with no documents - bot will run but cannot answer questions")
    
    # Создаем сессию с таймаутами для лучшей обработки ошибок
    session = AiohttpSession()
    bot = Bot(token=config.TELEGRAM_TOKEN, session=session)
    dp = Dispatcher()
    dp.include_router(router)
    
    logger.info("Starting bot polling...")
    try:
        await dp.start_polling(bot, close_bot_session=True)
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except (ClientError, ClientConnectorError) as e:
        logger.error(f"Network error: Cannot connect to Telegram API: {e}")
        logger.error("Please check your internet connection and try again")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Bot stopped with error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        await bot.session.close()
        logger.info("Bot shutdown complete")
        logger.info("=" * 50)

if __name__ == "__main__":
    asyncio.run(main())

