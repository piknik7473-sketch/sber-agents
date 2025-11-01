# 1. Импорты
from aiogram import Bot, Dispatcher
from aiogram.types import Message
import os
from dotenv import load_dotenv

# 2. Конфигурация
load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# 3. Инициализация
bot = Bot(token=TOKEN)
dp = Dispatcher()

# 4. Хендлеры
@dp.message()
async def echo_handler(message: Message):
    print(f"[INFO] User {message.from_user.id} sent: {message.text}")
    await message.answer(message.text)

# 5. Запуск
async def main():
    print("[START] Bot started")
    await dp.start_polling(bot)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

