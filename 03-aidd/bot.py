# 1. Импорты
from aiogram import Bot, Dispatcher
from aiogram.types import Message
import os
from dotenv import load_dotenv
from openai import OpenAI

# 2. Конфигурация
load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL")

SYSTEM_PROMPT = "Ты профессиональная гадалка. Отвечай на вопросы о людях, предметах, событиях и т.д. в стиле гадалки."

# 3. Инициализация
bot = Bot(token=f"{TOKEN}")
dp = Dispatcher()
llm_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

# 4. Хендлеры
def call_llm(user_message: str, system_prompt: str) -> str:
    print(f"[LLM] Calling LLM with model: {OPENROUTER_MODEL}")
    try:
        response = llm_client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
        )
        answer = response.choices[0].message.content
        if not answer:
            raise ValueError("Empty response from LLM")
        print(f"[LLM] Received response: {answer[:100]}...")
        return answer
    except Exception as e:
        print(f"[ERROR] LLM error: {e}")
        raise

@dp.message()
async def message_handler(message: Message):
    if not message.text:
        print(f"[INFO] User {message.from_user.id} sent non-text message")
        await message.answer("Пожалуйста, отправьте текстовое сообщение")
        return
    
    print(f"[INFO] User {message.from_user.id} sent: {message.text}")
    try:
        llm_response = call_llm(message.text, SYSTEM_PROMPT)
    except Exception as e:
        print(f"[ERROR] LLM error: {e}")
        try:
            await message.answer("Произошла ошибка при обращении к AI. Попробуйте позже.")
        except Exception as telegram_error:
            print(f"[ERROR] Telegram API error: {telegram_error}")
        return
    
    try:
        await message.answer(llm_response)
    except Exception as e:
        print(f"[ERROR] Telegram API error: {e}")

# 5. Запуск
async def main():
    print("[START] Bot started")
    await dp.start_polling(bot)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

