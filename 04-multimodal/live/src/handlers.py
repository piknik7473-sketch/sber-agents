import logging
import base64
from datetime import time
from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message
from openai import APIError, InternalServerError, NotFoundError
from llm import get_transaction_response_text, get_transaction_response_image
from models import Transaction
from config import config
from speech import download_and_transcribe_voice
from storage import load_transactions, add_transactions

logger = logging.getLogger(__name__)
router = Router()

# Глобальные словари для хранения данных
chat_conversations: dict[int, list[dict]] = {}
# Загружаем транзакции из файла при запуске
transactions: dict[int, list[Transaction]] = load_transactions()

# Максимальная длина сообщения пользователя
MAX_MESSAGE_LENGTH = 4000

@router.message(Command("start"))
async def cmd_start(message: Message):
    chat_id = message.chat.id
    logger.info(f"User {chat_id} started the bot")
    
    # Очищаем историю и транзакции для данного чата
    chat_conversations[chat_id] = [
        {"role": "system", "content": config.SYSTEM_PROMPT_TEXT}
    ]
    transactions[chat_id] = []
    # Сохраняем изменения (очистку транзакций)
    from storage import save_transactions
    save_transactions(transactions)
    
    await message.answer(
        "Привет! Я персональный финансовый советник.\n\n"
        "Я могу:\n"
        "• Извлекать транзакции из ваших сообщений\n"
        "• Записывать транзакции через голосовой ввод\n"
        "• Обрабатывать изображения чеков и скриншотов\n"
        "• Вести учет доходов и расходов\n"
        "• Предоставлять советы по управлению финансами\n\n"
        "Используйте /start для начала нового диалога и очистки истории."
    )

@router.message(Command("balance"))
async def cmd_balance(message: Message):
    chat_id = message.chat.id
    logger.info(f"Balance requested by {chat_id}")
    
    # Получаем транзакции пользователя
    user_transactions = transactions.get(chat_id, [])
    
    if not user_transactions:
        await message.answer(
            "💵 У вас пока нет транзакций.\n\n"
            "Отправьте сообщение с транзакцией или изображение чека для начала учета."
        )
        return
    
    # Расчет баланса, доходов и расходов
    total_income = sum(t.amount for t in user_transactions if t.type.value == "income")
    total_expense = sum(t.amount for t in user_transactions if t.type.value == "expense")
    balance = total_income - total_expense
    
    # Статистика по категориям
    category_stats: dict[str, float] = {}
    for t in user_transactions:
        category = t.category
        if category not in category_stats:
            category_stats[category] = 0.0
        if t.type.value == "income":
            category_stats[category] += t.amount
        else:
            category_stats[category] -= t.amount
    
    # Форматирование отчета
    report_lines = [
        "💵 **Отчет о балансе**\n",
        f"📊 Баланс: {balance:.2f} руб.",
        f"💰 Доходы: {total_income:.2f} руб.",
        f"💸 Расходы: {total_expense:.2f} руб.",
        f"\n📈 Всего транзакций: {len(user_transactions)}",
        "\n**Статистика по категориям:**"
    ]
    
    # Сортируем категории по сумме (от большей к меньшей)
    sorted_categories = sorted(category_stats.items(), key=lambda x: abs(x[1]), reverse=True)
    for category, amount in sorted_categories:
        sign = "💰" if amount > 0 else "💸"
        report_lines.append(f"{sign} {category}: {amount:+.2f} руб.")
    
    await message.answer("\n".join(report_lines))

@router.message(Command("transactions"))
async def cmd_transactions(message: Message):
    chat_id = message.chat.id
    logger.info(f"Transactions list requested by {chat_id}")
    
    # Получаем транзакции пользователя
    user_transactions = transactions.get(chat_id, [])
    
    if not user_transactions:
        await message.answer(
            "📋 У вас пока нет транзакций.\n\n"
            "Отправьте сообщение с транзакцией или изображение чека для начала учета."
        )
        return
    
    # Сортируем транзакции по дате (от новых к старым)
    def get_time_for_sort(t):
        """Получает время для сортировки."""
        if t.time is None:
            return time(0, 0)
        if isinstance(t.time, time):
            return t.time
        # Если это строка, преобразуем
        try:
            parts = t.time.split(':')
            if len(parts) >= 2:
                return time(int(parts[0]), int(parts[1]))
        except:
            pass
        return time(0, 0)
    
    sorted_transactions = sorted(user_transactions, key=lambda t: (t.date, get_time_for_sort(t)), reverse=True)
    
    # Форматирование списка транзакций
    report_lines = [
        f"📋 **Все транзакции** ({len(user_transactions)} шт.)\n"
    ]
    
    for i, t in enumerate(sorted_transactions, 1):
        # Форматирование даты и времени
        date_str = t.date.strftime("%d.%m.%Y")
        # Форматируем время (может быть строкой или объектом time)
        if t.time:
            if isinstance(t.time, time):
                time_str = f" {t.time.strftime('%H:%M')}"
            else:
                # Если это строка, используем как есть или форматируем
                time_str = f" {t.time}" if ':' in str(t.time) else ""
        else:
            time_str = ""
        
        # Знак и тип транзакции
        sign = "💰" if t.type.value == "income" else "💸"
        type_str = "Доход" if t.type.value == "income" else "Расход"
        
        # Форматирование суммы
        amount_str = f"{t.amount:.2f}".rstrip('0').rstrip('.')
        
        # Описание (если есть)
        desc_str = f"\n   {t.description}" if t.description else ""
        
        report_lines.append(
            f"{i}. {sign} **{type_str}** {amount_str} руб.\n"
            f"   📅 {date_str}{time_str}\n"
            f"   🏷️ {t.category}{desc_str}"
        )
    
    # Если транзакций много, разбиваем на несколько сообщений (Telegram лимит ~4096 символов)
    report_text = "\n\n".join(report_lines)
    if len(report_text) > 4000:
        # Разбиваем на части
        parts = []
        current_part = [report_lines[0]]  # Заголовок
        current_length = len(report_lines[0])
        
        for line in report_lines[1:]:
            line_length = len(line) + 2  # +2 для "\n\n"
            if current_length + line_length > 4000:
                parts.append("\n\n".join(current_part))
                current_part = [line]
                current_length = len(line)
            else:
                current_part.append(line)
                current_length += line_length
        
        if current_part:
            parts.append("\n\n".join(current_part))
        
        # Отправляем части
        for part in parts:
            await message.answer(part)
    else:
        await message.answer(report_text)

@router.message(lambda message: message.photo or (message.document and message.document.mime_type and message.document.mime_type.startswith("image/")))
async def handle_image(message: Message):
    chat_id = message.chat.id
    
    logger.info(f"Image received from {chat_id}")
    
    # Инициализируем историю если её нет
    if chat_id not in chat_conversations:
        chat_conversations[chat_id] = [
            {"role": "system", "content": config.SYSTEM_PROMPT_IMAGE}
        ]
    
    try:
        # Определяем источник изображения
        if message.photo:
            # Берем самое большое изображение
            photo = message.photo[-1]
            file_info = await message.bot.get_file(photo.file_id)
        elif message.document:
            file_info = await message.bot.get_file(message.document.file_id)
        else:
            await message.answer("Не удалось обработать изображение.")
            return
        
        # Скачиваем изображение
        file_buffer = await message.bot.download_file(file_info.file_path)
        image_bytes = file_buffer.getvalue()
        
        # Конвертируем в base64
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Получаем историю сообщений без системного промпта для контекста
        message_history = chat_conversations[chat_id][1:] if chat_conversations[chat_id] else []
        
        # Получаем ответ LLM с structured output
        response = await get_transaction_response_image(image_base64, message_history)
        
        # Детальное логирование ответа LLM
        logger.info(f"LLM response for image from {chat_id}: answer='{response.answer[:200]}...', transactions_count={len(response.transactions)}")
        if response.transactions:
            logger.info(f"Extracted {len(response.transactions)} transactions from image for {chat_id}: {[t.model_dump() for t in response.transactions]}")
        else:
            logger.warning(f"No transactions extracted from image for {chat_id}")
        
        # Сохраняем транзакции
        if response.transactions:
            add_transactions(chat_id, response.transactions, transactions)
        
        # Рассчитываем баланс
        balance = sum(
            t.amount if t.type.value == "income" else -t.amount 
            for t in transactions.get(chat_id, [])
        )
        
        # Формируем ответ пользователю
        answer_text = response.answer
        
        # Добавляем статус транзакций
        if response.transactions:
            count = len(response.transactions)
            answer_text += f"\n\n✅ Найдено и сохранено {count} транзакция{'и' if count > 1 else ''}"
        else:
            answer_text += "\n\nℹ️ Транзакции не найдены"
        
        # Добавляем баланс
        balance_str = f"{balance:.0f}" if balance == int(balance) else f"{balance:.2f}"
        answer_text += f"\n💵 Баланс: {balance_str} руб."
        
        # Добавляем изображение в историю как текстовое описание (для контекста)
        chat_conversations[chat_id].append(
            {"role": "user", "content": "[Изображение: чек/скриншот]"}
        )
        
        # Добавляем ответ LLM в историю
        chat_conversations[chat_id].append(
            {"role": "assistant", "content": response.answer}
        )
        
        await message.answer(answer_text)
    except (APIError, InternalServerError, NotFoundError) as e:
        logger.error(f"LLM API error for image from {chat_id}: {e}", exc_info=True)
        error_message = str(e)
        if "image input" in error_message.lower() or "404" in error_message or "not found" in error_message.lower():
            await message.answer(
                "Извините, используемая модель не поддерживает обработку изображений.\n\n"
                "Для работы с изображениями необходимо использовать vision-модель, например:\n"
                "• meta-llama/llama-3.2-11b-vision-instruct (OpenRouter)\n"
                "• llama3.2-vision (Ollama)\n\n"
                "Измените MODEL в файле .env на одну из этих моделей."
            )
        else:
            await message.answer(
                f"❌ Ошибка на стороне провайдера LLM при обработке изображения.\n\n{error_message}\n\n"
                f"Пожалуйста, попробуйте еще раз через несколько секунд."
            )
    except ValueError as e:
        # Обработка ошибок от провайдера (timeout, rate limit и т.д.)
        logger.error(f"LLM provider error for image from {chat_id}: {e}", exc_info=True)
        error_message = str(e)
        await message.answer(
            f"❌ {error_message}\n\n"
            f"Попробуйте отправить изображение еще раз или повторите попытку позже."
        )
    except Exception as e:
        logger.error(f"Error processing image from {chat_id}: {e}", exc_info=True)
        await message.answer(
            f"❌ Произошла ошибка при обработке изображения.\n\n"
            f"Попробуйте еще раз или используйте /start для начала нового диалога."
        )

@router.message(lambda message: message.voice is not None)
async def handle_voice(message: Message):
    """Обработчик голосовых сообщений - транскрибация и извлечение транзакций."""
    chat_id = message.chat.id
    
    logger.info(f"Voice message received from {chat_id}, duration: {message.voice.duration}s")
    
    # Проверяем длину голосового сообщения (максимум 60 секунд для быстрой обработки)
    if message.voice.duration > 60:
        await message.answer(
            f"Извините, голосовое сообщение слишком длинное ({message.voice.duration} секунд). "
            f"Максимальная длина: 60 секунд."
        )
        return
    
    try:
        # Показываем пользователю, что обрабатываем сообщение
        processing_msg = await message.answer("🎤 Обрабатываю голосовое сообщение...")
        
        # Транскрибируем голосовое сообщение
        transcribed_text = await download_and_transcribe_voice(
            message.bot,
            message.voice,
            model_name=config.WHISPER_MODEL,
            language=config.WHISPER_LANGUAGE
        )
        
        if not transcribed_text or not transcribed_text.strip():
            await processing_msg.edit_text("❌ Не удалось распознать речь в голосовом сообщении.")
            return
        
        logger.info(f"Voice transcribed for {chat_id}: {transcribed_text[:100]}...")
        
        # Обновляем сообщение о статусе
        await processing_msg.edit_text("📝 Распознанный текст получен. Обрабатываю транзакции...")
        
        # Инициализируем историю если её нет
        if chat_id not in chat_conversations:
            chat_conversations[chat_id] = [
                {"role": "system", "content": config.SYSTEM_PROMPT_TEXT}
            ]
        
        # Получаем историю сообщений без системного промпта для контекста
        message_history = chat_conversations[chat_id][1:] if chat_conversations[chat_id] else []
        
        try:
            # Получаем ответ LLM с structured output для извлечения транзакций
            response = await get_transaction_response_text(transcribed_text, message_history)
            
            # Детальное логирование ответа LLM
            logger.info(f"LLM response for voice from {chat_id}: answer='{response.answer[:200]}...', transactions_count={len(response.transactions)}")
            if response.transactions:
                logger.info(f"Extracted {len(response.transactions)} transactions from voice for {chat_id}: {[t.model_dump() for t in response.transactions]}")
            
            # Сохраняем транзакции
            if response.transactions:
                add_transactions(chat_id, response.transactions, transactions)
            
            # Рассчитываем баланс
            balance = sum(
                t.amount if t.type.value == "income" else -t.amount 
                for t in transactions.get(chat_id, [])
            )
            
            # Удаляем сообщение о обработке
            await processing_msg.delete()
            
            # Формируем ответ пользователю
            answer_lines = [
                f"📝 Распознанный текст:\n{transcribed_text}\n",
                response.answer
            ]
            
            # Добавляем статус транзакций
            if response.transactions:
                count = len(response.transactions)
                answer_lines.append(f"\n✅ Найдено и сохранено {count} транзакция{'и' if count > 1 else ''}")
            else:
                answer_lines.append("\nℹ️ Транзакции не найдены")
            
            # Добавляем баланс
            balance_str = f"{balance:.0f}" if balance == int(balance) else f"{balance:.2f}"
            answer_lines.append(f"💵 Баланс: {balance_str} руб.")
            
            answer_text = "\n".join(answer_lines)
            
            # Добавляем голосовое сообщение в историю как текстовое
            chat_conversations[chat_id].append(
                {"role": "user", "content": f"[Голосовое сообщение]: {transcribed_text}"}
            )
            
            # Добавляем ответ LLM в историю
            chat_conversations[chat_id].append(
                {"role": "assistant", "content": response.answer}
            )
            
            await message.answer(answer_text)
        
        except (APIError, InternalServerError) as e:
            logger.error(f"LLM API error for voice from {chat_id}: {e}", exc_info=True)
            error_message = str(e)
            await processing_msg.edit_text(
                f"❌ Ошибка при обработке распознанного текста.\n\n"
                f"{error_message}\n\n"
                f"Распознанный текст:\n{transcribed_text}"
            )
        except ValueError as e:
            # Обработка ошибок от провайдера (timeout, rate limit и т.д.)
            logger.error(f"LLM provider error for voice from {chat_id}: {e}", exc_info=True)
            error_message = str(e)
            await processing_msg.edit_text(
                f"❌ {error_message}\n\n"
                f"Распознанный текст:\n{transcribed_text}\n\n"
                f"Попробуйте отправить текстовое сообщение или повторите попытку позже."
            )
        except Exception as e:
            logger.error(f"Error processing transcribed text for {chat_id}: {e}", exc_info=True)
            await processing_msg.edit_text(
                f"❌ Ошибка при обработке.\n\n"
                f"Распознанный текст:\n{transcribed_text}\n\n"
                f"Попробуйте отправить текстовое сообщение."
            )
    
    except Exception as e:
        logger.error(f"Error processing voice message from {chat_id}: {e}", exc_info=True)
        await message.answer(
            "Произошла ошибка при обработке голосового сообщения. "
            "Попробуйте еще раз или отправьте текстовое сообщение."
        )

@router.message()
async def handle_message(message: Message):
    # Игнорируем сообщения без текста
    if not message.text:
        await message.answer("Извините, я работаю только с текстовыми сообщениями.")
        return
    
    # Проверяем длину сообщения
    if len(message.text) > MAX_MESSAGE_LENGTH:
        await message.answer(
            f"Извините, ваше сообщение слишком длинное ({len(message.text)} символов). "
            f"Максимальная длина: {MAX_MESSAGE_LENGTH} символов."
        )
        return
    
    chat_id = message.chat.id
    last_message = message.text
    
    logger.info(f"Message from {chat_id}: {last_message[:100]}...")
    
    # Инициализируем историю если её нет
    if chat_id not in chat_conversations:
        chat_conversations[chat_id] = [
            {"role": "system", "content": config.SYSTEM_PROMPT_TEXT}
        ]
    
    # Получаем историю сообщений без системного промпта для контекста
    message_history = chat_conversations[chat_id][1:] if chat_conversations[chat_id] else []
    
    try:
        # Получаем ответ LLM с structured output (извлечение транзакций только из последнего сообщения)
        response = await get_transaction_response_text(last_message, message_history)
        
        # Детальное логирование ответа LLM
        logger.info(f"LLM response for {chat_id}: answer='{response.answer[:200]}...', transactions_count={len(response.transactions)}")
        if response.transactions:
            logger.info(f"Extracted {len(response.transactions)} transactions for {chat_id}: {[t.model_dump() for t in response.transactions]}")
        else:
            logger.warning(f"No transactions extracted from message: '{last_message}' for {chat_id}")
        
        # Сохраняем транзакции
        if response.transactions:
            add_transactions(chat_id, response.transactions, transactions)
        
        # Рассчитываем баланс
        balance = sum(
            t.amount if t.type.value == "income" else -t.amount 
            for t in transactions.get(chat_id, [])
        )
        
        # Формируем ответ пользователю
        answer_text = response.answer
        
        # Добавляем статус транзакций
        if response.transactions:
            count = len(response.transactions)
            answer_text += f"\n\n✅ Найдено и сохранено {count} транзакция{'и' if count > 1 else ''}"
        else:
            answer_text += "\n\nℹ️ Транзакции не найдены"
        
        # Добавляем баланс
        balance_str = f"{balance:.0f}" if balance == int(balance) else f"{balance:.2f}"
        answer_text += f"\n💵 Баланс: {balance_str} руб."
        
        # Добавляем сообщение пользователя в историю
        chat_conversations[chat_id].append(
            {"role": "user", "content": last_message}
        )
        
        # Добавляем ответ LLM в историю
        chat_conversations[chat_id].append(
            {"role": "assistant", "content": response.answer}
        )
        
        await message.answer(answer_text)
    except (APIError, InternalServerError) as e:
        logger.error(f"LLM API error for {chat_id}: {e}", exc_info=True)
        error_message = str(e)
        await message.answer(
            f"❌ Ошибка на стороне провайдера LLM.\n\n{error_message}\n\n"
            f"Пожалуйста, попробуйте еще раз через несколько секунд."
        )
    except ValueError as e:
        # Обработка ошибок от провайдера (timeout, rate limit и т.д.)
        logger.error(f"LLM provider error for {chat_id}: {e}", exc_info=True)
        error_message = str(e)
        await message.answer(
            f"❌ {error_message}\n\n"
            f"Попробуйте повторить попытку позже."
        )
    except Exception as e:
        logger.error(f"Error in handle_message for {chat_id}: {e}", exc_info=True)
        await message.answer(
            f"❌ Произошла ошибка при обработке вашего сообщения.\n\n"
            f"Попробуйте еще раз или используйте /start для начала нового диалога."
        )

