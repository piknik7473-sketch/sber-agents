import logging
import json
import re
from openai import AsyncOpenAI
from openai import APIError, InternalServerError
from config import config
from models import TransactionResponse

logger = logging.getLogger(__name__)

def extract_json_from_markdown(content: str) -> str:
    """Извлекает JSON из markdown блоков кода или возвращает исходную строку."""
    if not content:
        logger.warning("extract_json_from_markdown received empty content")
        return content
    
    # Убираем пробелы в начале и конце
    original_content = content
    content = content.strip()
    logger.debug(f"extract_json_from_markdown: input length={len(original_content)}, stripped length={len(content)}")
    
    # Сначала пытаемся найти markdown блоки с кодом
    # Ищем все возможные варианты: ```json ... ```, ``` ... ```, и т.д.
    markdown_patterns = [
        r'```json\s*\n(.*?)\n```',      # ```json\n...\n```
        r'```json\s*(.*?)\n```',        # ```json...\n```
        r'```json\s*(.*?)```',          # ```json...```
        r'```\s*\n(.*?)\n```',          # ```\n...\n```
        r'```\s*(.*?)```',              # ```...```
    ]
    
    for pattern in markdown_patterns:
        match = re.search(pattern, content, re.DOTALL)
        if match:
            extracted = match.group(1).strip()
            logger.debug(f"extract_json_from_markdown: found match with pattern, extracted length={len(extracted)}")
            # Проверяем, что извлеченный контент начинается с {
            if extracted.startswith('{'):
                logger.debug(f"extract_json_from_markdown: returning extracted JSON (starts with {{)")
                return extracted
    
    # Если markdown блоков не найдено, ищем JSON напрямую
    # Находим первый { и последний }, учитывая вложенность
    first_brace = content.find('{')
    if first_brace == -1:
        # Если нет открывающей скобки, возвращаем исходную строку
        return content
    
    # Находим последнюю закрывающую скобку после первой открывающей
    # Это более надежный способ для многострочного JSON
    brace_count = 0
    last_brace = -1
    for i in range(first_brace, len(content)):
        if content[i] == '{':
            brace_count += 1
        elif content[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                last_brace = i
                break
    
    if last_brace != -1 and last_brace > first_brace:
        extracted = content[first_brace:last_brace + 1]
        # Проверяем, что это валидный JSON (начинается и заканчивается правильно)
        if extracted.strip().startswith('{') and extracted.strip().endswith('}'):
            return extracted
    
    # Если ничего не найдено, возвращаем исходную строку
    return content

client = AsyncOpenAI(
    api_key=config.OPENAI_API_KEY,
    base_url=config.OPENAI_BASE_URL
)

async def get_transaction_response_text(
    last_message: str,
    message_history: list[dict]
) -> TransactionResponse:
    try:
        # Пробуем сначала с strict mode, если не получится - без strict
        try:
            response = await client.chat.completions.create(
                model=config.MODEL_TEXT,
                messages=[
                    {"role": "system", "content": config.SYSTEM_PROMPT_TEXT},
                    *message_history[-10:],  # последние 10 сообщений для контекста
                    {"role": "user", "content": last_message}
                ],
                response_format={"type": "json_schema", "json_schema": {
                    "name": "transaction_response",
                    "schema": TransactionResponse.model_json_schema(),
                    "strict": True
                }}
            )
        except Exception as strict_error:
            logger.warning(f"Strict mode failed: {strict_error}, trying without strict")
            # Если strict mode не работает, пробуем без strict
            response = await client.chat.completions.create(
                model=config.MODEL_TEXT,
                messages=[
                    {"role": "system", "content": config.SYSTEM_PROMPT_TEXT},
                    *message_history[-10:],
                    {"role": "user", "content": last_message}
                ],
                response_format={"type": "json_schema", "json_schema": {
                    "name": "transaction_response",
                    "schema": TransactionResponse.model_json_schema(),
                    "strict": False
                }}
            )
        
        # Проверяем что ответ существует и содержит choices
        if not response:
            logger.error("LLM returned None response")
            raise ValueError("LLM returned None response")
        
        # Логируем полную информацию о response для диагностики
        logger.info(f"Response type: {type(response)}")
        if hasattr(response, 'model_dump'):
            response_dump = response.model_dump()
            logger.info(f"Response dump: {response_dump}")
            
            # Проверяем наличие ошибки от провайдера
            if 'error' in response_dump and response_dump['error']:
                error_info = response_dump['error']
                error_message = error_info.get('message', 'Unknown error')
                error_code = error_info.get('code', 'Unknown')
                provider = error_info.get('metadata', {}).get('provider_name', 'Unknown')
                
                logger.error(f"Provider error: {error_message} (code: {error_code}, provider: {provider})")
                
                # Формируем понятное сообщение об ошибке
                if error_code == 524:
                    raise ValueError("Превышено время ожидания ответа от модели. Попробуйте еще раз.")
                elif error_code == 429:
                    raise ValueError("Превышен лимит запросов. Подождите немного и попробуйте снова.")
                else:
                    raise ValueError(f"Ошибка провайдера: {error_message} (код: {error_code})")
        
        if not hasattr(response, 'choices') or not response.choices:
            error_msg = f"LLM returned empty or missing choices. Response type: {type(response)}"
            if hasattr(response, 'model_dump'):
                error_msg += f", Response: {response.model_dump()}"
            logger.error(error_msg)
            raise ValueError("LLM returned empty choices")
        
        if len(response.choices) == 0:
            logger.error("LLM returned empty choices list")
            raise ValueError("LLM returned empty choices list")
        
        if not response.choices[0] or not response.choices[0].message:
            logger.error(f"LLM returned invalid choice structure. Response: {response}")
            raise ValueError("LLM returned invalid choice structure")
        
        raw_content = response.choices[0].message.content
        logger.info(f"Raw LLM response (length: {len(raw_content) if raw_content else 0}): {raw_content[:1000] if raw_content else 'EMPTY'}")
        
        # Проверяем что ответ не пустой
        if not raw_content or not raw_content.strip():
            logger.error("LLM returned empty response content")
            raise ValueError("LLM returned empty response content")
        
        try:
            # Извлекаем JSON из markdown блоков (если есть)
            json_content = extract_json_from_markdown(raw_content)
            logger.info(f"Extracted JSON content (length: {len(json_content)}): {json_content[:300]}...")
            
            # Проверяем, что извлеченный контент похож на JSON
            if not json_content.strip().startswith('{'):
                logger.warning(f"Extracted content doesn't start with '{{', trying to find JSON in content")
                # Пытаемся найти JSON еще раз
                first_brace = json_content.find('{')
                if first_brace != -1:
                    json_content = json_content[first_brace:]
                    last_brace = json_content.rfind('}')
                    if last_brace != -1:
                        json_content = json_content[:last_brace + 1]
            
            # Парсим JSON ответ
            parsed_json = json.loads(json_content)
            
            # Обрабатываем случай, когда поле transactions отсутствует
            if "transactions" not in parsed_json:
                logger.warning("Field 'transactions' missing in LLM response, adding empty list")
                parsed_json["transactions"] = []
            
            # Убеждаемся, что answer есть
            if "answer" not in parsed_json:
                logger.warning("Field 'answer' missing in LLM response, adding default")
                parsed_json["answer"] = "Обработал ваше сообщение."
            
            parsed_response = TransactionResponse.model_validate(parsed_json)
            logger.info(f"Successfully parsed TransactionResponse: transactions={len(parsed_response.transactions)}")
            return parsed_response
        except json.JSONDecodeError as json_error:
            # Детальное логирование проблемы с JSON
            logger.error(f"Failed to parse JSON from LLM response: {json_error}")
            logger.error(f"Full response content ({len(raw_content)} chars): {raw_content}")
            logger.error(f"First 200 chars: {raw_content[:200]}")
            logger.error(f"Last 200 chars: {raw_content[-200:]}")
            raise
        except Exception as parse_error:
            # Детальное логирование для других ошибок парсинга
            logger.error(f"Failed to parse LLM response as TransactionResponse: {parse_error}")
            logger.error(f"Full response content ({len(raw_content)} chars): {raw_content}")
            logger.error(f"First 200 chars: {raw_content[:200]}")
            logger.error(f"Last 200 chars: {raw_content[-200:]}")
            raise
    except (APIError, InternalServerError) as e:
        logger.error(f"LLM API error: {e}")
        raise
    except Exception as e:
        logger.error(f"Error calling LLM: {e}", exc_info=True)
        raise

async def get_transaction_response_image(
    image_base64: str,
    message_history: list[dict]
) -> TransactionResponse:
    try:
        schema = TransactionResponse.model_json_schema()
        logger.info(f"Using model: {config.MODEL_IMAGE}, base_url: {config.OPENAI_BASE_URL}")
        
        # Логируем размер изображения в более понятном формате
        image_size_bytes = len(image_base64.encode('utf-8')) * 3 // 4  # примерная оценка
        image_size_kb = image_size_bytes / 1024
        logger.info(f"Image size: ~{image_size_kb:.1f} KB ({len(image_base64)} base64 chars)")
        logger.info(f"Message history length: {len(message_history)} messages")
        
        # Пробуем сначала с strict mode, если не получится - без strict
        try:
            response = await client.chat.completions.create(
                model=config.MODEL_IMAGE,
                messages=[
                    {"role": "system", "content": config.SYSTEM_PROMPT_IMAGE},
                    *message_history[-10:],  # последние 10 сообщений для контекста
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                            {"type": "text", "text": "Извлеки транзакции из этого изображения"}
                        ]
                    }
                ],
                response_format={"type": "json_schema", "json_schema": {
                    "name": "transaction_response",
                    "schema": schema,
                    "strict": True
                }}
            )
        except Exception as strict_error:
            logger.warning(f"Strict mode failed for image: {strict_error}, trying without strict")
            # Если strict mode не работает, пробуем без strict
            response = await client.chat.completions.create(
                model=config.MODEL_IMAGE,
                messages=[
                    {"role": "system", "content": config.SYSTEM_PROMPT_IMAGE},
                    *message_history[-10:],
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                            {"type": "text", "text": "Извлеки транзакции из этого изображения"}
                        ]
                    }
                ],
                response_format={"type": "json_schema", "json_schema": {
                    "name": "transaction_response",
                    "schema": schema,
                    "strict": False
                }}
            )
        
        # Проверяем что ответ существует и содержит choices
        if not response:
            logger.error("LLM returned None response for image")
            raise ValueError("LLM returned None response for image")
        
        # Логируем полную информацию о response для диагностики
        logger.info(f"Response type for image: {type(response)}")
        if hasattr(response, 'model_dump'):
            response_dump = response.model_dump()
            logger.info(f"Response dump for image: {response_dump}")
            
            # Проверяем наличие ошибки от провайдера
            if 'error' in response_dump and response_dump['error']:
                error_info = response_dump['error']
                error_message = error_info.get('message', 'Unknown error')
                error_code = error_info.get('code', 'Unknown')
                provider = error_info.get('metadata', {}).get('provider_name', 'Unknown')
                
                logger.error(f"Provider error for image: {error_message} (code: {error_code}, provider: {provider})")
                
                # Формируем понятное сообщение об ошибке
                if error_code == 524:
                    raise ValueError("Превышено время ожидания ответа от модели. Попробуйте еще раз.")
                elif error_code == 429:
                    raise ValueError("Превышен лимит запросов. Подождите немного и попробуйте снова.")
                else:
                    raise ValueError(f"Ошибка провайдера: {error_message} (код: {error_code})")
        
        if not hasattr(response, 'choices') or not response.choices:
            error_msg = f"LLM returned empty or missing choices for image. Response type: {type(response)}"
            if hasattr(response, 'model_dump'):
                error_msg += f", Response: {response.model_dump()}"
            logger.error(error_msg)
            raise ValueError("LLM returned empty choices for image")
        
        if len(response.choices) == 0:
            logger.error("LLM returned empty choices list for image")
            raise ValueError("LLM returned empty choices list for image")
        
        if not response.choices[0] or not response.choices[0].message:
            logger.error(f"LLM returned invalid choice structure for image. Response: {response}")
            raise ValueError("LLM returned invalid choice structure for image")
        
        # Логируем информацию о response объекте
        logger.info(f"Response object for image: {response}")
        logger.info(f"Response choices count for image: {len(response.choices)}")
        if response.choices:
            logger.info(f"First choice finish_reason for image: {response.choices[0].finish_reason}")
            logger.info(f"First choice message role for image: {response.choices[0].message.role}")
        
        raw_content = response.choices[0].message.content
        logger.info(f"Raw LLM response for image (length: {len(raw_content) if raw_content else 0}): {raw_content[:1000] if raw_content else 'EMPTY'}")
        
        # Проверяем что ответ не пустой
        if not raw_content or not raw_content.strip():
            logger.error("LLM returned empty response for image")
            logger.error(f"Response object details: {response}")
            logger.error(f"Finish reason: {response.choices[0].finish_reason if response.choices else 'no choices'}")
            raise ValueError("LLM returned empty response")
        
        try:
            # Извлекаем JSON из markdown блоков (если есть)
            json_content = extract_json_from_markdown(raw_content)
            logger.info(f"Extracted JSON content from image response (length: {len(json_content)}): {json_content[:300]}...")
            
            # Проверяем, что извлеченный контент похож на JSON
            if not json_content.strip().startswith('{'):
                logger.warning(f"Extracted content doesn't start with '{{', trying to find JSON in content")
                # Пытаемся найти JSON еще раз
                first_brace = json_content.find('{')
                if first_brace != -1:
                    json_content = json_content[first_brace:]
                    last_brace = json_content.rfind('}')
                    if last_brace != -1:
                        json_content = json_content[:last_brace + 1]
            
            # Парсим JSON ответ
            parsed_json = json.loads(json_content)
            logger.info(f"Successfully parsed JSON: transactions={len(parsed_json.get('transactions', []))}, has_answer={bool(parsed_json.get('answer'))}")
            
            # Обрабатываем случай, когда поле transactions отсутствует
            if "transactions" not in parsed_json:
                logger.warning("Field 'transactions' missing in LLM response, adding empty list")
                parsed_json["transactions"] = []
            
            # Убеждаемся, что answer есть
            if "answer" not in parsed_json:
                logger.warning("Field 'answer' missing in LLM response, adding default")
                parsed_json["answer"] = "Обработал изображение."
            
            # Логируем данные транзакций перед валидацией
            if parsed_json.get("transactions"):
                for i, txn in enumerate(parsed_json["transactions"]):
                    logger.info(f"Transaction {i} before validation: date={txn.get('date')}, time={txn.get('time')}, type={txn.get('type')}, amount={txn.get('amount')}")
            
            # Валидируем через Pydantic
            logger.info("Starting Pydantic validation...")
            parsed_response = TransactionResponse.model_validate(parsed_json)
            logger.info(f"Successfully parsed TransactionResponse for image: transactions={len(parsed_response.transactions)}")
            return parsed_response
        except json.JSONDecodeError as json_error:
            # Детальное логирование проблемы с JSON
            logger.error(f"Failed to parse JSON from LLM response for image: {json_error}")
            logger.error(f"Raw response content ({len(raw_content)} chars): {raw_content}")
            json_content = extract_json_from_markdown(raw_content)
            logger.error(f"Extracted JSON content ({len(json_content)} chars): {json_content}")
            logger.error(f"First 300 chars of extracted: {json_content[:300]}")
            logger.error(f"Last 300 chars of extracted: {json_content[-300:]}")
            raise
        except Exception as parse_error:
            # Детальное логирование для других ошибок парсинга
            logger.error(f"Failed to parse LLM response as TransactionResponse for image: {parse_error}", exc_info=True)
            logger.error(f"Error type: {type(parse_error).__name__}")
            logger.error(f"Raw response content ({len(raw_content)} chars): {raw_content}")
            json_content = extract_json_from_markdown(raw_content)
            logger.error(f"Extracted JSON content ({len(json_content)} chars): {json_content}")
            
            # Пытаемся распарсить JSON еще раз для диагностики
            try:
                test_parsed = json.loads(json_content)
                logger.error(f"JSON can be parsed, but Pydantic validation failed")
                logger.error(f"Parsed JSON structure: {list(test_parsed.keys())}")
                if "transactions" in test_parsed and test_parsed["transactions"]:
                    logger.error(f"First transaction keys: {list(test_parsed['transactions'][0].keys())}")
                    logger.error(f"First transaction data: {test_parsed['transactions'][0]}")
            except Exception as e:
                logger.error(f"Even JSON parsing failed: {e}")
            
            logger.error(f"First 300 chars of extracted: {json_content[:300]}")
            logger.error(f"Last 300 chars of extracted: {json_content[-300:]}")
            raise
    except (APIError, InternalServerError) as e:
        logger.error(f"LLM API error: {e}")
        raise
    except Exception as e:
        logger.error(f"Error calling LLM: {e}", exc_info=True)
        raise

