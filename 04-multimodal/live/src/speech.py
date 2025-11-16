import logging
import tempfile
from pathlib import Path
import whisper
from aiogram.types import Voice

logger = logging.getLogger(__name__)

# Глобальная переменная для хранения модели Whisper
_whisper_model = None

def get_whisper_model(model_name: str = "base") -> whisper.Whisper:
    """
    Загружает модель Whisper (загружается один раз и переиспользуется).
    
    Args:
        model_name: Название модели Whisper (tiny, base, small, medium, large)
    
    Returns:
        Загруженная модель Whisper
    """
    global _whisper_model
    
    if _whisper_model is None:
        logger.info(f"Loading Whisper model: {model_name}")
        try:
            _whisper_model = whisper.load_model(model_name)
            logger.info(f"Whisper model {model_name} loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}", exc_info=True)
            raise
    
    return _whisper_model

async def transcribe_voice_message(
    voice_file_path: str,
    model_name: str = "base",
    language: str = "ru"
) -> str:
    """
    Транскрибирует голосовое сообщение в текст.
    
    Args:
        voice_file_path: Путь к файлу голосового сообщения (OGG формат)
        model_name: Название модели Whisper (tiny, base, small, medium, large)
        language: Язык для распознавания (ru, en, и т.д.)
    
    Returns:
        Распознанный текст
    """
    try:
        # Загружаем модель
        model = get_whisper_model(model_name)
        
        # Транскрибируем аудио
        logger.info(f"Transcribing voice message: {voice_file_path}")
        result = model.transcribe(
            voice_file_path,
            language=language,
            task="transcribe"
        )
        
        text = result["text"].strip()
        logger.info(f"Transcription completed: {len(text)} characters")
        
        return text
    
    except Exception as e:
        logger.error(f"Error transcribing voice message: {e}", exc_info=True)
        raise

async def download_and_transcribe_voice(
    bot,
    voice: Voice,
    model_name: str = "base",
    language: str = "ru"
) -> str:
    """
    Скачивает голосовое сообщение и транскрибирует его.
    
    Args:
        bot: Экземпляр бота aiogram
        voice: Объект Voice из сообщения
        model_name: Название модели Whisper
        language: Язык для распознавания
    
    Returns:
        Распознанный текст
    """
    # Создаем временную директорию для файлов
    temp_dir = Path(tempfile.gettempdir()) / "telegram_voice"
    temp_dir.mkdir(exist_ok=True)
    
    voice_file_path = None
    try:
        # Получаем информацию о файле
        file_info = await bot.get_file(voice.file_id)
        
        # Скачиваем файл
        voice_file_path = temp_dir / f"{voice.file_id}.ogg"
        await bot.download_file(file_info.file_path, voice_file_path)
        
        logger.info(f"Voice file downloaded: {voice_file_path}, duration: {voice.duration}s")
        
        # Транскрибируем
        text = await transcribe_voice_message(
            str(voice_file_path),
            model_name=model_name,
            language=language
        )
        
        return text
    
    finally:
        # Удаляем временный файл
        if voice_file_path and voice_file_path.exists():
            try:
                voice_file_path.unlink()
                logger.debug(f"Temporary voice file deleted: {voice_file_path}")
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {voice_file_path}: {e}")

