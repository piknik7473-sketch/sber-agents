.PHONY: install run

install:
	uv sync --python 3.13

run:
	@echo "Останавливаем старые процессы бота..."
	@pkill -f "python.*bot.py" 2>/dev/null || true
	@sleep 1
	@echo "Запускаем бота..."
	uv run python src/bot.py

stop:
	@echo "Останавливаем все процессы бота..."
	@pkill -f "python.*bot.py" 2>/dev/null || true
	@echo "Бот остановлен"

