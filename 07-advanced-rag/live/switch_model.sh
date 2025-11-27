#!/bin/bash
# Скрипт для быстрой смены бесплатных моделей OpenRouter

MODEL=$1

if [ -z "$MODEL" ]; then
    echo "Использование: ./switch_model.sh <model_name>"
    echo ""
    echo "Доступные бесплатные модели:"
    echo "  1. microsoft/phi-3-mini-128k-instruct:free"
    echo "  2. google/gemini-flash-1.5:free"
    echo "  3. qwen/qwen-2.5-7b-instruct:free"
    echo "  4. mistralai/mistral-7b-instruct:free"
    echo "  5. meta-llama/llama-3.1-8b-instruct:free"
    echo "  6. meta-llama/llama-3.2-3b-instruct:free"
    exit 1
fi

# Обновляем .env файл
sed -i '' "s|^MODEL=.*|MODEL=$MODEL|" .env
sed -i '' "s|^MODEL_QUERY_TRANSFORM=.*|MODEL_QUERY_TRANSFORM=$MODEL|" .env

echo "✓ Модель изменена на: $MODEL"
echo ""
echo "Текущие настройки:"
grep -E "^MODEL" .env

