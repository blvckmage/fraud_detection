#!/bin/bash

# Fraud Detection System - Launch Script

echo "🛡️ Fraud Detection System"
echo "=========================="

cd "$(dirname "$0")"

# Activate virtual environment
source venv/bin/activate

case "$1" in
    "streamlit"|"app"|"web")
        echo "🚀 Запуск Streamlit приложения..."
        streamlit run app.py
        ;;
    "api"|"server")
        echo "🚀 Запуск FastAPI сервера..."
        uvicorn api:app --reload --host 0.0.0.0 --port 8000
        ;;
    "train")
        echo "🎯 Обучение модели..."
        python train_model.py
        ;;
    "all")
        echo "🚀 Запуск Streamlit и API..."
        streamlit run app.py &
        uvicorn api:app --reload --host 0.0.0.0 --port 8000 &
        wait
        ;;
    *)
        echo "Использование: $0 {streamlit|api|train|all}"
        echo ""
        echo "  streamlit  - Запустить веб-интерфейс (http://localhost:8501)"
        echo "  api        - Запустить REST API (http://localhost:8000)"
        echo "  train      - Обучить модель"
        echo "  all        - Запустить оба сервиса"
        ;;
esac