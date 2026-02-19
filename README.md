# 🛡️ Fraud Detection System

Система обнаружения мошеннических финансовых транзакций с использованием машинного обучения.

## 📋 Описание проекта

Данный проект представляет собой MVP системы Fraud Detection, разработанный для обнаружения мошеннических транзакций в финансовых операциях. Система использует алгоритмы машинного обучения (XGBoost, Random Forest, Gradient Boosting) для классификации транзакций.

### Основные возможности:
- 🔍 Анализ отдельных транзакций в реальном времени
- 📁 Пакетная обработка CSV файлов
- 📈 Визуализация статистики и метрик модели
- 🚨 Оповещения о мошенничестве с указанием вероятности
- 🔌 REST API для интеграции с внешними системами

## 🏗️ Архитектура

```
HACKATON/
├── PS_20174392719_1491204439457_log.csv  # Датасет
├── train_model.py                          # Скрипт обучения модели
├── app.py                                  # Streamlit приложение
├── api.py                                  # FastAPI REST API
├── requirements.txt                        # Зависимости
├── models/                                 # Обученные модели
│   ├── fraud_detection_model.joblib
│   ├── scaler.joblib
│   ├── type_encoder.joblib
│   ├── feature_columns.joblib
│   └── model_info.joblib
└── plots/                                  # Графики
    ├── eda_analysis.png
    ├── model_evaluation.png
    ├── feature_importance.png
    └── model_comparison.png
```

## 🚀 Быстрый старт

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 2. Обучение модели

```bash
python train_model.py
```

Этот скрипт:
- Загружает и анализирует датасет
- Создает новые признаки (Feature Engineering)
- Обучает несколько моделей и выбирает лучшую
- Сохраняет модель и компоненты в папку `models/`
- Генерирует графики в папку `plots/`

### 3. Запуск Streamlit приложения

```bash
streamlit run app.py
```

Приложение будет доступно по адресу: http://localhost:8501

### 4. Запуск FastAPI сервера

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

API документация: http://localhost:8000/docs

## 📊 Используемые технологии

| Компонент | Технология |
|-----------|------------|
| Машинное обучение | XGBoost, Random Forest, Gradient Boosting |
| Балансировка данных | SMOTE (imbalanced-learn) |
| Веб-интерфейс | Streamlit |
| REST API | FastAPI |
| Обработка данных | Pandas, NumPy |
| Визуализация | Plotly, Matplotlib, Seaborn |

## 🔧 API Endpoints

### POST /predict
Предсказание для одной транзакции

```json
{
  "step": 1,
  "type": "CASH_OUT",
  "amount": 1000.0,
  "oldbalanceOrg": 5000.0,
  "newbalanceOrig": 4000.0,
  "oldbalanceDest": 0.0,
  "newbalanceDest": 1000.0
}
```

### POST /batch
Пакетное предсказание для списка транзакций

### POST /upload
Загрузка CSV файла для анализа

### GET /model/info
Информация о модели и метриках

### GET /health
Проверка состояния API

## 📈 Метрики модели

| Метрика | Описание |
|---------|----------|
| ROC-AUC | Площадь под ROC-кривой |
| Precision | Точность предсказаний |
| Recall | Полнота обнаружения |
| F1-Score | Баланс между precision и recall |
| Accuracy | Общая точность |

## 🔑 Признаки для анализа

Модель использует следующие признаки:

| Признак | Описание |
|---------|----------|
| step | Временной шаг (часы) |
| type | Тип транзакции |
| amount | Сумма транзакции |
| oldbalanceOrg | Баланс отправителя до |
| newbalanceOrig | Баланс отправителя после |
| oldbalanceDest | Баланс получателя до |
| newbalanceDest | Баланс получателя после |
| errorBalanceOrig | Ошибка баланса отправителя |
| errorBalanceDest | Ошибка баланса получателя |
| isZeroBalanceAfter | Обнулён ли баланс после |
| amountToBalanceRatio | Отношение суммы к балансу |
| hour | Час дня |
| day | День |
| destEmptyBefore | Пустой ли счёт получателя |

## 🎯 Типы транзакций

| Тип | Описание | Уровень риска |
|-----|----------|---------------|
| CASH_OUT | Снятие наличных | Высокий |
| TRANSFER | Перевод средств | Высокий |
| CASH_IN | Внесение наличных | Низкий |
| PAYMENT | Платеж | Низкий |
| DEBIT | Дебетовая операция | Средний |

## 📝 Пример использования API

### Python (requests)

```python
import requests

# Предсказание для одной транзакции
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "step": 1,
        "type": "CASH_OUT",
        "amount": 1000.0,
        "oldbalanceOrg": 5000.0,
        "newbalanceOrig": 4000.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 1000.0
    }
)
print(response.json())
```

### cURL

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "step": 1,
       "type": "CASH_OUT",
       "amount": 1000.0,
       "oldbalanceOrg": 5000.0,
       "newbalanceOrig": 4000.0,
       "oldbalanceDest": 0.0,
       "newbalanceDest": 1000.0
     }'
```

## 📋 Требования

- Python 3.8+
- 8GB+ RAM (для обработки датасета)
- ~500MB свободного места

## 👥 Авторы

Проект разработан в рамках хакатона для решения задач финансовой безопасности.

## 📄 Лицензия

MIT License