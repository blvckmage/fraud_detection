# Fraud Detection System

Система обнаружения мошеннических финансовых транзакций с использованием машинного обучения.

---

## Описание проекта

Данный проект представляет собой MVP системы Fraud Detection, разработанный для обнаружения мошеннических транзакций в финансовых операциях. Система использует алгоритмы машинного обучения (XGBoost) для классификации транзакций и предоставляет два интерфейса для работы: веб-приложение на Streamlit и REST API на FastAPI.

### Основные возможности

- Анализ отдельных транзакций в реальном времени
- Пакетная обработка CSV файлов
- Визуализация статистики и метрик модели
- Оповещения о мошенничестве с указанием вероятности
- REST API для интеграции с внешними системами
- Интерактивная документация API (Swagger UI)

---

## Архитектура системы

### Компоненты

Система построена по модульному принципу и состоит из трёх основных компонентов:

**1. ML Pipeline (Модуль машинного обучения)**
- XGBoost Classifier — градиентный бустинг для классификации
- StandardScaler — нормализация признаков
- LabelEncoder — кодирование категориальных признаков
- SMOTE — балансировка классов при обучении
- joblib — сериализация модели

**2. Backend (FastAPI)**
- REST API с автоматической документацией
- Асинхронная обработка запросов
- CORS для кросс-доменных запросов
- Pydantic для валидации данных

**3. Frontend (Streamlit)**
- Интерактивный веб-интерфейс
- Визуализация результатов (Plotly)
- Загрузка CSV файлов
- Многостраничная навигация

### Структура проекта

```
HACKATON/
├── PS_20174392719_1491204439457_log.csv  # Датасет
├── train_model.py                          # Скрипт обучения модели
├── app.py                                  # Streamlit приложение
├── api.py                                  # FastAPI REST API
├── requirements.txt                        # Зависимости
├── Procfile                                # Конфигурация для Render
├── render.yaml                             # Настройки Render
├── run.sh                                  # Скрипт запуска
├── test_transactions.csv                   # Тестовые данные
├── models/                                 # Обученные модели
│   ├── fraud_detection_model.joblib       # XGBoost модель
│   ├── scaler.joblib                      # StandardScaler
│   ├── type_encoder.joblib                # LabelEncoder
│   ├── feature_columns.joblib             # Список признаков
│   └── model_info.joblib                  # Информация о модели
└── plots/                                  # Графики
    ├── eda_analysis.png                   # Анализ данных
    ├── model_evaluation.png               # Оценка модели
    ├── feature_importance.png             # Важность признаков
    └── model_comparison.png               # Сравнение моделей
```

---

## Технологический стек

| Слой | Технологии | Назначение |
|------|------------|------------|
| Машинное обучение | XGBoost | Градиентный бустинг для классификации |
| | scikit-learn | Предобработка данных, метрики качества |
| | imbalanced-learn | SMOTE для балансировки классов |
| | joblib | Сохранение и загрузка модели |
| Backend | FastAPI | REST API фреймворк |
| | uvicorn | ASGI-сервер |
| | Pydantic | Валидация данных |
| Frontend | Streamlit | Веб-интерфейс |
| | Plotly | Интерактивные графики |
| | pandas | Обработка табличных данных |
| Обработка данных | NumPy | Матричные операции |
| | pandas | Работа с DataFrame |
| Deploy | Render | Облачный хостинг |
| | GitHub | Контроль версий |
| Язык | Python 3.10+ | Основной язык разработки |

---

## Установка и запуск

### Требования

- Python 3.8+
- 8GB+ RAM (для обработки датасета)
- ~500MB свободного места

### 1. Клонирование репозитория

```bash
git clone https://github.com/blvckmage/fraud_detection.git
cd fraud_detection
```

### 2. Создание виртуального окружения

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate  # Windows
```

### 3. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 4. Обучение модели

```bash
python train_model.py
```

Этот скрипт выполняет:
- Загрузку и анализ датасета (6.3 млн транзакций)
- Feature Engineering (создание 7 дополнительных признаков)
- Балансировку классов с помощью SMOTE
- Обучение XGBoost модели
- Оценку качества на тестовой выборке
- Сохранение модели и компонентов в папку `models/`
- Генерацию графиков в папку `plots/`

### 5. Запуск Streamlit приложения

```bash
streamlit run app.py
```

Приложение будет доступно по адресу: http://localhost:8501

### 6. Запуск FastAPI сервера

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

API документация: http://localhost:8000/docs

---

## Streamlit Веб-интерфейс

### Обзор

Веб-приложение предоставляет удобный интерфейс для анализа транзакций без необходимости программирования. Приложение разделено на 5 функциональных страниц, доступных через боковое меню навигации.

### Страница "Главная"

Главная страница предоставляет общий обзор системы и её возможностей.

**Раздел "Назначение системы"**
Содержит описание того, что система использует алгоритмы машинного обучения для выявления подозрительных финансовых транзакций в режиме реального времени.

**Раздел "Типы транзакций"**
Таблица с расшифровкой всех поддерживаемых типов транзакций:
- CASH_OUT — снятие наличных средств
- CASH_IN — внесение наличных средств
- TRANSFER — перевод между счетами
- PAYMENT — платёж за услуги
- DEBIT — дебетовая операция

**Раздел "Используемые технологии"**
Информация о модели (XGBoost), балансировке классов (SMOTE) и Feature Engineering.

### Страница "Анализ транзакции"

Страница предназначена для проверки отдельных транзакций на предмет мошенничества.

**Ввод данных**

Пользователь заполняет форму с параметрами транзакции:
- Тип транзакции (выпадающий список из 5 вариантов)
- Сумма транзакции (числовое поле)
- Баланс отправителя до транзакции
- Баланс отправителя после транзакции
- Баланс получателя до транзакции
- Баланс получателя после транзакции

**Обработка**

После нажатия кнопки "Анализировать транзакцию" система выполняет:
1. Feature Engineering — создание 7 дополнительных признаков
2. Масштабирование признаков через StandardScaler
3. Предсказание через XGBoost модель
4. Формирование результата

**Результат**

Система выводит результат в виде цветного блока:
- Зеленый блок "ТРАНЗАКЦИЯ ЛЕГИТИМНА" — если fraud_probability < 0.5
- Красный блок "ВЫЯВЛЕНО МОШЕННИЧЕСТВО" — если fraud_probability >= 0.5

Показывается точная вероятность мошенничества в процентах.

**Детали анализа**
- График вероятностей классов (столбчатая диаграмма)
- Индикаторы риска — список предупреждений о подозрительных признаках:
  - Крупная сумма транзакции (> 100,000)
  - Баланс обнулен после транзакции
  - Высокорисковый тип транзакции (CASH_OUT, TRANSFER)
  - Получатель не имел средств на счете
  - Несоответствие баланса отправителя

### Страница "Пакетная обработка"

Страница позволяет анализировать множество транзакций одновременно путем загрузки CSV файла.

**Требования к файлу**

Система проверяет наличие обязательных колонок:
- type — тип транзакции
- amount — сумма
- oldbalanceOrg — баланс отправителя до
- newbalanceOrig — баланс отправителя после
- oldbalanceDest — баланс получателя до
- newbalanceDest — баланс получателя после

Колонка step (временной шаг) необязательна — если отсутствует, устанавливается значение по умолчанию 1.

**Пример CSV файла**

```csv
type,amount,oldbalanceOrg,newbalanceOrig,oldbalanceDest,newbalanceDest
CASH_OUT,10000.0,50000.0,40000.0,0.0,10000.0
TRANSFER,500000.0,500000.0,0.0,0.0,500000.0
PAYMENT,5000.0,10000.0,5000.0,2000.0,7000.0
```

**Процесс обработки**

1. Загрузка файла и предпросмотр первых 10 строк
2. После нажатия "Анализировать все транзакции":
   - Создание дополнительных признаков для каждой транзакции
   - Пакетное предсказание моделью
   - Формирование статистики

**Результаты анализа**
- Всего транзакций — общее количество загруженных записей
- Выявлено мошеннических — количество транзакций, классифицированных как мошеннические
- Легитимных — количество безопасных транзакций
- Средняя вероятность мошенничества

**Визуализация**
- Круговая диаграмма распределения предсказаний (легитимные/мошеннические)
- Гистограмма распределения вероятностей мошенничества

**Экспорт результатов**

Таблица с подозрительными транзакциями отображается с возможностью сортировки. Кнопка "Скачать результаты (CSV)" позволяет сохранить полный анализ с колонками isFraud_Predicted и Fraud_Probability.

### Страница "Статистика модели"

Страница содержит подробную информацию о модели и результатах обучения.

**Метрики модели**

| Метрика | Значение | Описание |
|---------|----------|----------|
| Accuracy | 99.98% | Общая точность предсказаний |
| Precision | 85.41% | Доля истинных мошеннических среди предсказанных |
| Recall | 99.76% | Доля найденных мошеннических среди всех мошеннических |
| F1-Score | 92.03% | Гармоническое среднее Precision и Recall |
| ROC-AUC | 99.98% | Площадь под ROC-кривой |

**Информация об обучении**
- Название модели: XGBoost
- Размер обучающей выборки: 5,090,096 записей
- Размер тестовой выборки: 1,272,524 записей
- Доля мошенничества: 0.13%
- Техника балансировки: SMOTE (10%)
- Валидация: Stratified Split (80/20)

**Важность признаков**

Горизонтальная столбчатая диаграмма показывает вклад каждого признака в предсказание модели. Наиболее важные:
1. type_encoded — тип транзакции
2. amount — сумма транзакции
3. errorBalanceOrig — ошибка баланса отправителя
4. oldbalanceOrg — начальный баланс

**Графики оценки модели**
- Confusion Matrix (матрица ошибок)
- ROC-кривая
- Precision-Recall кривая

### Страница "О системе"

Информация о проекте, команде разработчиков и контактные данные.

---

## REST API Документация

### Обзор

API предоставляет программный доступ к функциям системы для интеграции с внешними сервисами. Базовый URL для локального запуска: `http://localhost:8000`

### Аутентификация

API не требует аутентификации для демонстрационного режима. В продакшене рекомендуется добавить API ключи или JWT токены.

### Формат данных

Все запросы и ответы используют формат JSON. Для POST запросов необходимо указывать заголовок `Content-Type: application/json`.

---

### Эндпоинты

#### GET /

**Описание:** Корневой эндпоинт с информацией об API.

**Ответ:**

```json
{
    "message": "Fraud Detection API",
    "version": "1.0.0",
    "docs": "/docs",
    "endpoints": {
        "predict": "/predict",
        "batch": "/batch",
        "upload": "/upload",
        "model_info": "/model/info",
        "health": "/health"
    }
}
```

---

#### GET /health

**Описание:** Проверка работоспособности API и загрузки модели.

**Ответ:**

```json
{
    "status": "healthy",
    "model_loaded": true,
    "timestamp": "2026-02-19T18:00:00"
}
```

**Поля ответа:**
- `status` — строка, "healthy" если модель загружена, иначе "degraded"
- `model_loaded` — булево значение, загружена ли модель
- `timestamp` — строка, время запроса в ISO формате

---

#### POST /predict

**Описание:** Предсказание мошенничества для одной транзакции.

**Входные данные (JSON):**

| Поле | Тип | Обязательно | Описание |
|------|-----|-------------|----------|
| step | integer | Нет | Временной шаг (по умолчанию 1) |
| type | string | Да | Тип транзакции: CASH_OUT, CASH_IN, TRANSFER, PAYMENT, DEBIT |
| amount | float | Да | Сумма транзакции (>= 0) |
| oldbalanceOrg | float | Да | Баланс отправителя до (>= 0) |
| newbalanceOrig | float | Да | Баланс отправителя после (>= 0) |
| oldbalanceDest | float | Да | Баланс получателя до (>= 0) |
| newbalanceDest | float | Да | Баланс получателя после (>= 0) |

**Пример запроса:**

```json
{
    "type": "TRANSFER",
    "amount": 500000.0,
    "oldbalanceOrg": 500000.0,
    "newbalanceOrig": 0.0,
    "oldbalanceDest": 0.0,
    "newbalanceDest": 500000.0
}
```

**Ответ:**

```json
{
    "is_fraud": true,
    "fraud_probability": 0.9999890327453613,
    "legitimate_probability": 1.0967254638671875e-05,
    "risk_level": "HIGH",
    "timestamp": "2026-02-19T18:00:00"
}
```

**Поля ответа:**
- `is_fraud` — булево значение, является ли транзакция мошеннической
- `fraud_probability` — float, вероятность мошенничества (0.0 - 1.0)
- `legitimate_probability` — float, вероятность легитимности (1 - fraud_probability)
- `risk_level` — строка, уровень риска: LOW (< 30%), MEDIUM (30-70%), HIGH (> 70%)
- `timestamp` — строка, время обработки запроса

**Коды ошибок:**
- 400 — Неверный тип транзакции или некорректные данные
- 503 — Модель не загружена

---

#### POST /batch

**Описание:** Пакетное предсказание для списка транзакций (до 10,000 за запрос).

**Входные данные (JSON array):**

Массив объектов транзакций. Формат каждой транзакции аналогичен `/predict`.

**Пример запроса:**

```json
[
    {
        "type": "CASH_OUT",
        "amount": 10000.0,
        "oldbalanceOrg": 50000.0,
        "newbalanceOrig": 40000.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 10000.0
    },
    {
        "type": "TRANSFER",
        "amount": 500000.0,
        "oldbalanceOrg": 500000.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 500000.0
    }
]
```

**Ответ:**

```json
{
    "total_transactions": 2,
    "fraud_detected": 1,
    "fraud_rate": 0.5,
    "predictions": [
        {
            "index": 0,
            "is_fraud": false,
            "fraud_probability": 0.00004
        },
        {
            "index": 1,
            "is_fraud": true,
            "fraud_probability": 0.99999
        }
    ]
}
```

**Поля ответа:**
- `total_transactions` — integer, общее количество транзакций
- `fraud_detected` — integer, количество выявленных мошеннических
- `fraud_rate` — float, доля мошеннических транзакций
- `predictions` — массив результатов для каждой транзакции

**Коды ошибок:**
- 400 — Превышен лимит транзакций (> 10000) или некорректные данные
- 503 — Модель не загружена

---

#### POST /upload

**Описание:** Загрузка CSV файла для массового анализа.

**Входные данные:**

Multipart form с файлом (ключ "file").

**Требования к CSV файлу:**

Обязательные колонки:
- type — тип транзакции
- amount — сумма
- oldbalanceOrg — баланс отправителя до
- newbalanceOrig — баланс отправителя после
- oldbalanceDest — баланс получателя до
- newbalanceDest — баланс получателя после

Колонка step необязательна.

**Ответ:**

```json
{
    "filename": "transactions.csv",
    "total_transactions": 100,
    "fraud_detected": 5,
    "fraud_rate": 0.05,
    "results": [
        {
            "isFraud_Predicted": 0,
            "Fraud_Probability": 0.0001
        },
        ...
    ],
    "statistics": {
        "avg_fraud_probability": 0.05,
        "max_fraud_probability": 0.9999,
        "min_fraud_probability": 0.0001
    }
}
```

**Коды ошибок:**
- 400 — Файл не является CSV или отсутствуют обязательные колонки
- 503 — Модель не загружена

---

#### GET /model/info

**Описание:** Получение информации о модели.

**Ответ:**

```json
{
    "model_name": "XGBoost",
    "metrics": {
        "accuracy": 0.9997768214980621,
        "precision": 0.8540906722251173,
        "recall": 0.9975654290931223,
        "f1": 0.9202695115103874,
        "roc_auc": 0.9998430854980441
    },
    "feature_count": 14,
    "training_samples": 5090096,
    "test_samples": 1272524
}
```

---

#### GET /model/features

**Описание:** Получение списка всех признаков, используемых моделью.

**Ответ:**

```json
{
    "feature_count": 14,
    "features": [
        "step",
        "type_encoded",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "errorBalanceOrig",
        "errorBalanceDest",
        "isZeroBalanceAfter",
        "amountToBalanceRatio",
        "hour",
        "day",
        "destEmptyBefore"
    ]
}
```

---

#### GET /model/importance

**Описание:** Получение важности каждого признака для предсказания.

**Ответ:**

```json
{
    "type_encoded": 0.35,
    "amount": 0.25,
    "errorBalanceOrig": 0.15,
    "oldbalanceOrg": 0.10,
    "newbalanceOrig": 0.05,
    "oldbalanceDest": 0.03,
    "newbalanceDest": 0.02,
    "errorBalanceDest": 0.02,
    "isZeroBalanceAfter": 0.01,
    "amountToBalanceRatio": 0.01,
    "hour": 0.005,
    "day": 0.003,
    "destEmptyBefore": 0.002,
    "step": 0.001
}
```

---

#### GET /docs

**Описание:** Интерактивная документация Swagger UI.

Позволяет:
- Просмотреть все доступные эндпоинты
- Увидеть формат входных и выходных данных
- Выполнить тестовые запросы прямо в браузере
- Скачать OpenAPI спецификацию

---

#### GET /redoc

**Описание:** Альтернативная документация ReDoc.

---

## Примеры использования API

### cURL

**Предсказание для одной транзакции:**

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "type": "TRANSFER",
    "amount": 500000,
    "oldbalanceOrg": 500000,
    "newbalanceOrig": 0,
    "oldbalanceDest": 0,
    "newbalanceDest": 500000
  }'
```

**Проверка здоровья:**

```bash
curl "http://localhost:8000/health"
```

**Информация о модели:**

```bash
curl "http://localhost:8000/model/info"
```

### Python (requests)

```python
import requests

# Базовый URL
BASE_URL = "http://localhost:8000"

# Предсказание для одной транзакции
def predict_transaction(type, amount, oldbalanceOrg, newbalanceOrig, 
                        oldbalanceDest, newbalanceDest):
    response = requests.post(
        f"{BASE_URL}/predict",
        json={
            "type": type,
            "amount": amount,
            "oldbalanceOrg": oldbalanceOrg,
            "newbalanceOrig": newbalanceOrig,
            "oldbalanceDest": oldbalanceDest,
            "newbalanceDest": newbalanceDest
        }
    )
    return response.json()

# Пример использования
result = predict_transaction(
    type="TRANSFER",
    amount=500000,
    oldbalanceOrg=500000,
    newbalanceOrig=0,
    oldbalanceDest=0,
    newbalanceDest=500000
)

print(f"Мошенничество: {result['is_fraud']}")
print(f"Вероятность: {result['fraud_probability']:.2%}")
print(f"Уровень риска: {result['risk_level']}")
```

**Пакетное предсказание:**

```python
import requests

transactions = [
    {
        "type": "CASH_OUT",
        "amount": 10000.0,
        "oldbalanceOrg": 50000.0,
        "newbalanceOrig": 40000.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 10000.0
    },
    {
        "type": "TRANSFER",
        "amount": 500000.0,
        "oldbalanceOrg": 500000.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 500000.0
    }
]

response = requests.post(
    "http://localhost:8000/batch",
    json=transactions
)

result = response.json()
print(f"Всего: {result['total_transactions']}")
print(f"Мошеннических: {result['fraud_detected']}")
print(f"Доля: {result['fraud_rate']:.2%}")
```

**Загрузка CSV файла:**

```python
import requests

with open("transactions.csv", "rb") as f:
    response = requests.post(
        "http://localhost:8000/upload",
        files={"file": f}
    )

result = response.json()
print(f"Файл: {result['filename']}")
print(f"Всего транзакций: {result['total_transactions']}")
print(f"Выявлено мошеннических: {result['fraud_detected']}")
```

### JavaScript (fetch)

```javascript
// Предсказание для одной транзакции
async function predictTransaction(transaction) {
    const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(transaction)
    });
    
    return await response.json();
}

// Пример использования
const result = await predictTransaction({
    type: 'TRANSFER',
    amount: 500000,
    oldbalanceOrg: 500000,
    newbalanceOrig: 0,
    oldbalanceDest: 0,
    newbalanceDest: 500000
});

console.log('Мошенничество:', result.is_fraud);
console.log('Вероятность:', (result.fraud_probability * 100).toFixed(2) + '%');
```

---

## Признаки для анализа

### Базовые признаки (из входных данных)

| Признак | Тип | Описание |
|---------|-----|----------|
| step | integer | Временной шаг (часы с начала наблюдения) |
| type | string | Тип транзакции |
| amount | float | Сумма транзакции |
| oldbalanceOrg | float | Баланс отправителя до транзакции |
| newbalanceOrig | float | Баланс отправителя после транзакции |
| oldbalanceDest | float | Баланс получателя до транзакции |
| newbalanceDest | float | Баланс получателя после транзакции |

### Производные признаки (Feature Engineering)

| Признак | Формула | Описание |
|---------|---------|----------|
| errorBalanceOrig | newbalanceOrig + amount - oldbalanceOrg | Ошибка баланса отправителя |
| errorBalanceDest | oldbalanceDest + amount - newbalanceDest | Ошибка баланса получателя |
| isZeroBalanceAfter | 1 if newbalanceOrig == 0 else 0 | Обнулен ли баланс отправителя |
| amountToBalanceRatio | amount / (oldbalanceOrg + 1) | Отношение суммы к балансу |
| hour | step % 24 | Час дня |
| day | step // 24 | День с начала наблюдения |
| destEmptyBefore | 1 if oldbalanceDest == 0 else 0 | Пустой ли счет получателя |

---

## Типы транзакций

| Тип | Описание | Уровень риска |
|-----|----------|---------------|
| CASH_OUT | Снятие наличных средств | Высокий |
| TRANSFER | Перевод между счетами | Высокий |
| CASH_IN | Внесение наличных средств | Низкий |
| PAYMENT | Платеж за услуги | Низкий |
| DEBIT | Дебетовая операция | Средний |

**Примечание:** Мошенничество обнаруживается только в типах TRANSFER и CASH_OUT согласно анализу датасета.

---

## Метрики модели

| Метрика | Значение | Описание |
|---------|----------|----------|
| Accuracy | 99.98% | Доля правильных предсказаний среди всех |
| Precision | 85.41% | Доля истинных мошеннических среди предсказанных как мошеннические |
| Recall | 99.76% | Доля найденных мошеннических среди всех мошеннических |
| F1-Score | 92.03% | Гармоническое среднее Precision и Recall |
| ROC-AUC | 99.98% | Площадь под ROC-кривой |

**Интерпретация:**

Высокий Recall (99.76%) означает, что система пропускает очень мало мошеннических транзакций. Это критически важно для финансовой безопасности, так как пропущенное мошенничество может стоить дорого.

Precision (85.41%) означает, что около 15% транзакций, помеченных как мошеннические, на самом деле легитимны. Это приемлемый уровень ложных срабатываний для ручной проверки.

---

## Обучение модели

### Параметры XGBoost

```python
XGBClassifier(
    n_estimators=100,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)
```

### Пайплайн обучения

1. **Загрузка данных** — чтение CSV с 6.3 млн транзакций
2. **EDA** — анализ распределений, корреляций, пропусков
3. **Feature Engineering** — создание 7 производных признаков
4. **Разделение** — Stratified Split 80/20
5. **Балансировка** — SMOTE до 10% миноритарного класса
6. **Масштабирование** — StandardScaler
7. **Обучение** — XGBoost на обучающей выборке
8. **Оценка** — метрики на тестовой выборке
9. **Сохранение** — joblib сериализация

### Запуск обучения

```bash
python train_model.py
```

Результат:
- `models/` — сохраненная модель и компоненты
- `plots/` — графики анализа и оценки

---

## Деплой

Проект настроен для автоматического деплоя на Render.com.

**Конфигурация (render.yaml):**

```yaml
services:
  - type: web
    name: fraud-detection-api
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn api:app --host 0.0.0.0 --port $PORT
    plan: free
```

**Ручной деплой:**

1. Создайте аккаунт на render.com
2. Создайте новый Web Service
3. Подключите GitHub репозиторий
4. Укажите Start Command: `uvicorn api:app --host 0.0.0.0 --port $PORT`
5. Добавьте переменную окружения PYTHON_VERSION=3.10.0

### Streamlit Cloud

Для деплоя Streamlit приложения:

1. Создайте аккаунт на streamlit.io
2. Подключите GitHub репозиторий
3. Выберите файл `app.py`
4. Укажите зависимости из `requirements.txt`

---

## Авторы

Проект разработан в рамках хакатона для решения задач финансовой безопасности.
