"""
Fraud Detection FastAPI
REST API для доступа к модели обнаружения мошеннических транзакций
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import pandas as pd
import numpy as np
import joblib
import io
from datetime import datetime

# Создание приложения
app = FastAPI(
    title="Fraud Detection API",
    description="API для обнаружения мошеннических финансовых транзакций с использованием машинного обучения",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальные переменные для модели
model = None
scaler = None
type_encoder = None
feature_columns = None
model_info = None


def load_model():
    """Загрузка модели и компонентов"""
    global model, scaler, type_encoder, feature_columns, model_info
    try:
        model = joblib.load('models/fraud_detection_model.joblib')
        scaler = joblib.load('models/scaler.joblib')
        type_encoder = joblib.load('models/type_encoder.joblib')
        feature_columns = joblib.load('models/feature_columns.joblib')
        model_info = joblib.load('models/model_info.joblib')
        return True
    except FileNotFoundError:
        return False


# Загрузка модели при старте
@app.on_event("startup")
async def startup_event():
    if not load_model():
        print("Warning: Model not found. Please run train_model.py first.")


# Модели данных
class Transaction(BaseModel):
    """Модель данных транзакции"""
    step: int = Field(default=1, description="Временной шаг (часы, по умолчанию 1)", ge=1, example=1)
    type: str = Field(..., description="Тип транзакции", example="CASH_OUT")
    amount: float = Field(..., description="Сумма транзакции", ge=0, example=1000.0)
    oldbalanceOrg: float = Field(..., description="Баланс отправителя до", ge=0, example=5000.0)
    newbalanceOrig: float = Field(..., description="Баланс отправителя после", ge=0, example=4000.0)
    oldbalanceDest: float = Field(..., description="Баланс получателя до", ge=0, example=0.0)
    newbalanceDest: float = Field(..., description="Баланс получателя после", ge=0, example=1000.0)

    class Config:
        json_schema_extra = {
            "example": {
                "step": 1,
                "type": "CASH_OUT",
                "amount": 1000.0,
                "oldbalanceOrg": 5000.0,
                "newbalanceOrig": 4000.0,
                "oldbalanceDest": 0.0,
                "newbalanceDest": 1000.0
            }
        }


class PredictionResponse(BaseModel):
    """Ответ с предсказанием"""
    is_fraud: bool = Field(..., description="Является ли транзакция мошеннической")
    fraud_probability: float = Field(..., description="Вероятность мошенничества")
    legitimate_probability: float = Field(..., description="Вероятность легитимности")
    risk_level: str = Field(..., description="Уровень риска")
    timestamp: str = Field(..., description="Время предсказания")


class BatchPredictionResponse(BaseModel):
    """Ответ для пакетного предсказания"""
    total_transactions: int
    fraud_detected: int
    fraud_rate: float
    predictions: List[dict]


class ModelInfo(BaseModel):
    """Информация о модели"""
    model_name: str
    metrics: dict
    feature_count: int
    training_samples: int
    test_samples: int


class HealthResponse(BaseModel):
    """Ответ для проверки здоровья"""
    status: str
    model_loaded: bool
    timestamp: str


# Эндпоинты API
@app.get("/", tags=["Root"])
async def root():
    """Корневой эндпоинт"""
    return {
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


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Проверка состояния API"""
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_single(transaction: Transaction):
    """
    Предсказание мошенничества для одной транзакции
    
    Возвращает вероятность мошенничества и классификацию транзакции.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train the model first.")
    
    # Проверка типа транзакции
    if transaction.type not in type_encoder.classes_:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid transaction type. Valid types: {list(type_encoder.classes_)}"
        )
    
    try:
        # Инженерия признаков
        type_encoded = type_encoder.transform([transaction.type])[0]
        errorBalanceOrig = transaction.newbalanceOrig + transaction.amount - transaction.oldbalanceOrg
        errorBalanceDest = transaction.oldbalanceDest + transaction.amount - transaction.newbalanceDest
        isZeroBalanceAfter = 1 if transaction.newbalanceOrig == 0 else 0
        amountToBalanceRatio = transaction.amount / (transaction.oldbalanceOrg + 1)
        hour = transaction.step % 24
        day = transaction.step // 24
        destEmptyBefore = 1 if transaction.oldbalanceDest == 0 else 0
        
        # Создание вектора признаков
        features = np.array([[
            transaction.step, type_encoded, transaction.amount,
            transaction.oldbalanceOrg, transaction.newbalanceOrig,
            transaction.oldbalanceDest, transaction.newbalanceDest,
            errorBalanceOrig, errorBalanceDest,
            isZeroBalanceAfter, amountToBalanceRatio,
            hour, day, destEmptyBefore
        ]])
        
        # Масштабирование
        features_scaled = scaler.transform(features)
        
        # Предсказание
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        # Определение уровня риска
        fraud_prob = probabilities[1]
        if fraud_prob < 0.3:
            risk_level = "LOW"
        elif fraud_prob < 0.7:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"
        
        return PredictionResponse(
            is_fraud=bool(prediction),
            fraud_probability=float(probabilities[1]),
            legitimate_probability=float(probabilities[0]),
            risk_level=risk_level,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(transactions: List[Transaction]):
    """
    Пакетное предсказание для списка транзакций
    
    Принимает список транзакций и возвращает предсказания для каждой.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train the model first.")
    
    if len(transactions) > 10000:
        raise HTTPException(status_code=400, detail="Maximum 10000 transactions per batch")
    
    try:
        # Подготовка данных
        features_list = []
        for t in transactions:
            type_encoded = type_encoder.transform([t.type])[0] if t.type in type_encoder.classes_ else -1
            errorBalanceOrig = t.newbalanceOrig + t.amount - t.oldbalanceOrg
            errorBalanceDest = t.oldbalanceDest + t.amount - t.newbalanceDest
            isZeroBalanceAfter = 1 if t.newbalanceOrig == 0 else 0
            amountToBalanceRatio = t.amount / (t.oldbalanceOrg + 1)
            hour = t.step % 24
            day = t.step // 24
            destEmptyBefore = 1 if t.oldbalanceDest == 0 else 0
            
            features_list.append([
                t.step, type_encoded, t.amount,
                t.oldbalanceOrg, t.newbalanceOrig,
                t.oldbalanceDest, t.newbalanceDest,
                errorBalanceOrig, errorBalanceDest,
                isZeroBalanceAfter, amountToBalanceRatio,
                hour, day, destEmptyBefore
            ])
        
        features = np.array(features_list)
        features_scaled = scaler.transform(features)
        
        # Предсказания
        predictions = model.predict(features_scaled)
        probabilities = model.predict_proba(features_scaled)[:, 1]
        
        # Формирование результатов
        results = []
        for i, t in enumerate(transactions):
            results.append({
                "index": i,
                "is_fraud": bool(predictions[i]),
                "fraud_probability": float(probabilities[i])
            })
        
        fraud_count = sum(predictions)
        
        return BatchPredictionResponse(
            total_transactions=len(transactions),
            fraud_detected=int(fraud_count),
            fraud_rate=float(fraud_count / len(transactions)),
            predictions=results
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.post("/upload", tags=["Prediction"])
async def upload_csv(file: UploadFile = File(...)):
    """
    Загрузка CSV файла с транзакциями для анализа
    
    Файл должен содержать колонки: step, type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train the model first.")
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted")
    
    try:
        # Чтение файла
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        
        # Проверка колонок
        required_columns = ['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(status_code=400, detail=f"Missing columns: {missing_columns}")
        
        # Если step не указан, устанавливаем по умолчанию
        if 'step' not in df.columns:
            df['step'] = 1
        
        # Инженерия признаков
        df['errorBalanceOrig'] = df['newbalanceOrig'] + df['amount'] - df['oldbalanceOrg']
        df['errorBalanceDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']
        df['isZeroBalanceAfter'] = (df['newbalanceOrig'] == 0).astype(int)
        df['amountToBalanceRatio'] = df['amount'] / (df['oldbalanceOrg'] + 1)
        df['hour'] = df['step'] % 24
        df['day'] = df['step'] // 24
        df['destEmptyBefore'] = (df['oldbalanceDest'] == 0).astype(int)
        
        # Кодирование типа
        df['type_encoded'] = df['type'].apply(
            lambda x: type_encoder.transform([x])[0] if x in type_encoder.classes_ else -1
        )
        
        # Выбор признаков
        features = df[feature_columns].values
        features_scaled = scaler.transform(features)
        
        # Предсказания
        predictions = model.predict(features_scaled)
        probabilities = model.predict_proba(features_scaled)[:, 1]
        
        # Добавление результатов
        df['isFraud_Predicted'] = predictions
        df['Fraud_Probability'] = probabilities
        
        # Статистика
        fraud_count = int(sum(predictions))
        
        return {
            "filename": file.filename,
            "total_transactions": len(df),
            "fraud_detected": fraud_count,
            "fraud_rate": float(fraud_count / len(df)),
            "results": df[['isFraud_Predicted', 'Fraud_Probability']].to_dict('records')[:100],  # Первые 100 записей
            "statistics": {
                "avg_fraud_probability": float(probabilities.mean()),
                "max_fraud_probability": float(probabilities.max()),
                "min_fraud_probability": float(probabilities.min())
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File processing error: {str(e)}")


@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def get_model_info():
    """Получение информации о модели"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfo(
        model_name=model_info['model_name'],
        metrics=model_info['metrics'],
        feature_count=len(feature_columns),
        training_samples=model_info['training_samples'],
        test_samples=model_info['test_samples']
    )


@app.get("/model/features", tags=["Model"])
async def get_features():
    """Получение списка используемых признаков"""
    if feature_columns is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "feature_count": len(feature_columns),
        "features": feature_columns
    }


@app.get("/model/importance", tags=["Model"])
async def get_feature_importance():
    """Получение важности признаков"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if hasattr(model, 'feature_importances_'):
        importance = dict(zip(feature_columns, model.feature_importances_.tolist()))
        sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        return sorted_importance
    else:
        return {"message": "Feature importance not available for this model type"}


# Запуск сервера
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)