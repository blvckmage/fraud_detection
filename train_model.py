"""
Fraud Detection Model Training Script
Обучение модели для обнаружения мошеннических транзакций
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_curve, roc_curve, f1_score, accuracy_score,
    precision_score, recall_score
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import warnings
import os

warnings.filterwarnings('ignore')

# Создание директорий
os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)

print("=" * 60)
print("FRAUD DETECTION MODEL TRAINING")
print("=" * 60)

# Загрузка данных
print("\n[1/7] Загрузка данных...")
df = pd.read_csv('PS_20174392719_1491204439457_log.csv')
print(f"Размер датасета: {df.shape[0]:,} записей, {df.shape[1]} признаков")

# EDA - Разведочный анализ данных
print("\n[2/7] Разведочный анализ данных (EDA)...")

# Статистика по целевой переменной
fraud_counts = df['isFraud'].value_counts()
fraud_percentage = df['isFraud'].value_counts(normalize=True) * 100
print(f"\nРаспределение классов:")
print(f"  Легитимные транзакции: {fraud_counts[0]:,} ({fraud_percentage[0]:.3f}%)")
print(f"  Мошеннические транзакции: {fraud_counts[1]:,} ({fraud_percentage[1]:.3f}%)")

# Типы транзакций и их связь с мошенничеством
print("\nТипы транзакций:")
print(df['type'].value_counts())

fraud_by_type = df.groupby('type')['isFraud'].agg(['sum', 'count'])
fraud_by_type['fraud_rate'] = fraud_by_type['sum'] / fraud_by_type['count'] * 100
print("\nУровень мошенничества по типам транзакций:")
print(fraud_by_type.sort_values('fraud_rate', ascending=False))

# Создание графиков EDA
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# График 1: Распределение классов
ax1 = axes[0, 0]
colors = ['#2ecc71', '#e74c3c']
df['isFraud'].value_counts().plot(kind='bar', ax=ax1, color=colors)
ax1.set_title('Распределение классов', fontsize=12, fontweight='bold')
ax1.set_xlabel('Класс (0 = Легитимный, 1 = Мошеннический)')
ax1.set_ylabel('Количество транзакций')
ax1.set_xticklabels(['Легитимный', 'Мошеннический'], rotation=0)
for i, v in enumerate(df['isFraud'].value_counts()):
    ax1.text(i, v + 10000, f'{v:,}', ha='center', fontsize=10)

# График 2: Мошенничество по типам транзакций
ax2 = axes[0, 1]
fraud_by_type['fraud_rate'].sort_values(ascending=True).plot(kind='barh', ax=ax2, color='#3498db')
ax2.set_title('Уровень мошенничества по типам транзакций (%)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Процент мошенничества')

# График 3: Распределение суммы транзакций
ax3 = axes[1, 0]
df[df['isFraud'] == 0]['amount'].hist(bins=100, ax=ax3, alpha=0.7, label='Легитимные', color='#2ecc71', density=True)
df[df['isFraud'] == 1]['amount'].hist(bins=100, ax=ax3, alpha=0.7, label='Мошеннические', color='#e74c3c', density=True)
ax3.set_xlim(0, df['amount'].quantile(0.99))
ax3.set_title('Распределение суммы транзакций', fontsize=12, fontweight='bold')
ax3.set_xlabel('Сумма')
ax3.legend()

# График 4: Корреляция с целевой переменной
ax4 = axes[1, 1]
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols.remove('isFraud')
numeric_cols.remove('isFlaggedFraud')
correlations = df[numeric_cols + ['isFraud']].corr()['isFraud'].drop('isFraud').sort_values(key=abs, ascending=False)
correlations.plot(kind='bar', ax=ax4, color='#9b59b6')
ax4.set_title('Корреляция признаков с мошенничеством', fontsize=12, fontweight='bold')
ax4.set_ylabel('Коэффициент корреляции')

plt.tight_layout()
plt.savefig('plots/eda_analysis.png', dpi=150, bbox_inches='tight')
print("\nГрафики EDA сохранены в plots/eda_analysis.png")

# Инженерия признаков
print("\n[3/7] Инженерия признаков...")

# Создание новых признаков
df['errorBalanceOrig'] = df['newbalanceOrig'] + df['amount'] - df['oldbalanceOrg']
df['errorBalanceDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']

# Признак: равен ли баланс нулю после транзакции
df['isZeroBalanceAfter'] = (df['newbalanceOrig'] == 0).astype(int)

# Признак: отношение суммы к исходному балансу
df['amountToBalanceRatio'] = df['amount'] / (df['oldbalanceOrg'] + 1)

# Признак: час дня (step = 1 час)
df['hour'] = df['step'] % 24
df['day'] = df['step'] // 24

# Признак: транзакция с пустым счетом получателя
df['destEmptyBefore'] = (df['oldbalanceDest'] == 0).astype(int)

# Кодирование категориальных признаков
print("Кодирование категориальных признаков...")
type_encoder = LabelEncoder()
df['type_encoded'] = type_encoder.fit_transform(df['type'])

# Сохранение encoder
joblib.dump(type_encoder, 'models/type_encoder.joblib')

# Выбор признаков для модели
feature_columns = [
    'step', 'type_encoded', 'amount', 
    'oldbalanceOrg', 'newbalanceOrig', 
    'oldbalanceDest', 'newbalanceDest',
    'errorBalanceOrig', 'errorBalanceDest',
    'isZeroBalanceAfter', 'amountToBalanceRatio',
    'hour', 'day', 'destEmptyBefore'
]

X = df[feature_columns]
y = df['isFraud']

print(f"Количество признаков: {len(feature_columns)}")
print(f"Признаки: {feature_columns}")

# Разделение данных
print("\n[4/7] Разделение данных...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Обучающая выборка: {X_train.shape[0]:,} записей")
print(f"Тестовая выборка: {X_test.shape[0]:,} записей")

# Масштабирование признаков
print("\n[5/7] Масштабирование признаков...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Сохранение scaler
joblib.dump(scaler, 'models/scaler.joblib')
joblib.dump(feature_columns, 'models/feature_columns.joblib')

# Применение SMOTE для балансировки классов
print("\nПрименение SMOTE для балансировки классов...")
print(f"До SMOTE - класс 0: {sum(y_train == 0):,}, класс 1: {sum(y_train == 1):,}")
smote = SMOTE(random_state=42, sampling_strategy=0.1)  # Увеличиваем миноритарный класс до 10%
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
print(f"После SMOTE - класс 0: {sum(y_train_resampled == 0):,}, класс 1: {sum(y_train_resampled == 1):,}")

# Обучение моделей
print("\n[6/7] Обучение моделей...")

models = {
    'XGBoost': XGBClassifier(
        n_estimators=100,
        max_depth=8,
        learning_rate=0.1,
        scale_pos_weight=99,
        random_state=42,
        n_jobs=-1,
        eval_metric='auc'
    )
}

results = {}

for name, model in models.items():
    print(f"\nОбучение {name}...")
    model.fit(X_train_resampled, y_train_resampled)
    
    # Предсказания
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Метрики
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  ROC-AUC: {roc_auc:.4f}")

# Выбор лучшей модели
best_model_name = max(results, key=lambda x: results[x]['roc_auc'])
best_model = results[best_model_name]['model']

print(f"\n{'='*60}")
print(f"Лучшая модель: {best_model_name}")
print(f"ROC-AUC: {results[best_model_name]['roc_auc']:.4f}")
print(f"{'='*60}")

# Сохранение лучшей модели
joblib.dump(best_model, 'models/fraud_detection_model.joblib')
print(f"\nМодель сохранена в models/fraud_detection_model.joblib")

# Детальный отчет по лучшей модели
print("\n[7/7] Генерация отчетов и графиков...")

y_pred = results[best_model_name]['y_pred']
y_pred_proba = results[best_model_name]['y_pred_proba']

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Легитимная', 'Мошенническая']))

# Матрица ошибок
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
ax1 = axes[0]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
            xticklabels=['Легитимная', 'Мошенническая'],
            yticklabels=['Легитимная', 'Мошенническая'])
ax1.set_title(f'Матрица ошибок ({best_model_name})', fontsize=12, fontweight='bold')
ax1.set_xlabel('Предсказанный класс')
ax1.set_ylabel('Истинный класс')

# ROC-кривая
ax2 = axes[1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
ax2.plot(fpr, tpr, color='#3498db', lw=2, label=f'ROC curve (AUC = {results[best_model_name]["roc_auc"]:.4f})')
ax2.plot([0, 1], [0, 1], color='#95a5a6', lw=2, linestyle='--')
ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('ROC-кривая', fontsize=12, fontweight='bold')
ax2.legend(loc='lower right')

# Precision-Recall кривая
ax3 = axes[2]
precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
ax3.plot(recall_curve, precision_curve, color='#e74c3c', lw=2)
ax3.set_xlabel('Recall')
ax3.set_ylabel('Precision')
ax3.set_title('Precision-Recall кривая', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('plots/model_evaluation.png', dpi=150, bbox_inches='tight')
print("Графики оценки модели сохранены в plots/model_evaluation.png")

# Feature Importance (если доступно)
if hasattr(best_model, 'feature_importances_'):
    fig, ax = plt.subplots(figsize=(10, 6))
    importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=True)
    
    importance.plot(kind='barh', x='feature', y='importance', ax=ax, color='#9b59b6')
    ax.set_title('Важность признаков', fontsize=12, fontweight='bold')
    ax.set_xlabel('Важность')
    plt.tight_layout()
    plt.savefig('plots/feature_importance.png', dpi=150, bbox_inches='tight')
    print("График важности признаков сохранен в plots/feature_importance.png")

# Сравнение моделей
fig, ax = plt.subplots(figsize=(10, 6))
metrics_df = pd.DataFrame({
    name: {
        'Accuracy': res['accuracy'],
        'Precision': res['precision'],
        'Recall': res['recall'],
        'F1-Score': res['f1'],
        'ROC-AUC': res['roc_auc']
    }
    for name, res in results.items()
}).T

metrics_df.plot(kind='bar', ax=ax, colormap='viridis')
ax.set_title('Сравнение моделей', fontsize=12, fontweight='bold')
ax.set_xlabel('Модель')
ax.set_ylabel('Значение метрики')
ax.legend(loc='lower right')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
plt.tight_layout()
plt.savefig('plots/model_comparison.png', dpi=150, bbox_inches='tight')
print("График сравнения моделей сохранен в plots/model_comparison.png")

# Сохранение информации о модели
model_info = {
    'model_name': best_model_name,
    'feature_columns': feature_columns,
    'metrics': {
        'accuracy': results[best_model_name]['accuracy'],
        'precision': results[best_model_name]['precision'],
        'recall': results[best_model_name]['recall'],
        'f1': results[best_model_name]['f1'],
        'roc_auc': results[best_model_name]['roc_auc']
    },
    'training_samples': X_train.shape[0],
    'test_samples': X_test.shape[0],
    'fraud_ratio_train': y_train.mean(),
    'fraud_ratio_test': y_test.mean()
}
joblib.dump(model_info, 'models/model_info.joblib')

print("\n" + "=" * 60)
print("ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
print("=" * 60)
print(f"\nСохраненные файлы:")
print(f"  - models/fraud_detection_model.joblib (обученная модель)")
print(f"  - models/scaler.joblib (масштабирование признаков)")
print(f"  - models/type_encoder.joblib (кодировщик типов)")
print(f"  - models/feature_columns.joblib (список признаков)")
print(f"  - models/model_info.joblib (информация о модели)")
print(f"  - plots/eda_analysis.png (анализ данных)")
print(f"  - plots/model_evaluation.png (оценка модели)")
print(f"  - plots/feature_importance.png (важность признаков)")
print(f"  - plots/model_comparison.png (сравнение моделей)")