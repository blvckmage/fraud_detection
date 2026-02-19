"""
Fraud Detection Streamlit Application
Веб-интерфейс для обнаружения мошеннических транзакций
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
from datetime import datetime

# Настройка страницы
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Кастомные стили
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .fraud-alert {
        background-color: #ff4444;
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .legitimate {
        background-color: #44aa44;
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .stMetric > div {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Загрузка модели и компонентов
@st.cache_resource
def load_model():
    """Загрузка обученной модели и компонентов"""
    try:
        model = joblib.load('models/fraud_detection_model.joblib')
        scaler = joblib.load('models/scaler.joblib')
        type_encoder = joblib.load('models/type_encoder.joblib')
        feature_columns = joblib.load('models/feature_columns.joblib')
        model_info = joblib.load('models/model_info.joblib')
        return model, scaler, type_encoder, feature_columns, model_info
    except FileNotFoundError:
        return None, None, None, None, None

model, scaler, type_encoder, feature_columns, model_info = load_model()

# Боковая панель
with st.sidebar:
    st.title("Fraud Detection")
    st.markdown("---")
    
    page = st.radio(
        "Навигация",
        ["Главная", "Анализ транзакции", "Пакетная обработка", "Статистика модели", "О системе"]
    )

# Главная страница
if page == "Главная":
    st.markdown('<h1 class="main-header">Система обнаружения мошеннических транзакций</h1>', unsafe_allow_html=True)
    
    if model is None:
        st.error("Модель не найдена! Сначала запустите train_model.py для обучения модели.")
        st.code("python train_model.py", language="bash")
        st.stop()
    
    # Описание системы
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.markdown("""
        ### Назначение системы
        
        Данная система использует алгоритмы машинного обучения для выявления 
        подозрительных финансовых транзакций в режиме реального времени.
        
        **Основные возможности:**
        - Анализ отдельных транзакций
        - Пакетная обработка CSV файлов
        - Визуализация статистики
        - Оповещения о мошенничестве
        
        **Используемые технологии:**
        - XGBoost для классификации
        - SMOTE для балансировки классов
        - Feature Engineering для улучшения точности
        """)
    
    with col_right:
        st.markdown("""
        ### Типы транзакций
        
        | Тип | Описание |
        |-----|----------|
        | CASH_OUT | Снятие наличных |
        | CASH_IN | Внесение наличных |
        | TRANSFER | Перевод средств |
        | PAYMENT | Платеж |
        | DEBIT | Дебетовая операция |
        """)
    

# Анализ отдельной транзакции
elif page == "Анализ транзакции":
    st.title("Анализ транзакции")
    
    if model is None:
        st.error("Модель не найдена!")
        st.stop()
    
    st.markdown("Введите данные транзакции для проверки на мошенничество:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Основные данные")
        step = 1  # Значение по умолчанию
        trans_type = st.selectbox("Тип транзакции", ['CASH_OUT', 'CASH_IN', 'TRANSFER', 'PAYMENT', 'DEBIT'])
        amount = st.number_input("Сумма транзакции", min_value=0.0, value=1000.0, step=100.0)
    
    with col2:
        st.subheader("Баланс отправителя")
        oldbalanceOrg = st.number_input("Баланс до транзакции", min_value=0.0, value=5000.0, step=100.0)
        newbalanceOrig = st.number_input("Баланс после транзакции", min_value=0.0, value=4000.0, step=100.0)
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Баланс получателя")
        oldbalanceDest = st.number_input("Баланс получателя до", min_value=0.0, value=0.0, step=100.0)
        newbalanceDest = st.number_input("Баланс получателя после", min_value=0.0, value=1000.0, step=100.0)
    
    # Кнопка анализа
    if st.button("Анализировать транзакцию", type="primary"):
        # Подготовка признаков
        type_encoded = type_encoder.transform([trans_type])[0]
        
        # Инженерия признаков
        errorBalanceOrig = newbalanceOrig + amount - oldbalanceOrg
        errorBalanceDest = oldbalanceDest + amount - newbalanceDest
        isZeroBalanceAfter = 1 if newbalanceOrig == 0 else 0
        amountToBalanceRatio = amount / (oldbalanceOrg + 1)
        hour = step % 24
        day = step // 24
        destEmptyBefore = 1 if oldbalanceDest == 0 else 0
        
        # Создание вектора признаков
        features = np.array([[
            step, type_encoded, amount, oldbalanceOrg, newbalanceOrig,
            oldbalanceDest, newbalanceDest, errorBalanceOrig, errorBalanceDest,
            isZeroBalanceAfter, amountToBalanceRatio, hour, day, destEmptyBefore
        ]])
        
        # Масштабирование
        features_scaled = scaler.transform(features)
        
        # Предсказание
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        
        # Отображение результата
        st.markdown("---")
        
        col_result1, col_result2, col_result3 = st.columns([1, 2, 1])
        
        with col_result2:
            if prediction == 1:
                st.markdown(f"""
                <div class="fraud-alert">
                    ВЫЯВЛЕНО МОШЕННИЧЕСТВО!<br>
                    Вероятность: {probability[1]*100:.2f}%
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="legitimate">
                    ТРАНЗАКЦИЯ ЛЕГИТИМНА<br>
                    Вероятность мошенничества: {probability[1]*100:.2f}%
                </div>
                """, unsafe_allow_html=True)
        
        # Детали анализа
        st.markdown("### Детали анализа")
        
        col_det1, col_det2 = st.columns(2)
        
        with col_det1:
            st.markdown("**Вероятности классов:**")
            prob_df = pd.DataFrame({
                'Класс': ['Легитимная', 'Мошенническая'],
                'Вероятность': [probability[0], probability[1]]
            })
            fig_prob = px.bar(prob_df, x='Класс', y='Вероятность', 
                             color='Класс', 
                             color_discrete_map={'Легитимная': '#2ecc71', 'Мошенническая': '#e74c3c'})
            st.plotly_chart(fig_prob, use_container_width=True)
        
        with col_det2:
            st.markdown("**Индикаторы риска:**")
            risk_factors = []
            
            if amount > 100000:
                risk_factors.append("Крупная сумма транзакции")
            if newbalanceOrig == 0 and amount > 0:
                risk_factors.append("Баланс обнулён после транзакции")
            if trans_type in ['CASH_OUT', 'TRANSFER']:
                risk_factors.append("Высокорисковый тип транзакции")
            if oldbalanceDest == 0:
                risk_factors.append("Получатель не имел средств на счёте")
            if abs(errorBalanceOrig) > 1000:
                risk_factors.append("Несоответствие баланса отправителя")
            
            if risk_factors:
                for factor in risk_factors:
                    st.warning(factor)
            else:
                st.success("Явных признаков мошенничества не обнаружено")

# Пакетная обработка
elif page == "Пакетная обработка":
    st.title("Пакетная обработка транзакций")
    
    if model is None:
        st.error("Модель не найдена!")
        st.stop()
    
    st.markdown("Загрузите CSV файл с транзакциями для пакетного анализа")
    
    # Пример формата
    with st.expander("Требования к формату файла"):
        st.markdown("""
        Файл должен содержать следующие колонки:
        - `step` - шаг (время)
        - `type` - тип транзакции (CASH_OUT, CASH_IN, TRANSFER, PAYMENT, DEBIT)
        - `amount` - сумма транзакции
        - `oldbalanceOrg` - баланс отправителя до
        - `newbalanceOrig` - баланс отправителя после
        - `oldbalanceDest` - баланс получателя до
        - `newbalanceDest` - баланс получателя после
        """)
    
    uploaded_file = st.file_uploader("Выберите CSV файл", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"Загружено {len(df)} транзакций")
            
            # Отображение данных
            st.markdown("### Предпросмотр данных")
            st.dataframe(df.head(10), use_container_width=True)
            
            if st.button("Анализировать все транзакции", type="primary"):
                with st.spinner("Анализ транзакций..."):
                    # Подготовка данных
                    df_analysis = df.copy()
                    
                    # Инженерия признаков
                    df_analysis['errorBalanceOrig'] = df_analysis['newbalanceOrig'] + df_analysis['amount'] - df_analysis['oldbalanceOrg']
                    df_analysis['errorBalanceDest'] = df_analysis['oldbalanceDest'] + df_analysis['amount'] - df_analysis['newbalanceDest']
                    df_analysis['isZeroBalanceAfter'] = (df_analysis['newbalanceOrig'] == 0).astype(int)
                    df_analysis['amountToBalanceRatio'] = df_analysis['amount'] / (df_analysis['oldbalanceOrg'] + 1)
                    df_analysis['hour'] = df_analysis['step'] % 24
                    df_analysis['day'] = df_analysis['step'] // 24
                    df_analysis['destEmptyBefore'] = (df_analysis['oldbalanceDest'] == 0).astype(int)
                    
                    # Кодирование типа
                    df_analysis['type_encoded'] = df_analysis['type'].apply(
                        lambda x: type_encoder.transform([x])[0] if x in type_encoder.classes_ else -1
                    )
                    
                    # Выбор признаков
                    features = df_analysis[feature_columns].values
                    features_scaled = scaler.transform(features)
                    
                    # Предсказания
                    predictions = model.predict(features_scaled)
                    probabilities = model.predict_proba(features_scaled)[:, 1]
                    
                    # Добавление результатов
                    df['isFraud_Predicted'] = predictions
                    df['Fraud_Probability'] = probabilities
                    
                    # Статистика
                    st.markdown("### Результаты анализа")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    fraud_count = sum(predictions)
                    total_count = len(predictions)
                    
                    with col1:
                        st.metric("Всего транзакций", f"{total_count:,}")
                    with col2:
                        st.metric("Выявлено мошеннических", f"{fraud_count:,}", delta=f"{fraud_count/total_count*100:.2f}%")
                    with col3:
                        st.metric("Легитимных", f"{total_count - fraud_count:,}")
                    with col4:
                        avg_fraud_prob = df['Fraud_Probability'].mean()
                        st.metric("Ср. вероятность мошенничества", f"{avg_fraud_prob:.4f}")
                    
                    # Визуализация
                    col_vis1, col_vis2 = st.columns(2)
                    
                    with col_vis1:
                        fig_pie = px.pie(
                            values=[total_count - fraud_count, fraud_count],
                            names=['Легитимные', 'Мошеннические'],
                            title='Распределение предсказаний',
                            color=['Легитимные', 'Мошеннические'],
                            color_discrete_map={'Легитимные': '#2ecc71', 'Мошеннические': '#e74c3c'}
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
                    
                    with col_vis2:
                        fig_hist = px.histogram(
                            df, x='Fraud_Probability', 
                            title='Распределение вероятностей мошенничества',
                            nbins=50,
                            color_discrete_sequence=['#3498db']
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)
                    
                    # Таблица с подозрительными транзакциями
                    st.markdown("### Подозрительные транзакции")
                    fraud_df = df[df['isFraud_Predicted'] == 1].sort_values('Fraud_Probability', ascending=False)
                    
                    if len(fraud_df) > 0:
                        st.dataframe(fraud_df, use_container_width=True)
                        
                        # Скачивание результатов
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Скачать результаты (CSV)",
                            data=csv,
                            file_name='fraud_detection_results.csv',
                            mime='text/csv'
                        )
                    else:
                        st.success("Мошеннических транзакций не выявлено!")
                        
        except Exception as e:
            st.error(f"Ошибка при обработке файла: {str(e)}")

# Статистика модели
elif page == "Статистика модели":
    st.title("Статистика модели")
    
    if model is None:
        st.error("Модель не найдена!")
        st.stop()
    
    # Информация о модели
    st.markdown("### Метрики модели")
    
    metrics_df = pd.DataFrame({
        'Метрика': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
        'Значение': [
            model_info['metrics']['accuracy'],
            model_info['metrics']['precision'],
            model_info['metrics']['recall'],
            model_info['metrics']['f1'],
            model_info['metrics']['roc_auc']
        ]
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    with col2:
        fig_metrics = px.bar(
            metrics_df, x='Метрика', y='Значение',
            title='Метрики качества модели',
            color='Значение',
            color_continuous_scale='viridis'
        )
        fig_metrics.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig_metrics, use_container_width=True)
    
    # Информация об обучении
    st.markdown("### Информация об обучении")
    
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.markdown(f"""
        **Модель:** {model_info['model_name']}  
        **Обучающая выборка:** {model_info['training_samples']:,} записей  
        **Тестовая выборка:** {model_info['test_samples']:,} записей  
        **Доля мошенничества (train):** {model_info['fraud_ratio_train']*100:.4f}%  
        **Доля мошенничества (test):** {model_info['fraud_ratio_test']*100:.4f}%
        """)
    
    with col_info2:
        st.markdown(f"""
        **Используемые признаки:** {len(feature_columns)}  
        **Техника балансировки:** SMOTE (10%)  
        **Масштабирование:** StandardScaler  
        **Валидация:** Stratified Split (80/20)
        """)
    
    # Важность признаков
    if hasattr(model, 'feature_importances_'):
        st.markdown("### Важность признаков")
        
        importance_df = pd.DataFrame({
            'Признак': feature_columns,
            'Важность': model.feature_importances_
        }).sort_values('Важность', ascending=True)
        
        fig_importance = go.Figure(go.Bar(
            x=importance_df['Важность'],
            y=importance_df['Признак'],
            orientation='h',
            marker=dict(color=importance_df['Важность'], colorscale='plasma')
        ))
        fig_importance.update_layout(
            title='Важность признаков модели',
            xaxis_title='Важность',
            yaxis_title='Признак'
        )
        st.plotly_chart(fig_importance, use_container_width=True)
    
    # Графики (если существуют)
    if os.path.exists('plots/model_evaluation.png'):
        st.markdown("### Графики оценки модели")
        st.image('plots/model_evaluation.png', use_container_width=True)
    
    if os.path.exists('plots/eda_analysis.png'):
        st.markdown("### Анализ данных (EDA)")
        st.image('plots/eda_analysis.png', use_container_width=True)

# О системе
elif page == "О системе":
    st.title("О системе Fraud Detection")
    
    st.markdown("""
    ## Система обнаружения мошеннических транзакций
    
    ### Описание
    
    Данная система разработана для обнаружения мошеннических финансовых транзакций 
    с использованием алгоритмов машинного обучения. Система анализирует различные 
    параметры транзакции и определяет вероятность мошенничества.
    
    ### Технологии
    
    - **Язык программирования:** Python 3.x
    - **Машинное обучение:** XGBoost
    - **Балансировка данных:** SMOTE (imbalanced-learn)
    - **Веб-интерфейс:** Streamlit
    - **API:** FastAPI
    - **Обработка данных:** Pandas, NumPy
    - **Визуализация:** Plotly, Matplotlib, Seaborn
    
    ### Используемые признаки
    
    Для анализа транзакций используются следующие признаки:
    
    | Признак | Описание |
    |---------|----------|
    | step | Временной шаг (часы) |
    | type | Тип транзакции |
    | amount | Сумма транзакции |
    | oldbalanceOrg | Баланс отправителя до |
    | newbalanceOrig | Баланс отправителя после |
    | oldbalanceDest | Баланс получателя до |
    | newbalanceDest | Баланс получателя после |
    | errorBalanceOrig | Ошибка баланса отправителя (newbalanceOrig + amount - oldbalanceOrg) |
    | errorBalanceDest | Ошибка баланса получателя (oldbalanceDest + amount - newbalanceDest) |
    | isZeroBalanceAfter | Обнулён ли баланс после |
    | amountToBalanceRatio | Отношение суммы к балансу |
    | hour | Час дня |
    | day | День |
    | destEmptyBefore | Пустой ли счёт получателя |
    
    ### API
    
    Для доступа к модели извне используйте API:
    
    ```bash
    # Запуск API сервера
    uvicorn api:app --reload --host 0.0.0.0 --port 8000
    ```
    
    ### Авторы
    
    Проект разработан в рамках хакатона для решения задач финансовой безопасности.
    """)