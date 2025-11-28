# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

# Set page config
st.set_page_config(
    page_title="Система прогнозирования абитуриентов",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

class ApplicantPredictionModel:
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.history = None
        self.feature_names = None
    
    def preprocess_data(self, df, is_training=True):
        df = df.copy()
        
        if 'applicant_id' in df.columns:
            df = df.drop('applicant_id', axis=1)
        
        if 'enrolled' in df.columns:
            y = df['enrolled'].values
            X = df.drop('enrolled', axis=1)
        else:
            y = None
            X = df
        
        if is_training:
            self.feature_names = X.columns.tolist()
        
        categorical_cols = X.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if is_training:
                self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col])
            else:
                X[col] = self.label_encoders[col].transform(X[col])
        
        if is_training:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return X_scaled, y
    
    def build_model(self, input_dim):
        model = models.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(128, activation='relu', name='dense_1'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu', name='dense_2'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu', name='dense_3'),
            layers.Dense(1, activation='sigmoid', name='output')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.AUC()]
        )
        
        self.model = model
        return model
    
    def train_model(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, progress_callback=None):
        if self.model is None:
            self.build_model(X_train.shape[1])
        
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=0
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=0
        )
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )
        
        return self.history
    
    def evaluate_model(self, X_test, y_test):
        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        accuracy = accuracy_score(y_test, y_pred)
        auc_roc = roc_auc_score(y_test, y_pred_proba)
        
        return {
            'accuracy': accuracy,
            'auc_roc': auc_roc,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'y_test': y_test
        }
    
    def save_model(self, model_path='applicant_model.h5', scaler_path='scaler.pkl'):
        self.model.save(model_path)
        
        preprocessing_data = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names
        }
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(preprocessing_data, f)
    
    def load_model(self, model_path='applicant_model.h5', scaler_path='scaler.pkl'):
        self.model = keras.models.load_model(model_path)
        
        with open(scaler_path, 'rb') as f:
            preprocessing_data = pickle.load(f)
        
        self.scaler = preprocessing_data['scaler']
        self.label_encoders = preprocessing_data['label_encoders']
        self.feature_names = preprocessing_data['feature_names']
    
    def predict(self, X):
        X_processed, _ = self.preprocess_data(X, is_training=False)
        predictions = self.model.predict(X_processed, verbose=0)
        return (predictions > 0.5).astype(int).flatten(), predictions.flatten()


# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = ApplicantPredictionModel()
if 'data' not in st.session_state:
    st.session_state.data = None
if 'trained' not in st.session_state:
    st.session_state.trained = False
if 'results' not in st.session_state:
    st.session_state.results = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Загрузка данных"

# Header
st.title("🎓 Система прогнозирования поведения абитуриентов")
st.markdown("### Нейронная сеть для предсказания зачисления абитуриентов")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Панель управления")
    
    st.markdown("### 📋 Навигация")
    
    if st.button("📊 Загрузка данных", use_container_width=True, 
                 type="primary" if st.session_state.current_page == "Загрузка данных" else "secondary"):
        st.session_state.current_page = "Загрузка данных"
        st.rerun()
    
    if st.button("🤖 Обучение модели", use_container_width=True,
                 type="primary" if st.session_state.current_page == "Обучение модели" else "secondary"):
        st.session_state.current_page = "Обучение модели"
        st.rerun()
    
    if st.button("📈 Результаты и метрики", use_container_width=True,
                 type="primary" if st.session_state.current_page == "Результаты и метрики" else "secondary"):
        st.session_state.current_page = "Результаты и метрики"
        st.rerun()
    
    if st.button("🔮 Прогнозирование", use_container_width=True,
                 type="primary" if st.session_state.current_page == "Прогнозирование" else "secondary"):
        st.session_state.current_page = "Прогнозирование"
        st.rerun()
    
    st.markdown("---")
    
    st.markdown("### 🔧 Быстрые действия")
    
    if st.button("🔄 Сбросить приложение", use_container_width=True):
        st.session_state.model = ApplicantPredictionModel()
        st.session_state.data = None
        st.session_state.trained = False
        st.session_state.results = None
        st.session_state.current_page = "Загрузка данных"
        st.rerun()
    
    st.markdown("---")
    
    st.markdown("### 📊 Статус системы")
    
    if st.session_state.data is not None:
        st.success(f"✅ Данные загружены: {len(st.session_state.data)} записей")
    else:
        st.info("ℹ️ Данные не загружены")
    
    if st.session_state.trained:
        st.success("✅ Модель обучена")
    else:
        st.info("ℹ️ Модель не обучена")

# Main content
if st.session_state.current_page == "Загрузка данных":
    st.header("📊 Загрузка и просмотр данных")
    
    st.subheader("📁 Загрузка файла с данными")
    uploaded_file = st.file_uploader("Выберите CSV или Excel файл", type=['csv', 'xlsx', 'xls'])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.session_state.data = df
            st.success(f"✅ Файл успешно загружен! Загружено {len(df)} записей")
        except Exception as e:
            st.error(f"❌ Ошибка при загрузке файла: {e}")
    
    st.markdown("---")
    
    if st.session_state.data is not None:
        st.subheader("📋 Обзор датасета")
        
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Просмотр данных", "📈 Статистика", "🎨 Визуализации", "💾 Скачать"])
        
        with tab1:
            st.dataframe(st.session_state.data, use_container_width=True, height=400)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Всего записей", len(st.session_state.data))
            with col2:
                st.metric("Признаков", len(st.session_state.data.columns))
            with col3:
                if 'enrolled' in st.session_state.data.columns:
                    enrolled_count = st.session_state.data['enrolled'].sum()
                    st.metric("Зачислено", enrolled_count)
            with col4:
                if 'enrolled' in st.session_state.data.columns:
                    not_enrolled = len(st.session_state.data) - enrolled_count
                    st.metric("Не зачислено", not_enrolled)
        
        with tab2:
            st.write("**Описательная статистика**")
            st.dataframe(st.session_state.data.describe(), use_container_width=True)
            
            st.write("**Пропущенные значения**")
            missing = st.session_state.data.isnull().sum()
            if missing.sum() == 0:
                st.success("✅ Пропущенных значений не обнаружено!")
            else:
                st.dataframe(missing[missing > 0], use_container_width=True)

        
        with tab3:
            st.write("**Распределение данных**")
            
            # Numeric columns distribution
            numeric_cols = st.session_state.data.select_dtypes(include=[np.number]).columns.tolist()
            if 'applicant_id' in numeric_cols:
                numeric_cols.remove('applicant_id')
            
            if numeric_cols:
                selected_col = st.selectbox("Выберите столбец для отображения", numeric_cols)
                
                fig = px.histogram(st.session_state.data, x=selected_col, nbins=30, 
                                   title=f"Распределение {selected_col}")
                st.plotly_chart(fig, use_container_width=True)
            
            # Categorical distribution
            if 'enrolled' in st.session_state.data.columns:
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.pie(st.session_state.data, names='enrolled', 
                                title='Распределение зачислений',
                                labels={'enrolled': 'Статус'})
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    if 'ad_source' in st.session_state.data.columns:
                        source_counts = st.session_state.data['ad_source'].value_counts()
                        fig = px.bar(x=source_counts.index, y=source_counts.values,
                                    title='Распределение источников рекламы',
                                    labels={'x': 'Источник', 'y': 'Количество'})
                        st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.write("**Скачать датасет**")
            
            csv = st.session_state.data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Скачать как CSV",
                data=csv,
                file_name="applicants_data.csv",
                mime="text/csv",
                use_container_width=True
            )

elif st.session_state.current_page == "Обучение модели":
    st.header("🤖 Обучение модели")
    
    if st.session_state.data is None:
        st.warning("⚠️ Пожалуйста, сначала загрузите данные!")
    else:
        st.subheader("Конфигурация обучения")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            epochs = st.slider("Количество эпох", min_value=10, max_value=200, value=50, step=10)
        with col2:
            batch_size = st.selectbox("Размер батча", [16, 32, 64, 128], index=1)
        with col3:
            test_size = st.slider("Размер тестовой выборки (%)", min_value=10, max_value=30, value=15, step=5)
        
        st.markdown("---")
        
        if st.button("🚀 Начать обучение", use_container_width=True, type="primary"):
            with st.spinner("Обучение модели... Это может занять несколько минут."):
                try:
                    # Preprocess data
                    X, y = st.session_state.model.preprocess_data(st.session_state.data, is_training=True)
                    
                    # Split data
                    X_temp, X_test, y_temp, y_test = train_test_split(
                        X, y, test_size=test_size/100, random_state=42
                    )
                    X_train, X_val, y_train, y_val = train_test_split(
                        X_temp, y_temp, test_size=0.176, random_state=42
                    )
                    
                    # Progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("Построение архитектуры модели...")
                    progress_bar.progress(10)
                    
                    # Train model
                    status_text.text("Обучение нейронной сети...")
                    progress_bar.progress(30)
                    
                    history = st.session_state.model.train_model(
                        X_train, y_train, X_val, y_val,
                        epochs=epochs,
                        batch_size=batch_size
                    )
                    
                    progress_bar.progress(80)
                    status_text.text("Оценка модели...")
                    
                    # Evaluate
                    results = st.session_state.model.evaluate_model(X_test, y_test)
                    st.session_state.results = results
                    st.session_state.trained = True
                    
                    progress_bar.progress(100)
                    status_text.text("Обучение завершено!")
                    
                    st.success(f"✅ Модель успешно обучена! Точность: {results['accuracy']*100:.2f}%")
                    
                    # Save model
                    st.session_state.model.save_model()
                    
                except Exception as e:
                    st.error(f"❌ Ошибка во время обучения: {e}")
        
        st.markdown("---")
        
        if st.session_state.trained:
            st.subheader("📊 Архитектура модели")
            
            with st.expander("Посмотреть структуру модели"):
                buffer = io.StringIO()
                st.session_state.model.model.summary(print_fn=lambda x: buffer.write(x + '\n'))
                st.text(buffer.getvalue())

elif st.session_state.current_page == "Результаты и метрики":
    st.header("📈 Результаты и метрики")
    
    if not st.session_state.trained:
        st.warning("⚠️ Пожалуйста, сначала обучите модель!")
    else:
        results = st.session_state.results
        
        # Metrics overview
        st.subheader("🎯 Метрики производительности")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Точность (Accuracy)",
                f"{results['accuracy']*100:.2f}%",
                delta=f"{(results['accuracy']-0.7)*100:.2f}%" if results['accuracy'] >= 0.7 else None
            )
        
        with col2:
            st.metric("AUC-ROC", f"{results['auc_roc']:.4f}")
        
        with col3:
            precision = (results['y_pred'] == results['y_test']).sum() / len(results['y_pred'])
            st.metric("Точность (Precision)", f"{precision:.4f}")
        
        with col4:
            from sklearn.metrics import recall_score
            recall = recall_score(results['y_test'], results['y_pred'])
            st.metric("Полнота (Recall)", f"{recall:.4f}")
        
        st.markdown("---")
        
        # Tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["📊 История обучения", "🎯 Матрица ошибок", "📉 ROC-кривая", "📋 Отчет классификации"])
        
        with tab1:
            st.subheader("История обучения")
            
            history = st.session_state.model.history.history
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Точность', 'Потери', 'AUC-ROC', 'Precision & Recall')
            )
            
            # Accuracy
            fig.add_trace(
                go.Scatter(y=history['accuracy'], name='Обучение', mode='lines'),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(y=history['val_accuracy'], name='Валидация', mode='lines'),
                row=1, col=1
            )
            
            # Loss
            fig.add_trace(
                go.Scatter(y=history['loss'], name='Обучение', mode='lines'),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(y=history['val_loss'], name='Валидация', mode='lines'),
                row=1, col=2
            )
            
            # AUC
            fig.add_trace(
                go.Scatter(y=history['auc'], name='Обучение', mode='lines'),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(y=history['val_auc'], name='Валидация', mode='lines'),
                row=2, col=1
            )
            
            # Precision & Recall
            fig.add_trace(
                go.Scatter(y=history['precision'], name='Precision', mode='lines'),
                row=2, col=2
            )
            fig.add_trace(
                go.Scatter(y=history['recall'], name='Recall', mode='lines'),
                row=2, col=2
            )
            
            fig.update_layout(height=700, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Матрица ошибок")
            
            cm = confusion_matrix(results['y_test'], results['y_pred'])
            
            fig = px.imshow(
                cm,
                labels=dict(x="Предсказано", y="Фактически", color="Количество"),
                x=['Не зачислен', 'Зачислен'],
                y=['Не зачислен', 'Зачислен'],
                text_auto=True,
                color_continuous_scale='Blues'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Истинно отрицательные", cm[0, 0])
                st.metric("Ложно положительные", cm[0, 1])
            with col2:
                st.metric("Ложно отрицательные", cm[1, 0])
                st.metric("Истинно положительные", cm[1, 1])
        
        with tab3:
            st.subheader("ROC-кривая")
            
            fpr, tpr, thresholds = roc_curve(results['y_test'], results['y_pred_proba'])
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC = {results["auc_roc"]:.4f})'))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Случайная', line=dict(dash='dash')))
            
            fig.update_layout(
                title='ROC-кривая (Receiver Operating Characteristic)',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.subheader("Отчет классификации")
            
            report = classification_report(
                results['y_test'],
                results['y_pred'],
                target_names=['Не зачислен', 'Зачислен'],
                output_dict=True
            )
            
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.highlight_max(axis=0), use_container_width=True)

elif st.session_state.current_page == "Прогнозирование":
    st.header("🔮 Прогнозирование")
    
    if not st.session_state.trained:
        st.warning("⚠️ Пожалуйста, сначала обучите модель!")
    else:
        st.subheader("Одиночное предсказание")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Возраст", 16, 35, 20)
            gender = st.selectbox("Пол", ['M', 'F'])
            region = st.selectbox("Регион", ['Central', 'North-West', 'South', 'Siberia', 'Far-East'])
            exam_score = st.slider("Балл ЕГЭ", 30.0, 100.0, 65.0, 0.1)
        
        with col2:
            ad_clicks = st.number_input("Кликов по рекламе", 0, 50, 3)
            site_visits = st.number_input("Посещений сайта", 0, 50, 5)
            time_on_site = st.number_input("Время на сайте (минуты)", 0.0, 200.0, 20.0, 0.1)
            ad_source = st.selectbox("Источник рекламы", ['Yandex', 'Google', 'VK', 'Social'])
        
        if st.button("🔮 Выполнить прогноз", use_container_width=True, type="primary"):
            input_data = pd.DataFrame({
                'age': [age],
                'gender': [gender],
                'region': [region],
                'exam_score': [exam_score],
                'ad_clicks': [ad_clicks],
                'site_visits': [site_visits],
                'time_on_site': [time_on_site],
                'ad_source': [ad_source]
            })
            
            prediction, probability = st.session_state.model.predict(input_data)
            
            st.markdown("---")
            st.subheader("🎯 Результат прогноза")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction[0] == 1:
                    st.success("### ✅ ЗАЧИСЛЕН")
                else:
                    st.error("### ❌ НЕ ЗАЧИСЛЕН")
            
            with col2:
                st.metric("Вероятность зачисления", f"{probability[0]*100:.2f}%")
            
            # Probability gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability[0]*100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Вероятность зачисления"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgray"},
                        {'range': [30, 70], 'color': "gray"},
                        {'range': [70, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        st.subheader("Массовое прогнозирование")
        
        uploaded_file = st.file_uploader("Загрузите CSV файл для массового прогнозирования", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df_predict = pd.read_csv(uploaded_file)
                
                st.write("**Предварительный просмотр данных:**")
                st.dataframe(df_predict.head(), use_container_width=True)
                
                if st.button("🔮 Прогнозировать все", use_container_width=True):
                    predictions, probabilities = st.session_state.model.predict(df_predict)
                    
                    df_predict['Прогноз'] = ['Зачислен' if p == 1 else 'Не зачислен' for p in predictions]
                    df_predict['Вероятность'] = probabilities
                    
                    st.write("**Результаты прогнозирования:**")
                    st.dataframe(df_predict, use_container_width=True)
                    
                    # Download predictions
                    csv = df_predict.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Скачать прогнозы",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                    # Summary
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Всего прогнозов", len(predictions))
                    with col2:
                        st.metric("Прогноз: Зачислено", predictions.sum())
                    
            except Exception as e:
                st.error(f"❌ Ошибка: {e}")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Система прогнозирования поведения абитуриентов v1.0 | На базе TensorFlow & Streamlit</p>
    </div>
""", unsafe_allow_html=True)
