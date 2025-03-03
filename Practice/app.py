import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve

# Заголовок приложения
st.title("Проектная работа: Разработка веб-проекта для анализа данных")

# Загрузка данных
file_path = "train.csv"
df = pd.read_csv(file_path)

# Sidebar для навигации
page = st.sidebar.selectbox("Выберите страницу", ["Общее описание", "EDA", "Модели", "Визуализация"])

# Инициализация и обучение моделей
if 'log_reg' not in st.session_state or 'knn' not in st.session_state:
    # Раздел 2: Обработка данных и обучение моделей
    st.write("Обучение моделей...")

    # Выбор признаков и целевой переменной
    feature_cols = ['ph', 'osmo']
    X = df[feature_cols].values
    y = (df['target'] > df['target'].median()).astype(int)

    # Разделение на train и test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Нормализация
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Реализация KNN и логистической регрессии
    class KNNClassifier:
        def __init__(self, k=5):
            self.k = k

        def fit(self, X, y):
            self.X_train = X
            self.y_train = np.array(y)

        def predict(self, X):
            predictions = []
            for x in X:
                distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
                neighbors = np.argsort(distances)[:self.k]
                labels = self.y_train[neighbors]
                predictions.append(Counter(labels).most_common(1)[0][0])
            return np.array(predictions)

    class LogisticRegressionCustom:
        def __init__(self, lr=0.1, epochs=5000):
            self.lr = lr
            self.epochs = epochs

        def sigmoid(self, z):
            return 1 / (1 + np.exp(-z))

        def fit(self, X, y):
            self.theta = np.zeros(X.shape[1])
            for _ in range(self.epochs):
                z = np.dot(X, self.theta)
                predictions = self.sigmoid(z)
                gradient = np.dot(X.T, (predictions - y)) / y.size
                self.theta -= self.lr * gradient

        def predict(self, X):
            return (self.sigmoid(np.dot(X, self.theta)) >= 0.5).astype(int)

    knn = KNNClassifier(k=5)
    knn.fit(X_train, y_train)
    log_reg = LogisticRegressionCustom(lr=0.1, epochs=5000)
    log_reg.fit(X_train, y_train)

    # Сохраняем модели в session_state
    st.session_state.knn = knn
    st.session_state.log_reg = log_reg
    st.session_state.X_train = X_train
    st.session_state.y_train = y_train
    st.session_state.X_test = X_test
    st.session_state.y_test = y_test
    st.session_state.feature_cols = feature_cols
    st.write("Модели обучены.")

else:
    # Получаем обученные модели из session_state
    knn = st.session_state.knn
    log_reg = st.session_state.log_reg
    X_train = st.session_state.X_train
    y_train = st.session_state.y_train
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    feature_cols = st.session_state.feature_cols

if page == "Общее описание":
    # Раздел 1: Общее описание проекта
    st.header("Общее описание проекта")
    st.write("""
    Этот проект представляет собой веб-приложение для анализа данных с использованием библиотеки Streamlit.
    В проекте используются данные из набора данных "train.csv", и применяются модели машинного обучения для классификации.
    """)

elif page == "EDA":
    # Раздел 2: Разведочный анализ данных (EDA)
    st.header("Разведочный анализ данных (EDA)")

    # Основная информация о данных
    st.subheader("Основная информация о данных")
    st.write(f"Количество строк: {df.shape[0]}, Количество столбцов: {df.shape[1]}")
    st.write(f"Размер датафрейма в памяти: {df.memory_usage(deep=True).sum():} байт")

    # Статистики по числовым переменным
    st.subheader("Статистики по числовым переменным")
    st.write(df.describe(percentiles=[0.25, 0.5, 0.75]).T)

    # Анализ моды для категориальных переменных
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if not categorical_cols:
        st.write("В данных отсутствуют категориальные переменные.")
    else:
        for col in categorical_cols:
            mode_value = df[col].mode()[0]
            mode_count = df[col].value_counts()[mode_value]
            st.write(f"Мода для {col}: {mode_value} (встречается {mode_count} раз)")

    # Обработка пропусков
    st.subheader("Обработка пропусков")
    missing_values = df.isnull().sum()
    st.write("Пропущенные значения в данных:")
    st.write(missing_values[missing_values > 0])

    df.dropna(inplace=True)
    st.write(f"После удаления пропусков осталось {df.shape[0]} строк.")

    # Обнаружение выбросов с помощью IQR
    st.subheader("Обнаружение выбросов с помощью IQR")


    def detect_outliers(df, columns):
        outliers_info = {}
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outliers_info[col] = len(outliers)
        return outliers_info

    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    outliers = detect_outliers(df, numerical_cols)

    st.write("Количество выбросов по каждому признаку:")
    for col, count in outliers.items():
        st.write(f"{col}: {count}")

    # Удаление выбросов
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    st.write(f"После удаления выбросов осталось {df.shape[0]} строк.")

elif page == "Модели":
    # Обучение моделей
    st.header("Обучение моделей")

    # Модели уже обучены, выводим информацию о них
    st.write("Модели обучены. Переходите к следующей странице для оценки их качества.")

elif page == "Визуализация":
    # Визуализация моделей и ROC кривых
    st.header("Визуализация моделей")

    # Оценка качества моделей
    def evaluate_model(model, X_test, y_test, name):
        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        conf_matrix = confusion_matrix(y_test, predictions)
        roc_auc = roc_auc_score(y_test, predictions)

        st.write(f"\nМетрики для {name}:")
        st.write(f"Точность (A): {accuracy:.4f}")
        st.write(f"Точность положительного класса (P): {precision:.4f}")
        st.write(f"Полнота (R): {recall:.4f}")
        st.write(f"Среднее между точностью и полнотой (E): {f1:.4f}")
        st.write(f"Оценки качества: {roc_auc:.4f}")
        st.write("Матрица ошибок:")
        st.write(conf_matrix)

    st.header("Оценка качества моделей")
    evaluate_model(knn, X_test, y_test, "KNN")
    evaluate_model(log_reg, X_test, y_test, "Logistic Regression")

    # Визуализация ROC-кривых
    st.header("ROC-кривые")

    y_train_probs_log_reg = log_reg.sigmoid(np.dot(X_train, log_reg.theta))
    y_test_probs_log_reg = log_reg.sigmoid(np.dot(X_test, log_reg.theta))

    fpr_train_log, tpr_train_log, _ = roc_curve(y_train, y_train_probs_log_reg)
    fpr_test_log, tpr_test_log, _ = roc_curve(y_test, y_test_probs_log_reg)

    # ROC-кривые для KNN
    y_train_probs_knn = knn.predict(X_train)
    y_test_probs_knn = knn.predict(X_test)

    fpr_train_knn, tpr_train_knn, _ = roc_curve(y_train, y_train_probs_knn)
    fpr_test_knn, tpr_test_knn, _ = roc_curve(y_test, y_test_probs_knn)

    fig, ax = plt.subplots()
    ax.plot(fpr_train_log, tpr_train_log, color="green", linestyle="--", label="ROC Train (LogReg)")
    ax.plot(fpr_test_log, tpr_test_log, color="blue", linestyle="--", label="ROC Test (LogReg)")
    ax.plot(fpr_train_knn, tpr_train_knn, color="darkgreen", label="ROC Train (KNN)")
    ax.plot(fpr_test_knn, tpr_test_knn, color="darkblue", label="ROC Test (KNN)")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic (ROC)")
    ax.legend()
    st.pyplot(fig)

    def plot_decision_boundary(model, X, y, title):
        h = .1  # Шаг сетки
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        fig, ax = plt.subplots()
        ax.contourf(xx, yy, Z, alpha=0.3)
        ax.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=plt.cm.coolwarm)
        ax.set_xlabel(feature_cols[0])
        ax.set_ylabel(feature_cols[1])
        ax.set_title(title)
        st.pyplot(fig)


    st.header("Границы решений")
    st.write("Визуализация границ решений для моделей KNN и логистической регрессии.")

    plot_decision_boundary(knn, X_train, y_train, "KNN Decision Boundary")
    plot_decision_boundary(log_reg, X_train, y_train, "Logistic Regression Decision Boundary")
