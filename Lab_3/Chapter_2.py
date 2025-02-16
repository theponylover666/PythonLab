from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve

# Подключение к базе данных Sakila
db_url = "postgresql+psycopg2://mpi:135a1@povt-cluster.tstu.tver.ru:5432/sakila_postgres"
engine = create_engine(db_url)

# Загрузка данных (пример: таблица film)
query = """
SELECT f.film_id, f.rental_duration, f.rental_rate, f.length, f.replacement_cost, c.name as category
FROM film f
JOIN film_category fc ON f.film_id = fc.film_id
JOIN category c ON fc.category_id = c.category_id
"""
df = pd.read_sql(query, engine)
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', '{:.3f}'.format)

# Первичный анализ
print("Количество строк и столбцов:", df.shape)
print("Объем памяти (МБ):", df.memory_usage(deep=True).sum() / (1024**2))

# Анализ числовых переменных
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
numeric_stats = df[numeric_cols].describe(percentiles=[0.25, 0.5, 0.75]).T
print(numeric_stats)

# Анализ категориальных переменных
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
categorical_modes = df[categorical_cols].mode().iloc[0]
categorical_counts = df[categorical_cols].apply(lambda x: x.value_counts().iloc[0])
print(pd.DataFrame({"Mode": categorical_modes, "Count": categorical_counts}))

# Обработка пропусков
df.dropna(inplace=True)

# Обработка выбросов
Q1 = df[numeric_cols].quantile(0.25)
Q3 = df[numeric_cols].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[~((df[numeric_cols] < lower_bound) | (df[numeric_cols] > upper_bound)).any(axis=1)]

# Кодирование категориальных переменных
encoder = OneHotEncoder()
categorical_encoded = encoder.fit_transform(df[categorical_cols]).toarray()
categorical_df = pd.DataFrame(categorical_encoded, columns=encoder.get_feature_names_out(categorical_cols))
df = df.drop(columns=categorical_cols).reset_index(drop=True)
df = pd.concat([df, categorical_df], axis=1)

# Разделение данных на train и test
X = df.drop(columns=['film_id'])
y = df['rental_duration'] > df['rental_duration'].median()  # Бинаризация целевой переменной
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Нормализация
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Определение моделей
models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression": LogisticRegression(),
    "SVM": SVC(probability=True)
}

# Обучение и оценка моделей
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    metrics = {
        "Доля правильных предсказаний": accuracy_score(y_test, y_pred),
        "Точность": precision_score(y_test, y_pred, average="binary"),
        "Полнота": recall_score(y_test, y_pred, average="binary"),
        "Среднее между точностью и полнотой": f1_score(y_test, y_pred, average="binary"),
        "Качество классификации": roc_auc_score(y_test, y_prob) if y_prob is not None else None,
        "Таблица ошибок": confusion_matrix(y_test, y_pred)
    }
    results[name] = metrics

# Визуализация ROC-кривых
plt.figure(figsize=(8, 6))
for name, model in models.items():
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=name)

plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-кривые моделей")
plt.legend()
plt.grid(True)
plt.show()

# Вывод результатов
results_df = pd.DataFrame(results).T
print(results_df)
